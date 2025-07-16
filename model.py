import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from link_prediction_head import CoxHead
from link_prediction_head import LinkMLPPredictor
from link_prediction_head import CosineLinkPredictor
from link_prediction_head import TimeAwareCosineLinkPredictor
from risk_calibration import fit_absolute_risk_calibrator
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv, GCNConv
from torch_geometric.nn import Linear
from torch.nn import Module, ModuleDict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import datetime
from lifelines.utils import concordance_index
import os
import json
from sklearn.metrics import average_precision_score
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
import math
from losses import (
    binary_focal_loss_with_logits,
    simple_smote,
    smooth_labels,
    mixup,
    get_class_balanced_weight,
    differentiable_ap_loss,
    dynamic_negative_sampling,
    compute_total_loss,
    advanced_link_loss
)
from init import (
    model,
    link_head,
    cox_head,
    joint_head,
    patient_classifier,
    in_dims,
    metadata,
    device
)
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# chosen conditions
chosen = ["Essential hypertension", "Other specified conditions associated with the spine (intervertebral disc displacement)",
    "Osteoporosis","Diabetes mellitus, type unspecified",
    "Non-alcoholic fatty liver disease","Coronary atherosclerosis",
    "Malignant neoplasms of breast"
]


class HeteroGAT(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim: int = 64,
        out_dim:   int = 64,
        num_heads: int = 2,
        dropout:   float = 0.5
    ):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)
        self.hidden_dim = hidden_dim

        # 1) input projections
        self.input_proj = nn.ModuleDict({
            n: Linear(in_dims[n], hidden_dim) for n in metadata[0]
        })
        self.shrink = nn.ModuleDict({
            n: nn.Identity() for n in metadata[0]
        })

        # 2) build two hetero?conv layers
        def make_layer(in_d, out_d):
            convs = {}
            for et in metadata[1]:
                if et in [
                    ('patient','to','signature'),
                    ('signature','to_rev','patient')
                ]:
                    convs[et] = GATConv(
                        (in_d, in_d),
                        out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        edge_dim=1,
                    )
                elif et in [
                    ('patient','follows','patient'),
                    ('patient','follows_rev','patient')
                ]:
                    # simpler aggregator for temporal links
                    convs[et] = GCNConv(in_d, out_d)
                else:
                    convs[et] = GATConv(
                        (in_d, in_d),
                        out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
            return HeteroConv(convs, aggr='sum')

        self.conv1 = make_layer(hidden_dim, hidden_dim)
        self.conv2 = make_layer(hidden_dim, out_dim)

        # 3) final node?type projections
        self.linear_proj = nn.ModuleDict({
            n: Linear(out_dim, out_dim) for n in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # embed inputs
        x0 = {n: self.input_proj[n](x) for n,x in x_dict.items()}

        # prepare edge?attrs only for signature relations
        ew = {}
        fwd = ('patient','to','signature')
        rev = ('signature','to_rev','patient')
        if edge_attr_dict and fwd in edge_attr_dict:
            w = edge_attr_dict[fwd].view(-1,1).float()
            ew[fwd] = w
            ew[rev] = w.flip(0)

        # --- first hetero conv + residual + ReLU ---
        x1_raw = self.conv1(x0, edge_index_dict, edge_attr_dict=ew)
        x1 = {}
        for n in self.metadata[0]:
            h0 = x0[n]
            h1 = x1_raw.get(n)
            if h1 is None:
                x1[n] = h0
            else:
                x1[n] = F.relu(h1 + h0)

        # --- second hetero conv + residual + ReLU ---
        x2_raw = self.conv2(x1, edge_index_dict, edge_attr_dict=ew)
        x2 = {}
        for n in self.metadata[0]:
            h1 = x1[n]
            h2 = x2_raw.get(n)
            if h2 is None:
                x2[n] = h1
            else:
                x2[n] = F.relu(h2 + h1)

        # --- final projections ---
        out = {}
        for n, h in x2.items():
            out[n] = self.linear_proj[n](h)
        return out


class JointHead(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize raw (unbounded) parameters at 0.0
        self.s_link = nn.Parameter(torch.tensor(0.0))
        self.s_cox = nn.Parameter(torch.tensor(0.0))

    def forward(self, link_loss: torch.Tensor, cox_loss: torch.Tensor) -> torch.Tensor:
        # compute positive log-variances via softplus
        log_var_link = F.softplus(self.s_link)
        log_var_cox = F.softplus(self.s_cox)

        term_link = 0.5 * link_loss * torch.exp(-log_var_link) + 0.5 * log_var_link
        term_cox = 0.5 * cox_loss * torch.exp(-log_var_cox) + 0.5 * log_var_cox

        return term_link + term_cox

def generate_pseudo_labels(
    train_graphs: list,
    model: torch.nn.Module,
    link_head: torch.nn.Module,
    cond_embeds: torch.Tensor,
    eps: float,
    device: torch.device,
    gender_map: dict,
    BREAST_IDX: int
) -> list:
    """
    For each window-graph in train_graphs, run inference on ALL unlabeled
    (patient?condition) pairs, pick the top N_c for each condition c (where
      N_c = max_j |V^l_j| - |V^l_c|),
    subject to prob > eps, and return those as new pseudo-edges.

    Returns a list pseudo_edges_per_graph of length len(train_graphs),
    where each element is a list of (patient_node_index, condition_index)
    pairs to add to that graph.
    """
    model.eval()
    link_head.eval()

    C = cond_embeds.size(0)  # number of conditions
    device = cond_embeds.device

    # 1) Count how many real edges each class already has:
    labeled_count = torch.zeros(C, dtype=torch.long, device=device)
    for g in train_graphs:
        if ('patient','has','condition') in g.edge_types:
            ei = g['patient','has','condition'].edge_index.to(device)  # [2, E]
            dst = ei[1]
            for c in range(C):
                labeled_count[c] += (dst == c).sum()

    # 2) Compute N_hat[c] = max_j |V^l_j| - |V^l_c|
    max_count = int(labeled_count.max().item())
    N_hat = [ (max_count - int(labeled_count[c].item())) for c in range(C) ]

    pseudo_edges_per_graph = [[] for _ in range(len(train_graphs))]

    # 3) For each window-graph, build embeddings and score unlabeled pairs:
    for wi, g in enumerate(train_graphs):
        data = g.to(device)
        # Build edge_index_dict and edge_attr_dict exactly as in your forward:
        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
            if et != ('patient','has','condition')
        }
        edge_attr_dict = {}
        if ('patient','to','signature') in data.edge_types \
           and hasattr(data['patient','to','signature'], 'edge_attr'):
            edge_attr_dict[('patient','to','signature')] = data['patient','to','signature'].edge_attr.to(device)

        # (a) Compute patient embeddings for this graph:
        out = model(data.x_dict, edge_index_dict, edge_attr_dict)
        P = out['patient']            # [N_p, D]
        C_fixed = cond_embeds         # [C, D]

        Np = P.size(0)
        if Np == 0:
            continue

        # (b) Build set of ?already-labeled? edges:
        existing = set()
        if ('patient','has','condition') in g.edge_types:
            existing_ei = g['patient','has','condition'].edge_index.to(device)
            for idx in range(existing_ei.size(1)):
                i_node = int(existing_ei[0,idx].item())
                c_node = int(existing_ei[1,idx].item())
                existing.add((i_node, c_node))

        # (c) Build candidate list of all missing (i, c) pairs:
        all_pairs = []
        names = g['patient'].name  # RegistrationCode list for i=0..Np-1
        for i_node in range(Np):
            for c_node in range(C):
                if (i_node, c_node) not in existing:
                    is_female = (gender_map.get(names[i_node], 'female') == 'female')
                    # skip ?breast?male? entirely
                    if c_node == BREAST_IDX and not is_female:
                        continue
                    all_pairs.append((i_node, c_node))

        if len(all_pairs) == 0:
            continue

        # (d) Score all candidates in one batch:
        edge_i = torch.tensor([p[0] for p in all_pairs], dtype=torch.long, device=device)
        edge_c = torch.tensor([p[1] for p in all_pairs], dtype=torch.long, device=device)
        candidate_ei = torch.stack([edge_i, edge_c], dim=0)  # [2, M]

        # extract time-to-event for each candidate (i,c)
        tte = data['patient'].duration[candidate_ei[0], candidate_ei[1]]  # [M]
        with torch.no_grad():
            logits = link_head(P, C_fixed, candidate_ei, tte)  # now 4-arg
            probs = torch.sigmoid(logits)               # [M]

        # (e) Group ?candidate? indices by class c if prob>eps:
        candidates_by_class = [[] for _ in range(C)]
        for idx_in_all, (i_node, c_node) in enumerate(all_pairs):
            p_val = float(probs[idx_in_all].item())
            if p_val > eps:
                candidates_by_class[c_node].append((i_node, p_val))

        # (f) For each class c, take top N_hat[c] by confidence:
        for c_node in range(C):
            if N_hat[c_node] <= 0:
                continue
            M_c = candidates_by_class[c_node]
            if not M_c:
                continue
            M_c.sort(key=lambda x: x[1], reverse=True)
            top_k = M_c[: N_hat[c_node] ]
            for (i_node, _) in top_k:
                pseudo_edges_per_graph[wi].append((i_node, c_node))

    return pseudo_edges_per_graph


def plot_link_prediction_curves(
    link_scores_diag: list[np.ndarray],
    link_scores_non: list[np.ndarray],
    diag_windows: list[list[int]],
    disease_name: str,
    save_path: str
):
    """
    link_scores_diag : list of length T, each an array of scores for patients who *will* be diagnosed
    link_scores_non  : list of length T, each an array of scores for patients who *never* get diagnosed
    diag_windows     : list of lists, true diagnosis windows for the *diagnosed* patients
    disease_name     : title
    save_path        : where to write the PNG
    """
    T = len(link_scores_diag)
    x = np.arange(T)

    # compute means & stds
    mean_d = np.array([s.mean()    if s.size else np.nan for s in link_scores_diag])
    std_d  = np.array([s.std(ddof=0) if s.size else np.nan for s in link_scores_diag])
    mean_n = np.array([s.mean()    if s.size else np.nan for s in link_scores_non])
    std_n  = np.array([s.std(ddof=0) if s.size else np.nan for s in link_scores_non])

    # flatten diag_windows for median/IQR
    flat = [w for sub in diag_windows for w in sub]
    if flat:
        median = np.median(flat)
        p25, p75 = np.percentile(flat, [25, 75])
    else:
        median = p25 = p75 = None

    sns.set(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(12,6))

    # diagnosed (blue)
    ax.plot(x, mean_d, color="blue",  lw=2, label="Will be diagnosed")
    ax.fill_between(x, mean_d - std_d, mean_d + std_d, color="blue", alpha=0.2)

    # never diagnosed (red)
    ax.plot(x, mean_n, color="red",   lw=2, label="Never diagnosed")
    ax.fill_between(x, mean_n - std_n, mean_n + std_n, color="red",  alpha=0.2)

    # axes & legend
    ax.set_title(f"Link?scores for {disease_name}", fontsize=16)
    ax.set_xlabel("Window Index", fontsize=14)
    ax.set_ylabel("Predicted Score", fontsize=14)
    ax.set_xlim(0, T-1)
    ax.set_ylim(0.0, 1.02)

    # diagnosis median/IQR
    if median is not None and p25 is not None and p75 is not None:
        ax.axvline(float(median), color="black", linestyle="--", label="Median Diagnosis")
        ax.axvspan(float(p25), float(p75), color="gray", alpha=0.2, label="25?75% Diagnosis")

    ax.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
# get gender information per patients
study_ids = [10, 1001, 1002]
bm = BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first').df.join(BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first').df_metadata)

age_gender = bm[['age', 'gender', 'yob']].reset_index()
del age_gender['Date']
age_gender = age_gender[~age_gender['gender'].isna()]
age_gender = age_gender[~age_gender['age'].isna()]

gender_dictionary = {1:'male', 0:'female'}

age_gender['gender'] = (
    age_gender['gender']
    .fillna(0)
    .astype(int)
    .map(gender_dictionary)
    .astype('category')
)
# gender_df has columns ['RegistrationCode','gender'], where gender is 'male' or 'female'
gender_map = dict(zip(age_gender['RegistrationCode'], age_gender['gender']))

BREAST_IDX = chosen.index("Malignant neoplasms of breast")
focus_idx = chosen.index("Diabetes mellitus, type unspecified") # index to make the model focus on 1 disease

def build_diag_by_win(full_map, graphs):
    pts = {p for g in graphs for p in g['patient'].name}
    m = defaultdict(list)
    for d in full_map:
        if d["patient"] in pts:
            m[d["window"]].append((d["patient"], d["cond"]))
    return m

def subsample_patients_in_graphs(graphs, focus_idx=None, seed=42):
    """
    Returns a new list of graphs where only:
    1. Positive patients (diagnosed with any condition, or with focus_idx if set)
    2. Negative patients (never diagnosed with any condition)
    are kept. The number of negatives is at most 2x the number of positives.
    After filtering, remap patient node indices in all edge_index tensors.
    """
    all_patients = set()
    for g in graphs:
        all_patients.update(g['patient'].name)
    positive_patients = set()
    for g in graphs:
        if ('patient','has','condition') in g.edge_types:
            ei = g['patient','has','condition'].edge_index
            names = g['patient'].name
            src = ei[0]
            dst = ei[1]
            for i in range(src.size(0)):
                p_idx = src[i].item()
                c_idx = dst[i].item()
                if focus_idx is not None:
                    if c_idx == focus_idx:
                        positive_patients.add(names[p_idx])
                else:
                    positive_patients.add(names[p_idx])
    negative_patients = all_patients - positive_patients
    negative_patients = list(negative_patients)
    random.seed(seed)
    num_positives = len(positive_patients)
    num_negatives = min(2 * num_positives, len(negative_patients))
    sampled_negatives = set(random.sample(negative_patients, num_negatives))
    keep_patients = positive_patients | sampled_negatives
    filtered_graphs = []
    for g in graphs:
        names = g['patient'].name
        keep_idx = torch.tensor([i for i, name in enumerate(names) if name in keep_patients], dtype=torch.long)
        if keep_idx.numel() == 0:
            continue
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_idx.tolist())}
        g['patient'].x = g['patient'].x[keep_idx]
        g['patient'].name = [names[i] for i in keep_idx.tolist()]
        for et in g.edge_types:
            ei = g[et].edge_index
            num_edges = ei.shape[1]
            mask = torch.ones(num_edges, dtype=torch.bool)
            # If patient is source
            if et[0] == 'patient':
                src = ei[0]
                src_mask = torch.tensor([i in old_to_new for i in src.tolist()], dtype=torch.bool)
                mask &= src_mask
            # If patient is target
            if et[-1] == 'patient':
                dst = ei[1]
                dst_mask = torch.tensor([i in old_to_new for i in dst.tolist()], dtype=torch.bool)
                mask &= dst_mask
            ei = ei[:, mask]
            # Remap indices
            if et[0] == 'patient':
                ei[0] = torch.tensor([old_to_new[i] for i in ei[0].tolist()], dtype=torch.long)
            if et[-1] == 'patient':
                ei[1] = torch.tensor([old_to_new[i] for i in ei[1].tolist()], dtype=torch.long)
            g[et].edge_index = ei
            if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None:
                g[et].edge_attr = g[et].edge_attr[mask]
        filtered_graphs.append(g)
    return filtered_graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEGATIVE_MULTIPLIER = 10  # you can adjust this multiplier later
PAIRWISE_TAU = 5.0       # scale for time-weighted ranking loss
SMOTE_MULTIPLIER = 5.0   # oversample positives beyond negatives
HORIZON = 10   # how many graphs ahead to predict over
CONTRASTIVE_WEIGHT = 0.2

# --- Advanced Loss and Regularization Utilities ---


# Remove train() and evaluate() function definitions from this file.
# Update DataLoader instantiations:
# train_loader = DataLoader(train_graphs, batch_size=1,sampler=train_sampler, num_workers=2)
# val_loader = DataLoader(val_graphs, batch_size=1, num_workers=2)
# test_loader = DataLoader(test_graphs, batch_size=1, num_workers=2)


# Model and head definitions (before main)
# in_dims = {
#     'patient': 138,
#     'signature': 96,
#     'condition': 7,
# }
# metadata = (
#     ['patient','condition','signature'],
#     [
#       ('patient','to','signature'),
#       ('signature','to_rev','patient'),
#       ('patient','has','condition'),
#       ('condition','has_rev','patient'),
#       ('patient','follows','patient'),
#       ('patient','follows_rev','patient'),
#     ]
# )
# model = HeteroGAT(
#     metadata=metadata,
#     in_dims=in_dims,
#     hidden_dim=64,
#     out_dim =64,
#     dropout = 0.5,
#     num_heads = 2
# ).to(device)

# link_head = TimeAwareCosineLinkPredictor(init_scale = 5.0 , use_bias = True).to(device)
# cox_head = CoxHead(input_dim=64, num_conditions=7).to(device)
# joint_head = JointHead().to(device)
# patient_classifier = nn.Linear(
#     64,               # same as your final patient embedding dim (e.g. 128)
#     7         # output one score per condition
# ).to(device)


def main():
    from train import train
    from eval import evaluate
    # 1. Load graphs
    train_graphs = torch.load("split/train_graphs_3d.pt", weights_only=False)
    val_graphs = torch.load("split/val_graphs_3d.pt", weights_only=False)
    test_graphs = torch.load("split/test_graphs_3d.pt", weights_only=False)

    # 2. Subsample patients in training set
    train_graphs = subsample_patients_in_graphs(train_graphs, focus_idx=focus_idx)

    # 3. Load diagnosis mapping
    with open("diagnosis_mapping.json") as f:
        full_diag_map = json.load(f)

    diag_by_win_train = build_diag_by_win(full_diag_map, train_graphs)
    diag_by_win_val = build_diag_by_win(full_diag_map, val_graphs)
    diag_by_win_test = build_diag_by_win(full_diag_map, test_graphs)

    # 4. DataLoaders
    POS_WINDOW_WEIGHT = 10.0
    NEG_WINDOW_WEIGHT = 1.0
    cond = focus_idx
    train_pos_counts = []
    for g in train_graphs:
        if ('patient','has','condition') in g.edge_types:
            ei = g['patient','has','condition'].edge_index
            count_cond = (ei[1] == cond).sum().item()
        else:
            count_cond = 0
        train_pos_counts.append(count_cond)
    sample_weights = [POS_WINDOW_WEIGHT if cnt > 0 else NEG_WINDOW_WEIGHT for cnt in train_pos_counts]
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_graphs, batch_size=1, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size=1, num_workers=0)

    # Collect all unique patient names from train_graphs
    all_patient_names = sorted({name for g in train_graphs for name in g['patient'].name})

    # 5. Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(link_head.parameters())
        + list(cox_head.parameters())
        + list(patient_classifier.parameters())
        + list(joint_head.parameters()),
        lr=1e-3
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Full advanced training loop
    USE_PSEUDO = True  # Set to None to disable pseudo-labeling
    plot_idx = focus_idx  # Set to None to disable plotting
    EPOCHS = 200
    pseudo_eps = 0.80
    max_pr_auc = 0.5
    WARMUP_EPOCHS = 8
    history = []
    pseudo_label_by_window = {wi: {} for wi in range(len(train_graphs))}

    for epoch in range(1, EPOCHS + 1):
        # Warm-up phase
        if WARMUP_EPOCHS is not None and epoch <= WARMUP_EPOCHS:
            print(f"[Epoch {epoch}] Warm-up: training only link_head")
            for module in (model, cox_head, patient_classifier):
                for p in module.parameters():
                    p.requires_grad = False
            for p in link_head.parameters():
                p.requires_grad = True
            for p in joint_head.parameters():
                p.requires_grad = True
        else:
            print(f"[Epoch {epoch}] Unfreezing all modules")
            for module in (model, link_head, cox_head, patient_classifier, joint_head):
                for p in module.parameters():
                    p.requires_grad = True

        # Pseudo-label regeneration
        if USE_PSEUDO and epoch > 20 and (epoch-1)%5==0:
            with torch.no_grad():
                C = in_dims['condition']
                eye = torch.eye(C, device=device)
                h = model.input_proj['condition'](eye)
                h = F.relu(h)
                h = F.relu(h)
                h = F.relu(model.shrink['condition'](h))
                cond_embeds = model.linear_proj['condition'](h)
            pse = generate_pseudo_labels(
                train_graphs, model, link_head, cond_embeds,
                eps=pseudo_eps, device=device,
                gender_map=gender_map, BREAST_IDX=BREAST_IDX
            )
            for wi, g in enumerate(train_graphs):
                new_edges = pse[wi]
                if not new_edges:
                    continue
                old_ei = g.get(('patient', 'has', 'condition'), torch.empty((2, 0), dtype=torch.long))
                src = torch.tensor([i for i, c in new_edges], dtype=torch.long, device=device)
                dst = torch.tensor([c for i, c in new_edges], dtype=torch.long, device=device)
                merged = torch.cat([old_ei.to(device), torch.stack([src, dst], dim=0)], dim=1)
                g[('patient', 'has', 'condition')].edge_index = merged
                g[('condition', 'has_rev', 'patient')].edge_index = merged.flip(0)
            pseudo_label_by_window.clear()
            for wi, edges in enumerate(pse):
                m = {}
                for i, c in edges:
                    if i not in m:
                        m[i] = c
                pseudo_label_by_window[wi] = m

        # Training step
        train_loss = train(
            model, link_head, cox_head, patient_classifier,
            train_loader, optimizer, device,
            pseudo_label_by_window,
            diag_by_win_train,
            all_patient_names
        )

        # Validation step
        with torch.no_grad():
            C = in_dims['condition']
            eye = torch.eye(C, device=device)
            h = model.input_proj['condition'](eye)
            h = F.relu(h)
            h = F.relu(h)
            h = F.relu(model.shrink['condition'](h))
            cond_embeds = model.linear_proj['condition'](h)
        (
            avg_link, avg_cox, avg_node_ce, node_acc,
            cidx_list, mean_cidx, link_auc,
            pr_auc, pr_per_cond, val_scores,
            calibrators,
        ) = evaluate(
            model, link_head, cox_head, joint_head, patient_classifier,
            val_loader, device,
            NEGATIVE_MULTIPLIER, diag_by_win_val,
            cond_embeds, plot_idx=focus_idx
        )
        scheduler.step(pr_per_cond[plot_idx] if plot_idx is not None else pr_auc)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_link_loss": avg_link,
            "val_cox_loss": avg_cox,
            "val_node_ce": avg_node_ce,
            "val_node_acc": node_acc,
            "val_c_indices": cidx_list,
            "val_mean_c": mean_cidx,
            "val_link_auc": link_auc,
            "val_pr_auc": pr_auc,
            "val_pr_per_cond": pr_per_cond,
            "calibrators": bool(calibrators is not None),
            "timestamp": datetime.datetime.now().isoformat(),
        })
        print(f"[Epoch {epoch}] Train {train_loss:.4f} | ValLink {avg_link:.4f} "
              f"Val Cox Loss: {avg_cox:.4f} | Node Acc: {node_acc} | "
              f"Val C-Index: {cidx_list} | PR AUC per condition: {pr_per_cond}")

        # Plotting
        if plot_idx is not None and max_pr_auc < (pr_per_cond[plot_idx] if pr_per_cond[plot_idx] is not None else 0):
            max_pr_auc = pr_per_cond[plot_idx]
            val_true_windows = [
                [wi for (p, c) in diag_by_win_val.get(wi, []) if c == plot_idx]
                for wi in range(len(val_graphs))
            ]
            pos_patients = {
                p
                for (win, pairs) in diag_by_win_val.items()
                for (p, c) in pairs
                if c == plot_idx
            }
            val_scores_diag = []
            val_scores_non = []
            for scores, batch in zip(val_scores, val_loader):
                names = [
                    code
                    for sub in batch['patient'].name
                    for code in (sub if isinstance(sub, list) else [sub])
                ]
                mask = np.array([code in pos_patients for code in names])
                arr = np.asarray(scores)
                val_scores_diag.append(arr[mask])
                val_scores_non.append(arr[~mask])
            os.makedirs("outputs", exist_ok=True)
            plot_link_prediction_curves(
                link_scores_diag=val_scores_diag,
                link_scores_non=val_scores_non,
                diag_windows=val_true_windows,
                disease_name=chosen[plot_idx],
                save_path=f"outputs/val_{chosen[plot_idx].replace(' ', '_')}_epoch{epoch}.png",
            )
    print("Training complete.")


if __name__ == "__main__":
    mp.freeze_support()
    main()