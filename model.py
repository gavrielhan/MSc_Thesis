from link_prediction_head import CoxHead
from link_prediction_head import LinkMLPPredictor
from link_prediction_head import CosineLinkPredictor
from link_prediction_head import TimeAwareCosineLinkPredictor
from risk_calibration import fit_absolute_risk_calibrator
import pandas as pd
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import math
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# chosen conditions
chosen = ["Essential hypertension", "Other specified conditions associated with the spine (intervertebral disc displacement)",
    "Osteoporosis","Diabetes mellitus, type unspecified",
    "Non-alcoholic fatty liver disease","Coronary atherosclerosis",
    "Malignant neoplasms of breast"
]

def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary focal loss:
      FL = - ?_t * (1?p_t)^? * log(p_t)
    """
    prob = torch.sigmoid(logits)
    ce   = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t  = prob * labels + (1 - prob) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    fl   = alpha_t * (1 - p_t).pow(gamma) * ce
    if reduction == "sum":
        return fl.sum()
    elif reduction == "mean":
        return fl.mean()
    return fl

class HeteroGAT(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim: int = 128,
        out_dim:   int = 128,
        num_heads: int = 4,
        dropout:   float = 0.3
    ):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)

        # Input projections for each node type
        self.input_proj = nn.ModuleDict({
            n: Linear(in_dims[n], hidden_dim) for n in metadata[0]
        })
        self.hidden_dim = hidden_dim

        def make_layer(in_d, out_d):
            convs = {}
            for et in metadata[1]:
                if et in [('patient','to','signature'),
                          ('signature','to_rev','patient')]:
                    # GAT with built-in edge_attr support (edge_dim=1)
                    convs[et] = GATConv(
                        in_channels=(in_d, in_d),
                        out_channels=out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        edge_dim=1,
                    )
                elif et == ('patient','follows','patient') or et == ('patient','follows_rev','patient'):
                    convs[et] = GATConv(
                        in_channels=(in_d, in_d),
                        out_channels=out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                        edge_dim=1,
                    )
                else:
                    # all other relations (including condition has/has_rev)
                    convs[et] = GATConv(
                        in_channels=(in_d, in_d),
                        out_channels=out_d // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
            return HeteroConv(convs, aggr='sum')

        # Three hetero?layers
        self.convs1 = make_layer(hidden_dim, hidden_dim)
        self.convs2 = make_layer(hidden_dim, hidden_dim)
        self.convs3 = make_layer(hidden_dim, out_dim)

        # Final projections
        self.linear_proj = nn.ModuleDict({
            n: Linear(out_dim, out_dim)
            for n in metadata[0]
        })
        self.shrink = nn.ModuleDict({
            n: Linear(hidden_dim, out_dim)
            for n in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # 1) Input proj
        x0 = {n: self.input_proj[n](x) for n, x in x_dict.items()}

        # 2) Build the 1?dim edge_attr dict for signature edges
        ew = {}
        rel_fwd = ('patient','to','signature')
        rel_rev = ('signature','to_rev','patient')
        if edge_attr_dict and rel_fwd in edge_attr_dict:
            w = edge_attr_dict[rel_fwd].float().view(-1,1)  # cast to float32
            ew[rel_fwd] = w
            ew[rel_rev] = w.flip(0)

        # 3) conv1 + fallback + ReLU
        x1 = self.convs1(x0, edge_index_dict, edge_attr_dict=ew)
        for n in self.metadata[0]:
            if n not in x1 or x1[n] is None:
                x1[n] = x0[n]
            else:
                x1[n] = F.relu(x1[n])

        # 4) conv2 + fallback + ReLU
        x2 = self.convs2(x1, edge_index_dict, edge_attr_dict=ew)
        for n in self.metadata[0]:
            if n not in x2 or x2[n] is None:
                x2[n] = x1[n]
            else:
                x2[n] = F.relu(x2[n])

        # 5) conv3 + fallback + ReLU
        x3 = self.convs3(x2, edge_index_dict, edge_attr_dict=ew)
        for n in self.metadata[0]:
            h = x3.get(n, x2[n])
            h = F.relu(h)

            # **NEW**: if for any node?type we still have 256 dims,
            # shrink it down to 128 before going to linear_proj
            if h.size(1) == self.hidden_dim:
                h = F.relu(self.shrink[n](h))

            x3[n] = h

            # final projections (now guaranteed to be 128?128 everywhere)
        return {n: self.linear_proj[n](h) for n, h in x3.items()}

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
    if median is not None:
        ax.axvline(median, color="black", linestyle="--", label="Median Diagnosis")
        ax.axvspan(p25, p75, color="gray", alpha=0.2, label="25?75% Diagnosis")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEGATIVE_MULTIPLIER = 5  # you can adjust this multiplier later
HORIZON = 10   # how many graphs ahead to predict over



def train(model, link_head, cox_head, patient_classifier,loader, optimizer, device, pseudo_label_by_window):
    model.train()
    link_head.train()
    cox_head.train()
    patient_classifier.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)
        # 1) Sanity check inputs
        for ntype, x in batch.x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError(f"[Batch {batch_idx}] Bad input on node '{ntype}': contains NaN or Inf")

        # 2) Skip if no useful edges
        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types
            and ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            continue

        optimizer.zero_grad()
        # build edge_attr dict
        edge_attr_dict = {}
        for rel in [('patient','to','signature'), ('signature','to_rev','patient'),
                    ('patient','follows','patient'), ('patient','follows_rev','patient')]:
            if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
                edge_attr_dict[rel] = batch[rel].edge_attr

        # 3) Forward through GNN
        out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        P = out['patient']    # [N_pat, D]
        C = out['condition']  # [C, D]
        zero = P.sum() * 0.0

        # 4) Gather true positives in horizon
        t = int(batch.window)
        pos_triples = []
        for w in range(t+1, t+1+HORIZON):
            for (p, ci) in diag_by_win_train.get(w, []):
                if focus_idx is None or ci == focus_idx:
                    pos_triples.append((p, ci, w - t))

        # 5) Map to indices & filter
        flat_names = [n for sub in batch['patient'].name
                      for n in (sub if isinstance(sub, list) else [sub])]
        filtered = []
        for p, ci, d in pos_triples:
            if ci == BREAST_IDX:
                if p in flat_names and gender_map.get(p,'female')=='female':
                    filtered.append((p,ci,d))
            else:
                if p in flat_names:
                    filtered.append((p,ci,d))

        src, dst, dists = [], [], []
        for p, ci, d in filtered:
            idx = flat_names.index(p)
            src.append(idx); dst.append(ci); dists.append(d)

        if src:
            # positives
            pos_ei = torch.stack([torch.tensor(src, device=device),
                                   torch.tensor(dst, device=device)], dim=0)
            tte_pos = batch['patient'].duration[pos_ei[0], pos_ei[1]]
            pos_logits = link_head(P, C, pos_ei, tte_pos)

            # mine negatives
            with torch.no_grad():
                all_src = torch.arange(P.size(0), device=device)
                all_dst = torch.full((P.size(0),), focus_idx, device=device)
                all_ei  = torch.stack([all_src, all_dst], dim=0)
                tte_all = batch['patient'].duration[all_src, all_dst]
                all_logits = link_head(P, C, all_ei, tte_all)
            pos_pairs = set((int(i),int(c)) for i,c in zip(pos_ei[0],pos_ei[1]))
            neg_mask = [(s.item(),) not in pos_pairs for s in all_src]
            cand_src   = all_src[neg_mask]
            cand_logits = all_logits[neg_mask]
            n_neg = pos_logits.size(0) * NEGATIVE_MULTIPLIER
            if cand_logits.numel() >= n_neg:
                _, topk = cand_logits.topk(n_neg, largest=True)
                neg_src = cand_src[topk]
            else:
                neg_src = cand_src
            neg_dst = torch.full_like(neg_src, focus_idx, device=device)
            neg_ei  = torch.stack([neg_src, neg_dst], dim=0)
            tte_neg = batch['patient'].duration[neg_ei[0], neg_ei[1]]
            neg_logits = link_head(P, C, neg_ei, tte_neg)

            # --- START replacement: per-batch weighted focal + global ranking ---
            # concatenate
            logits_all = torch.cat([pos_logits, neg_logits], dim=0)
            labels_all = torch.cat([
                torch.ones_like(pos_logits),
                torch.zeros_like(neg_logits)
            ], dim=0)

            # pos?weight
            pos_count = labels_all.sum()
            neg_count = labels_all.numel() - pos_count
            pos_weight = torch.clamp(neg_count / (pos_count + 1e-6), max=100.0)

            # focal (?=0.99, ?=2), reduction=none
            fl = binary_focal_loss_with_logits(
                logits_all, labels_all,
                alpha=0.99, gamma=2.0,
                reduction="none"
            )
            weights = labels_all * pos_weight + (1.0 - labels_all)
            focal_loss = (fl * weights).mean()

            # global pairwise ranking
            if pos_count > 0 and neg_count > 0:
                pl = logits_all[labels_all==1]
                nl = logits_all[labels_all==0]
                pairwise_loss = -F.logsigmoid(
                    pl.unsqueeze(1) - nl.unsqueeze(0)
                ).mean()
            else:
                pairwise_loss = torch.tensor(0.0, device=logits_all.device)

            link_loss = focal_loss + pairwise_loss
            # --- END replacement ---
        else:
            link_loss = zero.clone()

        # 6) Cox loss
        cox_loss = zero.clone()
        if hasattr(batch['patient'],'event') and hasattr(batch['patient'],'duration'):
            ev = batch['patient'].event
            du = batch['patient'].duration
            rs = cox_head(P)
            valid = 0
            for ci in range(C.size(0)):
                if ev[:,ci].sum().item()>0:
                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        rs[:,ci].unsqueeze(1),
                        du[:,ci].unsqueeze(1),
                        ev[:,ci].unsqueeze(1)
                    )
                    cox_loss += loss_ci
                    valid += 1
            if valid>0:
                cox_loss = cox_loss / valid

        # 7) Pseudo?label CE
        node_ps = zero.clone()
        lm = pseudo_label_by_window.get(t,{})
        if lm:
            idxs = torch.tensor(list(lm.keys()), device=device)
            labs = torch.tensor([lm[i] for i in idxs.cpu().tolist()],
                                device=device)
            logits = patient_classifier(P[idxs])
            node_ps = F.cross_entropy(logits, labs)

        # 8) Backward
        total = joint_head(link_loss, cox_loss) + 0.2 * node_ps
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(link_head.parameters()) +
            list(cox_head.parameters()) +
            list(patient_classifier.parameters()) +
            list(joint_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += total.item()
        num_batches += 1

    return total_loss / max(1, num_batches)




@torch.no_grad()
def evaluate(
    model,
    link_head,
    cox_head,
    patient_classifier,       # unused here but kept for signature
    loader,
    device,
    NEGATIVE_MULTIPLIER,      # unused in eval
    diag_by_win_map,
    frozen_condition_embeddings,
    plot_idx,                 # unused in this function
):
    model.eval()
    link_head.eval()
    cox_head.eval()

    # Accumulators
    total_link_loss = 0.0
    total_cox_loss  = 0.0
    window_scores  = []
    all_link_preds, all_link_labels = [], []
    per_cond_preds,  per_cond_labels = {}, {}
    all_risk_scores, all_durations, all_events = [], [], []
    num_batches = 0

    BLEND_LINK, BLEND_COX = 1.0, 0.0
    num_conditions = cox_head.linear.out_features
    for ci in range(num_conditions):
        per_cond_preds[ci]  = []
        per_cond_labels[ci] = []

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # 1) skip if no patients
        if batch['patient'].x.size(0) == 0:
            continue

        # 2) GNN forward
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient','has','condition')
        }
        edge_attr_dict = {}

        # 1) signature edges (as before)
        rel_sig = ('patient', 'to', 'signature')
        if rel_sig in batch.edge_types and hasattr(batch[rel_sig], 'edge_attr'):
            edge_attr_dict[rel_sig] = batch[rel_sig].edge_attr
            # if you also use the reverse relation in your model:
            edge_attr_dict[('signature', 'to_rev', 'patient')] = batch[rel_sig].edge_attr

        # 2) temporal 'follows' edges (3-day intervals)
        for rel in [
            ('patient', 'follows', 'patient'),
            ('patient', 'follows_rev', 'patient')
        ]:
            if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
                edge_attr_dict[rel] = batch[rel].edge_attr

        out   = model(batch.x_dict, edge_index_dict, edge_attr_dict)
        P     = out['patient']                            # [N_pat, D]
        C_emb = frozen_condition_embeddings.to(device)    # [C, D]
        Np, Nc = P.size(0), C_emb.size(0)

        # 3) build full grid of (patient,cond)
        src_all = torch.arange(Np, device=device).repeat_interleave(Nc)  # [Np*Nc]
        dst_all = torch.arange(Nc, device=device).repeat(Np)            # [Np*Nc]
        full_ei = torch.stack([src_all, dst_all], dim=0)               # [2, Np*Nc]

        # 4) compute & blend logits using the fixed evaluation weights
        link_logits = link_head(P, C_emb, full_ei)     # [Np*Nc]
        risks       = cox_head(P)                      # [Np, Nc]
        cox_logits  = risks[src_all, dst_all]          # [Np*Nc]

        # now blend using the constants you set
        logits_all = BLEND_LINK * link_logits + BLEND_COX * cox_logits
        probs_all  = torch.sigmoid(logits_all)         # [Np*Nc]

        # save for sliding-window plot
        window_scores.append(probs_all.view(Np, Nc).cpu().numpy())

        # 5) gather true positives in this window?s HORIZON
        flat_names = [
            n for sub in batch['patient'].name
              for n in (sub if isinstance(sub, list) else [sub])
        ]
        is_female_flat = torch.tensor(
            [gender_map.get(p,'female')=='female' for p in flat_names],
            device=device, dtype=torch.bool
        )
        pos_pairs = []
        for w in range(window_idx+1, window_idx+1+HORIZON):
            pos_pairs.extend(diag_by_win_map.get(w, []))

        # filter male?breast
        filtered = []
        for (p,ci) in pos_pairs:
            if ci==BREAST_IDX:
                if p in flat_names and is_female_flat[flat_names.index(p)]:
                    filtered.append((p,ci))
            else:
                filtered.append((p,ci))

        # map to node indices
        src_nodes, dst_nodes = [], []
        for (p,ci) in filtered:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        # pack into pos_ei
        if src_nodes:
            pos_ei = torch.stack([
                torch.tensor(src_nodes, device=device),
                torch.tensor(dst_nodes, device=device)
            ], dim=0)  # [2, N_pos]
        else:
            pos_ei = torch.empty((2,0), device=device, dtype=torch.long)

        # 6) build dense labels over full grid
        labels_all = torch.zeros_like(probs_all)
        if pos_ei.numel()>0:
            flat_pos = pos_ei[0]*Nc + pos_ei[1]
            labels_all[flat_pos] = 1.0

        # 7) collect for PR?AUC
        all_link_preds .append(probs_all.cpu())
        all_link_labels.append(labels_all.cpu())
        for ci in range(num_conditions):
            mask_ci = (dst_all.cpu()==ci)
            per_cond_preds [ci].append(probs_all.cpu()[mask_ci])
            per_cond_labels[ci].append(labels_all.cpu()[mask_ci])

        n_pos = int(labels_all.sum().item())
        n_all = labels_all.numel()


        # 8) balanced 1:1 BCE link loss
        pos_idx = (labels_all==1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_all==0).nonzero(as_tuple=True)[0]
        if pos_idx.numel()>0:
            k = pos_idx.numel()
            if neg_idx.numel()>k:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:k]]
            sample_idx = torch.cat([pos_idx, neg_idx], dim=0)
        else:
            sample_idx = neg_idx

        logits_s = logits_all[sample_idx]
        labels_s = labels_all[sample_idx]
        total_link_loss += F.binary_cross_entropy_with_logits(logits_s, labels_s).item()

        # 9) Cox partial-likelihood
        if hasattr(batch['patient'],'event'):
            evts = batch['patient'].event
            durs = batch['patient'].duration
            all_risk_scores.append(risks.cpu())
            all_durations .append(durs.cpu())
            all_events    .append(evts.cpu())

            for ci in range(num_conditions):
                ev_col = evts[:,ci]
                if ev_col.sum().item()>0:
                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        risks[:,ci].unsqueeze(1),
                        durs [:,ci].unsqueeze(1),
                        evts[:,ci].unsqueeze(1)
                    )
                    total_cox_loss += loss_ci.item()

        num_batches += 1

    # finalize
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)

    if all_link_preds:
        y_true  = torch.cat(all_link_labels).numpy()
        y_score = torch.cat(all_link_preds) .numpy()
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc   = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    pr_per_cond = []
    for ci in range(num_conditions):
        if per_cond_preds[ci]:
            y_t = torch.cat(per_cond_labels[ci]).numpy()
            y_s = torch.cat(per_cond_preds [ci]).numpy()
            pr_per_cond.append(average_precision_score(y_t, y_s))
        else:
            pr_per_cond.append(None)

    # C-index
    c_idxs = []
    if all_risk_scores:
        R = torch.cat(all_risk_scores).numpy()
        D = torch.cat(all_durations) .numpy()
        E = torch.cat(all_events   ).numpy()
        for ci in range(num_conditions):
            mask = E[:,ci]>=0
            if mask.sum()>0:
                try:
                    c_idxs.append(concordance_index(
                        D[mask,ci],
                        -R[mask,ci],
                        E[mask,ci]
                    ))
                except ZeroDivisionError:
                    c_idxs.append(None)
            else:
                c_idxs.append(None)
    mean_c = None
    valid  = [c for c in c_idxs if c is not None]
    if valid:
        mean_c = sum(valid)/len(valid)

    return (
        avg_link,    # balanced BCE link loss
        avg_cox,     # Cox loss
        None,        # node?CE (not computed here)
        None,        # node?acc (not computed here)
        c_idxs,
        mean_c,
        link_auc,
        pr_auc,
        pr_per_cond,
        window_scores,
    )


in_dims = {
    'patient': 138,
    'signature': 96,
    'condition': 7,
}
metadata = (
    ['patient','condition','signature'],
    [
      ('patient','to','signature'),
      ('signature','to_rev','patient'),
      ('patient','has','condition'),
      ('condition','has_rev','patient'),
      ('patient','follows','patient'),
      ('patient','follows_rev','patient'),
    ]
)
model = HeteroGAT(
    metadata=metadata,
    in_dims=in_dims,
    hidden_dim=256,
    out_dim =128,
    dropout = 0.4,
    num_heads = 8
).to(device)

#link_head = LinkMLPPredictor(input_dim=128).to(device)
#link_head = CosineLinkPredictor(input_dim=128, init_scale=10.0, use_bias=True).to(device)
link_head = TimeAwareCosineLinkPredictor(init_scale = 10.0 , use_bias = True).to(device)
cox_head = CoxHead(input_dim=128, num_conditions=7).to(device)

USE_PSEUDO = True

joint_head = JointHead().to(device)
patient_classifier = nn.Linear(
    128,               # same as your final patient embedding dim (e.g. 128)
    7         # output one score per condition
).to(device)

# Add to optimizer, alongside model, link_head, cox_head, JointHead
optimizer = torch.optim.Adam(
    list(model.parameters())
    + list(link_head.parameters())
    + list(cox_head.parameters())
    + list(patient_classifier.parameters())
    + list(joint_head.parameters()),
    lr=1e-3
)

#scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


def main():
    plot_idx = 3
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    global train_graphs, val_graphs, test_graphs
    global diag_by_win_train, diag_by_win_val, diag_by_win_test
    global global_pos_weight, scheduler


    train_graphs = torch.load("split/train_graphs_3d.pt", weights_only=False)
    val_graphs = torch.load("split/val_graphs_3d.pt", weights_only=False)
    test_graphs = torch.load("split/test_graphs_3d.pt", weights_only=False)

    with open("diagnosis_mapping.json") as f:
        full_diag_map = json.load(f)

    diag_by_win_train = build_diag_by_win(full_diag_map, train_graphs)
    diag_by_win_val = build_diag_by_win(full_diag_map, val_graphs)
    diag_by_win_test = build_diag_by_win(full_diag_map, test_graphs)
    # --- condition?specific positive counts per window ---
    cond = focus_idx   # e.g. 3 for diabetes
    train_pos_counts = []
    for wi, g in enumerate(train_graphs):
        if ('patient','has','condition') in g.edge_types:
            ei = g['patient','has','condition'].edge_index
            # count only those edges where the condition == cond
            count_cond = (ei[1] == cond).sum().item()
        else:
            count_cond = 0
        train_pos_counts.append(count_cond)

    # now build sample weights: every window gets at least weight=1,
    # windows with positives get boosted by (1 + cnt)^gamma
    gamma = 0.5   # you can tune (0.0 = uniform sampling, 1.0 = linear to count)
    sample_weights = [(1 + cnt)**gamma for cnt in train_pos_counts]
    # 2) Compute global pos weight
    total_pos = sum(
        g[('patient', 'has', 'condition')].edge_index.size(1)
        for g in train_graphs
        if ('patient', 'has', 'condition') in g.edge_types
    )
    total_neg = total_pos * NEGATIVE_MULTIPLIER
    global global_pos_weight
    global_pos_weight = torch.tensor([1.0], device=device)

    # 3) Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4) DataLoaders
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,    # so that high?weight windows can appear multiple times
    )
    train_loader = DataLoader(train_graphs, batch_size=1,sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_graphs, batch_size=1, num_workers=4)
    test_loader = DataLoader(test_graphs, batch_size=1, num_workers=4)

    # 5) Prepare a placeholder for pseudo?labels
    pseudo_label_by_window = {wi: {} for wi in range(len(train_graphs))}

    # 6) Training loop
    history = []
    EPOCHS =200
    pseudo_eps = 0.90
    max_pr_auc = 0.5
    WARMUP_EPOCHS = 5

    for epoch in range(1, EPOCHS + 1):
        #  warm-up phase
        if epoch <= WARMUP_EPOCHS:
            print(f"[Epoch {epoch}] Warm-up: training only link_head")
            # freeze backbone, Cox head, and patient_classifier
            for module in (model, cox_head, patient_classifier):
                for p in module.parameters():
                    p.requires_grad = False
            # keep link_head AND joint_head trainable
            for p in link_head.parameters():
                p.requires_grad = True
            for p in joint_head.parameters():
                p.requires_grad = True
        else:
            print(f"[Epoch {epoch}] Unfreezing all modules")
            # unfreeze everything
            for module in (model, link_head, cox_head, patient_classifier, joint_head):
                for p in module.parameters():
                    p.requires_grad = True

        # every 5 epochs, regenerate pseudo?labels
        if USE_PSEUDO and epoch > 20 and (epoch-1)%5==0:
            # a) recompute frozen condition embeddings
            with torch.no_grad():
                C = in_dims['condition']
                eye = torch.eye(C, device=device)
                h = model.input_proj['condition'](eye)
                h = F.relu(h);
                h = F.relu(h)
                h = F.relu(model.shrink['condition'](h))
                cond_embeds = model.linear_proj['condition'](h)

            # b) generate and inject pseudo?edges
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

            # c) build the per?window label map
            pseudo_label_by_window.clear()
            for wi, edges in enumerate(pse):
                m = {}
                for i, c in edges:
                    if i not in m:
                        m[i] = c
                pseudo_label_by_window[wi] = m

        # d) one epoch of train()
        train_loss = train(
            model, link_head, cox_head, patient_classifier,
            train_loader, optimizer, device,
            pseudo_label_by_window
        )

        # e) recompute condition embeds for eval
        with torch.no_grad():
            C = in_dims['condition']
            eye = torch.eye(C, device=device)
            h = model.input_proj['condition'](eye)
            h = F.relu(h);
            h = F.relu(h)
            h = F.relu(model.shrink['condition'](h))
            train_cond_embeds = model.linear_proj['condition'](h)

        # f) validation
        val_metrics = evaluate(
            model, link_head, cox_head, patient_classifier,
            val_loader, device,
            NEGATIVE_MULTIPLIER, diag_by_win_val,
            train_cond_embeds, plot_idx = plot_idx
        )
        avg_link, avg_cox, avg_node_ce, node_acc, cidx_list, mean_cidx, link_auc, pr_auc, pr_per_cond, val_scores = val_metrics

        scheduler.step(pr_per_cond[plot_idx])

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
            "timestamp": datetime.datetime.now().isoformat(),
        })
        print(f"[Epoch {epoch}] Train {train_loss:.4f} | ValLink {avg_link:.4f} "
            f"Val Cox Loss: {avg_cox:.4f} |"
            f"Node Acc: {node_acc} |"
            f"Val C-Index: {cidx_list} |"
            f"PR AUC per condition: {pr_per_cond}")

        if max_pr_auc < pr_per_cond[plot_idx]:
            max_pr_auc = pr_per_cond[plot_idx]

            # ? 1) Build the list of true diagnosis windows per window ?
            val_true_windows = [
                # for each window wi, collect the window?indices where a diagnosis of plot_idx occurs
                [wi for (p, c) in diag_by_win_val.get(wi, []) if c == plot_idx]
                for wi in range(len(val_graphs))
            ]

            # ? 2) Precompute the set of all patient IDs in val_graphs who ever get this diagnosis ?
            pos_patients = {
                p
                for (win, pairs) in diag_by_win_val.items()
                for (p, c) in pairs
                if c == plot_idx
            }

            # ? 3) Split val_scores into two groups ?
            val_scores_diag = []
            val_scores_non = []

            # zip together each window?s score?array with the corresponding batch
            for scores, batch in zip(val_scores, val_loader):
                # flatten batch['patient'].name into a flat list of RegistrationCodes
                names = [
                    code
                    for sub in batch['patient'].name
                    for code in (sub if isinstance(sub, list) else [sub])
                ]

                # boolean mask: True for those who will be diagnosed
                mask = np.array([code in pos_patients for code in names])

                arr = np.asarray(scores)
                val_scores_diag.append(arr[mask])
                val_scores_non.append(arr[~mask])

            # ? 4) Plot both curves together ?
            os.makedirs("outputs", exist_ok=True)
            plot_link_prediction_curves(
                link_scores_diag=val_scores_diag,
                link_scores_non=val_scores_non,
                diag_windows=val_true_windows,
                disease_name=chosen[plot_idx],
                save_path=f"outputs/val_{chosen[plot_idx].replace(' ', '_')}_epoch{epoch}.png",
            )


    # 7) Save history
    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/history_{ts}.json", "w") as f:
        json.dump(history, f, indent=2)

    # 8) Final test?time inference for plotting
    test_scores = []
    true_windows = []
    model.eval()
    link_head.eval()
    with torch.no_grad():
        # recompute final condition embeds
        eye = torch.eye(in_dims['condition'], device=device)
        h = model.input_proj['condition'](eye)
        h = F.relu(h)
        h = F.relu(h)
        h = F.relu(model.shrink['condition'](h))
        cond_embeds = model.linear_proj['condition'](h)

        for wi, batch in enumerate(DataLoader(test_graphs, batch_size=1)):
            batch = batch.to(device)
            out = model(batch.x_dict,
                        {et: batch[et].edge_index for et in batch.edge_types if et != ('patient', 'has', 'condition')},
                        {})
            P = out['patient']
            if P.size(0) == 0:
                test_scores.append(torch.zeros(0, dtype=torch.float, device=device))
                true_windows.append([])
                continue
            # score every patient ? disease 3
            idx = 3
            ei = torch.stack([torch.arange(P.size(0), device=device), torch.full((P.size(0),), idx, device=device)],
                             dim=0)
            scores = torch.sigmoid(link_head(P, cond_embeds, ei))  # [N_pat]
            test_scores.append(scores.cpu().numpy())
            # record true diag-window(s) for disease 3 from your mapping
            tw = [wi for (p, c) in diag_by_win_test.get(wi, []) if c == idx]
            true_windows.append(tw)

    # 9) Plot & save
    os.makedirs("outputs", exist_ok=True)
    plot_link_prediction_curves(
        link_scores=test_scores,
        diag_windows=true_windows,
        disease_name=chosen[3],
        save_path="outputs/diabetes_curves.png"
    )

    # 10) Save model checkpoint
    os.makedirs("models", exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'link_head': link_head.state_dict(),
        'cox_head': cox_head.state_dict(),
        'patient_classifier': patient_classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': EPOCHS,
    }
    torch.save(ckpt, f"models/checkpoint_{EPOCHS}.pth")
    print(f"Saved checkpoint to models/checkpoint_{EPOCHS}.pth")


if __name__ == "__main__":
    mp.freeze_support()
    main()