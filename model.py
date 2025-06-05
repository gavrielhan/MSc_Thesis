from link_prediction_head import CoxHead
from link_prediction_head import LinkMLPPredictor
from link_prediction_head import CosineLinkPredictor
from risk_calibration import fit_absolute_risk_calibrator
import torch
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
        hidden_dim: int = 128,
        out_dim:   int = 128,
        num_heads: int = 4,
        dropout:   float = 0.2
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
                    convs[et] = GCNConv(in_d, out_d)
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
    def __init__(self, link_weight: float = 1.0, cox_weight: float = 0.2):
        super().__init__()
        # store as simple floats (or as buffers if you like)
        self.link_weight = link_weight
        self.cox_weight  = cox_weight

    def forward(self, link_loss: torch.Tensor, cox_loss: torch.Tensor) -> torch.Tensor:
        return self.link_weight * link_loss + self.cox_weight * cox_loss

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

        with torch.no_grad():
            logits = link_head(P, C_fixed, candidate_ei)  # [M]
            probs  = torch.sigmoid(logits)                # [M]

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


# get gender informaition per patients
study_ids = [10, 1001, 1002]
bm = BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first').df.join(BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first').df_metadata)

age_gender = bm[['age', 'gender', 'yob']].reset_index()
del age_gender['Date']
age_gender = age_gender[~age_gender['gender'].isna()]
age_gender = age_gender[~age_gender['age'].isna()]

gender_dictionary = {1:'male', 0:'female'}

age_gender.loc[:,'gender'] = age_gender['gender'].fillna(0).map(gender_dictionary)
# gender_df has columns ['RegistrationCode','gender'], where gender is 'male' or 'female'
gender_map = dict(zip(age_gender['RegistrationCode'], age_gender['gender']))

BREAST_IDX = chosen.index("Malignant neoplasms of breast")

all_graphs =  torch.load("glucose_sleep_graphs_3d.pt", weights_only = False)


# Load your pre-split graph lists
train_graphs = torch.load("split/train_graphs_3d.pt", weights_only = False)
val_graphs = torch.load("split/val_graphs_3d.pt", weights_only = False)
test_graphs = torch.load("split/test_graphs_3d.pt", weights_only = False)


# Load our diagnosis?window mapping
# Load the full diagnosis?window mapping:
with open("diagnosis_mapping.json") as f:
    full_diag_map = json.load(f)


def build_diag_by_win(full_map, graphs):
    pts = {p for g in graphs for p in g['patient'].name}
    m = defaultdict(list)
    for d in full_map:
        if d["patient"] in pts:
            m[d["window"]].append((d["patient"], d["cond"]))
    return m

diag_by_win_train = build_diag_by_win(full_diag_map, train_graphs)
diag_by_win_val   = build_diag_by_win(full_diag_map, val_graphs)
diag_by_win_test  = build_diag_by_win(full_diag_map, test_graphs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEGATIVE_MULTIPLIER = 2  # you can adjust this multiplier later
HORIZON = 10   # how many graphs ahead to predict over

# Count total positive edges in the training split
total_pos = sum(
    g[('patient','has','condition')].edge_index.size(1)
    for g in train_graphs
    if ('patient','has','condition') in g.edge_types
)
# Negatives sampled per batch is NEGATIVE_MULTIPLIER × positives
total_neg = total_pos * NEGATIVE_MULTIPLIER

global_pos_weight = total_neg / (total_pos + 1e-6)
global_pos_weight = torch.tensor([global_pos_weight], device=device)



#  HYPERPARAMETERS FOR PSEUDO-LABELING
# ----------------------------------------------------------------------------------
EPS_PSEUDO    = 0.90   # ?: confidence threshold (choose between 0.8?0.95 in practice)
LAMBDA_PSEUDO = 0.50   # ?: weight on the pseudo?label loss
# ----------------------------------------------------------------------------------

def train(model, link_head, cox_head, patient_classifier, loader, optimizer, device):
    model.train()
    link_head.train()
    cox_head.train()
    patient_classifier.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)
        # ??? Sanity?check inputs ???????????????????????????????????????????????????????????????????
        for ntype, x in batch.x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError(f"[Batch {batch_idx}] Bad input on node '{ntype}': contains NaN or Inf")

        # ??? Skip if no useful edges at all ???????????????????????????????????????????????????????
        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types
            and ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            # nothing to train on here
            continue

        optimizer.zero_grad()
        # ??? Build edge_attr_dict for signature edges ?????????????????????????????????????????
        edge_attr_dict = {}
        rel = ('patient', 'to', 'signature')
        if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
            edge_attr_dict[rel] = batch[rel].edge_attr

        # ??? 1) Forward through HeteroGAT ?????????????????????????????????????????????????????
        out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)

        # ??? Sanity?check GNN outputs ?????????????????????????????????????????????????????????????
        for ntype, h in out.items():
            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError(f"[Batch {batch_idx}] NaNs in model output for '{ntype}'")

        patient_embeds   = out['patient']    # [N_pat, out_dim]
        condition_embeds = out['condition']  # [C, out_dim]

        # ??? Prepare ZERO tensor to ?fill in? empty?loss cases ???????????????????????????????????
        zero = patient_embeds.sum() * 0.0

        # ??? 2) Link?prediction loss over horizon ????????????????????????????????????????????????
        window_idx = int(batch.window)  # which sliding?window this graph corresponds to
        # (You have already stored `batch.window = wi` when building `train_graphs`.)

        # 2a) Collect real ?future? positives exactly as before (HORIZON?based) ?????????????????????
        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend([
                (p, ci)
                for (p, ci) in diag_by_win_train.get(w, [])
            ])

        # 2b) Filter out male?breast (just like you had done) ????????????????????????????????????
        flat_names = [
            n for sub in batch['patient'].name
                for n in (sub if isinstance(sub, list) else [sub])
        ]  # e.g. ['patA', 'patB', ?]

        # Build a bool mask per patient in this batch
        is_female = torch.tensor(
            [gender_map.get(p, 'female') == 'female' for p in flat_names],
            device=device, dtype=torch.bool
        )

        # Keep only (p,ci) if not (breast & male)
        filtered = []
        for (p,ci) in pos_pairs:
            if ci == BREAST_IDX:
                # if it?s breast, only keep if that patient is female
                if p in flat_names:
                    p_idx = flat_names.index(p)
                    if is_female[p_idx]:
                        filtered.append((p,ci))
            else:
                filtered.append((p,ci))
        pos_pairs = filtered

        # 2c) Convert names?indices in [0 .. N_pat-1]
        src_nodes = []
        dst_nodes = []
        for (p,ci) in pos_pairs:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        if src_nodes:
            pos_ei = torch.stack([
                torch.tensor(src_nodes, device=device),
                torch.tensor(dst_nodes, device=device)
            ], dim=0)  # shape [2, N_pos]
        else:
            pos_ei = None

        # 2d) Compute BCE(u) for link?prediction same as before
        if pos_ei is not None:
            pos_preds  = link_head(patient_embeds, condition_embeds, pos_ei)
            pos_labels = torch.ones_like(pos_preds)

            num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, condition_embeds.size(0), (num_neg,), device=device)

            # enforce female?only for breast negatives:
            mask_b = (neg_dst == BREAST_IDX)
            if mask_b.any():
                female_idx = torch.where(is_female)[0]
                neg_src[mask_b] = female_idx[
                    torch.randint(0, female_idx.size(0), (int(mask_b.sum()),), device=device)
                ]

            neg_ei    = torch.stack([neg_src, neg_dst], dim=0)
            neg_preds = link_head(patient_embeds, condition_embeds, neg_ei)
            neg_labels= torch.zeros_like(neg_preds)

            preds  = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            link_loss = F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=global_pos_weight
            )
        else:
            link_loss = zero.clone()

        # ??? 3) Cox regression loss (unchanged) ????????????????????????????????????????????????
        cox_loss = zero.clone()
        valid_cox_conds = 0
        num_conditions = condition_embeds.size(0)

        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            events   = batch['patient'].event      # [N_pat, C]
            durations= batch['patient'].duration   # [N_pat, C]
            risk_scores = cox_head(patient_embeds) # [N_pat, C]

            for ci in range(num_conditions):
                ev_col = events[:, ci]
                if ev_col.sum().item() > 0:
                    dur_col   = durations[:, ci]
                    score_col = risk_scores[:, ci]

                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        score_col.unsqueeze(1),
                        dur_col.unsqueeze(1),
                        ev_col.unsqueeze(1)
                    )
                    cox_loss += loss_ci
                    valid_cox_conds += 1

            if valid_cox_conds > 0:
                cox_loss = cox_loss / valid_cox_conds
        else:
            print(f"[Batch {batch_idx}] Missing duration/event attrs")

        # ??? 4) Pseudo?label cross?entropy loss L?? ????????????????????????????????????????????
        #    For this batch?s window_idx, look up exactly one pseudo?label per patient:
        node_ps_loss = zero.clone()

        label_map = pseudo_label_by_window.get(window_idx, {})
        # label_map: { patient_idx_in_this_graph ? cond_idx }  or {} if none

        if label_map:
            # Build a tensor of indices and labels
            node_indices = torch.tensor(
                list(label_map.keys()),  # e.g. [3, 10, 25, ?]
                device=device, dtype=torch.long
            )  # shape [N_pseudo]

            pseudo_labels = torch.tensor(
                [label_map[i.item()] for i in node_indices.cpu().tolist()],
                device=device, dtype=torch.long
            )  # shape [N_pseudo], each in [0..C-1]

            # Forward pass through patient_classifier
            # patient_embeds: [N_pat, out_dim]; we index rows by node_indices
            selected_embeds = patient_embeds[node_indices]  # [N_pseudo, out_dim]
            logits = patient_classifier(selected_embeds)    # [N_pseudo, C]

            node_ps_loss = F.cross_entropy(logits, pseudo_labels)
        else:
            node_ps_loss = zero.clone()

        # ??? 5) Total loss = JointHead(link_loss, cox_loss) + ? ? node_ps_loss ????????????
        lambda_1 = 0.5  # you can tune this pseudo?label weight
        total_batch_loss = joint_head(link_loss, cox_loss) + lambda_1 * node_ps_loss

        # ??? 6) Backprop ???????????????????????????????????????????????????????????????????
        for name, l in (("link_loss", link_loss),
                        ("cox_loss",  cox_loss),
                        ("ps_loss",   node_ps_loss)):
            if isinstance(l, torch.Tensor) and not torch.isfinite(l).all():
                raise RuntimeError(f"[Batch {batch_idx}] {name} is non-finite: {l}")
            elif not isinstance(l, torch.Tensor) and not math.isfinite(l):
                raise RuntimeError(f"[Batch {batch_idx}] {name} is non-finite: {l}")

        total_batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(link_head.parameters())
            + list(cox_head.parameters())
            + list(patient_classifier.parameters())
            + list(joint_head.parameters()),
            max_norm=1.0
        )

        # Final step: optimizer step
        optimizer.step()

        total_loss += total_batch_loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)




@torch.no_grad()
def evaluate(model, link_head, cox_head, patient_classifier,loader,device,NEGATIVE_MULTIPLIER,diag_by_win_map,frozen_condition_embeddings):
    model.eval()
    link_head.eval()
    cox_head.eval()
    patient_classifier.eval()   # <? new

    total_link_loss = 0.0
    total_cox_loss  = 0.0
    total_node_loss = 0.0   # accumulate pseudo?node (CE) over all windows
    total_node_count = 0    # count number of patients on which we computed CE
    total_node_correct = 0  # to track accuracy

    all_link_preds,  all_link_labels  = [], []
    per_cond_preds,  per_cond_labels  = {}, {}
    all_risk_scores, all_durations, all_events = [], [], []
    num_batches = 0

    # Number of conditions (C)
    num_conditions = cox_head.linear.out_features
    for ci in range(num_conditions):
        per_cond_preds[ci]  = []
        per_cond_labels[ci] = []

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # ??? 1) Skip if no patients in this graph ????????????????????????????????????????
        if 'patient' not in batch.x_dict or batch['patient'].x.size(0) == 0:
            continue

        # ??? 2) Prepare edge_index_dict & edge_attr_dict for GNN forward ?????????????????
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient', 'has', 'condition')
        }
        edge_attr_dict = {}
        rel = ('patient', 'to', 'signature')
        if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
            edge_attr_dict[rel] = batch[rel].edge_attr

        # ??? 3) Forward pass through HeteroGAT ???????????????????????????????????????????
        out = model(batch.x_dict, edge_index_dict, edge_attr_dict)
        P = out['patient']    # [N_pat, out_dim]
        C_emb = frozen_condition_embeddings.to(device)  # [C, out_dim]

        # ??? 4) Skip if embeddings are empty ?????????????????????????????????????????????
        if P.size(0) == 0 or C_emb.size(0) == 0:
            continue

        # ??? 5) Build ?future positives? over the evaluation horizon ?????????????????????
        #      exactly as in train(): gather (patient_id, cond_idx) for windows [t+1 .. t+HORIZON].
        flat_names = [
            n for sub in batch['patient'].name
                for n in (sub if isinstance(sub, list) else [sub])
        ]  # e.g. ['patA', 'patB', ?]  length = N_pat

        # Build a bool mask for female?only logic for breast:
        is_female_flat = torch.tensor(
            [gender_map.get(p, 'female') == 'female' for p in flat_names],
            device=device, dtype=torch.bool
        )

        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend(diag_by_win_map.get(w, []))

        # Filter out male?breast exactly as in train()
        filtered = []
        for (p, ci) in pos_pairs:
            if ci == BREAST_IDX:
                if p in flat_names:
                    idx = flat_names.index(p)
                    if is_female_flat[idx]:
                        filtered.append((p, ci))
            else:
                filtered.append((p, ci))
        pos_pairs = filtered

        # Convert each (p,ci) into (patient_node_idx, cond_idx)
        src_nodes = []
        dst_nodes = []
        for (p, ci) in pos_pairs:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        if not src_nodes:
            # No true positives in this window?s horizon ? skip link + node loss
            continue

        pos_ei = torch.stack([
            torch.tensor(src_nodes, device=device),
            torch.tensor(dst_nodes, device=device)
        ], dim=0)  # [2, N_pos]

        # ??? 6) Link?prediction loss & metrics (per?condition) ??????????????????????????
        for cond_idx in range(num_conditions):
            mask = (pos_ei[1] == cond_idx)
            if not mask.any():
                continue

            # Positive logits:
            pe  = pos_ei[:, mask]                    # [2, #pos_{ci}]
            pp  = link_head(P, C_emb, pe)            # [#pos_{ci}]
            pl  = torch.ones_like(pp)

            # Negative sampling:
            n_neg   = pp.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, P.size(0), (n_neg,), device=device)
            neg_dst = torch.full((n_neg,), cond_idx, device=device)

            # If cond_idx == breast, restrict neg_src to female only:
            if cond_idx == BREAST_IDX:
                female_idx = torch.where(is_female_flat)[0]
                mask_b = (neg_dst == BREAST_IDX)
                neg_src[mask_b] = female_idx[
                    torch.randint(0, female_idx.size(0), (int(mask_b.sum()),), device=device)
                ]

            ne   = torch.stack([neg_src, neg_dst], dim=0)  # [2, n_neg]
            npred= link_head(P, C_emb, ne)                 # [n_neg]
            nl   = torch.zeros_like(npred)

            # Collect for metrics:
            preds  = torch.cat([pp, npred], dim=0)
            labels = torch.cat([pl, nl], dim=0)
            all_link_preds .append(preds.cpu())
            all_link_labels.append(labels.cpu())
            per_cond_preds [cond_idx].append(preds.cpu())
            per_cond_labels[cond_idx].append(labels.cpu())

            # Accumulate BCE loss:
            total_link_loss += F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=global_pos_weight
            ).item()

        # ??? 7) Cox regression loss (unchanged) ???????????????????????????????????????????
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risks = cox_head(P)                      # [N_pat, C]
            durs  = batch['patient'].duration        # [N_pat, C]
            evts  = batch['patient'].event           # [N_pat, C]

            all_risk_scores.append(risks.cpu())
            all_durations .append(durs.cpu())
            all_events    .append(evts.cpu())

            # Compute per?condition partial?likelihood:
            for ci in range(num_conditions):
                ev_col = evts[:, ci]
                if ev_col.sum().item() > 0:
                    dur_col   = durs[:, ci]
                    score_col = risks[:, ci]
                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        score_col.unsqueeze(1),
                        dur_col.unsqueeze(1),
                        ev_col.unsqueeze(1)
                    )
                    total_cox_loss += loss_ci.item()
            # Note: in validate we keep the sum of losses.  (You can average by C if you like.)

        # ??? 8) Node?classifier CE loss on ?true? future positives ?????????????????????????
        # Find all **unique** patient?indices that actually got diagnosed in [t+1..t+HORIZON].
        # Build a mapping { node_idx : cond_idx } for those.
        label_map = {}
        for (p, ci) in pos_pairs:
            if p in flat_names:
                idx = flat_names.index(p)
                # If the same patient appears twice with different ci, we just overwrite:
                #   e.g. if they had two conditions in the horizon, we use the *last one* here.
                label_map[idx] = ci

        if len(label_map) > 0:
            node_indices = torch.tensor(
                list(label_map.keys()), device=device, dtype=torch.long
            )  # e.g. [i1, i2, i5, ?]

            pseudo_labels = torch.tensor(
                [label_map[i.item()] for i in node_indices.cpu().tolist()],
                device=device, dtype=torch.long
            )  # shape [#unique_patients]

            # Forward through patient_classifier:
            selected_embeds = P[node_indices]                # [#unique_patients, out_dim]
            logits = patient_classifier(selected_embeds)     # [#unique_patients, C]

            node_ce = F.cross_entropy(logits, pseudo_labels)
            total_node_loss += node_ce.item()

            # Optional: track accuracy on those patients
            preds_class = logits.argmax(dim=1)  # [#unique_patients]
            total_node_correct += (preds_class == pseudo_labels).sum().item()
            total_node_count   += node_indices.size(0)
        else:
            # no real?future?positives ? skip node CE
            pass

        num_batches += 1

    # ??? 9) Finalize all averaged losses & metrics ????????????????????????????????????????
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)
    avg_node_ce = (total_node_loss / total_node_count) if total_node_count > 0 else None
    node_acc = (total_node_correct / total_node_count) if total_node_count > 0 else None

    # Compute global AUC / PR?AUC for link?prediction
    if all_link_preds:
        y_true  = torch.cat(all_link_labels).numpy()
        y_score = torch.cat(all_link_preds).numpy()
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc   = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    # PR?AUC per condition
    pr_per_cond = [None] * num_conditions
    for ci in range(num_conditions):
        if per_cond_preds[ci]:
            y_t = torch.cat(per_cond_labels[ci]).numpy()
            y_s = torch.cat(per_cond_preds [ci]).numpy()
            pr_per_cond[ci] = average_precision_score(y_t, y_s)

    # C?index per condition
    c_idxs, mean_c = [], None
    if all_risk_scores:
        R = torch.cat(all_risk_scores).numpy()
        D = torch.cat(all_durations).numpy()
        E = torch.cat(all_events   ).numpy()
        for ci in range(num_conditions):
            mask = E[:, ci] >= 0
            if mask.sum() > 0:
                try:
                    c = concordance_index(D[mask, ci], -R[mask, ci], E[mask, ci])
                except ZeroDivisionError:
                    c = None
            else:
                c = None
            c_idxs.append(c)
        valid = [c for c in c_idxs if c is not None]
        if valid:
            mean_c = sum(valid) / len(valid)

    return (
        avg_link,                # average link?prediction BCE
        avg_cox,                 # average Cox partial?likelihood
        avg_node_ce,             # average node cross?entropy on true positives
        node_acc,                # classification accuracy on those same patients
        c_idxs,                  # list of per?condition C?indices
        mean_c,                  # mean C?index
        link_auc,                # global AUC
        pr_auc,                  # global PR?AUC
        pr_per_cond              # PR?AUC per condition
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
    hidden_dim=128,
    out_dim=128
).to(device)

#link_head = LinkMLPPredictor(input_dim=128).to(device)
link_head = CosineLinkPredictor(input_dim=128, init_scale=1.0, use_bias=True).to(device)
cox_head = CoxHead(input_dim=128, num_conditions=7).to(device)

joint_head = JointHead(link_weight=1.0, cox_weight=1.5).to(device)
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

# Option B: reduce on plateau of validation link?loss
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
 )
# Create DataLoaders for each split
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True, num_workers = 4)
val_loader = DataLoader(val_graphs, batch_size=1, num_workers = 4)
test_loader = DataLoader(test_graphs, batch_size=1, num_workers = 4)

num_conditions = 7
with torch.no_grad():
    # 1) Start from one?hot identity for all conditions
    eye = torch.eye(num_conditions, device=device)  # [C, C]

    # 2) Input?project into hidden_dim
    h = model.input_proj['condition'](eye)          # [C, hidden_dim]
    h = F.relu(h)                                   # first ?fake? conv?fallback
    h = F.relu(h)                                   # second ?fake? conv?fallback

    # 3) Shrink from hidden_dim?out_dim (only if still at hidden_dim)
    #    (Your forward does exactly this before final projection.)
    h = F.relu(model.shrink['condition'](h))        # [C, out_dim]

    # 4) Final linear projection to produce the embedding that link_head expects
    cond_embeds = model.linear_proj['condition'](h)  # [C, out_dim]
    # cond_embeds is now a [C×out_dim] Tensor on `device`

# Choose a confidence threshold eps (e.g. 0.9 or 0.95)
pseudo_eps = 0.9

# NOTE: you must have already defined `gender_map` and `BREAST_IDX` somewhere above
#   `gender_map: { RegistrationCode ? 'male' or 'female' }`
#   `BREAST_IDX` = index of ?breast cancer? in your condition list (e.g. 6 if it?s the 7th)

pseudo_edges_per_graph = generate_pseudo_labels(
    train_graphs,
    model=model,
    link_head=link_head,
    cond_embeds=cond_embeds,
    eps=pseudo_eps,
    device=device,
    gender_map=gender_map,
    BREAST_IDX=BREAST_IDX
)
# ??? Add pseudo edges into each training graph ???????????????????????????????
for wi, g in enumerate(train_graphs):
    pse = pseudo_edges_per_graph[wi]  # list of (i_node, c_node)

    if not pse:
        continue

    # Extract existing real edges (if any):
    if ('patient','has','condition') in g.edge_types:
        old_ei = g['patient','has','condition'].edge_index  # [2, E_real]
    else:
        old_ei = torch.empty((2, 0), dtype=torch.long)

    # Build a tensor of new pseudo?edges
    src = torch.tensor([i for (i,c) in pse], dtype=torch.long, device=device)
    dst = torch.tensor([c for (i,c) in pse], dtype=torch.long, device=device)
    new_ei = torch.stack([src, dst], dim=0)  # [2, E_pseudo]

    # Concatenate real + pseudo
    merged_ei = torch.cat([old_ei.to(device), new_ei], dim=1)  # [2, E_real + E_pseudo]
    g['patient','has','condition'].edge_index = merged_ei

    # Also add the reverse edges in ('condition','has_rev','patient'):
    merged_rev = merged_ei.flip(0)  # flip rows?[condition, patient]
    g['condition','has_rev','patient'].edge_index = merged_rev

pseudo_label_by_window = {}  # will map window w ? { patient_idx: cond_idx }

for w, edge_list in enumerate(pseudo_edges_per_graph):
    # edge_list is a list like [(i?, c?), (i?, c?), ?], already sorted by confidence descending.
    seen = set()
    label_map = {}
    for (p_idx, c_idx) in edge_list:
        if p_idx not in seen:
            label_map[p_idx] = c_idx
            seen.add(p_idx)
    pseudo_label_by_window[w] = label_map
EPOCHS = 200
history = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, link_head, cox_head,joint_head,train_loader, optimizer, device)
    with torch.no_grad():
        # a) Build C×C identity (one?hot) for all conditions
        C = in_dims['condition']
        eye = torch.eye(C, device=device)  # [C, C]

        # 2) Input proj
        h = model.input_proj['condition'](eye)  # [C, hidden_dim]

        # 3) Three ?conv?fallback? steps (just ReLU)
        h = F.relu(h)
        h = F.relu(h)

        # **now** shrink to 128
        h = F.relu(model.shrink['condition'](h))  #  [7×128]

        # final projection (now matches 128?128)
        train_cond_embeds = model.linear_proj['condition'](h)
        # Now pass train_cond_embeds to evaluation
    val_avg_link, avg_cox, cidx_list, mean_cidx, link_auc, pr_auc, pr_auc_per_cond = evaluate(
        model, link_head, cox_head, val_loader, device,
        NEGATIVE_MULTIPLIER, diag_by_win_val, train_cond_embeds, patient_classifier=patient_classifier
    )
    # After training finished
    results = {
        "final_train_loss": train_loss,
        "final_val_link_loss": val_avg_link,
        "final_val_link_auc": link_auc,
        "final_val_pr_auc": pr_auc,
        "final_val_cox_loss": avg_cox,
        "c_indices_per_condition": cidx_list,
        "pr_auc_per_condition": pr_auc_per_cond,
        "mean_c_index": mean_cidx,
        "epoch": epoch,
        "NEGATIVE_MULTIPLIER": NEGATIVE_MULTIPLIER,
        "timestamp": datetime.datetime.now().isoformat()
    }
    history.append(results)
    # ? for StepLR:
    #scheduler.step()

    # ? if using ReduceLROnPlateau, comment out the above line and instead do:
    scheduler.step(val_avg_link)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
          f"Val Link Loss: {val_avg_link:.4f} |"
          f"Val Cox Loss: {avg_cox:.4f} | Val C-Index: {cidx_list}")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/results_{timestamp}.json"

with open(filename, 'w') as f:
    json.dump(history, f, indent=2)

print(f"Saved training results to {filename}")
