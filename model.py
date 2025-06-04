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

def train(model, predictor, cox_head, joint_head, loader, optimizer, device):
    """
    Combined training loop with Balanced Pseudo?Label Generation (Section 3.2).
    """
    model.train()
    predictor.train()
    cox_head.train()


    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)

        # 1) Sanity?check: no NaNs / Infs in node features
        for ntype, x in batch.x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError(
                    f"[Batch {batch_idx}] Bad input on node '{ntype}': contains NaN or Inf"
                )

        # 2) Skip if no useful edges
        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types
            and ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            continue

        optimizer.zero_grad()

        # 3) Forward through HeteroGAT
        edge_attr_dict = {}
        rel_sig = ('patient', 'to', 'signature')
        if rel_sig in batch.edge_types and hasattr(batch[rel_sig], 'edge_attr'):
            edge_attr_dict[rel_sig] = batch[rel_sig].edge_attr

        out_dict = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        patient_embeds   = out_dict['patient']    # [N_pat, D_hidden]
        condition_embeds = out_dict['condition']  # [C,     D_hidden]

        zero      = patient_embeds.sum() * 0.0
        link_loss = zero.clone()
        cox_loss  = zero.clone()

        # 4) HORIZON?based link?prediction loss
        window_idx = int(batch.window)
        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend([
                (p, ci) for (p, ci) in diag_by_win_train.get(w, [])
            ])

        # Flatten patient IDs
        names = [
            n for sub in batch['patient'].name
                 for n in (sub if isinstance(sub, list) else [sub])
        ]
        is_female = torch.tensor(
            [gender_map.get(p, 'female') == 'female' for p in names],
            device=device, dtype=torch.bool
        )

        # Remove (male, BREAST_IDX) from positives
        pos_pairs = [
            (p, ci) for (p, ci) in pos_pairs
            if not (ci == BREAST_IDX and not is_female[names.index(p)])
        ]

        src_nodes, dst_nodes = [], []
        for p, ci in pos_pairs:
            if p in names:
                src_nodes.append(names.index(p))
                dst_nodes.append(ci)

        if src_nodes:
            pos_ei = torch.stack([
                torch.tensor(src_nodes, device=device),
                torch.tensor(dst_nodes, device=device),
            ], dim=0)
            pos_preds = predictor(patient_embeds, condition_embeds, pos_ei)
            pos_labels = torch.ones_like(pos_preds)

            num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, condition_embeds.size(0), (num_neg,), device=device)
            mask_b = (neg_dst == BREAST_IDX)
            if mask_b.any():
                female_idx = torch.where(is_female)[0]
                neg_src[mask_b] = female_idx[
                    torch.randint(0, female_idx.size(0), (int(mask_b.sum()),), device=device)
                ]
            neg_ei = torch.stack([neg_src, neg_dst], dim=0)
            neg_preds = predictor(patient_embeds, condition_embeds, neg_ei)
            neg_labels = torch.zeros_like(neg_preds)

            preds = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            link_loss = F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=global_pos_weight
            )
        else:
            link_loss = zero.clone()

        # 5) Cox?regression loss (per condition)
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            events    = batch['patient'].event    # [N_pat, C]
            durations = batch['patient'].duration # [N_pat, C]
            risk_scores = cox_head(patient_embeds)  # [N_pat, C]
            valid_cox_conds = 0
            for ci in range(condition_embeds.size(0)):
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
                cox_loss /= valid_cox_conds
        else:
            print(f"[Batch {batch_idx}] Missing duration/event attrs")

        L_original = joint_head(link_loss, cox_loss)

        # 6) Balanced Pseudo?Label Generation
        N_pat = patient_embeds.size(0)
        Cdim  = condition_embeds.size(0)

        row_repeat = torch.arange(N_pat, device=device).repeat_interleave(Cdim)
        col_tile   = torch.arange(Cdim,   device=device).repeat(N_pat)
        dense_ei = torch.stack([row_repeat, col_tile], dim=0)  # [2, N_pat*C]

        dense_logits = predictor(patient_embeds, condition_embeds, dense_ei)
        dense_logits = dense_logits.view(N_pat, Cdim)         # [N_pat, C]
        probs = F.softmax(dense_logits, dim=1)                # [N_pat, C]

        if ('patient', 'has', 'condition') in batch.edge_types:
            true_edges = batch[('patient', 'has', 'condition')].edge_index
            true_src   = true_edges[0]
            true_dst   = true_edges[1]
        else:
            true_src = torch.empty((0,), dtype=torch.long, device=device)
            true_dst = torch.empty((0,), dtype=torch.long, device=device)

        already_labeled_mask = torch.zeros(N_pat, dtype=torch.bool, device=device)
        if true_src.numel() > 0:
            already_labeled_mask[ true_src ] = True
        unlabeled_mask = ~already_labeled_mask

        counts_per_class = torch.zeros(Cdim, device=device, dtype=torch.long)
        for cidx in range(Cdim):
            counts_per_class[cidx] = (true_dst == cidx).sum()
        M_max = counts_per_class.max().item()
        Nhat  = (M_max - counts_per_class).clamp(min=0)

        pseudo_edges = []
        top_d = torch.argmax(probs, dim=1)  # [N_pat]
        candidate_pool = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)

        for cidx in range(Cdim):
            num_to_add = Nhat[cidx].item()
            if num_to_add <= 0 or candidate_pool.numel() == 0:
                continue
            sample_n = min(num_to_add * 3, candidate_pool.size(0))
            perm = torch.randperm(candidate_pool.size(0), device=device)[:sample_n]
            sample_idx = candidate_pool[perm]

            p_c = probs[sample_idx, cidx]
            mask = (top_d[sample_idx] == cidx) & (p_c > EPS_PSEUDO)
            if mask.sum().item() == 0:
                continue
            cand_indices = sample_idx[mask]
            cand_probs = p_c[mask]
            topk = min(num_to_add, cand_indices.size(0))
            _, topk_idxs = torch.topk(cand_probs, k=topk, largest=True)
            sel_patients = cand_indices[topk_idxs]
            for i_sel in sel_patients.tolist():
                pseudo_edges.append((i_sel, cidx))
            if sel_patients.numel() > 0:
                remove_mask = (candidate_pool.unsqueeze(1) == sel_patients.view(-1)).any(dim=1)
                candidate_pool = candidate_pool[~remove_mask]

        M_pseudo = len(pseudo_edges)
        if M_pseudo == 0:
            L_ps = zero.clone()
        else:
            pe_rows = torch.tensor(
                [p[0] for p in pseudo_edges],
                device=device, dtype=torch.long
            )
            pe_cols = torch.tensor(
                [p[1] for p in pseudo_edges],
                device=device, dtype=torch.long
            )
            all_logits_for_sel = dense_logits[pe_rows]  # [M_pseudo, C]
            target_labels     = pe_cols                # [M_pseudo]
            L_ps = F.cross_entropy(
                all_logits_for_sel,
                target_labels,
                reduction='mean'
            )

        # 7) Total loss and backward
        L_total = L_original + (LAMBDA_PSEUDO * L_ps)
        L_total.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(predictor.parameters()) +
            list(cox_head.parameters()) +
            list(joint_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += L_total.item()
        num_batches += 1

    return total_loss / max(1, num_batches)



@torch.no_grad()
def evaluate(model, predictor, cox_head, loader, device,
             NEGATIVE_MULTIPLIER, diag_by_win_map, frozen_condition_embeddings):
    """
    Evaluate loop over validation/test graphs.

    - Uses the same HORIZON?based logic for link-prediction as in train()
    - Applies the same female?only filtering for BREAST_IDX
    - Computes per?condition AUC/PR, plus Cox c?index
    """
    model.eval()
    predictor.eval()
    cox_head.eval()

    total_link_loss = 0.0
    total_cox_loss  = 0.0
    all_link_preds  = []
    all_link_labels = []
    per_cond_preds  = {}
    per_cond_labels = {}
    all_risk_scores = []
    all_durations   = []
    all_events      = []
    num_batches     = 0

    # Number of conditions (Cdim)
    num_conditions = cox_head.linear.out_features
    for ci in range(num_conditions):
        per_cond_preds[ci]  = []
        per_cond_labels[ci] = []

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # 1) Skip if no patient nodes
        if 'patient' not in batch.x_dict or batch['patient'].x.size(0) == 0:
            continue

        # 2) Build edge_index_dict for all relations except the 'has?condition' edges
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient', 'has', 'condition')
        }

        # 3) Gather edge_attr for ('patient','to','signature'), if it exists
        edge_attr_dict = {}
        rel_sig = ('patient', 'to', 'signature')
        if rel_sig in batch.edge_types and hasattr(batch[rel_sig], 'edge_attr'):
            edge_attr_dict[rel_sig] = batch[rel_sig].edge_attr

        # 4) Forward pass through HeteroGAT
        out_dict = model(batch.x_dict, edge_index_dict, edge_attr_dict)
        P = out_dict['patient']    # [N_pat, D_hidden]
        C = frozen_condition_embeddings.to(device)  # [Cdim, D_hidden]

        # 5) Skip if embeddings are empty
        if P.size(0) == 0 or C.size(0) == 0:
            continue

        # 6) Build all positive pairs in [window_idx+1 .. window_idx+HORIZON]
        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend(diag_by_win_map.get(w, []))

        # 7) Flatten the list of patient IDs in this batch
        flat_names = [
            nid for sub in batch['patient'].name
                 for nid in (sub if isinstance(sub, list) else [sub])
        ]
        # 7.a) Build boolean mask "is_female_flat" for each node?index in this batch
        is_female_flat = torch.tensor(
            [gender_map.get(pid, 'female') == 'female' for pid in flat_names],
            device=device, dtype=torch.bool
        )

        # 8) Drop any (male, BREAST_IDX) from pos_pairs
        pos_pairs = [
            (p, ci) for (p, ci) in pos_pairs
            if not (ci == BREAST_IDX and not is_female_flat[flat_names.index(p)])
        ]

        # 9) Convert pos_pairs (RegistrationCode, condition_idx) ? node?indices
        src_nodes, dst_nodes = [], []
        for (p, ci) in pos_pairs:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        # 10) If no positives remain in horizon, skip this batch
        if len(src_nodes) == 0:
            continue

        pos_ei = torch.stack([
            torch.tensor(src_nodes, device=device),
            torch.tensor(dst_nodes, device=device)
        ], dim=0)  # shape [2, #pos]

        # ?? LINK?PREDICTION LOSS & METRICS ?? #
        for cond_idx in range(num_conditions):
            # 10.a) Mask for positives of this condition
            mask_pos = (pos_ei[1] == cond_idx)
            # 10.b) Drop any male?breast positives if cond_idx == BREAST_IDX
            if cond_idx == BREAST_IDX:
                mask_pos &= is_female_flat[pos_ei[0]]

            if not mask_pos.any():
                continue

            # 10.c) Positive edge_index for this cond_idx
            pe = pos_ei[:, mask_pos]
            pp = predictor(P, C, pe)           # [#pos, ]
            pl = torch.ones_like(pp)           # [#pos, ]

            # 10.d) Negative sampling
            n_neg   = pp.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, P.size(0), (n_neg,), device=device)
            neg_dst = torch.full((n_neg,), cond_idx, device=device)

            # If cond_idx == BREAST_IDX, restrict neg_src to female indices
            if cond_idx == BREAST_IDX:
                female_idx = torch.where(is_female_flat)[0]
                mask_b = (neg_dst == BREAST_IDX)
                neg_src[mask_b] = female_idx[
                    torch.randint(
                        0, female_idx.size(0),
                        (int(mask_b.sum()),),
                        device=device
                    )
                ]

            ne = torch.stack([neg_src, neg_dst], dim=0)  # [2, n_neg]
            npred = predictor(P, C, ne)                  # [n_neg, ]
            nl    = torch.zeros_like(npred)              # [n_neg, ]

            # 10.e) Collect for metrics
            all_preds  = torch.cat([pp, npred], dim=0)
            all_labels = torch.cat([pl, nl], dim=0)

            all_link_preds .append(all_preds.cpu())
            all_link_labels.append(all_labels.cpu())
            per_cond_preds [cond_idx].append(all_preds.cpu())
            per_cond_labels[cond_idx].append(all_labels.cpu())

            # 10.f) Accumulate BCE loss (with same global_pos_weight)
            total_link_loss += F.binary_cross_entropy_with_logits(
                all_preds, all_labels, pos_weight=global_pos_weight
            ).item()

        # ?? COX PARTIAL LOG?LIKELIHOOD LOSS ?? #
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risks = cox_head(P)                   # [N_pat, Cdim]
            durs = batch['patient'].duration      # [N_pat, Cdim]
            evts = batch['patient'].event         # [N_pat, Cdim]

            all_risk_scores.append(risks.cpu())
            all_durations  .append(durs.cpu())
            all_events     .append(evts.cpu())

            total_cox_loss += CoxHead.cox_partial_log_likelihood(
                risks, durs, evts
            ).item()

        num_batches += 1

    # ?? FINAL LOSS AVERAGES ?? #
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)

    # ?? GLOBAL AUC / PR?AUC ?? #
    if all_link_preds:
        y_true  = torch.cat(all_link_labels).numpy()
        y_score = torch.cat(all_link_preds).numpy()
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc   = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    # ?? PER?CONDITION PR?AUC ?? #
    pr_per_cond = [None] * num_conditions
    for ci in range(num_conditions):
        if per_cond_preds[ci]:
            y_t = torch.cat(per_cond_labels[ci]).numpy()
            y_s = torch.cat(per_cond_preds [ci]).numpy()
            pr_per_cond[ci] = average_precision_score(y_t, y_s)

    # ?? PER?CONDITION C?INDEX ?? #
    c_idxs, mean_c = [], None
    if all_risk_scores:
        R = torch.cat(all_risk_scores).numpy()  # [total_batches*N_pat, Cdim]
        D = torch.cat(all_durations).numpy()    # same shape
        E = torch.cat(all_events).numpy()       # same shape

        for ci in range(num_conditions):
            mask = E[:, ci] >= 0
            if mask.sum() > 0:
                try:
                    c = concordance_index(
                        D[mask, ci],
                        -R[mask, ci],
                        E[mask, ci]
                    )
                except ZeroDivisionError:
                    c = None
            else:
                c = None
            c_idxs.append(c)

        valid_c = [c for c in c_idxs if c is not None]
        if valid_c:
            mean_c = sum(valid_c) / len(valid_c)

    return avg_link, avg_cox, c_idxs, mean_c, link_auc, pr_auc, pr_per_cond



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
optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(link_head.parameters()) +
    list(cox_head.parameters()) +
    list(joint_head.parameters()),
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
        NEGATIVE_MULTIPLIER, diag_by_win_val, train_cond_embeds
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
