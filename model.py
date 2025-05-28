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


def train(model, predictor, cox_head, loader, optimizer, device):
    model.train()
    predictor.train()
    cox_head.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)
        # Sanity?check inputs
        for ntype, x in batch.x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError(f"[Batch {batch_idx}] Bad input on node '{ntype}': contains NaN or Inf")

        # Skip batches with no useful edges
        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types and
            ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            print("[SKIP] Skipping batch with no useful edges.")
            continue

        optimizer.zero_grad()
        edge_attr_dict = {}
        rel = ('patient', 'to', 'signature')
        if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
            edge_attr_dict[rel] = batch[rel].edge_attr
        out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        # Sanity?check GNN outputs
        for ntype, h in out.items():
            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError(f"[Batch {batch_idx}] NaNs in model output for '{ntype}'")

        patient_embeds   = out['patient']
        condition_embeds = out['condition']

        # ------ FIX: initialize losses as tensors that require grad ------
        zero = patient_embeds.sum() * 0.0
        # Link prediction loss
        # --- build our positive edges over the next HORIZON windows ---
        # Link prediction loss over the next HORIZON windows
        window_idx = int(batch.window)  # your build_graph stashed wi ? batch.window
        # 1) collect all (patient, condition) diagnoses in [t+1 .. t+HORIZON]
        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend([
                (p, ci)
                for (p, ci) in diag_by_win_train.get(w, [])
            ])

        # 2) turn names ? indices
        names = [n for sub in batch['patient'].name
                 for n in (sub if isinstance(sub, list) else [sub])]  # list of patient IDs in this batch
        is_female = torch.tensor(
            [gender_map.get(p, 'female') == 'female' for p in names],
            device=device, dtype=torch.bool
        )
        pos_pairs = [
            (p, ci) for (p, ci) in pos_pairs
            if not (ci == BREAST_IDX and not is_female[names.index(p)])
        ]
        src_nodes = []
        dst_nodes = []
        for p, ci in pos_pairs:
            if p in names:
                src_nodes.append(names.index(p))
                dst_nodes.append(ci)

        # 3) build pos_ei if any positives exist
        if src_nodes:
            pos_ei = torch.stack([
                torch.tensor(src_nodes, device=device),
                torch.tensor(dst_nodes, device=device),
            ], dim=0)  # shape [2, N_pos]
        else:
            pos_ei = None

        # 4) if we have any future positives, do BCE; otherwise zero
        if pos_ei is not None:
            pos_preds = predictor(patient_embeds, condition_embeds, pos_ei)
            pos_labels = torch.ones_like(pos_preds)

            num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
            # sample initial negatives
            neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, condition_embeds.size(0), (num_neg,), device=device)
            # enforce female-only for breast?cancer negatives
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

        # --- now, _outside_ of that horizon logic_, do your Cox loss exactly once per batch ---
        cox_loss = zero.clone()
        valid_cox_conds = 0
        num_conditions = condition_embeds.size(0)
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            events = batch['patient'].event
            durations = batch['patient'].duration
            risk_scores = cox_head(patient_embeds)
            for ci in range(num_conditions):
                ev_col = events[:, ci]
                if ev_col.sum().item() > 0:
                    dur_col = durations[:, ci]
                    score_col = risk_scores[:, ci]
                    # guards?
                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        score_col.unsqueeze(1),
                        dur_col.unsqueeze(1),
                        ev_col.unsqueeze(1),
                    )
                    cox_loss += loss_ci
                    valid_cox_conds += 1
            if valid_cox_conds > 0:
                cox_loss /= valid_cox_conds
        else:
            print(f"[Batch {batch_idx}] Missing duration/event attrs")
        # --- Final loss and backward ---
        loss = joint_head(link_loss, cox_loss)
        # Sanity?check losses
        for name, l in (("link_loss", link_loss), ("cox_loss", cox_loss)):
            if isinstance(l, torch.Tensor):
                if not torch.isfinite(l).all():
                    raise RuntimeError(f"[Batch {batch_idx}] {name} is non-finite: {l}")
            else:
                if not math.isfinite(l):
                    raise RuntimeError(f"[Batch {batch_idx}] {name} is non-finite: {l}")

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"[Batch {batch_idx}] total loss is non-finite: {loss}")

        loss.backward()

        # Clip gradients and check
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(predictor.parameters()) +
            list(cox_head.parameters()),
            max_norm=1.0
        )
        for name, param in list(model.named_parameters()) + \
                           list(predictor.named_parameters()) + \
                           list(cox_head.named_parameters()):
            if param.grad is not None and torch.isnan(param.grad).any():
                raise RuntimeError(f"[Batch {batch_idx}] NaN in grad of '{name}'")

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)




@torch.no_grad()
def evaluate(model, predictor, cox_head, loader, device,
             NEGATIVE_MULTIPLIER, diag_by_win_map, frozen_condition_embeddings):
    model.eval(); predictor.eval(); cox_head.eval()

    total_link_loss = 0.0
    total_cox_loss  = 0.0
    all_link_preds,  all_link_labels  = [], []
    per_cond_preds,  per_cond_labels  = {}, {}
    all_risk_scores, all_durations, all_events = [], [], []
    num_batches = 0

    # Number of conditions
    num_conditions = cox_head.linear.out_features
    for ci in range(num_conditions):
        per_cond_preds[ci]  = []
        per_cond_labels[ci] = []

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # Skip if no patients
        if 'patient' not in batch.x_dict or batch['patient'].x.size(0) == 0:
            continue

        # Prepare edges + attrs for GNN forward
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient', 'has', 'condition')
        }
        edge_attr_dict = {}
        rel = ('patient', 'to', 'signature')
        if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
            edge_attr_dict[rel] = batch[rel].edge_attr

        # Forward pass
        out = model(batch.x_dict, edge_index_dict, edge_attr_dict)
        P = out['patient']
        C = frozen_condition_embeddings.to(device)

        # Skip empty embeddings
        if P.size(0) == 0 or C.size(0) == 0:
            continue
        pos_pairs = []
        for w in range(window_idx + 1, window_idx + 1 + HORIZON):
            pos_pairs.extend(diag_by_win_map.get(w, []))
        # Flatten the patient names list once
        flat_names = [n for sub in batch['patient'].name
                 for n in (sub if isinstance(sub, list) else [sub])]
        is_female_flat = torch.tensor(
            [gender_map.get(p, 'female') == 'female' for p in flat_names],
            device=device, dtype=torch.bool
        )
        pos_pairs = [
            (p, ci) for (p, ci) in pos_pairs
            if not (ci == BREAST_IDX and not is_female_flat[flat_names.index(p)])
        ]

        # Map to node indices
        src_nodes, dst_nodes = [], []
        for p, ci in pos_pairs:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        # If no positives in horizon, skip this batch
        if not src_nodes:
            continue

        pos_ei = torch.stack([
            torch.tensor(src_nodes, device=device),
            torch.tensor(dst_nodes, device=device),
        ], dim=0)  # shape [2, #pos]

        # ----- Link prediction loss & metrics -----
        for cond_idx in range(num_conditions):
            mask = (pos_ei[1] == cond_idx)
            # drop any male?breast positives
            if cond_idx == BREAST_IDX:
                mask &= is_female_flat[pos_ei[0]]
            if not mask.any():
                continue

            # Positive logits
            pe  = pos_ei[:, mask]
            pp  = predictor(P, C, pe)
            pl  = torch.ones_like(pp)

            # Negative sampling
            n_neg   = pp.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, P.size(0), (n_neg,), device=device)
            neg_dst = torch.full((n_neg,), cond_idx, device=device)
            if cond_idx == BREAST_IDX:
                female_idx = torch.where(is_female_flat)[0]
                mask_b = (neg_dst == BREAST_IDX)
                neg_src[mask_b] = female_idx[
                    torch.randint(0, female_idx.size(0), (int(mask_b.sum()),), device=device)
                ]
            ne      = torch.stack([neg_src, neg_dst], dim=0)
            npred   = predictor(P, C, ne)
            nl      = torch.zeros_like(npred)

            # Collect for metrics
            preds  = torch.cat([pp, npred])
            labels = torch.cat([pl, nl])

            all_link_preds .append(preds.cpu())
            all_link_labels.append(labels.cpu())
            per_cond_preds [cond_idx].append(preds.cpu())
            per_cond_labels[cond_idx].append(labels.cpu())

            # Accumulate loss
            total_link_loss += F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=global_pos_weight
            ).item()

        # ----- Cox regression loss -----
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risks = cox_head(P)
            durs  = batch['patient'].duration
            evts  = batch['patient'].event

            all_risk_scores.append(risks.cpu())
            all_durations .append(durs.cpu())
            all_events    .append(evts.cpu())

            total_cox_loss += CoxHead.cox_partial_log_likelihood(
                risks, durs, evts
            ).item()

        num_batches += 1

    # Compute averaged losses
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss   / max(1, num_batches)

    # Global AUC / PR?AUC
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
    train_loss = train(model, link_head, cox_head, train_loader, optimizer, device)
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
