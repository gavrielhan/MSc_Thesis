from link_prediction_head import CoxHead
from link_prediction_head import LinkMLPPredictor
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


class HeteroGAT(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, out_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleDict({
            node_type: Linear(in_dims[node_type], hidden_dim)
            for node_type in metadata[0]
        })

        self.convs1 = HeteroConv({
            edge_type: (
                GCNConv(hidden_dim, hidden_dim)
                if edge_type == ('patient', 'follows', 'patient') else
                GATConv((hidden_dim, hidden_dim), hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.convs2 = HeteroConv({
            edge_type: (
                GCNConv(hidden_dim, hidden_dim)
                if edge_type == ('patient', 'follows', 'patient') else
                GATConv((hidden_dim, hidden_dim), hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        # ? Add a 3rd convolutional layer
        self.convs3 = HeteroConv({
            edge_type: (
                GCNConv(hidden_dim, out_dim)
                if edge_type == ('patient', 'follows', 'patient') else
                GATConv((hidden_dim, hidden_dim), out_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.linear_proj = nn.ModuleDict({
            node_type: Linear(out_dim, out_dim)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.input_proj[k](v) for k, v in x_dict.items()}

        x_dict1 = self.convs1(x_dict, edge_index_dict)
        for node_type in self.metadata[0]:
            if node_type not in x_dict1 or x_dict1[node_type] is None:
                x_dict1[node_type] = x_dict[node_type]
            else:
                x_dict1[node_type] = F.relu(x_dict1[node_type])

        x_dict2 = self.convs2(x_dict1, edge_index_dict)
        for node_type in self.metadata[0]:
            if node_type not in x_dict2 or x_dict2[node_type] is None:
                x_dict2[node_type] = x_dict1[node_type]
            else:
                x_dict2[node_type] = F.relu(x_dict2[node_type])

        x_dict3 = self.convs3(x_dict2, edge_index_dict)
        for node_type in self.metadata[0]:
            if node_type not in x_dict3 or x_dict3[node_type] is None:
                x_dict3[node_type] = x_dict2[node_type]
            else:
                x_dict3[node_type] = F.relu(x_dict3[node_type])

        x_dict3 = {key: self.linear_proj[key](x) for key, x in x_dict3.items()}
        return x_dict3





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
NEGATIVE_MULTIPLIER = 3  # you can adjust this multiplier later
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

    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)

        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types and
            ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            print("[SKIP] Skipping batch with no useful edges.")
            continue

        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)

        patient_embeds = out.get('patient')
        condition_embeds = out.get('condition')

        link_loss = 0
        valid_conditions = 0  # <<< count conditions where at least 1 pos edge

        if ('patient', 'has', 'condition') in batch.edge_index_dict:
            pos_edge_index = batch[('patient', 'has', 'condition')].edge_index
            src_nodes = pos_edge_index[0]
            dst_nodes = pos_edge_index[1]

            for cond_idx in range(condition_embeds.size(0)):
                mask = (dst_nodes == cond_idx)
                if mask.sum() > 0:
                    cond_src = src_nodes[mask]
                    cond_dst = dst_nodes[mask]

                    cond_edge_index = torch.stack([cond_src, cond_dst], dim=0)

                    pos_preds = predictor(patient_embeds, condition_embeds, cond_edge_index)
                    pos_labels = torch.ones(pos_preds.size(0), device=device)

                    num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
                    neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
                    neg_dst = torch.full((num_neg,), cond_idx, device=device)

                    neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                    neg_preds = predictor(patient_embeds, condition_embeds, neg_edge_index)
                    neg_labels = torch.zeros(neg_preds.size(0), device=device)

                    preds = torch.cat([pos_preds, neg_preds], dim=0)
                    labels = torch.cat([pos_labels, neg_labels], dim=0)


                    # use the same global weight every time
                    pos_weight = torch.tensor([global_pos_weight], device=device)
                    link_loss += F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

                    valid_conditions += 1  # <<< only count if condition had positive edges
                else:
                    ok = 0

            if valid_conditions > 0:
                link_loss /= valid_conditions
            else:
                link_loss = 0.0 * patient_embeds.sum()  # <<< no valid conditions, set loss to 0

        # --- Cox Regression Loss ---
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            events = batch['patient'].event
            num_events = (events > 0).sum()


            if num_events > 0:
                risk_scores = cox_head(patient_embeds)
                durations = batch['patient'].duration
                cox_loss = CoxHead.cox_partial_log_likelihood(risk_scores, durations, events)
            else:

                cox_loss = 0.0 * patient_embeds.sum()

        else:
            print("Batch missing duration/event attributes")  # <<<
            cox_loss = 0.0 * patient_embeds.sum()


        # --- Final loss ---
        loss = link_loss + cox_loss
        loss.backward()
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

    # Get number of conditions from CoxHead output dimension
    num_conditions = cox_head.linear.out_features
    for ci in range(num_conditions):
        per_cond_preds[ci]  = []
        per_cond_labels[ci] = []

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # --- Skip graphs with no patient nodes ---
        if 'patient' not in batch.x_dict or batch['patient'].x.size(0) == 0:
            continue

        # --- Remove true disease edges ---
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient', 'has', 'condition')
        }

        # --- Get patient embeddings ---
        out = model(batch.x_dict, edge_index_dict)
        P = out['patient']  # [n_patients, hidden]

        # --- Use fixed condition embeddings from training ---
        C = frozen_condition_embeddings.to(device)

        # --- Skip if embeddings are empty ---
        if P.size(0) == 0 or C.size(0) == 0:
            continue

        names = [n for sublist in batch['patient'].name for n in (sublist if isinstance(sublist, list) else [sublist])]
        name_set = set(names)


        # --- Reconstruct diagnosis edges (truth) ---
        positives = [
            (names.index(p), ci)
            for (p, ci) in diag_by_win_map.get(window_idx, [])
            if p in name_set
        ]
        if not positives:
            continue  # no positives in this window

        src = torch.tensor([i for i, ci in positives], device=device)
        dst = torch.tensor([ci for i, ci in positives], device=device)
        pos_ei = torch.stack([src, dst], dim=0)

        # --- Link prediction loss ---
        for cond_idx in range(num_conditions):
            mask = dst == cond_idx
            if not mask.any():
                continue

            pe = pos_ei[:, mask]              # [2, #pos]
            pp = predictor(P, C, pe)          # logits
            pl = torch.ones_like(pp)

            # --- Negative sampling ---
            n_neg   = pp.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, P.size(0), (n_neg,), device=device)
            neg_dst = torch.full((n_neg,), cond_idx, device=device)
            ne = torch.stack([neg_src, neg_dst], dim=0)
            npred = predictor(P, C, ne)

            nl    = torch.zeros_like(npred)

            # --- Metrics ---
            preds  = torch.cat([pp, npred])
            labels = torch.cat([pl, nl])

            all_link_preds.append(preds.cpu())
            all_link_labels.append(labels.cpu())
            per_cond_preds[cond_idx].append(preds.cpu())
            per_cond_labels[cond_idx].append(labels.cpu())

            total_link_loss += F.binary_cross_entropy_with_logits(
                preds, labels, pos_weight=global_pos_weight
            ).item()

        # --- Cox regression loss ---
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risks = cox_head(P)
            durs  = batch['patient'].duration
            evts  = batch['patient'].event

            all_risk_scores.append(risks.cpu())
            all_durations.append(durs.cpu())
            all_events.append(evts.cpu())

            total_cox_loss += CoxHead.cox_partial_log_likelihood(
                risks, durs, evts
            ).item()

        num_batches += 1

    # --- Final loss averages ---
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)

    # --- Global AUC / PR-AUC ---
    if all_link_preds:
        y_true  = torch.cat(all_link_labels).numpy()
        y_score = torch.cat(all_link_preds).numpy()
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc   = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    # --- PR-AUC per condition ---
    pr_per_cond = [None] * num_conditions
    for ci in range(num_conditions):
        if per_cond_preds[ci]:
            y_t = torch.cat(per_cond_labels[ci]).numpy()
            y_s = torch.cat(per_cond_preds[ci]).numpy()
            pr_per_cond[ci] = average_precision_score(y_t, y_s)

    # --- C-index per condition ---
    c_idxs, mean_c = [], None
    if all_risk_scores:
        R = torch.cat(all_risk_scores).numpy()
        D = torch.cat(all_durations).numpy()
        E = torch.cat(all_events).numpy()
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

model = HeteroGAT(
    metadata=all_graphs[0].metadata(),
    in_dims=in_dims,
    hidden_dim=128,
    out_dim=128
).to(device)

link_head = LinkMLPPredictor(input_dim=128).to(device)
cox_head = CoxHead(input_dim=128, num_conditions=7).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) +
                             list(link_head.parameters()) +
                             list(cox_head.parameters()), lr=1e-3)

# Create DataLoaders for each split
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1)
test_loader = DataLoader(test_graphs, batch_size=1)

EPOCHS = 50
history = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, link_head, cox_head, train_loader, optimizer, device)
    with torch.no_grad():
        train_cond_embeds = None
        for g in train_graphs:
            g = g.to(device)
            ei_masked = {et: g[et].edge_index for et in g.edge_types if et != ('patient', 'has', 'condition')}
            out = model(g.x_dict, ei_masked)
            C = out["condition"]
            train_cond_embeds = C if train_cond_embeds is None else train_cond_embeds + C
        train_cond_embeds /= len(train_graphs)

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