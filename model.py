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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEGATIVE_MULTIPLIER = 3  # you can adjust this multiplier later
# Count total positive edges in the training split
total_pos = sum(g.edge_index_dict[('patient','has','condition')].size(1)
                for g in train_graphs if ('patient','has','condition') in g.edge_index_dict
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
def evaluate(model, predictor, cox_head, loader, device, NEGATIVE_MULTIPLIER):
    model.eval()
    predictor.eval()
    cox_head.eval()

    total_link_loss = 0.0
    total_cox_loss = 0.0
    all_link_preds = []
    all_link_labels = []
    all_risk_scores = []
    all_durations = []
    all_events = []

    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)

        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types and
            ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            continue

        num_batches += 1

        out = model(batch.x_dict, batch.edge_index_dict)
        patient_embeds = out.get('patient')
        condition_embeds = out.get('condition')

        # --- Link prediction ---
        if ('patient', 'has', 'condition') in batch.edge_index_dict:
            pos_edge_index = batch[('patient', 'has', 'condition')].edge_index
            src_nodes, dst_nodes = pos_edge_index
            for cond_idx in range(condition_embeds.size(0)):
                mask = (dst_nodes == cond_idx)
                if mask.sum() == 0:
                    continue

                cond_src = src_nodes[mask]
                cond_edge_index = torch.stack([cond_src, dst_nodes[mask]], dim=0)

                pos_preds = predictor(patient_embeds, condition_embeds, cond_edge_index)
                pos_labels = torch.ones_like(pos_preds)
                num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
                neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
                neg_dst = torch.full((num_neg,), cond_idx, device=device)
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

                neg_preds = predictor(patient_embeds, condition_embeds, neg_edge_index)
                neg_labels = torch.zeros_like(neg_preds)

                preds = torch.cat([pos_preds, neg_preds])
                labels = torch.cat([pos_labels, neg_labels])

                all_link_preds.append(preds.detach().cpu())
                all_link_labels.append(labels.detach().cpu())

                pos_weight = torch.tensor([global_pos_weight], device=device)
                total_link_loss += F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight).item()

        # --- Cox regression ---
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risk_scores = cox_head(patient_embeds)
            durations = batch['patient'].duration
            events = batch['patient'].event

            all_risk_scores.append(risk_scores.detach().cpu())
            all_durations.append(durations.detach().cpu())
            all_events.append(events.detach().cpu())

            total_cox_loss += CoxHead.cox_partial_log_likelihood(risk_scores, durations, events).item()

    # --- Average losses ---
    avg_link_loss = total_link_loss / max(1, num_batches)
    avg_cox_loss = total_cox_loss / max(1, num_batches)

    # --- Compute ROC AUC for link prediction ---
    if all_link_preds:
        y_true = torch.cat(all_link_labels).numpy()
        y_score = torch.cat(all_link_preds).numpy()
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    # --- Compute C-index per condition and mean C-index ---
    c_indices = []
    if all_risk_scores:
        risks = torch.cat(all_risk_scores).cpu().numpy()  # shape (T, C)
        durs = torch.cat(all_durations).cpu().numpy()  # shape (T, C)
        evs = torch.cat(all_events).cpu().numpy()  # shape (T, C)

        c_indices = []
        num_conditions = risks.shape[1]

        for cond_idx in range(num_conditions):
            mask = evs[:, cond_idx] >= 0
            if mask.sum() > 0:
                # Extract the 1-D vectors for this condition
                dur_vec = durs[mask, cond_idx]
                pred_vec = risks[mask, cond_idx]
                ev_vec = evs[mask, cond_idx]

                try:
                    c_idx = concordance_index(dur_vec, -pred_vec, ev_vec)
                except ZeroDivisionError:
                    c_idx = None
            else:
                c_idx = None

            c_indices.append(c_idx)
        valid_cs = [c for c in c_indices if c is not None]
        mean_c_index = sum(valid_cs) / len(valid_cs) if valid_cs else None
    else:
        mean_c_index = None

    return avg_link_loss, avg_cox_loss, c_indices, mean_c_index, link_auc, pr_auc




in_dims = {
    'patient': 138,
    'signature': 96,
    'condition': 9,
}

model = HeteroGAT(
    metadata=all_graphs[0].metadata(),
    in_dims=in_dims,
    hidden_dim=128,
    out_dim=128
).to(device)

link_head = LinkMLPPredictor(input_dim=128).to(device)
cox_head = CoxHead(input_dim=128, num_conditions=9).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) +
                             list(link_head.parameters()) +
                             list(cox_head.parameters()), lr=1e-3)

# Create DataLoaders for each split
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1)
test_loader = DataLoader(test_graphs, batch_size=1)

EPOCHS = 50
for epoch in range(1, EPOCHS + 1):
    train_loss = train(model, link_head, cox_head, train_loader, optimizer, device)
    val_link_loss, val_cox_loss, val_c_indices, val_mean_c_index, val_link_auc, val_pr_auc = evaluate(model, link_head, cox_head, val_loader, device, NEGATIVE_MULTIPLIER)
    # After training finished
    results = {
        "final_train_loss": train_loss,
        "final_val_link_loss": val_link_loss,
        "final_val_link_auc": val_link_auc,
        "final_val_pr_auc": val_pr_auc,
        "final_val_cox_loss": val_cox_loss,
        "c_indices_per_condition": val_c_indices,
        "mean_c_index": val_mean_c_index,
        "epoch": epoch,
        "NEGATIVE_MULTIPLIER": NEGATIVE_MULTIPLIER,
        "timestamp": datetime.datetime.now().isoformat()
    }

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
          f"Val Link Loss: {val_link_loss:.4f} |"
          f"Val Cox Loss: {val_cox_loss:.4f} | Val C-Index: {val_c_indices}")

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/results_{timestamp}.json"

with open(filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Saved training results to {filename}")