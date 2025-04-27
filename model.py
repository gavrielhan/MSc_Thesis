from link_prediction_head import CoxHead
from link_prediction_head import LinkMLPPredictor
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.nn import Linear
from torch.nn import Module, ModuleDict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class HeteroGAT(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, out_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim

        # Input projection to hidden_dim
        self.input_proj = nn.ModuleDict({
            node_type: Linear(in_dims[node_type], hidden_dim)
            for node_type in metadata[0]
        })

        # First conv layer
        self.convs1 = HeteroConv({
            edge_type: (
                GCNConv(hidden_dim, hidden_dim)
                if edge_type == ('patient', 'follows', 'patient') else
                GATConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False
                )
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        # Second conv layer
        self.convs2 = HeteroConv({
            edge_type: (
                GCNConv(hidden_dim, out_dim)
                if edge_type == ('patient', 'follows', 'patient') else
                GATConv(
                    (hidden_dim, hidden_dim),
                    out_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False
                )
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.linear_proj = nn.ModuleDict({
            node_type: Linear(out_dim, out_dim)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        # Step 1: Input projection
        x_dict = {k: self.input_proj[k](v) for k, v in x_dict.items()}

        # Step 2: First convolution
        x_dict1 = self.convs1(x_dict, edge_index_dict)
        for node_type in self.metadata[0]:
            if node_type not in x_dict1 or x_dict1[node_type] is None:
                x_dict1[node_type] = x_dict[node_type]
            else:
                x_dict1[node_type] = F.relu(x_dict1[node_type])

        # Step 3: Second convolution
        x_dict2 = self.convs2(x_dict1, edge_index_dict)
        for node_type in self.metadata[0]:
            if node_type not in x_dict2 or x_dict2[node_type] is None:
                x_dict2[node_type] = x_dict1[node_type]
            else:
                x_dict2[node_type] = F.relu(x_dict2[node_type])

        # Step 4: Final projection
        x_dict2 = {key: self.linear_proj[key](x) for key, x in x_dict2.items()}
        return x_dict2




all_graphs =  torch.load("glucose_sleep_graphs.pt", weights_only = False)


# Load your pre-split graph lists
train_graphs = torch.load("split/train_graphs.pt", weights_only = False)
val_graphs = torch.load("split/val_graphs.pt", weights_only = False)
test_graphs = torch.load("split/test_graphs.pt", weights_only = False)


NEGATIVE_MULTIPLIER = 3  # you can adjust this multiplier later

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

        # --- Link Prediction Loss ---
        if ('patient', 'has', 'condition') in batch.edge_index_dict:
            pos_edge_index = batch[('patient', 'has', 'condition')].edge_index

            # Positive samples
            pos_preds = predictor(patient_embeds, condition_embeds, pos_edge_index)
            pos_labels = torch.ones(pos_preds.size(0), device=device)

            # Dynamic Negative Sampling
            num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, condition_embeds.size(0), (num_neg,), device=device)
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

            neg_preds = predictor(patient_embeds, condition_embeds, neg_edge_index)
            neg_labels = torch.zeros(neg_preds.size(0), device=device)

            # Combine
            preds = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            pos_weight = torch.tensor([neg_labels.size(0) / (pos_labels.size(0) + 1e-6)], device=device)
            link_loss = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        else:
            link_loss = 0

        # --- Cox Regression Loss ---
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risk_scores = cox_head(patient_embeds)
            durations = batch['patient'].duration
            events = batch['patient'].event
            cox_loss = CoxHead.cox_partial_log_likelihood(risk_scores, durations, events)
        else:
            cox_loss = 0

        loss = link_loss + cox_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(model, predictor, cox_head, loader, device):
    model.eval()
    predictor.eval()
    cox_head.eval()

    total_link_loss = 0
    total_cox_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)

        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types and
            ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            print("[SKIP] Skipping batch with no useful edges.")
            continue

        out = model(batch.x_dict, batch.edge_index_dict)

        patient_embeds = out.get('patient')
        condition_embeds = out.get('condition')

        # --- Link Prediction Loss ---
        if ('patient', 'has', 'condition') in batch.edge_index_dict:
            pos_edge_index = batch[('patient', 'has', 'condition')].edge_index

            pos_preds = predictor(patient_embeds, condition_embeds, pos_edge_index)
            pos_labels = torch.ones(pos_preds.size(0), device=device)

            num_neg = pos_preds.size(0) * NEGATIVE_MULTIPLIER
            neg_src = torch.randint(0, patient_embeds.size(0), (num_neg,), device=device)
            neg_dst = torch.randint(0, condition_embeds.size(0), (num_neg,), device=device)
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

            neg_preds = predictor(patient_embeds, condition_embeds, neg_edge_index)
            neg_labels = torch.zeros(neg_preds.size(0), device=device)

            preds = torch.cat([pos_preds, neg_preds], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            pos_weight = torch.tensor([neg_labels.size(0) / (pos_labels.size(0) + 1e-6)], device=device)
            link_loss = F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

            total_link_loss += link_loss.item()
        else:
            pass  # no link_loss added if no patient->condition edges

        # --- Cox Regression Loss ---
        if hasattr(batch['patient'], 'event') and hasattr(batch['patient'], 'duration'):
            risk_scores = cox_head(patient_embeds)
            durations = batch['patient'].duration
            events = batch['patient'].event
            total_cox_loss += CoxHead.cox_partial_log_likelihood(risk_scores, durations, events).item()

        num_batches += 1

    return (
        total_link_loss / max(1, num_batches),
        total_cox_loss / max(1, num_batches)
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
cox_head = CoxHead(input_dim=128).to(device)

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
    val_link_loss, val_cox_loss = evaluate(model, link_head, cox_head, val_loader, device)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Link Loss: {val_link_loss:.4f} | Val Cox Loss: {val_cox_loss:.4f}")
