import torch
import torch.nn as nn

class LinkMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # for binary classification (link or no link)
        )

    def forward(self, patient_embeds, condition_embeds, edge_index):
        # edge_index: shape [2, num_edges] ? [patient_idx, condition_idx]
        src = patient_embeds[edge_index[0]]  # [N, input_dim]
        dst = condition_embeds[edge_index[1]]  # [N, input_dim]
        x = torch.cat([src, dst], dim=1)  # [N, 2 * input_dim]
        return self.mlp(x).squeeze()  # [N]
