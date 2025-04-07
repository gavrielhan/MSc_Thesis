import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CoxHead(nn.Module):
    """
    Cox regression head for time-to-event modeling from GNN patient embeddings.
    Predicts a continuous risk score (log hazard ratio) per patient.
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim (int): Dimensionality of the patient embedding.
        """
        super(CoxHead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, patient_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patient_embeddings (Tensor): [N, input_dim] tensor of patient embeddings

        Returns:
            risk_scores (Tensor): [N] tensor of log hazard scores
        """
        return self.linear(patient_embeddings).squeeze(-1)

    @staticmethod
    def cox_partial_log_likelihood(risk_scores: torch.Tensor,
                                   durations: torch.Tensor,
                                   events: torch.Tensor) -> torch.Tensor:
        """
        Compute negative partial log-likelihood for Cox proportional hazards.

        Args:
            risk_scores (Tensor): [N] predicted log hazard scores
            durations (Tensor): [N] observed times (either to event or censoring)
            events (Tensor): [N] binary indicator (1 = event occurred, 0 = censored)

        Returns:
            loss (Tensor): scalar loss value
        """
        # Sort by descending duration
        order = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[order]
        events = events[order]

        log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)
        likelihood = (risk_scores - log_cumsum_exp) * events

        # Return negative partial log-likelihood
        return -torch.sum(likelihood) / torch.sum(events)