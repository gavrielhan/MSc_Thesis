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
        )

    def forward(self, patient_embeds, condition_embeds, edge_index):
        # edge_index: shape [2, num_edges] ? [patient_idx, condition_idx]
        src = patient_embeds[edge_index[0]]  # [N, input_dim]
        dst = condition_embeds[edge_index[1]]  # [N, input_dim]
        x = torch.cat([src, dst], dim=1)  # [N, 2 * input_dim]
        return self.mlp(x).view(-1)  # [N]



class CoxHead(nn.Module):
    """
    Cox regression head for time-to-event modeling from GNN patient embeddings.
    Predicts a continuous risk score (log hazard ratio) per patient.
    """

    def __init__(self, input_dim: int, num_conditions: int):
        """
        Args:
            input_dim (int): Dimensionality of the patient embedding.
            num_conditions (int): Number of conditions to predict risk for.
        """
        super(CoxHead, self).__init__()
        self.linear = nn.Linear(input_dim, num_conditions)

    def forward(self, patient_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patient_embeddings (Tensor): [N, input_dim] tensor of patient embeddings

        Returns:
            risk_scores (Tensor): [N, num_conditions] tensor of log hazard scores
        """
        return self.linear(patient_embeddings)

    @staticmethod
    def cox_partial_log_likelihood(risk_scores: torch.Tensor,
                                   durations: torch.Tensor,
                                   events: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamically-weighted negative partial log-likelihood across conditions.
        """
        losses = []
        weights = []

        num_conditions = risk_scores.size(1)
        for cond_idx in range(num_conditions):
            cond_risk = risk_scores[:, cond_idx]
            cond_duration = durations[:, cond_idx]
            cond_event = events[:, cond_idx]

            mask = (cond_event >= 0)
            cond_risk = cond_risk[mask]
            cond_duration = cond_duration[mask]
            cond_event = cond_event[mask]

            if cond_risk.numel() == 0:
                continue

            order = torch.argsort(cond_duration, descending=True)
            cond_risk = cond_risk[order]
            cond_event = cond_event[order]

            log_cumsum_exp = torch.logcumsumexp(cond_risk, dim=0)
            likelihood = (cond_risk - log_cumsum_exp) * cond_event

            if cond_event.sum() > 0:
                loss = -torch.sum(likelihood) / cond_event.sum()
                losses.append(loss)
                weights.append(loss)   # use loss magnitude as proxy for dynamic importance

        if len(losses) == 0:
            return 0.0 * risk_scores.sum()

        weights = torch.stack(weights)
        weights = weights / weights.sum()  # normalize
        losses = torch.stack(losses)

        return (losses * weights).sum()

