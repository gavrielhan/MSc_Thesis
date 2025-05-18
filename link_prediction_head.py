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
        Compute dynamically-weighted negative partial log-likelihood across conditions,
        with a fallback to uniform weights if dynamic weights are all zero.
        """
        losses = []

        num_conditions = risk_scores.size(1)
        for ci in range(num_conditions):
            # slice out one condition
            rs = risk_scores[:, ci]
            du = durations[:, ci]
            ev = events[:, ci]

            # keep only those with an event and positive duration
            mask = (ev > 0) & (du > 0)
            rs = rs[mask]
            du = du[mask]
            ev = ev[mask]

            if rs.numel() == 0:
                continue

            # sort by descending duration
            order = torch.argsort(du, descending=True)
            rs = rs[order]
            ev = ev[order]

            # partial?likelihood: logcumsumexp trick
            log_cum = torch.logcumsumexp(rs, dim=0)
            lik = (rs - log_cum) * ev

            # only if there was at least one event
            n_ev = ev.sum()
            if n_ev > 0:
                loss_ci = -lik.sum() / n_ev
                losses.append(loss_ci)

        # if no conditions produced a loss, return zero
        if len(losses) == 0:
            return 0.0 * risk_scores.sum()

        losses = torch.stack(losses)  # shape [K]

        # dynamic weights = loss magnitude
        weights = losses.clone()
        wsum = weights.sum()
        if wsum.abs() < 1e-8:
            # fallback to uniform weights when dynamic ones vanish
            weights = torch.ones_like(weights) / weights.numel()
        else:
            weights = weights / wsum

        # final, weighted sum of condition losses
        return (losses * weights).sum()

