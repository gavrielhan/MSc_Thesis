import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        A 2?layer MLP with a skip (residual) connection from the first layer's input
        into the second layer.

        Args:
            input_dim  (int): dimensionality of each patient / condition embed
            hidden_dim (int): dimensionality of the hidden layers
        """
        super().__init__()
        # we'll concatenate patient + condition ? 2 * input_dim
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        # If the concatenated dimension ? hidden_dim, project it to match:
        if input_dim * 2 != hidden_dim:
            self.res_proj = nn.Linear(input_dim * 2, hidden_dim)
        else:
            self.res_proj = nn.Identity()

        self.act = nn.ReLU()

    def forward(self, patient_embeds, condition_embeds, edge_index):
        """
        Args:
          patient_embeds   Tensor [N_pat, input_dim]
          condition_embeds Tensor [N_cond, input_dim]
          edge_index       LongTensor [2, E]  (rows: [pat_idx, cond_idx])

        Returns:
          logits           Tensor [E]
        """
        src = patient_embeds[edge_index[0]]  # [E, input_dim]
        dst = condition_embeds[edge_index[1]]  # [E, input_dim]
        x = torch.cat([src, dst], dim=1)  # [E, 2 * input_dim]

        # 1) First layer
        h1 = self.act(self.fc1(x))  # [E, hidden_dim]

        # 2) Residual: project original x ? hidden_dim
        res = self.res_proj(x)  # [E, hidden_dim]

        # 3) Second layer + skip
        h2 = self.act(self.fc2(h1) + res)  # [E, hidden_dim]

        # 4) Final logit
        return self.fc_out(h2).view(-1)  # [E]



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

class CosineLinkPredictor(nn.Module):
    def __init__(self, input_dim, init_scale: float = 1.0, use_bias: bool = True):
        """
        A link predictor that scores (p?c) by
            logits = scale * cosine_similarity(p, c) + bias
        so that you can train it via BCEWithLogits.

        Args:
            input_dim  (int): dimensionality of each patient/condition embed
            init_scale (float): initial value for the learnable scale
            use_bias   (bool): whether to include a learnable bias term
        """
        super().__init__()
        self.cos   = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("bias", torch.zeros(1))

    def forward(self, patient_embeds, condition_embeds, edge_index):
        """
        Args:
          patient_embeds   Tensor [N_pat, input_dim]
          condition_embeds Tensor [N_cond, input_dim]
          edge_index       LongTensor [2, E]  (rows: [pat_idx, cond_idx])

        Returns:
          logits           Tensor [E]
        """
        # gather the embeddings for each edge
        src = patient_embeds[edge_index[0]]    # [E, input_dim]
        dst = condition_embeds[edge_index[1]]  # [E, input_dim]

        # compute cosine similarity in [?1, 1]
        sim = self.cos(src, dst)               # [E]

        # scale (and bias) turn it into unbounded logits
        return sim * self.scale + self.bias   # [E]