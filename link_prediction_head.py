import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkMLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # now input_dim*2 (p+c) + 1 (tte)
        self.fc1     = nn.Linear(input_dim*2 + 1, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, 1)

        if input_dim*2 + 1 != hidden_dim:
            self.res_proj = nn.Linear(input_dim*2 + 1, hidden_dim)
        else:
            self.res_proj = nn.Identity()

        self.act = nn.ReLU()

    def forward(self, patient_embeds, condition_embeds, edge_index, tte=None):
        src = patient_embeds[edge_index[0]]    # [E, D]
        dst = condition_embeds[edge_index[1]]  # [E, D]
        x   = torch.cat([src, dst], dim=1)     # [E, 2D]

        if tte is not None:
            tte_feat = tte.view(-1,1).float()  # [E,1]
            x = torch.cat([x, tte_feat], dim=1)  # [E, 2D+1]
        else:
            # if no tte, pad with zeros
            zeros = torch.zeros(x.size(0), 1, device=x.device)
            x = torch.cat([x, zeros], dim=1)

        h1  = self.act(self.fc1(x))
        res = self.res_proj(x)
        h2  = self.act(self.fc2(h1) + res)
        return self.fc_out(h2).view(-1)        # [E]


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
        losses = []
        C = risk_scores.size(1)

        for ci in range(C):
            rs = risk_scores[:, ci]  # [N]
            du = durations[:, ci]  # [N]
            ev = events[:, ci]  # [N]

            # 1) drop any zero/negative durations (if your data has them)
            valid = du > 0
            if valid.sum() == 0:
                continue
            rs = rs[valid]
            du = du[valid]
            ev = ev[valid]

            # 2) sort *all* by descending time
            order = torch.argsort(du, descending=True)
            rs = rs[order]
            ev = ev[order]

            # 3) build log?cumulative?hazard over the full set
            log_cum = torch.logcumsumexp(rs, dim=0)  # denominator uses everyone

            # 4) only event?times contribute to numerator
            #    (censored will get zero because ev=0)
            #    lik_i = h_i - log(?_{j?i} e^{h_j})
            lik = (rs - log_cum) * ev

            # 5) average over the #events
            n_ev = ev.sum()
            if n_ev > 0:
                losses.append(-lik.sum() / n_ev)

        if not losses:
            # no events at all: zero?loss
            return torch.tensor(0.0, device=risk_scores.device)

        losses = torch.stack(losses)  # one per condition
        # your dynamic weighting (optional)
        weights = losses.clone()
        wsum = weights.sum()
        if wsum.abs() < 1e-8:
            weights = torch.ones_like(weights) / weights.numel()
        else:
            weights = weights / wsum

        return (losses * weights).sum()

class CosineLinkPredictor(nn.Module):
    def __init__(self, input_dim, init_scale: float = 10.0, use_bias: bool = True):
        """
        Cosine similarity + learnable temperature:
          logit = scale * cos(h_p, h_c) + bias
        so that after sigmoid you get a full [0,1] range.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = None

    def forward(self, patient_embeds, condition_embeds, edge_index):
        # edge_index: [2, E], first row patient idx, second row cond idx
        src = patient_embeds[edge_index[0]]      # [E, D]
        dst = condition_embeds[edge_index[1]]    # [E, D]
        cos = F.cosine_similarity(src, dst, dim=1)  # [E]
        logits = cos * self.scale
        if self.bias is not None:
            logits = logits + self.bias
        return logits

class TimeAwareCosineLinkPredictor(nn.Module):
    def __init__(self, init_scale: float = 10.0, use_bias: bool = True):
        super().__init__()
        # cosine temperature & bias
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias  = nn.Parameter(torch.zeros(1)) if use_bias else None
        # learned weight on time?to?event
        self.time_coeff = nn.Parameter(torch.tensor(1.0))

    def forward(self,
                patient_embeds:   torch.Tensor,
                condition_embeds: torch.Tensor,
                edge_index:       torch.Tensor,
                tte  = None
               ) -> torch.Tensor:
        """
        edge_index: [2, E]
        tte:        [E] if provided, else None?zeros
        """
        # if no tte passed, zero it out
        if tte is None:
            tte = torch.zeros(edge_index.size(1), device=patient_embeds.device)

        # basic cosine part
        src  = patient_embeds[edge_index[0]]     # [E, D]
        dst  = condition_embeds[edge_index[1]]   # [E, D]
        cos  = F.cosine_similarity(src, dst, dim=1)
        logits = cos * self.scale

        # time boost only where tte>0
        boost = torch.where(
            tte > 0,
            1.0 / (tte + 1.0),
            torch.zeros_like(tte),
        )
        logits = logits + self.time_coeff * boost

        if self.bias is not None:
            logits = logits + self.bias
        return logits
