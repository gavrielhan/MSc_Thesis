import torch
import torch.nn.functional as F
import numpy as np

def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce   = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t  = prob * labels + (1 - prob) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    fl   = alpha_t * (1 - p_t).pow(gamma) * ce
    if reduction == "sum":
        return fl.sum()
    elif reduction == "mean":
        return fl.mean()
    return fl

def simple_smote(X: torch.Tensor, n_samples: int, k: int = 5) -> torch.Tensor:
    N, D = X.size()
    if n_samples <= 0 or N == 0:
        return torch.empty((0, D), device=X.device)
    if N == 1:
        return X[0].unsqueeze(0).repeat(n_samples, 1)
    k = min(k, N - 1)
    if k == 0:
        idx = torch.randint(0, N, (n_samples,), device=X.device)
        return X[idx]
    dist = torch.cdist(X, X)
    nn_idx = dist.topk(k + 1, largest=False).indices[:, 1:]
    idx = torch.randint(0, N, (n_samples,), device=X.device)
    idx_nn = nn_idx[idx, torch.randint(0, k, (n_samples,), device=X.device)]
    lam = torch.rand(n_samples, 1, device=X.device)
    synth = X[idx] + lam * (X[idx_nn] - X[idx])
    return synth

def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing

def mixup(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

def get_class_balanced_weight(labels, beta=0.9999):
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    effective_num_pos = 1.0 - beta ** n_pos
    effective_num_neg = 1.0 - beta ** n_neg
    w_pos = (1.0 - beta) / (effective_num_pos + 1e-8)
    w_neg = (1.0 - beta) / (effective_num_neg + 1e-8)
    return w_pos, w_neg

def differentiable_ap_loss(logits, labels, delta=1.0):
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    diff = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)
    loss = torch.nn.functional.relu(1 + diff / delta)
    return loss.mean()

def dynamic_negative_sampling(logits, labels, n_pos, n_neg=None):
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    if n_neg is None:
        n_neg = n_pos * 5
    if neg_idx.numel() > n_neg:
        neg_scores = logits[neg_idx]
        _, topk = torch.topk(neg_scores, n_neg, largest=True)
        neg_idx = neg_idx[topk]
    sample_idx = torch.cat([pos_idx, neg_idx])
    return sample_idx

def compute_total_loss(
    logits, labels,
    loss_weights=None,
    smoothing=0.1,
    mixup_alpha=0.2,
    cb_beta=0.9999,
    ap_delta=1.0,
    focal_alpha=0.95,
    focal_gamma=2.5,
    device=None
):
    labels_smooth = smooth_labels(labels, smoothing)
    if mixup_alpha > 0:
        logits, labels_smooth = mixup(logits.unsqueeze(1), labels_smooth.unsqueeze(1), mixup_alpha)
        logits = logits.squeeze(1)
        labels_smooth = labels_smooth.squeeze(1)
    w_pos, w_neg = get_class_balanced_weight(labels, beta=cb_beta)
    weights = labels * w_pos + (1 - labels) * w_neg
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_smooth, reduction='none')
    prob = torch.sigmoid(logits)
    p_t = prob * labels + (1 - prob) * (1 - labels)
    alpha_t = focal_alpha * labels + (1 - focal_alpha) * (1 - labels)
    focal = alpha_t * (1 - p_t).pow(focal_gamma) * bce
    focal_loss = (focal * weights).mean()
    ap_loss = differentiable_ap_loss(logits, labels, delta=ap_delta)
    total = 0.0
    if loss_weights is None:
        loss_weights = {'focal': 1.0, 'ap': 1.0}
    total += loss_weights.get('focal', 1.0) * focal_loss
    total += loss_weights.get('ap', 1.0) * ap_loss
    return total, {'focal': focal_loss.item(), 'ap': ap_loss.item()}

def advanced_link_loss(
    P, C, link_head, pos_ei, neg_ei, device,
    loss_weights=None,
    smoothing=0.1,
    mixup_alpha=0.2,
    cb_beta=0.9999,
    ap_delta=1.0,
    focal_alpha=0.95,
    focal_gamma=2.5,
    smote_multiplier=2.0
):
    pos_logits = link_head(P, C, pos_ei)
    neg_logits = link_head(P, C, neg_ei)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    n_pos = pos_logits.size(0)
    n_neg = neg_logits.size(0)
    # SMOTE oversampling
    if n_pos > 0 and smote_multiplier > 1.0:
        target_pos = int(n_neg * smote_multiplier)
        if n_pos < target_pos:
            n_extra = target_pos - n_pos
            synth_pat = simple_smote(P[pos_ei[0]], n_extra)
            synth_ei = torch.stack([
                torch.arange(P.size(0), P.size(0) + n_extra, device=device),
                torch.full((n_extra,), pos_ei[1][0], device=device)
            ], dim=0)
            synth_logits = link_head(torch.cat([P, synth_pat], dim=0), C, synth_ei)
            logits = torch.cat([logits, synth_logits], dim=0)
            labels = torch.cat([labels, torch.ones_like(synth_logits)], dim=0)
    # Dynamic negative sampling
    sample_idx = dynamic_negative_sampling(logits, labels, n_pos)
    logits = logits[sample_idx]
    labels = labels[sample_idx]
    # Compute loss
    total_loss, loss_dict = compute_total_loss(
        logits, labels,
        loss_weights=loss_weights,
        smoothing=smoothing,
        mixup_alpha=mixup_alpha,
        cb_beta=cb_beta,
        ap_delta=ap_delta,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        device=device
    )
    return total_loss, loss_dict