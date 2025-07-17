from model import (
    model, link_head, cox_head, joint_head, patient_classifier, in_dims, focus_idx, BREAST_IDX, gender_map, HORIZON, NEGATIVE_MULTIPLIER, build_diag_by_win
)
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from lifelines.utils import concordance_index
from tqdm import tqdm
import torch.nn.functional as F
from link_prediction_head import CoxHead
from risk_calibration import  fit_absolute_risk_calibrator

# --- Evaluation function and helpers ---
# (Move evaluate() and related logic here)

def evaluate(
    model,
    link_head,
    cox_head,
    joint_head,
    patient_classifier,       # unused here but kept for signature
    loader,
    device,
    NEGATIVE_MULTIPLIER,      # unused in eval
    diag_by_win_map,
    frozen_condition_embeddings,
    plot_idx,                 # unused in this function
):
    model.eval()
    link_head.eval()
    cox_head.eval()
    joint_head.eval()
    import torch

    # Accumulators
    total_link_loss = 0.0
    link_loss_batches = 0
    total_cox_loss  = 0.0
    window_scores  = []
    # store the maximum predicted probability for each (patient, condition)
    patient_cond_max = defaultdict(float)
    all_risk_scores, all_durations, all_events = [], [], []
    num_batches = 0
    mse_losses = []
    weighted_losses = []

    with torch.no_grad():
        w_link = torch.exp(-F.softplus(joint_head.s_link)).item()
        w_cox  = torch.exp(-F.softplus(joint_head.s_cox)).item()
    norm = w_link + w_cox + 1e-8
    BLEND_LINK = w_link / norm
    BLEND_COX  = w_cox  / norm
    num_conditions = cox_head.linear.out_features

    for window_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        batch = batch.to(device)

        # 1) skip if no patients
        if batch['patient'].x.size(0) == 0:
            continue

        # 2) GNN forward
        edge_index_dict = {
            et: batch[et].edge_index
            for et in batch.edge_types
            if et != ('patient','has','condition')
        }
        edge_attr_dict = {}

        # 1) signature edges (as before)
        rel_sig = ('patient', 'to', 'signature')
        if rel_sig in batch.edge_types and hasattr(batch[rel_sig], 'edge_attr'):
            edge_attr_dict[rel_sig] = batch[rel_sig].edge_attr
            edge_attr_dict[('signature', 'to_rev', 'patient')] = batch[rel_sig].edge_attr

        for rel in [
            ('patient', 'follows', 'patient'),
            ('patient', 'follows_rev', 'patient')
        ]:
            if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
                edge_attr_dict[rel] = batch[rel].edge_attr

        out   = model(batch.x_dict, edge_index_dict, edge_attr_dict)
        P     = out['patient']                            # [N_pat, D]
        C_emb = frozen_condition_embeddings.to(device)    # [C, D]
        Np, Nc = P.size(0), C_emb.size(0)
        # Robustly flatten names if they are lists (possibly nested)
        names = batch['patient'].name[0]

        # --- Predict for all (patient, condition) pairs ---
        pred_count = 0
        window_ys, window_yt = [], []
        mse_targets, mse_preds = [], []
        weighted_targets, weighted_preds, weighted_weights = [], [], []
        if focus_idx is not None:
            conds = [focus_idx]
        else:
            conds = list(range(Nc))
        # Compute baseline risks for this window
        from model import baseline_risk_diabetes
        baseline_risks = baseline_risk_diabetes(names, window_idx)
        for pi, pname in enumerate(names):
            for ci in conds:
                edge_index = torch.tensor([[pi], [ci]], dtype=torch.long, device=device)
                pred = link_head(P, C_emb, edge_index)
                score = pred.item()
                # Clamp prediction to [0.0, 1.0]
                score = max(0.0, min(1.0, score))
                key = (pname, ci)
                if score > patient_cond_max.get(key, float('-inf')):
                    patient_cond_max[key] = score
                pred_count += 1
                # For link loss: get true label for this (patient, condition) in this window
                # True label is 1 if (pname, ci) in any diag_by_win_map[wi] for wi in [window_idx, window_idx+HORIZON]
                H = HORIZON if 'HORIZON' in globals() else 50  # fallback if not imported
                future_pos = False
                for wi in range(window_idx, window_idx + H + 1):
                    if (pname, ci) in set(diag_by_win_map.get(wi, [])):
                        future_pos = True
                        break
                true_label = 1.0 if future_pos else 0.0
                window_ys.append(score)
                window_yt.append(true_label)
                # --- Calculate edge attribute as in training ---
                # Find all future diagnosis windows for this patient-condition
                future_diags = [w for w, pairs in diag_by_win_map.items() if w >= window_idx and (pname, ci) in pairs]
                if not future_diags:
                    # Never diagnosed
                    edge_attr = baseline_risks[pi]
                else:
                    diag_win = min(future_diags)
                    if window_idx == diag_win:
                        edge_attr = 1.0
                    elif window_idx < diag_win:
                        total = max(1, diag_win - window_idx)
                        progress = 1 - total / (diag_win - window_idx + 1)
                        baseline = baseline_risks[pi]
                        edge_attr = baseline + (1.0 - baseline) * progress
                        edge_attr = min(edge_attr, 1.0)
                mse_targets.append(edge_attr)
                mse_preds.append(score)
                # For weighted loss: 1.0 if edge_attr==1.0, else 0.0
                weighted_targets.append(1.0 if edge_attr == 1.0 else 0.0)
                weighted_preds.append(score)
                weighted_weights.append(10.0 if edge_attr == 1.0 else 1.0)  # Example: 10x weight for positives
        # Compute link loss for this window
        if window_ys and window_yt:
            ys_tensor = torch.tensor(window_ys)
            yt_tensor = torch.tensor(window_yt)
            pos_weight = 1000.0  # Weight for positive samples
            weights = torch.ones_like(yt_tensor)
            weights[yt_tensor == 1.0] = pos_weight
            batch_link_loss = torch.mean(weights * (ys_tensor - yt_tensor) ** 2).item()
            total_link_loss += batch_link_loss
            link_loss_batches += 1
        # --- Compute MSE loss on edge attributes ---
        if mse_targets:
            mse_loss = F.mse_loss(torch.tensor(mse_preds), torch.tensor(mse_targets)).item()
            mse_losses.append(mse_loss)
        # --- Compute weighted binary loss for diagnosis ---
        if weighted_targets:
            pred_tensor = torch.tensor(weighted_preds)
            target_tensor = torch.tensor(weighted_targets)
            weight_tensor = torch.tensor(weighted_weights)
            # Weighted BCE loss
            bce = F.binary_cross_entropy(pred_tensor, target_tensor, weight=weight_tensor, reduction='mean').item()
            weighted_losses.append(bce)

        # 9) Cox partial-likelihood (unchanged)
        risks = cox_head(P)
        if hasattr(batch['patient'],'event'):
            evts = batch['patient'].event
            durs = batch['patient'].duration
            all_risk_scores.append(risks.cpu())
            all_durations .append(durs.cpu())
            all_events    .append(evts.cpu())

            for ci in range(num_conditions):
                ev_col = evts[:,ci]
                if ev_col.sum().item()>0:
                    loss_ci = CoxHead.cox_partial_log_likelihood(
                        risks[:,ci].unsqueeze(1),
                        durs [:,ci].unsqueeze(1),
                        evts[:,ci].unsqueeze(1)
                    )
                    total_cox_loss += loss_ci.item()

        num_batches += 1

    # finalize
    avg_link = total_link_loss / max(1, link_loss_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)
    avg_mse_loss = sum(mse_losses) / max(1, len(mse_losses))
    avg_weighted_loss = sum(weighted_losses) / max(1, len(weighted_losses))
    total_eval_loss = avg_mse_loss + avg_weighted_loss

    # --- Patient level AUC calculations ---
    pos_pairs = set()
    for pairs in diag_by_win_map.values():
        pos_pairs.update(pairs)

    # Debug: count positive and predicted pairs for focus_idx
    if plot_idx is not None:
        n_pos_pairs = sum(1 for (p, c) in pos_pairs if c == plot_idx)
        n_pred_pairs = sum(1 for (p, c) in patient_cond_max if c == plot_idx)

    y_true, y_score = [], []
    for (p, ci), score in patient_cond_max.items():
        if plot_idx is not None and ci != plot_idx:
            continue
        y_score.append(score)
        y_true.append(1 if (p, ci) in pos_pairs else 0)

    if len(set(y_true)) > 1:
        link_auc = roc_auc_score(y_true, y_score)
        pr_auc   = average_precision_score(y_true, y_score)
    else:
        link_auc = pr_auc = None

    pr_per_cond = []
    for ci in range(num_conditions):
        if plot_idx is not None and ci != plot_idx:
            pr_per_cond.append(None)
            continue
        ys, yt = [], []
        for (p, c), score in patient_cond_max.items():
            if c == ci:
                ys.append(score)
                yt.append(1 if (p, c) in pos_pairs else 0)
        if yt and sum(yt) > 0:
            pr_per_cond.append(average_precision_score(yt, ys))
        else:
            pr_per_cond.append(None)

    # C-index (unchanged)
    c_idxs = []
    if all_risk_scores:
        R = torch.cat(all_risk_scores).detach().numpy()
        D = torch.cat(all_durations).detach().numpy()
        E = torch.cat(all_events).detach().numpy()
        for ci in range(num_conditions):
            mask = E[:,ci]>=0
            if mask.sum()>0:
                try:
                    c_idxs.append(concordance_index(
                        D[mask,ci],
                        -R[mask,ci],
                        E[mask,ci]
                    ))
                except ZeroDivisionError:
                    c_idxs.append(None)
            else:
                c_idxs.append(None)
    mean_c = None
    valid  = [c for c in c_idxs if c is not None]
    if valid:
        mean_c = sum(valid)/len(valid)
    calibrators = None
    if all_risk_scores:
        R_np = torch.cat(all_risk_scores).detach().numpy()
        D_np = torch.cat(all_durations).detach().numpy()
        E_np = torch.cat(all_events).detach().numpy()
        calibrators = fit_absolute_risk_calibrator(R_np, D_np, E_np)
    print(f"[EVAL] MSE loss: {avg_mse_loss:.4f} | Weighted diagnosis loss: {avg_weighted_loss:.4f} | Total eval loss: {total_eval_loss:.4f}")
    return (
        avg_link,    # MSE link loss (legacy)
        avg_cox,     # Cox loss
        avg_mse_loss, # New: MSE on edge attributes
        avg_weighted_loss, # New: weighted binary loss for diagnosis
        total_eval_loss,   # Sum of both losses
        c_idxs,
        mean_c,
        link_auc,
        pr_auc,
        pr_per_cond,
        window_scores,
        calibrators,
        patient_cond_max,  # <-- add this to the return tuple
    )