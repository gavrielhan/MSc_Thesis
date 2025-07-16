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

    # Accumulators
    total_link_loss = 0.0
    total_cox_loss  = 0.0
    window_scores  = []
    # store the maximum predicted probability for each (patient, condition)
    patient_cond_max = defaultdict(float)
    all_risk_scores, all_durations, all_events = [], [], []
    num_batches = 0

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
            # if you also use the reverse relation in your model:
            edge_attr_dict[('signature', 'to_rev', 'patient')] = batch[rel_sig].edge_attr

        # 2) temporal 'follows' edges (3-day intervals)
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

        # 3) build full grid of (patient,cond)
        src_all = torch.arange(Np, device=device).repeat_interleave(Nc)  # [Np*Nc]
        dst_all = torch.arange(Nc, device=device).repeat(Np)            # [Np*Nc]
        full_ei = torch.stack([src_all, dst_all], dim=0)               # [2, Np*Nc]

        # 4) compute & blend logits using the fixed evaluation weights
        link_logits = link_head(P, C_emb, full_ei)     # [Np*Nc]
        risks       = cox_head(P)                      # [Np, Nc]
        cox_logits  = risks[src_all, dst_all]          # [Np*Nc]

        # now blend using the constants you set
        logits_all = BLEND_LINK * link_logits + BLEND_COX * cox_logits
        probs_all  = torch.sigmoid(logits_all)         # [Np*Nc]

        # save for sliding-window plot
        window_scores.append(probs_all.view(Np, Nc).cpu().detach().numpy())

        # 5) gather true positives in this window?s HORIZON
        flat_names = [
            n for sub in batch['patient'].name
              for n in (sub if isinstance(sub, list) else [sub])
        ]
        is_female_flat = torch.tensor(
            [gender_map.get(p,'female')=='female' for p in flat_names],
            device=device, dtype=torch.bool
        )
        pos_pairs = []
        for w in range(window_idx+1, window_idx+1+HORIZON):
            pos_pairs.extend(diag_by_win_map.get(w, []))

        # filter male?breast
        filtered = []
        for (p,ci) in pos_pairs:
            if ci==BREAST_IDX:
                if p in flat_names and is_female_flat[flat_names.index(p)]:
                    filtered.append((p,ci))
            else:
                filtered.append((p,ci))

        # map to node indices
        src_nodes, dst_nodes = [], []
        for (p,ci) in filtered:
            if p in flat_names:
                src_nodes.append(flat_names.index(p))
                dst_nodes.append(ci)

        # pack into pos_ei
        if src_nodes:
            pos_ei = torch.stack([
                torch.tensor(src_nodes, device=device),
                torch.tensor(dst_nodes, device=device)
            ], dim=0)  # [2, N_pos]
        else:
            pos_ei = torch.empty((2,0), device=device, dtype=torch.long)

        # 6) build dense labels over full grid
        labels_all = torch.zeros_like(probs_all)
        if pos_ei.numel()>0:
            flat_pos = pos_ei[0]*Nc + pos_ei[1]
            labels_all[flat_pos] = 1.0

        # 7) update patient-level maximum scores
        prob_matrix = probs_all.view(Np, Nc)
        if focus_idx is not None:
            cond_range = [focus_idx]
        else:
            cond_range = range(num_conditions)
        for i, pname in enumerate(flat_names):
            for ci in cond_range:
                score = prob_matrix[i, ci].item()
                key = (pname, ci)
                if score > patient_cond_max.get(key, 0.0):
                    patient_cond_max[key] = score

        # 8) balanced 1:1 BCE link loss
        pos_idx = (labels_all==1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_all==0).nonzero(as_tuple=True)[0]
        if pos_idx.numel()>0:
            k = pos_idx.numel()
            if neg_idx.numel()>k:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel())[:k]]
            sample_idx = torch.cat([pos_idx, neg_idx], dim=0)
        else:
            sample_idx = neg_idx

        logits_s = logits_all[sample_idx]
        labels_s = labels_all[sample_idx]
        total_link_loss += F.binary_cross_entropy_with_logits(logits_s, labels_s).item()

        # 9) Cox partial-likelihood
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
    avg_link = total_link_loss / max(1, num_batches)
    avg_cox  = total_cox_loss  / max(1, num_batches)

    # --- Patient level AUC calculations ---
    pos_pairs = set()
    for pairs in diag_by_win_map.values():
        pos_pairs.update(pairs)

    y_true, y_score = [], []
    for (p, ci), score in patient_cond_max.items():
        if focus_idx is not None and ci != focus_idx:
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
        if focus_idx is not None and ci != focus_idx:
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

    # C-index
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
    return (
        avg_link,    # balanced BCE link loss
        avg_cox,     # Cox loss
        None,        # node?CE (not computed here)
        None,        # node?acc (not computed here)
        c_idxs,
        mean_c,
        link_auc,
        pr_auc,
        pr_per_cond,
        window_scores,
        calibrators,
    )