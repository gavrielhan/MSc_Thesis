from model import (
    model, link_head, cox_head, patient_classifier, joint_head, in_dims, focus_idx, BREAST_IDX, gender_map, HORIZON, NEGATIVE_MULTIPLIER, SMOTE_MULTIPLIER, CONTRASTIVE_WEIGHT, build_diag_by_win
)
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import os
import json
import datetime
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
from losses import advanced_link_loss
from link_prediction_head import CoxHead
import numpy as np

# --- Training function and main loop ---
# (Move train() and main() logic here, update DataLoader num_workers)

def train(model, link_head, cox_head, patient_classifier, loader, optimizer, device, pseudo_label_by_window, diag_by_win_train, all_patient_names):
    model.train()
    link_head.train()
    cox_head.train()
    patient_classifier.train()

    total_loss = 0.0
    num_batches = 0

    name_to_idx = {name: idx for idx, name in enumerate(all_patient_names)}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        batch = batch.to(device)
        # 1) Sanity check inputs
        for ntype, x in batch.x_dict.items():
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError(f"[Batch {batch_idx}] Bad input on node '{ntype}': contains NaN or Inf")

        # 2) Skip if no useful edges
        if len(batch.edge_types) == 0 or (
            ('patient', 'has', 'condition') not in batch.edge_types
            and ('patient', 'follows', 'patient') not in batch.edge_types
        ):
            continue

        optimizer.zero_grad()
        # build edge_attr dict
        edge_attr_dict = {}
        for rel in [('patient','to','signature'), ('signature','to_rev','patient'),
                    ('patient','follows','patient'), ('patient','follows_rev','patient')]:
            if rel in batch.edge_types and hasattr(batch[rel], 'edge_attr'):
                edge_attr_dict[rel] = batch[rel].edge_attr

        # --- Positive edge oversampling for ('patient','has','condition') ---
        if 'patient' in batch.x_dict and ('patient','has','condition') in batch.edge_types and hasattr(batch['patient','has','condition'], 'edge_attr'):
            edge_attr = batch['patient','has','condition'].edge_attr.view(-1)
            pos_indices = (edge_attr == 1.0).nonzero(as_tuple=True)[0]
            if pos_indices.numel() > 0:
                edge_index = batch['patient','has','condition'].edge_index
                pos_edge_index = edge_index[:, pos_indices]
                pos_edge_attr = edge_attr[pos_indices]
                n_dup = 5
                dup_edge_index = pos_edge_index.repeat(1, n_dup)
                dup_edge_attr = pos_edge_attr.repeat(n_dup)
                # Concatenate to original edge_index and edge_attr
                batch['patient','has','condition'].edge_index = torch.cat([edge_index, dup_edge_index], dim=1)
                batch['patient','has','condition'].edge_attr = torch.cat([edge_attr, dup_edge_attr], dim=0)
                print(f"[DEBUG] Duplicated {pos_indices.numel()} positive edges {n_dup}x for oversampling.")

        # 3) Forward through GNN
        out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        P = out['patient']    # [N_pat, D]
        C = out['condition']  # [C, D]
        zero = P.sum() * 0.0

        # --- New: Link regression loss for all patient-condition edges ---
        if ('patient','has','condition') in batch.edge_types and hasattr(batch['patient','has','condition'], 'edge_attr'):
            edge_index = batch['patient','has','condition'].edge_index
            edge_attr = batch['patient','has','condition'].edge_attr.view(-1)
            pred = link_head(P, C, edge_index)
            mse_loss = F.mse_loss(pred, edge_attr)
            # --- Updated: diagnosis_targets is 1 if diagnosis occurs within HORIZON ---
            diagnosis_targets = torch.zeros_like(edge_attr, device=pred.device)
            # batch.window is the current window index (int)
            t = int(batch.window)
            H = HORIZON if 'HORIZON' in globals() else 50
            for idx in range(edge_index.size(1)):
                pi = edge_index[0, idx].item()
                ci = edge_index[1, idx].item()
                pname = batch['patient'].name[0][pi] if isinstance(batch['patient'].name[0], list) else batch['patient'].name[pi]
                # Check if (pname, ci) is diagnosed in [t, t+HORIZON]
                positive = False
                for wi in range(t, t + H + 1):
                    if (pname, ci) in set(diag_by_win_train.get(wi, [])):
                        positive = True
                        break
                if positive:
                    diagnosis_targets[idx] = 1.0
            diagnosis_weights = torch.ones_like(edge_attr, device=pred.device)
            diagnosis_weights[diagnosis_targets == 1.0] = 20.0  # Increased pos_weight
            weighted_loss = F.binary_cross_entropy_with_logits(pred, diagnosis_targets, weight=diagnosis_weights, reduction='mean')
            link_loss = mse_loss + weighted_loss
            # Debug print for batch composition
            if batch_idx < 5:
                n_pos = diagnosis_targets.sum().item()
                n_neg = len(diagnosis_targets) - n_pos
                print(f"[DEBUG] Batch {batch_idx}: {n_pos} positives, {n_neg} negatives (HORIZON positives)")
            if batch_idx == 0:
                print(f"[TRAIN] MSE loss: {mse_loss.item():.4f} | Weighted diagnosis loss: {weighted_loss.item():.4f} | Total link loss: {link_loss.item():.4f}")
        else:
            link_loss = zero.clone()

        # 6) Cox loss (unchanged)
        cox_loss = zero.clone()
        if hasattr(batch['patient'],'event') and hasattr(batch['patient'],'duration'):
            batch_names = [n for sub in batch['patient'].name for n in (sub if isinstance(sub, list) else [sub])]
            batch_indices = torch.tensor([name_to_idx[n] for n in batch_names], device=device)
            du = batch['patient'].duration[batch_indices]
            ev = batch['patient'].event[batch_indices]
            rs = cox_head(P)
            if (ev > 0).sum().item() > 0:
                cox_loss = CoxHead.cox_partial_log_likelihood(rs, du, ev)

        # 7) Pseudo?label CE (unchanged)
        node_ps = zero.clone()
        t = int(batch.window)
        lm = pseudo_label_by_window.get(t,{})
        if lm:
            idxs = torch.tensor(list(lm.keys()), device=device)
            labs = torch.tensor([lm[i] for i in idxs.cpu().tolist()], device=device)
            logits = patient_classifier(P[idxs])
            node_ps = F.cross_entropy(logits, labs)

        # 8) Backward
        total = joint_head(link_loss, cox_loss) + 0.2 * node_ps
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(link_head.parameters()) +
            list(cox_head.parameters()) +
            list(patient_classifier.parameters()) +
            list(joint_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += total.item()
        num_batches += 1

    return total_loss / max(1, num_batches) 