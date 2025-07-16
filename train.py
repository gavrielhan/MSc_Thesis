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

        # 3) Forward through GNN
        out = model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        P = out['patient']    # [N_pat, D]
        C = out['condition']  # [C, D]
        zero = P.sum() * 0.0

        # 4) Gather true positives in horizon
        t = int(batch.window)
        pos_triples = []
        for w in range(t+1, t+1+HORIZON):
            for (p, ci) in diag_by_win_train.get(w, []):
                if focus_idx is None or ci == focus_idx:
                    pos_triples.append((p, ci, w - t))

        # 5) Map to indices & filter
        flat_names = [n for sub in batch['patient'].name
                      for n in (sub if isinstance(sub, list) else [sub])]
        filtered = []
        for p, ci, d in pos_triples:
            if ci == BREAST_IDX:
                if p in flat_names and gender_map.get(p,'female')=='female':
                    filtered.append((p,ci,d))
            else:
                if p in flat_names:
                    filtered.append((p,ci,d))

        src, dst, dists = [], [], []
        for p, ci, d in filtered:
            idx = flat_names.index(p)
            src.append(idx); dst.append(ci); dists.append(d)

        if src:
            # positives
            pos_ei = torch.stack([torch.tensor(src, device=device),
                                   torch.tensor(dst, device=device)], dim=0)
            tte_pos = batch['patient'].duration[pos_ei[0], pos_ei[1]]
            # negatives
            with torch.no_grad():
                all_src = torch.arange(P.size(0), device=device)
                all_dst = torch.full((P.size(0),), focus_idx, device=device)
                all_ei  = torch.stack([all_src, all_dst], dim=0)
                tte_all = batch['patient'].duration[all_src, all_dst]
                all_logits = link_head(P, C, all_ei, tte_all)
            pos_pairs = set((int(i), int(c)) for i, c in zip(pos_ei[0], pos_ei[1]))
            neg_mask = [(s.item(), focus_idx) not in pos_pairs for s in all_src]
            cand_src   = all_src[neg_mask]
            cand_logits = all_logits[neg_mask]
            n_neg = pos_ei.size(1) * NEGATIVE_MULTIPLIER
            if cand_logits.numel() >= n_neg:
                _, topk = cand_logits.topk(n_neg, largest=True)
                neg_src = cand_src[topk]
            else:
                neg_src = cand_src
            neg_dst = torch.full_like(neg_src, focus_idx, device=device)
            neg_ei  = torch.stack([neg_src, neg_dst], dim=0)
            tte_neg = batch['patient'].duration[neg_ei[0], neg_ei[1]]
            # --- Use advanced_link_loss ---
            link_loss, loss_dict = advanced_link_loss(
                P, C, link_head, pos_ei, neg_ei, device,
                loss_weights={'focal': 1.0, 'ap': 1.0},
                smoothing=0.1,
                mixup_alpha=0.2,
                cb_beta=0.9999,
                ap_delta=1.0,
                focal_alpha=0.95,
                focal_gamma=2.5,
                smote_multiplier=SMOTE_MULTIPLIER
            )
        else:
            link_loss = zero.clone()

        # 6) Cox loss
        cox_loss = zero.clone()
        if hasattr(batch['patient'],'event') and hasattr(batch['patient'],'duration'):
            batch_names = [n for sub in batch['patient'].name for n in (sub if isinstance(sub, list) else [sub])]
            batch_indices = torch.tensor([name_to_idx[n] for n in batch_names], device=device)
            du = batch['patient'].duration[batch_indices]
            ev = batch['patient'].event[batch_indices]
            rs = cox_head(P)
            if (ev > 0).sum().item() > 0:
                cox_loss = CoxHead.cox_partial_log_likelihood(rs, du, ev)

        # 7) Pseudo?label CE
        node_ps = zero.clone()
        lm = pseudo_label_by_window.get(t,{})
        if lm:
            idxs = torch.tensor(list(lm.keys()), device=device)
            labs = torch.tensor([lm[i] for i in idxs.cpu().tolist()],
                                device=device)
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