import torch
import os
import random

# Paths to your split files
splits = {
    'train': 'split/train_graphs_3d.pt',
    'val': 'split/test_graphs_3d.pt',
    'test': 'split/val_graphs_3d.pt',
}

def check_temporal_edges(graphs, split_name):
    print(f"\nChecking temporal edges for {split_name}...")
    for wi in range(len(graphs)-1):
        g_now = graphs[wi]
        g_next = graphs[wi+1]
        names_now = g_now['patient'].name
        names_next = g_next['patient'].name
        name_to_idx_next = {n: i for i, n in enumerate(names_next)}
        # For each patient in current window, check follows edge to self in next window
        if ('patient','follows','patient') in g_now.edge_types:
            ei = g_now['patient','follows','patient'].edge_index
            for i, name in enumerate(names_now):
                if name in name_to_idx_next:
                    idx_next = name_to_idx_next[name]
                    found = False
                    for j in range(ei.size(1)):
                        if ei[0,j] == i and ei[1,j] == idx_next:
                            found = True
                            break
                    if not found:
                        print(f"[ERROR] Patient {name} (idx {i}) in window {wi} does not follow self in window {wi+1}.")
        # Check reverse
        if ('patient','follows_rev','patient') in g_next.edge_types:
            ei_rev = g_next['patient','follows_rev','patient'].edge_index
            for i, name in enumerate(names_now):
                if name in name_to_idx_next:
                    idx_next = name_to_idx_next[name]
                    found = False
                    for j in range(ei_rev.size(1)):
                        if ei_rev[0,j] == idx_next and ei_rev[1,j] == i:
                            found = True
                            break
                    if not found:
                        print(f"[ERROR] Patient {name} (idx {i}) in window {wi+1} does not have follows_rev from self in window {wi}.")

def check_patient_removal_after_diagnosis(graphs, split_name):
    if split_name != 'train':
        return
    print(f"\nChecking patient removal after diagnosis for {split_name}...")
    diagnosed_patients = set()
    for wi, g in enumerate(graphs):
        names = g['patient'].name
        # Find patients diagnosed in this window
        if ('patient','has','condition') in g.edge_types:
            ei = g['patient','has','condition'].edge_index
            for i in range(ei.size(1)):
                p_idx = ei[0,i].item()
                diagnosed_patients.add(names[p_idx])
        # Check that diagnosed patients do not appear in later windows
        for name in names:
            if name in diagnosed_patients:
                print(f"[ERROR] Patient {name} reappears in window {wi} after diagnosis.")

def check_reverse_edges(graphs, split_name):
    print(f"\nChecking reverse edges for {split_name}...")
    reverse_map = {
        ('patient','to','signature'): ('signature','to_rev','patient'),
        ('signature','to_rev','patient'): ('patient','to','signature'),
        ('patient','has','condition'): ('condition','has_rev','patient'),
        ('condition','has_rev','patient'): ('patient','has','condition'),
        ('patient','follows','patient'): ('patient','follows_rev','patient'),
        ('patient','follows_rev','patient'): ('patient','follows','patient'),
    }
    for wi, g in enumerate(graphs):
        for et in g.edge_types:
            rev = reverse_map.get(et, None)
            if rev and rev not in g.edge_types:
                print(f"[ERROR] Window {wi}: Edge type {et} missing reverse {rev}.")

def check_signature_edge_attr_sum(graphs, split_name):
    print(f"\nChecking signature edge attribute sums for {split_name}...")
    for wi, g in enumerate(graphs):
        if ('patient','to','signature') in g.edge_types:
            ei = g['patient','to','signature'].edge_index
            ea = g['patient','to','signature'].edge_attr
            names = g['patient'].name
            for i, name in enumerate(names):
                mask = (ei[0] == i)
                s = ea[mask].sum().item()
                if not abs(s - 1.0) < 1e-4:
                    print(f"[ERROR] Window {wi}: Patient {name} sum of signature edge attrs is {s} (should be 1.0)")

def check_node_embeddings(graphs, split_name):
    print(f"\nChecking node embeddings for {split_name}...")
    if not graphs:
        print("[WARNING] No graphs to check.")
        return
    # Sample a random window
    wi = random.randint(0, len(graphs)-1)
    g = graphs[wi]
    print(f"Randomly sampling from window {wi}...")
    for ntype in g.x_dict:
        x = g.x_dict[ntype]
        if x.size(0) == 0:
            print(f"[WARNING] No nodes of type {ntype} in window {wi}.")
            continue
        idx = random.randint(0, x.size(0)-1)
        emb = x[idx]
        print(f"Node type '{ntype}', sampled node {idx}, embedding: {emb[:10]}{'...' if emb.numel() > 10 else ''}")

def main():
    for split_name, path in splits.items():
        if not os.path.exists(path):
            print(f"[WARNING] {path} not found, skipping {split_name}.")
            continue
        graphs = torch.load(path, weights_only=False)
        check_temporal_edges(graphs, split_name)
        check_patient_removal_after_diagnosis(graphs, split_name)
        check_reverse_edges(graphs, split_name)
        check_signature_edge_attr_sum(graphs, split_name)
        check_node_embeddings(graphs, split_name)
    print("\nGraph integrity checks complete.")

if __name__ == "__main__":
    main()