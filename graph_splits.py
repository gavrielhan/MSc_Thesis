import torch
from torch_geometric.data import HeteroData
import os
import random
from tqdm import tqdm
from collections import Counter

# CONFIG
GRAPHS_PATH = "glucose_sleep_graphs_3d.pt"
SAVE_DIR = "split"
SPLIT_RATIO = (0.6, 0.2, 0.2)
SEED = 42
NUM_CONDITIONS = 7
once = 1
# Load graphs
graphs = torch.load(GRAPHS_PATH, weights_only=False)
random.seed(SEED)

# Get all patient names
all_patients = set()
for g in graphs:
    if 'patient' in g.node_types and 'name' in g['patient']:
        all_patients.update(g['patient']['name'])

all_patients = list(all_patients)
random.shuffle(all_patients)
n = len(all_patients)
train_patients = set(all_patients[:int(SPLIT_RATIO[0]*n)])
val_patients = set(all_patients[int(SPLIT_RATIO[0]*n):int((SPLIT_RATIO[0]+SPLIT_RATIO[1])*n)])
test_patients = set(all_patients[int((SPLIT_RATIO[0]+SPLIT_RATIO[1])*n):])

# Build diagnosis index: patient -> {condition_idx: graph_idx}
patient_condition_diagnosis_idx = {}
for idx, g in enumerate(graphs):
    if ('patient', 'has', 'condition') in g.edge_types:
        edges = g[('patient', 'has', 'condition')].edge_index
        src_nodes = edges[0].tolist()
        dst_conditions = edges[1].tolist()
        for src, cond in zip(src_nodes, dst_conditions):
            name = g['patient']['name'][src]
            patient_condition_diagnosis_idx.setdefault(name, {})[cond] = idx

# Patient appearances (for temporal edges)
patient_occurrences = {}
for idx, g in enumerate(graphs):
    for i, name in enumerate(g['patient']['name']):
        patient_occurrences.setdefault(name, []).append((idx, i))

# Helper
def get_event_duration(idx, name, cond_idx, patient_cond_idx_map):
    if name in patient_cond_idx_map and cond_idx in patient_cond_idx_map[name]:
        event_idx = patient_cond_idx_map[name][cond_idx]
        if event_idx == idx:
            event = 1
            # Avoid zero duration which would be ignored by Cox loss
            duration = 1.0
        elif event_idx > idx:
            event = 0
            duration = event_idx - idx
        else:
            # Diagnosis already happened in the past
            event = 0
            duration = 0.0
    else:
        event = 0
        duration = 0.0
    return event, duration


# Filter and relabel graphs
split_graphs = {'train': [], 'val': [], 'test': []}
for i, g in enumerate(tqdm(graphs, desc="Filtering and labeling")):
    for split_name, split_set in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
        if 'patient' not in g.node_types:
            continue

        patient_mask_list = []
        for name in g['patient']['name']:
            # Normal split assignment
            in_split = name in split_set

            # Additionally: if this patient is having an event now, we need to keep them
            has_event_now = False
            if name in patient_condition_diagnosis_idx:
                for cond_idx, event_graph_idx in patient_condition_diagnosis_idx[name].items():
                    if event_graph_idx == i:  # i = current graph index
                        has_event_now = True
                        break

            patient_mask_list.append(name in split_set)

        patient_mask = torch.tensor(patient_mask_list, dtype=torch.bool)

        if patient_mask.sum() == 0:
            continue

        new_g = HeteroData()
        kept_indices = torch.nonzero(patient_mask, as_tuple=False).squeeze()
        old_to_new_index = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(kept_indices)}

        for node_type in g.node_types:
            if node_type == 'patient':
                for k, v in g[node_type].items():
                    if isinstance(v, torch.Tensor) and v.size(0) == patient_mask.size(0):
                        new_g[node_type][k] = v[patient_mask]
                    elif isinstance(v, list) and len(v) == patient_mask.size(0):
                        new_g[node_type][k] = [val for val, keep in zip(v, patient_mask.tolist()) if keep]
                    else:
                        new_g[node_type][k] = v

                filtered_names = new_g[node_type]['name']

                # Calculate event and duration per condition
                events = []
                durations = []
                for name in filtered_names:
                    patient_events = []
                    patient_durations = []
                    for cond_idx in range(NUM_CONDITIONS):
                        e, d = get_event_duration(i, name, cond_idx, patient_condition_diagnosis_idx)
                        patient_events.append(e)
                        patient_durations.append(d)
                    events.append(patient_events)
                    durations.append(patient_durations)

                new_g[node_type]['event'] = torch.tensor(events, dtype=torch.long)  # [num_patients, num_conditions]
                new_g[node_type]['duration'] = torch.tensor(durations, dtype=torch.float)  # [num_patients, num_conditions]


            else:
                for k, v in g[node_type].items():
                    new_g[node_type][k] = v

        # Filter edges
        for edge_type in g.edge_types:
            src, rel, dst = edge_type
            edge_index = g[edge_type].edge_index
            edge_attr = g[edge_type].edge_attr if 'edge_attr' in g[edge_type] else None

            if src == 'patient' or dst == 'patient':
                new_src, new_dst, new_attr = [], [], []
                for j in range(edge_index.size(1)):
                    s, d = edge_index[0, j].item(), edge_index[1, j].item()
                    if src == 'patient' and s not in old_to_new_index:
                        continue
                    if dst == 'patient' and d not in old_to_new_index:
                        continue
                    new_src.append(old_to_new_index[s] if src == 'patient' else s)
                    new_dst.append(old_to_new_index[d] if dst == 'patient' else d)
                    if edge_attr is not None:
                        new_attr.append(edge_attr[j])

                if len(new_src) > 0:
                    new_g[edge_type].edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
                    if edge_attr is not None:
                        new_g[edge_type].edge_attr = torch.stack(new_attr)
            else:
                new_g[edge_type].edge_index = edge_index
                if edge_attr is not None:
                    new_g[edge_type].edge_attr = edge_attr

        split_graphs[split_name].append(new_g)

# Temporal patient edges
for split_name, graphs in split_graphs.items():
    name_to_indices = {}
    for t, g in enumerate(graphs):
        for i, name in enumerate(g['patient']['name']):
            name_to_indices.setdefault(name, []).append((t, i))

    for name, occurrences in name_to_indices.items():
        for k in range(1, len(occurrences)):
            prev_t, prev_i = occurrences[k-1]
            curr_t, curr_i = occurrences[k]
            edge = torch.tensor([[prev_i], [curr_i]], dtype=torch.long)
            if ('patient', 'follows', 'patient') in graphs[curr_t].edge_types:
                graphs[curr_t][('patient', 'follows', 'patient')].edge_index = torch.cat([
                    graphs[curr_t][('patient', 'follows', 'patient')].edge_index, edge
                ], dim=1)
            else:
                graphs[curr_t][('patient', 'follows', 'patient')].edge_index = edge

# Save
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(split_graphs['train'], os.path.join(SAVE_DIR, 'train_graphs_3d.pt'))
torch.save(split_graphs['val'], os.path.join(SAVE_DIR, 'val_graphs_3d.pt'))
torch.save(split_graphs['test'], os.path.join(SAVE_DIR, 'test_graphs_3d.pt'))

# Quick Checks
print("\nRunning checks...")
for split_name, gset in split_graphs.items():
    for idx, g in enumerate(gset):
        if 'patient' not in g.node_types:
            print(f"[{split_name}] Graph {idx} missing patients.")
        if 'event' not in g['patient'] or 'duration' not in g['patient']:
            print(f"[{split_name}] Graph {idx} missing event/duration.")
        if g['patient']['event'].ndim != 2 or g['patient']['duration'].ndim != 2:
            print(f"[{split_name}] Graph {idx} wrong event/duration shape!")

def check_split(graphs, split_name):
    total_graphs = len(graphs)
    # Initialize counters and stats
    cond_counter = Counter()
    total_cond_edges = 0
    cond_graphs_with = 0

    total_sig_edges = 0
    sig_graphs_with = 0

    total_follow_edges = 0
    follow_graphs_with = 0

    for g in graphs:
        # Condition edges
        if ('patient', 'has', 'condition') in g.edge_types:
            num = g['patient', 'has', 'condition'].edge_index.size(1)
            total_cond_edges += num
            if num > 0:
                cond_graphs_with += 1
            # count distribution by condition index
            cond_counter.update(g['patient', 'has', 'condition'].edge_index[1].tolist())
        else:
            num = 0

        # Signature edges
        if ('patient', 'to', 'signature') in g.edge_types:
            num_sig = g['patient', 'to', 'signature'].edge_index.size(1)
        else:
            num_sig = 0
        total_sig_edges += num_sig
        if num_sig > 0:
            sig_graphs_with += 1

        # Follow edges
        if ('patient', 'follows', 'patient') in g.edge_types:
            num_follow = g['patient', 'follows', 'patient'].edge_index.size(1)
        else:
            num_follow = 0
        total_follow_edges += num_follow
        if num_follow > 0:
            follow_graphs_with += 1

    print(f"\n=== {split_name.upper()} SPLIT ===")
    print(f"Total graphs: {total_graphs}")
    # Condition edges summary
    print(f"Condition edges: total={total_cond_edges}, graphs with ?1={cond_graphs_with} ({cond_graphs_with/total_graphs:.1%}), avg/graph={total_cond_edges/total_graphs:.2f}")
    # Signature edges summary
    print(f"Signature edges: total={total_sig_edges}, graphs with ?1={sig_graphs_with} ({sig_graphs_with/total_graphs:.1%}), avg/graph={total_sig_edges/total_graphs:.2f}")
    # Follow edges summary
    print(f"Follow edges: total={total_follow_edges}, graphs with ?1={follow_graphs_with} ({follow_graphs_with/total_graphs:.1%}), avg/graph={total_follow_edges/total_graphs:.2f}")
    # Condition distribution
    print("Condition edge distribution (condition_idx: count):")
    for cond_idx, count in sorted(cond_counter.items()):
        print(f"  {cond_idx:2d}: {count}")
    print("-" * 40)

# Load splits
train_graphs = torch.load("split/train_graphs_3d.pt", weights_only=False)
val_graphs   = torch.load("split/val_graphs_3d.pt",   weights_only=False)
test_graphs  = torch.load("split/test_graphs_3d.pt",  weights_only=False)

# Run checks
check_split(train_graphs, "train")
check_split(val_graphs,   "val")
check_split(test_graphs,  "test")

print("All checks passed.")
