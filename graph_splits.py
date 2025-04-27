import torch
from torch_geometric.data import HeteroData
import os
import random
from tqdm import tqdm

# CONFIG
GRAPHS_PATH = "glucose_sleep_graphs.pt"  # path to the list of daily graphs
SAVE_DIR = "split"  # where to save train/val/test
SPLIT_RATIO = (0.7, 0.15, 0.15)
SEED = 42

# Load graphs
graphs = torch.load(GRAPHS_PATH, weights_only=False)
random.seed(SEED)

# Get all patient names from all graphs
all_patients = set()
for g in graphs:
    if 'patient' in g.node_types and 'name' in g['patient']:
        all_patients.update(g['patient']['name'])

# Split patients
all_patients = list(all_patients)
random.shuffle(all_patients)
n = len(all_patients)
train_patients = set(all_patients[:int(SPLIT_RATIO[0]*n)])
val_patients = set(all_patients[int(SPLIT_RATIO[0]*n):int((SPLIT_RATIO[0]+SPLIT_RATIO[1])*n)])
test_patients = set(all_patients[int((SPLIT_RATIO[0]+SPLIT_RATIO[1])*n):])

# Helper to get event/duration labels
def get_event_duration(idx, name, patient_diagnosis_idx):
    if name in patient_diagnosis_idx:
        event_idx = patient_diagnosis_idx[name]
        event = 1
        duration = event_idx - idx
    else:
        event = 0
        duration = 0.0
    return event, duration

# Build diagnosis index
patient_diagnosis_idx = {}
for idx, g in enumerate(graphs):
    if ('patient', 'has', 'condition') in g.edge_types:
        edges = g[('patient', 'has', 'condition')].edge_index
        for src in edges[0].tolist():
            name = g['patient']['name'][src]
            if name not in patient_diagnosis_idx:
                patient_diagnosis_idx[name] = idx

# Track patient indices over time to preserve temporal edges
patient_occurrences = {}
for idx, g in enumerate(graphs):
    for i, name in enumerate(g['patient']['name']):
        patient_occurrences.setdefault(name, []).append((idx, i))

# Filter graphs per split and label
split_graphs = {'train': [], 'val': [], 'test': []}
for i, g in enumerate(tqdm(graphs, desc="Filtering and labeling")):
    for split_name, split_set in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
        if 'patient' not in g.node_types:
            continue

        patient_mask = torch.tensor([name in split_set for name in g['patient']['name']], dtype=torch.bool)
        if patient_mask.sum() == 0:
            continue

        new_g = HeteroData()
        kept_indices = torch.nonzero(patient_mask, as_tuple=False).squeeze()
        old_to_new_index = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(kept_indices)}

        # Copy all node types
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
                new_g[node_type]['event'] = torch.tensor([
                    get_event_duration(i, name, patient_diagnosis_idx)[0] for name in filtered_names
                ], dtype=torch.long)
                new_g[node_type]['duration'] = torch.tensor([
                    get_event_duration(i, name, patient_diagnosis_idx)[1] for name in filtered_names
                ], dtype=torch.float)
            else:
                for k, v in g[node_type].items():
                    new_g[node_type][k] = v

        # Copy and filter edges
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

# Add temporal patient-patient edges between graphs in each split
for split_name, graphs in split_graphs.items():
    name_to_indices = {}
    for t, g in enumerate(graphs):
        for i, name in enumerate(g['patient']['name']):
            name_to_indices.setdefault(name, []).append((t, i))

    for name, occurrences in name_to_indices.items():
        for k in range(1, len(occurrences)):
            prev_t, prev_i = occurrences[k - 1]
            curr_t, curr_i = occurrences[k]
            edge = torch.tensor([[prev_i], [curr_i]], dtype=torch.long)
            if ('patient', 'follows', 'patient') in graphs[curr_t].edge_types:
                graphs[curr_t][('patient', 'follows', 'patient')].edge_index = torch.cat([
                    graphs[curr_t][('patient', 'follows', 'patient')].edge_index, edge
                ], dim=1)
            else:
                graphs[curr_t][('patient', 'follows', 'patient')].edge_index = edge

# Save graphs
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(split_graphs['train'], os.path.join(SAVE_DIR, 'train_graphs.pt'))
torch.save(split_graphs['val'], os.path.join(SAVE_DIR, 'val_graphs.pt'))
torch.save(split_graphs['test'], os.path.join(SAVE_DIR, 'test_graphs.pt'))

# --- TESTING ---
print("\nRunning checks...")
for split_name, gset in split_graphs.items():
    for idx, g in enumerate(gset):
        if 'patient' not in g.node_types:
            print(f"[{split_name}] Graph {idx} has no patient nodes.")
        if 'signature' not in g.node_types:
            print(f"[{split_name}] Graph {idx} has no signature nodes.")
        if any(v is None for v in g['patient'].values()):
            print(f"[{split_name}] Graph {idx} patient has None features.")
        if 'event' not in g['patient'] or 'duration' not in g['patient']:
            print(f"[{split_name}] Graph {idx} patient nodes missing event or duration labels.")
        for edge_type in g.edge_types:
            if 'edge_index' not in g[edge_type] or g[edge_type].edge_index.shape[0] != 2:
                print(f"[{split_name}] Graph {idx} has malformed edge {edge_type}.")
        if idx > 0 and ('patient', 'follows', 'patient') not in g.edge_types:
            print(f"[{split_name}] Graph {idx} is missing temporal patient-patient edges.")

print("All checks complete.")
