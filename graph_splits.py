import torch
from torch_geometric.data import HeteroData
import os
import random
from tqdm import tqdm

# CONFIG
GRAPHS_PATH = "glucose_sleep_graphs.pt"  # path to the list of daily graphs
SAVE_DIR = "/split"  # where to save train/val/test
SPLIT_RATIO = (0.7, 0.15, 0.15)
SEED = 42

# Load graphs
graphs = torch.load(GRAPHS_PATH, weights_only = False)
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
def get_event_duration(graph_idx, patient_name, diagnosis_dict):
    if patient_name not in diagnosis_dict:
        return 0, len(graphs) - graph_idx  # no event, censored at last observation
    diag_idx = diagnosis_dict[patient_name]
    if graph_idx > diag_idx:
        return None, None  # patient should not appear after diagnosis
    return int(graph_idx == diag_idx), diag_idx - graph_idx

# Build diagnosis index
patient_diagnosis_idx = {}
for idx, g in enumerate(graphs):
    if ('patient', 'has', 'condition') in g.edge_index_dict:
        edges = g.edge_index_dict[('patient', 'has', 'condition')]
        for src in edges[0].tolist():
            name = g['patient']['name'][src]
            if name not in patient_diagnosis_idx:
                patient_diagnosis_idx[name] = idx

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
        # Copy all node types
        for node_type in g.node_types:
            if node_type == 'patient':
                for k, v in g[node_type].items():
                    if isinstance(v, torch.Tensor) and v.size(0) == patient_mask.size(0):
                        new_g[node_type][k] = v[patient_mask]
                    elif isinstance(v, list) and len(v) == patient_mask.size(0):
                        new_g[node_type][k] = [val for val, keep in zip(v, patient_mask) if keep]
                    else:
                        new_g[node_type][k] = v  # e.g., scalars or non-maskable fields

                # Get filtered names from the masked version
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

        # Filter edges that involve patient nodes
        for (src, rel, dst), edge_index in g.edge_index_dict.items():
            if src == 'patient' and dst == 'glucose_signature':
                src_masked = patient_mask[edge_index[0]]
                edge_index_mask = src_masked
                new_g[(src, rel, dst)].edge_index = edge_index[:, edge_index_mask]
            elif src == 'patient' and dst == 'condition':
                src_masked = patient_mask[edge_index[0]]
                edge_index_mask = src_masked
                new_g[(src, rel, dst)].edge_index = edge_index[:, edge_index_mask]
            elif src == dst == 'glucose_signature':
                new_g[(src, rel, dst)].edge_index = edge_index  # keep all temporal edges

        split_graphs[split_name].append(new_g)

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
        if 'glucose_signature' not in g.node_types:
            print(f"[{split_name}] Graph {idx} has no glucose_signature nodes.")
        if any(v is None for v in g['patient'].values()):
            print(f"[{split_name}] Graph {idx} patient has None features.")
        if 'event' not in g['patient'] or 'duration' not in g['patient']:
            print(f"[{split_name}] Graph {idx} patient nodes missing event or duration labels.")
        for (src, _, dst), edge_index in g.edge_index_dict.items():
            if src not in g.node_types or dst not in g.node_types:
                print(f"[{split_name}] Graph {idx} has invalid edge types.")

print("All checks complete.")
