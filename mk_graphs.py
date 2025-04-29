import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from scipy.interpolate import interp1d
from LabData.DataLoaders.CGMLoader import CGMLoader
from multiprocessing import Pool
from functools import partial
import os
import threading
import time

# ========== Timer function ==========
def start_pulse_timer(message="running", delay=1800, interval=1800):
    stop_event = threading.Event()
    def pulse():
        time.sleep(delay)
        while not stop_event.is_set():
            print(message)
            time.sleep(interval)
    thread = threading.Thread(target=pulse)
    thread.daemon = True
    thread.start()
    return stop_event

# ========== Load data ==========
print("Loading data...")
df_conditions_followup = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/follow_up_conditions_all.csv")
df_conditions_baseline = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv")

df_conditions_baseline["created_at"] = pd.to_datetime(df_conditions_baseline["created_at"], format='mixed', utc=True).dt.tz_convert(None)
df_conditions_followup["Date"] = pd.to_datetime(df_conditions_followup["Date"], format='mixed', utc=True).dt.tz_convert(None)

cgm = CGMLoader().get_data(study_ids=[10, 1001, 1002])
df_cgm = cgm.df.reset_index()
df_cgm['Date'] = pd.to_datetime(df_cgm['Date']).dt.date

nmf_signatures = np.load("nmf_cgm_signatures.npy")
num_signatures = nmf_signatures.shape[0]

baseline_embeddings = pd.read_csv("filtered_patient_embeddings.csv")
followup_embeddings = pd.read_csv("followup_filtered_patient_embeddings.csv")
sleep_embeddings = pd.read_csv("filtered_sleep_embeddings.csv").set_index('RegistrationCode').groupby('RegistrationCode').mean()
average_sleep_embedding = sleep_embeddings.mean(axis=0).values

patients_in_cgm = df_cgm["RegistrationCode"].unique()
baseline_patients = baseline_embeddings["RegistrationCode"].unique()
valid_patients = np.intersect1d(patients_in_cgm, baseline_patients)

# Filter
df_conditions_baseline = df_conditions_baseline[df_conditions_baseline["RegistrationCode"].isin(valid_patients)]
df_cgm = df_cgm[df_cgm["RegistrationCode"].isin(valid_patients)].copy()
baseline_embeddings = baseline_embeddings[baseline_embeddings["RegistrationCode"].isin(valid_patients)].copy()
followup_embeddings = followup_embeddings[followup_embeddings["RegistrationCode"].isin(valid_patients)].copy()
sleep_embeddings = sleep_embeddings[sleep_embeddings.index.isin(valid_patients)].copy()
df_conditions_followup = df_conditions_followup[
    df_conditions_followup['RegistrationCode'].isin(valid_patients)
]
grouped_cgm = df_cgm.groupby(["RegistrationCode", "Date"])["GlucoseValue"].apply(lambda x: x.values)
grouped_cgm_dict = grouped_cgm.to_dict()

followup_embeddings["Date"] = pd.to_datetime(followup_embeddings["Date"]).dt.date
followup_dict = dict(tuple(followup_embeddings.groupby("RegistrationCode")))
baseline_embeddings_indexed = baseline_embeddings.set_index("RegistrationCode")

chosen_conditions = [
    "Hyperlipoproteinaemia ", "Essential hypertension", "Intermediate hyperglycaemia",
    "Other specified conditions associated with the spine (intervertebral disc displacement)",
    "Osteoporosis", "Diabetes mellitus, type unspecified", "Non-alcoholic fatty liver disease",
    "Coronary atherosclerosis", "Malignant neoplasms of breast"
]
condition_to_idx = {cond: i for i, cond in enumerate(chosen_conditions)}
num_conditions = len(chosen_conditions)

all_patients = sorted(set(valid_patients))
pname_to_index = {p: i for i, p in enumerate(all_patients)}

# ========== Helpers ==========
def align_and_rescale(signal, target_len=96):
    signal = np.nan_to_num(signal, nan=0.0, posinf=1000, neginf=0.0)
    if len(signal) < 2:
        return np.zeros(target_len)
    interp = interp1d(np.linspace(0, 1, len(signal)), signal, bounds_error=False, fill_value=(signal[0], signal[-1]))
    return interp(np.linspace(0, 1, target_len))

def compute_weights(signatures, signal):
    try:
        weights = np.linalg.lstsq(signatures.T, signal, rcond=None)[0]
        return np.clip(weights, 0, 1)
    except:
        return np.ones(signatures.shape[0]) / signatures.shape[0]

def find_closest_window(diag_date, window_centers):
    min_diff = float('inf')
    closest_idx = -1
    for idx, center in enumerate(window_centers):
        diff = abs((center - diag_date).days)
        if diff < min_diff:
            min_diff = diff
            closest_idx = idx
    return closest_idx

def get_patient_embedding(patient, date):
    emb = followup_dict.get(patient)
    if emb is not None and "Date" in emb.columns:
        valid = emb[emb["Date"] <= date]
    else:
        valid = pd.DataFrame()
    if not valid.empty:
        embedding = valid.sort_values("Date").iloc[-1].filter(like='emb_').values
    else:
        if patient not in baseline_embeddings_indexed.index:
            raise ValueError(f"No baseline embedding for patient {patient}.")
        embedding = baseline_embeddings_indexed.loc[patient].filter(like='emb_').values
    sleep = sleep_embeddings.loc[patient].values if patient in sleep_embeddings.index else average_sleep_embedding
    return np.concatenate([embedding, sleep]).astype(np.float32)

# ========== Window processing function ==========
def build_graph(w_idx, w_start, w_end, patients_in_window, grouped_cgm_dict,
                patient_diagnosis_to_graph, last_patient_locations):
    print(f"Building window {w_idx} from {w_start} to {w_end}")

    hetero = HeteroData()
    hetero['signature'].x = torch.tensor(nmf_signatures, dtype=torch.float)
    hetero['condition'].x = torch.eye(num_conditions, dtype=torch.float)

    patient_feats, patient_names = [], []
    edge_index, edge_weights, condition_edges = [], [], []

    # Step 1: build patients
    for patient in patients_in_window:
        baseline_or_followup_embedding = get_patient_embedding(patient, w_end)
        baseline_or_followup_embedding = torch.from_numpy(baseline_or_followup_embedding).float()
        patient_feats.append(baseline_or_followup_embedding)
        patient_names.append(patient)

    if patient_feats:
        hetero['patient'].x = torch.stack(patient_feats)
        hetero['patient'].name = patient_names
    else:
        hetero['patient'].x = torch.empty((0, 138))
        hetero['patient'].name = []

    hetero['signature'].name = [f"sig_{i}" for i in range(num_signatures)]
    hetero['condition'].name = list(condition_to_idx.keys())

    # ====== Local patient index map ======
    local_pname_to_idx = {name: idx for idx, name in enumerate(patient_names)}

    # Step 2: build edges (patient ? signature)
    for patient in patients_in_window:
        if patient not in local_pname_to_idx:
            continue  # Safety

        local_idx = local_pname_to_idx[patient]

        for day in pd.date_range(w_start, w_end).date:
            signal = grouped_cgm_dict.get((patient, day))
            if signal is not None:
                signal = align_and_rescale(signal)
                weights = compute_weights(nmf_signatures, signal)
                for sig_idx, w in enumerate(weights):
                    if w > 0.01:
                        edge_index.append([local_idx, sig_idx])
                        edge_weights.append(w)

    if edge_index:
        hetero['patient', 'to', 'signature'].edge_index = torch.tensor(edge_index, dtype=torch.long).T
        hetero['patient', 'to', 'signature'].edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # ====== Step 3: Add patient ? condition edges based on closest diagnosis
    for patient, cond_idx, diag_date in patient_diagnosis_to_graph.get(w_idx, []):
        if patient in local_pname_to_idx:
            local_idx = local_pname_to_idx[patient]
            condition_edges.append([local_idx, cond_idx])

    if condition_edges:
        hetero['patient', 'has', 'condition'].edge_index = torch.tensor(condition_edges, dtype=torch.long).T

    # ====== Step 4: Temporal patient self-follow edges
    temporal_src = []
    temporal_dst = []

    for new_idx, patient in enumerate(patient_names):
        if patient in last_patient_locations:
            prev_graph_idx, prev_node_idx = last_patient_locations[patient]
            temporal_src.append(prev_node_idx)
            temporal_dst.append(new_idx)

        last_patient_locations[patient] = (w_idx, new_idx)

    if temporal_src:
        hetero['patient', 'follows', 'patient'].edge_index = torch.tensor(
            [temporal_src, temporal_dst],
            dtype=torch.long
        )

    # ====== DEBUG PRINTS ======
    if w_idx % (len(window_centers) // 20 + 1) == 0:  # Every ~5% of graphs
        print(f"[Graph {w_idx}] {hetero['patient'].x.size(0)} patients, "
              f"{hetero['signature'].x.size(0)} signatures, "
              f"{hetero['condition'].x.size(0)} conditions")

        if ('patient', 'has', 'condition') in hetero.edge_types:
            num_edges = hetero['patient', 'has', 'condition'].edge_index.size(1)
            print(f"    'patient' ? 'condition' edges: {num_edges}")
        else:
            print(f"    No 'patient' ? 'condition' edges.")

        if ('patient', 'to', 'signature') in hetero.edge_types:
            num_edges = hetero['patient', 'to', 'signature'].edge_index.size(1)
            print(f"    'patient' ? 'signature' edges: {num_edges}")
        else:
            print(f"    No 'patient' ? 'signature' edges.")

        if ('patient', 'follows', 'patient') in hetero.edge_types:
            num_edges = hetero['patient', 'follows', 'patient'].edge_index.size(1)
            print(f"    'patient' ? 'patient' (follows) edges: {num_edges}")
        else:
            print(f"    No 'patient' ? 'patient' (follows) edges.")

    return (w_idx, hetero)

def count_events(valid_patients, df_cgm, df_conditions_followup, chosen_conditions):
    """
    Count diagnosis events:
    - Only for valid patients
    - Only for chosen conditions
    - Only within the CGM date range
    - Keep multiple conditions per patient if they occur on the same day
    - But count unique (patient, date) pairs once if all events are duplicates
    """
    df = df_conditions_followup.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', utc=True).dt.tz_convert(None)

    # Filter by valid patients and conditions
    df = df[df['RegistrationCode'].isin(valid_patients)]
    df = df[df['english_name'].isin(chosen_conditions)]

    # Filter by CGM date range
    min_cgm_date = pd.to_datetime(df_cgm['Date'].min())
    max_cgm_date = pd.to_datetime(df_cgm['Date'].max())
    df = df[(df['Date'] >= min_cgm_date) & (df['Date'] <= max_cgm_date)]

    # Group by patient and date, keep all diagnoses on same day
    df_unique = df.drop_duplicates(subset=['RegistrationCode', 'Date'])

    correct_total_events = len(df_unique)
    print(f"? Correct number of diagnosis events (filtered): {correct_total_events}")

    return correct_total_events
correct_total_events = count_events(valid_patients, df_cgm, df_conditions_followup, chosen_conditions)
# ========== Main ==========
print("Building graphs...")
pulse = start_pulse_timer("Still building graphs...", delay=1800)

all_dates = sorted(df_cgm['Date'].unique())
window_size = 3
step_size = 3

# Precompute window start and end dates
window_ranges = []
window_centers = []
for w_start_idx in range(0, len(all_dates) - window_size + 1, step_size):
    w_start = all_dates[w_start_idx]
    w_end = all_dates[w_start_idx + window_size - 1]
    center = w_start + (w_end - w_start) / 2
    window_centers.append(center)
    window_ranges.append((w_start, w_end))

# Precompute diagnosis assignment
patient_diagnosis_to_graph = {}  # graph_idx -> list of (patient, condition_idx, diag_date)
for idx, row in df_conditions_followup.iterrows():
    if row['english_name'] in condition_to_idx and row['RegistrationCode'] in valid_patients:
        patient = row['RegistrationCode']
        cond_idx = condition_to_idx[row['english_name']]
        diag_date = row['Date'].date()

        # Assign to a graph whose window contains the diagnosis
        assigned = False
        for graph_idx, (w_start, w_end) in enumerate(window_ranges):
            if w_start <= diag_date <= w_end:
                patient_diagnosis_to_graph.setdefault(graph_idx, []).append((patient, cond_idx, diag_date))
                assigned = True
                break
        # If no exact match, assign to the closest past window (latest one before diagnosis)
        if not assigned:
            past_windows = [(idx, center) for idx, center in enumerate(window_centers) if center <= diag_date]
            if past_windows:
                closest_graph = max(past_windows, key=lambda x: (x[1]))[0]
            else:
                # If no past windows exist, fallback to first graph
                closest_graph = 0
            patient_diagnosis_to_graph.setdefault(closest_graph, []).append((patient, cond_idx, diag_date))

# Initialize last_patient_locations
last_patient_locations = {}

patient_diag_dates = {}
for idx, row in df_conditions_followup.iterrows():
    if row['english_name'] in chosen_conditions:
        patient_diag_dates.setdefault(row['RegistrationCode'], []).append(row['Date'].date())

remaining_patients = set(all_patients)
jobs = []
for w_idx, (w_start, w_end) in enumerate(window_ranges):
    # 1) Who is diagnosed in this window?
    diag_now = {
        patient
        for patient, _, diag_date in patient_diagnosis_to_graph.get(w_idx, [])
        if w_start <= diag_date <= w_end
    }

    # 2) Undiagnosed yet
    undiagnosed = remaining_patients - diag_now

    # 3) Our window?s patient set:
    patients_in_window = list(undiagnosed | diag_now)

    # 4) Schedule this job
    jobs.append((
        w_idx, w_start, w_end,
        patients_in_window,
        grouped_cgm_dict,
        patient_diagnosis_to_graph,
        last_patient_locations
    ))

    # 5) Remove exactly those just?diagnosed from future windows
    remaining_patients -= diag_now
# Start pulse while building graphs
pulse = start_pulse_timer(message="Still building graphs...", delay=1800, interval=1800)

# Build graphs (sequential for simplicity)
results = [build_graph(*job) for job in jobs]
# Stop pulse after graphs are ready
pulse.set()

# Sort and save
all_graphs = [g for idx, g in sorted(results, key=lambda x: x[0])]
# Sum up the patient?condition edges actually created
total_edges_created = sum(
    g['patient', 'has', 'condition'].edge_index.size(1)
    if ('patient', 'has', 'condition') in g.edge_types else 0
    for g in all_graphs
)

print(f"Total patient?condition edges created: {total_edges_created}")

if total_edges_created == correct_total_events:
    print(" All events correctly captured. No event loss.")
else:
    print(f"Warning: {abs(correct_total_events - total_edges_created)} events missing!")
print(f"Generated {len(all_graphs)} Graphs")
torch.save(all_graphs, "glucose_sleep_graphs_3d.pt")
print(f"Saved {len(all_graphs)} graphs to glucose_sleep_graphs_3d.pt")
