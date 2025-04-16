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

def start_pulse_timer(message="running", delay=1800, interval=60):
    """Start a pulse print after `delay` seconds and repeat every `interval` seconds."""
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

def save_graph(date, grouped_cgm_dict, patient_list, followup_dict):
    graph = build_graph_for_date(date, grouped_cgm_dict, patient_list, followup_dict)
    path = os.path.join("daily_graphs", f"{date}.pt")
    torch.save(graph, path)
    return path

# Load data
print("Loading data...")
df_conditions_followup = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/follow_up_conditions_all.csv")
df_conditions_baseline = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv")

# Convert to timezone-naive UTC datetimes
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

# Filter all data
df_conditions_baseline = df_conditions_baseline[df_conditions_baseline["RegistrationCode"].isin(valid_patients)]
df_cgm = df_cgm[df_cgm["RegistrationCode"].isin(valid_patients)].copy()
baseline_embeddings = baseline_embeddings[baseline_embeddings["RegistrationCode"].isin(valid_patients)].copy()
followup_embeddings = followup_embeddings[followup_embeddings["RegistrationCode"].isin(valid_patients)].copy()
sleep_embeddings = sleep_embeddings[sleep_embeddings.index.isin(valid_patients)].copy()

# Group CGM
grouped_cgm = df_cgm.groupby(["RegistrationCode", "Date"])["GlucoseValue"].apply(lambda x: x.values)
grouped_cgm_dict = grouped_cgm.to_dict()

# Index followups
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

def build_graph_for_date(date, grouped_cgm_dict, patient_list, followup_dict):
    print(f"Processing date: {date}")
    pulse_stop = start_pulse_timer(f"Still processing date: {date}", delay=1800)
    try:
        signature_nodes = torch.tensor(nmf_signatures, dtype=torch.float)
        condition_nodes = torch.eye(num_conditions, dtype=torch.float)
        patient_nodes, patient_names = [], []
        edge_index, edge_weights, condition_edges = [], [], []

        for patient in patient_list:
            patient_names.append(patient)

        # --- Get embedding ---
            emb = followup_dict.get(patient)
            if emb is not None and "Date" in emb.columns:
                valid = emb[emb["Date"] <= date]
            else:
                valid = pd.DataFrame()

            if not valid.empty:
                embedding = valid.sort_values("Date").iloc[-1].filter(like='emb_').values
            else:
                if patient not in baseline_embeddings_indexed.index:
                    raise ValueError(f"No baseline embedding found for patient {patient}.")
                embedding = baseline_embeddings_indexed.loc[patient].filter(like='emb_').values

            sleep = sleep_embeddings.loc[patient].values if patient in sleep_embeddings.index else average_sleep_embedding

            if embedding.shape[0] != 128:
                raise ValueError(f"Follow-up or baseline embedding for patient {patient} on date {date} is not 128D but {embedding.shape[0]}D.")
            if sleep.shape[0] != 10:
                raise ValueError(f"Sleep embedding for patient {patient} is not 10D but {sleep.shape[0]}D.")

            final_embedding = np.concatenate([embedding, sleep]).astype(np.float32)
            if final_embedding.shape[0] != 138:
                raise ValueError(f"Final embedding for patient {patient} is not 138D but {final_embedding.shape[0]}D.")
            patient_nodes.append(torch.tensor(final_embedding, dtype=torch.float))

        # --- Get glucose signal ---
            signal = grouped_cgm_dict.get((patient, date))
            if signal is not None:
                signal = align_and_rescale(signal)
                weights = compute_weights(nmf_signatures, signal)
                for sig_idx, w in enumerate(weights):
                    if w > 0.01:
                        edge_index.append([pname_to_index[patient], sig_idx])
                        edge_weights.append(w)

        # --- Diagnosis ---
            diagnoses = df_conditions_followup[(df_conditions_followup["RegistrationCode"] == patient) & (df_conditions_followup["Date"] == pd.Timestamp(date))]
            for cond in diagnoses["english_name"]:
                if cond in condition_to_idx:
                    condition_edges.append([pname_to_index[patient], condition_to_idx[cond]])

        hetero = HeteroData()
        hetero['signature'].x = signature_nodes
        hetero['condition'].x = condition_nodes
        hetero['patient'].x = torch.stack(patient_nodes)
        hetero['patient'].name = patient_names
        hetero['signature'].name = [f"sig_{i}" for i in range(num_signatures)]
        hetero['condition'].name = list(condition_to_idx.keys())

        if edge_index:
            hetero['patient', 'to', 'signature'].edge_index = torch.tensor(edge_index, dtype=torch.long).T
            hetero['patient', 'to', 'signature'].edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        if condition_edges:
            hetero['patient', 'has', 'condition'].edge_index = torch.tensor(condition_edges, dtype=torch.long).T

        return hetero
    finally:
        pulse_stop.set()


# Main
all_dates = sorted(df_cgm['Date'].unique())
os.makedirs("daily_graphs", exist_ok=True)
with Pool() as pool:
    save_fn = partial(save_graph, grouped_cgm_dict=grouped_cgm_dict, patient_list=all_patients,
                      followup_dict=followup_dict)
    saved_paths = list(pool.map(save_fn, all_dates))

print("Adding temporal edges...")
name_to_day_idx = {}
all_graphs = [torch.load(p, weights_only=False) for p in saved_paths]
for d, g in enumerate(all_graphs):
    for i, name in enumerate(g['patient'].name):
        name_to_day_idx.setdefault(name, []).append((d, i))

for name, appearances in name_to_day_idx.items():
    appearances = sorted(appearances)
    for k in range(1, len(appearances)):
        prev_day, prev_idx = appearances[k-1]
        curr_day, curr_idx = appearances[k]
        edge = torch.tensor([[prev_idx], [curr_idx]], dtype=torch.long)
        g = all_graphs[curr_day]
        if ('patient', 'follows', 'patient') in g:
            g['patient', 'follows', 'patient'].edge_index = torch.cat([g['patient', 'follows', 'patient'].edge_index, edge], dim=1)
        else:
            g['patient', 'follows', 'patient'].edge_index = edge

print("Saving graphs...")
for p in saved_paths:
    os.remove(p)
torch.save(all_graphs, "glucose_sleep_graphs.pt")
print(f"Saved {len(all_graphs)} graphs with temporal edges.")
