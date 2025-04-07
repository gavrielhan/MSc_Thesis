import numpy as np
import pandas as pd
from torch_geometric.data import Data,HeteroData
import torch
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from LabData.DataLoaders.CGMLoader import CGMLoader

# Load data
print("Loading condition data...")
df_conditions_followup = pd.read_csv(
    '/net/mraid20/export/genie/LabData/Data/10K/for_review/follow_up_conditions_all.csv')
df_conditions_baseline = pd.read_csv(
    '/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv')

# Convert to timezone-naive UTC datetimes
df_conditions_baseline["created_at"] = pd.to_datetime(
    df_conditions_baseline["created_at"],
    format='mixed',  # Handle varied formats
    utc=True  # Parse timezone info and return aware objects
).dt.tz_convert(None)  # Convert to UTC and remove timezone

# For follow-up conditions
df_conditions_followup["Date"] = pd.to_datetime(
    df_conditions_followup["Date"],
    format='mixed',
    utc=True
).dt.tz_convert(None)
print("Loading CGM data...")
cgm = CGMLoader().get_data(study_ids=[10, 1001, 1002])
df_cgm = cgm.df.reset_index()
df_cgm['Date'] = pd.to_datetime(df_cgm['Date']).dt.date  # Convert to date format

print("Loading NMF signatures...")
nmf_signatures = np.load("nmf_cgm_signatures.npy")  # Precomputed NMF signatures
num_signatures = nmf_signatures.shape[0]

print("Loading patient embeddings...")
baseline_embeddings = pd.read_csv("filtered_patient_embeddings.csv")
followup_embeddings = pd.read_csv("followup_filtered_patient_embeddings.csv")

# Load sleep embeddings
print("Loading sleep embeddings...")
sleep_embeddings = pd.read_csv("filtered_sleep_embeddings.csv")
sleep_embeddings = sleep_embeddings.set_index('RegistrationCode').dropna()
# Before processing patients, calculate average sleep embedding
average_sleep_embedding = sleep_embeddings.mean(axis=0).values
# Unify multiple sleep embeddings by averaging them per patient
sleep_embeddings = sleep_embeddings.groupby('RegistrationCode').mean()


# Helper function: Align glucose signal to 96D
def align_and_rescale_signal(signal, target_length=96):
    if len(signal) == target_length:
        return signal
    signal = np.nan_to_num(signal, nan=0.0, posinf=1000, neginf=0.0)
    if len(signal) < 2:
        return np.zeros(target_length)
    x_existing = np.linspace(0, 1, len(signal))
    x_target = np.linspace(0, 1, target_length)
    interp_func = interp1d(x_existing, signal, kind='linear', bounds_error=False, fill_value=(signal[0], signal[-1]))
    return interp_func(x_target)

def compute_weights(signatures, signal):
    try:
        weights = np.linalg.lstsq(signatures.T, signal, rcond=None)[0]
        return np.clip(weights, 0, 1)
    except np.linalg.LinAlgError:
        return np.ones(signatures.shape[0]) / signatures.shape[0]

# Graph Construction
print("Building graphs...")
all_graphs = []
patients_in_cgm = df_cgm["RegistrationCode"].unique()
df_conditions_baseline = df_conditions_baseline[df_conditions_baseline["RegistrationCode"].isin(patients_in_cgm)]
patients_in_baseline = df_conditions_baseline["RegistrationCode"].unique()
df_conditions_followup = df_conditions_followup[df_conditions_followup["RegistrationCode"].isin(patients_in_baseline)]

grouped_cgm = df_cgm.groupby(["RegistrationCode", "Date"])["GlucoseValue"].apply(lambda x: x.values).reset_index()


# Define chosen medical conditions
chosen_medical_conditions = ["Hyperlipoproteinaemia ", "Essential hypertension", "Intermediate hyperglycaemia",
                             "Other specified conditions associated with the spine (intervertebral disc displacement)",
                             "Osteoporosis","Diabetes mellitus, type unspecified","Non-alcoholic fatty liver disease",
                             "Coronary atherosclerosis","Malignant neoplasms of breast"]  # Replace with actual conditions
condition_to_idx = {cond: i for i, cond in enumerate(chosen_medical_conditions)}
num_conditions = len(chosen_medical_conditions)

patient_last_index = {}
remaining_patients = set(patients_in_baseline)

#get the diagnosis dates for each patient and disease they have
diagnosis_dates = df_conditions_followup.groupby(["RegistrationCode", "Date"])["english_name"].apply(list).reset_index()

for date, daily_data in df_cgm.groupby('Date'):
    print(f"Processing date: {date}")
    signature_nodes = torch.tensor(nmf_signatures, dtype=torch.float)
    patient_nodes, patient_to_idx = [], {}
    condition_nodes = torch.eye(num_conditions, dtype=torch.float)
    edge_index, edge_weights, condition_edges, patient_temporal_edges = [], [], [], []

    # Identify patients that should still be in the graph
    valid_patients = remaining_patients.copy()
    for patient in list(valid_patients):
        if patient in diagnosis_dates and diagnosis_dates[patient] <= date:
            remaining_patients.discard(patient)

    for i, patient in enumerate(remaining_patients):
        patient_to_idx[patient] = len(patient_nodes)
        embedding = followup_embeddings[
            followup_embeddings["RegistrationCode"] == patient].copy()  # Make a copy of the slice
        embedding.loc[:, "Date"] = pd.to_datetime(embedding["Date"]).dt.date  # Convert datetime to date

        # Keep only embeddings with a valid past date
        valid_embeddings = embedding[embedding["Date"] <= date]

        if not valid_embeddings.empty:
            # Select the latest available embedding (closest date)
            latest_embedding = valid_embeddings.sort_values(by="Date").iloc[-1]  # Most recent row
            embedding = latest_embedding.drop(labels=["RegistrationCode", "Date"]).to_numpy()
        else:
            embedding = np.array([])

        if embedding is None or embedding.size == 0:
            baseline_embed = baseline_embeddings[baseline_embeddings["RegistrationCode"] == patient].iloc[:, 1:].values
            embedding = baseline_embed if baseline_embed.size > 0 else None

        if embedding is None or embedding.size == 0:
            continue

        sleep_embedding = sleep_embeddings.loc[
            patient].values if patient in sleep_embeddings.index else average_sleep_embedding
        embedding = embedding.flatten().astype(np.float32)
        sleep_embedding = sleep_embedding.astype(np.float32)
        patient_nodes.append(torch.tensor(np.append(embedding, sleep_embedding), dtype=torch.float))
        # Get glucose signal for this patient on this day
        glucose_row = grouped_cgm[(grouped_cgm["RegistrationCode"] == patient) &
                                  (grouped_cgm["Date"] == date)]
        if not glucose_row.empty:
            signal = glucose_row.iloc[0]["GlucoseValue"]
            signal = align_and_rescale_signal(signal)

            # Compute weights using NMF signatures
            weights = compute_weights(nmf_signatures, signal)

            for sig_idx, w in enumerate(weights):
                if w > 0.01:  # Optional: filter very weak connections
                    edge_index.append([patient_to_idx[patient], sig_idx])
                    edge_weights.append(w)
        # Temporal edges between consecutive appearances
        if patient in patient_last_index:
            patient_temporal_edges.append([patient_last_index[patient], patient_to_idx[patient]])
        patient_last_index[patient] = patient_to_idx[patient]

    hetero_graph = HeteroData()
    hetero_graph['signature'].x = signature_nodes
    hetero_graph['patient'].x = torch.stack(patient_nodes)
    hetero_graph['condition'].x = condition_nodes

    if edge_index:

        hetero_graph['patient', 'to', 'signature'].edge_index = torch.tensor(edge_index, dtype=torch.long).T
        hetero_graph['patient', 'to', 'signature'].edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    if condition_edges:

        hetero_graph['patient', 'has', 'condition'].edge_index = torch.tensor(condition_edges, dtype=torch.long).T

    if patient_temporal_edges:

        hetero_graph['patient', 'temporal', 'patient'].edge_index = torch.tensor(patient_temporal_edges,
                                                                                 dtype=torch.long).T

    all_graphs.append(hetero_graph)
torch.save(all_graphs, "glucose_sleep_graphs.pt")
print(f"Graphs saved successfully to 'glucose_sleep_graphs.pt'.")
print(f"Graph building complete! {len(all_graphs)} graphs generated.")
loaded_graphs = torch.load("glucose_sleep_graphs.pt", weights_only=False)
if loaded_graphs:
    print('Graphs are loadable')
