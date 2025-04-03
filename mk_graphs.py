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
        reg = 1e-6 * np.eye(signatures.shape[1])
        weights = np.linalg.lstsq(signatures.T @ signatures + reg, signatures.T @ signal, rcond=None)[0]
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

for date, daily_data in df_cgm.groupby('Date'):
    print(f"Processing date: {date} ({len(daily_data)} measurements)")

    patients_in_day = daily_data["RegistrationCode"].unique()
    signature_nodes = torch.tensor(nmf_signatures, dtype=torch.float)

    patient_nodes, patient_to_idx = [], {}
    edge_index, edge_weights = [], []

    for i, patient in enumerate(patients_in_day):
        print(f"Processing patient {patient}...")
        patient_to_idx[patient] = len(patient_nodes)
        embedding = None

        # Get follow-up dates directly from embeddings (not conditions)
        patient_embeddings = followup_embeddings[followup_embeddings["RegistrationCode"] == patient]

        if not patient_embeddings.empty:
            # Convert all dates to datetime.date for comparison
            embedding_dates = pd.to_datetime(patient_embeddings["Date"]).dt.date
            current_date = pd.to_datetime(date).date()  # Ensure comparison as date objects

            # Find closest date <= current date
            valid_dates = embedding_dates[embedding_dates <= current_date]
            if not valid_dates.empty:
                closest_date = valid_dates.max()
                embedding = patient_embeddings.loc[embedding_dates == closest_date].drop(columns=['RegistrationCode','Date']).values


        # Fallback to baseline if no valid follow-up
        if embedding is None or embedding.size == 0:
            baseline_embed = baseline_embeddings[
                                 baseline_embeddings["RegistrationCode"] == patient
                                 ].iloc[:, 1:].values
            embedding = baseline_embed if baseline_embed.size > 0 else None

        print(f"Checking patient {patient} embeddings...")

        if embedding is None or embedding.size == 0:
            print(f"?? Warning: No embedding found for patient {patient} (date: {date})")
            continue  # Skip patient if no valid embedding exists

        sleep_embedding = sleep_embeddings.loc[patient].values if patient in sleep_embeddings.index else average_sleep_embedding
        embedding = embedding.flatten().astype(np.float32)  # Ensure float32
        sleep_embedding = sleep_embedding.astype(np.float32)
        patient_nodes.append(torch.tensor(np.append(embedding,sleep_embedding), dtype=torch.float))

    # Create edges
    for patient in patients_in_day:
        patient_signal = grouped_cgm[
            (grouped_cgm["RegistrationCode"] == patient) & (grouped_cgm["Date"] == date)
            ]["GlucoseValue"].values

        if len(patient_signal) == 0:
            print(f"?? Warning: No glucose signal found for patient {patient} on {date}")
            continue  # Skip if no data

        signal = patient_signal[0]  # Extract the array

        if len(signal) != 96:
            signal = align_and_rescale_signal(signal)

        try:
            weights = compute_weights(nmf_signatures, signal)
            if not np.isfinite(weights).all():
                weights = np.ones(num_signatures) / num_signatures
        except:
            weights = np.ones(num_signatures) / num_signatures

        patient_idx = patient_to_idx.get(patient, None)
        if patient_idx is None:
            print(f"Skipping patient {patient} (no embedding found).")
            continue

        for sig_idx, weight in enumerate(weights):
            edge_index.append([patient_idx, num_signatures + sig_idx])
            edge_weights.append(weight)

    if not patient_nodes:  # Check if patient_nodes is empty
        print(f"!! Skipping date {date} because no valid patient embeddings were found.")
        continue  # Skip this date and move to the next one
        # Create HeteroData graph
    hetero_graph = HeteroData()

    # 1. Add signature nodes (type: 'signature')
    hetero_graph['signature'].x = signature_nodes  # Features: 96D

    # 2. Add patient nodes (type: 'patient')
    patient_nodes = torch.stack(patient_nodes)  # Shape: [num_patients, 131]
    hetero_graph['patient'].x = patient_nodes

    # 3. Add edges (patient -> signature)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        # Key change: Specify edge type ('patient', 'to', 'signature')
        hetero_graph['patient', 'to', 'signature'].edge_index = edge_index
        hetero_graph['patient', 'to', 'signature'].edge_attr = edge_weights
    # 4. Add temporal edge
        if len(all_graphs) > 0:  # Ensure there is a previous graph
            prev_graph = all_graphs[-1]

            if 'signature' in prev_graph and 'signature' in hetero_graph:
                num_signatures = signature_nodes.shape[0]
                temporal_edges = torch.tensor(
                    [[i, i] for i in range(num_signatures)], dtype=torch.long
                ).T  # Creates edges (i at t ? i at t-1)

                hetero_graph['signature', 'temporal', 'signature'].edge_index = temporal_edges
        all_graphs.append(hetero_graph)

torch.save(all_graphs, "glucose_sleep_graphs.pt")
print(f"Graphs saved successfully to 'glucose_sleep_graphs.pt'.")
print(f"Graph building complete! {len(all_graphs)} graphs generated.")
loaded_graphs = torch.load("glucose_sleep_graphs.pt", weights_only=False)
if loaded_graphs:
    print('Graphs are loadable')
