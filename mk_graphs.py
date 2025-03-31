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
followup_embeddings = pd.read_csv("followup_filtered_patient_embeddings.csv").drop(columns=["research_stage"])


# Helper function: Align glucose signal to 96D
def align_and_rescale_signal(signal, target_length=96):
    """Align signal to target length with safe interpolation"""
    if len(signal) == target_length:
        return signal

    # Ensure signal has valid values
    signal = np.nan_to_num(signal, nan=0.0, posinf=1000, neginf=0.0)

    # Handle edge cases
    if len(signal) < 2:
        return np.zeros(target_length)

    x_existing = np.linspace(0, 1, len(signal))
    x_target = np.linspace(0, 1, target_length)

    with np.errstate(divide='ignore', invalid='ignore'):
        interp_func = interp1d(
            x_existing,
            signal,
            kind='linear',
            bounds_error=False,
            fill_value=(signal[0], signal[-1])  # Extend first/last values
        )
    return interp_func(x_target)
def compute_weights(signatures, signal):
    """Robust weight computation with regularization"""
    try:
        # Add small regularization (ridge regression)
        reg = 1e-6 * np.eye(signatures.shape[1])
        weights = np.linalg.lstsq(
            signatures.T @ signatures + reg,
            signatures.T @ signal,
            rcond=None
        )[0]
        return np.clip(weights, 0, 1)
    except LinAlgError:
        # Fallback to uniform weights if unstable
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

        # Get closest past embedding
        patient_baseline_date = df_conditions_baseline[df_conditions_baseline["RegistrationCode"] == patient][
            "created_at"].min()
        patient_followup_dates = df_conditions_followup[df_conditions_followup["RegistrationCode"] == patient]["Date"].dt.tz_localize(None)
        patient_followup_dates = pd.Series([dt.date() for dt in patient_followup_dates])  # Convert to Pandas Series
        embedding = None
        if patient_followup_dates.empty:
            past_dates = patient_followup_dates[patient_followup_dates <= date]
            if not past_dates.empty:
                closest_past_date = past_dates.max()
                embedding = followup_embeddings[followup_embeddings["RegistrationCode"] == patient].iloc[:, 1:].values
                if not followup_embeddings.empty:
                    embedding = followup_embeddings.iloc[:, 1:].values.flatten()
        if embedding is None:
            embedding = baseline_embeddings[baseline_embeddings["RegistrationCode"] == patient].iloc[:, 1:].values

        print(f"Checking patient {patient} embeddings...")
        print(f"Baseline date: {patient_baseline_date}")
        print(f"Follow-up dates: {patient_followup_dates}")

        if embedding is None or embedding.size == 0:
            print(f"?? Warning: No embedding found for patient {patient} (date: {date})")
            continue  # Skip patient if no valid embedding exists

        patient_nodes.append(torch.tensor(embedding.flatten(), dtype=torch.float))

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

        all_graphs.append(hetero_graph)

print(f"? Graph building complete! {len(all_graphs)} graphs generated.")
