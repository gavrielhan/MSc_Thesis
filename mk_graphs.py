import numpy as np
import pandas as pd
from torch_geometric.data import Data
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

df_conditions_followup["Date"] = pd.to_datetime(df_conditions_followup["Date"]).dt.date
df_conditions_baseline["created_at"] = pd.to_datetime(df_conditions_baseline["created_at"]).dt.date

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
    """Align a short CGM signal with the existing 96D vectors and interpolate/extrapolate as needed."""
    if len(signal) == target_length:
        return signal

    distances = [fastdtw(signal, ref_signal)[0] for ref_signal in nmf_signatures]
    reference_signal = nmf_signatures[np.argmin(distances)]  # Best matched 96D vector

    x_existing = np.linspace(0, 1, len(signal))
    x_target = np.linspace(0, 1, target_length)
    interp_func = interp1d(x_existing, signal, kind='linear', fill_value='extrapolate')
    return interp_func(x_target)


# Graph Construction
print("Building graphs...")
all_graphs = []
for date, daily_data in df_cgm.groupby("Date"):
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
        patient_followup_dates = df_conditions_followup[df_conditions_followup["RegistrationCode"] == patient]["Date"]

        embedding = None
        if not patient_followup_dates.empty:
            past_dates = patient_followup_dates[patient_followup_dates <= date]
            if not past_dates.empty:
                closest_past_date = past_dates.max()
                embedding = followup_embeddings[followup_embeddings["RegistrationCode"] == patient].iloc[:, 1:].values

        if embedding is None and pd.notna(patient_baseline_date) and patient_baseline_date <= date:
            embedding = baseline_embeddings[baseline_embeddings["RegistrationCode"] == patient].iloc[:, 1:].values

        if embedding is None:
            print(f"?? Warning: No embedding found for patient {patient} (date: {date})")
            continue  # Skip patient if no valid embedding exists

        patient_nodes.append(torch.tensor(embedding.flatten(), dtype=torch.float))

    # Create edges
    for _, row in daily_data.iterrows():
        patient = row["RegistrationCode"]
        signal = row["GlucoseValue"]

        if len(signal) != 96:
            signal = align_and_rescale_signal(signal)

        weights = np.linalg.lstsq(nmf_signatures.T, signal, rcond=None)[0]
        weights = np.clip(weights, 0, 1)

        patient_idx = patient_to_idx.get(patient, None)
        if patient_idx is None:
            print(f"Skipping patient {patient} (no embedding found).")
            continue

        for sig_idx, weight in enumerate(weights):
            edge_index.append([patient_idx, num_signatures + sig_idx])
            edge_weights.append(weight)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        patient_nodes = torch.stack(patient_nodes)
        graph = Data(x=torch.cat((signature_nodes, patient_nodes)), edge_index=edge_index, edge_attr=edge_weights)
        all_graphs.append(graph)

print(f"? Graph building complete! {len(all_graphs)} graphs generated.")
