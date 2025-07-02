import os
import csv
import glob
from datetime import datetime
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

# Directory containing all patient subdirectories with OD/OS images
EYES_ROOT = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/"
MANIFEST_CSV = "retina_manifest.csv"

# 1. Index all retina image pairs and create a manifest CSV
def create_retina_manifest(eyes_root: str, manifest_csv: str):
    rows = []
    for subdir in sorted(os.listdir(eyes_root)):
        subdir_path = os.path.join(eyes_root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        # Expect subdir name: <patient_code>_<date>
        try:
            patient_code, date_str = subdir.split("_", 1)
        except ValueError:
            print(f"Skipping malformed directory: {subdir}")
            continue
        # Add '10K_' prefix and rename to RegistrationCode
        registration_code = f"10K_{patient_code}"
        # Normalize date to ISO format (YYYY-MM-DD) using pandas
        try:
            norm_date = pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d')
        except Exception:
            norm_date = date_str
        # Find OD and OS images
        od_path = glob.glob(os.path.join(subdir_path, "*OD*.jpg"))
        os_path = glob.glob(os.path.join(subdir_path, "*OS*.jpg"))
        if not od_path or not os_path:
            print(f"Missing OD/OS in {subdir}")
            continue
        # Use first match for each
        od_path = od_path[0]
        os_path = os_path[0]
        rows.append({
            "RegistrationCode": registration_code,
            "date": norm_date,
            "od_path": od_path,
            "os_path": os_path
        })
    # Write CSV
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["RegistrationCode", "date", "od_path", "os_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote manifest with {len(rows)} entries to {manifest_csv}")

# 2. PyTorch Dataset for retina images
class RetinaDataset(Dataset):
    def __init__(self, manifest_csv: str, transform=None):
        self.entries = []
        with open(manifest_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        entry = self.entries[idx]
        od_img = Image.open(entry["od_path"]).convert("RGB")
        os_img = Image.open(entry["os_path"]).convert("RGB")
        od_tensor = self.transform(od_img)
        os_tensor = self.transform(os_img)
        return {
            "od": od_tensor,
            "os": os_tensor,
            "RegistrationCode": entry["RegistrationCode"],
            "date": entry["date"]
        }

def create_diagnosis_csv(
    baseline_csv="/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv",
    followup_csv="/net/mraid20/export/genie/LabData/Data/10K/for_review/follow_up_conditions_all.csv",
    output_csv="retina_patient_diagnosis.csv"
):
    """
    Extracts and orders diagnosis info from baseline and follow-up CSVs.
    Output columns: RegistrationCode, baseline (yes/no), disease, date
    Each row: one disease per patient per date.
    """
    # Load data
    df_base = pd.read_csv(baseline_csv)
    df_follow = pd.read_csv(followup_csv)

    # Baseline: research_stage == 'baseline'
    base_rows = df_base[df_base['research_stage'] == 'baseline'][[
        'RegistrationCode', 'english_name', 'Date']].copy()
    base_rows['baseline'] = 'yes'
    base_rows = base_rows.rename(columns={'english_name': 'disease', 'Date': 'date'})

    # Follow-up: research_stage != 'baseline'
    follow_rows = df_follow[df_follow['research_stage'] != 'baseline'][[
        'RegistrationCode', 'english_name', 'Date']].copy()
    follow_rows['baseline'] = 'no'
    follow_rows = follow_rows.rename(columns={'english_name': 'disease', 'Date': 'date'})

    # Concat and sort
    all_rows = pd.concat([base_rows, follow_rows], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset=['RegistrationCode', 'disease', 'date'])
    all_rows = all_rows.sort_values(['RegistrationCode', 'baseline', 'disease', 'date'])
    # Normalize date to ISO format (YYYY-MM-DD) for all rows
    all_rows['date'] = pd.to_datetime(all_rows['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    # Write to CSV
    all_rows[['RegistrationCode', 'baseline', 'disease', 'date']].to_csv(output_csv, index=False)
    print(f"Wrote diagnosis info to {output_csv}")

def create_future_diagnosis_csv(
    manifest_csv="retina_manifest.csv",
    diagnosis_csv="retina_patient_diagnosis.csv",
    output_csv="retina_future_diagnosis.csv",
    diseases=None
):
    """
    For each patient and each disease, for each image, label as 1 if there is a diagnosis date for that disease after the image date, else 0.
    Overlap only by RegistrationCode, not by exact date.
    Output columns: RegistrationCode, date, od_path, os_path, disease, label, diagnosis_date (if any future diagnosis exists).
    """
    if diseases is None:
        diseases = [
            "Obesity",
            "Essential hypertension",
            "Diabetes mellitus, type unspecified"
        ]
    manifest = pd.read_csv(manifest_csv)
    diagnosis = pd.read_csv(diagnosis_csv)
    # Ensure date columns are datetime
    manifest['date'] = pd.to_datetime(manifest['date'], errors='coerce')
    diagnosis['date'] = pd.to_datetime(diagnosis['date'], errors='coerce')
    # For each patient and disease, get all diagnosis dates
    diag_grouped = diagnosis[diagnosis['disease'].isin(diseases)].groupby(['RegistrationCode','disease'])['date'].apply(list).reset_index()
    # Build a lookup: (RegistrationCode, disease) -> list of diagnosis_dates
    diag_lookup = {(row['RegistrationCode'], row['disease']): row['date'] for _, row in diag_grouped.iterrows()}
    # For each image, for each disease, assign label
    rows = []
    for _, img in manifest.iterrows():
        reg = img['RegistrationCode']
        img_date = img['date']
        for disease in diseases:
            diag_dates = diag_lookup.get((reg, disease), [])
            # Find any diagnosis date after the image date
            future_diag_dates = [d for d in diag_dates if not pd.isnull(d) and img_date < d]
            if pd.isnull(img_date):
                continue
            if future_diag_dates:
                label = 1
                diagnosis_date = min(future_diag_dates).strftime('%Y-%m-%d')
            else:
                label = 0
                diagnosis_date = ''
            rows.append({
                'RegistrationCode': reg,
                'date': img_date.strftime('%Y-%m-%d'),
                'od_path': img['od_path'],
                'os_path': img['os_path'],
                'disease': disease,
                'label': label,
                'diagnosis_date': diagnosis_date
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote future diagnosis prediction CSV to {output_csv} with {len(out_df)} rows.")

# 3. Main: generate manifest and test dataset
if __name__ == "__main__":
    # Step 1: Create manifest
    create_retina_manifest(EYES_ROOT, MANIFEST_CSV)
    # Step 2: Create diagnosis CSV
    create_diagnosis_csv()
    # Step 3: Create future diagnosis prediction CSV
    create_future_diagnosis_csv()
    # Step 4: Load dataset and show a batch
    dataset = RetinaDataset(MANIFEST_CSV)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("OD batch shape:", batch["od"].shape)
    print("OS batch shape:", batch["os"].shape)
    print("Patient codes:", batch["RegistrationCode"])
    print("Dates:", batch["date"]) 