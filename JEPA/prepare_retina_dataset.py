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
            "date": date_str,
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

    # Write to CSV
    all_rows[['RegistrationCode', 'baseline', 'disease', 'date']].to_csv(output_csv, index=False)
    print(f"Wrote diagnosis info to {output_csv}")

# 3. Main: generate manifest and test dataset
if __name__ == "__main__":
    # Step 1: Create manifest
    create_retina_manifest(EYES_ROOT, MANIFEST_CSV)
    # Step 2: Create diagnosis CSV
    create_diagnosis_csv()
    # Step 3: Load dataset and show a batch
    dataset = RetinaDataset(MANIFEST_CSV)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("OD batch shape:", batch["od"].shape)
    print("OS batch shape:", batch["os"].shape)
    print("Patient codes:", batch["RegistrationCode"])
    print("Dates:", batch["date"]) 