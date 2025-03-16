import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.MedicalConditionLoader import MedicalConditionLoader
from LabData.DataLoaders.MedicalProceduresLoader import MedicalProceduresLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.DemographicsLoader import DemographicsLoader
from LabData.DataLoaders.ECGTextLoader import ECGTextLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.IBSTenkLoader import IBSTenkLoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.FamilyMedicalConditionsLoader import FamilyMedicalConditionsLoader
from LabData.DataLoaders.ChildrenLoader import ChildrenLoader
from LabData.DataLoaders.MentalLoader import MentalLoader
from LabData.DataLoaders.TimelineLoader import TimelineLoader
from LabData.DataLoaders.SubjectRelationsLoader import SubjectRelationsLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.PAStepsLoader import PAStepsLoader

# get general medical info fo reach patient
study_ids = [10, 1001, 1002]
bm = BodyMeasuresLoader().get_data(study_ids=study_ids).df.join(BodyMeasuresLoader().get_data(study_ids=study_ids).df_metadata)
general_info = bm.reset_index()
general_info = general_info[~general_info['gender'].isna()]
general_info = general_info[~general_info['age'].isna()]
gender_dictionary = {1:'male', 0:'female'}
general_info.loc[:,'gender'] = general_info['gender'].fillna(0).map(gender_dictionary)

# get the patient embeddings obtained with ClinicalBERT
embeddings_df_baseline = pd.read_csv("patient_embeddings.csv")
# get the baseline conditions of each patient, to then use them to test clustering power
path_file = '/net/mraid20/export/genie/LabData/Data/10K/for_review/'
file_name = 'baseline_conditions_all.csv'
full_path = path_file + file_name

# Read the CSV file into a DataFrame
df_conditions_baseline = pd.read_csv(full_path)


# Group baseline conditions by RegistrationCode and aggregate the english_name values.
baseline_conditions = df_conditions_baseline.groupby("RegistrationCode")["english_name"] \
                    .apply(lambda x: "; ".join(x.astype(str).unique())) \
                    .reset_index()
baseline_conditions.rename(columns={"english_name": "baseline_conditions"}, inplace=True)

# Merge the aggregated baseline conditions with general_info.
merged_df = pd.merge(general_info, baseline_conditions, on="RegistrationCode", how="left")

# ----- Define the Condition to Color By -----
# For example, choose a condition from baseline_conditions. You can modify this as needed.
chosen_condition = "Diabetes mellitus, type unspecified"

# Create a new boolean column: True if the chosen condition is found in the baseline_conditions string.
merged_df["has_condition"] = merged_df["baseline_conditions"].fillna("").apply(lambda x: chosen_condition in x)
# (Ensure there are no missing values in the features you choose.)
features_df = merged_df[["age", "gender_numeric"]].dropna()

# Keep the indices so that we can align the condition labels later.
umap_features = features_df.values

# ----- Run UMAP -----
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(umap_features)

# ----- Plot UMAP Results with Color Based on Condition Presence -----
# Align the "has_condition" column with the indices of features_df.
condition_labels = merged_df.loc[features_df.index, "has_condition"]

plt.figure(figsize=(10, 8))
# Color the points: convert boolean to int (0 or 1) so that we can use a colormap.
scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                      c=condition_labels.astype(int), cmap="coolwarm", alpha=0.7)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title(f"UMAP Clustering of Patients (Colored by '{chosen_condition}' Presence)")
cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.set_label("Has Condition (0=False, 1=True)")
plt.show()