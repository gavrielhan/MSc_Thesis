import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from collections import Counter
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

# get the patient embeddings obtained with ClinicalBERT
embeddings_df_baseline = pd.read_csv("patient_embeddings.csv")
# get the baseline conditions of each patient, to then use them to test clustering power
path_file = '/net/mraid20/export/genie/LabData/Data/10K/for_review/'
file_name = 'baseline_conditions_all.csv'
full_path = path_file + file_name

# Read the CSV file into a DataFrame
df_conditions_baseline = pd.read_csv(full_path)

# ===== Step 1: Data Preprocessing =====

# Remove columns in general_info that have >20% missing values.
missing_threshold = 0.8 * len(general_info)
general_info_filtered = general_info.dropna(axis=1, thresh=missing_threshold)

# Aggregate baseline conditions from df_conditions_baseline by RegistrationCode.
baseline_conditions = df_conditions_baseline.groupby("RegistrationCode")["english_name"] \
    .apply(lambda x: "; ".join(x.astype(str).unique())).reset_index()
baseline_conditions.rename(columns={"english_name": "baseline_conditions"}, inplace=True)

# Merge aggregated baseline conditions with general_info.
merged_df = pd.merge(general_info_filtered, baseline_conditions, on="RegistrationCode", how="left")

# Remove duplicate patients, keeping only the first encounter.
merged_df = merged_df.drop_duplicates(subset=["RegistrationCode"], keep="first")

# Create a primary condition column (taking the first condition listed).
def get_primary_condition(cond_str):
    if pd.isna(cond_str) or cond_str == "":
        return "None"
    else:
        return cond_str.split("; ")[0]

merged_df["primary_condition"] = merged_df["baseline_conditions"].apply(get_primary_condition)

# Encode the condition labels.
le = LabelEncoder()
merged_df["condition_label"] = le.fit_transform(merged_df["primary_condition"])

# ===== Step 2: Restrict to Common Patients =====
# Also drop duplicates from embeddings_df_baseline if necessary.
embeddings_df_baseline = embeddings_df_baseline.drop_duplicates(subset=["RegistrationCode"], keep="first")

# Restrict to patients present in both merged_df and embeddings_df_baseline.
common_ids = set(merged_df["RegistrationCode"]).intersection(set(embeddings_df_baseline["RegistrationCode"]))
merged_df = merged_df[merged_df["RegistrationCode"].isin(common_ids)]
embeddings_df = embeddings_df_baseline[embeddings_df_baseline["RegistrationCode"].isin(common_ids)]

# ===== Step 3: Extract Features and Labels =====
# For original data, select numeric features (remove identifier and condition columns).
feature_columns = merged_df.select_dtypes(include=[np.number]).columns.tolist()
for col in ["condition_label"]:  # remove label columns if present
    if col in feature_columns:
        feature_columns.remove(col)
for col in ["RegistrationCode"]:
    if col in feature_columns:
        feature_columns.remove(col)
for col in ["primary_condition", "baseline_conditions"]:
    if col in feature_columns:
        feature_columns.remove(col)

X_original = merged_df[feature_columns].dropna()
y = merged_df.loc[X_original.index, "condition_label"]

# For embeddings, select columns that start with "emb_".
embedding_cols = [col for col in embeddings_df.columns if col.startswith("emb_")]
X_embedding = embeddings_df[embedding_cols].dropna()

# Restrict both representations to the same set of patients.
common_index = set(X_original.index).intersection(set(X_embedding.index))
X_original = X_original.loc[list(common_index)]
y = y.loc[list(common_index)]
X_embedding = X_embedding.loc[list(common_index)]

print("Number of patients used before filtering small classes:", len(X_original))

# ===== Remove Classes with Fewer Than Two Samples =====
def filter_classes(X, y):
    counts = Counter(y)
    valid_classes = {cls for cls, count in counts.items() if count > 1}
    valid_idx = [idx for idx in y.index if y.loc[idx] in valid_classes]
    return X.loc[valid_idx], y.loc[valid_idx]

X_original, y = filter_classes(X_original, y)
X_embedding, _ = filter_classes(X_embedding, y)  # y is the same for both representations

print("Number of patients used after filtering small classes:", len(X_original))

# ===== Step 4: Supervised Classification =====
def evaluate_classification(X, y, representation_name=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"{representation_name} - Accuracy: {acc:.3f}, F1-score: {f1:.3f}")
    return acc, f1

print("=== Supervised Classification ===")
acc_orig, f1_orig = evaluate_classification(X_original, y, "Original Data")
acc_emb, f1_emb = evaluate_classification(X_embedding, y, "Embeddings")

# ===== Step 5: Unsupervised Clustering Evaluation =====
def evaluate_clustering(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    return silhouette_score(X, cluster_labels)

sil_orig = evaluate_clustering(X_original)
sil_emb = evaluate_clustering(X_embedding)
print(f"Clustering Silhouette Score - Original Data: {sil_orig:.3f}")
print(f"Clustering Silhouette Score - Embeddings: {sil_emb:.3f}")

# ===== Step 6: Mutual Information =====
mi_orig = mutual_info_classif(X_original, y, discrete_features="auto").mean()
mi_emb = mutual_info_classif(X_embedding, y, discrete_features="auto").mean()
print(f"Mutual Information (mean) - Original Data: {mi_orig:.3f}")
print(f"Mutual Information (mean) - Embeddings: {mi_emb:.3f}")

# ===== Step 7: UMAP Visualization =====
umap_reducer_orig = umap.UMAP(random_state=42)
umap_orig = umap_reducer_orig.fit_transform(X_original.values)

umap_reducer_emb = umap.UMAP(random_state=42)
umap_emb = umap_reducer_emb.fit_transform(X_embedding.values)

plt.figure(figsize=(12, 5))

# UMAP Projection for Original Data
plt.subplot(1, 2, 1)
plt.scatter(umap_orig[:, 0], umap_orig[:, 1], c="blue", alpha=0.7, edgecolor="k")
plt.title("UMAP Projection - Original Data (blue)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# UMAP Projection for Embeddings
plt.subplot(1, 2, 2)
plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c="blue", alpha=0.7, edgecolor="k")
plt.title("UMAP Projection - Embeddings (blue)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.tight_layout()
plt.show()

# ----- Helper Functions -----
def evaluate_classification_no_print(X, y):
    """Trains logistic regression and returns accuracy and weighted F1-score."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return acc, f1

def evaluate_clustering(X, n_clusters=5):
    """Applies KMeans and returns the silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    return silhouette_score(X, cluster_labels)

# ----- Assume the following objects are already defined from your preprocessing pipeline: -----
# X_original: original numeric features (from merged general_info) for common patients.
# y: condition labels for these patients.
# X_embedding: a DataFrame containing all 768 dimensions of embeddings for the same patients.

# For demonstration, here is an example of k values we want to test.
k_values = [8, 16, 32, 64, 128, 256, 500, 768]

# Compute constant metrics for the original data.
acc_orig, f1_orig = evaluate_classification_no_print(X_original, y)
sil_orig = evaluate_clustering(X_original)
mi_orig = mutual_info_classif(X_original, y, discrete_features="auto").mean()

print("Original Data Metrics:")
print(f"  Accuracy: {acc_orig:.3f}, F1-score: {f1_orig:.3f}")
print(f"  Silhouette: {sil_orig:.3f}")
print(f"  Mutual Information (mean): {mi_orig:.3f}")

# Initialize lists to store metrics for embeddings (using only first k dimensions).
accs_emb = []
f1s_emb = []
sil_emb_list = []
mi_emb_list = []

# Loop over k values, select the first k embedding dimensions and evaluate.
for k in k_values:
    X_emb_k = X_embedding.iloc[:, :k]
    acc, f1 = evaluate_classification_no_print(X_emb_k, y)
    accs_emb.append(acc)
    f1s_emb.append(f1)
    sil_score = evaluate_clustering(X_emb_k)
    sil_emb_list.append(sil_score)
    mi_score = mutual_info_classif(X_emb_k, y, discrete_features="auto").mean()
    mi_emb_list.append(mi_score)
    print(f"k={k:3d}: Acc={acc:.3f}, F1={f1:.3f}, Silhouette={sil_score:.3f}, MI={mi_score:.3f}")

# ----- Plotting -----
plt.figure(figsize=(14, 10))

# Plot Accuracy
plt.subplot(2, 2, 1)
plt.plot(k_values, accs_emb, marker='o', color='blue', label='Embeddings')
plt.hlines(acc_orig, xmin=min(k_values), xmax=max(k_values), colors='orange', linestyles='--', label='Original Data')
plt.xlabel("Number of Embedding Dimensions (k)")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy vs. k")
plt.legend()

# Plot F1-score
plt.subplot(2, 2, 2)
plt.plot(k_values, f1s_emb, marker='o', color='blue', label='Embeddings')
plt.hlines(f1_orig, xmin=min(k_values), xmax=max(k_values), colors='orange', linestyles='--', label='Original Data')
plt.xlabel("Number of Embedding Dimensions (k)")
plt.ylabel("F1-score")
plt.title("Classification F1-score vs. k")
plt.legend()

# Plot Silhouette Score
plt.subplot(2, 2, 3)
plt.plot(k_values, sil_emb_list, marker='o', color='blue', label='Embeddings')
plt.hlines(sil_orig, xmin=min(k_values), xmax=max(k_values), colors='orange', linestyles='--', label='Original Data')
plt.xlabel("Number of Embedding Dimensions (k)")
plt.ylabel("Silhouette Score")
plt.title("Clustering Silhouette Score vs. k")
plt.legend()

# Plot Mutual Information
plt.subplot(2, 2, 4)
plt.plot(k_values, mi_emb_list, marker='o', color='blue', label='Embeddings')
plt.hlines(mi_orig, xmin=min(k_values), xmax=max(k_values), colors='orange', linestyles='--', label='Original Data')
plt.xlabel("Number of Embedding Dimensions (k)")
plt.ylabel("Mean Mutual Information")
plt.title("Mutual Information vs. k")
plt.legend()

plt.tight_layout()
plt.show()