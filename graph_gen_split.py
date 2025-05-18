import json
import os
import threading
import time
from functools import partial
from multiprocessing import Pool
import random
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch_geometric.data import HeteroData
from LabData.DataLoaders.CGMLoader import CGMLoader

# ========== CONFIG ==========
WINDOW_SIZE = 3  # days
STEP_SIZE   = 3  # days
NEGATIVE_THRESHOLD = 0.01
DIAG_JSON_PATH = "diagnosis_mapping.json"
OUTPUT_DIR = "split"
random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== UTILS ==========
def start_pulse_timer(message="running", delay=1800, interval=1800):
    stop_event = threading.Event()
    def pulse():
        time.sleep(delay)
        while not stop_event.is_set():
            print(message)
            time.sleep(interval)
    t = threading.Thread(target=pulse)
    t.daemon = True
    t.start()
    return stop_event

def align_and_rescale(signal, target_len=96):
    signal = np.nan_to_num(signal, nan=0.0, posinf=1000, neginf=0.0)
    if len(signal) < 2:
        return np.zeros(target_len)
    interp = interp1d(np.linspace(0, 1, len(signal)), signal,
                      bounds_error=False, fill_value=(signal[0], signal[-1]))
    return interp(np.linspace(0, 1, target_len))

def compute_weights(signatures, signal):
    try:
        w = np.linalg.lstsq(signatures.T, signal, rcond=None)[0]
        return np.clip(w, 0, 1)
    except:
        return np.ones(signatures.shape[0]) / signatures.shape[0]

# ========== LOAD & PREP DATA ==========
print("Loading data?")
df_fb = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/follow_up_conditions_all.csv")
df_bb = pd.read_csv("/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv")
df_bb["created_at"] = pd.to_datetime(df_bb["created_at"], format='mixed', utc=True).dt.tz_convert(None)
df_fb["Date"] = pd.to_datetime(df_fb["Date"], format='mixed', utc=True).dt.tz_convert(None)

cgm = CGMLoader().get_data(study_ids=[10,1001,1002])
df_cgm = cgm.df.reset_index()
df_cgm["Date"] = pd.to_datetime(df_cgm["Date"]).dt.date

nmf_signatures = np.load("nmf_cgm_signatures.npy")
num_signatures = nmf_signatures.shape[0]

# embeddings
emb_base = pd.read_csv("filtered_patient_embeddings.csv")
emb_follow= pd.read_csv("followup_filtered_patient_embeddings.csv")
sleep_df  = (pd.read_csv("filtered_sleep_embeddings.csv")
               .set_index("RegistrationCode")
               .groupby("RegistrationCode").mean())
avg_sleep = sleep_df.mean(axis=0).values

# valid patients intersection
p_cgm  = df_cgm["RegistrationCode"].unique()
p_base = emb_base["RegistrationCode"].unique()
valid_patients = np.intersect1d(p_cgm, p_base)

# filter down
df_fb = df_fb[df_fb["RegistrationCode"].isin(valid_patients)]
df_cgm = df_cgm[df_cgm["RegistrationCode"].isin(valid_patients)]
emb_base   = emb_base[emb_base["RegistrationCode"].isin(valid_patients)]
emb_follow = emb_follow[emb_follow["RegistrationCode"].isin(valid_patients)]
sleep_df   = sleep_df.loc[sleep_df.index.isin(valid_patients)]

# group CGM by (patient, date)
grouped = df_cgm.groupby(["RegistrationCode","Date"])["GlucoseValue"].apply(lambda x: x.values)
cgm_dict = grouped.to_dict()

# index follow-ups
emb_follow["Date"] = pd.to_datetime(emb_follow["Date"]).dt.date
follow_dict = dict(tuple(emb_follow.groupby("RegistrationCode")))
base_idx    = emb_base.set_index("RegistrationCode")

# chosen conditions
chosen = ["Essential hypertension", "Other specified conditions associated with the spine (intervertebral disc displacement)",
    "Osteoporosis","Diabetes mellitus, type unspecified",
    "Non-alcoholic fatty liver disease","Coronary atherosclerosis",
    "Malignant neoplasms of breast"
]
cond2idx = {c:i for i,c in enumerate(chosen)}
num_conds = len(chosen)
all_patients = sorted(valid_patients.tolist())

# build patient?[diag dates]
patient_diag_dates = {}
for _,row in df_fb.iterrows():
    if row["english_name"] in cond2idx:
        patient_diag_dates.setdefault(row["RegistrationCode"],[]).append(row["Date"].date())

# ========== SLIDING WINDOWS ==========
all_dates = sorted(df_cgm["Date"].unique())
window_ranges  = []
window_centers = []
for i in range(0, len(all_dates)-WINDOW_SIZE+1, STEP_SIZE):
    start = all_dates[i]
    end   = all_dates[i+WINDOW_SIZE-1]
    center = start + (end - start)/2
    window_ranges.append((start,end))
    window_centers.append(center)
num_windows = len(window_ranges)

# assign each diagnosis to a window
diag_map = []  # entries of form {patient,cond_idx,window_idx,diag_date}
for _,row in df_fb.iterrows():
    p = row["RegistrationCode"]
    cond = row["english_name"]
    if cond not in cond2idx or p not in valid_patients: continue
    di = row["Date"].date()
    # find containing window
    assigned=False
    for wi,(s,e) in enumerate(window_ranges):
        if s<=di<=e:
            diag_map.append({"patient":p,"cond":cond2idx[cond],
                             "window":wi,"date":str(di)})
            assigned=True; break
    if not assigned:
        # fallback to closest center
        diffs = [abs((c - di).days) for c in window_centers]
        wi = int(np.argmin(diffs))
        diag_map.append({"patient":p,"cond":cond2idx[cond],
                         "window":wi,"date":str(di)})

# save mapping JSON
with open(DIAG_JSON_PATH,"w") as f:
    json.dump(diag_map, f, indent=2)

# reorganize by window
from collections import defaultdict
diag_by_win = defaultdict(list)
for d in diag_map:
    diag_by_win[d["window"]].append((d["patient"], d["cond"], d["date"]))

# ========== build_graph fn ==========
def build_graph(wi, wstart, wend, patients, do_train, last_loc):
    """
    do_train=True  -> add has-condition edges and drop after diag
    do_train=False -> never add has-condition; never drop
    """
    if do_train:
        # Remove patients diagnosed before the current window `wi`
        diagnosed_before = {
            d["patient"]
            for d in diag_map
            if d["window"] < wi
        }
        patients = [p for p in patients if p not in diagnosed_before]
        for p in diagnosed_before:
            if p in last_loc:
                del last_loc[p]
    het = HeteroData()
    het["signature"].x = torch.tensor(nmf_signatures, dtype=torch.float)
    het["condition"].x = torch.eye(num_conds, dtype=torch.float)
    het['meta'] = {}
    het['meta']['start_date'] = str(wstart)  # store as string to avoid serialization issues
    het['meta']['end_date'] = str(wend)
    # collect patient embeddings & names
    feats, names = [], []
    for p in patients:
        # get embedding at wend
        dfp = follow_dict.get(p)
        if dfp is not None:
            dfp = dfp[dfp["Date"]<=wend]
        emb = (dfp.sort_values("Date").filter(like="emb_").values[-1]
               if dfp is not None and not dfp.empty
               else base_idx.loc[p].filter(like="emb_").values)
        sleep = sleep_df.loc[p].values if p in sleep_df.index else avg_sleep
        feats.append(torch.from_numpy(np.concatenate([emb,sleep])).float())
        names.append(p)
    het["patient"].x = torch.stack(feats) if feats else torch.empty((0,138))
    het["patient"].name = names

    # event/duration matrix
    ev, du = [], []
    for p in names:
        ev_row, du_row = [], []
        for ci in range(num_conds):
            # Find first diagnosis window (if any)
            future_diags = [d["window"] for d in diag_map if
                            d["patient"] == p and d["cond"] == ci and d["window"] >= wi]
            if future_diags:
                ev_row.append(1)
                du_row.append(min(future_diags) - wi)
            else:
                ev_row.append(0)
                du_row.append(0.0)

        ev.append(ev_row); du.append(du_row)
    het["patient"].event    = torch.tensor(ev, dtype=torch.long)
    het["patient"].duration = torch.tensor(du, dtype=torch.float)

    # patient?signature edges
    eidx, wts = [], []
    for pi,p in enumerate(names):
        for day in pd.date_range(wstart, wend).date:
            sig = cgm_dict.get((p,day))
            if sig is None: continue
            v = align_and_rescale(sig)
            kws = compute_weights(nmf_signatures,v)
            for si,ww in enumerate(kws):
                if ww>NEGATIVE_THRESHOLD:
                    eidx.append([pi,si]); wts.append(ww)
    if eidx:
        het["patient","to","signature"].edge_index = torch.tensor(eidx).T
        het["patient","to","signature"].edge_attr  = torch.tensor(wts)

    # patient?condition (only in train)
    if do_train and wi in diag_by_win:
        ci_edges = []
        for p, ci, _ in diag_by_win[wi]:
            if p in names:  # only if patient still in this window
                ci_edges.append([names.index(p), ci])
        if ci_edges:
            het["patient", "has", "condition"].edge_index = torch.tensor(ci_edges).T

    # temporal follows (keep everyone in val/test)
    # ? Build valid follows edges, avoid stale indices
    src, dst = [], []
    name_to_idx = {p: i for i, p in enumerate(names)}

    # ? Clean up stale last_loc entries
    for p in list(last_loc.keys()):
        if p not in name_to_idx:
            del last_loc[p]

    # ? Now build follows safely
    for p, ni in name_to_idx.items():
        if p in last_loc:
            prev_wi, prev_idx = last_loc[p]
            if prev_idx < het['patient'].x.size(0):  # Should always be true now, but safe
                src.append(prev_idx)
                dst.append(ni)
        last_loc[p] = (wi, ni)
    if src:
        het["patient", "follows", "patient"].edge_index = torch.tensor([src, dst])
    if ('patient', 'follows', 'patient') in het.edge_types:
        ei = het['patient', 'follows', 'patient'].edge_index
        assert ei.max().item() < het['patient'].x.size(0), \
            f"[Window {wi}] Invalid index in 'follows' edge: max={ei.max().item()} vs num_patients={het['patient'].x.size(0)}"
    return wi, het

# store last_loc per split
build_graph.last_loc = {}

# ========== RUN ==========
print("Building and splitting?")
# ===== Define the patient splits =====
shuffled_patients = all_patients.copy()
random.shuffle(shuffled_patients)

n = len(shuffled_patients)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

train_set = shuffled_patients[:n_train]
val_set   = shuffled_patients[n_train : n_train + n_val]
test_set  = shuffled_patients[n_train + n_val :]

# ========== BUILD EACH SPLIT SEPARATELY ==========
pulse = start_pulse_timer("Still building?", delay=1800)
# --- TRAIN ---
build_graph.last_loc = {}
train_graphs = []
split_last_locs = {
    "train": {},
    "val": {},
    "test": {}
}

for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[TRAIN]   Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, train_set, True, split_last_locs["train"])
    train_graphs.append(g)

# --- VAL ---
build_graph.last_loc = {}
val_graphs = []
for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[VALID]   Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, val_set, False, split_last_locs["val"])
    val_graphs.append(g)

# --- TEST ---
build_graph.last_loc = {}
test_graphs = []
for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[TEST]    Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, test_set, False, split_last_locs["test"])
    test_graphs.append(g)

pulse.set()

for split_name, patient_list in [("train", train_set), ("val", val_set), ("test", test_set)]:
    num_diags = sum(
        1 for d in diag_map
        if d["patient"] in patient_list
    )
    print(f"{split_name}: {num_diags} future diagnoses in diag_map")

# Then save as before:
torch.save(train_graphs, os.path.join(OUTPUT_DIR, "train_graphs_3d.pt"))
torch.save(val_graphs,   os.path.join(OUTPUT_DIR, "val_graphs_3d.pt"))
torch.save(test_graphs,  os.path.join(OUTPUT_DIR, "test_graphs_3d.pt"))


print("Done. Graphs and diagnosis_mapping.json are ready.")
