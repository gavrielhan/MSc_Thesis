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
from torch_geometric.utils import to_undirected
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader

study_ids = [10,1001,1002]
bm = BodyMeasuresLoader().get_data(study_ids=study_ids, groupby_reg='first')
df_meta = bm.df.join(bm.df_metadata)
df_meta = df_meta.reset_index().rename(columns={'index': 'RegistrationCode'})

# demographic columns
sex_col = next((c for c in df_meta.columns if 'sex' in c.lower() or 'gender' in c.lower()), None)
bmi_col = next((c for c in df_meta.columns if 'bmi' in c.lower()), None)

# drop patients with missing sex information
if sex_col:
    n_missing_sex = df_meta[sex_col].isna().sum()
    if n_missing_sex:
        print(f"Excluding {n_missing_sex} patients due to missing sex")
    df_meta = df_meta.dropna(subset=[sex_col])
    sex_map = (
        df_meta.set_index('RegistrationCode')[sex_col]
        .astype('category')
        .cat.codes
        .to_dict()
    )
else:
    sex_map = {}

# bmi: fill missing with median and report how many were imputed
if bmi_col:
    bmi_series = df_meta.set_index('RegistrationCode')[bmi_col].astype(float)
    bmi_median = bmi_series.median(skipna=True)
    n_infer = bmi_series.isna().sum()
    if n_infer:
        print(f"Inferred BMI for {n_infer} patients using median {bmi_median:.2f}")
    bmi_series = bmi_series.fillna(bmi_median)
    bmi_map = bmi_series.to_dict()
else:
    bmi_map = {}

age_gender = df_meta[['RegistrationCode', 'age', 'yob']].dropna(subset=['age', 'yob'])
yob_map = dict(zip(age_gender['RegistrationCode'], age_gender['yob']))

# ========== CONFIG ==========
WINDOW_SIZE = 3  # days
STEP_SIZE   = 3  # days
NEGATIVE_THRESHOLD = 0.01
DIAG_JSON_PATH = "diagnosis_mapping.json"
OUTPUT_DIR = "split"
random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# toggle demographic vs. precomputed patient embeddings
age_sex_bmi = True

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


if age_sex_bmi:
    # only sleep embeddings are loaded; demographics will be derived later
    sleep_df = (
        pd.read_csv("filtered_sleep_embeddings.csv")
          .set_index("RegistrationCode")
          .groupby("RegistrationCode").mean()
    )
    avg_sleep = sleep_df.mean(axis=0).values
    patient_dim = 128
else:
    # precomputed patient embeddings
    emb_base = pd.read_csv("filtered_patient_embeddings.csv")
    emb_follow = pd.read_csv("followup_filtered_patient_embeddings.csv")
    sleep_df = (
        pd.read_csv("filtered_sleep_embeddings.csv")
          .set_index("RegistrationCode")
          .groupby("RegistrationCode").mean()
    )
    avg_sleep = sleep_df.mean(axis=0).values
    patient_dim = len([c for c in emb_base.columns if c.startswith("emb_")])
sleep_dim = sleep_df.shape[1]

# valid patients intersection
p_cgm = df_cgm["RegistrationCode"].unique()
if age_sex_bmi:
    p_src = df_meta["RegistrationCode"].unique()
else:
    p_src = emb_base["RegistrationCode"].unique()
valid_patients = np.intersect1d(p_cgm, p_src)

# filter down
df_fb = df_fb[df_fb["RegistrationCode"].isin(valid_patients)]
df_cgm = df_cgm[df_cgm["RegistrationCode"].isin(valid_patients)]
sleep_df = sleep_df.loc[sleep_df.index.isin(valid_patients)]
if age_sex_bmi:
    df_meta = df_meta[df_meta["RegistrationCode"].isin(valid_patients)]
else:
    emb_base   = emb_base[emb_base["RegistrationCode"].isin(valid_patients)]
    emb_follow = emb_follow[emb_follow["RegistrationCode"].isin(valid_patients)]

# group CGM by (patient, date)
grouped = df_cgm.groupby(["RegistrationCode","Date"])["GlucoseValue"].apply(lambda x: x.values)
cgm_dict = grouped.to_dict()

# Build a per-patient queue of all available CGM dates:
from collections import defaultdict
pending_signals = defaultdict(list)
for (p, day) in cgm_dict.keys():
    pending_signals[p].append(day)
for p in pending_signals:
    pending_signals[p].sort()

# index follow-ups
if not age_sex_bmi:
    emb_follow["Date"] = pd.to_datetime(emb_follow["Date"]).dt.date
    follow_dict = dict(tuple(emb_follow.groupby("RegistrationCode")))
    base_idx = emb_base.set_index("RegistrationCode")

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
    do_train=True  -> drop patients diagnosed before this window
    do_train=False -> keep everyone
    """
    # 1) drop diagnosed-before (train only)
    if do_train:
        to_drop = {d["patient"] for d in diag_map if d["window"] < wi}
        patients = [p for p in patients if p not in to_drop]
        for p in to_drop:
            last_loc.pop(p, None)

    het = HeteroData()
    # --- signature & condition nodes ---
    het["signature"].x = torch.tensor(nmf_signatures, dtype=torch.float)
    het["condition"].x = torch.eye(num_conds, dtype=torch.float)
    het.start_date = str(wstart)
    het.end_date   = str(wend)

    # --- patient features & names ---
    feats, names = [], []
    for p in patients:
        if age_sex_bmi:
            # project [age, sex, bmi] ? 128-d
            age = wend.year - int(yob_map.get(p, wend.year))
            sex = sex_map.get(p, 0)
            bmi = bmi_map.get(p, 0.0)
            demo = np.array([age, sex, bmi], dtype=float)
            # simple linear interpolation to 128
            x_old = np.linspace(0, 1, len(demo))
            x_new = np.linspace(0, 1, patient_dim)
            emb   = np.interp(x_new, x_old, demo)
        else:
            dfp = follow_dict.get(p)
            if dfp is not None:
                dfp = dfp[dfp["Date"] <= wend]
            if dfp is not None and not dfp.empty:
                emb = dfp.sort_values("Date").filter(like="emb_").values[-1]
            else:
                emb = base_idx.loc[p].filter(like="emb_").values
        # always append sleep
        sleep = sleep_df.loc[p].values if p in sleep_df.index else avg_sleep
        feats.append(torch.from_numpy(np.concatenate([emb, sleep])).float())
        names.append(p)

    feat_dim = patient_dim + sleep_dim
    het["patient"].x    = torch.stack(feats) if feats else torch.empty((0, feat_dim))
    het["patient"].name = names

    # --- event/duration for Cox ---
    ev, du = [], []
    for p in names:
        e_row, d_row = [], []
        for ci in range(num_conds):
            future = [d["window"] for d in diag_map
                      if d["patient"]==p and d["cond"]==ci and d["window"]>=wi]
            if future:
                e_row.append(1)
                delta = min(future) - wi
                d_row.append(delta if delta>0 else 1.0)
            else:
                e_row.append(0)
                d_row.append(0.0)
        ev.append(e_row); du.append(d_row)
    het["patient"].event    = torch.tensor(ev, dtype=torch.long)
    het["patient"].duration = torch.tensor(du, dtype=torch.float)

    # --- patient?signature (1 signal per patient per window) ---
    eidx, wts = [], []
    for pi, p in enumerate(names):
        # pick earliest pending day in this window
        days = [d for d in pending_signals[p] if wstart <= d <= wend]
        if not days:
            continue
        day = days.pop(0)  # consume it
        sig = cgm_dict[(p, day)]
        v   = align_and_rescale(sig)
        kws = compute_weights(nmf_signatures, v)
        # if none above threshold, force the top one
        pos = [(si,ww) for si,ww in enumerate(kws) if ww>NEGATIVE_THRESHOLD]
        if not pos:
            best = int(np.argmax(kws))
            pos = [(best, kws[best])]
        for si, ww in pos:
            eidx.append([pi, si]); wts.append(ww)

    if eidx:
        sig_ei = torch.tensor(eidx).T                  # [2,E]
        wts_t  = torch.tensor(wts, dtype=torch.float)
        # normalize per-patient sum?1
        sums = defaultdict(float)
        for idx, pi in enumerate(sig_ei[0].tolist()):
            sums[pi] += wts_t[idx].item()
        norm = [ wts_t[i]/(sums[pi]+1e-9) for i,pi in enumerate(sig_ei[0].tolist()) ]
        sig_w = torch.tensor(norm, dtype=torch.float).view(-1,1)
        het["patient","to","signature"].edge_index = sig_ei
        het["patient","to","signature"].edge_attr  = sig_w
        rev_ei = sig_ei.flip(0)
        het["signature","to_rev","patient"].edge_index = rev_ei
        het["signature","to_rev","patient"].edge_attr  = sig_w

    # --- patient?condition edges (train only) ---
    if do_train and wi in diag_by_win:
        ci_edges = [[names.index(p),ci] for p,ci,_ in diag_by_win[wi] if p in names]
        if ci_edges:
            cond_ei = torch.tensor(ci_edges).T
            het["patient","has","condition"].edge_index     = cond_ei
            het["condition","has_rev","patient"].edge_index = cond_ei.flip(0)

    # --- temporal follows edges (only connect same patient) ---
    split_id   = id(last_loc)
    prev_names = build_graph.last_names.get(split_id)
    name2idx = {p: i for i, p in enumerate(names)}
    src, dst = [], []
    for p, ni in name2idx.items():
        # if this patient was in last_loc *and* wasn't just dropped
        if p in last_loc and prev_names is not None:
            _, prev_i = last_loc[p]
            # bounds-check: prev_i must still be a valid index
            if prev_i < len(prev_names) and prev_names[prev_i] == p:
                # AND ensure that prev_i < current n_src
                if prev_i < len(names):
                    src.append(prev_i)
                    dst.append(ni)
                # otherwise they were dropped ? no edge
        # update last_loc to current window and new index
        last_loc[p] = (wi, ni)


    if src:
        ei_f = torch.tensor([src,dst], dtype=torch.long)
        ages = [ (wend.year - int(yob_map.get(names[d], wend.year))) / 100.0
                 for d in dst ]
        wt_f = torch.tensor(ages, dtype=torch.float).view(-1,1)
        het["patient","follows","patient"].edge_index = ei_f
        het["patient","follows","patient"].edge_attr  = wt_f
        het["patient","follows_rev","patient"].edge_index = ei_f.flip(0)
        het["patient","follows_rev","patient"].edge_attr  = wt_f

    # remember names for next window
    build_graph.last_names[split_id] = names
    het.window = wi
    return wi, het

# store last_loc per split
build_graph.last_loc = {}
build_graph.last_names = {}
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
build_graph.last_names = {}
train_graphs = []
split_last_locs = {
    "train": {},
    "val": {},
    "test": {}
}

for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[TRAIN]   Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, train_set, True, split_last_locs["train"])
    for rel in g.edge_types:
        ei = g[rel].edge_index
        src_type, _, dst_type = rel
        n_src = g[src_type].x.size(0)
        n_dst = g[dst_type].x.size(0)
        # Row 0 must index into src, row 1 into dst
        assert ei[0].max().item() < n_src, f"{rel} src OOB: {ei[0].max()} ? {n_src}"
        assert ei[1].max().item() < n_dst, f"{rel} dst OOB: {ei[1].max()} ? {n_dst}"

    train_graphs.append(g)

# --- VAL ---
build_graph.last_loc = {}
build_graph.last_names = {}
val_graphs = []
for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[VALID]   Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, val_set, False, split_last_locs["val"])
    for rel in g.edge_types:
        ei = g[rel].edge_index
        src_type, _, dst_type = rel
        n_src = g[src_type].x.size(0)
        n_dst = g[dst_type].x.size(0)
        # Row 0 must index into src, row 1 into dst
        assert ei[0].max().item() < n_src, f"{rel} src OOB: {ei[0].max()} ? {n_src}"
        assert ei[1].max().item() < n_dst, f"{rel} dst OOB: {ei[1].max()} ? {n_dst}"

    val_graphs.append(g)

# --- TEST ---
build_graph.last_loc = {}
build_graph.last_names = {}
test_graphs = []
for wi, (wstart, wend) in enumerate(window_ranges):
    print(f"[TEST]    Window {wi}: {wstart} ? {wend}")
    _, g = build_graph(wi, wstart, wend, test_set, False, split_last_locs["test"])
    for rel in g.edge_types:
        ei = g[rel].edge_index
        src_type, _, dst_type = rel
        n_src = g[src_type].x.size(0)
        n_dst = g[dst_type].x.size(0)
        # Row 0 must index into src, row 1 into dst
        assert ei[0].max().item() < n_src, f"{rel} src OOB: {ei[0].max()} ? {n_src}"
        assert ei[1].max().item() < n_dst, f"{rel} dst OOB: {ei[1].max()} ? {n_dst}"

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
