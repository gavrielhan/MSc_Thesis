# === Fine-tune 3 models on Messidor (5 epochs each), pick a test image all 3 get right,
# === then save green-halo attention overlays for each model to outputs/images/ ===

import os, math, json, random, glob, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os, numpy as np
from PIL import Image, ImageFilter

def attn_vec_to_grid(attn_in, img_size, patch_size, assume_cls_first=True, verbose=False):
    """
    Converts common ViT attention forms to a [gh, gw] grid (no CLS).
    Handles:
      - dict/list wrappers
      - [L, H, T, T]  -> last layer, mean heads
      - [H, T, T]     -> mean heads
      - [T, T]        -> square (uses CLS->patch row if CLS exists)
      - [T]           -> vector (with/without CLS)
      - flattened squares [T*T]
    Tolerates off-by-one lengths by zero-padding or truncating.
    """
    import torch

    H, W = int(img_size[0]), int(img_size[1])
    gh, gw = H // patch_size, W // patch_size
    Np = gh * gw               # patches
    N  = Np + 1                # patches + CLS

    def to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
        return np.asarray(x, dtype=np.float32)

    # unwrap containers
    attn = attn_in
    if isinstance(attn, dict):
        for k in ('attn', 'attn_weights', 'attn_probs', 'attn_map', 'attentions'):
            if k in attn:
                attn = attn[k]; break
    if isinstance(attn, (list, tuple)):
        attn = attn[-1]

    attn = to_np(attn)
    attn = np.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"[attn_vec_to_grid] incoming shape: {attn.shape}")

    # 4D -> [T,T]
    if attn.ndim == 4 and attn.shape[-1] == attn.shape[-2]:
        attn = attn[-1].mean(0)
    # 3D -> [T,T]
    if attn.ndim == 3 and attn.shape[-1] == attn.shape[-2]:
        attn = attn.mean(0)

    # 2D square [T,T]
    if attn.ndim == 2 and attn.shape[0] == attn.shape[1]:
        T = attn.shape[0]
        # pick CLS->patch row when CLS is present, else average queries
        if T == N:
            v = attn[0, 1:]
        elif T == Np:
            v = attn.mean(0)
        else:
            # near-square fallback
            v = attn[0]
            if v.size >= N:
                v = v[1:]
        # pad/truncate to Np
        if v.size < Np:
            v = np.pad(v, (0, Np - v.size))
        elif v.size > Np:
            v = v[:Np]
        return v.reshape(gh, gw)

    # 1D vector / flattened square
    if attn.ndim == 1:
        L = int(attn.size)

        # direct matches
        if L == Np:
            v = attn
        elif L == N:  # with CLS
            v = attn[1:]
        elif L == N * N:
            M = attn.reshape(N, N)
            v = M[0, 1:]
        elif L == Np * Np:
            M = attn.reshape(Np, Np)
            v = M.mean(0)
        else:
            # tolerate small off-by-k (e.g., 1935)
            if L in (Np - 2, Np - 1):
                v = np.pad(attn, (0, Np - L))              # pad zeros
            elif L in (N - 2, N - 1, N + 1):               # near N (CLS + patches)
                # drop presumed CLS, then pad/truncate to Np
                base = attn[1:] if assume_cls_first and L >= 2 else attn
                if base.size < Np:
                    v = np.pad(base, (0, Np - base.size))
                else:
                    v = base[:Np]
            else:
                # try nearest square
                n = int(round(np.sqrt(L)))
                if n > 1 and n * n == L:
                    M = attn.reshape(n, n)
                    if n >= N:      v = M[0, 1:1+Np].reshape(-1)
                    elif n == Np:   v = M.mean(0)
                    else:           v = M.mean(0)
                else:
                    raise RuntimeError(
                        f"Unrecognized attention vector length {L}; expected {Np}, {N}, {Np*Np}, or {N*N}."
                    )
        # finalize
        v = np.nan_to_num(v, nan=0.0)
        if v.size < Np:
            v = np.pad(v, (0, Np - v.size))
        elif v.size > Np:
            v = v[:Np]
        return v.reshape(gh, gw)

    raise RuntimeError(f"Unrecognized attention ndim={attn.ndim}, shape={getattr(attn, 'shape', None)}")


def green_halo_overlay(img_path, attn_in, img_size, patch_size, save_path,
                       top_p=None, blur_px=16, alpha=0.45, verbose=False, **kwargs):
    """
    Draw a green halo over the highest-attention patches and save the image.
    Accepts both 'top_p' (fraction) and legacy 'top_percent' in kwargs.
    """
    # alias: accept top_percent as 0..1 or 0..100
    if top_p is None and 'top_percent' in kwargs:
        tp = float(kwargs['top_percent'])
        top_p = tp / 100.0 if tp > 1.0 else tp
    if top_p is None:
        top_p = 0.05  # default top 5%

    img = Image.open(img_path).convert('RGB')
    H, W = int(img_size[0]), int(img_size[1])
    if img.size != (W, H):
        img = img.resize((W, H), resample=Image.BILINEAR)

    attn_grid = attn_vec_to_grid(attn_in, img_size, patch_size, verbose=verbose)

    # normalize
    attn = attn_grid.astype(np.float32)
    attn -= attn.min()
    if attn.max() > 0:
        attn /= attn.max()

    # threshold (top_p)
    thr = np.quantile(attn, 1.0 - top_p) if 0.0 < top_p < 1.0 else 1.0
    mask_small = (attn >= thr).astype(np.uint8) * 255

    # upsample mask to image size
    mask_img = Image.fromarray(mask_small).resize((W, H), resample=Image.NEAREST)

    # halo = big blur ? small blur
    blur_big = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_px))
    blur_small = mask_img.filter(ImageFilter.GaussianBlur(radius=max(1, blur_px // 3)))
    halo = np.clip(np.array(blur_big, dtype=np.float32) - np.array(blur_small, dtype=np.float32), 0, 255)

    # alpha map
    halo_norm = halo / (halo.max() + 1e-6)
    halo_alpha = (halo_norm * alpha)
    halo_alpha_img = Image.fromarray(np.uint8(halo_alpha * 255))

    # compose green glow
    green_layer = Image.new('RGBA', (W, H), (0, 255, 0, 0))
    green_layer.putalpha(halo_alpha_img)
    out = Image.alpha_composite(img.convert('RGBA'), green_layer).convert('RGB')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out.save(save_path)

# -------------------- PATHS --------------------
EXTERNAL_ROOT = '/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/external_datasets'
MESSIDOR_IMG_DIR  = os.path.join(EXTERNAL_ROOT, 'messidor', 'IMAGES')
MESSIDOR_DATA_CSV = os.path.join(EXTERNAL_ROOT, 'messidor', 'messidor_data.csv')
MESSIDOR_PATIENT_CSV = os.path.join(EXTERNAL_ROOT, 'messidor', 'messidor-2.csv')  # for patient-wise split

STRATEGIES = [
    {
        "name": "retina_feature_finetune",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_retina_finetune.pth",
    },
    {
        "name": "imagenet_finetune",
        "ckpt": "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA/pretrained_IN/IN22K-vit.h.14-900e.pth.tar",
    },
    {
        "name": "retina_pretrain_finetune",
        "ckpt": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/gavrielh/checkpoint_pretrain_newrun.pth",
    },
]

OUT_DIR = os.path.join('outputs', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- CONFIG --------------------
SEED = 21
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'img_size': (616, 616),
    'patch_size': 14,
    'embed_dim': 1280,
    'depth': 32,
    'num_heads': 16,
    'use_lora': True,
    'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.2,
    'batch_size': 2,
    'epochs': 5,
    'lr_lora': 1e-4,
    'lr_head': 3e-4,
    'weight_decay': 1e-2,
    'n_classes': 2,
}

# -------------------- TRANSFORMS --------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
trsf = transforms.Compose([
    transforms.Resize(CONFIG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -------------------- DATASET --------------------
class MessidorDataset(Dataset):
    def __init__(self, df, img_dir, transform, label_col='adjudicated_dr_grade'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        label = int(row[self.label_col] != 0)  # binarize
        p = os.path.join(self.img_dir, img_name)
        if not os.path.exists(p):
            base = os.path.splitext(img_name)[0]
            for ext in ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG']:
                cand = os.path.join(self.img_dir, base+ext)
                if os.path.exists(cand): p = cand; break
        img = Image.open(p).convert('RGB')
        return self.transform(img), label, p

# -------------------- IJepa ViT --------------------

# >>> set this to the JEPA repo root (folder that contains the "ijepa" directory) <<<
REPO_ROOT = "/home/gavrielh/PycharmProjects/MSc_Thesis/JEPA"

# Make both 'ijepa' and top-level 'src' importable
sys.path.insert(0, REPO_ROOT)                      # enables 'ijepa.*'
sys.path.insert(0, os.path.join(REPO_ROOT, "ijepa"))  # enables 'src.*'
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "ijepa")))
from ijepa.src.models.vision_transformer import VisionTransformer

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__(); self.head = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        if x.ndim == 3: x = x[:,0]  # CLS
        return self.head(x)

def build_encoder(cfg):
    return VisionTransformer(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], in_chans=3,
        embed_dim=cfg['embed_dim'], depth=cfg['depth'], num_heads=cfg['num_heads'],
        use_lora=cfg['use_lora'], lora_r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout']
    )

def _strip_module(sd):
    return {(k[7:] if isinstance(k,str) and k.startswith('module.') else k): v for k,v in sd.items()}

def load_into_encoder_any(enc, ckpt_path):
    """Robustly load encoder weights from pretrain or finetuned (nn.Sequential) ckpts."""
    if not (ckpt_path and os.path.isfile(ckpt_path)):
        print(f"[load] no ckpt: {ckpt_path}"); return
    raw = torch.load(ckpt_path, map_location=DEVICE)
    state_like = None
    if isinstance(raw, dict):
        if 'model_state_dict' in raw and isinstance(raw['model_state_dict'], dict):
            enc_keys = {k.replace('0.','',1): v for k,v in raw['model_state_dict'].items() if k.startswith('0.')}
            state_like = enc_keys if enc_keys else raw['model_state_dict']
        for k in ['enc','encoder','state_dict','model']:
            if state_like is None and k in raw and isinstance(raw[k], dict):
                state_like = raw[k]
        if state_like is None:
            state_like = raw
    else:
        state_like = {}
    state_like = _strip_module(state_like)
    # remap linear names (matches your training code)
    remapped = {}
    for k, v in state_like.items():
        k2 = k
        k2 = k2.replace('.attn.qkv.weight','.attn.qkv.linear.weight').replace('.attn.qkv.bias','.attn.qkv.linear.bias')
        k2 = k2.replace('.attn.proj.weight','.attn.proj.linear.weight').replace('.attn.proj.bias','.attn.proj.linear.bias')
        remapped[k2] = v

    enc_keys = set(enc.state_dict().keys())
    filtered = {k:v for k,v in remapped.items() if k in enc_keys or k in ['patch_embed.proj.weight','pos_embed']}
    # handle 6->3 conv
    w = filtered.get('patch_embed.proj.weight')
    if w is not None and w.shape[1]==6 and enc.patch_embed.proj.weight.shape[1]==3:
        filtered['patch_embed.proj.weight'] = 0.5*(w[:,0:3,:,:] + w[:,3:6,:,:])
    # drop mismatched pos_embed
    if 'pos_embed' in filtered and getattr(enc,'pos_embed', None) is not None:
        if filtered['pos_embed'].shape != enc.pos_embed.shape:
            filtered.pop('pos_embed')
    missing, unexpected = enc.load_state_dict(filtered, strict=False)
    print(f"[load] loaded {len(filtered)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")

def freeze_all_but_lora_and_head(model_seq):
    # model_seq = nn.Sequential(encoder, head)
    for p in model_seq[0].parameters():
        p.requires_grad = False
    for n,p in model_seq[0].named_parameters():
        if 'lora_' in n: p.requires_grad = True
    for p in model_seq[1].parameters():
        p.requires_grad = True

# -------------------- ATTENTION (rollout via qkv) --------------------
@torch.no_grad()
def collect_attn_softmax_from_qkv(encoder, x):
    """Hook each block's attn.qkv, reconstruct softmax(QK^T). Returns list of [B,H,N,N]."""
    qkv_outs = []
    hooks = []
    for name, m in encoder.named_modules():
        if name.endswith('attn.qkv'):
            hooks.append(m.register_forward_hook(lambda m, inp, out: qkv_outs.append(out.detach())))
    _ = encoder(x)  # trigger hooks
    for h in hooks: h.remove()
    # find num heads
    num_heads = None
    for name, m in encoder.named_modules():
        if name.endswith('attn') and hasattr(m, 'num_heads'):
            num_heads = int(m.num_heads); break
    if num_heads is None: num_heads = CONFIG['num_heads']
    attn_all = []
    for qkv in qkv_outs:
        B, N, threeD = qkv.shape
        D = threeD // 3
        q, k = qkv[...,:D], qkv[...,D:2*D]
        head_dim = D // num_heads
        q = q.reshape(B, N, num_heads, head_dim).permute(0,2,1,3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0,2,1,3)
        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(head_dim)
        attn = torch.softmax(attn, dim=-1)  # [B,H,N,N]
        attn_all.append(attn)
    return attn_all

def attention_rollout(attn_list, head_fuse='mean', discard_ratio=0.0):
    """Fuse heads per layer, add residual, row-normalize, cumulative product. Returns [B, N-1] (CLS->patch)."""
    assert len(attn_list) > 0
    B, H, N, _ = attn_list[0].shape
    result = None
    for A in attn_list:
        A = (A.max(dim=1).values if head_fuse=='max' else A.mean(dim=1))  # [B,N,N]
        if discard_ratio > 0:
            k = int(A.size(-1)*discard_ratio)
            if k > 0:
                topv, topi = torch.topk(A, A.size(-1)-k, dim=-1)
                Z = torch.zeros_like(A); Z.scatter_(-1, topi, topv)
                A = Z / (Z.sum(dim=-1, keepdim=True)+1e-6)
        I = torch.eye(N, device=A.device).unsqueeze(0)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True)+1e-6)
        result = A if result is None else result @ A
    cls_to_patches = result[:,0,1:]  # [B, N-1]
    return cls_to_patches


# -------------------- PREP SPLIT (patient-wise 80/20) --------------------
df_all = pd.read_csv(MESSIDOR_DATA_CSV)
df_all['adjudicated_dr_grade'] = (df_all['adjudicated_dr_grade'] != 0).astype(int)

if os.path.exists(MESSIDOR_PATIENT_CSV):
    df_pat = pd.read_csv(MESSIDOR_PATIENT_CSV)
    df_pat.columns = df_pat.columns.str.strip()
    col = df_pat.columns[0]
    patient_groups = []
    for _, row in df_pat.iterrows():
        imgs = [x.strip() for x in str(row[col]).split(';') if x and x.strip().lower() != 'nan']
        if imgs: patient_groups.append(imgs)
    rng = np.random.default_rng(SEED)
    rng.shuffle(patient_groups)
    cut = int(0.8*len(patient_groups))
    train_imgs = set([im for g in patient_groups[:cut] for im in g])
    test_imgs  = set([im for g in patient_groups[cut:] for im in g])
    df_train = df_all[df_all['image_id'].isin(train_imgs)].reset_index(drop=True)
    df_test  = df_all[df_all['image_id'].isin(test_imgs)].reset_index(drop=True)
else:
    # fallback image-wise
    idx = np.arange(len(df_all)); rng = np.random.default_rng(SEED); rng.shuffle(idx)
    s = int(0.8*len(idx))
    df_train = df_all.iloc[idx[:s]].reset_index(drop=True)
    df_test  = df_all.iloc[idx[s:]].reset_index(drop=True)

train_ds = MessidorDataset(df_train, MESSIDOR_IMG_DIR, trsf)
test_ds  = MessidorDataset(df_test,  MESSIDOR_IMG_DIR, trsf)
train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

# -------------------- TRAIN/EVAL PER STRATEGY --------------------
correct_paths_per_model = {}
trained_states = {}  # keep in RAM only

def compute_class_weights(loader, n_classes):
    labels = []
    for _, y, _ in DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=0):
        labels.extend(y.numpy())
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=n_classes)
    inv = 1.0/(counts+1e-6)
    w = inv/np.sum(inv)*n_classes
    return torch.tensor(w, dtype=torch.float, device=DEVICE)

@torch.no_grad()
def evaluate_and_collect(model, loader):
    model.eval()
    correct_paths = set()
    for x, y, paths in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        y = y.numpy()
        for i,p in enumerate(paths):
            if preds[i] == y[i]:
                correct_paths.add(p)
    return correct_paths

for strat in STRATEGIES:
    name, ckpt = strat['name'], strat['ckpt']
    print(f"\n=== Fine-tuning: {name} ===")
    # build model
    enc = build_encoder(CONFIG).to(DEVICE)
    load_into_encoder_any(enc, ckpt)
    head = ClassificationHead(CONFIG['embed_dim'], CONFIG['n_classes']).to(DEVICE)
    model = nn.Sequential(enc, head).to(DEVICE)
    # freeze all but LoRA + head
    for p in model[0].parameters(): p.requires_grad = False
    for n,p in model[0].named_parameters():
        if 'lora_' in n: p.requires_grad = True
    for p in model[1].parameters(): p.requires_grad = True

    # loss/opt
    class_weights = compute_class_weights(train_loader, CONFIG['n_classes'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model[0].parameters() if p.requires_grad], 'lr': CONFIG['lr_lora'], 'weight_decay': CONFIG['weight_decay']},
        {'params': model[1].parameters(), 'lr': CONFIG['lr_head'], 'weight_decay': 0.0},
    ])

    # train 5 epochs
    model.train()
    for epoch in range(1, CONFIG['epochs']+1):
        running, n = 0.0, 0
        for x,y,_ in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()*x.size(0); n += x.size(0)
        print(f"  Epoch {epoch}/{CONFIG['epochs']}  loss={running/max(n,1):.4f}")

    # evaluate & collect correctly predicted paths
    correct_paths = evaluate_and_collect(model, test_loader)
    print(f"  Correct predictions: {len(correct_paths)} / {len(test_ds)}")
    correct_paths_per_model[name] = correct_paths

    # keep trained weights in RAM (CPU) only; free GPU
    state_cpu = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    trained_states[name] = state_cpu
    del model, enc, head
    torch.cuda.empty_cache()

# -------------------- PICK COMMON CORRECT IMAGE --------------------
common = None
for k,v in correct_paths_per_model.items():
    common = v if common is None else (common & v)
if not common:
    raise RuntimeError("No test image was correctly predicted by all three models. Try another seed/epoch count.")
chosen_path = sorted(list(common))[0]
print(f"\nChosen image correctly predicted by all models:\n  {chosen_path}")

# -------------------- VISUALIZE ATTENTION FOR EACH TRAINED MODEL --------------------
def build_model_and_load(state_dict_cpu):
    enc = build_encoder(CONFIG).to(DEVICE)
    head = ClassificationHead(CONFIG['embed_dim'], CONFIG['n_classes']).to(DEVICE)
    model = nn.Sequential(enc, head).to(DEVICE)
    model.load_state_dict(state_dict_cpu, strict=False)
    model.eval()
    return model

@torch.no_grad()
def rollout_for_image(enc, img_path):
    x = trsf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    attn_list = collect_attn_softmax_from_qkv(enc, x)
    cls2patch = attention_rollout(attn_list, head_fuse='mean')[0]  # [N-1]
    return cls2patch

for strat in STRATEGIES:
    name = strat['name']
    model = build_model_and_load(trained_states[name])
    attn_vec = rollout_for_image(model[0], chosen_path)
    base = os.path.splitext(os.path.basename(chosen_path))[0]
    save_path = os.path.join('outputs', 'images', f'halo_{name}_{os.path.basename(chosen_path)}')
    green_halo_overlay(
        img_path=chosen_path,
        attn_in=attn_vec,  # whatever you cached (any common format)
        img_size=CONFIG['img_size'],
        patch_size=CONFIG['patch_size'],
        save_path=save_path,
        top_p=0.05,  # or top_percent=5
        blur_px=16,
        alpha=0.45,
        verbose=False
    )
    print(f"Saved: {save_path}")
    del model
    torch.cuda.empty_cache()

print("\nDone.")
