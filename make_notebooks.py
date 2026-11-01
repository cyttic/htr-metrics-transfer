# -*- coding: utf-8 -*-
"""Generates the A5000 notebooks of the augmentation / training-free-metrics study (waves 1 and 2).

    /mnt/ssd2/cyttic/ml_env/bin/python make_notebooks.py
-> train_trocrctc.ipynb   TrOCR-base ViT encoder + one linear CTC classifier (the exp38/Option-B recipe), greedy decoding
   train_htrvt.ipynb      CNN + transformer encoder + CTC trained from scratch (our implementation after HTR-VT), greedy decoding
   queue_wave1.ipynb      2 models x (none + 4 families x 4 levels) = 34 runs, seed 42; each run = one papermill call with parameters
   train_trocrsmall.ipynb the same TrOCR-CTC recipe on the TrOCR-small STAGE-1 encoder (DeiT-small, synthetic pretraining only, no IAM)
   queue_wave2.ipynb      TrOCR-small-stage1 x 17 conditions (seed 42), then HTR-VT x 17 conditions with seed 43 = 34 runs

Both training notebooks are parameterised (papermill cell tag "parameters": COND, SEED, REPO_DIR, ...) and import augment.py from
the repo, the same file the metric computation uses. Data: cyttic/hhd-optionb-split (paragraph- and writer-disjoint). Train = Option-B
train minus the dev writers (20% of the train writers, seed 42, the same dev as the Option-B runs); the dev writers are the unlabeled
target sample for the metrics and are scored (greedy) every few epochs for a learning curve only; nothing is selected on them.
The test set is scored once, after the last epoch. No language model anywhere.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42,)
FAMILIES = ("geom", "elastic", "stroke", "photo")
CONDS = ["none"] + [f"{f}{l}" for f in FAMILIES for l in (1, 2, 3, 4)]


def nb(cells):
    out = []
    for kind, src, *tags in cells:
        c = {"cell_type": kind, "metadata": {"tags": tags} if tags else {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code": c.update(outputs=[], execution_count=None)
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


SETUP = r'''
!pip install -q "transformers==5.15.0" datasets jiwer opencv-python-headless safetensors
'''

IMPORTS = r'''
import os, sys, io, re, json, math, time, random, unicodedata
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, jiwer
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi, get_token, whoami
assert get_token() is not None, "run `hf auth login` on the server first"
print("HF user:", whoami()["name"])
sys.path.insert(0, os.path.expanduser(REPO_DIR))
import augment as A   # the SAME file the metric computation uses
FAMILY, LEVEL = A.parse_condition(COND)
device = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(device, torch.cuda.get_device_name(0) if device == "cuda" else "", "| amp", AMP_DTYPE, "| condition", COND, (FAMILY, LEVEL), "| seed", SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = os.path.abspath(f"run_{MODEL}_{COND}_s{SEED}"); os.makedirs(OUT, exist_ok=True)
OUT_REPO = f"cyttic/hmt-{MODEL}-{COND}-s{SEED}"   # public
print("OUT", OUT, "->", OUT_REPO)
'''

DATA = r'''
CHARS = json.load(open(hf_hub_download("cyttic/exp35-trocr-large-en-ctc-seqhead-allreal-synpunct", "charset.json")))["chars"]
assert CHARS[0] == ""   # class 0 = CTC blank; the same 198 classes as exp32-39 / Option B
C2I = {c: i for i, c in enumerate(CHARS) if i}; N_CLASSES = len(CHARS)
def encode(text): return [C2I[c] for c in text if c in C2I]
def greedy(lp, n):
    best = lp[:n].argmax(-1).tolist(); out, prev = [], 0
    for k in best:
        if k != prev and k != 0: out.append(CHARS[k])
        prev = k
    return "".join(out)

hhd = load_dataset("cyttic/hhd-optionb-split")
train_full, test = hhd["train"], hhd["test"]
tw = list(train_full["writer"]); wr = sorted(set(tw)); random.Random(42).shuffle(wr); dev_w = set(wr[:int(len(wr) * 0.20)])   # FIXED dev
dev = train_full.select([i for i, w in enumerate(tw) if w in dev_w])
train = train_full.select([i for i, w in enumerate(tw) if w not in dev_w])
assert not set(train["writer"]) & set(test["writer"]) and not set(train["paragraph"]) & set(test["paragraph"])
print(f"train {len(train)} lines / {len(set(train['writer']))} writers | dev {len(dev)} / {len(dev_w)} | test {len(test)} / {len(set(test['writer']))}")
if SMOKE:
    train, dev, test = train.select(range(64)), dev.select(range(32)), test.select(range(32)); EPOCHS, EVAL_EVERY = 2, 1

class LineDS(torch.utils.data.Dataset):
    """HF split -> (model input, CTC length, label ids, text). Augmentation (train only) on the 64 px grayscale line, then fmt()."""
    def __init__(self, split, aug): self.s, self.aug, self.rng = split, aug, np.random.default_rng(SEED)
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        ex = self.s[i]; g = A.to_line(ex["image"], ROW_H)
        if self.aug: g = A.augment(g, FAMILY, LEVEL, self.rng)
        x, n = fmt(g)
        return x, n, encode(ex["text"]), ex["text"]

def loader(split, train_mode, epoch=0):
    ds = LineDS(split, aug=train_mode and LEVEL > 0)
    def init(wid):   # a distinct, reproducible augmentation stream per (seed, epoch, worker)
        torch.utils.data.get_worker_info().dataset.rng = np.random.default_rng([SEED, epoch, wid])
    return torch.utils.data.DataLoader(ds, batch_size=BATCH if train_mode else EVAL_BS, shuffle=train_mode, num_workers=NUM_WORKERS,
                                       collate_fn=collate, worker_init_fn=init, drop_last=False)

def scores(R, H):
    """CER on the raw text, CER without punctuation (Unicode P* removed from both sides), WER. Empty references skipped."""
    nop = lambda t: re.sub(r" +", " ", "".join(c for c in t if not unicodedata.category(c).startswith("P"))).strip()
    k = [i for i, r in enumerate(R) if r.strip()]; k2 = [i for i in k if nop(R[i])]
    return {"cer": jiwer.cer([R[i] for i in k], [H[i] for i in k]), "wer": jiwer.wer([R[i] for i in k], [H[i] for i in k]),
            "cer_nopunct": jiwer.cer([nop(R[i]) for i in k2], [nop(H[i]) for i in k2])}

@torch.no_grad()
def predict(model, split):
    model.eval(); H, R = [], []
    for x, n, _, texts in loader(split, False):
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
            lp = model(x.to(device), n.to(device))
        lp = lp.float().cpu().numpy(); H += [greedy(lp[i], int(n[i])) for i in range(len(texts))]; R += texts
    return R, H
'''

TRAIN = r'''
def train_model(model, param_groups):
    opt = torch.optim.AdamW(param_groups, weight_decay=WD); base_lr = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and AMP_DTYPE == torch.float16)
    spe = math.ceil(len(train) / BATCH); total = EPOCHS * spe; warm = max(1, int(WARMUP * total)); step = 0; t0 = time.time(); curve = []
    log = open(os.path.join(OUT, "train_log.tsv"), "w"); log.write("epoch\tstep\ttrain_loss\tdev_cer\tdev_wer\tsec\n")
    print(f"{len(train)} train lines | {spe} steps/epoch | {EPOCHS} epochs = {total} steps | augmentation {COND}")
    for ep in range(1, EPOCHS + 1):
        model.train(); losses = []
        for x, n, labels, _ in loader(train, True, ep):
            keep = [i for i, l in enumerate(labels) if 0 < len(l) <= int(n[i])]
            if not keep: continue
            f = schedule(step, total, warm)
            for g, b in zip(opt.param_groups, base_lr): g["lr"] = b * f
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
                lp = model(x[keep].to(device), n[keep].to(device))
            lab = [labels[i] for i in keep]
            loss = F.ctc_loss(lp.float().transpose(0, 1), torch.tensor(sum(lab, [])), n[keep], torch.tensor([len(l) for l in lab]),
                              blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            losses.append(float(loss)); step += 1
        tl = float(np.mean(losses)); row = {"epoch": ep, "step": step, "train_loss": tl}
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            R, H = predict(model, dev); row.update({f"dev_{k}": v for k, v in scores(R, H).items()})
        curve.append(row)
        log.write(f"{ep}\t{step}\t{tl:.4f}\t{row.get('dev_cer', '')}\t{row.get('dev_wer', '')}\t{time.time()-t0:.0f}\n"); log.flush()
        print(f"[epoch {ep}/{EPOCHS}] loss {tl:.4f}" + (f" | dev CER {100*row['dev_cer']:.2f}" if "dev_cer" in row else "") +
              f" | {time.time()-t0:.0f}s", flush=True)
    return curve, time.time() - t0
'''

FINAL = r'''
curve, sec = train_model(model, PARAM_GROUPS)
R, H = predict(model, test); T = scores(R, H)
print("HHD Option-B TEST (greedy, last epoch):", {k: round(v, 4) for k, v in T.items()})
with open(os.path.join(OUT, "test_predictions.tsv"), "w", encoding="utf-8") as fh:
    fh.write("ref\thyp\n"); [fh.write(f"{r}\t{h}\n") for r, h in zip(R, H)]
results = {"model": MODEL, "condition": COND, "family": FAMILY, "level": LEVEL, "seed": SEED, "config": CONFIG,
           "augment_max": A.MAX.get(FAMILY), "train_lines": len(train), "dev_lines": len(dev), "test_lines": len(test),
           "train_hours": sec / 3600, "dev_curve": curve, "final": {"test_greedy": T}}
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
'''

PUSH_CELL = r'''
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(OUT, "model.safetensors"))
json.dump({"chars": CHARS}, open(os.path.join(OUT, "charset.json"), "w"), ensure_ascii=False)
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
    f"---\nlicense: mit\nlanguage:\n- he\ntags:\n- htr\n- ctc\n---\n\n# {OUT_REPO}\n\nAugmentation study (htr-metrics-transfer). "
    f"Model `{MODEL}`, augmentation `{COND}`, seed {SEED}. Trained on `cyttic/hhd-optionb-split` train minus the dev writers; "
    f"HHD Option-B test, greedy decoding, no LM: CER {T['cer']:.4f}, WER {T['wer']:.4f}. Details in results.json.\n")
if PUSH and not SMOKE:
    api = HfApi()
    for attempt in range(5):
        try:
            api.create_repo(OUT_REPO, private=False, exist_ok=True)
            api.upload_folder(repo_id=OUT_REPO, folder_path=OUT, commit_message=f"{MODEL} {COND} seed {SEED}"); break
        except Exception as e:
            print("upload failed, retrying:", repr(e)[:200]); time.sleep(30 * (attempt + 1))
    print("pushed -> https://huggingface.co/" + OUT_REPO)
else:
    print("not pushed; files in", OUT)
'''

# ------------------------------------------------------------------ TrOCR-CTC
TROCR_PARAMS = r'''
COND = "none"            # "none" or <family><level>, family in geom / elastic / stroke / photo, level 1..4 (augment.py)
SEED = 42
REPO_DIR = "~/htr-metrics-transfer"
PUSH, SMOKE = True, False
'''
TROCR_CFG = r'''
import os, math
MODEL = "__MODEL__"
CONFIG = {"enc": "__ENC__", "row_h": 64, "sub": 2, "batch": 32, "lr_enc": 5e-05, "lr_head": 3e-4, "warmup": 0.05,
          "wd": 0.01, "epochs": 40, "eval_every": 5, "schedule": "linear warm-up + linear decay"}
ENC_SRC, ROW_H, SUB, BATCH = CONFIG["enc"], CONFIG["row_h"], CONFIG["sub"], CONFIG["batch"]
LR_ENC, LR_HEAD, WARMUP, WD, EPOCHS, EVAL_EVERY = CONFIG["lr_enc"], CONFIG["lr_head"], CONFIG["warmup"], CONFIG["wd"], CONFIG["epochs"], CONFIG["eval_every"]
NUM_WORKERS, EVAL_BS = min(8, os.cpu_count()), 64
def schedule(step, total, warm): return min(1.0, (step + 1) / warm) * max(0.0, (total - step) / max(1, total - warm))
'''
TROCR_FMT = r'''
C, P = 384, 16; RROWS = C // ROW_H; MAX_COLS = RROWS * (C // P)
def fmt(g):
    """64 px gray line -> mirror (RTL -> LTR) -> wrapped into the 384x384 ViT square in rows of 384 px -> (3,384,384) in [-1,1] + CTC length"""
    g = g[:, ::-1]; nw = g.shape[1]; cont = np.full((C, C), 255, np.uint8); sx, r = 0, 0
    while sx < nw and r < RROWS:
        seg = g[:, sx:sx + C]; cont[r * ROW_H:(r + 1) * ROW_H, :seg.shape[1]] = seg; sx += seg.shape[1]; r += 1
    t = torch.from_numpy(cont).float().div(255).sub(0.5).div(0.5).unsqueeze(0).expand(3, C, C).contiguous()
    return t, min(MAX_COLS, math.ceil(min(nw, RROWS * C) / P)) * SUB
def collate(b):
    return torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b]), [x[2] for x in b], [x[3] for x in b]
'''
TROCR_MODEL = r'''
from transformers import VisionEncoderDecoderModel
class EncoderLinearCTC(nn.Module):
    """ViT patch tokens (24x24) -> reading-order frames (one per 16 px column, the ROW_H//16 patch rows of a text row concatenated)
    -> ONE Linear giving SUB CTC steps per column (exp38 / Option-B recipe)."""
    def __init__(self, vit):
        super().__init__(); self.vit = vit
        if getattr(self.vit, "pooler", None) is not None: self.vit.pooler = None
        self.Hd = vit.config.hidden_size; self.k = ROW_H // 16
        self.classifier = nn.Linear(self.k * self.Hd, SUB * N_CLASSES)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02); nn.init.zeros_(self.classifier.bias)
    def forward(self, px, lens=None):   # lens unused: every input is a full 384x384 square
        tok = self.vit(pixel_values=px).last_hidden_state[:, -576:]; B = tok.shape[0]   # the 24x24 patch tokens (ViT: after CLS; DeiT: after CLS + distillation)
        g = tok.reshape(B, 24, 24, self.Hd)[:, :RROWS * self.k]
        g = g.reshape(B, RROWS, self.k, 24, self.Hd).permute(0, 1, 3, 2, 4).reshape(B, RROWS * 24, self.k * self.Hd)
        return F.log_softmax(self.classifier(g).reshape(B, RROWS * 24 * SUB, N_CLASSES).float(), dim=-1)
torch.manual_seed(SEED)
model = EncoderLinearCTC(VisionEncoderDecoderModel.from_pretrained(ENC_SRC).encoder).to(device)
PARAM_GROUPS = [{"params": list(model.vit.parameters()), "lr": LR_ENC}, {"params": list(model.classifier.parameters()), "lr": LR_HEAD}]
print(f"encoder {sum(p.numel() for p in model.vit.parameters())/1e6:.1f}M | classifier {sum(p.numel() for p in model.classifier.parameters())/1e6:.2f}M")
'''

# ------------------------------------------------------------------ HTR-VT
HTRVT_PARAMS = TROCR_PARAMS
HTRVT_CFG = r'''
import os, math
MODEL = "htrvt"
CONFIG = {"row_h": 64, "max_w": 2048, "batch": 32, "lr": 5e-4, "warmup": 0.05, "wd": 0.05, "epochs": 300, "eval_every": 10,
          "cnn_width": 256, "dim": 384, "depth": 6, "heads": 6, "mlp_ratio": 4, "dropout": 0.1, "mask_ratio": 0.4, "mask_span": 4,
          "schedule": "linear warm-up + cosine decay to 1% of the peak"}
ROW_H, MAX_W, BATCH = CONFIG["row_h"], CONFIG["max_w"], CONFIG["batch"]
LR, WARMUP, WD, EPOCHS, EVAL_EVERY = CONFIG["lr"], CONFIG["warmup"], CONFIG["wd"], CONFIG["epochs"], CONFIG["eval_every"]
NUM_WORKERS, EVAL_BS = min(8, os.cpu_count()), 32
def schedule(step, total, warm):
    if step < warm: return (step + 1) / warm
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
'''
HTRVT_FMT = r'''
STRIDE = 4   # the CNN reduces width 4x -> one CTC step per 4 px
def fmt(g):
    """64 px gray line -> mirror (RTL -> LTR) -> width capped at MAX_W -> (1,64,W) in [-1,1] (ink dark) + CTC length"""
    g = g[:, ::-1]
    if g.shape[1] > MAX_W: g = A.cv2.resize(np.ascontiguousarray(g), (MAX_W, ROW_H), interpolation=A.cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(g)).float().div(255).sub(0.5).div(0.5).unsqueeze(0)
    return t, max(1, t.shape[-1] // STRIDE)
def collate(b):
    W = max(x[0].shape[-1] for x in b); W = int(math.ceil(W / STRIDE) * STRIDE)
    px = torch.ones(len(b), 1, ROW_H, W)   # pad with white (+1)
    for i, x in enumerate(b): px[i, :, :, :x[0].shape[-1]] = x[0]
    return px, torch.tensor([x[1] for x in b]), [x[2] for x in b], [x[3] for x in b]
'''
HTRVT_MODEL = r'''
class ResBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout)
        self.sc = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.pool = nn.MaxPool2d(pool) if pool != (1, 1) else nn.Identity()
    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return self.pool(F.relu(y + self.sc(x)))

class HTRVT(nn.Module):
    """Our implementation after HTR-VT (Li et al., Pattern Recognition 2025): a residual CNN turns the 64 px line into a feature
    sequence (height 64 -> 4, folded into channels; width / 4), span masking on that sequence during training, sinusoidal positions,
    a pre-norm transformer encoder, a linear CTC head. Trained from scratch."""
    def __init__(self, n_classes, cnn_width, dim, depth, heads, mlp_ratio, dropout, mask_ratio, mask_span, **_):
        super().__init__()
        w1, w2, w3 = cnn_width // 4, cnn_width // 2, cnn_width
        self.cnn = nn.Sequential(nn.Conv2d(1, w1, 3, padding=1, bias=False), nn.BatchNorm2d(w1), nn.ReLU(),
                                 ResBlock(w1, w1, (2, 2)), ResBlock(w1, w2, (2, 2)), ResBlock(w2, w2, (2, 1)),
                                 ResBlock(w2, w3, (2, 1)), ResBlock(w3, w3, (1, 1)))
        self.proj = nn.Linear(w3 * 4, dim); self.mask_ratio, self.mask_span = mask_ratio, mask_span
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, n_classes)
        pe = torch.zeros(4096, dim); pos = torch.arange(4096).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); self.register_buffer("pe", pe, persistent=False)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
    def span_mask(self, x):
        B, T, _ = x.shape; n = max(1, int(T * self.mask_ratio / self.mask_span))
        st = torch.randint(0, max(1, T - self.mask_span), (B, n), device=x.device); m = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for o in range(self.mask_span): m.scatter_(1, (st + o).clamp(max=T - 1), True)
        return x.masked_fill(m[..., None], 0.0)
    def forward(self, px, lens=None):   # lens = CTC steps per line; padded steps are masked out of attention
        f = self.cnn(px); B, Cc, Hh, W = f.shape
        x = self.proj(f.permute(0, 3, 1, 2).reshape(B, W, Cc * Hh))
        if self.training and self.mask_ratio > 0: x = self.span_mask(x)
        pad = None if lens is None else torch.arange(W, device=x.device)[None, :] >= lens[:, None]
        x = self.enc(self.drop(x + self.pe[:W]), src_key_padding_mask=pad)
        return F.log_softmax(self.head(self.norm(x)).float(), dim=-1)

torch.manual_seed(SEED)
model = HTRVT(N_CLASSES, **CONFIG).to(device)
PARAM_GROUPS = [{"params": list(model.parameters()), "lr": LR}]
print(f"HTR-VT (ours) {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, trained from scratch")
'''


def training_nb(title, desc, params, cfg, fmt_src, model_src):
    return nb([
        ("markdown", f"# {title}\n\n{desc}\n\nParameters (papermill): `COND`, `SEED`, `REPO_DIR`, `PUSH`, `SMOKE`. "
                     "Output: public HF repo `cyttic/hmt-<model>-<COND>-s<SEED>` (results.json, train_log.tsv, test_predictions.tsv, weights)."),
        ("code", params, "parameters"),
        ("code", SETUP), ("code", cfg), ("code", IMPORTS), ("code", fmt_src), ("code", DATA), ("code", model_src),
        ("code", TRAIN), ("code", FINAL), ("code", PUSH_CELL)])


COMMON = ("Augmentation study, wave 1. Data `cyttic/hhd-optionb-split`: train = Option-B train minus the dev writers (20%, seed 42); "
          "online augmentation `COND` from `augment.py`; dev scored every few epochs (learning curve only); the HHD Option-B test scored once "
          "after the last epoch, **greedy CTC decoding, no language model** (CER, CER without punctuation, WER).")
TRO = training_nb("TrOCR-CTC (pretrained ViT + linear CTC head), greedy",
                  "TrOCR-base handwritten ViT encoder + ONE linear CTC classifier (exp38 / Option-B recipe), 40 epochs fixed "
                  "(the epoch count the Option-B HHD-only runs selected). " + COMMON, TROCR_PARAMS,
                  TROCR_CFG.replace("__MODEL__", "trocrctc").replace("__ENC__", "microsoft/trocr-base-handwritten"), TROCR_FMT, TROCR_MODEL)
TRS = training_nb("TrOCR-small stage 1 + linear CTC head, greedy",
                  "Encoder of `microsoft/trocr-small-stage1` (DeiT-small, 384-dim, pretrained on large synthetic text-line data only, "
                  "NOT fine-tuned on IAM) + ONE linear CTC classifier: exactly the TrOCR-CTC recipe (same tiling, learning rates, 40 epochs), "
                  "only the encoder differs. " + COMMON, TROCR_PARAMS,
                  TROCR_CFG.replace("__MODEL__", "trocrsmall").replace("__ENC__", "microsoft/trocr-small-stage1"), TROCR_FMT, TROCR_MODEL)
HTV = training_nb("HTR-VT (CNN + transformer + CTC, from scratch), greedy",
                  "Our implementation after HTR-VT (Li et al., Pattern Recognition 2025): residual CNN front-end, span masking, 6-layer "
                  "transformer encoder (dim 384), linear CTC head, trained from scratch, 300 epochs fixed (dev curve logged to check the budget). "
                  + COMMON, HTRVT_PARAMS, HTRVT_CFG, HTRVT_FMT, HTRVT_MODEL)

def queue_nb(wave, runs, intro):
    src = r"""
import os, json, time, traceback, shutil
import papermill as pm
from huggingface_hub import HfApi, hf_hub_download
WAVE = "__WAVE__"
RUNS = __RUNS__   # (model, condition, seed)
NB = {"trocrctc": "train_trocrctc.ipynb", "trocrsmall": "train_trocrsmall.ipynb", "htrvt": "train_htrvt.ipynb"}
HERE = os.getcwd(); BASE = os.path.expanduser(f"~/work/hmt_{WAVE}"); os.makedirs(BASE, exist_ok=True)
api = HfApi(); LOGS = "cyttic/a5000-queue-logs"   # PUBLIC dataset: executed notebooks of failed runs (tracebacks)
api.create_repo(LOGS, repo_type="dataset", private=False, exist_ok=True)
def repo(m, c, s): return f"cyttic/hmt-{m}-{c}-s{s}"
def done(r):
    try: return "results.json" in api.list_repo_files(r)
    except Exception: return False
def push_log(path, name):
    try: api.upload_file(path_or_fileobj=path, path_in_repo=f"{name}.ipynb", repo_id=LOGS, repo_type="dataset", commit_message=f"failed run {name}"); print("log ->", f"{LOGS}/{name}.ipynb", flush=True)
    except Exception as e: print("log upload failed", repr(e)[:200], flush=True)
status = {}
for i, (m, c, s) in enumerate(RUNS, 1):
    r, name = repo(m, c, s), f"hmt_{m}_{c}_s{s}"
    if done(r): print(f"[{i}/{len(RUNS)}] {name}: already on HF, skipped", flush=True); status[name] = "skipped (done)"; continue
    wd = os.path.join(BASE, name); os.makedirs(wd, exist_ok=True); t0 = time.time()
    print(f"[{i}/{len(RUNS)}] {name} -> {r}  started {time.strftime('%H:%M')}", flush=True)
    try:
        pm.execute_notebook(os.path.join(HERE, NB[m]), os.path.join(wd, name + ".ipynb"), cwd=wd, log_output=True, kernel_name="python3",
                            parameters={"COND": c, "SEED": s, "REPO_DIR": HERE})
        status[name] = f"ok ({(time.time() - t0) / 3600:.2f} h)" if done(r) else "finished, but results.json not on HF"
        if done(r): shutil.rmtree(wd, ignore_errors=True)   # pushed: free the disk
    except Exception as e:
        status[name] = "FAILED: " + repr(e)[:300]; traceback.print_exc(); push_log(os.path.join(wd, name + ".ipynb"), f"hmt_{WAVE}__" + name)
    print(f"[{i}/{len(RUNS)}] {name}: {status[name]}", flush=True)
print("\n==== queue summary"); [print(f"{k:45s} {v}") for k, v in status.items()]
""".replace("__WAVE__", wave).replace("__RUNS__", json.dumps(runs))
    res = r"""
for m, c, s in RUNS:
    r = repo(m, c, s)
    if not done(r): print(f"{r}: no results"); continue
    T = json.load(open(hf_hub_download(r, "results.json", force_download=True)))["final"]["test_greedy"]
    print(f"{r:44s} CER {T['cer']:.4f} | WER {T['wer']:.4f}")
"""
    return nb([("markdown", f"# Queue: augmentation study, {wave} — {len(runs)} runs\n\n{intro}\n\nEach run pushes to its own public HF repo "
                            "`cyttic/hmt-<model>-<cond>-s<seed>`; a run whose repo already holds `results.json` is skipped, so re-running resumes "
                            "the queue. A failed run's executed notebook goes to the public HF dataset `cyttic/a5000-queue-logs`. The local run "
                            f"folder is deleted after a successful push.\n\nStart on the server: `./run.sh queue_{wave}.ipynb` (inside tmux)."),
               ("code", src), ("markdown", "## Results on HF"), ("code", res)])


RUNS1 = [("trocrctc", "none", 42), ("htrvt", "none", 42)] + [("trocrctc", c, 42) for c in CONDS[1:]] + [("htrvt", c, 42) for c in CONDS[1:]]
RUNS2 = [("trocrsmall", c, 42) for c in CONDS] + [("htrvt", c, 43) for c in CONDS]
QUEUE = queue_nb("wave1", RUNS1, "Models: TrOCR-CTC (`train_trocrctc.ipynb`) and HTR-VT from scratch (`train_htrvt.ipynb`), seed 42. Conditions: "
                 "`none` + 4 families (geom, elastic, stroke, photo) x levels 1-4 (`augment.py`). Order: both baselines, then all TrOCR-CTC runs, "
                 "then all HTR-VT runs.")
QUEUE2 = queue_nb("wave2", RUNS2, "First TrOCR-small stage 1 (`train_trocrsmall.ipynb`, seed 42, all 17 conditions, ~10-15 min each), then "
                  "HTR-VT (`train_htrvt.ipynb`) with seed 43 on all 17 conditions (~2.2 h each): the replication of the wave-1 HTR-VT runs.")

for name, obj in (("train_trocrctc.ipynb", TRO), ("train_trocrsmall.ipynb", TRS), ("train_htrvt.ipynb", HTV),
                  ("queue_wave1.ipynb", QUEUE), ("queue_wave2.ipynb", QUEUE2)):
    json.dump(obj, open(os.path.join(HERE, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", os.path.join(HERE, name))
print(len(RUNS1), "runs in wave 1 |", len(RUNS2), "runs in wave 2")
