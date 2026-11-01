# -*- coding: utf-8 -*-
"""Training-free metrics of every wave-1 augmentation condition, against the dev reference.

For each condition (none + 4 families x 4 levels) and draw k (K draws; `none` has no randomness -> 1 draw) the whole
training set (Option-B train minus the 20% dev writers, exactly the lines the recognizers were trained on) is augmented
once with augment.py and embedded; every metric is computed against the un-augmented dev lines (the reference).

Same protocol as the paper scorer (scorer-ocrs-dataset / paper_v3/analysis/score_all.py), code ported unchanged:
  granularities  patch (Otsu, 64 px, <=3 square windows/line), word (gap segmentation, gap calibrated on 300 train lines),
                 line (ink crop, pad to square), vit_token (TrOCR-base encoder tokens on the 384 tiled container)
  encoder        DINOv2-small (vit_small_patch14_dinov2.lvd142m, 224) for patch/word/line; TrOCR-base-handwritten for vit_token
  metrics        coverage, density, precision (k=5), Vendi + mean pairwise distance (2000-vector subsample),
                 Vendi SD over 10 subsamples, proxy A-distance (5-fold logistic regression), KID x1000 (10 x 1000)
  caps           every candidate granularity capped at 6000 vectors (seed 42); vit_token draws 6000 tokens from all lines

One JSON per (condition, draw) is pushed to HF dataset OUT_REPO/per/; a run skips files already there (resumable).
Usage: python metrics_hmt.py --shard i --nshards n [--draws 3] [--smoke]
"""
import argparse, io, json, os, sys, time

import numpy as np
import torch
from PIL import Image, ImageOps

import augment as A

OUT_REPO = "cyttic/hmt-metrics"
SEED, M_CAP, TOKEN_CAP, CHUNK_LINES, N_CALIB, ROW_H = 42, 6000, 6000, 100, 300, 64
DINO = "vit_small_patch14_dinov2.lvd142m"
TROCR = "microsoft/trocr-base-handwritten"
DEV = "cuda:0"


# ---------------------------------------------------------------- preprocessing (compare_datasets / common.py)
def otsu_binarize(img):
    a = np.asarray(img)
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    w = hist.cumsum(); m = (hist * np.arange(256)).cumsum()
    w1 = w[-1] - w; mu0 = np.where(w > 0, m / np.maximum(w, 1e-9), 0)
    mu1 = np.where(w1 > 0, (m[-1] - m) / np.maximum(w1, 1e-9), 0)
    t = int(np.argmax(w * w1 * (mu0 - mu1) ** 2))
    return Image.fromarray(np.where(a <= t, 0, 255).astype(np.uint8))


def patches(img, side, max_p=3, min_ink=0.01):
    a = np.asarray(img); ys, xs = np.where(a < 200)
    if len(ys) < 10: return []
    m = 4
    img = img.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(img.width, xs.max() + 1 + m), min(img.height, ys.max() + 1 + m)))
    w, h = img.size; nw = max(side, int(w * side / h)); img = img.resize((nw, side), Image.LANCZOS)
    xs = [0] if nw <= side else list(np.linspace(0, nw - side, min(max_p, 1 + nw // side)).astype(int))
    res = []
    for x in xs:
        p = img.crop((x, 0, x + side, side))
        if (np.asarray(p) < 200).mean() >= min_ink: res.append(p.convert("RGB"))
    return res


def resize_to_h(img, h=64):
    w0, h0 = img.size
    return img if h0 == h else img.resize((max(1, int(w0 * h / h0)), h), Image.LANCZOS)


def ink_crop(img, margin=4):
    a = np.asarray(img); ys, xs = np.where(a < 200)
    if len(ys) < 10: return None
    return img.crop((max(0, xs.min() - margin), max(0, ys.min() - margin), min(img.width, xs.max() + 1 + margin), min(img.height, ys.max() + 1 + margin)))


def pad_to_square_resize(img, target=224, bg=255):
    w, h = img.size; s = target / max(w, h); nw, nh = max(1, int(w * s)), max(1, int(h * s))
    c = Image.new("L", (target, target), bg); c.paste(img.resize((nw, nh), Image.LANCZOS), ((target - nw) // 2, (target - nh) // 2))
    return c


def resize_h64_pad(img, target_h=64, canvas=224, bg=255):
    w, h = img.size; nw = max(1, int(w * target_h / h)); r = img.resize((nw, target_h), Image.LANCZOS)
    out = Image.new("L", (canvas, canvas), bg); x0 = max(0, (canvas - nw) // 2)
    if nw > canvas:
        cx0 = (nw - canvas) // 2; r = r.crop((cx0, 0, cx0 + canvas, target_h)); x0 = 0
    out.paste(r, (x0, (canvas - target_h) // 2))
    return out


def segment_words(img, gap_thresh):
    b = otsu_binarize(resize_to_h(img, 64)); ink = (np.asarray(b) < 128).any(axis=0)
    segs, in_word, gap, start, W = [], False, 0, 0, len(ink)
    for x in range(W):
        if ink[x]:
            if not in_word: start, in_word = x, True
            gap = 0
        elif in_word:
            gap += 1
            if gap > gap_thresh: segs.append((start, x - gap + 1)); in_word, gap = False, 0
    if in_word: segs.append((start, W - gap if gap else W))
    return [(s, e) for s, e in segs if e > s], b


def crop_word(b, x0, x1, margin=3):
    a = np.asarray(b); ys = np.where((a[:, max(0, x0 - margin):min(a.shape[1], x1 + margin)] < 128).any(axis=1))[0]
    y0, y1 = (0, a.shape[0]) if len(ys) == 0 else (max(0, ys.min() - margin), min(a.shape[0], ys.max() + 1 + margin))
    return b.crop((max(0, x0 - margin), y0, min(a.shape[1], x1 + margin), y1))


def calibrate_gap(lines):
    med_words = np.median([len(t.split()) for _, t in lines]); best = None
    for g in range(1, 41):
        d = abs(np.median([len(segment_words(im, g)[0]) for im, _ in lines]) - med_words)
        if best is None or d < best[0]: best = (d, g)
    return best[1]


# ---------------------------------------------------------------- embedders
class Dino:
    def __init__(self):
        import timm
        from timm.data import resolve_data_config, create_transform
        self.model = timm.create_model(DINO, pretrained=True, num_classes=0, img_size=224).to(DEV).eval()
        cfg = resolve_data_config({}, model=self.model); cfg["input_size"] = (3, 224, 224)
        self.tf, self.side = create_transform(**cfg), 224

    @torch.no_grad()
    def __call__(self, imgs, bs=64):
        feats = []
        for s in range(0, len(imgs), bs):
            x = torch.stack([self.tf(im.convert("RGB")) for im in imgs[s:s + bs]]).to(DEV)
            feats.append(self.model(x).float().cpu().numpy())
        E = np.concatenate(feats) if feats else np.zeros((0, 384), np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def tile384(img):
    """HebrewBlockProcessor with AUTOCROP off: mirror (RTL->LTR), 64 px high, tiled row by row into 384x384, [-1, 1]."""
    img = ImageOps.mirror(img.convert("RGB")); w, h = img.size; nw = max(1, round(w * 64 / h))
    a = np.array(img.resize((nw, 64), Image.LANCZOS)); c = Image.new("RGB", (384, 384), (255, 255, 255))
    sx = dx = dy = 0
    while sx < nw and dy < 384:
        cw = min(nw - sx, 384 - dx); c.paste(Image.fromarray(a[:, sx:sx + cw]), (dx, dy)); sx += cw; dx += cw
        if dx >= 384: dx, dy = 0, dy + 64
    return (torch.tensor(np.array(c), dtype=torch.float32).permute(2, 0, 1) / 255.0 - 0.5) / 0.5


class TrocrTokens:
    def __init__(self):
        from transformers import VisionEncoderDecoderModel
        self.enc = VisionEncoderDecoderModel.from_pretrained(TROCR).encoder.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, imgs, bs=16):
        feats = []
        for s in range(0, len(imgs), bs):
            tok = self.enc(torch.stack([tile384(im) for im in imgs[s:s + bs]]).to(DEV)).last_hidden_state[:, 1:, :]
            feats.append(tok.reshape(-1, tok.shape[-1]).float().cpu().numpy())
        E = np.concatenate(feats) if feats else np.zeros((0, 768), np.float32)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def token_sample(vit, imgs, seed=SEED):
    rng = np.random.default_rng(seed); nch = -(-len(imgs) // CHUNK_LINES); per = -(-TOKEN_CAP // nch); out = []
    for c in range(nch):
        e = vit(imgs[c * CHUNK_LINES:(c + 1) * CHUNK_LINES]); out.append(e[rng.choice(len(e), min(per, len(e)), replace=False)])
    X = np.concatenate(out)
    return X[rng.choice(len(X), TOKEN_CAP, replace=False)] if len(X) > TOKEN_CAP else X


def cap(E, m=M_CAP, seed=SEED):
    return E if len(E) <= m else E[np.random.default_rng(seed).choice(len(E), m, replace=False)]


def embed_all(lines, dino, vit, gap):
    """lines: list of (PIL L image, text). Returns {granularity: vectors}, word retention."""
    imgs = [im for im, _ in lines]
    P = [p for im in imgs for p in patches(otsu_binarize(resize_to_h(im, 64)), dino.side)]
    W, kept = [], 0
    for im, t in lines:
        segs, b = segment_words(im, gap)
        if len(segs) != len(t.split()): continue
        crops = [crop_word(b, x0, x1) for x0, x1 in segs]
        if any(c.width < 2 or c.height < 2 for c in crops): continue
        kept += 1; W += [resize_h64_pad(c) for c in crops]
    L = [pad_to_square_resize(c) for c in (ink_crop(otsu_binarize(im)) for im in imgs) if c is not None]
    X = {"patch": cap(dino(P)), "word": cap(dino(W)), "line": cap(dino(L)), "vit_token": token_sample(vit, imgs)}
    return X, kept / len(lines)


# ---------------------------------------------------------------- metrics (compare_datasets.py, unchanged)
def kid(X, Y, subset=1000, reps=10, seed=0):
    d = X.shape[1]; rng = np.random.default_rng(seed); vals = []
    k = lambda a, b: (a @ b.T / d + 1.0) ** 3
    for _ in range(reps):
        x = X[rng.choice(len(X), min(subset, len(X)), replace=False)]; y = Y[rng.choice(len(Y), min(subset, len(Y)), replace=False)]
        m, n = len(x), len(y); kxx, kyy, kxy = k(x, x), k(y, y), k(x, y); np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)) * 1000, float(np.std(vals)) * 1000


def coverage_density(X, R, k=5):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances
    radii = NearestNeighbors(n_neighbors=k + 1).fit(R).kneighbors(R)[0][:, -1]
    d = pairwise_distances(R, X); within = d < radii[:, None]
    return float((d.min(axis=1) < radii).mean()), float(within.sum(axis=0).mean() / k), float(within.any(axis=0).mean())


def vendi(X, cap_=2000, seed=0):
    x = X[np.random.default_rng(seed).choice(len(X), min(cap_, len(X)), replace=False)]
    ev = np.clip(np.linalg.eigvalsh(x @ x.T / len(x)), 1e-12, None); ev /= ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


def mean_pairwise(X, cap_=2000, seed=0):
    x = X[np.random.default_rng(seed).choice(len(X), min(cap_, len(X)), replace=False)]
    S = x @ x.T; n = len(x)
    return float(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def a_distance(X, R, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    n = min(len(X), len(R)); rng = np.random.default_rng(seed)
    Z = np.concatenate([X[rng.choice(len(X), n, replace=False)], R[rng.choice(len(R), n, replace=False)]])
    acc = cross_val_score(LogisticRegression(max_iter=2000), Z, np.r_[np.zeros(n), np.ones(n)], cv=5).mean()
    return float(2 * (2 * acc - 1))


def score(X, R):
    cov, dens, prec = coverage_density(X, R); km, ks = kid(X, R)
    return dict(Cover=cov, Dens=dens, Prec=prec, Vendi=vendi(X), Vendi_sd=float(np.std([vendi(X, seed=s) for s in range(10)])),
                PairD=mean_pairwise(X), A_dist=a_distance(X, R), KID=km, KID_sd=ks, n=int(len(X)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--draws", type=int, default=3); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo
    api = HfApi(); create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
    done = {f for f in api.list_repo_files(OUT_REPO, repo_type="dataset")}

    hhd = load_dataset("cyttic/hhd-optionb-split")["train"]
    tw = list(hhd["writer"]); wr = sorted(set(tw)); __import__("random").Random(42).shuffle(wr); dev_w = set(wr[:int(len(wr) * 0.20)])
    to = lambda ex: (Image.fromarray(A.to_line(ex["image"], ROW_H)), ex["text"])
    train = [to(hhd[i]) for i, w in enumerate(tw) if w not in dev_w]          # (64 px L image, text): what the models trained on
    dev = [to(hhd[i]) for i, w in enumerate(tw) if w in dev_w]
    if a.smoke: train, dev = train[:120], dev[:60]
    print(f"train {len(train)} lines | dev {len(dev)} lines (reference)", flush=True)

    conds = ["none"] + [f"{f}{l}" for f in A.FAMILIES for l in A.LEVELS[1:]]
    jobs = [(c, k) for c in conds for k in range(1 if c == "none" else a.draws)]
    mine = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard and f"per/{j[0]}_k{j[1]}.json" not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(jobs)} jobs to do", flush=True)
    if not mine: return

    dino, vit = Dino(), TrocrTokens()
    gap = calibrate_gap(train[:N_CALIB])
    t0 = time.time(); R, ret_dev = embed_all(dev, dino, vit, gap)
    print(f"gap {gap} | reference sizes { {g: len(v) for g, v in R.items()} } | {time.time() - t0:.0f}s", flush=True)

    for c, k in mine:
        t0 = time.time(); fam, lvl = A.parse_condition(c); rng = np.random.default_rng([SEED, 1000 + k, conds.index(c)])
        cand = [(Image.fromarray(A.augment(np.asarray(im), fam, lvl, rng)), t) for im, t in train]
        X, ret = embed_all(cand, dino, vit, gap)
        res = dict(condition=c, family=fam or "none", level=lvl, draw=k, n_lines=len(cand), word_retention=ret,
                   dev_word_retention=ret_dev, gap_thresh=int(gap), protocol=dict(encoder=DINO, vit_token=TROCR, M_CAP=M_CAP,
                   TOKEN_CAP=TOKEN_CAP, seed=SEED, reference="dev writers (20% of Option-B train, seed 42), un-augmented"),
                   **{g: score(X[g], R[g]) for g in ("patch", "word", "line", "vit_token")})
        os.makedirs("per", exist_ok=True); json.dump(res, open(f"per/{c}_k{k}.json", "w"), indent=1)   # local copy too
        for attempt in range(5 if not a.smoke else 0):                    # two shards commit to one repo + flaky network: retry
            try:
                api.upload_file(path_or_fileobj=f"per/{c}_k{k}.json", path_in_repo=f"per/{c}_k{k}.json",
                                repo_id=OUT_REPO, repo_type="dataset", commit_message=f"{c} draw {k}"); break
            except Exception as e:
                print(f"upload retry {attempt + 1}: {e}", flush=True); time.sleep(30)
        print(f"[{c} k{k}] {time.time() - t0:.0f}s  Vendi p/w/l/t " + " ".join(f"{res[g]['Vendi']:.4f}" for g in ("patch", "word", "line", "vit_token"))
              + f"  KID line {res['line']['KID']:.4f}  ret {ret:.4f}", flush=True)


if __name__ == "__main__":
    main()
