# -*- coding: utf-8 -*-
"""Model-aware predictors of augmentation benefit (H2: why do the image-only metrics predict HTR-VT's CER but not TrOCR's?).

Each recognizer trained WITHOUT augmentation (cyttic/hmt-<arch>-none-s<seed>) is the probe; nothing is trained here.
  affinity  -- its greedy CER on the dev lines augmented with each condition (3 draws; Gontijo-Lopes et al., ICLR 2021), minus the
               clean-dev CER: how far the augmentation moves the data away from what this model already reads
  own       -- the training-free metrics of metrics_hmt.py (score(): Cover/Dens/Prec/Vendi/PairD/A_dist/KID), computed in the model's
               OWN feature space = the input of its CTC classifier. frame: 6000 valid frames sampled over all lines; line: the mean of
               each line's valid frames. Candidates = the training set augmented with exactly metrics_hmt.py's draws (same rng);
               reference = the clean dev lines. Vectors are L2-normalised; above 1024 dims (TrOCR frames = 4 patch rows x hidden)
               they are first PCA-reduced to 768 dims, fitted once on the model's dev reference.
Model and preprocessing code are the training notebooks' own (model_src.json, written by make_model_aware_notebook.py).

    python model_aware.py --shard i --nshards n     one worker per GPU; each finished job -> HF cyttic/hmt-metrics model_aware/
    python model_aware.py --analyze                 merge + leave-one-family-out test -> model_aware/summary.json
"""
import argparse, json, math, os, time

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import augment as A

OUT_REPO, PREFIX, SEED, CAP, ROW_H = "cyttic/hmt-metrics", "model_aware", 42, 6000, 64
DEV = "cuda:0"
PROBES = {"htrvt-s42": ("htrvt", 42), "htrvt-s43": ("htrvt", 43), "trocrctc-s42": ("trocrctc", 42),
          "trocrsmall-s42": ("trocrsmall", 42), "trocrscratch-s42": ("trocrscratch", 42)}
CONDS = ["none"] + [f"{f}{l}" for f in A.FAMILIES for l in A.LEVELS[1:]]
DRAWS = 3


def load_split():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    chars = json.load(open(hf_hub_download("cyttic/exp35-trocr-large-en-ctc-seqhead-allreal-synpunct", "charset.json")))["chars"]
    hhd = load_dataset("cyttic/hhd-optionb-split")["train"]
    tw = list(hhd["writer"]); wr = sorted(set(tw)); __import__("random").Random(42).shuffle(wr); dev_w = set(wr[:int(len(wr) * 0.20)])
    train = [A.to_line(hhd[i]["image"], ROW_H) for i, w in enumerate(tw) if w not in dev_w]   # same order as metrics_hmt.py
    dev = [(A.to_line(hhd[i]["image"], ROW_H), hhd[i]["text"]) for i, w in enumerate(tw) if w in dev_w]
    return chars, train, dev


def build(arch, seed, n_classes):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    S = json.load(open("model_src.json"))[arch]
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "math": math, "os": os, "A": A, "SEED": seed, "N_CLASSES": n_classes, "device": DEV}
    for k in ("cfg", "fmt", "model"): exec(S[k], ns)
    ns["model"].load_state_dict(load_file(hf_hub_download(f"cyttic/hmt-{arch}-none-s{seed}", "model.safetensors")))
    ns["model"].eval()
    return ns


def greedy(lp, n, chars):
    out, prev = [], 0
    for k in lp[:n].argmax(-1).tolist():
        if k != prev and k != 0: out.append(chars[k])
        prev = k
    return "".join(out)


@torch.no_grad()
def run(ns, grays, chars, rng, bs=32):
    """-> greedy hypotheses, sampled frames (<= CAP over all lines), per-line mean frames. Frames = the CTC classifier's input."""
    model = ns["model"]; head = model.classifier if hasattr(model, "classifier") else model.head; sub = ns.get("SUB", 1)
    per = -(-CAP // len(grays)) + 1; cap = {}; hyps, frames, means = [], [], []
    hk = head.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))
    for s in range(0, len(grays), bs):
        items = [ns["fmt"](g) for g in grays[s:s + bs]]
        px, n = ns["collate"]([(x, k, [], "") for x, k in items])[:2]
        with torch.autocast("cuda", dtype=torch.float16):
            lp = model(px.to(DEV), n.to(DEV))
        lp = lp.float().cpu().numpy(); X = cap["x"].float().cpu().numpy()
        for i in range(len(items)):
            k = int(n[i]); hyps.append(greedy(lp[i], k, chars)); f = X[i, :max(1, k // sub)]
            frames.append(f[rng.choice(len(f), min(per, len(f)), replace=False)]); means.append(f.mean(0))
    hk.remove()
    F_ = np.concatenate(frames)
    return hyps, (F_[rng.choice(len(F_), CAP, replace=False)] if len(F_) > CAP else F_), np.stack(means)


def cer(refs, hyps):
    import jiwer
    k = [i for i, r in enumerate(refs) if r.strip()]
    return float(jiwer.cer([refs[i] for i in k], [hyps[i] for i in k]))


def projector(R):
    """Fitted on the dev reference: PCA to 768 dims above 1024 dims, then L2 normalisation."""
    if R.shape[1] > 1024:
        mu = R.mean(0); V = np.linalg.svd(R - mu, full_matrices=False)[2][:768]
        f = lambda X: (X - mu) @ V.T
    else:
        f = lambda X: X
    return lambda X: (lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9))(f(X)).astype(np.float32)


def work(shard, nshards, smoke):
    from huggingface_hub import HfApi, create_repo
    from metrics_hmt import score
    api = HfApi(); create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
    done = set(api.list_repo_files(OUT_REPO, repo_type="dataset"))
    jobs = [(p, c, k) for p in PROBES for c in CONDS for k in range(1 if c == "none" else DRAWS)]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard and f"{PREFIX}/{j[0]}/{j[1]}_k{j[2]}.json" not in done]
    print(f"shard {shard}/{nshards}: {len(mine)} of {len(jobs)} jobs to do", flush=True)
    if not mine: return
    chars, train, dev = load_split()
    if smoke: train, dev = train[:96], dev[:48]
    refs, dev_g = [t for _, t in dev], [g for g, _ in dev]
    print(f"train {len(train)} | dev {len(dev)}", flush=True)
    cur = None
    for p, c, k in mine:
        t0 = time.time()
        if cur != p:                                             # jobs are grouped by probe: load it once, embed the clean dev once
            arch, seed = PROBES[p]; ns = build(arch, seed, len(chars)); cur = p
            h0, Rf, Rl = run(ns, dev_g, chars, np.random.default_rng([SEED, 7]))
            clean = cer(refs, h0); proj = {"frame": projector(Rf), "line": projector(Rl)}
            R = {"frame": proj["frame"](Rf), "line": proj["line"](Rl)}
            print(f"[{p}] loaded | clean dev CER {100 * clean:.2f} | frame dim {Rf.shape[1]}", flush=True)
        fam, lvl = A.parse_condition(c); ci = CONDS.index(c)
        rng = np.random.default_rng([SEED, 1000 + k, ci])        # = metrics_hmt.py: the same augmented training set
        cand = [A.augment(g, fam, lvl, rng) for g in train]
        _, Xf, Xl = run(ns, cand, chars, np.random.default_rng([SEED, 8, ci, k]))
        rng = np.random.default_rng([SEED, 2000 + k, ci])        # affinity: the dev lines, augmented
        hyp, _, _ = run(ns, [A.augment(g, fam, lvl, rng) for g in dev_g], chars, np.random.default_rng([SEED, 9]))
        res = dict(probe=p, condition=c, family=fam or "none", level=lvl, draw=k, clean_dev_cer=clean, aug_dev_cer=cer(refs, hyp),
                   frame=score(proj["frame"](Xf), R["frame"]), line=score(proj["line"](Xl), R["line"]))
        path = f"{PREFIX}/{p}/{c}_k{k}.json"; os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(res, open(path, "w"), indent=1)
        for attempt in range(0 if smoke else 5):                # two workers commit to one repo + flaky network: retry
            try: api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=OUT_REPO, repo_type="dataset", commit_message=f"{p} {c} k{k}"); break
            except Exception as e: print(f"upload retry {attempt + 1}: {e}", flush=True); time.sleep(30)
        print(f"[{p} {c} k{k}] {time.time() - t0:.0f}s | dev CER {100 * res['aug_dev_cer']:.2f} | Vendi frame/line "
              f"{res['frame']['Vendi']:.4f}/{res['line']['Vendi']:.4f} | KID line {res['line']['KID']:.4f}", flush=True)


def analyze():
    """Leave-one-family-out (lofo.py's protocol and permutation test) with three predictor sets per probe:
    affinity alone; own-feature metrics (frame/line x lofo.COMBOS); affinity + the fixed image-only combo (word Vendi+KID+Prec)."""
    import glob, pandas as pd
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from scipy.stats import spearmanr
    import lofo as L
    os.makedirs(L.RES, exist_ok=True)
    d = snapshot_download(OUT_REPO, repo_type="dataset", allow_patterns=[f"{PREFIX}/*/*.json", "metrics.csv"])
    rows = [json.load(open(f)) for f in glob.glob(f"{d}/{PREFIX}/*/*.json")]
    print(len(rows), "of", len(PROBES) * (1 + 16 * DRAWS), "jobs present")
    cer = L.fetch_cer()
    img = pd.read_csv(f"{d}/metrics.csv"); img = img[img.granularity == "word"].groupby("condition")[["Vendi", "KID", "Prec"]].mean().loc[L.CONDS]
    out, rng = {}, np.random.default_rng(0)
    for p, (arch, seed) in PROBES.items():
        R = [r for r in rows if r["probe"] == p]
        if len({(r["condition"], r["draw"]) for r in R}) < 1 + 16 * DRAWS: print(p, "incomplete, skipped"); continue
        aff = pd.Series({c: np.mean([r["aug_dev_cer"] - r["clean_dev_cer"] for r in R if r["condition"] == c]) for c in L.CONDS})
        own = {g: pd.DataFrame({c: pd.DataFrame([r[g] for r in R if r["condition"] == c])[L.M].mean() for c in L.CONDS}).T for g in ("frame", "line")}
        for g in own: own[g]["Vendi_x_A"] = own[g].Vendi * own[g].A_dist
        y = 100 * np.array([cer[f"{arch}-{c}-s{seed}"] for c in L.CONDS])
        cands = {"affinity": aff.values[:, None], "affinity+img word Vendi+KID+Prec": np.c_[aff.values, img.values]}
        cands.update({f"own {g} {'+'.join(c)}": own[g][list(c)].values for g in own for c in L.COMBOS})
        b0 = L.mae(L.lofo(np.zeros((17, 0)), y), y)
        def ratios(yy):
            b = L.mae(L.lofo(np.zeros((17, 0)), yy), yy)
            return {k: L.mae(L.lofo(X, yy), yy) / b for k, X in cands.items()}
        obs = ratios(y)
        groups = {"affinity": ["affinity"], "own": [k for k in cands if k.startswith("own")], "affinity+img": ["affinity+img word Vendi+KID+Prec"]}
        null = {g: [] for g in groups}
        for _ in range(300):
            yp = y.copy(); yp[L.aug] = rng.permutation(y[L.aug]); rr = ratios(yp)
            for g, ks in groups.items(): null[g].append(min(rr[k] for k in ks))
        res = {"baseline_MAE": b0, "affinity_rho_vs_cer": float(spearmanr(aff.values[L.aug], y[L.aug])[0])}
        for g, ks in groups.items():
            kb = min(ks, key=lambda k: obs[k]); pr = L.lofo(cands[kb], y)
            res[g] = {"best": kb, "ratio": obs[kb], "MAE": obs[kb] * b0, "rho": float(spearmanr(pr[L.aug], y[L.aug])[0]),
                      "perm_median": float(np.median(null[g])), "perm_p": float((np.array(null[g]) <= obs[kb]).mean())}
        out[p] = res
        print(f"\n== {p}: baseline MAE {b0:.2f} | rho(affinity, CER) {res['affinity_rho_vs_cer']:.4f}")
        for g in groups:
            r = res[g]; print(f"   {g:13s} {r['best']:40s} ratio {r['ratio']:.4f} rho {r['rho']:.4f} p {r['perm_p']:.4f}")
    json.dump(out, open("summary.json", "w"), indent=1)
    HfApi().upload_file(path_or_fileobj="summary.json", path_in_repo=f"{PREFIX}/summary.json", repo_id=OUT_REPO, repo_type="dataset",
                        commit_message="model-aware summary")
    print(f"-> {OUT_REPO}/{PREFIX}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else work(a.shard, a.nshards, a.smoke)
