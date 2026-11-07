# -*- coding: utf-8 -*-
"""Training-free probe: the metrics in an OFF-THE-SHELF encoder's frame space (nothing trained on HHD, no CER needed).

Why: in model_aware.py the metrics predict TrOCR's CER only in the TRAINED TrOCR's own frame space (trocrctc own line Vendi+Dens
ratio 0.56), which is not training-free. The image-only vit_token granularity (metrics_hmt.py) already uses the off-the-shelf
TrOCR-base encoder, but as 6000 tokens drawn at random over all lines, not as reading-order frames, and it fails. This probe keeps
model_aware.py's exact feature extraction (frames = the 4 patch rows of a 16 px column concatenated = what the CTC head would read;
frame = 6000 sampled frames, line = per-line mean; PCA to 768 above 1024 dims, fitted on the dev reference; L2) and swaps ONLY the
weights for untrained ones:
  trocrbase-raw    microsoft/trocr-base-handwritten encoder as downloaded = trocrctc before fine-tuning on HHD
  trocrsmall-raw   microsoft/trocr-small-stage1 encoder as downloaded     = trocrsmall before fine-tuning
  deitsmall-rand   the same DeiT-small architecture, random weights (seed 42): control -- if this works, the space does not matter
  clip-b16-raw     openai/clip-vit-base-patch16 vision tower (384 px via position interpolation, CLIP normalisation): a generic
                   pretrained ViT that never saw handwriting-specific training
Same tiling as the TrOCR recognizers (TROCR_FMT), same candidate draws as metrics_hmt.py / model_aware.py (same rng), reference =
the clean dev lines.

    python probe_raw.py --shard i --nshards n     one worker per GPU; each job -> HF cyttic/hmt-metrics raw_probe/<probe>/<cond>_k<k>.json
    python probe_raw.py --analyze                 leave-one-family-out, every probe x every recognizer's CER -> raw_probe/summary.json
"""
import argparse, json, math, os, time

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import augment as A
import model_aware as MA

OUT_REPO, PREFIX, SEED = "cyttic/hmt-metrics", "raw_probe", 42
PROBES = {"trocrbase-raw": "trocrctc", "trocrsmall-raw": "trocrsmall", "deitsmall-rand": "trocrscratch", "clip-b16-raw": "clip"}
CONDS, DRAWS = MA.CONDS, MA.DRAWS
# recognizers whose test CER the probes are scored against (only those with all 17 conditions on HF are used)
TARGETS = ["trocrctc-s42", "trocrctc-s43", "trocrsmall-s42", "trocrscratch-s42", "trocrscratchreg-s42", "trocrscratchline-s42",
           "htrvt-s42", "htrvt-s43"]


def build_raw(src, n_classes):
    """The training notebooks' own model code (model_src.json), WITHOUT loading any HHD-trained weights."""
    S = json.load(open("model_src.json"))[src]
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "math": math, "os": os, "A": A, "SEED": SEED, "N_CLASSES": n_classes, "device": MA.DEV}
    for k in ("cfg", "fmt", "model"): exec(S[k], ns)
    ns["model"].eval()
    return ns


def work(shard, nshards, smoke):
    from huggingface_hub import HfApi, create_repo
    from metrics_hmt import score
    api = HfApi(); create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
    done = set(api.list_repo_files(OUT_REPO, repo_type="dataset"))
    jobs = [(p, c, k) for p in PROBES for c in CONDS for k in range(1 if c == "none" else DRAWS)]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard and f"{PREFIX}/{j[0]}/{j[1]}_k{j[2]}.json" not in done]
    print(f"shard {shard}/{nshards}: {len(mine)} of {len(jobs)} jobs to do", flush=True)
    if not mine: return
    chars, train, dev = MA.load_split()
    if smoke: train, dev = train[:96], dev[:48]
    dev_g = [g for g, _ in dev]
    print(f"train {len(train)} | dev {len(dev)}", flush=True)
    cur = None
    for p, c, k in mine:
        t0 = time.time()
        if cur != p:                                             # jobs are grouped by probe: load it once, embed the clean dev once
            ns = build_raw(PROBES[p], len(chars)); cur = p
            _, Rf, Rl = MA.run(ns, dev_g, chars, np.random.default_rng([SEED, 7]))
            proj = {"frame": MA.projector(Rf), "line": MA.projector(Rl)}
            R = {"frame": proj["frame"](Rf), "line": proj["line"](Rl)}
            print(f"[{p}] loaded | frame dim {Rf.shape[1]}", flush=True)
        fam, lvl = A.parse_condition(c); ci = CONDS.index(c)
        rng = np.random.default_rng([SEED, 1000 + k, ci])        # = metrics_hmt.py / model_aware.py: the same augmented training set
        cand = [A.augment(g, fam, lvl, rng) for g in train]
        _, Xf, Xl = MA.run(ns, cand, chars, np.random.default_rng([SEED, 8, ci, k]))
        res = dict(probe=p, condition=c, family=fam or "none", level=lvl, draw=k,
                   frame=score(proj["frame"](Xf), R["frame"]), line=score(proj["line"](Xl), R["line"]))
        path = f"{PREFIX}/{p}/{c}_k{k}.json"; os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(res, open(path, "w"), indent=1)
        for attempt in range(0 if smoke else 5):                # two workers commit to one repo + flaky network: retry
            try: api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=OUT_REPO, repo_type="dataset", commit_message=f"{p} {c} k{k}"); break
            except Exception as e: print(f"upload retry {attempt + 1}: {e}", flush=True); time.sleep(30)
        print(f"[{p} {c} k{k}] {time.time() - t0:.0f}s | Vendi frame/line {res['frame']['Vendi']:.4f}/{res['line']['Vendi']:.4f} | "
              f"KID line {res['line']['KID']:.4f}", flush=True)


def analyze():
    """For every (probe, recognizer): leave-one-family-out over frame/line x lofo.COMBOS, best ratio vs the 300-shuffle null in which
    the best combo is re-picked on every shuffle (lofo.py's protocol), so the combo search is paid for."""
    import glob, pandas as pd
    from huggingface_hub import HfApi, snapshot_download
    from scipy.stats import spearmanr
    import lofo as L
    os.makedirs(L.RES, exist_ok=True)
    d = snapshot_download(OUT_REPO, repo_type="dataset", allow_patterns=[f"{PREFIX}/*/*.json"])
    rows = [json.load(open(f)) for f in glob.glob(f"{d}/{PREFIX}/*/*.json") if not f.endswith("summary.json")]
    print(len(rows), "of", len(PROBES) * (1 + 16 * DRAWS), "jobs present")
    cer = L.fetch_cer()
    tg = [t for t in TARGETS if all(f"{t.rsplit('-', 1)[0]}-{c}-{t.rsplit('-', 1)[1]}" in cer for c in L.CONDS)]
    print("targets:", tg)
    out, rng = {}, np.random.default_rng(0)
    for p in PROBES:
        R = [r for r in rows if r["probe"] == p]
        if len({(r["condition"], r["draw"]) for r in R}) < 1 + 16 * DRAWS: print(p, "incomplete, skipped"); continue
        own = {g: pd.DataFrame({c: pd.DataFrame([r[g] for r in R if r["condition"] == c])[L.M].mean() for c in L.CONDS}).T for g in ("frame", "line")}
        for g in own: own[g]["Vendi_x_A"] = own[g].Vendi * own[g].A_dist
        cands = {f"{g} {'+'.join(c)}": own[g][list(c)].values for g in own for c in L.COMBOS}
        out[p] = {}
        print(f"\n== probe {p}")
        for t in tg:
            arch, s = t.rsplit("-", 1)
            y = 100 * np.array([cer[f"{arch}-{c}-{s}"] for c in L.CONDS])
            def ratios(yy):
                b = L.mae(L.lofo(np.zeros((17, 0)), yy), yy)
                return {k: L.mae(L.lofo(X, yy), yy) / b for k, X in cands.items()}
            obs = ratios(y); kb = min(obs, key=obs.get); pr = L.lofo(cands[kb], y)
            null = []
            for _ in range(300):
                yp = y.copy(); yp[L.aug] = rng.permutation(y[L.aug]); null.append(min(ratios(yp).values()))
            null = np.array(null)
            out[p][t] = {"best": kb, "ratio": obs[kb], "rho": float(spearmanr(pr[L.aug], y[L.aug])[0]),
                         "perm_median": float(np.median(null)), "perm_p": float((null <= obs[kb]).mean()),
                         "MAE_per_family": {f: float(np.abs(pr[L.fam == f] - y[L.fam == f]).mean()) for f in L.FAMS}}
            o = out[p][t]
            print(f"   {t:22s} {kb:24s} ratio {o['ratio']:.4f} rho {o['rho']:.4f} p {o['perm_p']:.4f} | per family "
                  + ", ".join(f"{f} {v:.2f}" for f, v in o["MAE_per_family"].items()))
    json.dump(out, open("raw_summary.json", "w"), indent=1)
    HfApi().upload_file(path_or_fileobj="raw_summary.json", path_in_repo=f"{PREFIX}/summary.json", repo_id=OUT_REPO, repo_type="dataset",
                        commit_message="raw-probe summary")
    print(f"-> {OUT_REPO}/{PREFIX}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else work(a.shard, a.nshards, a.smoke)
