# -*- coding: utf-8 -*-
"""Leave-one-family-out test: can training-free image metrics predict the test CER of an augmentation they were not fitted on?

CER  = final.test_greedy.cer of every HF run cyttic/hmt-<model>-<cond>-s<seed> (fetched -> results/cer.json).
X    = results/metrics.csv (metrics_hmt.py on Kaggle, cyttic/hmt-metrics; image-only, so the same for every model), mean over draws,
       per granularity (patch / word / line / vit_token); combos: Vendi alone, Vendi + one other metric, two triples, Vendi x A_dist.
LOFO = for each family (geom / elastic / stroke / photo): OLS on z-scored X fitted on the other 3 families + 'none', predict its 4 levels.
Scores on the 16 augmented conditions: MAE (CER points) and Spearman rho (prediction vs CER). Baselines: mean of the other families;
strength level only. Permutation test (300 shuffles of CER over the 16 conditions): ratio best-combo MAE / baseline MAE vs its null
(the best combo is re-picked on every shuffle, so the search over 4 x 10 combos is accounted for).
Targets: trocrctc s42, trocrsmall s42, trocrscratch s42 (TrOCR-small architecture, random init), htrvt s42, htrvt s43, htrvt mean of s42+s43; plus seed agreement of htrvt (s42 vs s43).

    /mnt/ssd2/cyttic/ml_env/bin/python lofo.py
-> results/cer.json, results/lofo.json
"""
import json, os, time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
FAMS = ["geom", "elastic", "stroke", "photo"]
CONDS = ["none"] + [f"{f}{l}" for f in FAMS for l in range(1, 5)]
fam = np.array(["none"] + [f for f in FAMS for _ in range(4)]); lvl = np.array([0] + [l for _ in FAMS for l in range(1, 5)], float)
aug = fam != "none"
M = ["Vendi", "PairD", "KID", "Cover", "Dens", "Prec", "A_dist"]
COMBOS = [("Vendi",)] + [("Vendi", k) for k in M[1:]] + [("Vendi", "KID", "Prec"), ("Vendi", "A_dist", "Cover"), ("Vendi_x_A",)]


def fetch_cer():
    from huggingface_hub import HfApi, hf_hub_download
    out = {}
    for m in sorted(x.id for x in HfApi().list_models(author="cyttic", search="hmt-")):
        if m.endswith("hmt-metrics"): continue
        for k in range(3):
            try: out[m.split("/")[1][4:]] = json.load(open(hf_hub_download(m, "results.json")))["final"]["test_greedy"]["cer"]; break
            except Exception: time.sleep(5)          # flaky router DNS
    json.dump(out, open(os.path.join(RES, "cer.json"), "w"), indent=1)
    return out


def lofo(X, y):
    pred = np.full(len(y), np.nan)
    for f in FAMS:
        te = fam == f; tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12; A = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ w
    return pred


mae = lambda p, y: float(np.abs(p[aug] - y[aug]).mean())


def main():
    cer = fetch_cer()
    df = pd.read_csv(os.path.join(RES, "metrics.csv")); T = {}
    for g in ["patch", "word", "line", "vit_token"]:
        t = df[df.granularity == g].groupby("condition")[M].mean().loc[CONDS]; t["Vendi_x_A"] = t.Vendi * t.A_dist; T[g] = t
    Y = {"trocrctc_s42": [cer[f"trocrctc-{c}-s42"] for c in CONDS], "trocrsmall_s42": [cer[f"trocrsmall-{c}-s42"] for c in CONDS],
         "trocrscratch_s42": [cer[f"trocrscratch-{c}-s42"] for c in CONDS],
         "htrvt_s42": [cer[f"htrvt-{c}-s42"] for c in CONDS], "htrvt_s43": [cer[f"htrvt-{c}-s43"] for c in CONDS]}
    if all(f"trocrscratchreg-{c}-s42" in cer for c in CONDS):          # ladder A, once all 17 runs are on HF
        Y["trocrscratchreg_s42"] = [cer[f"trocrscratchreg-{c}-s42"] for c in CONDS]
    if all(f"trocrscratchline-{c}-s42" in cer for c in CONDS):         # ladder B, once all 17 runs are on HF
        Y["trocrscratchline_s42"] = [cer[f"trocrscratchline-{c}-s42"] for c in CONDS]
    if all(f"trocrctc-{c}-s43" in cer for c in CONDS):                 # TrOCR-CTC seed 43, once all 17 runs are on HF
        Y["trocrctc_s43"] = [cer[f"trocrctc-{c}-s43"] for c in CONDS]
        Y["trocrctc_mean"] = list((np.array(Y["trocrctc_s42"]) + np.array(Y["trocrctc_s43"])) / 2)
    Y["htrvt_mean"] = list((np.array(Y["htrvt_s42"]) + np.array(Y["htrvt_s43"])) / 2)
    Y = {k: 100 * np.array(v) for k, v in Y.items()}
    a, b = Y["htrvt_s42"], Y["htrvt_s43"]
    out = {"htrvt_seed_agreement": {"spearman_16aug": float(spearmanr(a[aug], b[aug])[0]), "mean_abs_diff": float(np.abs(a - b).mean())}}
    print(f"HTR-VT s42 vs s43 on the 16 augmented conditions: rho {out['htrvt_seed_agreement']['spearman_16aug']:.4f}, "
          f"mean |diff| {out['htrvt_seed_agreement']['mean_abs_diff']:.2f} CER points")
    if "trocrctc_s43" in Y:
        a, b = Y["trocrctc_s42"], Y["trocrctc_s43"]
        out["trocrctc_seed_agreement"] = {"spearman_16aug": float(spearmanr(a[aug], b[aug])[0]), "mean_abs_diff": float(np.abs(a - b).mean())}
        print(f"TrOCR-CTC s42 vs s43 on the 16 augmented conditions: rho {out['trocrctc_seed_agreement']['spearman_16aug']:.4f}, "
              f"mean |diff| {out['trocrctc_seed_agreement']['mean_abs_diff']:.2f} CER points")

    def best_ratio(y):
        b0 = mae(lofo(np.zeros((17, 0)), y), y)
        return min(mae(lofo(T[g][list(c)].values, y), y) for g in T for c in COMBOS) / b0

    rng = np.random.default_rng(0)
    for name, y in Y.items():
        base = lofo(np.zeros((17, 0)), y); lv = lofo(lvl[:, None], y)
        rows = [{"gran": g, "metrics": "+".join(c), "MAE": mae(p, y), "rho": float(spearmanr(p[aug], y[aug])[0])}
                for g in T for c in COMBOS for p in [lofo(T[g][list(c)].values, y)]]
        rows.sort(key=lambda r: r["MAE"]); best = rows[0]
        obs = best_ratio(y); null = []
        for _ in range(300):
            yp = y.copy(); yp[aug] = rng.permutation(y[aug]); null.append(best_ratio(yp))
        null = np.array(null)
        bp = lofo(T[best["gran"]][best["metrics"].split("+")].values, y)
        out[name] = {"baseline_mean_MAE": mae(base, y), "baseline_level_MAE": mae(lv, y), "baseline_level_rho": float(spearmanr(lv[aug], y[aug])[0]),
                     "best": best, "top5": rows[:5], "best_over_baseline": obs, "perm_median": float(np.median(null)), "perm_p": float((null <= obs).mean()),
                     "best_MAE_per_family": {f: float(np.abs(bp[fam == f] - y[fam == f]).mean()) for f in FAMS}}
        o = out[name]
        print(f"\n== {name}: baseline mean MAE {o['baseline_mean_MAE']:.2f} | level MAE {o['baseline_level_MAE']:.2f} (rho {o['baseline_level_rho']:.4f})")
        for r in rows[:5]: print(f"   {r['gran']:9s} {r['metrics']:20s} MAE {r['MAE']:.2f} rho {r['rho']:.4f}")
        print(f"   best/baseline {obs:.4f} | permuted median {o['perm_median']:.4f} | p {o['perm_p']:.4f} | per family "
              + ", ".join(f"{f} {v:.2f}" for f, v in o["best_MAE_per_family"].items()))
    json.dump(out, open(os.path.join(RES, "lofo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
