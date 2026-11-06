# -*- coding: utf-8 -*-
"""Which metric carries the prediction, and does it beat "stronger level = better"?

Same leave-one-family-out protocol as lofo.py; spaces = image (patch / word / line / vit_token, results/metrics.csv) and each probe's
own feature space (frame / line, model_aware/ on cyttic/hmt-metrics). Ratio = LOFO MAE / MAE of the no-metric baseline (< 1 = helps).
  1. single   -- every metric alone, in every space, per target; permutation p for that fixed predictor (300 shuffles)
  2. drop     -- the winning combos with one metric removed (the ratio rise = what that metric contributes)
  3. level    -- level-only regression vs the winning combos, and combo + level (does the metric add anything beyond the level?);
                 p = permutation test of the combo+level ratio against level-only on the same shuffles
  4. cross    -- one TrOCR's feature space predicting the other TrOCR's CER (fixed combos and best-of-combos with re-picked permutation)
  5. regret   -- selection regret of the own-feature winner vs level-only vs random (regret.py's definitions)

    /mnt/ssd2/cyttic/ml_env/bin/python metric_importance.py      (needs results/cer.json + results/metrics.csv)
-> results/metric_importance.json
"""
import glob, json, os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lofo import CONDS, COMBOS, M, RES, aug, lofo, lvl, mae
from regret import random_regrets, regrets

NPERM = 300
PROBES = {"htrvt_s42": ("htrvt-s42", "htrvt-{}-s42"), "htrvt_s43": ("htrvt-s43", "htrvt-{}-s43"),
          "trocrctc_s42": ("trocrctc-s42", "trocrctc-{}-s42"), "trocrsmall_s42": ("trocrsmall-s42", "trocrsmall-{}-s42"),
          "trocrscratch_s42": ("trocrscratch-s42", "trocrscratch-{}-s42")}
WIN = {"htrvt_s42": ("img word", ("Vendi", "KID", "Prec")), "htrvt_s43": ("img word", ("Vendi", "KID", "Prec")),
       "trocrctc_s42": ("own line", ("Vendi", "Dens")), "trocrsmall_s42": ("own line", ("Vendi", "Prec")),
       "trocrscratch_s42": ("own frame", ("Vendi",))}


def load():
    from huggingface_hub import snapshot_download
    cer = json.load(open(os.path.join(RES, "cer.json")))
    df = pd.read_csv(os.path.join(RES, "metrics.csv"))
    img = {f"img {g}": df[df.granularity == g].groupby("condition")[M].mean().loc[CONDS] for g in ["patch", "word", "line", "vit_token"]}
    d = snapshot_download("cyttic/hmt-metrics", repo_type="dataset", allow_patterns=["model_aware/*/*.json"])
    rows = [json.load(open(f)) for f in glob.glob(f"{d}/model_aware/*/*.json") if not f.endswith("summary.json")]
    own = {}
    for p, _ in PROBES.values():
        R = [r for r in rows if r["probe"] == p]
        own[p] = {f"own {g}": pd.DataFrame({c: pd.DataFrame([r[g] for r in R if r["condition"] == c])[M].mean() for c in CONDS}).T
                  for g in ("frame", "line")}
    Y = {k: 100 * np.array([cer[v.format(c)] for c in CONDS]) for k, (_, v) in PROBES.items()}
    return img, own, Y


def ratio(X, y):
    return mae(lofo(X, y), y) / mae(lofo(np.zeros((17, 0)), y), y)


def shuffles(y, rng):
    for _ in range(NPERM):
        yp = y.copy(); yp[aug] = rng.permutation(y[aug]); yield yp


def main():
    img, own, Y = load(); rng = np.random.default_rng(0); out = {}
    for name, y in Y.items():
        S = {**img, **own[PROBES[name][0]]}; r = {}
        perms = list(shuffles(y, rng))
        # 1. single metrics
        single = {}
        for s, t in S.items():
            for m in M:
                X = t[[m]].values; obs = ratio(X, y)
                single[f"{s} {m}"] = {"ratio": obs, "rho": float(spearmanr(lofo(X, y)[aug], y[aug])[0]),
                                      "p": float(np.mean([ratio(X, yp) <= obs for yp in perms]))}
        r["single"] = single
        # 2. drop-one on the winning combo
        ws, wc = WIN[name]; Xw = S[ws][list(wc)].values
        r["win"] = {"combo": f"{ws} {'+'.join(wc)}", "ratio": ratio(Xw, y)}
        r["drop"] = {m: ratio(S[ws][[k for k in wc if k != m]].values, y) if len(wc) > 1 else ratio(np.zeros((17, 0)), y) for m in wc}
        # 3. level baseline
        L = lvl[:, None]; Xwl = np.c_[Xw, L]
        lv, wl = ratio(L, y), ratio(Xwl, y)
        gain = lv - wl; null = [ratio(L, yp) - ratio(np.c_[Xw, L], yp) for yp in perms]
        r["level"] = {"level_only": lv, "win": r["win"]["ratio"], "win+level": wl,
                      "p_win_beats_level": float(np.mean([ratio(L, yp) - ratio(Xw, yp) >= lv - r["win"]["ratio"] for yp in perms])),
                      "p_win+level_beats_level": float(np.mean([g >= gain for g in null]))}
        # 5. regret: own winner vs level vs random
        sel = {"win": lofo(Xw, y), "win+level": lofo(Xwl, y), "level": lofo(L, y)}
        rr = {k: dict(zip(["within", "global", "global_top3"], regrets(p, y))) for k, p in sel.items()}
        rw, rg, r3 = random_regrets(y)
        rr["random"] = {"within": float(rw.mean()), "global": float(rg.mean()), "global_top3": float(r3.mean())}
        rr["p_win_vs_random"] = {"within": float((rw <= rr["win"]["within"] + 1e-9).mean()),
                                 "global": float((rg <= rr["win"]["global"] + 1e-9).mean()),
                                 "global_top3": float((r3 <= rr["win"]["global_top3"] + 1e-9).mean())}
        r["regret"] = rr
        out[name] = r
    # 4. cross: one TrOCR's feature space predicting the other's CER
    cross = {}
    for src, dst in [("trocrctc-s42", "trocrsmall_s42"), ("trocrsmall-s42", "trocrctc_s42"), ("trocrctc-s42", "htrvt_s42"),
                     ("trocrsmall-s42", "htrvt_s42"), ("htrvt-s42", "trocrctc_s42"), ("htrvt-s42", "trocrsmall_s42")]:
        y = Y[dst]; cands = {f"{g} {'+'.join(c)}": own[src][g][list(c)].values for g in ("own frame", "own line") for c in COMBOS
                             if "Vendi_x_A" not in c}
        obs = {k: ratio(X, y) for k, X in cands.items()}; kb = min(obs, key=obs.get)
        null = [min(ratio(X, yp) for X in cands.values()) for yp in shuffles(y, rng)]
        cross[f"{src} -> {dst}"] = {"best": kb, "ratio": obs[kb], "p": float(np.mean([n <= obs[kb] for n in null])),
                                    "line Vendi+Dens": obs["own line Vendi+Dens"], "line Vendi+Prec": obs["own line Vendi+Prec"]}
    out["cross"] = cross
    json.dump(out, open(os.path.join(RES, "metric_importance.json"), "w"), indent=1)

    for name in Y:
        r = out[name]; print(f"\n===== {name}")
        print("  best single metric per space (ratio / rho / p):")
        for s in sorted({k.rsplit(' ', 1)[0] for k in r["single"]}):
            k = min((k for k in r["single"] if k.rsplit(' ', 1)[0] == s), key=lambda k: r["single"][k]["ratio"]); v = r["single"][k]
            print(f"    {s:14s} {k.rsplit(' ', 1)[1]:7s} {v['ratio']:.4f} {v['rho']:+.4f} p {v['p']:.4f}")
        print("  Vendi alone:  " + " | ".join(f"{s} {r['single'][f'{s} Vendi']['ratio']:.4f}" for s in sorted({k.rsplit(' ', 1)[0] for k in r['single']})))
        print(f"  winner {r['win']['combo']} {r['win']['ratio']:.4f}; drop-one: " + ", ".join(f"-{m} {v:.4f}" for m, v in r["drop"].items()))
        l = r["level"]
        print(f"  level-only {l['level_only']:.4f} | winner {l['win']:.4f} (p beats level {l['p_win_beats_level']:.4f}) | "
              f"winner+level {l['win+level']:.4f} (p {l['p_win+level_beats_level']:.4f})")
        g = r["regret"]
        print("  regret within/global/top3: " + " | ".join(f"{k} {g[k]['within']:.4f}/{g[k]['global']:.4f}/{g[k]['global_top3']:.4f}"
                                                       for k in ["win", "win+level", "level", "random"]))
        print(f"  winner p vs random: " + " / ".join(f"{v:.4f}" for v in g["p_win_vs_random"].values()))
    print("\n===== cross-space")
    for k, v in cross.items():
        print(f"  {k:32s} best {v['best']:26s} {v['ratio']:.4f} p {v['p']:.4f} | line V+Dens {v['line Vendi+Dens']:.4f} | line V+Prec {v['line Vendi+Prec']:.4f}")
    print("->", os.path.join(RES, "metric_importance.json"))


if __name__ == "__main__":
    main()
