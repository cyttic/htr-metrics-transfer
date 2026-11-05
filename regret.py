# -*- coding: utf-8 -*-
"""Selection regret: does choosing an augmentation by the metrics' PREDICTED CER (no training) land close to the truly best one?

Predictions are lofo.py's leave-one-family-out ones: every condition is predicted by a model fitted WITHOUT its family.
Two decisions, scored in test-CER points of the chosen condition minus the best one (regret; 0 = perfect):
  within  -- for each family separately: pick its level (1-4) by predicted CER; mean regret over the 4 families
  global  -- pick one of the 16 augmented conditions by predicted CER (also: best of the predicted top 3, i.e. train 3 instead of 16)
Selectors: FIXED = word Vendi+KID+Prec (HTR-VT s42's best LOFO combo, held fixed for every target), OWN = the target's own best LOFO combo
(chosen on the same data, so optimistic), level-only regression, always the strongest level (4), and random (expected value).
Random-chance p for FIXED: share of random picks (exact, over all choices) whose regret <= the observed one.

    /mnt/ssd2/cyttic/ml_env/bin/python regret.py      (needs results/cer.json + results/lofo.json from lofo.py)
-> results/regret.json
"""
import itertools, json, os

import numpy as np
import pandas as pd

from lofo import CONDS, FAMS, M, RES, aug, fam, lofo, lvl

FIXED = ("word", ("Vendi", "KID", "Prec"))
TARGETS = ["htrvt_s42", "htrvt_s43", "htrvt_mean", "trocrctc_s42", "trocrsmall_s42", "trocrscratch_s42"]


def regrets(p, y):
    """-> within-family mean regret, global regret, global top-3 regret (CER points)."""
    w = [y[fam == f][np.argmin(p[fam == f])] - y[fam == f].min() for f in FAMS]
    ya, pa = y[aug], p[aug]; order = np.argsort(pa)
    return float(np.mean(w)), float(ya[order[0]] - ya.min()), float(ya[order[:3]].min() - ya.min())


def random_regrets(y):
    """Expected regret of random picks, and the full distributions for the p-values."""
    per_fam = [y[fam == f] - y[fam == f].min() for f in FAMS]
    within = np.array([np.mean(c) for c in itertools.product(*per_fam)])     # 4^4 joint picks
    ya = y[aug] - y[aug].min()
    top3 = np.array([ya[list(c)].min() for c in itertools.combinations(range(16), 3)])
    return within, ya, top3


def main():
    cer = json.load(open(os.path.join(RES, "cer.json"))); best = json.load(open(os.path.join(RES, "lofo.json")))
    df = pd.read_csv(os.path.join(RES, "metrics.csv")); T = {}
    for g in ["patch", "word", "line", "vit_token"]:
        t = df[df.granularity == g].groupby("condition")[M].mean().loc[CONDS]; t["Vendi_x_A"] = t.Vendi * t.A_dist; T[g] = t
    Y = {"htrvt_s42": "htrvt-{}-s42", "htrvt_s43": "htrvt-{}-s43", "trocrctc_s42": "trocrctc-{}-s42",
         "trocrsmall_s42": "trocrsmall-{}-s42", "trocrscratch_s42": "trocrscratch-{}-s42"}
    Y = {k: 100 * np.array([cer[v.format(c)] for c in CONDS]) for k, v in Y.items()}
    Y["htrvt_mean"] = (Y["htrvt_s42"] + Y["htrvt_s43"]) / 2
    if "trocrscratchreg_s42" in best:                                   # ladder A, once lofo.py has scored it
        Y["trocrscratchreg_s42"] = 100 * np.array([cer[f"trocrscratchreg-{c}-s42"] for c in CONDS]); TARGETS.append("trocrscratchreg_s42")

    out = {}
    for name in TARGETS:
        y = Y[name]; ob = best[name]["best"]
        sel = {"fixed": lofo(T[FIXED[0]][list(FIXED[1])].values, y),
               "own": lofo(T[ob["gran"]][ob["metrics"].split("+")].values, y),
               "level": lofo(lvl[:, None], y),
               "strongest": np.where(aug, -lvl, np.inf)}
        r = {k: dict(zip(["within", "global", "global_top3"], regrets(p, y))) for k, p in sel.items()}
        rw, rg, r3 = random_regrets(y)
        r["random"] = {"within": float(rw.mean()), "global": float(rg.mean()), "global_top3": float(r3.mean())}
        f = r["fixed"]
        r["fixed_p_vs_random"] = {"within": float((rw <= f["within"] + 1e-9).mean()), "global": float((rg <= f["global"] + 1e-9).mean()),
                                  "global_top3": float((r3 <= f["global_top3"] + 1e-9).mean())}
        ya = y[aug]; conds = np.array(CONDS)[aug]
        r["best_condition"] = {"name": str(conds[np.argmin(ya)]), "cer": float(ya.min()), "none_cer": float(y[0])}
        r["fixed_pick"] = {"name": str(conds[np.argmin(sel["fixed"][aug])]), "cer": float(ya[np.argmin(sel["fixed"][aug])])}
        r["own_combo"] = f"{ob['gran']} {ob['metrics']}"
        out[name] = r
        print(f"\n== {name}: best {r['best_condition']['name']} {r['best_condition']['cer']:.2f} (none {y[0]:.2f}); "
              f"FIXED picks {r['fixed_pick']['name']} {r['fixed_pick']['cer']:.2f}")
        print(f"   {'selector':10s} {'within':>8s} {'global':>8s} {'top3':>8s}")
        for k in ["fixed", "own", "level", "strongest", "random"]:
            print(f"   {k:10s} {r[k]['within']:8.2f} {r[k]['global']:8.2f} {r[k]['global_top3']:8.2f}")
        pv = r["fixed_p_vs_random"]
        print(f"   FIXED p vs random: within {pv['within']:.4f} | global {pv['global']:.4f} | top3 {pv['global_top3']:.4f}")
    json.dump(out, open(os.path.join(RES, "regret.json"), "w"), indent=1)
    print("->", os.path.join(RES, "regret.json"))


if __name__ == "__main__":
    main()
