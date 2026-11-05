# -*- coding: utf-8 -*-
"""Writes model_aware_kaggle.ipynb: affinity + metrics in each model's own feature space (model_aware.py) on Kaggle 2x T4, then the
leave-one-family-out analysis. augment.py, metrics_hmt.py, lofo.py, model_aware.py and the training notebooks' model / preprocessing
sources (model_src.json, from make_notebooks.py) are embedded verbatim (%%writefile), so the notebook needs nothing but HF.
Usage: python make_model_aware_notebook.py
"""
import json, os

import make_notebooks as MN

HERE = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

trocr_cfg = lambda m, enc: MN.TROCR_CFG.replace("__MODEL__", m).replace("__ENC__", enc)
MODEL_SRC = {
    "htrvt": {"cfg": MN.HTRVT_CFG, "fmt": MN.HTRVT_FMT, "model": MN.HTRVT_MODEL},
    "trocrctc": {"cfg": trocr_cfg("trocrctc", "microsoft/trocr-base-handwritten"), "fmt": MN.TROCR_FMT, "model": MN.TROCR_MODEL},
    "trocrsmall": {"cfg": trocr_cfg("trocrsmall", "microsoft/trocr-small-stage1"), "fmt": MN.TROCR_FMT, "model": MN.TROCR_MODEL},
    "trocrscratch": {"cfg": MN.SCRATCH_CFG, "fmt": MN.TROCR_FMT, "model": MN.SCRATCH_MODEL},
}

cells = [
    md("# htr-metrics-transfer: model-aware predictors (H2)\n\n"
       "Why do the image-only metrics predict HTR-VT's CER but not TrOCR's? Each recognizer trained **without** augmentation "
       "(`cyttic/hmt-<arch>-none-s<seed>`: HTR-VT s42/s43, TrOCR-CTC, TrOCR-small, TrOCR scratch) is used as a probe; nothing is trained.\n\n"
       "- **Affinity:** its greedy CER on the dev lines augmented with each condition (3 draws) minus the clean-dev CER.\n"
       "- **Own-feature metrics:** the same metrics as `metrics_hmt.py` (Cover, Dens, Prec, Vendi, PairD, A_dist, KID), computed in the "
       "model's own feature space (the CTC classifier's input frames; `frame` = 6000 sampled frames, `line` = per-line mean), training set "
       "augmented with exactly the image-metric draws vs the clean dev lines.\n\n"
       "5 probes x 49 jobs, split over the 2 GPUs. Each finished job goes to `cyttic/hmt-metrics` (`model_aware/<probe>/<cond>_k<k>.json`), "
       "so **Save & Run All again resumes** after a timeout. The last cell runs the leave-one-family-out test with permutation p-values "
       "-> `model_aware/summary.json`. Expected about 2-4 h.\n\n"
       "**Setup:** Accelerator = GPU T4 x2, Internet = ON, secret `HF_TOKEN` (write)."),
    code("!pip install -q 'pillow==11.2.1' 'transformers==5.15.0' jiwer   # the training version; workers are fresh processes, so no restart"),
    code("from huggingface_hub import login\n"
         "try:\n"
         "    from kaggle_secrets import UserSecretsClient\n"
         "    tok = UserSecretsClient().get_secret('HF_TOKEN')\n"
         "except Exception:\n"
         "    tok = None\n"
         "login(token=tok) if tok else login()"),
    code("%%writefile augment.py\n" + src("augment.py")),
    code("%%writefile metrics_hmt.py\n" + src("metrics_hmt.py")),
    code("%%writefile lofo.py\n" + src("lofo.py")),
    code("%%writefile model_aware.py\n" + src("model_aware.py")),
    code("%%writefile model_src.json\n" + json.dumps(MODEL_SRC, ensure_ascii=False, indent=1)),
    md("## Run: one worker per GPU (logs in `worker<i>.log`)\n\n`SMOKE = True` runs one worker on a few lines without uploading."),
    code("import subprocess, os, time\n"
         "SMOKE = False\n"
         "n = 1 if SMOKE else 2\n"
         "procs = [subprocess.Popen(f'python model_aware.py --shard {i} --nshards {n}' + (' --smoke' if SMOKE else '') + f' > worker{i}.log 2>&1',\n"
         "                          shell=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)}) for i in range(n)]\n"
         "seen = [0] * n\n"
         "while any(p.poll() is None for p in procs):\n"
         "    time.sleep(30)\n"
         "    for i in range(n):\n"
         "        lines = [l for l in open(f'worker{i}.log', errors='ignore').read().splitlines() if l.startswith(('[', 'train', 'shard', 'upload'))]\n"
         "        for l in lines[seen[i]:]: print(f'w{i}', l)\n"
         "        seen[i] = len(lines)\n"
         "print('exit codes', [p.returncode for p in procs])\n"
         "for i in range(n):\n"
         "    if procs[i].returncode: print(f'--- worker{i}.log tail ---'); print(open(f'worker{i}.log').read()[-3000:])"),
    md("## Leave-one-family-out: affinity / own-feature metrics / affinity + image metrics -> `model_aware/summary.json`"),
    code("!python model_aware.py --analyze"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = os.path.join(HERE, "model_aware_kaggle.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
