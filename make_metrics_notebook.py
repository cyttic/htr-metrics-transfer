# -*- coding: utf-8 -*-
"""Writes metrics_kaggle.ipynb: the training-free metrics of all wave-1 conditions on Kaggle 2x T4.
augment.py and metrics_hmt.py are embedded verbatim (%%writefile), so the notebook needs nothing but HF.
Usage: python make_metrics_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("# htr-metrics-transfer: training-free metrics of the wave-1 augmentations\n\n"
       "For every condition (`none` + geom/elastic/stroke/photo x levels 1-4) and draw k = 0..2 (`none`: 1 draw), the training set "
       "(Option-B train minus the 20% dev writers, the exact lines the recognizers saw) is augmented once with `augment.py` and scored "
       "against the un-augmented dev lines at 4 granularities (patch / word / line with DINOv2-small, vit_token with the TrOCR-base encoder): "
       "coverage, density, precision, Vendi (+ SD over 10 subsamples), mean pairwise distance, proxy A-distance, KID.\n\n"
       "49 jobs, split over the 2 GPUs (one process per T4). Every finished job is pushed to `cyttic/hmt-metrics` (`per/<cond>_k<k>.json`), "
       "so **Save & Run All again resumes** after a timeout. Expected about 2-3 h.\n\n"
       "**Setup:** Accelerator = GPU T4 x2, Internet = ON, secret `HF_TOKEN` (write)."),
    code("!pip install -q 'pillow==11.2.1'   # Kaggle's base Pillow breaks transformers; the workers are fresh processes, so no restart needed"),
    code("from huggingface_hub import login\n"
         "try:\n"
         "    from kaggle_secrets import UserSecretsClient\n"
         "    tok = UserSecretsClient().get_secret('HF_TOKEN')\n"
         "except Exception:\n"
         "    tok = None\n"
         "login(token=tok) if tok else login()"),
    code("%%writefile augment.py\n" + src("augment.py")),
    code("%%writefile metrics_hmt.py\n" + src("metrics_hmt.py")),
    md("## Run: one worker per GPU (logs in `worker<i>.log`)"),
    code("import subprocess, os, time\n"
         "DRAWS = 3\n"
         "procs = [subprocess.Popen(f'python metrics_hmt.py --shard {i} --nshards 2 --draws {DRAWS} > worker{i}.log 2>&1', shell=True,\n"
         "                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)}) for i in range(2)]\n"
         "seen = [0, 0]\n"
         "while any(p.poll() is None for p in procs):\n"
         "    time.sleep(60)\n"
         "    for i in range(2):\n"
         "        lines = [l for l in open(f'worker{i}.log', errors='ignore').read().splitlines() if l.startswith(('[', 'train', 'shard', 'gap', 'upload'))]\n"
         "        for l in lines[seen[i]:]: print(f'w{i}', l)\n"
         "        seen[i] = len(lines)\n"
         "print('exit codes', [p.returncode for p in procs])\n"
         "for i in range(2):\n"
         "    if procs[i].returncode: print(f'--- worker{i}.log tail ---'); print(open(f'worker{i}.log').read()[-3000:])"),
    md("## Merge -> `metrics.json` + `metrics.csv` (one row per condition, draw, granularity)"),
    code("import json, glob, pandas as pd\n"
         "from huggingface_hub import HfApi, snapshot_download\n"
         "d = snapshot_download('cyttic/hmt-metrics', repo_type='dataset', allow_patterns=['per/*.json'])\n"
         "runs = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/per/*.json'))]\n"
         "print(len(runs), 'of 49 jobs present')\n"
         "rows = [dict(condition=r['condition'], family=r['family'], level=r['level'], draw=r['draw'], granularity=g,\n"
         "             word_retention=r['word_retention'], **r[g]) for r in runs for g in ('patch', 'word', 'line', 'vit_token')]\n"
         "df = pd.DataFrame(rows); df.to_csv('metrics.csv', index=False); json.dump(runs, open('metrics.json', 'w'), indent=1)\n"
         "api = HfApi()\n"
         "for f in ('metrics.csv', 'metrics.json'):\n"
         "    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id='cyttic/hmt-metrics', repo_type='dataset', commit_message='merge')\n"
         "df.groupby(['granularity', 'condition'])[['Vendi', 'Vendi_sd', 'KID', 'Cover', 'Prec', 'A_dist']].mean().round(4)"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = os.path.join(HERE, "metrics_kaggle.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
