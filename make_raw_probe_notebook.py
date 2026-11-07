# -*- coding: utf-8 -*-
"""Writes raw_probe_kaggle.ipynb: the metrics in OFF-THE-SHELF encoders' frame spaces (probe_raw.py) on Kaggle 2x T4, then the
leave-one-family-out analysis against every recognizer's CER. augment.py, metrics_hmt.py, lofo.py, model_aware.py, probe_raw.py and
the model / preprocessing sources (model_src.json: the training notebooks' TrOCR code from make_notebooks.py + a CLIP wrapper) are
embedded verbatim (%%writefile), so the notebook needs nothing but HF.
Usage: python make_raw_probe_notebook.py
"""
import json, os

import make_notebooks as MN

HERE = os.path.dirname(os.path.abspath(__file__))
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
trocr_cfg = lambda m, enc: MN.TROCR_CFG.replace("__MODEL__", m).replace("__ENC__", enc)

# CLIP ViT-B/16 vision tower on the same tiled 384x384 square: 24x24 patches (position embeddings interpolated from 14x14), the same
# reading-order frames as EncoderLinearCTC; inputs arrive in [-1,1] (TROCR_FMT) and are re-normalised with CLIP's mean / std.
CLIP_MODEL = r'''
from transformers import CLIPVisionModel
class ClipFrames(nn.Module):
    def __init__(self, vit):
        super().__init__(); self.vit = vit; self.Hd = vit.config.hidden_size; self.k = ROW_H // 16
        self.classifier = nn.Linear(self.k * self.Hd, SUB * N_CLASSES)   # never trained: only its INPUT (the frames) is read
        self.register_buffer("mu", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    def forward(self, px, lens=None):
        x = ((px + 1) / 2 - self.mu) / self.sd
        tok = self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, -576:]; B = tok.shape[0]
        g = tok.reshape(B, 24, 24, self.Hd)[:, :RROWS * self.k]
        g = g.reshape(B, RROWS, self.k, 24, self.Hd).permute(0, 1, 3, 2, 4).reshape(B, RROWS * 24, self.k * self.Hd)
        return F.log_softmax(self.classifier(g).reshape(B, RROWS * 24 * SUB, N_CLASSES).float(), dim=-1)
torch.manual_seed(SEED)
model = ClipFrames(CLIPVisionModel.from_pretrained(ENC_SRC)).to(device)
'''
MODEL_SRC = {
    "trocrctc": {"cfg": trocr_cfg("trocrctc", "microsoft/trocr-base-handwritten"), "fmt": MN.TROCR_FMT, "model": MN.TROCR_MODEL},
    "trocrsmall": {"cfg": trocr_cfg("trocrsmall", "microsoft/trocr-small-stage1"), "fmt": MN.TROCR_FMT, "model": MN.TROCR_MODEL},
    "trocrscratch": {"cfg": MN.SCRATCH_CFG, "fmt": MN.TROCR_FMT, "model": MN.SCRATCH_MODEL},
    "clip": {"cfg": trocr_cfg("clip", "openai/clip-vit-base-patch16"), "fmt": MN.TROCR_FMT, "model": CLIP_MODEL},
}

cells = [
    md("# htr-metrics-transfer: training-free probe in off-the-shelf encoder spaces\n\n"
       "The metrics predict TrOCR's CER only in the **trained** TrOCR's own frame space (`model_aware`: trocrctc ratio 0.56, p 0.0033) -- "
       "not training-free. Here the same frame / line features are computed with encoders **nobody trained on HHD**:\n\n"
       "- `trocrbase-raw`: `microsoft/trocr-base-handwritten` encoder as downloaded (= TrOCR-CTC before fine-tuning)\n"
       "- `trocrsmall-raw`: `microsoft/trocr-small-stage1` encoder as downloaded\n"
       "- `deitsmall-rand`: the same DeiT-small with random weights (control)\n"
       "- `clip-b16-raw`: `openai/clip-vit-base-patch16` vision tower (generic pretrained ViT)\n\n"
       "Same tiling, frames (4 patch rows per 16 px column), candidate draws and reference as `model_aware.py`; metrics = "
       "`metrics_hmt.score` (Cover, Dens, Prec, Vendi, PairD, A_dist, KID). 4 probes x 49 jobs over the 2 GPUs; each job goes to "
       "`cyttic/hmt-metrics` (`raw_probe/<probe>/<cond>_k<k>.json`), so **Save & Run All again resumes** after a timeout. The last cell "
       "runs leave-one-family-out for every probe x recognizer (TrOCR-CTC s42/s43, TrOCR-small, TrOCR scratch, ladder A/B when complete, "
       "HTR-VT s42/s43) with permutation p-values -> `raw_probe/summary.json`. Expected about 1.5-3 h.\n\n"
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
    code("%%writefile probe_raw.py\n" + src("probe_raw.py")),
    code("%%writefile model_src.json\n" + json.dumps(MODEL_SRC, ensure_ascii=False, indent=1)),
    md("## Run: one worker per GPU (logs in `worker<i>.log`)\n\n`SMOKE = True` runs one worker on a few lines without uploading."),
    code("import subprocess, os, time\n"
         "SMOKE = False\n"
         "n = 1 if SMOKE else 2\n"
         "procs = [subprocess.Popen(f'python probe_raw.py --shard {i} --nshards {n}' + (' --smoke' if SMOKE else '') + f' > worker{i}.log 2>&1',\n"
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
    md("## Leave-one-family-out: every probe x every recognizer -> `raw_probe/summary.json`"),
    code("!python probe_raw.py --analyze"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = os.path.join(HERE, "raw_probe_kaggle.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out)
