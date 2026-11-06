# htr-metrics-transfer

Can training-free distribution metrics predict how much data augmentation helps a handwriting recognizer trained on a small real
dataset? Real data only: HHD, Option-B split (`cyttic/hhd-optionb-split`, paragraph- and writer-disjoint). Greedy CTC decoding, no
language model. Results go to Hugging Face (public); this repo holds code only.

| file | purpose |
|------|---------|
| `augment.py` | 4 augmentation families (geom, elastic, stroke, photo) x levels 0-4; used by training AND by the metric computation |
| `augment_preview.png` | two random draws of every family/level on one HHD line |
| `make_notebooks.py` | generates the notebooks below |
| `train_trocrctc.ipynb` | TrOCR-base ViT + one linear CTC head (pretrained encoder), 40 epochs |
| `train_htrvt.ipynb` | our implementation after HTR-VT (CNN + transformer + CTC, from scratch), 300 epochs |
| `queue_wave1.ipynb` | wave 1: 2 models x (none + 4 families x 4 levels) = 34 runs, seed 42 -> `cyttic/hmt-<model>-<cond>-s42` |
| `train_trocrsmall.ipynb` / `queue_wave2.ipynb` | wave 2: TrOCR-small stage 1 x 17 conditions (seed 42) + HTR-VT seed 43 x 17 |
| `train_trocrscratch.ipynb` | TrOCR-small architecture, random init + linear CTC head, from scratch (HTR-VT schedule: lr 5e-4, 300 epochs) |
| `queue_scratchcheck.ipynb` | feasibility check: trocrscratch x (none, geom2, elastic4), seed 42 -> `cyttic/hmt-trocrscratch-<cond>-s42` |
| `queue_wave3.ipynb` | trocrscratch x all 17 conditions, seed 42 (the 3 scratch-check runs are skipped: 14 new runs, ~18 h) |
| `train_trocrscratchreg.ipynb` / `queue_ladderA.ipynb` | ladder A: trocrscratch + HTR-VT's dropout 0.1 and span masking (40% of 16 px columns), 17 conditions, seed 42 (~22 h) |
| `queue_trocrctc_s43.ipynb` | TrOCR-CTC x 17 conditions, seed 43 (~6.5 h) -> `cyttic/hmt-trocrctc-<cond>-s43`: is TrOCR's CER ranking reliable enough to predict? |
| `lofo.py` | leave-one-family-out test of the metrics against every run's test CER (local; writes `results/`, not in git) |
| `model_aware.py` / `model_aware_kaggle.ipynb` (`make_model_aware_notebook.py`) | H2 probes on Kaggle 2x T4: each no-augmentation model's affinity (CER on augmented dev) + the metrics in its own feature space, then leave-one-family-out -> HF `cyttic/hmt-metrics` `model_aware/` |
| `regret.py` | selection regret: pick the augmentation by the LOFO-predicted CER (within a family / among all 16) vs the truly best -> `results/regret.json` |
| `metric_importance.py` | single metrics, drop-one ablation, level-only baseline, cross-model feature spaces, own-feature selection regret -> `results/metric_importance.json` |

Split: train = Option-B train minus the dev writers (20%, seed 42); dev = the unlabeled target sample for the metrics (also scored every
few epochs as a learning curve, nothing selected on it); test = Option-B test, scored once after the last epoch.

## Server (A5000)

Once: `git clone https://github.com/cyttic/htr-metrics-transfer.git` (venv `~/env` with papermill, `hf auth login` done).

Run: `cd ~/htr-metrics-transfer && git pull`, then `tmux new -s hmt` and `./run.sh queue_<name>.ipynb` (e.g. `queue_scratchcheck.ipynb`).
Re-running the queue skips runs already on HF. Failed runs' executed notebooks go to the HF dataset `cyttic/a5000-queue-logs`.
