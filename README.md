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

Split: train = Option-B train minus the dev writers (20%, seed 42); dev = the unlabeled target sample for the metrics (also scored every
few epochs as a learning curve, nothing selected on it); test = Option-B test, scored once after the last epoch.

## Server (A5000)

Once: `git clone https://github.com/cyttic/htr-metrics-transfer.git` (venv `~/env` with papermill, `hf auth login` done).

Run: `cd ~/htr-metrics-transfer && git pull`, then `tmux new -s hmt` and `./run.sh queue_wave1.ipynb`.
Re-running the queue skips runs already on HF. Failed runs' executed notebooks go to the HF dataset `cyttic/a5000-queue-logs`.
