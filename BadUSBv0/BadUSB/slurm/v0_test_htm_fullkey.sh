#!/bin/bash
# ============================================================
# SLURM job: HTM fullkey — evaluate test split
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_test_htm_fullkey.sh
# ============================================================
#SBATCH --job-name=v0_htm_test
#SBATCH --output=logs/v0_htm_test_%j.out
#SBATCH --error=logs/v0_htm_test_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00

set -e

ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Task  : HTM fullkey — test split evaluation"
echo "Start : $(date)"
echo "========================================"

python -X utf8 - <<'PYEOF'
import os, sys, json, pickle
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

WORK = os.environ.get("SLURM_SUBMIT_DIR", ".")
HTM_DIR   = os.path.join(WORK, "HTM")
MODEL_PKL = os.path.join(WORK, "results", "HTM", "models",
                          "htm_model_0002_vf10.6429_fullkey.pkl")
CACHE_PKL = os.path.join(HTM_DIR, "windows_cache_fullkey.pkl")
SPLIT_JSON = os.path.join(WORK, "data_split.json")

sys.path.insert(0, HTM_DIR)
from htm.bindings.sdr import SDR

with open(MODEL_PKL, "rb") as f:
    md = pickle.load(f)
with open(CACHE_PKL, "rb") as f:
    wcache = pickle.load(f)
with open(SPLIT_JSON) as f:
    split = json.load(f)

sp          = md["sp"]
tm          = md["tm"]
encoder     = md["encoder"]
al_bytes    = md["al_live_bytes"]
input_width = md["input_width"]
warmup      = md["warmup_steps"]
best_thresh = md["best_thresh"]
active_cols = SDR(sp.getColumnDimensions())

print(f"best_thresh = {best_thresh:.4f}  warmup = {warmup}")
print(f"window_size = {md['window_size']}  window_step = {md['window_step']}")

def score_file(fp):
    seq = wcache["sequences"].get(fp)
    if not seq:
        return None
    tm.reset()
    al_eval = pickle.loads(al_bytes)
    raw = []
    for stats, key_idx, dwell, flight, dist in seq:
        sdr = SDR(input_width)
        sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
        sp.compute(sdr, False, active_cols)
        tm.compute(active_cols, learn=False)
        raw.append(al_eval.compute(float(tm.anomaly)))
    return raw[warmup:]

for split_name in ("val", "test"):
    labels, preds = [], []
    skipped = 0
    for is_bot, flist in [(0, split[split_name]["humans"]),
                          (1, split[split_name]["bots"])]:
        for fp in flist:
            scores = score_file(fp)
            if scores is None or len(scores) == 0:
                skipped += 1
                continue
            # first-crossing detection
            pred = 1 if any(s > best_thresh for s in scores) else 0
            labels.append(is_bot)
            preds.append(pred)

    labels = np.array(labels)
    preds  = np.array(preds)
    f1   = f1_score(labels, preds, zero_division=0)
    bacc = balanced_accuracy_score(labels, preds)
    cm   = confusion_matrix(labels, preds)
    n_bot = int((labels == 1).sum())
    print(f"\n── {split_name.upper()} ({len(labels)} files, {n_bot} bots, skipped={skipped})")
    print(f"   File-F1={f1:.4f}  BAcc={bacc:.4f}")
    if cm.shape == (2, 2):
        print(f"   Confusion (rows=actual, cols=pred):")
        print(f"            Human  Bot")
        print(f"   Human  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"   Bot    {cm[1,0]:5d}  {cm[1,1]:5d}")
PYEOF

echo "========================================"
echo "End : $(date)"
echo "========================================"
