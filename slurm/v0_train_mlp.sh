#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 MLP pipeline
#   sbatch slurm/v0_train_mlp.sh
# ============================================================
#SBATCH --job-name=v0_mlp
#SBATCH --output=logs/v0_mlp_%j.out
#SBATCH --error=logs/v0_mlp_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e   # stop immediately on any error

source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── Absolute paths ────────────────────────────────────────────────────────────
ROOT="$SLURM_SUBMIT_DIR"          # project root (where sbatch was run)
WORK="$ROOT/BadUSBv0/BadUSB"
DATA="$WORK/dataset_generator"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "ROOT  : $ROOT"
echo "Start : $(date)"
echo "========================================"

# ── Step 0: symlink UB dataset s2/ into dataset_generator/ (Newton only) ─────
UB_S2="$ROOT/../UB_keystroke_dataset/s2"
if [ ! -e "$DATA/s2" ] && [ -d "$UB_S2" ]; then
    ln -s "$UB_S2" "$DATA/s2"
    echo "Created symlink: $DATA/s2 -> $UB_S2"
fi

# ── Step 1: generate bots + humans (skip if already done) ────────────────────
if [ ! -d "$DATA/Synthetic_Bots" ]; then
    echo "--- Generating datasets ---"
    cd "$DATA"
    python -X utf8 bot_generator.py     -o Synthetic_Bots      -f 25 -e 80
    python -X utf8 bot_generator.py     -o Synthetic_Bots_test -f  5 -e 80
    python -X utf8 human_generator.py   -o Balanced_Humans      -f 124 -l 80 -e 1
    python -X utf8 human_generator.py   -o Balanced_Humans_test -f  24 -l 80 -e 0
    cd "$ROOT"
fi

# ── Step 2: person-disjoint split (skip if already done) ─────────────────────
if [ ! -f "$WORK/data_split.json" ]; then
    echo "--- Creating person-disjoint split ---"
    cd "$WORK"
    python -X utf8 split_persons.py \
        --bots-dir dataset_generator/Synthetic_Bots \
        --ub-dir   "$ROOT/../UB_keystroke_dataset" \
        --sessions s0 s1 s2 \
        --tasks    1
    cd "$ROOT"
fi

# ── Step 3: train polynomial regressor (skip if pkl already present) ─────────
cd "$WORK/MLP/regressor"
if [ ! -f "poly_regressor.pkl" ]; then
    echo "--- Training polynomial regressor ---"
    python -X utf8 regressor_train.py \
        -hu "$DATA/Balanced_Humans" \
        -m  poly_regressor.pkl
    python -X utf8 test_regressor.py \
        -hu "$DATA/Balanced_Humans_test" \
        -b  "$DATA/Synthetic_Bots_test" \
        -m  poly_regressor.pkl
else
    echo "--- Skipping regressor training (poly_regressor.pkl already exists) ---"
fi
cp poly_regressor.pkl "$WORK/MLP/poly_regressor.pkl"

# ── Step 4: feature extraction → 3 CSVs ──────────────────────────────────────
echo "--- Feature extraction (person-disjoint splits) ---"
cd "$WORK/MLP"
python -X utf8 dataset_csv_generator.py --split-json "$WORK/data_split.json"

# ── Step 5: train MLP ─────────────────────────────────────────────────────────
echo "--- Training MLP ---"
python -X utf8 model_training.py

echo "========================================"
echo "End : $(date)"
echo "========================================"
