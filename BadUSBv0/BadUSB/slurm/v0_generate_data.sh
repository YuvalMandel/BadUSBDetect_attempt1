#!/bin/bash
# ============================================================
# SLURM job: Generate all data for the BadUSBv0 pipeline
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_generate_data.sh
#
# Steps:
#   1. Synthetic bot files (8 types × 100 = 800 files)
#   2. Balanced human files for the polynomial regressor
#   3. Person-disjoint data_split.json
#   4. Train polynomial regressor + copy to MLP/
# ============================================================
#SBATCH --job-name=v0_generate
#SBATCH --output=logs/v0_generate_%j.out
#SBATCH --error=logs/v0_generate_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e

ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

WORK="$SLURM_SUBMIT_DIR"
DATA="$WORK/dataset_generator"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

# ── Step 0: symlink UB dataset s2/ (Newton only) ──────────────────────────────
UB_S2="$WORK/../../UB_keystroke_dataset/s2"
if [ ! -e "$DATA/s2" ] && [ -d "$UB_S2" ]; then
    ln -s "$UB_S2" "$DATA/s2"
    echo "Created symlink: $DATA/s2 -> $UB_S2"
fi

# ── Step 1: generate synthetic bots ───────────────────────────────────────────
cd "$DATA"
if [ ! -d "Synthetic_Bots" ]; then
    echo "--- Generating bots (8 types × 100 files, 200 events each) ---"
    python -X utf8 bot_generator.py -o Synthetic_Bots      -f 100 -e 200
    python -X utf8 bot_generator.py -o Synthetic_Bots_test -f  20 -e 200
else
    echo "--- Synthetic_Bots already exists, skipping ---"
fi

# ── Step 2: generate balanced humans for regressor ────────────────────────────
if [ ! -d "Balanced_Humans" ]; then
    echo "--- Generating balanced human files for regressor ---"
    python -X utf8 human_generator.py -o Balanced_Humans      -f 124 -l 80 -e 1
    python -X utf8 human_generator.py -o Balanced_Humans_test -f  24 -l 80 -e 0
else
    echo "--- Balanced_Humans already exists, skipping ---"
fi

# ── Step 3: person-disjoint split ─────────────────────────────────────────────
cd "$WORK"
if [ ! -f "data_split.json" ]; then
    echo "--- Creating person-disjoint split ---"
    python -X utf8 split_persons.py \
        --bots-dir dataset_generator/Synthetic_Bots \
        --ub-dir   "$WORK/../../UB_keystroke_dataset" \
        --sessions s0 s1 s2 \
        --tasks    1
else
    echo "--- data_split.json already exists, skipping ---"
fi

# ── Step 4: train polynomial regressor ────────────────────────────────────────
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
    echo "--- poly_regressor.pkl already exists, skipping ---"
fi
cp poly_regressor.pkl "$WORK/MLP/poly_regressor.pkl"
echo "Regressor ready: $WORK/MLP/poly_regressor.pkl"

echo "========================================"
echo "End : $(date)"
echo "========================================"
