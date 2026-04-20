#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 MLP pipeline
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_train_mlp.sh
#
# Pipeline env vars (set by v0_full_pipeline.sh):
#   PIPELINE_MODE   = partial|full   (default: partial)
#   PIPELINE_SEARCH = --search       (default: empty / disabled)
# ============================================================
#SBATCH --job-name=v0_mlp
#SBATCH --output=logs/v0_mlp_%j.out
#SBATCH --error=logs/v0_mlp_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
##SBATCH --gres=gpu:1              # Uncomment to request a GPU
##SBATCH --partition=gpu           # Uncomment if GPU partition has a name
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

# Accept overrides from v0_full_pipeline.sh or use defaults
MODE="${PIPELINE_MODE:-partial}"
SEARCH="${PIPELINE_SEARCH:-}"
N_CONFIGS="${PIPELINE_N_CONFIGS:-50}"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "Mode  : $MODE  |  HP search: ${SEARCH:-disabled}  |  n-configs: $N_CONFIGS"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

# ── Step 0: symlink UB dataset s2/ into dataset_generator/ (Newton only) ─────
UB_S2="$WORK/../../UB_keystroke_dataset/s2"
if [ ! -e "$DATA/s2" ] && [ -d "$UB_S2" ]; then
    ln -s "$UB_S2" "$DATA/s2"
    echo "Created symlink: $DATA/s2 -> $UB_S2"
fi

# ── Step 1: generate bots + humans (skip if already done) ────────────────────
if [ ! -d "$DATA/Synthetic_Bots" ]; then
    echo "--- Generating datasets (8 types × 100 files, 200 events) ---"
    cd "$DATA"
    python -X utf8 bot_generator.py     -o Synthetic_Bots      -f 100 -e 200
    python -X utf8 bot_generator.py     -o Synthetic_Bots_test -f  20 -e 200
    python -X utf8 human_generator.py   -o Balanced_Humans      -f 124 -l 80 -e 1
    python -X utf8 human_generator.py   -o Balanced_Humans_test -f  24 -l 80 -e 0
    cd "$WORK"
fi

# ── Step 2: person-disjoint split (skip if already done) ─────────────────────
if [ ! -f "$WORK/data_split.json" ]; then
    echo "--- Creating person-disjoint split ---"
    cd "$WORK"
    python -X utf8 split_persons.py \
        --bots-dir dataset_generator/Synthetic_Bots \
        --ub-dir   "$WORK/../../UB_keystroke_dataset" \
        --sessions s0 s1 s2 \
        --tasks    1
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
echo "--- Feature extraction (mode=$MODE) ---"
cd "$WORK/MLP"
python -X utf8 dataset_csv_generator.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE"

# ── Step 5: train MLP ─────────────────────────────────────────────────────────
echo "--- Training MLP (mode=$MODE ${SEARCH:-}) ---"
python -X utf8 model_training.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE" \
    --n-configs "$N_CONFIGS" \
    $SEARCH

echo "========================================"
echo "End : $(date)"
echo "========================================"
