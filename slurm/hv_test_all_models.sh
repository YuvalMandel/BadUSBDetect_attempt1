#!/bin/bash
# ============================================================
# SLURM job array: test ALL HTM-Velocity models in parallel
#
# One job per .pkl file in hv_models/.
# Results (plots + optional decision logs) go to hv_plots/ and logs/.
#
# Usage:
#   sbatch slurm/hv_test_all_models.sh
#
# To test only a subset, override the array range:
#   sbatch --array=0-9 slurm/hv_test_all_models.sh
#
# Modes:
#   orig            — val/test splits only (fastest)
#   all_non_train   — all non-training files, uses hv_windows_cache (default)
#   all_other_files — new .txt files from filesystem (needs poly_regressor.pkl)
# ============================================================
#SBATCH --job-name=hv_test
#SBATCH --output=logs/hv_test_%A_%a.out
#SBATCH --error=logs/hv_test_%A_%a.err
#SBATCH --array=0-999%50
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Edit these -------------------------------------------------
MODE="all_non_train"          # orig | all_non_train | all_other_files
DATA_ROOT="../UB_keystroke_dataset/"
BOTS_ROOT="../BadUSBdataset"
# -----------------------------------------------------------------

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Pick model for this task -----------------------------------
mapfile -t MODELS < <(ls hv_models/*.pkl 2>/dev/null | sort)
N_MODELS=${#MODELS[@]}

if [[ $N_MODELS -eq 0 ]]; then
    echo "ERROR: No .pkl files found in hv_models/"
    exit 1
fi

if [[ $SLURM_ARRAY_TASK_ID -ge $N_MODELS ]]; then
    echo "Task $SLURM_ARRAY_TASK_ID >= N_MODELS ($N_MODELS), nothing to do."
    exit 0
fi

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "Task  : $SLURM_ARRAY_TASK_ID / $((N_MODELS-1))"
echo "Model : $MODEL"
echo "Mode  : $MODE"
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "========================================"

python -u htm_velocity/htm_velocity_test_model.py \
    --model     "$MODEL" \
    --mode      "$MODE"  \
    --data_root "$DATA_ROOT" \
    --bots_root "$BOTS_ROOT"

echo "========================================"
echo "End   : $(date)"
echo "========================================"
