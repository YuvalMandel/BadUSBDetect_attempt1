#!/bin/bash
# ============================================================
# SLURM script — test one HTM-Distance model
#
# Edit MODEL and MODE before submitting.
#
# Submit with:
#   sbatch slurm/dist_test_model.sh
#
# 128 CPUs are used only for the 'all_other_files' mode
# (parallel parsing of raw .txt files).
# For 'orig' and 'all_non_train' modes, fewer CPUs are fine.
# ============================================================
#SBATCH --job-name=dist_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dist_test_%j.out
#SBATCH --error=logs/dist_test_%j.err
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source /home/yuval.mandel/miniconda3/etc/profile.d/conda.sh
conda activate htm_keyboard_1

echo "Python:    $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CPUs:      $SLURM_CPUS_PER_TASK"

# Prevent NumPy/BLAS from spawning extra threads inside each worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Edit these two lines before submitting --------------------
MODEL="dist_models/dist0121_sp40_enc16w9_tm16_act13_wu5_al15_vf10.8571_tf10.9189.pkl"
MODE="all_other_files"   # orig | all_non_train | all_other_files
# ----------------------------------------------------------------

echo "Model: $MODEL"
echo "Mode:  $MODE"
echo "========================================"

python htm_distance/htm_distance_test_model.py \
    --model  "$MODEL" \
    --mode   "$MODE" \
    --data_root   ../UB_keystroke_dataset/ \
    --bots_root   ../BadUSBdataset \
    --write_decision_log
