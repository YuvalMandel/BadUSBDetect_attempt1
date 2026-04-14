#!/bin/bash
# ============================================================
# SLURM job: HTM-Combined model evaluation
#
# Edit MODEL, MODE (and optionally DATA_ROOT / BOTS_ROOT)
# then submit with:
#   sbatch slurm/hc_test_model.sh
#
# all_other_files mode uses SLURM_CPUS_PER_TASK workers
# automatically for parallel feature extraction.
# orig and all_non_train modes are single-threaded.
# ============================================================
#SBATCH --job-name=hc_test
#SBATCH --output=logs/hc_test_%j.out
#SBATCH --error=logs/hc_test_%j.err
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Edit these -------------------------------------------------
MODEL="hc_models/hc0191_sp25_d16w9_f16w5_q24w7_dk16w5_fk16w5_qk8w7_ws10s1_tm16_act10_wu2_al5_vf10.7273_tf10.9189.pkl"
MODE="all_other_files"        # orig | all_non_train | all_other_files
DATA_ROOT="../UB_keystroke_dataset/"
BOTS_ROOT="Synthetic_Bots/"
# -----------------------------------------------------------------

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "Model : $MODEL"
echo "Mode  : $MODE"
echo "Start : $(date)"
echo "========================================"

python -u htm_combined/htm_combined_test_model.py \
    --model    "$MODEL" \
    --mode     "$MODE"  \
    --data_root  "$DATA_ROOT" \
    --bots_root  "$BOTS_ROOT"

echo "========================================"
echo "End   : $(date)"
echo "========================================"
