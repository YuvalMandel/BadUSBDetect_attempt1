#!/bin/bash
# ============================================================
# SLURM job: HTM-Velocity model evaluation
#
# Runs both all_non_train and all_other_files modes sequentially.
# all_other_files uses SLURM_CPUS_PER_TASK workers via mp.Pool.
#
# Edit MODEL (and optionally DATA_ROOT / BOTS_ROOT) then submit:
#   sbatch slurm/hv_test_model.sh
# ============================================================
#SBATCH --job-name=hv_test
#SBATCH --output=logs/hv_test_%j.out
#SBATCH --error=logs/hv_test_%j.err
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Edit these -------------------------------------------------
MODEL="hv_models/hv0102_sp30_d24w9_f16w9_q24w7_dk8w3_fk8w3_qk16w7_pm16w7_pd8w5_ps16w5_ws15s1_tm16_act10_wu2_al10_vf10.9952_tf10.9964.pkl"
DATA_ROOT="../UB_keystroke_dataset/"
BOTS_ROOT="../BadUSBdataset"
# -----------------------------------------------------------------

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "Model : $MODEL"
echo "Start : $(date)"
echo "========================================"

echo ""
echo "--- Mode: all_non_train ---"
python -u htm_velocity/htm_velocity_test_model.py \
    --model "$MODEL" \
    --mode  all_non_train

echo ""
echo "--- Mode: all_other_files ---"
python -u htm_velocity/htm_velocity_test_model.py \
    --model     "$MODEL" \
    --mode      all_other_files \
    --data_root "$DATA_ROOT" \
    --bots_root "$BOTS_ROOT"

echo "========================================"
echo "End   : $(date)"
echo "========================================"
