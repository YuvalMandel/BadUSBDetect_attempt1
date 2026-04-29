#!/bin/bash
# ============================================================
# SLURM job: GRU — threshold sweep on saved fullkey model
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_tunethresh_gru.sh
# ============================================================
#SBATCH --job-name=v0_gru_tune
#SBATCH --output=logs/v0_gru_tune_%j.out
#SBATCH --error=logs/v0_gru_tune_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

set -e

ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Task  : GRU threshold sweep (mode=full, tag=fullkey)"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/GRU"

python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --hps-json "$WORK/results/GRU/gru_best_hps_fullkey.json" \
    --tune-threshold

echo "========================================"
echo "End : $(date)"
echo "========================================"
