#!/bin/bash
# ============================================================
# SLURM job: HTM — full training, no keystroke truncation
#   Trains best config (cfg_0002) with unlimited keystroke windows.
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_fullkey_htm.sh
# ============================================================
#SBATCH --job-name=v0_htm_fullkey
#SBATCH --output=logs/v0_htm_fullkey_%j.out
#SBATCH --error=logs/v0_htm_fullkey_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

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
echo "Config: cfg_0002 (best) | tag: fullkey | no keystroke limit"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

echo "--- Building HTM windows cache (no keystroke limit) ---"
cd "$WORK"
python -X utf8 HTM/htm_prepare_data.py --tag fullkey

echo "--- Training HTM cfg_0002 (tag=fullkey) ---"
python -X utf8 HTM/htm_train.py \
    --config HTM/configs/config_0002.json \
    --cache  HTM/windows_cache_fullkey.pkl \
    --tag    fullkey

echo "========================================"
echo "End : $(date)"
echo "========================================"
