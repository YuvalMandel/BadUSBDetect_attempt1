#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 GRU pipeline
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_train_gru.sh
# ============================================================
#SBATCH --job-name=v0_gru
#SBATCH --output=logs/v0_gru_%j.out
#SBATCH --error=logs/v0_gru_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -e

ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

# WORK = BadUSBv0/BadUSB/ (the directory sbatch was run from)
WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

echo "--- Building RNN tensors ---"
cd "$WORK/GRU"
python -X utf8 translate_to_tensors.py --split-json "$WORK/data_split.json"

echo "--- Training GRU ---"
python -X utf8 train_gru.py --split-json "$WORK/data_split.json"

echo "========================================"
echo "End : $(date)"
echo "========================================"
