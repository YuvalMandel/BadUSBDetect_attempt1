#!/bin/bash
# ============================================================
# SLURM job: Collect HTM hyperparameter search results
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_collect_htm.sh
#
# Reads:   results/HTM/results/htm_model_*.json
# Writes:  results/HTM/leaderboard.txt
#          results/HTM/leaderboard.csv
#          results/HTM/final_test_results.json   (best model on test)
#          results/HTM/plots/cfg_NNNN_final_confusion.png
# ============================================================
#SBATCH --job-name=v0_htm_collect
#SBATCH --output=logs/v0_htm_collect_%j.out
#SBATCH --error=logs/v0_htm_collect_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

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
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

cd "$WORK"
echo "--- Collecting HTM results ---"
python -X utf8 HTM/htm_collect_results.py

echo "========================================"
echo "End : $(date)"
echo "========================================"
