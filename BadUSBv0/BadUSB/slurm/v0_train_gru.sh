#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 GRU pipeline
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_train_gru.sh
#
# Pipeline env vars (set by v0_full_pipeline.sh):
#   PIPELINE_MODE   = partial|full   (default: partial)
#   PIPELINE_SEARCH = --search       (default: empty / disabled)
# ============================================================
#SBATCH --job-name=v0_gru
#SBATCH --output=logs/v0_gru_%j.out
#SBATCH --error=logs/v0_gru_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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

# Accept overrides from v0_full_pipeline.sh or use defaults
MODE="${PIPELINE_MODE:-partial}"
SEARCH="${PIPELINE_SEARCH:-}"
N_CONFIGS="${PIPELINE_N_CONFIGS:-50}"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Mode  : $MODE  |  HP search: ${SEARCH:-disabled}  |  n-configs: $N_CONFIGS"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

echo "--- Building RNN tensors (mode=$MODE) ---"
cd "$WORK/GRU"
python -X utf8 translate_to_tensors.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE"

echo "--- Training GRU (mode=$MODE ${SEARCH:-}) ---"
python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode "$MODE" \
    --n-configs "$N_CONFIGS" \
    $SEARCH

echo "========================================"
echo "End : $(date)"
echo "========================================"
