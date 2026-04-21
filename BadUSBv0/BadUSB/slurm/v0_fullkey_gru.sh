#!/bin/bash
# ============================================================
# SLURM job: GRU — full training, no keystroke truncation
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_fullkey_gru.sh
# ============================================================
#SBATCH --job-name=v0_gru_fullkey
#SBATCH --output=logs/v0_gru_fullkey_%j.out
#SBATCH --error=logs/v0_gru_fullkey_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

set -e

ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}$LD_LIBRARY_PATH"

python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null || {
    echo "Installing CUDA-enabled PyTorch (cu121)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall -q
}

WORK="$SLURM_SUBMIT_DIR"

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Mode  : full | tag: fullkey | HP search: 32 configs"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

echo "--- Building RNN tensors (mode=full, no keystroke limit) ---"
cd "$WORK/GRU"
python -X utf8 translate_to_tensors.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey

echo "--- Training GRU (mode=full, tag=fullkey, search 32 configs) ---"
python -X utf8 train_gru.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --search \
    --n-configs 32

echo "========================================"
echo "End : $(date)"
echo "========================================"
