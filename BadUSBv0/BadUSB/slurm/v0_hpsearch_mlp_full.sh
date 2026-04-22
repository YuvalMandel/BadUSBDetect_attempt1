#!/bin/bash
# ============================================================
# SLURM job: MLP — full-mode HP search (Optuna, 50 trials)
#   Uses existing fullkey CSVs (already generated).
#   cd BadUSBv0/BadUSB && sbatch slurm/v0_hpsearch_mlp_full.sh
# ============================================================
#SBATCH --job-name=v0_mlp_hpsearch
#SBATCH --output=logs/v0_mlp_hpsearch_%j.out
#SBATCH --error=logs/v0_mlp_hpsearch_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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
echo "Task  : MLP full-mode HP search (50 trials, tag=fullkey)"
echo "WORK  : $WORK"
echo "Start : $(date)"
echo "========================================"

cd "$WORK/MLP"

echo "--- HP search + final training (mode=full, tag=fullkey, 50 trials) ---"
python -X utf8 model_training.py \
    --split-json "$WORK/data_split.json" \
    --mode full \
    --tag fullkey \
    --search \
    --n-configs 50

echo "========================================"
echo "End : $(date)"
echo "========================================"
