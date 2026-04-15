#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 GRU pipeline
#
# Run AFTER v0_train_mlp.sh (needs data_split.json).
#   sbatch slurm/v0_train_gru.sh
# ============================================================
#SBATCH --job-name=v0_gru
#SBATCH --output=logs/v0_gru_%j.out
#SBATCH --error=logs/v0_gru_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

WORK=BadUSBv0/BadUSB

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "========================================"

# ── Step 1: build tensors from person-disjoint splits ────────────────────────
echo "--- Building RNN tensors ---"
cd $WORK/GRU
python -X utf8 translate_to_tensors.py --split-json ../data_split.json

# ── Step 2: train GRU ─────────────────────────────────────────────────────────
echo "--- Training GRU ---"
python -X utf8 train_gru.py
cd -

echo "========================================"
echo "End : $(date)"
echo "========================================"
