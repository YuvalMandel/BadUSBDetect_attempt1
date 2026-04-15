#!/bin/bash
# ============================================================
# SLURM job: BadUSBv0 MLP pipeline
#
# Runs from project root:
#   sbatch slurm/v0_train_mlp.sh
#
# Prerequisites on the cluster:
#   - data_split.json exists (run split_persons.py locally and push, or
#     run `python BadUSBv0/BadUSB/split_persons.py` as a pre-step)
#   - dataset_generator/Synthetic_Bots/ and s2/ are present
# ============================================================
#SBATCH --job-name=v0_mlp
#SBATCH --output=logs/v0_mlp_%j.out
#SBATCH --error=logs/v0_mlp_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

WORK=BadUSBv0/BadUSB

echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Node  : $(hostname)"
echo "CPUs  : $SLURM_CPUS_PER_TASK"
echo "Start : $(date)"
echo "========================================"

# ── Step 1: generate bots + humans (fast, skip if already done) ──────────────
if [ ! -d "$WORK/dataset_generator/Synthetic_Bots" ]; then
    echo "--- Generating bots ---"
    cd $WORK/dataset_generator && python -X utf8 bot_generator.py -o Synthetic_Bots -f 25 -e 80
    python -X utf8 bot_generator.py -o Synthetic_Bots_test -f 5 -e 80
    python -X utf8 human_generator.py -o Balanced_Humans -f 124 -l 80 -e 1
    python -X utf8 human_generator.py -o Balanced_Humans_test -f 24 -l 80 -e 0
    cd -
fi

# ── Step 2: person-disjoint split (skip if already done) ─────────────────────
if [ ! -f "$WORK/data_split.json" ]; then
    echo "--- Creating person-disjoint split ---"
    cd $WORK && python -X utf8 split_persons.py --bots-dir dataset_generator/Synthetic_Bots
    cd -
fi

# ── Step 3: train regressor ───────────────────────────────────────────────────
echo "--- Training polynomial regressor ---"
cd $WORK/MLP/regressor
python -X utf8 regressor_train.py \
    -hu ../../dataset_generator/Balanced_Humans \
    -m  poly_regressor.pkl
python -X utf8 test_regressor.py \
    -hu ../../dataset_generator/Balanced_Humans_test \
    -b  ../../dataset_generator/Synthetic_Bots_test \
    -m  poly_regressor.pkl
cp poly_regressor.pkl ../poly_regressor.pkl
cd -

# ── Step 4: feature extraction → 3 CSVs ──────────────────────────────────────
echo "--- Feature extraction (person-disjoint splits) ---"
cd $WORK/MLP
python -X utf8 dataset_csv_generator.py --split-json ../data_split.json

# ── Step 5: train MLP ─────────────────────────────────────────────────────────
echo "--- Training MLP ---"
python -X utf8 model_training.py
cd -

echo "========================================"
echo "End : $(date)"
echo "========================================"
