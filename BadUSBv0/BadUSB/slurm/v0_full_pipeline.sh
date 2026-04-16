#!/bin/bash
# ============================================================
# Full BadUSBv0 pipeline — submits all SLURM jobs with
# dependency chaining so they run in the correct order.
#
# Run from BadUSBv0/BadUSB/:
#   bash slurm/v0_full_pipeline.sh [OPTIONS]
#
# Options:
#   --mode partial|full   Data balance mode (default: partial)
#                           partial = undersample humans to match bots
#                           full    = all windows, class-weighted loss
#   --search              Enable Optuna HP search for MLP + GRU
#                           (~10-20 min extra per model, val F1 objective)
#   --n-configs N         HTM configs to search (default: 128)
#
# Examples:
#   bash slurm/v0_full_pipeline.sh
#   bash slurm/v0_full_pipeline.sh --mode full --search --n-configs 256
# ============================================================

set -euo pipefail
WORK=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORK"

# ── Parse arguments ────────────────────────────────────────────────────────────
MODE="partial"
SEARCH=""
N_CONFIGS=128

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";         shift 2 ;;
        --search)    SEARCH="--search"; shift   ;;
        --n-configs) N_CONFIGS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1  ;;
    esac
done

echo "========================================"
echo "BadUSBv0 Full Pipeline"
echo "  Mode      : $MODE"
echo "  HP search : ${SEARCH:-disabled}"
echo "  n-configs : $N_CONFIGS  (same for MLP Optuna, GRU Optuna, HTM random search)"
echo "  Submit dir: $WORK"
echo "========================================"
mkdir -p logs

# ── Step 1: Generate all data (bots, split, regressor) ───────────────────────
JOB_DATA=$(sbatch --parsable slurm/v0_generate_data.sh)
echo "Step 1  — Generate data    : job $JOB_DATA"

# ── Step 2a: MLP  (depends on data) ──────────────────────────────────────────
JOB_MLP=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    --export=ALL,PIPELINE_MODE=$MODE,PIPELINE_SEARCH="$SEARCH",PIPELINE_N_CONFIGS=$N_CONFIGS \
    slurm/v0_train_mlp.sh)
echo "Step 2a — MLP training     : job $JOB_MLP"

# ── Step 2b: GRU  (depends on data, runs in parallel with MLP) ───────────────
JOB_GRU=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    --export=ALL,PIPELINE_MODE=$MODE,PIPELINE_SEARCH="$SEARCH",PIPELINE_N_CONFIGS=$N_CONFIGS \
    slurm/v0_train_gru.sh)
echo "Step 2b — GRU training     : job $JOB_GRU"

# ── Step 2c: Generate HTM configs (fast — run directly on login node) ─────────
echo "Step 2c — Generating $N_CONFIGS HTM configs (random search)..."
python -X utf8 HTM/htm_generate_configs.py --n-configs "$N_CONFIGS"

# ── Step 2d: HTM HP search array (depends on data, parallel with MLP/GRU) ────
JOB_HTM=$(sbatch --parsable \
    --dependency=afterok:$JOB_DATA \
    slurm/v0_train_htm_array.sh)
echo "Step 2d — HTM HP search    : job array $JOB_HTM"

# ── Step 3: Collect HTM results (depends on entire array finishing) ───────────
JOB_COLLECT=$(sbatch --parsable \
    --dependency=afterok:$JOB_HTM \
    slurm/v0_collect_htm.sh)
echo "Step 3  — HTM collect      : job $JOB_COLLECT"

echo ""
echo "All jobs submitted. Dependency chain:"
echo "  $JOB_DATA (data) → $JOB_MLP (MLP)"
echo "  $JOB_DATA (data) → $JOB_GRU (GRU)"
echo "  $JOB_DATA (data) → $JOB_HTM (HTM search) → $JOB_COLLECT (collect)"
echo ""
echo "Monitor progress:  squeue -u \$USER"
echo "Pipeline complete when $JOB_MLP, $JOB_GRU, and $JOB_COLLECT all finish."
