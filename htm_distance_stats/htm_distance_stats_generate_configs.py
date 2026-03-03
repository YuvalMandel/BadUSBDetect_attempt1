"""
htm_distance_stats/htm_distance_stats_generate_configs.py
Generate random HTM-Distance-Stats hyperparameter configs and a SLURM script.

Search space combines:
  * SP / TM parameters from the htm_distance/ Round-5 space
  * Encoder: enc_bits_per_feature, enc_w (for 21-feature StatsEncoder)
  * Window:  window_size (MUST be small — bot files have 3–10 printable keys)
             window_step (sliding stride)
  * Detection: warmup_steps (in windows, not keystrokes)
               al_period    (AnomalyLikelihood history)

Cumulative workflow — results are never deleted:
  Each run finds the highest config_idx already present in ds_results/,
  removes old ds_configs/*.json files, then writes new configs numbered
  from last_idx + 1 onward.

Usage (from project root):
  python htm_distance_stats/htm_distance_stats_generate_configs.py [--n-configs 128] [--seed 0]

Outputs:
  ds_configs/config_NNNN.json …
  slurm/ds_submit_array.sh
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Round-1 search space
#
# Window size: 10–30 keystrokes (like htm/ which defaults to 15).
# No warmup: AnomalyLikelihood smooths the first-window spike; human files
# are long so no windows need to be discarded.
# Bot files shorter than window_size produce zero windows → short-file rule
# auto-misclassifies them (same as htm/ behaviour with window_size=15).
# ──────────────────────────────────────────────────────────────────────────────
PARAM_SPACE = {
    # SpatialPooler
    "sp_columnDimensions":    [2048],
    "sp_numActiveColumns":    [20, 25, 30, 35, 40],
    "sp_potentialPct":        [0.65, 0.80],
    "sp_synPermActiveInc":    [0.02, 0.05, 0.10],
    "sp_synPermConnected":    [0.10, 0.20],
    "sp_synPermInactiveDec":  [0.003, 0.005, 0.010],
    # TemporalMemory
    "tm_cellsPerColumn":      [16, 32],
    "tm_activationThreshold": [10, 13],
    "tm_initialPermanence":   [0.21, 0.31, 0.40],
    "tm_connectedPermanence": [0.30, 0.50],
    "tm_minThreshold":        [8, 10],
    "tm_maxNewSynapseCount":  [15, 20, 25, 30],
    "tm_permanenceIncrement": [0.05, 0.10],
    "tm_permanenceDecrement": [0.05, 0.10],
    # Encoder (for 21-dim StatsEncoder)
    "enc_bits_per_feature":   [16, 24, 32],
    "enc_w":                  [5, 7, 9],
    # Window — same range as htm/ (default 15, max 30)
    "window_size":            [10, 15, 20, 25, 30],
    "window_step":            [1],   # sliding stride (fixed for Round 1)
    # AnomalyLikelihood history window (warmup is fixed at 0)
    "al_period":              [5, 10, 15, 20],
}

# Default config: mirrors htm/ best config (cfg0083) with 21-dim encoder
DEFAULT_CONFIG = {
    "sp_columnDimensions":    2048,
    "sp_numActiveColumns":    40,
    "sp_potentialPct":        0.80,
    "sp_synPermActiveInc":    0.05,
    "sp_synPermConnected":    0.10,
    "sp_synPermInactiveDec":  0.005,
    "tm_cellsPerColumn":      16,
    "tm_activationThreshold": 13,
    "tm_initialPermanence":   0.21,
    "tm_connectedPermanence": 0.50,
    "tm_minThreshold":        8,
    "tm_maxNewSynapseCount":  20,
    "tm_permanenceIncrement": 0.10,
    "tm_permanenceDecrement": 0.10,
    "enc_bits_per_feature":   16,
    "enc_w":                  5,
    "window_size":            15,   # same default as htm/
    "window_step":            1,
    # warmup_steps absent → defaults to 0 (WARMUP_STEPS constant)
    "al_period":              15,
    "seed":                   42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Validity constraints
# ──────────────────────────────────────────────────────────────────────────────
def is_valid(cfg: dict) -> bool:
    if cfg["enc_w"] >= cfg["enc_bits_per_feature"]:
        return False
    if cfg["tm_minThreshold"] > cfg["tm_activationThreshold"]:
        return False
    if cfg["tm_activationThreshold"] > cfg["tm_maxNewSynapseCount"]:
        return False
    return True


def sample_config(rng: random.Random) -> dict | None:
    for _ in range(1000):
        cfg = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        if is_valid(cfg):
            return cfg
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Find next available config index from completed results
# ──────────────────────────────────────────────────────────────────────────────
def find_start_idx(results_dir: str = "ds_results") -> int:
    max_idx = -1
    for fpath in glob.glob(os.path.join(results_dir, "ds*.json")):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            idx = r.get("config_idx", -1)
            if isinstance(idx, int) and idx > max_idx:
                max_idx = idx
        except Exception:
            pass
    return max_idx + 1


# ──────────────────────────────────────────────────────────────────────────────
# SLURM script
# ──────────────────────────────────────────────────────────────────────────────
def write_slurm_script(n_configs: int, out_path: str):
    script = f"""\
#!/bin/bash
# ============================================================
# SLURM job array for HTM-Distance-Stats hyperparameter search
# Generated by htm_distance_stats/htm_distance_stats_generate_configs.py
#
# Submit with:
#   sbatch slurm/ds_submit_array.sh
# ============================================================
#SBATCH --job-name=ds_htm
#SBATCH --output=logs/ds_%A_%a.out
#SBATCH --error=logs/ds_%A_%a.err
#SBATCH --array=0-{n_configs - 1}%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Run this task's config ------------------------------------
mapfile -t CONFIGS < <(ls ds_configs/config_*.json | sort)
CONFIG="${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: No config for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "========================================"
echo "Task  : $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Node  : $(hostname)"
echo "========================================"

python htm_distance_stats/htm_distance_stats_train_single.py --config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:
        fh.write(script)
    print(f"SLURM script  → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Generate HTM-Distance-Stats hyperparameter configs and SLURM script")
    parser.add_argument("--n-configs", type=int, default=128,
                        help="Number of new configs to generate (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for config sampling (default: 0)")
    args = parser.parse_args()

    os.makedirs("ds_configs", exist_ok=True)
    os.makedirs("logs",       exist_ok=True)

    # ---- Find where to continue from ----
    start_idx = find_start_idx("ds_results")
    print(f"Completed ds runs: {start_idx}  (next config index: {start_idx})")

    # ---- Delete old (already-run) config files ----
    old_configs = glob.glob("ds_configs/config_*.json")
    if old_configs:
        for f in old_configs:
            os.remove(f)
        print(f"Deleted {len(old_configs)} old config file(s) from ds_configs/")

    # ---- Sample new configs ----
    rng = random.Random(args.seed + start_idx)
    configs = []
    seen: set = set()

    # Always include DEFAULT_CONFIG first
    default = DEFAULT_CONFIG.copy()
    configs.append(default)
    seen.add(json.dumps(
        {k: v for k, v in default.items() if k != "seed"},
        sort_keys=True))

    while len(configs) < args.n_configs:
        cfg = sample_config(rng)
        if cfg is None:
            print("WARNING: Could not sample more valid configs.")
            break
        cfg["seed"] = rng.randint(0, 99999)
        key = json.dumps(
            {k: v for k, v in cfg.items() if k != "seed"},
            sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    # ---- Write config files ----
    for i, cfg in enumerate(configs):
        global_idx = start_idx + i
        path = f"ds_configs/config_{global_idx:04d}.json"
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)

    n = len(configs)
    print(f"Generated {n} new configs → ds_configs/  "
          f"(indices {start_idx}–{start_idx + n - 1})")
    write_slurm_script(n, "slurm/ds_submit_array.sh")

    print(f"\nWorkflow:")
    print(f"  1. python htm_distance_stats/htm_distance_stats_prepare_data.py")
    print(f"  2. python htm_distance_stats/htm_distance_stats_generate_configs.py "
          f"--n-configs {args.n_configs}")
    print(f"  3. sbatch slurm/ds_submit_array.sh        # submit {n} jobs")
    print(f"  4. python htm_distance_stats/htm_distance_stats_collect_results.py")


if __name__ == "__main__":
    main()
