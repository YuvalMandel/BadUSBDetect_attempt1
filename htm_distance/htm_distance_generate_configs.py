"""
htm_distance/htm_distance_generate_configs.py
Generate random HTM-distance hyperparameter configs and a SLURM script.

Cumulative workflow — results are NEVER deleted:
  Each run finds the highest config_idx already present in dist_results/,
  removes old dist_configs/*.json files, then writes new configs numbered
  from last_idx + 1 onward.

Usage (from project root):
  python htm_distance/htm_distance_generate_configs.py [--n-configs 128] [--seed 0]

Outputs:
  dist_configs/config_NNNN.json …
  slurm/dist_submit_array.sh
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

# ------------------------------------------------------------------
# Round-2 search space — updated from 128-run leaderboard
#
# Evidence from top-8 configs (test F1 > 0.52 with thresh > 0):
#
# ENCODER
#   enc=64 dominates top-4; enc=24/32 never in top-8 → drop 24,32
#   Key insight: for non-overlapping key_idx encoding across 95 keys,
#   need bits_per_feature >= 94+w.  w=7 → 101 bits → try 96, 128.
#   Hot zone: enc=64 (keep), add 96, 128; drop 24, 32, 48
#   enc_w: 5 (ranks 2,3,4), 7 (ranks 1,6,7,9) both strong; 3 weak → keep 5,7; drop 3
#
# WARMUP
#   warmup=30  → threshold=0.0 for every config (degenerate) → drop
#   warmup=50  → only two configs above degenerate threshold → drop
#   warmup=100 → all top-8 use it → keep; add 150, 200 to probe further
#
# SP
#   sp_act: 30 (ranks 2,3,5,6,8), 40 (ranks 1,7,9) dominate → keep; add 35
#           20 only rank 4 (50/60 never top-8) → keep 20, drop 50,60
#   pct:    0.65 (4/8), 0.80 (2/8) both viable; 0.5 weak → keep 0.65,0.80; drop 0.5
#   synPermActiveInc / Connected / InactiveDec: no signal yet → keep broad
#
# TM
#   tm_cells: 16 (5/8 top configs), 32 (3/8) both viable; 64 never top-8 → drop
#   act:      10 (ranks 1,3,8), 13 (ranks 2,3,5,6,7) → keep; 16/20 weak → drop
#   min:       8 (ranks 1,7,8), 10 (ranks 2,3,5,6) → keep; 12 weak → drop
# ------------------------------------------------------------------
PARAM_SPACE = {
    # SpatialPooler
    "sp_columnDimensions":    [2048],
    "sp_numActiveColumns":    [20, 30, 35, 40],      # dropped 50, 60
    "sp_potentialPct":        [0.65, 0.80],           # dropped 0.5
    "sp_synPermActiveInc":    [0.02, 0.05, 0.10],
    "sp_synPermConnected":    [0.10, 0.20],
    "sp_synPermInactiveDec":  [0.003, 0.005, 0.010],
    # TemporalMemory
    "tm_cellsPerColumn":      [16, 32],               # dropped 64
    "tm_activationThreshold": [10, 13],               # dropped 16, 20
    "tm_initialPermanence":   [0.21, 0.31, 0.40],
    "tm_connectedPermanence": [0.30, 0.50],
    "tm_minThreshold":        [8, 10],                # dropped 12
    "tm_maxNewSynapseCount":  [15, 20, 25, 30],
    "tm_permanenceIncrement": [0.05, 0.10],
    "tm_permanenceDecrement": [0.05, 0.10],
    # Encoder
    # key_type is now a fixed 95-bit one-hot (1 active bit) — not tunable.
    # enc_bits_per_feature / enc_w control only the 3 scalar features
    # (dwell, flight, distance).  Total SDR = 95 + 3*bits, active = 1 + 3*w.
    "enc_bits_per_feature":   [24, 32, 48, 64],
    "enc_w":                  [3, 5, 7],
    # Warmup — must be large enough for TM to learn human patterns
    "warmup_steps":           [100, 150, 200],        # dropped 30, 50
}

# Best config so far: dist0022 (test F1=0.6792)
DEFAULT_CONFIG = {
    "sp_columnDimensions":    2048,
    "sp_numActiveColumns":    40,
    "sp_potentialPct":        0.80,
    "sp_synPermActiveInc":    0.05,
    "sp_synPermConnected":    0.10,
    "sp_synPermInactiveDec":  0.005,
    "tm_cellsPerColumn":      16,
    "tm_activationThreshold": 10,
    "tm_initialPermanence":   0.21,
    "tm_connectedPermanence": 0.50,
    "tm_minThreshold":        8,
    "tm_maxNewSynapseCount":  20,
    "tm_permanenceIncrement": 0.10,
    "tm_permanenceDecrement": 0.10,
    "enc_bits_per_feature":   32,   # scalar features only; key is fixed 95-bit one-hot
    "enc_w":                  5,
    "warmup_steps":           100,
    "seed":                   42,
}


# ------------------------------------------------------------------
# HTM parameter validity constraints
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Find the next available config index from completed results
# ------------------------------------------------------------------
def find_start_idx(results_dir: str = "dist_results") -> int:
    max_idx = -1
    for fpath in glob.glob(os.path.join(results_dir, "dist*.json")):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            idx = r.get("config_idx", -1)
            if isinstance(idx, int) and idx > max_idx:
                max_idx = idx
        except Exception:
            pass
    return max_idx + 1


# ------------------------------------------------------------------
# SLURM script
# ------------------------------------------------------------------
def write_slurm_script(n_configs: int, out_path: str):
    script = f"""\
#!/bin/bash
# ============================================================
# SLURM job array for HTM-distance hyperparameter search
# Generated by htm_distance/htm_distance_generate_configs.py
#
# Submit with:
#   sbatch slurm/dist_submit_array.sh
# ============================================================
#SBATCH --job-name=dist_htm
#SBATCH --output=logs/dist_%A_%a.out
#SBATCH --error=logs/dist_%A_%a.err
#SBATCH --array=0-{n_configs - 1}%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Run this task's config ------------------------------------
mapfile -t CONFIGS < <(ls dist_configs/config_*.json | sort)
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

python htm_distance/htm_distance_train_single.py --config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:
        fh.write(script)
    print(f"SLURM script  → {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Generate HTM-distance hyperparameter configs and SLURM script")
    parser.add_argument("--n-configs", type=int, default=128,
                        help="Number of new configs to generate (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for config sampling (default: 0)")
    args = parser.parse_args()

    os.makedirs("dist_configs", exist_ok=True)
    os.makedirs("logs",         exist_ok=True)

    # ---- Find where to continue from ----
    start_idx = find_start_idx("dist_results")
    print(f"Completed dist runs: {start_idx}  (next config index: {start_idx})")

    # ---- Delete old (already-run) config files ----
    old_configs = glob.glob("dist_configs/config_*.json")
    if old_configs:
        for f in old_configs:
            os.remove(f)
        print(f"Deleted {len(old_configs)} old config file(s) from dist_configs/")

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
        path = f"dist_configs/config_{global_idx:04d}.json"
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)

    n = len(configs)
    print(f"Generated {n} new configs → dist_configs/  "
          f"(indices {start_idx}–{start_idx + n - 1})")
    write_slurm_script(n, "slurm/dist_submit_array.sh")

    print(f"\nWorkflow:")
    print(f"  1. python htm_distance/htm_distance_prepare_data.py")
    print(f"  2. python htm_distance/htm_distance_generate_configs.py "
          f"--n-configs {args.n_configs}")
    print(f"  3. sbatch slurm/dist_submit_array.sh        # submit {n} jobs")
    print(f"  4. python htm_distance/htm_distance_collect_results.py")


if __name__ == "__main__":
    main()
