"""
htm_distance/htm_distance_generate_configs.py
Generate random HTM-distance hyperparameter configs and a SLURM script.

Fresh-start workflow (delete old results before re-running):
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
# Round-5 search space — bot files have very few valid keystrokes
#
# DIAGNOSIS (Round 4, 89 runs): ALL configs produced identical val F1=0.1509
# and test F1=0.5217 with thresh=0.0000.  Identical results regardless of any
# hyperparameter means the model's anomaly scores do NOT discriminate humans
# from bots — all decisions are made by the short-file rule:
#
#   * "Long" bot files (len > warmup) → predicted Bot at thresh=0 (TP)
#   * "Short" bot files (len ≤ warmup) → auto-predicted Human (FN)
#   * Human files (all long)          → predicted Bot at thresh=0 (FP)
#
# F1 is then purely a function of (long_bots / all_files), which is constant
# regardless of hyperparameters.  The fix: minimize warmup so more bot files
# become "long" and enter the actual detection zone.
#
# Most BadUSB files contain only a handful of printable keystrokes after
# filtering (many bot keystrokes are modifiers, function keys, ctrl+v, etc.
# that parse_file_distance drops).  Observed bot files often have < 5 valid
# keystrokes → warmup must be ≤ 2 to have any post-warmup data for them.
#
# WARMUP: [1, 2, 3, 5]
#   warmup=0 is excluded: step 0 always has anomaly=1.0 (TM has no prior
#   context) → first_crossing at any thresh ≤ 1 flags everything as Bot.
#   warmup=1 skips only that forced-1.0 step; warmup=2–3 give the TM
#   slightly more context before the detection zone.
#
# SP / TM / ENCODER: unchanged from Round 4 — no signal yet to guide pruning.
# ------------------------------------------------------------------
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
    # Encoder (scalar features only; key block is fixed 95-bit one-hot)
    "enc_bits_per_feature":   [16, 24],
    "enc_w":                  [5, 7, 9],
    # Warmup: must be tiny so even short bot files have post-warmup data.
    # warmup=0 excluded (step 0 anomaly=1.0 makes every file a "Bot").
    "warmup_steps":           [1, 2, 3, 5],
    # AnomalyLikelihood history window.
    # AL converts raw TM anomaly → probability of being anomalous relative
    # to the distribution seen during training on human files.
    # Small period → fast adaptation; range capped at 20 (bot files are short).
    "al_period":              [5, 10, 15, 20],
}

# Round-5 default: same SP/TM/enc as best Round-2 config (dist0027),
# warmup=2 (skip only the first two forced-high-anomaly steps)
DEFAULT_CONFIG = {
    "sp_columnDimensions":    2048,
    "sp_numActiveColumns":    30,
    "sp_potentialPct":        0.80,
    "sp_synPermActiveInc":    0.05,
    "sp_synPermConnected":    0.10,
    "sp_synPermInactiveDec":  0.005,
    "tm_cellsPerColumn":      32,
    "tm_activationThreshold": 13,
    "tm_initialPermanence":   0.21,
    "tm_connectedPermanence": 0.50,
    "tm_minThreshold":        8,
    "tm_maxNewSynapseCount":  20,
    "tm_permanenceIncrement": 0.10,
    "tm_permanenceDecrement": 0.10,
    "enc_bits_per_feature":   24,   # scalar features; key is fixed 95-bit one-hot
    "enc_w":                  7,
    "warmup_steps":           2,    # skip only first 2 forced-high-anomaly steps
    "al_period":              10,   # AnomalyLikelihood history window
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
