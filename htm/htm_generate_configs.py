"""
htm/htm_generate_configs.py
Generate random HTM hyperparameter configurations and the SLURM submission script.

Cumulative workflow — results are NEVER deleted:
  Each run finds the highest config_idx already present in results/,
  deletes old (already-run) configs/config_*.json files, then writes
  new configs numbered from last_idx + 1 onward.

Usage:
  python htm/htm_generate_configs.py [--n-configs 128] [--seed 0]

Outputs (relative to project root):
  configs/config_NNNN.json … (N new configs, starting after last completed)
  slurm/submit_array.sh
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

# ------------------------------------------------------------------
# Search space — Round 6 (after 254 runs)
#
# Eliminated by evidence across all rounds:
#   enc_bits=32   — never appeared in top-6
#   tm_cells=32   — never appeared in top-6
#   sp_act=10,15  — never appeared in top-6
#   enc_w=4       — never appeared in top-6
#   boost>0       — killed all configs in round 1
#   AL=True       — never helped
#
# Hot zone from top-6 analysis:
#   sp_act: 20 (ranks 2,5), 30 (rank 4), 40 (ranks 1,3,6) → add 25,35,45
#   pct:    0.7 (3/6), 0.75 (1/6), 0.6 (1/6), 0.5 (1/6) → add 0.65, 0.80
#   enc_w:  5 (2/6), 6 (2/6), 7 (1/6)
#   tm_cells: 16 (3/6), 8 (3/6) — both viable
#   act:    13 (2/6), 14,15,18,20 (1/6 each) — full range valid
#   min:    10 (4/6), 12 (1/6) — 10 dominant
# ------------------------------------------------------------------
PARAM_SPACE = {
    # SpatialPooler
    "sp_columnDimensions":   [2048],
    # Round 6: removed 10,15 (never top-6); added 25,35,45 to probe around hot zone 30-40
    "sp_numActiveColumns":   [20, 25, 30, 35, 40, 45],
    # Round 6: added 0.65,0.80 to bracket the 0.7-0.75 sweet spot
    "sp_potentialPct":       [0.5, 0.60, 0.65, 0.70, 0.75, 0.80],
    "sp_boostStrength":      [0.0],                # fixed: boost>0 kills learning
    "sp_synPermActiveInc":   [0.02, 0.05, 0.10],
    # Round 6: added 0.15 between 0.10 and 0.20
    "sp_synPermConnected":   [0.10, 0.15, 0.20],
    "sp_synPermInactiveDec": [0.003, 0.005, 0.010],
    # TemporalMemory
    # Round 6: removed 32 (never top-6); 8 and 16 both appear in top-6
    "tm_cellsPerColumn":      [8, 16],
    "tm_activationThreshold": [13, 14, 15, 18, 20],
    "tm_minThreshold":        [10, 11, 12],
    # Round 6: added 25 to probe between 20 and 30
    "tm_maxNewSynapseCount":  [15, 20, 25, 30],
    "tm_initialPermanence":   [0.21, 0.31, 0.40],
    "tm_connectedPermanence": [0.30, 0.50],
    "tm_permanenceIncrement": [0.05, 0.10, 0.20],
    "tm_permanenceDecrement": [0.03, 0.05, 0.10],
    # Encoder
    # Round 6: removed enc_bits=32 (never top-6); removed enc_w=4 (never top-6)
    "enc_bits_per_feature": [16],
    "enc_w":                [5, 6, 7],
    # Detection strategy — AL never helped; fixed to False
    "use_anomaly_likelihood": [False],
    # Warmup: initial windows skipped before any alarm can fire
    "warmup_steps": [3, 5, 8, 12, 20],
}

# DEFAULT_CONFIG = best config found so far (cfg0083: test F1=0.6207)
DEFAULT_CONFIG = {
    "sp_columnDimensions":   2048,
    "sp_numActiveColumns":   40,
    "sp_potentialPct":       0.70,
    "sp_boostStrength":      0.0,
    "sp_synPermActiveInc":   0.05,
    "sp_synPermConnected":   0.10,
    "sp_synPermInactiveDec": 0.005,
    "tm_cellsPerColumn":     16,
    "tm_activationThreshold":13,
    "tm_minThreshold":       10,
    "tm_maxNewSynapseCount": 20,
    "tm_initialPermanence":  0.21,
    "tm_connectedPermanence":0.50,
    "tm_permanenceIncrement":0.10,
    "tm_permanenceDecrement":0.10,
    "enc_bits_per_feature":  16,
    "enc_w":                 5,
    "detection_mode":         "first_crossing",
    "use_anomaly_likelihood": False,
    "al_learning_period":     20,
    "warmup_steps":           5,
    "seed":                   42,
}


# ------------------------------------------------------------------
# Constraint checker
# ------------------------------------------------------------------
def is_valid(cfg):
    """HTM requires certain ordering constraints between parameters."""
    if cfg["enc_w"] >= cfg["enc_bits_per_feature"]:
        return False
    if cfg["tm_minThreshold"] > cfg["tm_activationThreshold"]:
        return False
    if cfg["tm_activationThreshold"] > cfg["tm_maxNewSynapseCount"]:
        return False
    return True


def sample_config(rng):
    for _ in range(1000):
        cfg = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        if is_valid(cfg):
            return cfg
    return None


# ------------------------------------------------------------------
# Find the next available config index from completed results
# ------------------------------------------------------------------
def find_start_idx(results_dir="results"):
    """
    Scans results/cfg*.json to find the highest config_idx already completed.
    Returns that index + 1 (i.e., where the next batch should start).
    Returns 0 if no results exist yet.
    """
    max_idx = -1
    for fpath in glob.glob(os.path.join(results_dir, "cfg*.json")):
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
def write_slurm_script(n_configs, out_path):
    script = f"""\
#!/bin/bash
# ============================================================
# SLURM job array for HTM hyperparameter search
# Generated by htm/htm_generate_configs.py
#
# Submit with:
#   sbatch slurm/submit_array.sh
# ============================================================
#SBATCH --job-name=htm_search
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-{n_configs - 1}%200  # max 200 concurrent tasks (stays under 300-CPU QOS limit)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
# Adjust partition / account to match your Newton allocation:
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source /home/yuval.mandel/HWSecurity/BadUSBDetect_attempt1/.venv/bin/activate

# ---- Run this task's config ------------------------------------
# Array index → config file (zero-padded, sorted alphabetically)
# Old configs are deleted before each generation run, so configs/ only
# contains the current batch; array task IDs map 1:1 to sorted files.
mapfile -t CONFIGS < <(ls configs/config_*.json | sort)
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

python htm/htm_train_single.py --config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:   # Unix line-endings for bash
        fh.write(script)
    print(f"SLURM script  → {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Run from project root so configs/ and slurm/ are at root level
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Generate HTM hyperparameter configs and SLURM script")
    parser.add_argument("--n-configs", type=int, default=128,
                        help="Number of NEW configs to generate (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for config sampling (default: 0)")
    args = parser.parse_args()

    os.makedirs("configs", exist_ok=True)
    os.makedirs("logs",    exist_ok=True)

    # ---- Find where to continue from ----
    start_idx = find_start_idx("results")
    print(f"Existing completed runs: {start_idx}  (next config index: {start_idx})")

    # ---- Delete old (already-run) config files ----
    old_configs = glob.glob("configs/config_*.json")
    if old_configs:
        for f in old_configs:
            os.remove(f)
        print(f"Deleted {len(old_configs)} old config file(s) from configs/")

    # ---- Sample new configs ----
    rng = random.Random(args.seed + start_idx)   # seed shifts so we don't repeat configs
    configs = []
    seen    = set()

    # Always include the DEFAULT_CONFIG as the first new entry
    default = DEFAULT_CONFIG.copy()
    default["seed"] = 42
    configs.append(default)
    seen.add(json.dumps({k: v for k, v in default.items()
                         if k not in ("seed", "detection_mode", "al_learning_period")},
                        sort_keys=True))

    while len(configs) < args.n_configs:
        cfg = sample_config(rng)
        if cfg is None:
            print("WARNING: Could not sample more valid configs.")
            break
        cfg["seed"] = rng.randint(0, 99999)
        # Deduplicate on search-space keys only (exclude seed/detection_mode)
        key = json.dumps({k: v for k, v in cfg.items()
                          if k not in ("seed", "detection_mode", "al_learning_period")},
                         sort_keys=True)
        if key not in seen:
            seen.add(key)
            # Add fixed fields not in PARAM_SPACE
            cfg["detection_mode"]    = "first_crossing"
            cfg["al_learning_period"] = 20
            configs.append(cfg)

    # ---- Write config files with globally unique indices ----
    for i, cfg in enumerate(configs):
        global_idx = start_idx + i
        path = f"configs/config_{global_idx:04d}.json"
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)

    n = len(configs)
    print(f"Generated {n} new configs → configs/  "
          f"(indices {start_idx}–{start_idx + n - 1})")
    write_slurm_script(n, "slurm/submit_array.sh")

    print(f"\nWorkflow:")
    print(f"  1. python htm/htm_generate_configs.py --n-configs {args.n_configs}")
    print(f"  2. sbatch slurm/submit_array.sh        # submit {n} jobs")
    print(f"  3. python htm/htm_collect_results.py   # reads ALL results/ (cumulative)")


if __name__ == "__main__":
    main()
