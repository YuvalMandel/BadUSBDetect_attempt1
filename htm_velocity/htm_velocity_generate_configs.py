"""
htm_velocity/htm_velocity_generate_configs.py
Generate random HTM-Velocity hyperparameter configs and a SLURM script.

Extends the HTM-Combined search space with 6 additional poly error encoder HPs:
  poly_mean_enc_bits / poly_mean_enc_w  (mean   |predicted - actual| flight)
  poly_med_enc_bits  / poly_med_enc_w   (median)
  poly_std_enc_bits  / poly_std_enc_w   (std)

DEFAULT_CONFIG uses hc0195 (best-ever HTM-Combined architecture) params with al=5,
which was the hc round-6 DEFAULT, and adds poly encoder defaults.

Cumulative workflow — results are never deleted:
  Each run finds the highest config_idx in hv_results/, deletes old
  hv_configs/*.json, then writes new configs numbered from last_idx+1.

Usage (from project root):
  python htm_velocity/htm_velocity_generate_configs.py [--n-configs 128] [--seed 0]

Outputs:
  hv_configs/config_NNNN.json …
  slurm/hv_submit_array.sh
  slurm/hv_prepare_windows.sh
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Search space
# ──────────────────────────────────────────────────────────────────────────────
PARAM_SPACE = {
    # SpatialPooler
    "sp_columnDimensions":      [2048],
    "sp_numActiveColumns":      [25, 30, 35],
    "sp_potentialPct":          [0.65, 0.80],
    "sp_synPermActiveInc":      [0.02, 0.05],
    "sp_synPermConnected":      [0.10, 0.20],
    "sp_synPermInactiveDec":    [0.003, 0.005, 0.010],
    # TemporalMemory
    "tm_cellsPerColumn":        [16, 32],
    "tm_activationThreshold":   [10, 13],
    "tm_initialPermanence":     [0.21, 0.31, 0.40],
    "tm_connectedPermanence":   [0.30, 0.50],
    "tm_minThreshold":          [8, 10],
    "tm_maxNewSynapseCount":    [20, 25, 30],
    "tm_permanenceIncrement":   [0.05, 0.10],
    "tm_permanenceDecrement":   [0.05, 0.10],
    # Stats-channel encoders (7 features each, independent per channel)
    "dwell_stats_enc_bits":     [16, 24, 32],
    "dwell_stats_enc_w":        [5, 7, 9],
    "flight_stats_enc_bits":    [16, 24, 32],
    "flight_stats_enc_w":       [5, 7, 9],
    "dist_stats_enc_bits":      [16, 24, 32],
    "dist_stats_enc_w":         [5, 7, 9],
    # Per-keystroke scalar encoders (1 feature each)
    "dwell_scalar_enc_bits":    [8, 16],
    "dwell_scalar_enc_w":       [3, 5, 7],
    "flight_scalar_enc_bits":   [8, 16],
    "flight_scalar_enc_w":      [3, 5, 7],
    "dist_scalar_enc_bits":     [8, 16],
    "dist_scalar_enc_w":        [3, 5, 7],
    # Poly error encoder channels (1 feature each, fixed range 0-500 ms)
    "poly_mean_enc_bits":       [8, 16],
    "poly_mean_enc_w":          [3, 5, 7],
    "poly_med_enc_bits":        [8, 16],
    "poly_med_enc_w":           [3, 5, 7],
    "poly_std_enc_bits":        [8, 16],
    "poly_std_enc_w":           [3, 5, 7],
    # Window
    "window_size":              [10, 15],
    "window_step":              [1],
    # Detection
    "warmup_steps":             [2, 3],
    "al_period":                [5, 10, 15],
}

# Default: hc0195 combined params (best-ever test F1=0.9444) with al=5 and poly defaults.
DEFAULT_CONFIG = {
    "sp_columnDimensions":    2048,
    "sp_numActiveColumns":    25,
    "sp_potentialPct":        0.80,
    "sp_synPermActiveInc":    0.05,
    "sp_synPermConnected":    0.10,
    "sp_synPermInactiveDec":  0.010,
    "tm_cellsPerColumn":      16,
    "tm_activationThreshold": 13,
    "tm_initialPermanence":   0.31,
    "tm_connectedPermanence": 0.50,
    "tm_minThreshold":        10,
    "tm_maxNewSynapseCount":  20,
    "tm_permanenceIncrement": 0.10,
    "tm_permanenceDecrement": 0.10,
    "dwell_stats_enc_bits":   24,
    "dwell_stats_enc_w":      5,
    "flight_stats_enc_bits":  24,
    "flight_stats_enc_w":     7,
    "dist_stats_enc_bits":    32,
    "dist_stats_enc_w":       9,
    "dwell_scalar_enc_bits":  8,
    "dwell_scalar_enc_w":     5,
    "flight_scalar_enc_bits": 8,
    "flight_scalar_enc_w":    7,
    "dist_scalar_enc_bits":   8,
    "dist_scalar_enc_w":      3,
    "poly_mean_enc_bits":     8,
    "poly_mean_enc_w":        3,
    "poly_med_enc_bits":      8,
    "poly_med_enc_w":         3,
    "poly_std_enc_bits":      8,
    "poly_std_enc_w":         3,
    "window_size":            10,
    "window_step":            1,
    "warmup_steps":           2,
    "al_period":              5,
    "seed":                   42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Validity constraints
# ──────────────────────────────────────────────────────────────────────────────
_ENC_PAIRS = [
    ("dwell_stats_enc_w",    "dwell_stats_enc_bits"),
    ("flight_stats_enc_w",   "flight_stats_enc_bits"),
    ("dist_stats_enc_w",     "dist_stats_enc_bits"),
    ("dwell_scalar_enc_w",   "dwell_scalar_enc_bits"),
    ("flight_scalar_enc_w",  "flight_scalar_enc_bits"),
    ("dist_scalar_enc_w",    "dist_scalar_enc_bits"),
    ("poly_mean_enc_w",      "poly_mean_enc_bits"),
    ("poly_med_enc_w",       "poly_med_enc_bits"),
    ("poly_std_enc_w",       "poly_std_enc_bits"),
]


def is_valid(cfg: dict) -> bool:
    for w_key, bits_key in _ENC_PAIRS:
        if cfg[w_key] >= cfg[bits_key]:
            return False
    if cfg["tm_minThreshold"] > cfg["tm_activationThreshold"]:
        return False
    if cfg["tm_activationThreshold"] > cfg["tm_maxNewSynapseCount"]:
        return False
    return True


def sample_config(rng: random.Random):
    for _ in range(1000):
        cfg = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        if is_valid(cfg):
            return cfg
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Find next available config index from completed results
# ──────────────────────────────────────────────────────────────────────────────
def find_start_idx(results_dir: str = "hv_results") -> int:
    max_idx = -1
    for fpath in glob.glob(os.path.join(results_dir, "hv*.json")):
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
# SLURM scripts
# ──────────────────────────────────────────────────────────────────────────────
def write_slurm_script(n_configs: int, out_path: str):
    script = f"""\
#!/bin/bash
# ============================================================
# SLURM job array for HTM-Velocity hyperparameter search
# Generated by htm_velocity/htm_velocity_generate_configs.py
#
# Submit with:
#   sbatch slurm/hv_submit_array.sh
# ============================================================
#SBATCH --job-name=hv_htm
#SBATCH --output=logs/hv_%A_%a.out
#SBATCH --error=logs/hv_%A_%a.err
#SBATCH --array=0-{n_configs - 1}%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=16:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Run this task's config ------------------------------------
mapfile -t CONFIGS < <(ls hv_configs/config_*.json | sort)
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

python -u htm_velocity/htm_velocity_train_single.py --config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:
        fh.write(script)
    print(f"SLURM script  -> {out_path}")


def write_prepare_windows_slurm_script(out_path: str = "slurm/hv_prepare_windows.sh"):
    script = """\
#!/bin/bash
# ============================================================
# SLURM job: HTM-Velocity windows cache preparation
# Generated by htm_velocity/htm_velocity_generate_configs.py
#
# Run ONCE before submitting the hyperparameter search array.
# Safe to re-run -- existing cache files are skipped.
#
# Submit with:
#   sbatch slurm/hv_prepare_windows.sh
# ============================================================
#SBATCH --job-name=hv_prep_win
#SBATCH --output=logs/hv_prepare_windows_%A_%a.out
#SBATCH --error=logs/hv_prepare_windows_%A_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Run --------------------------------------------------------
echo "========================================"
echo "Job   : $SLURM_JOB_ID"
echo "Task  : $SLURM_ARRAY_TASK_ID"
echo "Node  : $(hostname)"
echo "Start : $(date)"
echo "========================================"

python -u htm_velocity/htm_velocity_prepare_windows.py --task-id $SLURM_ARRAY_TASK_ID

echo "========================================"
echo "End   : $(date)"
echo "========================================"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:
        fh.write(script)
    print(f"SLURM script  -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Generate HTM-Velocity hyperparameter configs and SLURM script")
    parser.add_argument("--n-configs", type=int, default=128,
                        help="Number of new configs to generate (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for config sampling (default: 0)")
    args = parser.parse_args()

    os.makedirs("hv_configs", exist_ok=True)
    os.makedirs("logs",       exist_ok=True)

    start_idx = find_start_idx("hv_results")
    print(f"Completed hv runs: {start_idx}  (next config index: {start_idx})")

    old_configs = glob.glob("hv_configs/config_*.json")
    if old_configs:
        for f in old_configs:
            os.remove(f)
        print(f"Deleted {len(old_configs)} old config file(s) from hv_configs/")

    rng     = random.Random(args.seed + start_idx)
    configs = []
    seen:  set = set()

    # Always include DEFAULT_CONFIG first
    default = DEFAULT_CONFIG.copy()
    configs.append(default)
    seen.add(json.dumps(
        {k: v for k, v in default.items() if k != "seed"}, sort_keys=True))

    while len(configs) < args.n_configs:
        cfg = sample_config(rng)
        if cfg is None:
            print("WARNING: Could not sample more valid configs.")
            break
        cfg["seed"] = rng.randint(0, 99999)
        key = json.dumps(
            {k: v for k, v in cfg.items() if k != "seed"}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    for i, cfg in enumerate(configs):
        global_idx = start_idx + i
        path = f"hv_configs/config_{global_idx:04d}.json"
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)

    n = len(configs)
    print(f"Generated {n} new configs -> hv_configs/  "
          f"(indices {start_idx}-{start_idx + n - 1})")
    write_slurm_script(n, "slurm/hv_submit_array.sh")
    write_prepare_windows_slurm_script("slurm/hv_prepare_windows.sh")

    print(f"\nWorkflow:")
    print(f"  1. python htm_velocity/htm_velocity_prepare_data.py")
    print(f"     (builds velocity_cache.pkl — run once)")
    print(f"  2. sbatch slurm/hv_prepare_windows.sh")
    print(f"     (pre-computes window sequences — run once)")
    print(f"  3. sbatch slurm/hv_submit_array.sh        # submit {n} jobs")
    print(f"  4. python htm_velocity/htm_velocity_collect_results.py")


if __name__ == "__main__":
    main()
