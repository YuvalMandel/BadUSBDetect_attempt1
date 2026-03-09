"""
htm_combined/htm_combined_generate_configs.py
Generate random HTM-Combined hyperparameter configs and a SLURM script.

Search space refined from round-5 results (384 runs total, 159/384 deployable):
  Best (Round 5): hc0191 — sp=25 pct=0.8, d16w9/f16w5/q24w7, dk16w5/fk16w5/qk8w7,
                            ws=10s1, tm=16, act=10, wu=2, al=5
                  → Best deployable: test_f1=0.9189, val_bacc=0.9667, live_thresh=0.030
  Non-deployable best: hc0195 — same sp but d24w5/f24w7/q32w9, dk8w5/fk8w7/qk8w3,
                            ws=10, act=13, wu=2, al=10
                  → test_f1=0.9444 but live_thresh=0 (not deployable)

New in Round 3: ALL 24 scalar-encoded parameters have independent enc_bits/enc_w:
  - 3 stats channels × (enc_bits, enc_w):
      dwell_stats_enc_bits / dwell_stats_enc_w    (7 dwell-time statistics)
      flight_stats_enc_bits / flight_stats_enc_w  (7 flight-time statistics)
      dist_stats_enc_bits   / dist_stats_enc_w    (7 QWERTY-distance statistics)
  - 3 per-keystroke scalars × (enc_bits, enc_w):
      dwell_scalar_enc_bits / dwell_scalar_enc_w  (last-key dwell time, 0-400 ms)
      flight_scalar_enc_bits/ flight_scalar_enc_w (last-key flight time, 0-500 ms)
      dist_scalar_enc_bits  / dist_scalar_enc_w   (last-key QWERTY dist, 0-12 u)

Round-6 analysis (384 runs, 159 deployable):
  window_size: ws=10 → 47% deployable, ws=5 → 32%, ws=15 → 47%
               → narrow to [10] (drop ws=5; ws=15 kept for re-evaluation)
  al_period: al=5 → 42% (80 dep), al=10 → 38% (57 dep), al=15 → 50% (22 dep)
             → keep [5, 10, 15] (al=15 surprise: highest rate, but small sample)
  warmup: wu=2 → 45% deployable, wu=3 → 37%   → bias toward [2], keep [2, 3]
  tm_act: act=10 → 45% deployable, act=13 → 37% → keep [10, 13]
  synPermActiveInc: 0.02 → 40%, 0.05 → 41%    → no signal, keep [0.02, 0.05]
  Best deployable test_f1=0.9189: hc0191 (ws=10, al=5, act=10, wu=2)
  Round-6 goal: test hc0195's architecture (test_f1=0.9444) with al=5 as DEFAULT
                and explore ws=15/al=15 combinations for higher deployability

Cumulative workflow — results are never deleted:
  Each run finds the highest config_idx in hc_results/, deletes old
  hc_configs/*.json, then writes new configs numbered from last_idx+1.

Usage (from project root):
  python htm_combined/htm_combined_generate_configs.py [--n-configs 128] [--seed 0]

Outputs:
  hc_configs/config_NNNN.json …
  slurm/hc_submit_array.sh
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Search space (Round 6 — DEFAULT=hc0195+al5, ws=[10,15], al=[5,10,15])
# ──────────────────────────────────────────────────────────────────────────────
PARAM_SPACE = {
    # SpatialPooler
    "sp_columnDimensions":     [2048],
    "sp_numActiveColumns":     [25, 30, 35],
    "sp_potentialPct":         [0.65, 0.80],
    "sp_synPermActiveInc":     [0.02, 0.05],
    "sp_synPermConnected":     [0.10, 0.20],
    "sp_synPermInactiveDec":   [0.003, 0.005, 0.010],
    # TemporalMemory
    "tm_cellsPerColumn":       [16, 32],
    "tm_activationThreshold":  [10, 13],
    "tm_initialPermanence":    [0.21, 0.31, 0.40],
    "tm_connectedPermanence":  [0.30, 0.50],
    "tm_minThreshold":         [8, 10],
    "tm_maxNewSynapseCount":   [20, 25, 30],
    "tm_permanenceIncrement":  [0.05, 0.10],
    "tm_permanenceDecrement":  [0.05, 0.10],
    # Stats-channel encoders — per physical quantity (7 features each)
    # All 3 channels have independent resolution settings
    "dwell_stats_enc_bits":    [16, 24, 32],
    "dwell_stats_enc_w":       [5, 7, 9],
    "flight_stats_enc_bits":   [16, 24, 32],
    "flight_stats_enc_w":      [5, 7, 9],
    "dist_stats_enc_bits":     [16, 24, 32],
    "dist_stats_enc_w":        [5, 7, 9],
    # Per-keystroke scalar encoders — per physical quantity (1 feature each)
    # scalar_enc_bits=8 dominated Round 2 (75%); 16 kept for coverage; 24 dropped
    "dwell_scalar_enc_bits":   [8, 16],
    "dwell_scalar_enc_w":      [3, 5, 7],
    "flight_scalar_enc_bits":  [8, 16],
    "flight_scalar_enc_w":     [3, 5, 7],
    "dist_scalar_enc_bits":    [8, 16],
    "dist_scalar_enc_w":       [3, 5, 7],
    # Window — ws=10 → 47% deployable, ws=5 → 32% (dropped); ws=15 re-added (47%)
    "window_size":             [10, 15],
    "window_step":             [1],
    # Detection — warmup=0/1 produced degenerate configs; wu=2 → 45%, wu=3 → 37%
    "warmup_steps":            [2, 3],
    # al=5→42%, al=10→38%, al=15→50% (small sample but promising); keep all three
    "al_period":               [5, 10, 15],
}

# Default: hc0195 params with al=5 — key Round-6 experiment.
# hc0195 achieved test_f1=0.9444 (best ever) but live_thresh=0 (al=10).
# Testing its exact architecture with al=5 to check if it becomes deployable.
# (hc0191 — best deployable — is kept as reference: test_f1=0.9189, val_bacc=0.9667)
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


def sample_config(rng: random.Random) -> dict | None:
    for _ in range(1000):
        cfg = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        if is_valid(cfg):
            return cfg
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Find next available config index from completed results
# ──────────────────────────────────────────────────────────────────────────────
def find_start_idx(results_dir: str = "hc_results") -> int:
    max_idx = -1
    for fpath in glob.glob(os.path.join(results_dir, "hc*.json")):
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
# SLURM job array for HTM-Combined hyperparameter search
# Generated by htm_combined/htm_combined_generate_configs.py
#
# Submit with:
#   sbatch slurm/hc_submit_array.sh
# ============================================================
#SBATCH --job-name=hc_htm
#SBATCH --output=logs/hc_%A_%a.out
#SBATCH --error=logs/hc_%A_%a.err
#SBATCH --array=0-{n_configs - 1}%200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

# ---- Environment ------------------------------------------------
source $(conda info --base)/etc/profile.d/conda.sh
conda activate htm_keyboard_1

# ---- Run this task's config ------------------------------------
mapfile -t CONFIGS < <(ls hc_configs/config_*.json | sort)
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

python -u htm_combined/htm_combined_train_single.py --config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='\n') as fh:
        fh.write(script)
    print(f"SLURM script  -> {out_path}")


def write_prepare_windows_slurm_script(out_path: str = "slurm/hc_prepare_windows.sh"):
    script = """\
#!/bin/bash
# ============================================================
# SLURM job: HTM-Combined windows cache preparation
# Generated by htm_combined/htm_combined_generate_configs.py
#
# Run ONCE before submitting the hyperparameter search array.
# Safe to re-run -- existing cache files are skipped.
#
# Submit with:
#   sbatch slurm/hc_prepare_windows.sh
# ============================================================
#SBATCH --job-name=hc_prep_win
#SBATCH --output=logs/hc_prepare_windows_%A_%a.out
#SBATCH --error=logs/hc_prepare_windows_%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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

python -u htm_combined/htm_combined_prepare_windows.py --task-id $SLURM_ARRAY_TASK_ID

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
        description="Generate HTM-Combined hyperparameter configs and SLURM script")
    parser.add_argument("--n-configs", type=int, default=128,
                        help="Number of new configs to generate (default: 128)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for config sampling (default: 0)")
    args = parser.parse_args()

    os.makedirs("hc_configs", exist_ok=True)
    os.makedirs("logs",       exist_ok=True)

    start_idx = find_start_idx("hc_results")
    print(f"Completed hc runs: {start_idx}  (next config index: {start_idx})")

    old_configs = glob.glob("hc_configs/config_*.json")
    if old_configs:
        for f in old_configs:
            os.remove(f)
        print(f"Deleted {len(old_configs)} old config file(s) from hc_configs/")

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

    for i, cfg in enumerate(configs):
        global_idx = start_idx + i
        path = f"hc_configs/config_{global_idx:04d}.json"
        with open(path, 'w') as fh:
            json.dump(cfg, fh, indent=2)

    n = len(configs)
    print(f"Generated {n} new configs -> hc_configs/  "
          f"(indices {start_idx}-{start_idx + n - 1})")
    write_slurm_script(n, "slurm/hc_submit_array.sh")
    write_prepare_windows_slurm_script("slurm/hc_prepare_windows.sh")

    print(f"\nWorkflow:")
    print(f"  1. python htm_combined/htm_combined_generate_configs.py "
          f"--n-configs {args.n_configs}")
    print(f"  2. sbatch slurm/hc_prepare_windows.sh"
          f"   # one-time; skips existing caches")
    print(f"  3. sbatch slurm/hc_submit_array.sh        # submit {n} jobs")
    print(f"  4. python htm_combined/htm_combined_collect_results.py")


if __name__ == "__main__":
    main()
