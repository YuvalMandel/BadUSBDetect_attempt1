"""
Generate random GRU HP configs and SLURM array / collect scripts.

Run from BadUSBv0/BadUSB/:
  python GRU/gru_generate_hp_configs.py [--n-configs 50] [--seed 42] [--tag fullkey]

Already-done trials (results/GRU/hp_results/trial_NNNN.json) are reported but not skipped
at generation time — each array task checks for its own result and skips if present.
"""
import argparse
import glob
import json
import math
import os
import random
from pathlib import Path

DROPOUT_VALS = [round(0.10 + i * 0.05, 2) for i in range(7)]   # 0.10 .. 0.40
THRESH_VALS  = [round(0.30 + i * 0.05, 2) for i in range(12)]  # 0.30 .. 0.85


def sample_config(rng: random.Random) -> dict:
    return {
        "hidden_dim": rng.choice([64, 128, 256]),
        "num_layers": 2,  # fixed: 2 layers was optimal in both partial and full mode
        "dropout":    rng.choice(DROPOUT_VALS),
        "lr":         math.exp(rng.uniform(math.log(5e-5), math.log(2e-3))),
        "batch_size": rng.choice([16, 32, 64]),
        "threshold":  rng.choice(THRESH_VALS),
    }


def find_done(results_dir: str) -> set:
    done = set()
    for f in glob.glob(os.path.join(results_dir, "trial_*.json")):
        try:
            done.add(int(os.path.basename(f)[len("trial_"):-len(".json")]))
        except ValueError:
            pass
    return done


def write_array_script(n: int, tag: str, out_path: str) -> None:
    tag_arg    = f"--tag {tag}" if tag else ""
    tag_suffix = f"_{tag}" if tag else ""
    script = f"""\
#!/bin/bash
# GRU HP-search array: each task trains 30 epochs with one random HP config.
# Submit from BadUSBv0/BadUSB/: sbatch slurm/v0_gru_hp_array{tag_suffix}.sh
#SBATCH --job-name=v0_gru_hp
#SBATCH --output=logs/v0_gru_hp_%A_%a.out
#SBATCH --error=logs/v0_gru_hp_%A_%a.err
#SBATCH --array=0-{n - 1}%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

set -e
ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${{CONDA_PREFIX:+$CONDA_PREFIX/lib:}}$LD_LIBRARY_PATH"

WORK="$SLURM_SUBMIT_DIR"
CONFIG=$(printf "gru_hp_configs/config_%04d.json" $SLURM_ARRAY_TASK_ID)
echo "Task $SLURM_ARRAY_TASK_ID | config: $CONFIG"
cd "$WORK/GRU"
python -X utf8 train_gru.py \\
    --split-json "$WORK/data_split.json" \\
    --mode full \\
    {tag_arg} \\
    --trial-config "$CONFIG"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(script, encoding="utf-8")
    print(f"Array script   -> {out_path}")


def write_collect_script(tag: str, out_path: str) -> None:
    tag_arg    = f"--tag {tag}" if tag else ""
    tag_suffix = f"_{tag}" if tag else ""
    script = f"""\
#!/bin/bash
# GRU: collect HP trial results -> best HPs JSON -> final 50-epoch training.
# Submit from BadUSBv0/BadUSB/: sbatch slurm/v0_gru_hp_collect{tag_suffix}.sh
#SBATCH --job-name=v0_gru_collect
#SBATCH --output=logs/v0_gru_collect_%j.out
#SBATCH --error=logs/v0_gru_collect_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1

set -e
ENV_NAME=htm_keyboard_1
export PATH="$HOME/miniconda3/envs/$ENV_NAME/bin:$HOME/anaconda3/envs/$ENV_NAME/bin:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate $ENV_NAME 2>/dev/null || true
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/$ENV_NAME/lib:${{CONDA_PREFIX:+$CONDA_PREFIX/lib:}}$LD_LIBRARY_PATH"

WORK="$SLURM_SUBMIT_DIR"
cd "$WORK/GRU"

echo "=== Collecting HP trial results ==="
python -X utf8 train_gru.py --mode full {tag_arg} --collect

echo "=== Final 50-epoch training with best HPs ==="
python -X utf8 train_gru.py \\
    --split-json "$WORK/data_split.json" \\
    --mode full \\
    {tag_arg} \\
    --hps-json "$WORK/results/GRU/gru_best_hps{tag_suffix}.json"
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(script, encoding="utf-8")
    print(f"Collect script -> {out_path}")


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)

    ap = argparse.ArgumentParser(description="Generate GRU HP search configs + SLURM scripts")
    ap.add_argument("--n-configs", type=int, default=50)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--tag",       default="fullkey")
    args = ap.parse_args()

    configs_dir = os.path.join(_here, "gru_hp_configs")
    results_dir = os.path.join(_root, "results", "GRU", "hp_results")
    slurm_dir   = os.path.join(_root, "slurm")

    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(slurm_dir,   exist_ok=True)

    done = find_done(results_dir)
    print(f"Already completed: {len(done)}/{args.n_configs} trials")

    rng = random.Random(args.seed)
    written = 0
    for i in range(args.n_configs):
        cfg = sample_config(rng)
        cfg["trial_idx"] = i
        path = os.path.join(configs_dir, f"config_{i:04d}.json")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                json.dump(cfg, fh, indent=2)
            written += 1

    print(f"Config files: {written} new written (total {args.n_configs} in {configs_dir})")

    tag_suffix = f"_{args.tag}" if args.tag else ""
    write_array_script(args.n_configs, args.tag,
                       os.path.join(slurm_dir, f"v0_gru_hp_array{tag_suffix}.sh"))
    write_collect_script(args.tag,
                         os.path.join(slurm_dir, f"v0_gru_hp_collect{tag_suffix}.sh"))


if __name__ == "__main__":
    main()
