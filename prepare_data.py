"""
prepare_data.py
One-time data preparation for the HTM hyperparameter search.
Run this ONCE (locally or on a login node) before submitting SLURM jobs.

Creates / updates:
  split.pkl          — Reproducible train / val / test file lists
  features_cache.pkl — Per-file keystroke feature sequences (window_size=15)

Usage:
  python prepare_data.py
"""

import glob
import multiprocessing
import os
import pickle

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from htm_common import (
    FOLDERS, DEFAULT_WINDOW_SIZE, DEFAULT_STEP_HUMAN, DEFAULT_STEP_BOT,
    is_task1, create_reference_pool, process_file_worker, build_split,
)

SPLIT_FILE = "split.pkl"
CACHE_FILE = "features_cache.pkl"


def main():
    print("=" * 60)
    print("  HTM Data Preparation")
    print("=" * 60)

    # ---- 1. Locate files ----
    print("\n[1] Locating files...")
    human_files = []
    for folder in FOLDERS["Humans"]:
        files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        human_files.extend(f for f in files if is_task1(f))

    bot_files = []
    for folder in FOLDERS["Bots"]:
        bot_files.extend(
            glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))

    print(f"  Human files : {len(human_files)}")
    print(f"  Bot files   : {len(bot_files)}")

    if not human_files:
        print("ERROR: No human files found. Check FOLDERS paths in htm_common.py.")
        return

    # ---- 2. Build / load split ----
    print("\n[2] Building split...")
    if os.path.exists(SPLIT_FILE):
        print(f"  {SPLIT_FILE} already exists — loading (delete to regenerate).")
        with open(SPLIT_FILE, 'rb') as fh:
            split = pickle.load(fh)
    else:
        split = build_split(human_files, bot_files)
        with open(SPLIT_FILE, 'wb') as fh:
            pickle.dump(split, fh)
        print(f"  Saved → {SPLIT_FILE}")

    print(f"  Train : {len(split['train_human'])} human files")
    print(f"  Val   : {len(split['val_human'])} human, "
          f"{len(split['val_bots'])} bot files")
    print(f"  Test  : {len(split['test_human'])} human, "
          f"{len(split['test_bots'])} bot files")

    # ---- 3. Reference pool (train humans only) ----
    print("\n[3] Building reference pool...")
    ref_dwells, ref_flights = create_reference_pool(
        split['train_human'], window_size=DEFAULT_WINDOW_SIZE)

    # ---- 4. Incremental feature extraction ----
    print("\n[4] Feature extraction (window_size=15)...")
    file_features = {}
    if os.path.exists(CACHE_FILE):
        print(f"  Loading existing cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as fh:
            file_features = pickle.load(fh)
        print(f"  {len(file_features)} files already cached.")

    all_needed = (
        [(fp, DEFAULT_STEP_HUMAN, ref_dwells, ref_flights, DEFAULT_WINDOW_SIZE)
         for fp in (split['train_human'] + split['val_human'] +
                    split['test_human'])]
        + [(fp, DEFAULT_STEP_BOT, ref_dwells, ref_flights, DEFAULT_WINDOW_SIZE)
           for fp in (split['val_bots'] + split['test_bots'])]
    )
    missing = [a for a in all_needed if a[0] not in file_features]

    if missing:
        print(f"  Extracting features for {len(missing)} files...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file_worker, a) for a in missing]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="  Processing"):
                fp, feats = future.result()
                if len(feats) > 0:
                    file_features[fp] = feats

        with open(CACHE_FILE, 'wb') as fh:
            pickle.dump(file_features, fh)
        print(f"  Cache saved → {CACHE_FILE}  ({len(file_features)} total files)")
    else:
        print("  All files already cached, nothing to do.")

    print("\nDone. Ready to generate configs and submit SLURM jobs.")
    print("  Next steps:")
    print("    python generate_configs.py")
    print("    sbatch slurm/submit_array.sh")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # needed on Windows
    main()
