"""
htm_distance/htm_distance_prepare_data.py
One-time data preparation for the HTM-distance variant.

Reads the same split.pkl as the original HTM so F1 scores are comparable.
Parses every file into a sequence of (key_idx, dwell_ms, flight_ms, dist_units)
tuples and saves them to dist_cache.pkl.

Usage (from project root):
  python htm_distance/htm_distance_prepare_data.py

Outputs:
  dist_cache.pkl   — dict: filepath → list of (key_idx, dwell, flight, dist)
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_dist_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_dist_dir not in sys.path:
    sys.path.insert(0, _htm_dist_dir)

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from common.keystroke_features import FOLDERS, is_task1
from htm_distance_common import parse_file_distance

SPLIT_FILE = "split.pkl"
CACHE_FILE = "dist_cache.pkl"


# Top-level worker (must be picklable for ProcessPoolExecutor)
def _worker(filepath: str):
    return filepath, parse_file_distance(filepath)


def main():
    os.chdir(_project_root)

    print("=" * 60)
    print("  HTM-Distance Data Preparation")
    print("=" * 60)

    # ── 1. Load split (same as original HTM) ──────────────────────
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: '{SPLIT_FILE}' not found. Run htm/htm_prepare_data.py first.")
        return

    with open(SPLIT_FILE, 'rb') as fh:
        split = pickle.load(fh)

    train_human = split['train_human']
    val_human   = split['val_human']
    test_human  = split['test_human']
    val_bots    = split['val_bots']
    test_bots   = split['test_bots']

    all_files = (train_human + val_human + test_human
                 + val_bots + test_bots)

    print(f"\nSplit loaded from {SPLIT_FILE}")
    print(f"  Train : {len(train_human)} human files")
    print(f"  Val   : {len(val_human)} human  +  {len(val_bots)} bot files")
    print(f"  Test  : {len(test_human)} human  +  {len(test_bots)} bot files")
    print(f"  Total : {len(all_files)} files to parse")

    # ── 2. Incremental extraction (skip already-cached files) ──────
    dist_cache: dict = {}
    if os.path.exists(CACHE_FILE):
        print(f"\nLoading existing cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as fh:
            dist_cache = pickle.load(fh)
        print(f"  {len(dist_cache)} files already cached.")

    missing = [f for f in all_files if f not in dist_cache]

    if missing:
        print(f"\nParsing {len(missing)} files (parallel)...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_worker, fp) for fp in missing]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="  Parsing"):
                fp, events = future.result()
                if events:
                    dist_cache[fp] = events

        with open(CACHE_FILE, 'wb') as fh:
            pickle.dump(dist_cache, fh)
        print(f"  Cache saved → {CACHE_FILE}  ({len(dist_cache)} total files)")
    else:
        print("\nAll files already cached — nothing to do.")

    # ── 3. Quick stats ─────────────────────────────────────────────
    total_events = sum(len(v) for v in dist_cache.values())
    print(f"\nTotal keystroke events cached : {total_events:,}")
    if dist_cache:
        sample = next(iter(dist_cache.values()))[:3]
        print(f"Sample events (key_idx, dwell, flight, dist):")
        for e in sample:
            print(f"  {e}")

    print("\nDone. Next:")
    print("  python htm_distance/htm_distance_generate_configs.py --n-configs 128")
    print("  sbatch slurm/dist_submit_array.sh")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
