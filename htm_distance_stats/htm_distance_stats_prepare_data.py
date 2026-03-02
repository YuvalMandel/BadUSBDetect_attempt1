"""
htm_distance_stats/htm_distance_stats_prepare_data.py
One-time data preparation for the HTM-Distance-Stats variant.

Reads split.pkl and parses every file into per-keystroke event sequences:
  (key_idx, dwell_ms, flight_ms, dist_units)

Saves the result to stats_cache.pkl.  The format is identical to
dist_cache.pkl — if that file already exists its entries are reused
to avoid redundant re-parsing.

NOTE: Feature extraction (windowing + statistics) is NOT performed here
because it depends on window_size, which is a per-config hyperparameter.
stats_cache.pkl stores raw events only; train_single.py extracts features
on-the-fly using each config's window_size and the training-derived reference pool.

Usage (from project root):
  python htm_distance_stats/htm_distance_stats_prepare_data.py

Outputs:
  stats_cache.pkl   — dict: filepath → list of (key_idx, dwell, flight, dist)
"""

import multiprocessing
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_dist_dir = os.path.join(_project_root, "htm_distance")
if _htm_dist_dir not in sys.path:
    sys.path.insert(0, _htm_dist_dir)

from common.keystroke_features import FOLDERS, is_task1  # noqa: F401
from htm_distance_common import parse_file_distance

SPLIT_FILE = "split.pkl"
CACHE_FILE = "stats_cache.pkl"
DIST_CACHE = "dist_cache.pkl"    # reuse if already built by htm_distance/


# Top-level worker — must be picklable for ProcessPoolExecutor
def _worker(filepath: str):
    return filepath, parse_file_distance(filepath)


def main():
    os.chdir(_project_root)

    print("=" * 60)
    print("  HTM-Distance-Stats Data Preparation")
    print("=" * 60)

    # ── 1. Load split ──────────────────────────────────────────────
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: '{SPLIT_FILE}' not found. Run htm/htm_prepare_data.py first.")
        return

    with open(SPLIT_FILE, "rb") as fh:
        split = pickle.load(fh)

    train_human = split["train_human"]
    val_human   = split["val_human"]
    test_human  = split["test_human"]
    val_bots    = split["val_bots"]
    test_bots   = split["test_bots"]
    all_files   = train_human + val_human + test_human + val_bots + test_bots

    print(f"\nSplit loaded from {SPLIT_FILE}")
    print(f"  Train : {len(train_human)} human files")
    print(f"  Val   : {len(val_human)} human  +  {len(val_bots)} bot files")
    print(f"  Test  : {len(test_human)} human  +  {len(test_bots)} bot files")
    print(f"  Total : {len(all_files)} files to parse")

    # ── 2. Load existing cache ─────────────────────────────────────
    # Prefer stats_cache.pkl; fall back to dist_cache.pkl (same format)
    stats_cache: dict = {}

    if os.path.exists(CACHE_FILE):
        print(f"\nLoading existing cache: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as fh:
            stats_cache = pickle.load(fh)
        print(f"  {len(stats_cache)} files already cached.")
    elif os.path.exists(DIST_CACHE):
        print(f"\nNo {CACHE_FILE} found — seeding from {DIST_CACHE} (same format)...")
        with open(DIST_CACHE, "rb") as fh:
            stats_cache = pickle.load(fh)
        print(f"  {len(stats_cache)} files loaded from {DIST_CACHE}.")

    # ── 3. Parse missing files ─────────────────────────────────────
    missing = [f for f in all_files if f not in stats_cache]

    if missing:
        print(f"\nParsing {len(missing)} files (parallel)...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_worker, fp) for fp in missing]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="  Parsing"):
                fp, events = future.result()
                if events:
                    stats_cache[fp] = events

        with open(CACHE_FILE, "wb") as fh:
            pickle.dump(stats_cache, fh)
        print(f"  Cache saved → {CACHE_FILE}  ({len(stats_cache)} total files)")

    else:
        print("\nAll files already cached — nothing to do.")
        if not os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "wb") as fh:
                pickle.dump(stats_cache, fh)
            print(f"  Saved {CACHE_FILE}")

    # ── 4. Quick stats ─────────────────────────────────────────────
    total_events = sum(len(v) for v in stats_cache.values())
    print(f"\nTotal keystroke events cached : {total_events:,}")

    # Show bot file lengths (critical: must be >= window_size for detection)
    bot_files = val_bots + test_bots
    bot_lens  = [len(stats_cache[f]) for f in bot_files if f in stats_cache]
    if bot_lens:
        print(f"\nBot file keystroke counts (after printable-key filtering):")
        print(f"  min={min(bot_lens)}  median={sorted(bot_lens)[len(bot_lens)//2]}"
              f"  max={max(bot_lens)}  (need >= window_size for detection)")
        short3 = sum(1 for l in bot_lens if l < 3)
        short5 = sum(1 for l in bot_lens if l < 5)
        print(f"  < 3 keystrokes: {short3}/{len(bot_lens)} bots")
        print(f"  < 5 keystrokes: {short5}/{len(bot_lens)} bots")

    if stats_cache:
        sample = next(iter(stats_cache.values()))[:3]
        print("\nSample events (key_idx, dwell_ms, flight_ms, dist_units):")
        for e in sample:
            print(f"  {e}")

    print("\nDone. Next steps:")
    print("  python htm_distance_stats/htm_distance_stats_generate_configs.py --n-configs 128")
    print("  sbatch slurm/ds_submit_array.sh")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
