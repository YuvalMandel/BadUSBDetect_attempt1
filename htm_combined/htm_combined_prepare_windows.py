"""
htm_combined/htm_combined_prepare_windows.py
Pre-compute per-window feature sequences for all (window_size, window_step)
combinations used by the HTM-Combined hyperparameter search.

This is a ONE-TIME preparation step that dramatically reduces per-job wall time.
Instead of recomputing KS/Wasserstein statistics inside every SLURM job
(128 configs × ~500 files each), statistics are computed once per
(window_size, window_step) pair and shared by all configs with those settings.

The 8 pairs in the Round-3 search space:
  window_size  ∈ {5, 10, 15, 20}
  window_step  ∈ {1, 2}

A fixed reference pool is built with RANDOM_SEED (not per-config seed).
Per-config seeds only affect SP/TM initialisation, not the feature statistics,
so the results remain comparable across configs.

Usage (from project root):
  python htm_combined/htm_combined_prepare_windows.py

Outputs:
  windows_cache/ws{N}s{S}.pkl  for each (window_size, window_step) pair

Cache file format (dict):
  {
    'window_size':  int,
    'window_step':  int,
    'ref_dwells':   list[np.ndarray],   # KS/Wasserstein reference windows
    'ref_flights':  list[np.ndarray],
    'ref_dists':    list[np.ndarray],
    'sequences': {
        filepath: [(stats_21, key_idx, dwell_ms, flight_ms, dist_units), ...],
        ...   (all train / val / test / bot files)
    }
  }
"""

import os
import sys
import pickle

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_combined_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_combined_dir not in sys.path:
    sys.path.insert(0, _htm_combined_dir)

os.chdir(_project_root)

from tqdm import tqdm
from common.keystroke_features import RANDOM_SEED
from htm_combined_common import create_reference_pool, get_file_combined_seq

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLIT_FILE  = "split.pkl"
CACHE_FILE  = "stats_cache.pkl"
DIST_CACHE  = "dist_cache.pkl"      # fallback — same format
WINDOWS_DIR = "windows_cache"

# All (window_size, window_step) pairs in the Round-3 search space
WINDOW_PAIRS = [
    (ws, st)
    for ws in [5, 10, 15, 20]
    for st in [1, 2]
]


def main():
    os.makedirs(WINDOWS_DIR, exist_ok=True)

    # ── Load split + keystroke cache ──────────────────────────────────────────
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: {SPLIT_FILE} not found. "
              "Run htm/htm_prepare_data.py first.")
        sys.exit(1)

    cache_path = CACHE_FILE if os.path.exists(CACHE_FILE) else DIST_CACHE
    if not os.path.exists(cache_path):
        print(f"ERROR: neither {CACHE_FILE} nor {DIST_CACHE} found. "
              "Run htm_distance_stats/htm_distance_stats_prepare_data.py first.")
        sys.exit(1)
    if cache_path == DIST_CACHE:
        print(f"  Note: {CACHE_FILE} not found; using {DIST_CACHE} (same format).")

    print(f"Loading split from {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'rb') as fh:
        split = pickle.load(fh)

    print(f"Loading keystroke cache from {cache_path}...")
    with open(cache_path, 'rb') as fh:
        cache = pickle.load(fh)

    train_human = split['train_human']
    all_files = (
        split['train_human'] + split['val_human'] + split['test_human'] +
        split['val_bots']   + split['test_bots']
    )
    # Deduplicate, preserving order
    seen: set = set()
    unique_files = []
    for fp in all_files:
        if fp not in seen:
            seen.add(fp)
            unique_files.append(fp)

    print(f"  {len(unique_files)} total files  "
          f"({len(split['train_human'])} train-h / "
          f"{len(split['val_human'])} val-h / "
          f"{len(split['test_human'])} test-h / "
          f"{len(split['val_bots'])} val-b / "
          f"{len(split['test_bots'])} test-b)")

    # ── Build cache for each (window_size, window_step) pair ──────────────────
    for window_size, window_step in WINDOW_PAIRS:
        out_path = os.path.join(WINDOWS_DIR,
                                f"ws{window_size}s{window_step}.pkl")

        if os.path.exists(out_path):
            print(f"\n[ws={window_size} step={window_step}]  "
                  f"Already exists: {out_path}  (skipping)")
            continue

        print(f"\n{'='*60}")
        print(f"[ws={window_size} step={window_step}]  Building cache...")

        # Fixed seed — all configs sharing this (window_size, window_step)
        # will use the same reference pool, keeping results comparable
        ref_dwells, ref_flights, ref_dists = create_reference_pool(
            train_human, cache,
            window_size=window_size,
            seed=RANDOM_SEED)

        if not ref_dwells:
            print(f"  WARNING: Could not build reference pool "
                  f"(window_size={window_size} may exceed training files). Skipping.")
            continue

        sequences: dict = {}
        skipped = 0
        for fp in tqdm(unique_files, desc=f"  ws={window_size}s={window_step}"):
            events = cache.get(fp)
            if not events:
                skipped += 1
                continue
            seq = get_file_combined_seq(events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists)
            if seq:
                sequences[fp] = seq

        total_windows = sum(len(s) for s in sequences.values())
        print(f"  Sequences: {len(sequences)}/{len(unique_files)} files "
              f"({skipped} missing from cache)")
        print(f"  Total windows: {total_windows:,}")

        payload = {
            'window_size':  window_size,
            'window_step':  window_step,
            'ref_dwells':   ref_dwells,
            'ref_flights':  ref_flights,
            'ref_dists':    ref_dists,
            'sequences':    sequences,
        }
        with open(out_path, 'wb') as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved -> {out_path}")

    print(f"\nDone. All window caches written to {WINDOWS_DIR}/")
    print("Next step: generate configs and submit SLURM jobs.")


if __name__ == "__main__":
    main()
