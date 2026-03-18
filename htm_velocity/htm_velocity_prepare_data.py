"""
htm_velocity/htm_velocity_prepare_data.py
Build velocity_cache.pkl from stats_cache.pkl + poly_regressor.pkl.

This is a ONE-TIME preparation step.  The velocity cache extends the
stats_cache by adding a per-event polynomial Fitts'-Law prediction error:
  poly_err = |actual_flight_ms - poly_model.predict(x1, y1, x2, y2)|

velocity_cache format:
  {filepath: [(key_idx, dwell_ms, flight_ms, dist_units, poly_err), ...]}

Prerequisites (run these first):
  python htm_distance_stats/htm_distance_stats_prepare_data.py  → stats_cache.pkl
  python mlp_vlad_velocity/regressor/regressor_train.py          → poly_regressor.pkl

Usage (from project root):
  python htm_velocity/htm_velocity_prepare_data.py
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_vel_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_vel_dir not in sys.path:
    sys.path.insert(0, _htm_vel_dir)

os.chdir(_project_root)

import pickle
import joblib

from htm_velocity_common import build_velocity_cache_from_stats

STATS_CACHE_FILE = "stats_cache.pkl"
DIST_CACHE_FILE  = "dist_cache.pkl"    # fallback, same format
POLY_MODEL_PATH  = "poly_regressor.pkl"
OUTPUT_PATH      = "velocity_cache.pkl"


def main():
    # ── Load poly model ──────────────────────────────────────────────────────
    if not os.path.exists(POLY_MODEL_PATH):
        print(f"ERROR: {POLY_MODEL_PATH} not found.")
        print("  Run: python mlp_vlad_velocity/regressor/regressor_train.py")
        sys.exit(1)

    print(f"Loading poly model from {POLY_MODEL_PATH}...")
    poly_model = joblib.load(POLY_MODEL_PATH)
    print("  Done.")

    # ── Load stats cache ─────────────────────────────────────────────────────
    cache_path = STATS_CACHE_FILE if os.path.exists(STATS_CACHE_FILE) else DIST_CACHE_FILE
    if not os.path.exists(cache_path):
        print(f"ERROR: neither {STATS_CACHE_FILE} nor {DIST_CACHE_FILE} found.")
        print("  Run: python htm_distance_stats/htm_distance_stats_prepare_data.py")
        sys.exit(1)
    if cache_path == DIST_CACHE_FILE:
        print(f"  Note: {STATS_CACHE_FILE} not found; using {DIST_CACHE_FILE}")

    print(f"Loading keystroke cache from {cache_path}...")
    with open(cache_path, 'rb') as fh:
        stats_cache = pickle.load(fh)
    print(f"  {len(stats_cache)} files in cache.")

    # ── Build velocity cache ─────────────────────────────────────────────────
    print("\nBuilding velocity cache (computing poly errors per event)...")
    print("  This may take a few minutes for large datasets.")
    vel_cache = build_velocity_cache_from_stats(stats_cache, poly_model,
                                                show_progress=True)

    # ── Stats ────────────────────────────────────────────────────────────────
    n_total = sum(len(v) for v in vel_cache.values())
    n_valid = sum(
        sum(1 for e in v if e[4] == e[4])   # non-NaN
        for v in vel_cache.values()
    )
    print(f"\n  Total events           : {n_total:,}")
    print(f"  Events with poly error : {n_valid:,}  "
          f"({100.0 * n_valid / max(n_total, 1):.1f}%)")
    print(f"  Events without (NaN)   : {n_total - n_valid:,}  "
          f"(first events + unmapped keys)")

    # Sample a few error values
    import numpy as np
    sample_errs = [
        e[4]
        for v in list(vel_cache.values())[:20]
        for e in v
        if e[4] == e[4]
    ]
    if sample_errs:
        arr = np.array(sample_errs)
        print(f"\n  Sample error stats (first 20 files):")
        print(f"    mean={arr.mean():.1f} ms  median={np.median(arr):.1f} ms  "
              f"std={arr.std():.1f} ms  max={arr.max():.1f} ms")

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving velocity cache to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as fh:
        pickle.dump(vel_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
