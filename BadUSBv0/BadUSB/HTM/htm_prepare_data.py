"""
BadUSBv0/BadUSB/HTM/htm_prepare_data.py
Pre-compute per-window feature sequences for the HTM-Combined model.

This script reads the person-disjoint splits from data_split.json,
parses the raw keystroke files, and generates feature windows for each file.
The output is a single cache file that can be used by the HTM training script.

Usage (from BadUSBv0/BadUSB/):
  python HTM/htm_prepare_data.py

Outputs:
  HTM/windows_cache.pkl
"""

import os
import sys
import json
import pickle
import multiprocessing as mp
from tqdm import tqdm

# --- Path setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_badusb_root = os.path.dirname(_here)
_project_root = os.path.dirname(os.path.dirname(_badusb_root))

for p in [_project_root, os.path.join(_project_root, "common"), os.path.join(_project_root, "htm_distance")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from htm_combined_common import create_reference_pool, get_file_combined_seq, parse_file_distance
from keystroke_features import RANDOM_SEED

# --- Configuration ---
SPLIT_JSON = os.path.join(_badusb_root, "data_split.json")
OUTPUT_CACHE = os.path.join(_here, "windows_cache.pkl")
WINDOW_SIZE = 10
WINDOW_STEP = 1

# --- Multiprocessing worker ---
_w_cache = None
_w_ref_dwells = None
_w_ref_flights = None
_w_ref_dists = None
_w_window_size = None
_w_window_step = None

def _init_worker(cache, ref_dwells, ref_flights, ref_dists, window_size, window_step):
    global _w_cache, _w_ref_dwells, _w_ref_flights, _w_ref_dists, _w_window_size, _w_window_step
    _w_cache = cache
    _w_ref_dwells = ref_dwells
    _w_ref_flights = ref_flights
    _w_ref_dists = ref_dists
    _w_window_size = window_size
    _w_window_step = window_step

def _process_file(fp):
    events = _w_cache.get(fp)
    if not events:
        return fp, None, True
    seq = get_file_combined_seq(events, _w_window_size, _w_window_step, _w_ref_dwells, _w_ref_flights, _w_ref_dists)
    return fp, seq if seq else None, False

def main():
    if not os.path.exists(SPLIT_JSON):
        print(f"ERROR: {SPLIT_JSON} not found. Run split_persons.py first.")
        sys.exit(1)

    with open(SPLIT_JSON) as f:
        split = json.load(f)

    print("Parsing all files to create a temporary event cache...")
    event_cache = {}
    all_files = []
    for s in ["train", "val", "test"]:
        all_files.extend(split[s]["humans"])
        all_files.extend(split[s]["bots"])
    
    unique_files = sorted(list(set(all_files)))

    for fp in tqdm(unique_files, desc="Parsing files"):
        events = parse_file_distance(fp)
        if events:
            event_cache[fp] = events

    print(f"Building reference pool from {len(split['train']['humans'])} training human files...")
    ref_dwells, ref_flights, ref_dists = create_reference_pool(
        split['train']['humans'], event_cache, window_size=WINDOW_SIZE, seed=RANDOM_SEED
    )

    if not ref_dwells:
        print(f"ERROR: Could not build reference pool. window_size={WINDOW_SIZE} may be too large.")
        sys.exit(1)

    sequences = {}
    skipped = 0
    n_workers = mp.cpu_count()
    print(f"Generating feature sequences with {n_workers} workers...")
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(event_cache, ref_dwells, ref_flights, ref_dists, WINDOW_SIZE, WINDOW_STEP),
    ) as pool:
        for fp, seq, missing in tqdm(
            pool.imap_unordered(_process_file, unique_files),
            total=len(unique_files),
            desc=f"ws={WINDOW_SIZE}s={WINDOW_STEP}",
        ):
            if missing:
                skipped += 1
            elif seq:
                sequences[fp] = seq

    total_windows = sum(len(s) for s in sequences.values())
    print(f"Generated sequences for {len(sequences)}/{len(unique_files)} files.")
    print(f"Total windows: {total_windows}")

    payload = {
        'window_size': WINDOW_SIZE,
        'window_step': WINDOW_STEP,
        'ref_dwells': ref_dwells,
        'ref_flights': ref_flights,
        'ref_dists': ref_dists,
        'sequences': sequences,
    }

    with open(OUTPUT_CACHE, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Successfully created windows cache at: {OUTPUT_CACHE}")

if __name__ == "__main__":
    main()
