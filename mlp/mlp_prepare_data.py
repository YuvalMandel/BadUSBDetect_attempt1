"""
mlp/mlp_prepare_data.py
Data preparation for the MLP supervised classifier.
Parses raw keystroke files, extracts 14-dim feature windows,
generates synthetic bot samples, and saves train/val/test CSV files.

Run before mlp_train.py:
  python mlp/mlp_prepare_data.py

Outputs (relative to project root):
  train_dataset.csv  — training windows (humans + real bots + synthetic bots)
  val_dataset.csv    — validation windows
  test_dataset.csv   — test windows
  reference_pool.npz — KS/Wasserstein reference windows (from train humans)
"""

import os
import sys

# Allow imports from project root (common/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import glob
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.keystroke_features import (
    FOLDERS, RANDOM_SEED, DEFAULT_WINDOW_SIZE, NUM_REFERENCES,
    is_task1, get_person_id, parse_file, extract_features, create_reference_pool,
)

# ------------------------------------------------------------------
# MLP-specific constants
# ------------------------------------------------------------------
WINDOW_SIZE       = DEFAULT_WINDOW_SIZE
STEP_SIZE_HUMAN   = 10   # coarser stride for humans (reduces dataset size)
STEP_SIZE_BOT     = 1
SYNTHETIC_SAMPLES = 4000  # for the train split; val/test get proportional counts

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

OUTPUT_TRAIN = "train_dataset.csv"
OUTPUT_VAL   = "val_dataset.csv"
OUTPUT_TEST  = "test_dataset.csv"
OUTPUT_REFS  = "reference_pool.npz"


# ==============================================================================
# 1. Person-based splitting (MLP uses ratio-based, not person-count-based)
# ==============================================================================

def split_by_person(all_files, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                    seed=RANDOM_SEED):
    """
    Splits files into train/val/test with no person appearing in more than one
    split (prevents data leakage through typing style).
    Returns: (train_files, train_persons), (val_files, val_persons), (test_files, test_persons)
    """
    person_ids = sorted(set(get_person_id(f) for f in all_files))
    rng = random.Random(seed)
    rng.shuffle(person_ids)

    n       = len(person_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_persons = set(person_ids[:n_train])
    val_persons   = set(person_ids[n_train: n_train + n_val])
    test_persons  = set(person_ids[n_train + n_val:])

    train_files = [f for f in all_files if get_person_id(f) in train_persons]
    val_files   = [f for f in all_files if get_person_id(f) in val_persons]
    test_files  = [f for f in all_files if get_person_id(f) in test_persons]

    return (train_files, train_persons), (val_files, val_persons), (test_files, test_persons)


# ==============================================================================
# 2. MLP worker: produces labeled CSV rows (label appended)
# ==============================================================================

def process_single_file(filepath, label, step_size, ref_dwells, ref_flights):
    """
    Parse a keystroke file and produce feature rows with their label.
    Used by ProcessPoolExecutor for parallel feature extraction.
    """
    rows = []
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))

    if min_len < WINDOW_SIZE:
        return []

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i: i + WINDOW_SIZE]
        w_f = f[i: i + WINDOW_SIZE]
        ft_d = extract_features(w_d, ref_dwells, WINDOW_SIZE)
        ft_f = extract_features(w_f, ref_flights, WINDOW_SIZE)
        if ft_d and ft_f:
            rows.append(ft_f + ft_d + [label])

    return rows


# ==============================================================================
# 3. Synthetic bot generator
# ==============================================================================

def generate_synthetic_bots(n_samples):
    """
    Generates n_samples synthetic BadUSB attack windows covering 13 attack types:
    machine_gun, gaussian, uniform, jitter, constant, exponential, log_normal,
    bimodal, very_fast, burst_pause, linear_ramp, pareto, sinusoidal.
    """
    synthetic_rows = []
    attack_types = [
        'machine_gun', 'gaussian', 'uniform', 'jitter',
        'constant', 'exponential', 'log_normal', 'bimodal',
        'very_fast', 'burst_pause', 'linear_ramp', 'pareto', 'sinusoidal',
    ]

    for _ in range(n_samples):
        attack_type = random.choice(attack_types)
        win_dwell = win_flight = None

        if attack_type == 'machine_gun':
            val = random.randint(4, 10)
            win_dwell  = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)
            win_flight = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)

        elif attack_type == 'gaussian':
            mean = random.randint(80, 150)
            std  = random.randint(10, 40)
            win_dwell  = np.random.normal(mean, std, WINDOW_SIZE)
            win_flight = np.random.normal(mean, std, WINDOW_SIZE)

        elif attack_type == 'uniform':
            low  = random.randint(10, 50)
            high = random.randint(60, 200)
            win_dwell  = np.random.uniform(low, high, WINDOW_SIZE)
            win_flight = np.random.uniform(low, high, WINDOW_SIZE)

        elif attack_type == 'jitter':
            win_dwell  = np.random.randint(5, 300, WINDOW_SIZE).astype(float)
            win_flight = np.random.randint(5, 300, WINDOW_SIZE).astype(float)

        elif attack_type == 'constant':
            val = random.uniform(20.0, 500.0)
            win_dwell  = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.05, WINDOW_SIZE)
            win_flight = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.05, WINDOW_SIZE)

        elif attack_type == 'exponential':
            scale = random.uniform(20.0, 200.0)
            win_dwell  = np.random.exponential(scale, WINDOW_SIZE)
            win_flight = np.random.exponential(scale, WINDOW_SIZE)

        elif attack_type == 'log_normal':
            mu    = random.uniform(3.0, 5.5)
            sigma = random.uniform(0.1, 0.6)
            win_dwell  = np.random.lognormal(mu, sigma, WINDOW_SIZE)
            win_flight = np.random.lognormal(mu, sigma, WINDOW_SIZE)

        elif attack_type == 'bimodal':
            mu1, std1 = random.randint(5, 30),   random.randint(1, 5)
            mu2, std2 = random.randint(100, 300), random.randint(10, 50)
            n1 = WINDOW_SIZE // 2
            combined = np.concatenate([
                np.random.normal(mu1, std1, n1),
                np.random.normal(mu2, std2, WINDOW_SIZE - n1),
            ])
            np.random.shuffle(combined)
            win_dwell  = combined.copy()
            win_flight = combined + np.random.normal(0, 2.0, WINDOW_SIZE)

        elif attack_type == 'very_fast':
            win_dwell  = np.random.uniform(0.3, 5.0, WINDOW_SIZE)
            win_flight = np.random.uniform(0.3, 5.0, WINDOW_SIZE)

        elif attack_type == 'burst_pause':
            burst_len = random.randint(3, 8)
            burst_val = random.uniform(5.0, 20.0)
            pause_val = random.uniform(200.0, 800.0)
            pattern = []
            i = 0
            while len(pattern) < WINDOW_SIZE:
                if i % (burst_len + 1) == burst_len:
                    pattern.append(pause_val + np.random.normal(0, 10))
                else:
                    pattern.append(burst_val + np.random.normal(0, 1))
                i += 1
            arr = np.array(pattern[:WINDOW_SIZE])
            win_dwell  = arr.copy()
            win_flight = arr + np.random.normal(0, 2.0, WINDOW_SIZE)

        elif attack_type == 'linear_ramp':
            start = random.uniform(5.0,  50.0)
            end   = random.uniform(100.0, 500.0)
            if random.random() < 0.5:
                start, end = end, start
            noise = random.uniform(1.0, 10.0)
            base = np.linspace(start, end, WINDOW_SIZE)
            win_dwell  = base + np.random.normal(0, noise, WINDOW_SIZE)
            win_flight = base + np.random.normal(0, noise, WINDOW_SIZE)

        elif attack_type == 'pareto':
            alpha = random.uniform(1.5, 3.0)
            scale = random.uniform(10.0, 50.0)
            win_dwell  = (np.random.pareto(alpha, WINDOW_SIZE) + 1.0) * scale
            win_flight = (np.random.pareto(alpha, WINDOW_SIZE) + 1.0) * scale

        elif attack_type == 'sinusoidal':
            base_val  = random.uniform(50.0, 200.0)
            amplitude = random.uniform(10.0, 80.0)
            period    = random.uniform(4.0, 14.0)
            t = np.arange(WINDOW_SIZE, dtype=float)
            wave = base_val + amplitude * np.sin(2.0 * np.pi * t / period)
            win_dwell  = wave + np.random.normal(0, 2.0, WINDOW_SIZE)
            win_flight = wave + np.random.normal(0, 2.0, WINDOW_SIZE)

        if win_dwell is not None:
            synthetic_rows.append((np.abs(win_dwell), np.abs(win_flight)))

    return synthetic_rows


# ==============================================================================
# 4. Process one split (parallel file processing + synthetic bots)
# ==============================================================================

def process_split(human_files, bot_files, ref_dwells, ref_flights,
                  n_synthetic, split_name):
    dataset_rows = []

    tasks = (
        [(f, 0, STEP_SIZE_HUMAN, ref_dwells, ref_flights) for f in human_files]
        + [(f, 1, STEP_SIZE_BOT,   ref_dwells, ref_flights) for f in bot_files]
    )

    if tasks:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, *t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks),
                               desc=f"Processing {split_name}"):
                try:
                    result_rows = future.result()
                    if result_rows:
                        dataset_rows.extend(result_rows)
                except Exception as e:
                    print(f"  Error in worker: {e}")

    if n_synthetic > 0:
        print(f"  Generating {n_synthetic} synthetic bots for {split_name}...")
        synth_data = generate_synthetic_bots(n_synthetic)
        for w_d, w_f in tqdm(synth_data, desc=f"  Synthetic ({split_name})"):
            ft_d = extract_features(w_d, ref_dwells, WINDOW_SIZE)
            ft_f = extract_features(w_f, ref_flights, WINDOW_SIZE)
            if ft_d and ft_f:
                dataset_rows.append(ft_f + ft_d + [1])

    return dataset_rows


# ==============================================================================
# 5. Main
# ==============================================================================

def main():
    # ---- 1. Collect all Task 1 human files ----
    all_human_files = []
    for folder in FOLDERS["Humans"]:
        files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        all_human_files.extend(f for f in files if is_task1(f))

    print(f"Task 1 human files found: {len(all_human_files)}")
    if not all_human_files:
        print("ERROR: No Task 1 human files found. Check FOLDERS paths in common/keystroke_features.py.")
        return

    # ---- 2. Person-based split ----
    (train_human, train_persons), \
    (val_human,   val_persons),   \
    (test_human,  test_persons)   = split_by_person(all_human_files)

    print(f"\nPerson split — Train: {len(train_persons)}, "
          f"Val: {len(val_persons)}, Test: {len(test_persons)}")
    print(f"File split   — Train: {len(train_human)}, "
          f"Val: {len(val_human)}, Test: {len(test_human)}")

    # ---- 3. Reference pool from TRAIN humans only ----
    ref_dwells, ref_flights = create_reference_pool(train_human)
    if not ref_dwells:
        return
    np.savez(OUTPUT_REFS, dwell=ref_dwells, flight=ref_flights)
    print("Reference pool saved.")

    # ---- 4. Collect and split real bot files ----
    all_bot_files = []
    for folder in FOLDERS["Bots"]:
        all_bot_files.extend(
            glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_bot_files)
    nb       = len(all_bot_files)
    nb_train = int(nb * TRAIN_RATIO)
    nb_val   = int(nb * VAL_RATIO)
    train_bots = all_bot_files[:nb_train]
    val_bots   = all_bot_files[nb_train: nb_train + nb_val]
    test_bots  = all_bot_files[nb_train + nb_val:]
    print(f"\nReal bot files — Train: {len(train_bots)}, "
          f"Val: {len(val_bots)}, Test: {len(test_bots)}")

    # ---- 5. Synthetic bot counts proportional to split sizes ----
    n_synth_train = SYNTHETIC_SAMPLES
    n_synth_val   = int(SYNTHETIC_SAMPLES * VAL_RATIO / TRAIN_RATIO)
    n_synth_test  = int(SYNTHETIC_SAMPLES * (1 - TRAIN_RATIO - VAL_RATIO) / TRAIN_RATIO)

    # ---- 6. Process each split and save ----
    cols = ["F_Mean", "F_Med", "F_Std", "F_Skew", "F_Kurt", "F_MinKS", "F_MinW",
            "D_Mean", "D_Med", "D_Std", "D_Skew", "D_Kurt", "D_MinKS", "D_MinW", "Label"]

    splits = [
        ("train", train_human, train_bots, n_synth_train, OUTPUT_TRAIN),
        ("val",   val_human,   val_bots,   n_synth_val,   OUTPUT_VAL),
        ("test",  test_human,  test_bots,  n_synth_test,  OUTPUT_TEST),
    ]

    for split_name, human_files, bot_files, n_synth, output_file in splits:
        print(f"\n{'='*60}")
        print(f" Processing {split_name.upper()} split")
        print(f"{'='*60}")

        rows = process_split(human_files, bot_files, ref_dwells, ref_flights,
                             n_synth, split_name)

        df = pd.DataFrame(rows, columns=cols)
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        df.to_csv(output_file, index=False)

        print(f"\nSaved: {output_file}  ({len(df)} rows)")
        print(df["Label"].value_counts().to_string())

    print("\nAll splits saved.")
    print("Next: run mlp/mlp_balance_train.py, then mlp/mlp_train.py")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
