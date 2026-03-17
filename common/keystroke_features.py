"""
common/keystroke_features.py
Shared keystroke-dynamics parsing and feature extraction utilities.
Used by both the HTM and MLP pipelines.
"""

import os
import random
import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
RANDOM_SEED         = 42
NUM_REFERENCES      = 50
DEFAULT_WINDOW_SIZE = 15
DEFAULT_STEP_HUMAN  = 1
DEFAULT_STEP_BOT    = 1

FOLDERS = {
    "Humans": [
        "../UB_keystroke_dataset/s0/rotation/",
        "../UB_keystroke_dataset/s1/rotation/",
        "../UB_keystroke_dataset/s2/rotation/",
    ],
    "Bots": ["../BadUSBdataset", "../only_timings_dataset", "Synthetic_Bots"],
}


# ------------------------------------------------------------------
# File helpers
# ------------------------------------------------------------------
def is_task1(filepath):
    """Returns True if the file belongs to Task 1 (6th char of stem == '1')."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return len(basename) == 6 and basename[5] == '1'


def get_person_id(filepath):
    """Extracts the 3-digit person ID (first 3 characters of the filename stem)."""
    return os.path.splitext(os.path.basename(filepath))[0][:3]


# ------------------------------------------------------------------
# Keystroke parsing
# ------------------------------------------------------------------
def parse_file(filepath):
    """
    Parse a keystroke log .txt file.
    Returns (dwells, flights) as numpy arrays of timing values in ms.
    """
    dwells, flights = [], []
    active_keys = {}
    last_keyup  = None

    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
    except Exception:
        return np.array([]), np.array([])

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key, action = parts[0], parts[1]
        try:
            ts = int(parts[2])
        except ValueError:
            continue
        scale = 10000.0 if ts > 10 ** 15 else 1.0

        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup is not None:
                delta = (ts - last_keyup) / scale
                if 0 < delta < 5000:
                    flights.append(delta)
        elif action == "KeyUp":
            last_keyup = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta   = (ts - down_ts) / scale
                if 0 < delta < 3000:
                    dwells.append(delta)

    return np.array(dwells), np.array(flights)


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def extract_features(window_data, reference_pool,
                     window_size=DEFAULT_WINDOW_SIZE):
    """
    Compute a 7-dimensional feature vector for one timing window:
    [mean, median, std, skew, kurtosis, min_KS_distance, min_Wasserstein_distance]
    """
    if len(window_data) < window_size or np.isnan(window_data).any():
        return None

    feat_mean = np.mean(window_data)
    feat_med  = np.median(window_data)
    feat_std  = np.std(window_data)

    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0.0, 10.0
    else:
        feat_skew = stats.skew(window_data)
        feat_kurt = stats.kurtosis(window_data)

    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))

    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt,
            np.min(ks_scores), np.min(w_scores)]


def create_reference_pool(train_human_files,
                          window_size=DEFAULT_WINDOW_SIZE,
                          num_references=NUM_REFERENCES,
                          seed=RANDOM_SEED):
    """
    Build KS/Wasserstein reference windows from training-set human files only
    (no leakage from validation/test data).
    Returns (ref_dwells, ref_flights) — lists of numpy arrays.
    """
    rng = random.Random(seed)
    print("--- Building Reference Pool ---")
    all_dwells, all_flights = [], []
    sample_files = list(train_human_files)
    rng.shuffle(sample_files)

    for filepath in sample_files[:50]:
        d, fl = parse_file(filepath)
        if len(d)  >= window_size: all_dwells.extend(d)
        if len(fl) >= window_size: all_flights.extend(fl)

    ref_dwells, ref_flights = [], []
    if len(all_dwells) > window_size:
        for _ in range(num_references):
            start = rng.randint(0, len(all_dwells) - window_size)
            ref_dwells.append(np.array(all_dwells[start:start + window_size]))
    if len(all_flights) > window_size:
        for _ in range(num_references):
            start = rng.randint(0, len(all_flights) - window_size)
            ref_flights.append(np.array(all_flights[start:start + window_size]))

    return ref_dwells, ref_flights


# ------------------------------------------------------------------
# Multiprocessing worker
# Top-level so it is picklable by ProcessPoolExecutor.
# args = (filepath, step_size, ref_dwells, ref_flights, window_size)
# ------------------------------------------------------------------
def process_file_worker(args):
    """
    Worker function for parallel feature extraction.
    Returns (filepath, feature_sequence) where feature_sequence is a
    numpy array of shape [num_windows, 14].
    """
    filepath, step_size, ref_dwells, ref_flights, window_size = args
    d, fl = parse_file(filepath)
    min_len = min(len(d), len(fl))
    features_seq = []
    if min_len < window_size:
        return filepath, np.array([])

    for i in range(0, min_len - window_size, step_size):
        w_d  = d[i:i + window_size]
        w_f  = fl[i:i + window_size]
        ft_d = extract_features(w_d, ref_dwells, window_size)
        ft_f = extract_features(w_f, ref_flights, window_size)
        if ft_d and ft_f:
            features_seq.append(ft_f + ft_d)

    return filepath, np.array(features_seq)
