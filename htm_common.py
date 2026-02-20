"""
htm_common.py
Shared data-processing and encoding utilities used by all HTM scripts.
HTM_training.py remains standalone and does NOT import from here.
"""

import os
import random
import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
RANDOM_SEED          = 42
NUM_REFERENCES       = 50
DEFAULT_WINDOW_SIZE  = 15
DEFAULT_STEP_HUMAN   = 1
DEFAULT_STEP_BOT     = 1

FOLDERS = {
    "Humans": [
        "../UB_keystroke_dataset/s0/rotation/",
        "../UB_keystroke_dataset/s1/rotation/",
        "../UB_keystroke_dataset/s2/rotation/",
    ],
    "Bots": ["../BadUSBdataset", "../only_timings_dataset"],
}


# ------------------------------------------------------------------
# File helpers
# ------------------------------------------------------------------
def is_task1(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return len(basename) == 6 and basename[5] == '1'


def get_person_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0][:3]


# ------------------------------------------------------------------
# Keystroke parsing
# ------------------------------------------------------------------
def parse_file(filepath):
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


# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------
class SimpleScalarEncoder:
    def __init__(self, min_val, max_val, n_bits, w):
        self.min_val = min_val
        self.max_val = max_val
        self.n_bits  = n_bits
        self.w       = w
        self.range   = max(float(max_val - min_val), 1e-9)

    def encode_into_array(self, value, dense_array, offset):
        value = float(np.clip(value, self.min_val, self.max_val))
        pos   = (value - self.min_val) / self.range
        idx   = int(pos * (self.n_bits - self.w))
        idx   = max(0, min(self.n_bits - self.w, idx))
        for i in range(self.w):
            dense_array[offset + idx + i] = 1


class MultiAttributeEncoder:
    def __init__(self, num_features, min_vals, max_vals,
                 bits_per_feature=32, w=5):
        self.encoders         = []
        self.total_bits       = num_features * bits_per_feature
        self.bits_per_feature = bits_per_feature
        for i in range(num_features):
            self.encoders.append(
                SimpleScalarEncoder(min_vals[i], max_vals[i],
                                    bits_per_feature, w)
            )

    def encode(self, features):
        dense = np.zeros(self.total_bits, dtype=np.uint8)
        for i, val in enumerate(features):
            self.encoders[i].encode_into_array(
                val, dense, i * self.bits_per_feature)
        return dense


# ------------------------------------------------------------------
# Data splitting  (deterministic given seed)
# ------------------------------------------------------------------
def build_split(human_files, bot_files,
                n_val_persons=15, seed=RANDOM_SEED):
    """Return a dict with train/val/test file lists."""
    rng = random.Random(seed)
    person_ids = sorted(set(get_person_id(f) for f in human_files))
    rng.shuffle(person_ids)
    n       = len(person_ids)
    n_val   = n_val_persons
    n_train = n - n_val - (n - n_val) // 5   # ~80 % train, rest test

    train_persons = set(person_ids[:n_train])
    val_persons   = set(person_ids[n_train:n_train + n_val])
    test_persons  = set(person_ids[n_train + n_val:])

    train_human = [f for f in human_files if get_person_id(f) in train_persons]
    val_human   = [f for f in human_files if get_person_id(f) in val_persons]
    test_human  = [f for f in human_files if get_person_id(f) in test_persons]

    shuffled_bots = list(bot_files)
    rng.shuffle(shuffled_bots)
    nb_val    = max(1, len(shuffled_bots) // 5)
    val_bots  = shuffled_bots[:nb_val]
    test_bots = shuffled_bots[nb_val:]

    return {
        'train_human': train_human,
        'val_human':   val_human,
        'test_human':  test_human,
        'val_bots':    val_bots,
        'test_bots':   test_bots,
    }
