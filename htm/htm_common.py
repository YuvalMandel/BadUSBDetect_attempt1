"""
htm/htm_common.py
HTM-specific shared utilities: SDR encoders and data splitting.
Shared keystroke-parsing utilities live in common/keystroke_features.py.
"""

import os
import sys

# Allow imports from the project root (e.g. common/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import random
import numpy as np

from common.keystroke_features import RANDOM_SEED, get_person_id


# ------------------------------------------------------------------
# Encoders
# ------------------------------------------------------------------
class SimpleScalarEncoder:
    """Encodes a single scalar value into a contiguous block of active bits."""
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
    """Concatenates independent scalar encodings for each feature dimension."""
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
    """
    Person-disjoint train/val/test split for the HTM pipeline.
    Returns a dict with keys: train_human, val_human, test_human,
    val_bots, test_bots.
    """
    rng = random.Random(seed)
    person_ids = sorted(set(get_person_id(f) for f in human_files))
    rng.shuffle(person_ids)
    n       = len(person_ids)
    n_val   = n_val_persons
    n_train = n - n_val - (n - n_val) // 5   # ~80 % train, remainder test

    train_persons = set(person_ids[:n_train])
    val_persons   = set(person_ids[n_train:n_train + n_val])

    train_human = [f for f in human_files if get_person_id(f) in train_persons]
    val_human   = [f for f in human_files if get_person_id(f) in val_persons]
    test_human  = [f for f in human_files
                   if get_person_id(f) not in train_persons
                   and get_person_id(f) not in val_persons]

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
