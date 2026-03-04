"""
htm_combined/htm_combined_common.py
Shared utilities for the HTM-Combined variant.

Design: each HTM timestep encodes ONE sliding window of printable keystrokes.

  Block 1 -- Stats (21 dims, data-derived ranges):
    7 statistics x 3 channels (dwell times, flight times, QWERTY distances)
    over ALL keystrokes in the window, identical to htm_distance_stats/.

  Block 2 -- Last-keystroke identity (95 + 3 x scalar_enc_bits bits, fixed ranges):
    key_type  : one-hot over 95 printable ASCII keys (exactly 1 active bit)
    dwell_ms  : 0-400 ms   (scalar block, scalar_enc_bits bits, scalar_enc_w active)
    flight_ms : 0-500 ms
    dist_units: 0-12  QWERTY key-units

Full SDR layout (left -> right):
  [ stats_block | key_one_hot | dwell | flight | dist ]
   21*stats_enc_bits  95 bits  scalar_enc_bits x3

Total bits   = 21*stats_enc_bits + 95 + 3*scalar_enc_bits
Active bits  = 21*stats_enc_w    +  1 + 3*scalar_enc_w

The stats block uses data-derived min/max (fitted on training windows).
The last-keystroke scalars use fixed physical ranges (same as htm_distance/).
Stats and scalar blocks can have independent enc_bits / enc_w settings.
"""

import os
import sys
import random

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_dist_dir = os.path.join(_project_root, "htm_distance")
if _htm_dist_dir not in sys.path:
    sys.path.insert(0, _htm_dist_dir)

from common.keystroke_features import extract_features, RANDOM_SEED, NUM_REFERENCES
from htm_distance_common import (           # noqa: F401
    parse_file_distance, make_anomaly_likelihood,
    SimpleScalarEncoder,
    DWELL_MIN, DWELL_MAX, FLIGHT_MIN, FLIGHT_MAX, DIST_MIN, DIST_MAX,
)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
NUM_STATS    = 21   # 7 stats x 3 channels (dwell, flight, dist)
NUM_KEYS     = 95   # printable ASCII 32-126 -> indices 0-94
WARMUP_STEPS = 0    # windows to skip (AnomalyLikelihood handles the early spike)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Combined SDR Encoder
# ──────────────────────────────────────────────────────────────────────────────
class CombinedEncoder:
    """
    Encodes (21-dim stats vector, key_idx, dwell_ms, flight_ms, dist_units)
    into a sparse binary SDR.

    Layout:
      [ stats_block | key_one_hot | dwell | flight | dist ]
       21*stats_enc_bits  95 bits  scalar_enc_bits x3

    Total bits   = 21*stats_enc_bits + 95 + 3*scalar_enc_bits
    Active bits  = 21*stats_enc_w    +  1 + 3*scalar_enc_w

    min_vals / max_vals (length-21 arrays) define data-derived encoding ranges
    for the stats block.  Last-keystroke scalars use fixed physical ranges.
    Stats and scalar blocks have independent enc_bits / enc_w settings.
    """

    NUM_STATS = NUM_STATS
    KEY_BITS  = NUM_KEYS

    def __init__(self, min_vals, max_vals,
                 stats_enc_bits: int = 16, stats_enc_w: int = 9,
                 scalar_enc_bits: int = 16, scalar_enc_w: int = 7):
        assert len(min_vals) == self.NUM_STATS, \
            f"Expected {self.NUM_STATS} min_vals, got {len(min_vals)}"
        assert len(max_vals) == self.NUM_STATS, \
            f"Expected {self.NUM_STATS} max_vals, got {len(max_vals)}"
        assert stats_enc_w < stats_enc_bits, \
            f"stats_enc_w ({stats_enc_w}) must be < stats_enc_bits ({stats_enc_bits})"
        assert scalar_enc_w < scalar_enc_bits, \
            f"scalar_enc_w ({scalar_enc_w}) must be < scalar_enc_bits ({scalar_enc_bits})"

        self.min_vals        = np.array(min_vals, dtype=float)
        self.max_vals        = np.array(max_vals, dtype=float)
        self.stats_enc_bits  = stats_enc_bits
        self.stats_enc_w     = stats_enc_w
        self.scalar_enc_bits = scalar_enc_bits
        self.scalar_enc_w    = scalar_enc_w
        self.total_bits      = (self.NUM_STATS * stats_enc_bits
                                + self.KEY_BITS
                                + 3 * scalar_enc_bits)
        self._ranges         = np.maximum(self.max_vals - self.min_vals, 1e-9)

        # Scalar encoders for the last-keystroke features (fixed physical ranges)
        self._dwell_enc  = SimpleScalarEncoder(DWELL_MIN,  DWELL_MAX,
                                               scalar_enc_bits, scalar_enc_w)
        self._flight_enc = SimpleScalarEncoder(FLIGHT_MIN, FLIGHT_MAX,
                                               scalar_enc_bits, scalar_enc_w)
        self._dist_enc   = SimpleScalarEncoder(DIST_MIN,   DIST_MAX,
                                               scalar_enc_bits, scalar_enc_w)

    def encode(self, stats_features, key_idx: int,
               dwell: float, flight: float, distance: float) -> np.ndarray:
        """Encode a combined feature set into a dense binary uint8 SDR."""
        dense            = np.zeros(self.total_bits, dtype=np.uint8)
        stats_enc_bits   = self.stats_enc_bits
        stats_enc_w      = self.stats_enc_w
        scalar_enc_bits  = self.scalar_enc_bits

        # ── Block 1: stats (21 features, data-derived ranges) ────────────────
        for i in range(self.NUM_STATS):
            val = float(np.clip(stats_features[i], self.min_vals[i], self.max_vals[i]))
            pos = (val - self.min_vals[i]) / self._ranges[i]
            idx = int(pos * (stats_enc_bits - stats_enc_w))
            idx = max(0, min(stats_enc_bits - stats_enc_w, idx))
            offset = i * stats_enc_bits
            for j in range(stats_enc_w):
                dense[offset + idx + j] = 1

        # ── Block 2: key one-hot (95 bits, exactly 1 active) ─────────────────
        key_offset = self.NUM_STATS * stats_enc_bits
        dense[key_offset + max(0, min(self.KEY_BITS - 1, key_idx))] = 1

        # ── Block 3: last-keystroke scalars ───────────────────────────────────
        offset = key_offset + self.KEY_BITS
        self._dwell_enc.encode_into(dwell, dense, offset);    offset += scalar_enc_bits
        self._flight_enc.encode_into(flight, dense, offset);  offset += scalar_enc_bits
        self._dist_enc.encode_into(distance, dense, offset)

        return dense


# ──────────────────────────────────────────────────────────────────────────────
# 2. Reference pool (dwell / flight / distance windows from training humans)
# ──────────────────────────────────────────────────────────────────────────────
def create_reference_pool(train_human_files, cache,
                          window_size: int = 10,
                          num_references: int = NUM_REFERENCES,
                          seed: int = RANDOM_SEED):
    """
    Build KS/Wasserstein reference windows for dwell, flight, and distance
    from training human files.

    Parameters
    ----------
    train_human_files : list[str]
    cache             : dict[str, list[tuple]]
        filepath -> [(key_idx, dwell_ms, flight_ms, dist_units), ...]
    window_size       : int -- must match the config window_size
    num_references    : int -- number of reference windows per channel
    seed              : int

    Returns
    -------
    (ref_dwells, ref_flights, ref_dists) : lists of numpy arrays
    """
    rng = random.Random(seed)
    all_dwells, all_flights, all_dists = [], [], []

    files = [f for f in train_human_files if f in cache]
    rng.shuffle(files)
    for fp in files[:50]:
        for _, dwell, flight, dist in cache[fp]:
            all_dwells.append(float(dwell))
            all_flights.append(float(flight))
            all_dists.append(float(dist))

    def _sample(arr):
        refs = []
        if len(arr) >= window_size:
            for _ in range(num_references):
                s = rng.randint(0, len(arr) - window_size)
                refs.append(np.array(arr[s:s + window_size], dtype=float))
        return refs

    print(f"  Reference pool: {len(all_dwells)} dwell / {len(all_flights)} flight "
          f"/ {len(all_dists)} dist values from {min(50, len(files))} files  "
          f"(window_size={window_size}, num_ref={num_references})")

    return _sample(all_dwells), _sample(all_flights), _sample(all_dists)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_combined_window(events, start: int, window_size: int,
                            ref_dwells, ref_flights, ref_dists):
    """
    Extract a combined feature tuple from events[start : start + window_size].

    Returns a 5-tuple:
      (stats_21, key_idx, dwell_ms, flight_ms, dist_units)

    where stats_21 is a list of 21 floats and the remaining 4 values
    are from the LAST keystroke in the window.

    Returns None if extraction fails (too short, NaN, empty refs).
    """
    if not ref_dwells or not ref_flights or not ref_dists:
        return None
    w = events[start:start + window_size]
    if len(w) < window_size:
        return None

    d_arr = np.array([e[1] for e in w], dtype=float)   # dwell_ms
    f_arr = np.array([e[2] for e in w], dtype=float)   # flight_ms
    q_arr = np.array([e[3] for e in w], dtype=float)   # qwerty dist

    fd = extract_features(d_arr, ref_dwells,  window_size)
    ff = extract_features(f_arr, ref_flights, window_size)
    fq = extract_features(q_arr, ref_dists,   window_size)

    if fd is None or ff is None or fq is None:
        return None

    stats = fd + ff + fq   # 21 floats

    last = w[-1]           # (key_idx, dwell_ms, flight_ms, dist_units)
    return stats, last[0], float(last[1]), float(last[2]), float(last[3])


def get_file_combined_seq(events, window_size: int, window_step: int,
                          ref_dwells, ref_flights, ref_dists) -> list:
    """
    Extract the full per-window feature sequence for one file.

    Returns a list of 5-tuples (stats, key_idx, dwell, flight, dist),
    one per valid sliding window.  Empty list if file is too short.
    """
    n   = len(events)
    seq = []
    for start in range(0, n - window_size + 1, window_step):
        item = extract_combined_window(events, start, window_size,
                                       ref_dwells, ref_flights, ref_dists)
        if item is not None:
            seq.append(item)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# 4. Detection (first-crossing, with short-file handling)
# ──────────────────────────────────────────────────────────────────────────────
def apply_detection(seqs: list, mode: str, thresh: float,
                    warmup: int = WARMUP_STEPS,
                    labels: list = None) -> list:
    """
    Apply file-level detection to per-window anomaly score sequences.

    mode='first_crossing' : Bot if any post-warmup score >= thresh.
    mode='mean'           : Bot if mean(post-warmup scores) >= thresh.

    Files with no post-warmup windows are always treated as misclassified;
    pass `labels` to enable this behaviour.
    """
    preds = []
    for i, seq in enumerate(seqs):
        post = seq[warmup:]
        if not post:
            preds.append(1 - labels[i] if labels is not None else 0)
            continue
        if mode == 'first_crossing':
            preds.append(1 if any(s >= thresh for s in post) else 0)
        else:
            preds.append(1 if float(np.mean(post)) >= thresh else 0)
    return preds
