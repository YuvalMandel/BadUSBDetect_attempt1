"""
htm_combined/htm_combined_common.py
Shared utilities for the HTM-Combined variant.

Design: each HTM timestep encodes ONE sliding window of printable keystrokes.

  Block 1 -- Stats (21 dims, data-derived ranges), split into 3 channels:
    Dwell stats   (7 features): per-channel settings (dwell_stats_enc_bits / dwell_stats_enc_w)
    Flight stats  (7 features): per-channel settings (flight_stats_enc_bits / flight_stats_enc_w)
    Dist stats    (7 features): per-channel settings (dist_stats_enc_bits  / dist_stats_enc_w)

  Block 2 -- Last-keystroke identity (95 + 3 scalar encoders, fixed physical ranges):
    key_type   : one-hot over 95 printable ASCII keys (exactly 1 active bit)
    dwell_ms   : 0-400 ms   (dwell_scalar_enc_bits  bits, dwell_scalar_enc_w  active)
    flight_ms  : 0-500 ms   (flight_scalar_enc_bits bits, flight_scalar_enc_w active)
    dist_units : 0-12 units (dist_scalar_enc_bits   bits, dist_scalar_enc_w   active)

Full SDR layout:
  [ dwell_stats | flight_stats | dist_stats | key_one_hot | dwell_key | flight_key | dist_key ]
    7×d_sb        7×f_sb        7×q_sb       95 bits       d_kb        f_kb         q_kb

Total bits  = 7*(d_sb + f_sb + q_sb) + 95 + d_kb + f_kb + q_kb
Active bits = 7*(d_sw + f_sw + q_sw) +  1 + d_kw + f_kw + q_kw

The 3 stats channels encode fundamentally different physical quantities
(dwell time ms, flight time ms, QWERTY distance units), so independent
encoder resolution is expected to improve model quality.
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
NUM_STATS    = 21   # 7 stats × 3 channels (dwell, flight, dist)
NUM_KEYS     = 95   # printable ASCII 32-126 → indices 0-94
WARMUP_STEPS = 0    # windows to skip (AnomalyLikelihood handles the early spike)

# Stats channel slice indices
_DWELL_SLICE  = slice(0,  7)
_FLIGHT_SLICE = slice(7,  14)
_DIST_SLICE   = slice(14, 21)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Combined SDR Encoder
# ──────────────────────────────────────────────────────────────────────────────
class CombinedEncoder:
    """
    Encodes (21-dim stats vector, key_idx, dwell_ms, flight_ms, dist_units)
    into a sparse binary SDR.

    All 24 scalar parameters have independent enc_bits / enc_w settings:
      - 7 dwell stats features  : dwell_stats_enc_bits  / dwell_stats_enc_w
      - 7 flight stats features : flight_stats_enc_bits / flight_stats_enc_w
      - 7 dist stats features   : dist_stats_enc_bits   / dist_stats_enc_w
      - 1 last-keystroke dwell  : dwell_scalar_enc_bits / dwell_scalar_enc_w
      - 1 last-keystroke flight : flight_scalar_enc_bits/ flight_scalar_enc_w
      - 1 last-keystroke dist   : dist_scalar_enc_bits  / dist_scalar_enc_w
      (+ key one-hot: 95 bits, always 1 active — not a scalar encoder)

    SDR layout:
      [ dwell_stats | flight_stats | dist_stats | key_one_hot | dwell_key | flight_key | dist_key ]

    min_vals / max_vals (length-21 arrays) define data-derived encoding ranges
    for the stats block.  Last-keystroke scalars use fixed physical ranges.
    """

    NUM_STATS = NUM_STATS
    KEY_BITS  = NUM_KEYS

    def __init__(self, min_vals, max_vals,
                 dwell_stats_enc_bits:  int = 24, dwell_stats_enc_w:  int = 7,
                 flight_stats_enc_bits: int = 24, flight_stats_enc_w: int = 7,
                 dist_stats_enc_bits:   int = 24, dist_stats_enc_w:   int = 7,
                 dwell_scalar_enc_bits: int = 8,  dwell_scalar_enc_w: int = 5,
                 flight_scalar_enc_bits:int = 8,  flight_scalar_enc_w:int = 5,
                 dist_scalar_enc_bits:  int = 8,  dist_scalar_enc_w:  int = 5):
        assert len(min_vals) == self.NUM_STATS
        assert len(max_vals) == self.NUM_STATS
        for bits, w, name in [
            (dwell_stats_enc_bits,   dwell_stats_enc_w,   "dwell_stats"),
            (flight_stats_enc_bits,  flight_stats_enc_w,  "flight_stats"),
            (dist_stats_enc_bits,    dist_stats_enc_w,    "dist_stats"),
            (dwell_scalar_enc_bits,  dwell_scalar_enc_w,  "dwell_scalar"),
            (flight_scalar_enc_bits, flight_scalar_enc_w, "flight_scalar"),
            (dist_scalar_enc_bits,   dist_scalar_enc_w,   "dist_scalar"),
        ]:
            assert w < bits, f"{name}: enc_w ({w}) must be < enc_bits ({bits})"

        self.min_vals  = np.array(min_vals, dtype=float)
        self.max_vals  = np.array(max_vals, dtype=float)
        self._ranges   = np.maximum(self.max_vals - self.min_vals, 1e-9)

        self.dwell_stats_enc_bits   = dwell_stats_enc_bits
        self.dwell_stats_enc_w      = dwell_stats_enc_w
        self.flight_stats_enc_bits  = flight_stats_enc_bits
        self.flight_stats_enc_w     = flight_stats_enc_w
        self.dist_stats_enc_bits    = dist_stats_enc_bits
        self.dist_stats_enc_w       = dist_stats_enc_w
        self.dwell_scalar_enc_bits  = dwell_scalar_enc_bits
        self.dwell_scalar_enc_w     = dwell_scalar_enc_w
        self.flight_scalar_enc_bits = flight_scalar_enc_bits
        self.flight_scalar_enc_w    = flight_scalar_enc_w
        self.dist_scalar_enc_bits   = dist_scalar_enc_bits
        self.dist_scalar_enc_w      = dist_scalar_enc_w

        self.total_bits = (
            7 * (dwell_stats_enc_bits + flight_stats_enc_bits + dist_stats_enc_bits)
            + self.KEY_BITS
            + dwell_scalar_enc_bits + flight_scalar_enc_bits + dist_scalar_enc_bits
        )

        # Scalar encoders for last-keystroke features (fixed physical ranges)
        self._dwell_enc  = SimpleScalarEncoder(DWELL_MIN,  DWELL_MAX,
                                               dwell_scalar_enc_bits,  dwell_scalar_enc_w)
        self._flight_enc = SimpleScalarEncoder(FLIGHT_MIN, FLIGHT_MAX,
                                               flight_scalar_enc_bits, flight_scalar_enc_w)
        self._dist_enc   = SimpleScalarEncoder(DIST_MIN,   DIST_MAX,
                                               dist_scalar_enc_bits,   dist_scalar_enc_w)

    def encode(self, stats_features, key_idx: int,
               dwell: float, flight: float, distance: float) -> np.ndarray:
        """Encode a combined feature set into a dense binary uint8 SDR."""
        dense  = np.zeros(self.total_bits, dtype=np.uint8)
        offset = 0

        # ── Block 1a: dwell stats (features 0-6, data-derived ranges) ────────
        bits, w = self.dwell_stats_enc_bits, self.dwell_stats_enc_w
        for i in range(0, 7):
            val = float(np.clip(stats_features[i], self.min_vals[i], self.max_vals[i]))
            pos = (val - self.min_vals[i]) / self._ranges[i]
            idx = max(0, min(bits - w, int(pos * (bits - w))))
            for j in range(w):
                dense[offset + idx + j] = 1
            offset += bits

        # ── Block 1b: flight stats (features 7-13) ────────────────────────────
        bits, w = self.flight_stats_enc_bits, self.flight_stats_enc_w
        for i in range(7, 14):
            val = float(np.clip(stats_features[i], self.min_vals[i], self.max_vals[i]))
            pos = (val - self.min_vals[i]) / self._ranges[i]
            idx = max(0, min(bits - w, int(pos * (bits - w))))
            for j in range(w):
                dense[offset + idx + j] = 1
            offset += bits

        # ── Block 1c: dist stats (features 14-20) ────────────────────────────
        bits, w = self.dist_stats_enc_bits, self.dist_stats_enc_w
        for i in range(14, 21):
            val = float(np.clip(stats_features[i], self.min_vals[i], self.max_vals[i]))
            pos = (val - self.min_vals[i]) / self._ranges[i]
            idx = max(0, min(bits - w, int(pos * (bits - w))))
            for j in range(w):
                dense[offset + idx + j] = 1
            offset += bits

        # ── Block 2: key one-hot (95 bits, exactly 1 active) ─────────────────
        dense[offset + max(0, min(self.KEY_BITS - 1, key_idx))] = 1
        offset += self.KEY_BITS

        # ── Block 3: per-keystroke scalars (independent per-feature settings) ─
        self._dwell_enc.encode_into(dwell, dense, offset)
        offset += self.dwell_scalar_enc_bits
        self._flight_enc.encode_into(flight, dense, offset)
        offset += self.flight_scalar_enc_bits
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
    """
    if not ref_dwells or not ref_flights or not ref_dists:
        return None
    w = events[start:start + window_size]
    if len(w) < window_size:
        return None

    d_arr = np.array([e[1] for e in w], dtype=float)
    f_arr = np.array([e[2] for e in w], dtype=float)
    q_arr = np.array([e[3] for e in w], dtype=float)

    fd = extract_features(d_arr, ref_dwells,  window_size)
    ff = extract_features(f_arr, ref_flights, window_size)
    fq = extract_features(q_arr, ref_dists,   window_size)

    if fd is None or ff is None or fq is None:
        return None

    stats = fd + ff + fq   # 21 floats: 0-6 dwell, 7-13 flight, 14-20 dist

    last = w[-1]
    return stats, last[0], float(last[1]), float(last[2]), float(last[3])


def get_file_combined_seq(events, window_size: int, window_step: int,
                          ref_dwells, ref_flights, ref_dists) -> list:
    """Extract the full per-window feature sequence for one file."""
    n   = len(events)
    seq = []
    for start in range(0, n - window_size + 1, window_step):
        item = extract_combined_window(events, start, window_size,
                                       ref_dwells, ref_flights, ref_dists)
        if item is not None:
            seq.append(item)
    return seq


# ──────────────────────────────────────────────────────────────────────────────
# 4. Detection
# ──────────────────────────────────────────────────────────────────────────────
def apply_detection(seqs: list, mode: str, thresh: float,
                    warmup: int = WARMUP_STEPS,
                    labels: list = None) -> list:
    """Apply file-level detection to per-window anomaly score sequences."""
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
