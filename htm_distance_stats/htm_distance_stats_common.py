"""
htm_distance_stats/htm_distance_stats_common.py
Shared utilities for the HTM-Distance-Stats variant.

Approach: sliding windows over printable-keystroke sequences, extracting a
21-dimensional statistical feature vector per window — fed into an HTM.

  Group 1 (features  0– 6): 7 stats on dwell times     (ms)
  Group 2 (features  7–13): 7 stats on flight times     (ms)
  Group 3 (features 14–20): 7 stats on QWERTY distances (key-unit Euclidean)

Each group uses the same 7-statistic pipeline as htm/:
  [mean, median, std, skew, kurtosis, min_KS_distance, min_Wasserstein_distance]

Key differences from htm/:
  * Parser:  parse_file_distance (only printable ASCII keys; computes QWERTY dist)
  * Features: 21-dim (adds a distance-stats group) instead of 14-dim
  * Window:  typically 3–7 keystrokes (small: bot files have few printable keys)

Key differences from htm_distance/:
  * Each HTM step is a WINDOW of keystrokes, not a single keystroke event
  * SDR encodes statistics, not raw (key_type, dwell, flight, dist) tuples
  * No key one-hot block — key identity is implicitly captured via distance stats
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
# Re-export parse_file_distance and AnomalyLikelihood utilities from htm_distance
from htm_distance_common import parse_file_distance, make_anomaly_likelihood  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
NUM_FEATURES = 21    # 7 stats × 3 channels (dwell, flight, qwerty_dist)
WARMUP_STEPS = 0     # No warmup: AnomalyLikelihood handles the first-window spike.
                     # Unlike htm_distance/, window_size is 10–30 keystrokes so human
                     # files produce many windows; no need to discard any.


# ──────────────────────────────────────────────────────────────────────────────
# 1. SDR Encoder
# ──────────────────────────────────────────────────────────────────────────────
class StatsEncoder:
    """
    Encodes a 21-dim statistical feature vector into a sparse binary SDR.

    Each of the 21 features is independently encoded as a contiguous
    active-bit block (same scheme as htm/'s MultiAttributeEncoder):

        total_bits  = 21 × bits_per_feature
        active_bits = 21 × w

    min_vals / max_vals (length-21 arrays) define the scalar encoding range
    per feature; values outside the range are clipped.
    """
    NUM_FEATURES = 21

    def __init__(self, min_vals, max_vals, bits_per_feature: int = 16, w: int = 5):
        assert len(min_vals) == self.NUM_FEATURES, \
            f"Expected {self.NUM_FEATURES} min values, got {len(min_vals)}"
        assert len(max_vals) == self.NUM_FEATURES, \
            f"Expected {self.NUM_FEATURES} max values, got {len(max_vals)}"
        self.min_vals         = np.array(min_vals, dtype=float)
        self.max_vals         = np.array(max_vals, dtype=float)
        self.bits_per_feature = bits_per_feature
        self.w                = w
        self.total_bits       = self.NUM_FEATURES * bits_per_feature
        self._ranges          = np.maximum(self.max_vals - self.min_vals, 1e-9)

    def encode(self, features) -> np.ndarray:
        """Encode a 21-dim feature vector into a dense binary uint8 array."""
        dense    = np.zeros(self.total_bits, dtype=np.uint8)
        feat_arr = np.asarray(features, dtype=float)
        n_bits   = self.bits_per_feature
        w        = self.w
        for i in range(self.NUM_FEATURES):
            val = float(np.clip(feat_arr[i], self.min_vals[i], self.max_vals[i]))
            pos = (val - self.min_vals[i]) / self._ranges[i]
            idx = int(pos * (n_bits - w))
            idx = max(0, min(n_bits - w, idx))
            offset = i * n_bits
            for j in range(w):
                dense[offset + idx + j] = 1
        return dense


# ──────────────────────────────────────────────────────────────────────────────
# 2. Reference pool
# ──────────────────────────────────────────────────────────────────────────────
def create_reference_pool_stats(train_human_files, stats_cache,
                                window_size: int = 5,
                                num_references: int = NUM_REFERENCES,
                                seed: int = RANDOM_SEED):
    """
    Build KS/Wasserstein reference windows for dwell, flight, and QWERTY
    distance from training human files using pre-parsed events in stats_cache.

    Parameters
    ----------
    train_human_files : list[str]
        File paths in the training split (human files only — no leakage).
    stats_cache : dict[str, list[tuple]]
        Maps filepath → [(key_idx, dwell_ms, flight_ms, dist_units), ...].
    window_size : int
        Length of each reference window (should match the config window_size).
    num_references : int
        Number of reference windows sampled per channel.
    seed : int
        RNG seed for deterministic sampling.

    Returns
    -------
    (ref_dwells, ref_flights, ref_dists) : tuple of lists of numpy arrays
        Each list contains `num_references` arrays of length `window_size`.
        Returns empty lists if not enough training data is available.
    """
    rng = random.Random(seed)

    all_dwells:  list = []
    all_flights: list = []
    all_dists:   list = []

    files = [f for f in train_human_files if f in stats_cache]
    rng.shuffle(files)

    for fp in files[:50]:
        for _, dwell, flight, dist in stats_cache[fp]:
            all_dwells.append(float(dwell))
            all_flights.append(float(flight))
            all_dists.append(float(dist))

    def _sample(arr: list) -> list:
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
# 3. Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_window_features(events, start: int, window_size: int,
                            ref_dwells, ref_flights, ref_dists):
    """
    Extract a 21-dim feature vector from events[start : start + window_size].

    Parameters
    ----------
    events      : list of (key_idx, dwell_ms, flight_ms, dist_units) tuples
    start       : first event index in the window
    window_size : number of keystrokes per window
    ref_dwells, ref_flights, ref_dists : lists of numpy arrays (reference windows)

    Returns
    -------
    list of 21 floats, or None if extraction fails (too short, NaN, empty refs).
    """
    if not ref_dwells or not ref_flights or not ref_dists:
        return None  # Reference pool not built — cannot compute KS/Wasserstein

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

    return fd + ff + fq   # list concatenation → 21 floats


def get_file_feature_seq(events, window_size: int, step_size: int,
                         ref_dwells, ref_flights, ref_dists) -> list:
    """
    Extract the full per-window feature sequence for one file.

    Returns a list of 21-dim feature vectors (one per valid sliding window).
    Returns an empty list if the file has fewer than `window_size` events.
    """
    n   = len(events)
    seq = []
    for start in range(0, n - window_size + 1, step_size):
        feat = extract_window_features(events, start, window_size,
                                       ref_dwells, ref_flights, ref_dists)
        if feat is not None:
            seq.append(feat)
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

    Files with no post-warmup windows (len(seq) <= warmup, or empty seq)
    are always treated as misclassified — pass `labels` to enable this
    behaviour; without labels, they fall back to predicting Human (0).
    """
    preds = []
    for i, seq in enumerate(seqs):
        post = seq[warmup:]
        if not post:
            # Short file: no evidence — always wrong
            preds.append(1 - labels[i] if labels is not None else 0)
            continue
        if mode == 'first_crossing':
            preds.append(1 if any(s >= thresh for s in post) else 0)
        else:
            preds.append(1 if float(np.mean(post)) >= thresh else 0)
    return preds
