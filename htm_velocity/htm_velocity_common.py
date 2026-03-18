"""
htm_velocity/htm_velocity_common.py
Shared utilities for the HTM-Velocity variant.

Extends HTM-Combined by adding 3 polynomial Fitts'-Law error channels
(mean, median, std of |predicted_flight - actual_flight|) to the SDR.

The polynomial model is Vlad's degree-12 Ridge regressor trained on the UB
keystroke dataset. It predicts flight_ms from (x1, y1, x2, y2) key positions.

SDR layout (extension of HTM-Combined):
  [ all htm_combined channels | poly_err_mean | poly_err_med | poly_err_std ]

The 3 poly error channels each have independent enc_bits / enc_w settings:
  poly_mean_enc_bits / poly_mean_enc_w  (mean  |pred - actual| over window)
  poly_med_enc_bits  / poly_med_enc_w   (median)
  poly_std_enc_bits  / poly_std_enc_w   (std)

All three use the fixed range [0, 500] ms.

velocity_cache format (per-file):
  {filepath: [(key_idx, dwell_ms, flight_ms, dist_units, poly_err), ...]}
  poly_err = |actual_flight_ms - poly_predicted_flight_ms|, or NaN if not computable.

Window sequence format (8-tuple per window):
  (stats_21, key_idx, dwell_ms, flight_ms, dist_units, poly_mean, poly_med, poly_std)
"""

import os
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_combined_dir = os.path.join(_project_root, "htm_combined")
if _htm_combined_dir not in sys.path:
    sys.path.insert(0, _htm_combined_dir)

from htm_combined_common import (               # noqa: F401
    CombinedEncoder, create_reference_pool,
    extract_combined_window, get_file_combined_seq,
    apply_detection, make_anomaly_likelihood, WARMUP_STEPS,
)
from htm_distance_common import SimpleScalarEncoder  # noqa: F401


# ── Polynomial error encoder range (ms) ──────────────────────────────────────
POLY_ERR_MIN   = 0.0
POLY_ERR_MAX   = 500.0
MIN_POLY_TRANS = 3   # minimum valid transitions per window for stats; fallback = 0/0/0


# ── Vlad's KEYBOARD_MAP (physical key positions, matches poly_regressor.pkl) ─
# (x, y) in key-unit coordinates.  Must match the training layout exactly.
KEYBOARD_MAP = {
    # Row 3 — QWERTY
    'q': (0, 3),  'w': (1, 3),  'e': (2, 3),  'r': (3, 3),  't': (4, 3),
    'y': (5, 3),  'u': (6, 3),  'i': (7, 3),  'o': (8, 3),  'p': (9, 3),
    'tab': (-0.5, 3),
    'oemopenbrackets': (10, 3), 'oemclosebrackets': (11, 3),
    # Row 2 — ASDF home row
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'oemsemicolon': (9.5, 2), 'oemquotes': (10.5, 2), 'return': (12.5, 2),
    # Row 1 — ZXCV
    'lshiftkey': (-1, 1),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'oemcomma': (8, 1), 'oemperiod': (9, 1), 'oemquestion': (10, 1),
    'rshiftkey': (12.5, 1),
    # Row 0 — bottom
    'lcontrolkey': (0, 0), 'space': (4.5, 0), 'rcontrolkey': (13, 0),
    # Row 4 — numbers
    'd1': (1, 4), 'd2': (2, 4), 'd3': (3, 4), 'd4': (4, 4), 'd5': (5, 4),
    'd6': (6, 4), 'd7': (7, 4), 'd8': (8, 4), 'd9': (9, 4), 'd0': (10, 4),
    'oemminus': (11, 4), 'oemplus': (12, 4), 'back': (13.5, 4),
    # Arrow cluster
    'up': (15, 1), 'left': (14, 0), 'down': (15, 0), 'right': (16, 0),
}


# ── CHAR_TO_KMAP: printable ASCII char (ord 32-126) → KEYBOARD_MAP key name ─
# stats_cache uses key_idx = ord(char) - 32, so char = chr(32 + key_idx).
# key_to_char() in htm_distance_common filters out Enter, Tab, modifiers, arrows,
# so only printable ASCII chars appear in stats_cache events.
CHAR_TO_KMAP: dict = {}

# Letters (upper and lower share the same physical key)
for _c in 'abcdefghijklmnopqrstuvwxyz':
    CHAR_TO_KMAP[_c]         = _c    # lowercase → same key name
    CHAR_TO_KMAP[_c.upper()] = _c    # uppercase → lowercase key name

# Space
CHAR_TO_KMAP[' '] = 'space'

# Digits 0–9
for _d in '0123456789':
    CHAR_TO_KMAP[_d] = f'd{_d}'

# Punctuation — unshifted and shifted variants share the same physical key
CHAR_TO_KMAP['.']  = 'oemperiod'
CHAR_TO_KMAP['>']  = 'oemperiod'       # Shift+.
CHAR_TO_KMAP[',']  = 'oemcomma'
CHAR_TO_KMAP['<']  = 'oemcomma'        # Shift+,
CHAR_TO_KMAP['/']  = 'oemquestion'
CHAR_TO_KMAP['?']  = 'oemquestion'     # Shift+/
CHAR_TO_KMAP['-']  = 'oemminus'
CHAR_TO_KMAP['_']  = 'oemminus'        # Shift+-
CHAR_TO_KMAP['=']  = 'oemplus'
CHAR_TO_KMAP['+']  = 'oemplus'         # Shift+=
CHAR_TO_KMAP[';']  = 'oemsemicolon'
CHAR_TO_KMAP[':']  = 'oemsemicolon'    # Shift+;
CHAR_TO_KMAP["'"]  = 'oemquotes'
CHAR_TO_KMAP['"']  = 'oemquotes'       # Shift+'
CHAR_TO_KMAP['[']  = 'oemopenbrackets'
CHAR_TO_KMAP['{']  = 'oemopenbrackets' # Shift+[
CHAR_TO_KMAP[']']  = 'oemclosebrackets'
CHAR_TO_KMAP['}']  = 'oemclosebrackets'# Shift+]
# Note: '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '\', '|'
# are absent from KEYBOARD_MAP — correctly omitted from CHAR_TO_KMAP.


# ──────────────────────────────────────────────────────────────────────────────
# 1. VelocityEncoder  (composition over CombinedEncoder)
# ──────────────────────────────────────────────────────────────────────────────
class VelocityEncoder:
    """
    Encodes a velocity feature set (21-dim stats + last-key scalars + 3 poly
    error scalars) into a sparse binary SDR.

    Wraps CombinedEncoder internally and appends 3 poly error channels:
      [ all CombinedEncoder bits | poly_err_mean | poly_err_med | poly_err_std ]

    Each poly channel uses the fixed range [POLY_ERR_MIN, POLY_ERR_MAX] = [0, 500] ms.

    Hyperparameters (18 total):
      Inherited from CombinedEncoder (12):
        dwell_stats_enc_{bits,w}, flight_stats_enc_{bits,w}, dist_stats_enc_{bits,w}
        dwell_scalar_enc_{bits,w}, flight_scalar_enc_{bits,w}, dist_scalar_enc_{bits,w}
      New poly error channels (6):
        poly_mean_enc_{bits,w}, poly_med_enc_{bits,w}, poly_std_enc_{bits,w}
    """

    def __init__(self, min_vals, max_vals,
                 # ── Inherited combined encoder params (12 HPs) ──
                 dwell_stats_enc_bits:   int = 24, dwell_stats_enc_w:   int = 7,
                 flight_stats_enc_bits:  int = 24, flight_stats_enc_w:  int = 7,
                 dist_stats_enc_bits:    int = 24, dist_stats_enc_w:    int = 7,
                 dwell_scalar_enc_bits:  int = 8,  dwell_scalar_enc_w:  int = 5,
                 flight_scalar_enc_bits: int = 8,  flight_scalar_enc_w: int = 5,
                 dist_scalar_enc_bits:   int = 8,  dist_scalar_enc_w:   int = 5,
                 # ── New poly error params (6 HPs) ──
                 poly_mean_enc_bits: int = 8,  poly_mean_enc_w: int = 3,
                 poly_med_enc_bits:  int = 8,  poly_med_enc_w:  int = 3,
                 poly_std_enc_bits:  int = 8,  poly_std_enc_w:  int = 3):

        # Build the inner CombinedEncoder
        self._combined = CombinedEncoder(
            min_vals, max_vals,
            dwell_stats_enc_bits=dwell_stats_enc_bits,
            dwell_stats_enc_w=dwell_stats_enc_w,
            flight_stats_enc_bits=flight_stats_enc_bits,
            flight_stats_enc_w=flight_stats_enc_w,
            dist_stats_enc_bits=dist_stats_enc_bits,
            dist_stats_enc_w=dist_stats_enc_w,
            dwell_scalar_enc_bits=dwell_scalar_enc_bits,
            dwell_scalar_enc_w=dwell_scalar_enc_w,
            flight_scalar_enc_bits=flight_scalar_enc_bits,
            flight_scalar_enc_w=flight_scalar_enc_w,
            dist_scalar_enc_bits=dist_scalar_enc_bits,
            dist_scalar_enc_w=dist_scalar_enc_w,
        )

        # Validate poly encoder params
        for bits, w, name in [
            (poly_mean_enc_bits, poly_mean_enc_w, "poly_mean"),
            (poly_med_enc_bits,  poly_med_enc_w,  "poly_med"),
            (poly_std_enc_bits,  poly_std_enc_w,  "poly_std"),
        ]:
            assert w < bits, f"{name}: enc_w ({w}) must be < enc_bits ({bits})"

        self.poly_mean_enc_bits = poly_mean_enc_bits
        self.poly_mean_enc_w    = poly_mean_enc_w
        self.poly_med_enc_bits  = poly_med_enc_bits
        self.poly_med_enc_w     = poly_med_enc_w
        self.poly_std_enc_bits  = poly_std_enc_bits
        self.poly_std_enc_w     = poly_std_enc_w

        self.total_bits = (
            self._combined.total_bits
            + poly_mean_enc_bits + poly_med_enc_bits + poly_std_enc_bits
        )

        # Poly error scalar encoders (fixed range 0–500 ms)
        self._poly_mean_enc = SimpleScalarEncoder(
            POLY_ERR_MIN, POLY_ERR_MAX, poly_mean_enc_bits, poly_mean_enc_w)
        self._poly_med_enc  = SimpleScalarEncoder(
            POLY_ERR_MIN, POLY_ERR_MAX, poly_med_enc_bits,  poly_med_enc_w)
        self._poly_std_enc  = SimpleScalarEncoder(
            POLY_ERR_MIN, POLY_ERR_MAX, poly_std_enc_bits,  poly_std_enc_w)

    def encode(self, stats_features, key_idx: int,
               dwell: float, flight: float, distance: float,
               poly_mean: float, poly_med: float, poly_std: float) -> np.ndarray:
        """Encode a velocity feature set into a dense binary uint8 SDR."""
        dense       = np.zeros(self.total_bits, dtype=np.uint8)
        parent_bits = self._combined.total_bits

        # Fill all CombinedEncoder channels
        dense[:parent_bits] = self._combined.encode(
            stats_features, key_idx, dwell, flight, distance)

        # Append poly error channels
        offset = parent_bits
        self._poly_mean_enc.encode_into(poly_mean, dense, offset)
        offset += self.poly_mean_enc_bits
        self._poly_med_enc.encode_into(poly_med, dense, offset)
        offset += self.poly_med_enc_bits
        self._poly_std_enc.encode_into(poly_std, dense, offset)

        return dense


# ──────────────────────────────────────────────────────────────────────────────
# 2. Velocity cache builder
# ──────────────────────────────────────────────────────────────────────────────
def build_velocity_cache_from_stats(stats_cache: dict, poly_model,
                                    show_progress: bool = False) -> dict:
    """
    Build velocity_cache from stats_cache by appending a poly_error per event.

    Input:  stats_cache = {filepath: [(key_idx, dwell_ms, flight_ms, dist_units), ...]}
    Output: vel_cache   = {filepath: [(key_idx, dwell_ms, flight_ms, dist_units, poly_err), ...]}

    poly_err = |actual_flight_ms - predicted_flight_ms|, or NaN if:
      - this is the first event in the file (no previous key to form a pair), or
      - either the previous or current key's char is not in CHAR_TO_KMAP / KEYBOARD_MAP.
    """
    if show_progress:
        try:
            from tqdm import tqdm
            items = tqdm(stats_cache.items(), desc="velocity_cache")
        except ImportError:
            items = stats_cache.items()
    else:
        items = stats_cache.items()

    vel_cache = {}
    for filepath, events in items:
        n = len(events)
        poly_errors = [float('nan')] * n

        # Batch poly predictions for efficiency (one call per file)
        valid_indices = []
        X_poly        = []
        y_true        = []

        for i in range(1, n):
            prev_key_idx = events[i - 1][0]
            curr_key_idx = events[i][0]
            flight_ms    = float(events[i][2])

            prev_char = chr(32 + prev_key_idx)
            curr_char = chr(32 + curr_key_idx)
            prev_kmap = CHAR_TO_KMAP.get(prev_char)
            curr_kmap = CHAR_TO_KMAP.get(curr_char)

            if (prev_kmap and curr_kmap
                    and prev_kmap in KEYBOARD_MAP and curr_kmap in KEYBOARD_MAP):
                x1, y1 = KEYBOARD_MAP[prev_kmap]
                x2, y2 = KEYBOARD_MAP[curr_kmap]
                valid_indices.append(i)
                X_poly.append([x1, y1, x2, y2])
                y_true.append(flight_ms)

        if X_poly:
            preds = poly_model.predict(np.array(X_poly, dtype=float))
            for idx, pred, actual in zip(valid_indices, preds, y_true):
                poly_errors[idx] = abs(actual - float(pred))

        vel_cache[filepath] = [
            (events[i][0], events[i][1], events[i][2], events[i][3], poly_errors[i])
            for i in range(n)
        ]

    return vel_cache


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_velocity_window(events, vel_events, start: int, window_size: int,
                            ref_dwells, ref_flights, ref_dists):
    """
    Extract a velocity feature 8-tuple from a window starting at `start`.

    Returns:
      (stats_21, key_idx, dwell_ms, flight_ms, dist_units, poly_mean, poly_med, poly_std)

    Falls back to poly_mean/med/std = 0.0 if fewer than MIN_POLY_TRANS valid
    (non-NaN) poly error values exist in the window.
    """
    combined = extract_combined_window(events, start, window_size,
                                       ref_dwells, ref_flights, ref_dists)
    if combined is None:
        return None

    stats_21, key_idx, dwell, flight, dist = combined

    # Collect valid (non-NaN) poly errors within the window
    w_vel = vel_events[start:start + window_size]
    poly_errors = [e[4] for e in w_vel if e[4] == e[4]]  # non-NaN filter

    if len(poly_errors) >= MIN_POLY_TRANS:
        arr       = np.array(poly_errors, dtype=float)
        poly_mean = float(np.mean(arr))
        poly_med  = float(np.median(arr))
        poly_std  = float(np.std(arr))
    else:
        poly_mean = poly_med = poly_std = 0.0

    return stats_21, key_idx, dwell, flight, dist, poly_mean, poly_med, poly_std


def get_file_velocity_seq(events, vel_events, window_size: int, window_step: int,
                          ref_dwells, ref_flights, ref_dists) -> list:
    """Extract the full per-window velocity feature sequence for one file."""
    n   = len(events)
    seq = []
    for start in range(0, n - window_size + 1, window_step):
        item = extract_velocity_window(events, vel_events, start, window_size,
                                       ref_dwells, ref_flights, ref_dists)
        if item is not None:
            seq.append(item)
    return seq
