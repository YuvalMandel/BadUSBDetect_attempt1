"""
htm_distance/htm_distance_common.py
Shared utilities for the HTM-distance variant.

Input to the HTM — one step per keystroke event (no windowing, no stats):
  key_idx  : 0–94  (printable ASCII 32–126: space → ~)
  dwell_ms : key hold duration in ms     (range 0–400 ms from data)
  flight_ms: time from previous key-up   (range 0–500 ms from data)
  dist     : Euclidean QWERTY distance from previous key (0–12 units)

Unknown / modifier keys are silently skipped — no unknown types reach the HTM.
"""

import inspect
import os
import sys
import math
import numpy as np

# ──────────────────────────────────────────────────────────────────
# 1. QWERTY physical layout (key positions in key-unit coordinates)
#    Each row is staggered as on a real keyboard:
#      row 0 (numbers):   y=0,  x starts at 0
#      row 1 (QWERTY):    y=1,  x starts at 0.5
#      row 2 (ASDF home): y=2,  x starts at 0.75
#      row 3 (ZXCV):      y=3,  x starts at 1.25
#      row 4 (space):     y=4
# ──────────────────────────────────────────────────────────────────
_R0 = 0.00   # number row
_R1 = 0.50   # QWERTY row stagger
_R2 = 0.75   # home row stagger
_R3 = 1.25   # bottom row stagger

QWERTY_POS: dict[str, tuple[float, float]] = {
    # (x, y) — x is column, y is row

    # ── Number row (y=0) ──────────────────────────────────────────
    '`': (_R0 + 0,  0), '1': (_R0 + 1,  0), '2': (_R0 + 2,  0),
    '3': (_R0 + 3,  0), '4': (_R0 + 4,  0), '5': (_R0 + 5,  0),
    '6': (_R0 + 6,  0), '7': (_R0 + 7,  0), '8': (_R0 + 8,  0),
    '9': (_R0 + 9,  0), '0': (_R0 + 10, 0), '-': (_R0 + 11, 0),
    '=': (_R0 + 12, 0),

    # ── QWERTY row (y=1) ─────────────────────────────────────────
    'q': (_R1 + 0,  1), 'w': (_R1 + 1,  1), 'e': (_R1 + 2,  1),
    'r': (_R1 + 3,  1), 't': (_R1 + 4,  1), 'y': (_R1 + 5,  1),
    'u': (_R1 + 6,  1), 'i': (_R1 + 7,  1), 'o': (_R1 + 8,  1),
    'p': (_R1 + 9,  1), '[': (_R1 + 10, 1), ']': (_R1 + 11, 1),
    '\\':(_R1 + 12, 1),

    # ── Home row (y=2) ───────────────────────────────────────────
    'a': (_R2 + 0,  2), 's': (_R2 + 1,  2), 'd': (_R2 + 2,  2),
    'f': (_R2 + 3,  2), 'g': (_R2 + 4,  2), 'h': (_R2 + 5,  2),
    'j': (_R2 + 6,  2), 'k': (_R2 + 7,  2), 'l': (_R2 + 8,  2),
    ';': (_R2 + 9,  2), "'": (_R2 + 10, 2),

    # ── Bottom row (y=3) ─────────────────────────────────────────
    'z': (_R3 + 0,  3), 'x': (_R3 + 1,  3), 'c': (_R3 + 2,  3),
    'v': (_R3 + 3,  3), 'b': (_R3 + 4,  3), 'n': (_R3 + 5,  3),
    'm': (_R3 + 6,  3), ',': (_R3 + 7,  3), '.': (_R3 + 8,  3),
    '/': (_R3 + 9,  3),

    # ── Space bar (y=4) ──────────────────────────────────────────
    ' ': (5.5, 4),
}

# Uppercase letters share the same physical key as lowercase
for _c in list(QWERTY_POS.keys()):
    if _c.isalpha() and _c.islower() and _c.upper() not in QWERTY_POS:
        QWERTY_POS[_c.upper()] = QWERTY_POS[_c]

# Shifted number-row symbols share the same physical key as their base
_SHIFT_BASE = {
    '~': '`', '!': '1', '@': '2', '#': '3', '$': '4',
    '%': '5', '^': '6', '&': '7', '*': '8', '(': '9',
    ')': '0', '_': '-', '+': '=',
    '{': '[', '}': ']', '|': '\\',
    ':': ';', '"': "'", '<': ',', '>': '.', '?': '/',
}
for _shifted, _base in _SHIFT_BASE.items():
    if _base in QWERTY_POS and _shifted not in QWERTY_POS:
        QWERTY_POS[_shifted] = QWERTY_POS[_base]

# Maximum possible distance (backtick at (0,0) to / at ~(10.25, 3)) ≈ 10.7
QWERTY_MAX_DIST = 12.0   # safe upper bound


def qwerty_distance(char_a: str, char_b: str) -> float:
    """
    Euclidean distance between two characters on the QWERTY keyboard.
    Returns 0.0 if either character is not in the layout.
    """
    pos_a = QWERTY_POS.get(char_a)
    pos_b = QWERTY_POS.get(char_b)
    if pos_a is None or pos_b is None:
        return 0.0
    return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


# ──────────────────────────────────────────────────────────────────
# 2. Key string → printable ASCII character
# ──────────────────────────────────────────────────────────────────
_OEM_MAP: dict = {
    # Space bar
    'Space': ' ',
    # Punctuation / symbol keys (US QWERTY)
    'Oem1':        ';',   # ;:
    'Oem2':        '/',   # /?
    'OemQuestion': '/',   # alias for Oem2
    'Oem3':        '`',   # `~
    'Oem4':        '[',   # [{
    'Oem5':        '\\',  # \|
    'Oem6':        ']',   # ]}
    'Oem7':        "'",   # '"
    'OemMinus':    '-',   # -_
    'OemPlus':     '=',   # =+
    'Oemcomma':    ',',   # ,<
    'OemPeriod':   '.',   # .>
}


def key_to_char(key_str: str):
    """
    Convert a raw key string from a Windows keystroke log to a single
    printable ASCII character (ord 32–126), or None to skip the event.

    Handles the Windows key-naming convention used by this dataset:
      A–Z          → single uppercase letter  (passed through as-is)
      D0–D9        → digit '0'–'9'
      Space        → ' '
      Oem1 … etc.  → punctuation character
      -  .  , etc. → already a printable ASCII char (bot-log format)
      Key.space    → ' '   (pynput legacy)
      Key.*        → None  (all other specials)
      Everything else (shift, ctrl, F-keys, arrows …) → None
    """
    s = key_str.strip()

    # ── OEM / named keys (includes 'Space') ──────────────────────
    if s in _OEM_MAP:
        return _OEM_MAP[s]

    # ── Digit keys: D0–D9 ────────────────────────────────────────
    if len(s) == 2 and s[0] == 'D' and s[1].isdigit():
        return s[1]          # 'D3' → '3'

    # ── pynput quoted format: 'a'  '\\'  "' '" ──────────────────
    if len(s) >= 3 and s[0] == "'" and s[-1] == "'":
        inner = s[1:-1]
        if inner == "\\'":
            inner = "'"
        elif inner == '\\\\':
            inner = '\\'
        s = inner

    # ── pynput special: Key.space ─────────────────────────────────
    if s == 'Key.space':
        return ' '
    if s.startswith('Key.'):
        return None

    # ── Single printable ASCII character (letters A–Z, symbols …) ─
    if len(s) == 1 and 32 <= ord(s) <= 126:
        return s

    return None   # modifier, function key, arrow, etc. — skip


def char_to_idx(char: str) -> int:
    """Map printable ASCII character to index 0–94 (ASCII 32 → 0, ASCII 126 → 94)."""
    return ord(char) - 32


# ──────────────────────────────────────────────────────────────────
# 3. File parser — produces one 4-tuple per valid keystroke
# ──────────────────────────────────────────────────────────────────
# Timing bounds (ms) — consistent with common/keystroke_features.py
_DWELL_MAX  = 3000.0
_FLIGHT_MAX = 5000.0

def parse_file_distance(filepath: str) -> list:
    """
    Parse a keystroke log and return a list of 4-tuples:
      (key_idx, dwell_ms, flight_ms, dist_units)

    Rules:
    - Only printable ASCII 32–126 keys are kept; all others are skipped.
    - flight_ms = 0.0 and dist_units = 0.0 for the very first keystroke.
    - Negative flight (overlapping keys) is clamped to 0.
    - Keystrokes with out-of-range dwell are discarded.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
            lines = fh.readlines()
    except Exception:
        return []

    # Detect timestamp scale (Windows 100-ns ticks vs. ms)
    scale = 1.0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                ts = int(parts[2])
                scale = 10000.0 if ts > 10 ** 15 else 1.0
                break
            except ValueError:
                pass

    # First pass: collect (char, key_down_ts, key_up_ts) triples
    active: dict[str, tuple[str, int]] = {}  # key_str → (char, down_ts)
    key_events: list[tuple[str, float, float]] = []  # (char, down_ms, up_ms)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key_str, action = parts[0], parts[1]
        try:
            ts = int(parts[2])
        except ValueError:
            continue

        char = key_to_char(key_str)
        if char is None:
            continue   # skip modifiers / unknown keys entirely

        if action == 'KeyDown':
            active[key_str] = (char, ts)
        elif action == 'KeyUp':
            if key_str in active:
                orig_char, down_ts = active.pop(key_str)
                dwell_ms = (ts - down_ts) / scale
                if 0 < dwell_ms < _DWELL_MAX:
                    key_events.append((orig_char, down_ts / scale, ts / scale))

    # Sort by press time
    key_events.sort(key=lambda e: e[1])

    # Second pass: compute flight and distance relative to previous event
    result: list[tuple[int, float, float, float]] = []
    prev_char: str | None = None
    prev_up_ms: float | None = None

    for char, down_ms, up_ms in key_events:
        dwell_ms = up_ms - down_ms

        if prev_up_ms is not None:
            flight_ms = down_ms - prev_up_ms
            if flight_ms < 0:
                flight_ms = 0.0          # overlapping keystrokes
            elif flight_ms > _FLIGHT_MAX:
                # Large gap — still valid but will be clipped by encoder
                flight_ms = min(flight_ms, _FLIGHT_MAX)
        else:
            flight_ms = 0.0              # first keystroke: no prior context

        dist = qwerty_distance(prev_char, char) if prev_char is not None else 0.0

        result.append((char_to_idx(char), dwell_ms, flight_ms, dist))
        prev_char  = char
        prev_up_ms = up_ms

    return result


# ──────────────────────────────────────────────────────────────────
# 4. SDR encoder
# ──────────────────────────────────────────────────────────────────
# Fixed physical ranges used by all configs (no data-derived min/max)
NUM_KEYS   = 95        # printable ASCII 32–126 → indices 0–94
DWELL_MIN,  DWELL_MAX  =  0, 400.0   # ms  (plot: 30–300 ms bulk)
FLIGHT_MIN, FLIGHT_MAX =  0, 500.0   # ms  (plot: 0–400 ms bulk)
DIST_MIN,   DIST_MAX   =  0,  12.0   # QWERTY Euclidean units


class SimpleScalarEncoder:
    """Encodes a single scalar into a contiguous active-bit block."""
    def __init__(self, min_val, max_val, n_bits, w):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.n_bits  = n_bits
        self.w       = w
        self.range   = max(float(max_val - min_val), 1e-9)

    def encode_into(self, value: float, arr: np.ndarray, offset: int):
        value = float(np.clip(value, self.min_val, self.max_val))
        pos   = (value - self.min_val) / self.range
        idx   = int(pos * (self.n_bits - self.w))
        idx   = max(0, min(self.n_bits - self.w, idx))
        arr[offset + idx : offset + idx + self.w] = 1


class DistanceEncoder:
    """
    SDR for one keystroke event:

      [key_type | dwell | flight | distance]

    key_type : 95 bits, exactly 1 active (true one-hot).
               Each of the 95 printable ASCII keys gets its own unique bit;
               no two keys ever share any active bit.

    dwell    : bits_per_feature bits, w active  (scalar, overlapping blocks)
    flight   : bits_per_feature bits, w active
    distance : bits_per_feature bits, w active

    Total bits   = 95 + 3 * bits_per_feature
    Active bits  =  1 + 3 * w
    """
    KEY_BITS = NUM_KEYS   # 95, fixed — not a tunable hyperparameter

    def __init__(self, bits_per_feature: int = 32, w: int = 5):
        self.bits_per_feature = bits_per_feature
        self.w                = w
        self.total_bits       = self.KEY_BITS + 3 * bits_per_feature
        self._scalar_enc = [
            SimpleScalarEncoder(DWELL_MIN,  DWELL_MAX,  bits_per_feature, w),
            SimpleScalarEncoder(FLIGHT_MIN, FLIGHT_MAX, bits_per_feature, w),
            SimpleScalarEncoder(DIST_MIN,   DIST_MAX,   bits_per_feature, w),
        ]

    def encode(self, key_idx: int, dwell: float,
               flight: float, distance: float) -> np.ndarray:
        dense = np.zeros(self.total_bits, dtype=np.uint8)

        # One-hot key: bit at position key_idx (clamped to 0–94)
        dense[max(0, min(self.KEY_BITS - 1, key_idx))] = 1

        # Scalar features starting right after the key block
        offset = self.KEY_BITS
        for enc, val in zip(self._scalar_enc, (dwell, flight, distance)):
            enc.encode_into(val, dense, offset)
            offset += self.bits_per_feature

        return dense


# ──────────────────────────────────────────────────────────────────
# 5. AnomalyLikelihood factory (version-safe)
# ──────────────────────────────────────────────────────────────────
class AnomalyLikelihoodWrapper:
    """
    Version-safe wrapper around htm AnomalyLikelihood.

    Exposes a single `compute(raw_anomaly) -> float` interface, hiding
    differences in method signatures across htm.core versions:

      Modern htm.core  : anomalyProbability(value, anomaly) — two args
      Some builds      : compute(anomaly)                   — one arg
      Other builds     : anomalyLikelihood(anomaly)         — one arg

    Picklable: stores only the method name string, not a bound method.
    If no scoring method is found, `compute()` returns the raw anomaly as-is.
    """

    # Methods tried in preference order; 2-arg methods must be in _TWO_ARG
    _CANDIDATES = ["anomalyProbability", "compute",
                   "anomalyLikelihood",  "update", "likelihood"]
    _TWO_ARG    = {"anomalyProbability"}

    def __init__(self, al, method_name: str):
        self._al          = al
        self._method_name = method_name   # only the name — stays picklable

    def compute(self, raw_anomaly: float) -> float:
        if self._al is None or self._method_name is None:
            return raw_anomaly
        method = getattr(self._al, self._method_name)
        if self._method_name in self._TWO_ARG:
            return float(method(raw_anomaly, raw_anomaly))
        return float(method(raw_anomaly))


def make_anomaly_likelihood(period: int = 10) -> AnomalyLikelihoodWrapper:
    """
    Construct an AnomalyLikelihoodWrapper compatible with the installed
    version of htm.core.

    Constructor parameter names vary by version:
      htm.core (htm-community, modern) : learningPeriod=N
      nupic-style / older builds       : claLearningPeriod=N
      Unknown / C++ binding only       : no-arg construction (uses defaults)

    Returns an AnomalyLikelihoodWrapper with a `.compute(raw)` interface.
    If htm.algorithms is unavailable, `.compute()` returns raw anomaly unchanged.
    """
    try:
        from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
    except ImportError:
        return AnomalyLikelihoodWrapper(None, None)

    # ── Build the AL object ───────────────────────────────────────
    try:
        params = set(inspect.signature(AnomalyLikelihood.__init__).parameters)
    except (ValueError, TypeError):
        params = set()

    kw: dict = {}
    if "learningPeriod" in params:
        kw["learningPeriod"]     = period
        kw["estimationSamples"]  = max(period, 10)
        kw["historicWindowSize"] = 8192
        kw["reestimationPeriod"] = period
    elif "claLearningPeriod" in params:
        kw["claLearningPeriod"]  = period
        kw["estimationSamples"]  = max(period, 10)
        kw["historicWindowSize"] = 8192
        kw["reestimationPeriod"] = period
    # else: use defaults (period parameter ignored — AL still helps)

    al = AnomalyLikelihood(**kw)

    # ── Discover which scoring method this version exposes ────────
    method_name = next(
        (m for m in AnomalyLikelihoodWrapper._CANDIDATES if hasattr(al, m)),
        None)

    return AnomalyLikelihoodWrapper(al, method_name)


# ──────────────────────────────────────────────────────────────────
# 6. Detection (first-crossing, same logic as htm/htm_common.py)
# ──────────────────────────────────────────────────────────────────
WARMUP_STEPS = 50   # keystrokes to skip before alarming (more than original
                    # because each step is a single keystroke, not a window)


def apply_detection(seqs: list, mode: str, thresh: float,
                    warmup: int = WARMUP_STEPS,
                    labels: list = None) -> list:
    """
    mode='first_crossing': file is Bot if any post-warmup score >= thresh.
    mode='mean':           file is Bot if mean(post-warmup scores) >= thresh.

    Files shorter than warmup have no post-warmup keystrokes and are always
    treated as misclassified: prediction = 1 - true_label.  Pass `labels` to
    enable this; without labels, short files fall back to predicting Human (0).
    """
    preds = []
    for i, seq in enumerate(seqs):
        post = seq[warmup:]   # empty if len(seq) <= warmup
        if not post:
            # Too short — always wrong
            preds.append(1 - labels[i] if labels is not None else 0)
            continue
        if mode == 'first_crossing':
            preds.append(1 if any(s >= thresh for s in post) else 0)
        else:
            preds.append(1 if float(np.mean(post)) >= thresh else 0)
    return preds
