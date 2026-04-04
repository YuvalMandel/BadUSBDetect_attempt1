"""
htm_velocity/htm_velocity_classify_file.py
Run a trained HTM-Velocity model on a single .txt keystroke file.

Outputs one line to stdout:
  <filepath>  HUMAN|BOT  score=<float>  thresh=<float>  windows=<int>

Usage (from project root):
  python htm_velocity/htm_velocity_classify_file.py \\
      --model hv_models/<slug>.pkl \\
      --file  path/to/recording.txt \\
      [--true_label human|bot]      # optional: prints CORRECT/WRONG
"""

import argparse
import os
import pickle
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_vel_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_vel_dir not in sys.path:
    sys.path.insert(0, _htm_vel_dir)

_htm_combined_dir = os.path.join(_project_root, "htm_combined")
if _htm_combined_dir not in sys.path:
    sys.path.insert(0, _htm_combined_dir)

os.chdir(_project_root)

import joblib
import numpy as np

try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_combined_common import parse_file_distance, WARMUP_STEPS
from htm_velocity_common import (
    get_file_velocity_seq, build_velocity_cache_from_stats,
)

POLY_MODEL_PATH = "poly_regressor.pkl"


def classify(model_data, filepath, poly_model):
    window_size  = model_data.get('window_size', 10)
    window_step  = model_data.get('window_step', 1)
    warmup       = model_data.get('warmup_steps', WARMUP_STEPS)
    best_thresh  = model_data['best_thresh']
    ref_dwells   = model_data['ref_dwells']
    ref_flights  = model_data['ref_flights']
    ref_dists    = model_data['ref_dists']
    sp           = model_data['sp']
    tm           = model_data['tm']
    encoder      = model_data['encoder']
    al           = model_data['al']
    input_width  = model_data['input_width']

    events = parse_file_distance(filepath)
    if not events:
        return None, 0, 0

    vel_dict   = build_velocity_cache_from_stats({filepath: events}, poly_model)
    vel_events = vel_dict.get(filepath, [])
    if not vel_events:
        return None, 0, 0

    seq = get_file_velocity_seq(events, vel_events, window_size, window_step,
                                ref_dwells, ref_flights, ref_dists)
    if not seq:
        return None, 0, 0

    active_columns = SDR(sp.getColumnDimensions())
    tm.reset()
    raw = []
    bot_triggered_at = None

    for step, (stats, key_idx, dwell, flight, dist, poly_mean, poly_med, poly_std) in enumerate(seq):
        enc_sdr       = SDR(input_width)
        enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist,
                                       poly_mean, poly_med, poly_std)
        sp.compute(enc_sdr, False, active_columns)
        tm.compute(active_columns, learn=False)
        score = al.compute(float(tm.anomaly))
        raw.append(score)

        if step >= warmup and bot_triggered_at is None and score >= best_thresh:
            bot_triggered_at = step

    valid = raw[warmup:]
    mean_score = float(np.mean(valid)) if valid else 0.0
    prediction = "BOT" if bot_triggered_at is not None else "HUMAN"
    return prediction, mean_score, len(seq)


def main():
    parser = argparse.ArgumentParser(description="Classify a single keystroke file")
    parser.add_argument("--model", required=True, help="Path to .pkl model file")
    parser.add_argument("--file",  required=True, help="Path to .txt keystroke file")
    parser.add_argument("--true_label", choices=["human", "bot"],
                        help="Optional ground-truth label for CORRECT/WRONG output")
    parser.add_argument("--poly_model", default=POLY_MODEL_PATH,
                        help=f"Path to poly_regressor.pkl (default: {POLY_MODEL_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.file):
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.poly_model):
        print(f"ERROR: poly_regressor not found: {args.poly_model}", file=sys.stderr)
        sys.exit(1)

    with open(args.model, 'rb') as fh:
        model_data = pickle.load(fh)

    poly_model = joblib.load(args.poly_model)

    prediction, mean_score, n_windows = classify(model_data, args.file, poly_model)

    if prediction is None:
        print(f"{args.file}  ERROR  (could not extract features)")
        sys.exit(2)

    thresh = model_data['best_thresh']
    parts  = [
        args.file,
        prediction,
        f"score={mean_score:.4f}",
        f"thresh={thresh:.4f}",
        f"windows={n_windows}",
    ]

    if args.true_label:
        true_pred = args.true_label.upper()
        verdict   = "CORRECT" if prediction == true_pred else "WRONG"
        parts.append(verdict)

    print("  ".join(parts))


if __name__ == "__main__":
    main()
