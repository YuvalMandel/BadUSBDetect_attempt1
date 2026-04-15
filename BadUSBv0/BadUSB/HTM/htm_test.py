"""
HTM Combined — session-level evaluator for the BadUSBv0 pipeline.

Loads a pre-trained hc model (pkl) and evaluates it on the person-disjoint
splits from data_split.json.  Each FILE is one session; per-window anomaly
likelihoods are aggregated to a session-level binary decision.

Run from BadUSBv0/BadUSB/:
    python HTM/htm_test.py --model results/HTM/models/htm_model_vf10.xxxx_tf10.xxxx.pkl
"""
import os, sys, json, pickle, argparse
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, balanced_accuracy_score)

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Script: BadUSBv0/BadUSB/HTM/htm_test.py
_here         = os.path.dirname(os.path.abspath(__file__))
_badusb_root  = os.path.dirname(_here)                       # BadUSBv0/BadUSB/
_project_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

for _p in (_project_root, os.path.join(_project_root, "htm_distance"), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HTM bindings
try:
    from htm.bindings.sdr import SDR                         # noqa: E402
except ImportError:
    print("ERROR: htm.core not installed. "
          "Activate the conda env used for HTM training.")
    sys.exit(1)

# HTM Combined helpers
from htm_combined_common import (                            # noqa: E402
    get_file_combined_seq, parse_file_distance)

# ==============================================================================
# Per-file scoring
# ==============================================================================

def score_session(filepath, model_data):
    """
    Run a single file (session) through the trained SP+TM+AL.

    Returns:
        al_scores  list[float]: anomaly-likelihood score per post-warmup window
                                (empty if file is too short)
    """
    window_size = model_data['window_size']
    window_step = model_data['window_step']
    warmup      = model_data['warmup_steps']
    ref_dwells  = model_data['ref_dwells']
    ref_flights = model_data['ref_flights']
    ref_dists   = model_data['ref_dists']

    events = parse_file_distance(filepath)
    if not events:
        return []

    seq = get_file_combined_seq(events, window_size, window_step,
                                ref_dwells, ref_flights, ref_dists)
    if not seq:
        return []

    sp          = model_data['sp']
    tm          = model_data['tm']
    encoder     = model_data['encoder']
    al          = model_data['al']           # shared AL state — NOT reset per file
    input_width = model_data['input_width']
    active_cols = SDR(sp.getColumnDimensions())

    tm.reset()                               # TM context reset per session
    raw = []
    for stats, key_idx, dwell, flight, dist in seq:
        sdr = SDR(input_width)
        sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
        sp.compute(sdr, False, active_cols)
        tm.compute(active_cols, learn=False)
        raw.append(al.compute(float(tm.anomaly)))

    return raw[warmup:]  # discard warmup windows


def session_label(al_scores, live_thresh, session_thresh):
    """
    1 = bot if fraction of windows above live_thresh >= session_thresh.
    """
    if not al_scores:
        return None
    frac = float(np.mean([s > live_thresh for s in al_scores]))
    return int(frac >= session_thresh)


# ==============================================================================
# Evaluate one split
# ==============================================================================

def eval_split(name, human_files, bot_files, model_data, session_thresh):
    live_thresh = model_data.get('live_thresh', 0.0)
    print(f"\n── {name.upper()}  "
          f"(humans={len(human_files)}, bots={len(bot_files)})  "
          f"live_thresh={live_thresh:.4f}  session_thresh={session_thresh}")

    labels, preds = [], []
    skipped = 0

    for fp in human_files:
        scores = score_session(fp, model_data)
        p = session_label(scores, live_thresh, session_thresh)
        if p is None:
            skipped += 1
            continue
        labels.append(0); preds.append(p)

    for fp in bot_files:
        scores = score_session(fp, model_data)
        p = session_label(scores, live_thresh, session_thresh)
        if p is None:
            skipped += 1
            continue
        labels.append(1); preds.append(p)

    if skipped:
        print(f"  Skipped {skipped} files (too short to produce any post-warmup windows).")

    if not labels:
        print("  No data to evaluate."); return

    labels = np.array(labels)
    preds  = np.array(preds)

    cm   = confusion_matrix(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    bacc = balanced_accuracy_score(labels, preds)

    print(f"  F1={f1:.4f}  BAcc={bacc:.4f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted, 0=Human/1=Bot):")
    if cm.shape == (2, 2):
        print(f"           Human  Bot")
        print(f"  Human  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"  Bot    {cm[1,0]:5d}  {cm[1,1]:5d}")
    else:
        print(f"  {cm}")
    print(classification_report(labels, preds,
                                 target_names=["Human", "Bot"],
                                 zero_division=0))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained hc model pkl")
    parser.add_argument("--split-json", default="data_split.json",
                        help="Path to data_split.json (default: data_split.json in BadUSBv0/BadUSB/)")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to evaluate (default: val test)")
    parser.add_argument("--session-thresh", type=float, default=0.5,
                        help="Fraction of windows above live_thresh needed to classify "
                             "as bot (default: 0.5)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    split_path = os.path.join(_badusb_root, args.split_json)

    for p, desc in [(model_path, "model"), (split_path, "data_split.json")]:
        if not os.path.exists(p):
            print(f"ERROR: {desc} not found: {p}"); return

    print(f"Model      : {os.path.basename(model_path)}")
    print(f"Split JSON : {split_path}")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print(f"live_thresh  = {model_data.get('live_thresh', 0):.4f}")
    print(f"window_size  = {model_data['window_size']}")
    print(f"warmup_steps = {model_data['warmup_steps']}")

    with open(split_path) as f:
        split = json.load(f)

    for name in args.splits:
        eval_split(name,
                   split[name]["humans"],
                   split[name]["bots"],
                   model_data,
                   args.session_thresh)


if __name__ == "__main__":
    main()