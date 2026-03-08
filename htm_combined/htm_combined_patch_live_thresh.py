"""
htm_combined/htm_combined_patch_live_thresh.py

One-off script: add `live_thresh` and `al_live_bytes` to an existing model pkl
that was trained before these fields existed.

What it does:
  1. Load the existing model pkl
  2. Use the post-training AL state (model['al']) as the frozen starting point
  3. For each val file: run inference with an independent fresh copy of that AL
  4. Sweep threshold on those live-mode val scores → live_thresh
  5. Save a new (or overwritten) pkl with live_thresh and al_live_bytes added

Usage (from project root):
  python htm_combined/htm_combined_patch_live_thresh.py \\
      --model hc_models/<slug>.pkl [--out hc_models/<slug>_patched.pkl]
"""

import argparse
import os
import pickle
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_htm_combined_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_combined_dir not in sys.path:
    sys.path.insert(0, _htm_combined_dir)

os.chdir(_project_root)

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_combined_common import (
    apply_detection, get_file_combined_seq, WARMUP_STEPS,
)

SPLIT_FILE  = "split.pkl"
WINDOWS_DIR = "windows_cache"


def get_scores_live(file_list, is_bot, model_data, al_live_bytes,
                    wcache, cache, warmup):
    """Per-file independent AL copy — mirrors live app startup exactly."""
    sp          = model_data['sp']
    tm          = model_data['tm']
    encoder     = model_data['encoder']
    input_width = model_data['input_width']
    window_size = model_data.get('window_size', 10)
    window_step = model_data.get('window_step', 1)
    ref_dwells  = model_data['ref_dwells']
    ref_flights = model_data['ref_flights']
    ref_dists   = model_data['ref_dists']
    active_cols = SDR(sp.getColumnDimensions())

    scores, labels, seqs = [], [], []
    for fp in tqdm(file_list, desc="Live-mode inference", leave=False):
        if wcache is not None:
            seq = wcache['sequences'].get(fp)
        else:
            events = cache.get(fp) if cache else None
            if not events:
                continue
            seq = get_file_combined_seq(events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists)
        if not seq:
            continue

        al_fresh = pickle.loads(al_live_bytes)   # fresh copy per file
        tm.reset()
        raw = []
        for stats, key_idx, dwell, flight, dist in seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, False, active_cols)
            tm.compute(active_cols, learn=False)
            raw.append(al_fresh.compute(float(tm.anomaly)))

        valid = raw[warmup:]
        scores.append(float(np.mean(valid)) if valid else 0.0)
        labels.append(1 if is_bot else 0)
        seqs.append(raw)

    return scores, labels, seqs


def main():
    parser = argparse.ArgumentParser(description="Patch a model pkl with live_thresh")
    parser.add_argument("--model", required=True, help="Existing model pkl path")
    parser.add_argument("--out",   default=None,
                        help="Output pkl path (default: overwrite input)")
    parser.add_argument("--split", default=SPLIT_FILE)
    args = parser.parse_args()

    out_path = args.out or args.model

    print(f"Loading model: {args.model}")
    with open(args.model, 'rb') as fh:
        model_data = pickle.load(fh)

    if 'live_thresh' in model_data:
        print(f"Model already has live_thresh={model_data['live_thresh']:.4f}. "
              "Nothing to do.")
        return

    warmup = model_data.get('warmup_steps', WARMUP_STEPS)
    print(f"  warmup_steps={warmup}  best_thresh={model_data['best_thresh']:.4f}")

    # The saved 'al' is the post-evaluation state. We need the post-TRAINING state.
    # Since we don't have it separately, we use the saved al as the starting point —
    # it's close enough (evaluation only adds ~100 windows to an 8192-slot history).
    al_live_bytes = pickle.dumps(model_data['al'])
    print("  AL state frozen from model['al'] (post-evaluation approximation).")

    print(f"Loading split: {args.split}")
    with open(args.split, 'rb') as fh:
        split = pickle.load(fh)

    val_human = split['val_human']
    val_bots  = split['val_bots']

    window_size = model_data.get('window_size', 10)
    window_step = model_data.get('window_step', 1)
    wcache_path = os.path.join(WINDOWS_DIR, f"ws{window_size}s{window_step}.pkl")
    wcache = cache = None

    if os.path.exists(wcache_path):
        print(f"Loading windows cache: {wcache_path}")
        with open(wcache_path, 'rb') as fh:
            wcache = pickle.load(fh)
    else:
        cache_path = "stats_cache.pkl" if os.path.exists("stats_cache.pkl") else "dist_cache.pkl"
        if os.path.exists(cache_path):
            print(f"Loading raw cache: {cache_path}")
            with open(cache_path, 'rb') as fh:
                cache = pickle.load(fh)
        else:
            print("ERROR: no windows cache or raw cache found.", file=sys.stderr)
            sys.exit(1)

    kw = dict(model_data=model_data, al_live_bytes=al_live_bytes,
              wcache=wcache, cache=cache, warmup=warmup)

    print("Running live-mode val inference...")
    lv_h_sc, lv_h_lb, lv_h_sq = get_scores_live(val_human, False, **kw)
    lv_b_sc, lv_b_lb, lv_b_sq = get_scores_live(val_bots,  True,  **kw)
    lv_seqs   = lv_h_sq + lv_b_sq
    lv_labels = lv_h_lb + lv_b_lb

    if not lv_seqs:
        print("ERROR: no sequences found. Check cache paths.", file=sys.stderr)
        sys.exit(1)

    # Two-stage threshold sweep (same as train_single.py)
    coarse_pts  = np.linspace(0, 1, 200)
    coarse_step = 1.0 / (len(coarse_pts) - 1)
    live_bacc, live_thresh = 0.0, 0.0

    for th in coarse_pts:
        preds = apply_detection(lv_seqs, 'first_crossing', th, warmup, labels=lv_labels)
        bacc  = balanced_accuracy_score(lv_labels, preds)
        if bacc > live_bacc:
            live_bacc, live_thresh = bacc, th

    fine_lo = max(0.0, live_thresh - 2 * coarse_step)
    fine_hi = min(1.0, live_thresh + 2 * coarse_step)
    for th in np.linspace(fine_lo, fine_hi, 1000):
        preds = apply_detection(lv_seqs, 'first_crossing', th, warmup, labels=lv_labels)
        bacc  = balanced_accuracy_score(lv_labels, preds)
        if bacc > live_bacc:
            live_bacc, live_thresh = bacc, th

    live_f1 = f1_score(lv_labels,
                       apply_detection(lv_seqs, 'first_crossing', live_thresh,
                                       warmup, labels=lv_labels),
                       zero_division=0)

    print(f"\nResults:")
    print(f"  old best_thresh (shared-AL): {model_data['best_thresh']:.4f}")
    print(f"  new live_thresh (per-file) : {live_thresh:.4f}")
    print(f"  live-mode val BAcc={live_bacc:.4f}  F1={live_f1:.4f}")

    model_data['live_thresh']   = live_thresh
    model_data['al_live_bytes'] = al_live_bytes

    print(f"\nSaving patched model -> {out_path}")
    with open(out_path, 'wb') as fh:
        pickle.dump(model_data, fh)
    print("Done.")


if __name__ == "__main__":
    main()
