"""
BadUSBv0/BadUSB/HTM/htm_collect_results.py
Aggregate all completed HTM runs into a leaderboard.

Usage (from BadUSBv0/BadUSB/):
  python HTM/htm_collect_results.py [--top 20]

Reads:   results/HTM/results/htm_model_*.json
Writes:  results/HTM/leaderboard.txt
         results/HTM/leaderboard.csv
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

_here = os.path.dirname(os.path.abspath(__file__))
_badusb_root = os.path.dirname(_here)
RESULTS_DIR = os.path.join(_badusb_root, "results", "HTM", "results")
LEADERBOARD_DIR = os.path.join(_badusb_root, "results", "HTM")

def load_results(tag: str = "") -> list:
    records = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "htm_model_*.json"))):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            if tag and tag not in r.get("model_file", ""):
                continue
            r["result_file"] = fpath
            records.append(r)
        except Exception as exc:
            print(f"  Warning: could not load {fpath}: {exc}", file=sys.stderr)
    return records

def format_row(rank: int, r: dict) -> str:
    val_bacc = r.get("val_bacc", float("nan"))
    bacc_str = f"{val_bacc:.4f}" if val_bacc == val_bacc else "  N/A "
    live_thresh = r.get("live_thresh", None)
    lt_str = f"{live_thresh:.4f}" if live_thresh is not None and live_thresh > 0 else "  ---  "
    vwf = r.get("val_win_f1"); vwf_str = f"{vwf:.4f}" if vwf is not None else " N/A  "

    return (
        f"{rank:>4}  "
        f"cfg_{r.get('config_idx', 0):04d}  "
        f"{bacc_str}  "
        f"vF={r.get('val_f1', 0):.4f}/wF={vwf_str}  "
        f"{r.get('best_thresh', 0):.4f}  "
        f"lt={lt_str}"
    )

def evaluate_best_on_test(best_record, args_tag=""):
    """Load the best-val model and evaluate it on val+test, plotting full confusion matrices."""
    if _here not in sys.path:
        sys.path.insert(0, _here)
    try:
        from htm.bindings.sdr import SDR
        from htm_combined_common import make_anomaly_likelihood, apply_detection
    except ImportError as e:
        print(f"  [test eval] Cannot import HTM modules: {e}"); return

    model_path       = best_record.get("model_file")
    _tag = args_tag if args_tag else ""
    _cache_name = f"windows_cache_{_tag}.pkl" if _tag else "windows_cache.pkl"
    windows_cache    = os.path.join(_here, _cache_name)
    split_json       = os.path.join(_badusb_root, "data_split.json")

    for path, label in [(model_path, "model"), (windows_cache, "windows_cache"), (split_json, "data_split.json")]:
        if not path or not os.path.exists(path):
            print(f"  [test eval] Not found: {label} ({path})"); return

    print(f"\nEvaluating best model on val + test: {os.path.basename(model_path)}")
    with open(model_path,    'rb') as fh: mdl    = pickle.load(fh)
    with open(windows_cache, 'rb') as fh: wcache = pickle.load(fh)
    with open(split_json)        as fh: split  = json.load(fh)

    sp          = mdl["sp"];   tm      = mdl["tm"];  encoder = mdl["encoder"]
    input_width = mdl["input_width"]
    best_thresh = mdl["best_thresh"]
    warmup      = mdl.get("warmup_steps", 0)
    al_period   = mdl.get("al_period", 10)
    al_bytes    = mdl.get("al_live_bytes")
    al = pickle.loads(al_bytes) if al_bytes else make_anomaly_likelihood(al_period)
    active_columns = SDR(sp.getColumnDimensions())

    def score_files(file_list, is_bot):
        seqs, labels = [], []
        for fp in file_list:
            seq = wcache['sequences'].get(fp)
            if not seq: continue
            tm.reset()
            al_eval = pickle.loads(al_bytes)  # fresh calibrated AL per file
            raw = []
            for stats, key_idx, dwell, flight, dist in seq:
                enc = SDR(input_width)
                enc.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
                sp.compute(enc, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(al_eval.compute(float(tm.anomaly)))
            seqs.append(raw); labels.append(1 if is_bot else 0)
        return seqs, labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"HTM Best Model (cfg_{best_record['config_idx']:04d}) — Confusion Matrices", fontsize=13)
    results = {}

    for col, split_name in enumerate(["val", "test"]):
        seqs_h, lbl_h = score_files(split[split_name]["humans"], False)
        seqs_b, lbl_b = score_files(split[split_name]["bots"],   True)
        all_seqs   = seqs_h + seqs_b
        all_labels = lbl_h  + lbl_b

        file_preds = apply_detection(all_seqs, 'first_crossing', best_thresh, warmup, labels=all_labels)
        win_labels, win_preds = [], []
        for seq in seqs_h:
            for s in seq[warmup:]:
                win_labels.append(0); win_preds.append(1 if s >= best_thresh else 0)
        for seq in seqs_b:
            for s in seq[warmup:]:
                win_labels.append(1); win_preds.append(1 if s >= best_thresh else 0)

        file_f1 = f1_score(all_labels, file_preds, zero_division=0)
        win_f1  = f1_score(win_labels,  win_preds,  zero_division=0)
        results[split_name] = {"file_f1": file_f1, "win_f1": win_f1}

        print(f"  {split_name.upper()}: File-F1={file_f1:.4f} ({len(all_labels)} files) | "
              f"Win-F1={win_f1:.4f} ({len(win_labels)} windows)")

        for row, (labels, preds, subtitle) in enumerate([
            (all_labels, file_preds, f"{split_name.capitalize()} — File-level"),
            (win_labels,  win_preds, f"{split_name.capitalize()} — Window-level"),
        ]):
            ax = axes[row, col]
            if labels:
                sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d',
                            cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{subtitle}\nF1={f1_score(labels, preds, zero_division=0):.4f}")
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_xticklabels(['Human', 'Bot']); ax.set_yticklabels(['Human', 'Bot'])

    plt.tight_layout()
    plots_dir = os.path.join(LEADERBOARD_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"cfg_{best_record['config_idx']:04d}_final_confusion.png")
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices -> {fig_path}")

    final = {
        "config_idx":  best_record["config_idx"],
        "val_f1":      results["val"]["file_f1"],
        "val_win_f1":  results["val"]["win_f1"],
        "test_f1":     results["test"]["file_f1"],
        "test_win_f1": results["test"]["win_f1"],
        "thresh":      float(best_thresh),
    }
    _tag_suffix = f"_{args_tag}" if args_tag else ""
    out = os.path.join(LEADERBOARD_DIR, f"final_test_results{_tag_suffix}.json")
    with open(out, 'w') as fh: json.dump(final, fh, indent=2)
    print(f"  Final results JSON -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Collect HTM hyperparameter search results")
    parser.add_argument("--top", type=int, default=20, help="Number of top configs to display")
    parser.add_argument("--tag", type=str, default="", help="Filter to results whose model_file contains this tag (e.g. 'fullkey')")
    args = parser.parse_args()

    records = load_results(tag=args.tag)
    if not records:
        print(f"No results found in {RESULTS_DIR}/. Run training jobs first.")
        sys.exit(0)

    records.sort(
        key=lambda r: (1 if r.get("live_thresh", 0) > 0 else 0, r.get("val_f1", 0), r.get("val_bacc", 0)),
        reverse=True
    )

    total = len(records)
    show = min(args.top, total)

    header = (f"{'Rank':>4}  {'Config':>9}  {'ValBAcc':>7}  "
              f"{'Val File-F1/Win-F1':>22}  "
              f"{'Thresh':>7}  {'LiveThresh':>10}")
    tag_label = f" [{args.tag}]" if args.tag else ""
    lines = ["=" * 80, f"HTM Hyperparameter Search{tag_label} -- {total} completed runs", "=" * 80, header, "-" * 80]

    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))

    if records:
        best = records[0]
        lines.extend([
            "=" * 80,
            f"Best config: HTM/configs/config_{best.get('config_idx', 0):04d}.json",
            f"Best model:  {best.get('model_file', 'N/A')}",
            f"Best val F1: {best.get('val_f1', 0):.4f}  (test evaluated separately below)",
            "=" * 80,
        ])

    report = "\n".join(lines)
    print(report)

    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    tag_suffix = f"_{args.tag}" if args.tag else ""
    txt_path = os.path.join(LEADERBOARD_DIR, f"leaderboard{tag_suffix}.txt")
    with open(txt_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nText report -> {txt_path}")

    csv_path = os.path.join(LEADERBOARD_DIR, f"leaderboard{tag_suffix}.csv")
    all_cfg_keys = sorted(list(set(k for r in records for k in r.get("config", {}).keys())))
    fieldnames = ["rank", "config_idx", "val_bacc", "val_f1", "val_win_f1", "best_thresh", "live_thresh"] + all_cfg_keys
    
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            row = {"rank": rank, **r}
            row.update(r.get("config", {}))
            writer.writerow(row)
    print(f"CSV report  -> {csv_path}")

    evaluate_best_on_test(records[0], args_tag=args.tag)

if __name__ == "__main__":
    main()
