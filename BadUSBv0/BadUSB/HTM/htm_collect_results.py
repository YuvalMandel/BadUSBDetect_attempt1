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
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_badusb_root = os.path.dirname(_here)
RESULTS_DIR = os.path.join(_badusb_root, "results", "HTM", "results")
LEADERBOARD_DIR = os.path.join(_badusb_root, "results", "HTM")

def load_results() -> list:
    records = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "htm_model_*.json"))):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            r["result_file"] = fpath
            records.append(r)
        except Exception as exc:
            print(f"  Warning: could not load {fpath}: {exc}", file=sys.stderr)
    return records

def format_row(rank: int, r: dict) -> str:
    cfg = r.get("config", {})
    val_bacc = r.get("val_bacc", float("nan"))
    bacc_str = f"{val_bacc:.4f}" if val_bacc == val_bacc else "  N/A "
    live_thresh = r.get("live_thresh", None)
    lt_str = f"{live_thresh:.4f}" if live_thresh is not None and live_thresh > 0 else "  ---  "
    
    return (
        f"{rank:>4}  "
        f"cfg_{r.get('config_idx', 0):04d}  "
        f"{bacc_str}  "
        f"{r.get('val_f1', 0):.4f}   "
        f"{r.get('test_f1', 0):.4f}   "
        f"{r.get('best_thresh', 0):.4f}  "
        f"lt={lt_str}"
    )

def main():
    parser = argparse.ArgumentParser(description="Collect HTM hyperparameter search results")
    parser.add_argument("--top", type=int, default=20, help="Number of top configs to display")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(f"No results found in {RESULTS_DIR}/. Run training jobs first.")
        sys.exit(0)

    records.sort(
        key=lambda r: (1 if r.get("live_thresh", 0) > 0 else 0, r.get("test_f1", 0), r.get("val_bacc", 0)),
        reverse=True
    )

    total = len(records)
    show = min(args.top, total)

    header = (f"{'Rank':>4}  {'Config':>9}  {'ValBAcc':>7}  {'Val F1':>7}   "
              f"{'Test F1':>7}   {'Thresh':>7}  {'LiveThresh':>10}")
    lines = ["=" * 80, f"HTM Hyperparameter Search -- {total} completed runs", "=" * 80, header, "-" * 80]

    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))

    if records:
        best = records[0]
        lines.extend([
            "=" * 80,
            f"Best config: HTM/configs/config_{best.get('config_idx', 0):04d}.json",
            f"Best model: {best.get('model_file', 'N/A')}",
            f"Best test F1: {best.get('test_f1', 0):.4f}",
            "=" * 80,
        ])

    report = "\n".join(lines)
    print(report)

    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    txt_path = os.path.join(LEADERBOARD_DIR, "leaderboard.txt")
    with open(txt_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nText report -> {txt_path}")

    csv_path = os.path.join(LEADERBOARD_DIR, "leaderboard.csv")
    all_cfg_keys = sorted(list(set(k for r in records for k in r.get("config", {}).keys())))
    fieldnames = ["rank", "config_idx", "val_bacc", "val_f1", "test_f1", "best_thresh", "live_thresh"] + all_cfg_keys
    
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            row = {"rank": rank, **r}
            row.update(r.get("config", {}))
            writer.writerow(row)
    print(f"CSV report  -> {csv_path}")

if __name__ == "__main__":
    main()
