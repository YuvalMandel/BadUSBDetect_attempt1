"""
htm/htm_collect_results.py
Aggregate all completed HTM hyperparameter search runs into a leaderboard.

Usage:
  python htm/htm_collect_results.py [--top 20]

Reads:   results/cfg*.json
Writes:  results/leaderboard.csv
         results/leaderboard.txt
"""

import argparse
import csv
import glob
import json
import os

RESULTS_DIR = "results"


def load_results():
    records = []
    pattern = os.path.join(RESULTS_DIR, "cfg*.json")
    for fpath in sorted(glob.glob(pattern)):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            r["result_file"] = fpath
            records.append(r)
        except Exception as exc:
            print(f"  Warning: could not load {fpath}: {exc}")
    return records


def format_row(rank, r):
    cfg = r.get("config", {})
    key = (
        f"sp_act={cfg.get('sp_numActiveColumns','?'):>2} "
        f"pct={cfg.get('sp_potentialPct','?')} "
        f"boost={cfg.get('sp_boostStrength','?')} "
        f"enc={cfg.get('enc_bits_per_feature','?')}w{cfg.get('enc_w','?')} "
        f"tm={cfg.get('tm_cellsPerColumn','?')} "
        f"act={cfg.get('tm_activationThreshold','?')} "
        f"min={cfg.get('tm_minThreshold','?')}"
    )
    return (
        f"{rank:>4}  "
        f"cfg{r.get('config_idx',0):04d}  "
        f"{r.get('val_f1',0):.4f}   "
        f"{r.get('test_f1',0):.4f}   "
        f"{r.get('best_thresh',0):.4f}  "
        f"{key}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect HTM hyperparameter search results")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to display (default: 20)")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(f"No results found in {RESULTS_DIR}/. "
              "Run htm/htm_train_single.py jobs first.")
        return

    # Sort by test F1 desc, break ties by val F1
    records.sort(key=lambda r: (r.get("test_f1", 0), r.get("val_f1", 0)),
                 reverse=True)

    total = len(records)
    show  = min(args.top, total)

    # ---- Console leaderboard ----
    sep    = "=" * 100
    header = (f"{'Rank':>4}  {'Config':>8}  {'Val F1':>7}   "
              f"{'Test F1':>7}   {'Thresh':>7}  Key params")
    lines  = [sep,
              f"  HTM Hyperparameter Search — {total} completed runs",
              sep, header, "-" * 100]
    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))
    lines += [sep,
              f"  Best config: configs/config_{records[0].get('config_idx',0):04d}.json",
              f"  Best model : {records[0].get('model_file','(see models/)')}",
              sep]

    report = "\n".join(lines)
    print(report)

    # ---- Write leaderboard.txt ----
    txt_path = os.path.join(RESULTS_DIR, "leaderboard.txt")
    with open(txt_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nText report → {txt_path}")

    # ---- Write leaderboard.csv ----
    all_cfg_keys = []
    for r in records:
        for k in r.get("config", {}):
            if k not in all_cfg_keys:
                all_cfg_keys.append(k)

    csv_path = os.path.join(RESULTS_DIR, "leaderboard.csv")
    fieldnames = (["rank", "config_idx", "val_f1", "test_f1", "best_thresh"]
                  + all_cfg_keys
                  + ["result_file", "model_file"])

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            row = {
                "rank":        rank,
                "config_idx":  r.get("config_idx", ""),
                "val_f1":      r.get("val_f1", ""),
                "test_f1":     r.get("test_f1", ""),
                "best_thresh": r.get("best_thresh", ""),
                "result_file": r.get("result_file", ""),
                "model_file":  r.get("model_file", ""),
            }
            row.update(r.get("config", {}))
            writer.writerow(row)

    print(f"CSV report  → {csv_path}")


if __name__ == "__main__":
    main()
