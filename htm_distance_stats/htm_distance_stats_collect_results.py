"""
htm_distance_stats/htm_distance_stats_collect_results.py
Aggregate all completed HTM-Distance-Stats runs into a leaderboard.

Usage (from project root):
  python htm_distance_stats/htm_distance_stats_collect_results.py [--top 20]

Reads:   ds_results/ds*.json
Writes:  ds_results/leaderboard.txt
         ds_results/leaderboard.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

RESULTS_DIR = "ds_results"


def load_results() -> list:
    records = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "ds*.json"))):
        try:
            with open(fpath) as fh:
                r = json.load(fh)
            r["result_file"] = fpath
            records.append(r)
        except Exception as exc:
            print(f"  Warning: could not load {fpath}: {exc}")
    return records


def format_row(rank: int, r: dict) -> str:
    cfg     = r.get("config", {})
    detect  = r.get("mean_bot_detect_step", -1)
    det_str = f"{detect:5.1f}" if detect >= 0 else "  N/A"
    caught  = r.get("n_bots_caught", "?")
    total   = r.get("n_bots_total",  "?")
    b_sc    = r.get("mean_bot_score",   0.0)
    h_sc    = r.get("mean_human_score", 0.0)
    key = (
        f"sp_act={cfg.get('sp_numActiveColumns','?'):>2} "
        f"pct={cfg.get('sp_potentialPct','?')} "
        f"enc={cfg.get('enc_bits_per_feature','?')}w{cfg.get('enc_w','?')} "
        f"ws={cfg.get('window_size','?')}s{cfg.get('window_step','?')} "
        f"tm={cfg.get('tm_cellsPerColumn','?')} "
        f"act={cfg.get('tm_activationThreshold','?')} "
        f"min={cfg.get('tm_minThreshold','?')} "
        f"wu={cfg.get('warmup_steps','?')} "
        f"al={cfg.get('al_period','?')}"
    )
    val_bacc = r.get("val_bacc", float("nan"))
    bacc_str = f"{val_bacc:.4f}" if val_bacc == val_bacc else "  N/A "
    return (
        f"{rank:>4}  "
        f"ds{r.get('config_idx', 0):04d}  "
        f"{bacc_str}  "
        f"{r.get('val_f1', 0):.4f}   "
        f"{r.get('test_f1', 0):.4f}   "
        f"{r.get('best_thresh', 0):.4f}  "
        f"det={det_str}  "
        f"caught={caught}/{total}  "
        f"bSc={b_sc:.3f}  hSc={h_sc:.3f}  "
        f"{key}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect HTM-Distance-Stats hyperparameter search results")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to display (default: 20)")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(f"No results found in {RESULTS_DIR}/. "
              "Run htm_distance_stats_train_single.py jobs first.")
        sys.exit(0)

    records.sort(key=lambda r: (r.get("test_f1", 0), r.get("val_bacc", r.get("val_f1", 0))),
                 reverse=True)

    total = len(records)
    show  = min(args.top, total)

    sep    = "=" * 152
    header = (f"{'Rank':>4}  {'Config':>8}  {'ValBAcc':>7}  {'Val F1':>7}   "
              f"{'Test F1':>7}   {'Thresh':>7}  "
              f"{'DetStep':>9}  {'Caught':>10}  "
              f"{'bSc':>7}  {'hSc':>7}  Key params")
    lines  = [sep,
              f"  HTM-Distance-Stats Hyperparameter Search — {total} completed runs",
              sep, header, "-" * 152]

    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))

    best = records[0]
    lines += [
        sep,
        f"  Best config : ds_configs/config_{best.get('config_idx', 0):04d}.json",
        f"  Best model  : {best.get('model_file', '(see ds_models/)')}",
        f"  Best val BAcc: {best.get('val_bacc', float('nan')):.4f}",
        f"  Best val F1 : {best.get('val_f1', 0):.4f}",
        f"  Best test F1: {best.get('test_f1', 0):.4f}",
        f"  Mean det window: {best.get('mean_bot_detect_step', -1):.1f}  "
        f"({best.get('n_bots_caught','?')}/{best.get('n_bots_total','?')} bots caught)",
        f"  Mean bot score  (post-warmup): {best.get('mean_bot_score', 0):.4f}",
        f"  Mean human score (post-warmup): {best.get('mean_human_score', 0):.4f}",
        sep,
    ]

    report = "\n".join(lines)
    print(report)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_path = os.path.join(RESULTS_DIR, "leaderboard.txt")
    with open(txt_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nText report → {txt_path}")

    # ── CSV ──────────────────────────────────────────────────────────
    all_cfg_keys: list = []
    for r in records:
        for k in r.get("config", {}):
            if k not in all_cfg_keys:
                all_cfg_keys.append(k)

    csv_path = os.path.join(RESULTS_DIR, "leaderboard.csv")
    fieldnames = (
        ["rank", "config_idx", "val_bacc", "val_f1", "test_f1", "best_thresh",
         "mean_bot_detect_step", "n_bots_caught", "n_bots_total",
         "mean_bot_score", "mean_human_score"]
        + all_cfg_keys
        + ["result_file", "model_file"]
    )

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            row = {
                "rank":                 rank,
                "config_idx":           r.get("config_idx", ""),
                "val_bacc":             r.get("val_bacc", ""),
                "val_f1":               r.get("val_f1", ""),
                "test_f1":              r.get("test_f1", ""),
                "best_thresh":          r.get("best_thresh", ""),
                "mean_bot_detect_step": r.get("mean_bot_detect_step", ""),
                "n_bots_caught":        r.get("n_bots_caught", ""),
                "n_bots_total":         r.get("n_bots_total", ""),
                "mean_bot_score":       r.get("mean_bot_score", ""),
                "mean_human_score":     r.get("mean_human_score", ""),
                "result_file":          r.get("result_file", ""),
                "model_file":           r.get("model_file", ""),
            }
            row.update(r.get("config", {}))
            writer.writerow(row)

    print(f"CSV report  → {csv_path}")


if __name__ == "__main__":
    main()
