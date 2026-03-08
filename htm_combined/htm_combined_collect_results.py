"""
htm_combined/htm_combined_collect_results.py
Aggregate all completed HTM-Combined runs into a leaderboard.

Usage (from project root):
  python htm_combined/htm_combined_collect_results.py [--top 20]

Reads:   hc_results/hc*.json
Writes:  hc_results/leaderboard.txt
         hc_results/leaderboard.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

RESULTS_DIR = "hc_results"


def load_results() -> list:
    records = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "hc*.json"))):
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
    if "dwell_stats_enc_bits" in cfg:
        # Round 3: fully per-channel / per-scalar
        enc_str = (
            f"dS={cfg.get('dwell_stats_enc_bits','?')}w{cfg.get('dwell_stats_enc_w','?')} "
            f"fS={cfg.get('flight_stats_enc_bits','?')}w{cfg.get('flight_stats_enc_w','?')} "
            f"qS={cfg.get('dist_stats_enc_bits','?')}w{cfg.get('dist_stats_enc_w','?')} "
            f"dK={cfg.get('dwell_scalar_enc_bits','?')}w{cfg.get('dwell_scalar_enc_w','?')} "
            f"fK={cfg.get('flight_scalar_enc_bits','?')}w{cfg.get('flight_scalar_enc_w','?')} "
            f"qK={cfg.get('dist_scalar_enc_bits','?')}w{cfg.get('dist_scalar_enc_w','?')}"
        )
    elif "stats_enc_bits" in cfg:
        enc_str = (f"se={cfg.get('stats_enc_bits','?')}w{cfg.get('stats_enc_w','?')} "
                   f"ke={cfg.get('scalar_enc_bits','?')}w{cfg.get('scalar_enc_w','?')}")
    else:
        enc_str = f"enc={cfg.get('enc_bits_per_feature','?')}w{cfg.get('enc_w','?')}"
    key = (
        f"sp_act={cfg.get('sp_numActiveColumns','?'):>2} "
        f"pct={cfg.get('sp_potentialPct','?')} "
        f"{enc_str} "
        f"ws={cfg.get('window_size','?')}s{cfg.get('window_step','?')} "
        f"tm={cfg.get('tm_cellsPerColumn','?')} "
        f"act={cfg.get('tm_activationThreshold','?')} "
        f"min={cfg.get('tm_minThreshold','?')} "
        f"wu={cfg.get('warmup_steps','?')} "
        f"al={cfg.get('al_period','?')}"
    )
    val_bacc = r.get("val_bacc", float("nan"))
    bacc_str = f"{val_bacc:.4f}" if val_bacc == val_bacc else "  N/A "
    live_thresh = r.get("live_thresh", None)
    lt_str = f"{live_thresh:.4f}" if live_thresh is not None and live_thresh > 0 else "  ---  "
    return (
        f"{rank:>4}  "
        f"hc{r.get('config_idx', 0):04d}  "
        f"{bacc_str}  "
        f"{r.get('val_f1', 0):.4f}   "
        f"{r.get('test_f1', 0):.4f}   "
        f"{r.get('best_thresh', 0):.4f}  "
        f"lt={lt_str}  "
        f"det={det_str}  "
        f"caught={caught}/{total}  "
        f"bSc={b_sc:.3f}  hSc={h_sc:.3f}  "
        f"{key}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect HTM-Combined hyperparameter search results")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to display (default: 20)")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(f"No results found in {RESULTS_DIR}/. "
              "Run htm_combined_train_single.py jobs first.")
        sys.exit(0)

    # Sort: test F1 first, then live_thresh>0 (deployable models first),
    # then val BAcc, then config_idx (newer = higher = better)
    records.sort(
        key=lambda r: (r.get("test_f1", 0),
                       1 if r.get("live_thresh", 0) > 0 else 0,
                       r.get("val_bacc", r.get("val_f1", 0)),
                       r.get("config_idx", 0)),
        reverse=True)

    total = len(records)
    show  = min(args.top, total)

    sep    = "=" * 165
    header = (f"{'Rank':>4}  {'Config':>8}  {'ValBAcc':>7}  {'Val F1':>7}   "
              f"{'Test F1':>7}   {'Thresh':>7}  "
              f"{'LiveThresh':>10}  "
              f"{'DetStep':>9}  {'Caught':>10}  "
              f"{'bSc':>7}  {'hSc':>7}  Key params")
    lines  = [sep,
              f"  HTM-Combined Hyperparameter Search -- {total} completed runs",
              sep, header, "-" * 165]

    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))

    best = records[0]
    best_lt = best.get('live_thresh', None)
    best_lt_str = f"{best_lt:.4f}" if best_lt is not None and best_lt > 0 else "N/A (model not deployable)"
    lines += [
        sep,
        f"  Best config : hc_configs/config_{best.get('config_idx', 0):04d}.json",
        f"  Best model  : {best.get('model_file', '(see hc_models/)')}",
        f"  Best val BAcc: {best.get('val_bacc', float('nan')):.4f}",
        f"  Best val F1 : {best.get('val_f1', 0):.4f}",
        f"  Best test F1: {best.get('test_f1', 0):.4f}",
        f"  Live thresh : {best_lt_str}",
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
    print(f"\nText report -> {txt_path}")

    # ── CSV ──────────────────────────────────────────────────────────
    all_cfg_keys: list = []
    for r in records:
        for k in r.get("config", {}):
            if k not in all_cfg_keys:
                all_cfg_keys.append(k)

    csv_path = os.path.join(RESULTS_DIR, "leaderboard.csv")
    fieldnames = (
        ["rank", "config_idx", "val_bacc", "val_f1", "test_f1",
         "best_thresh", "live_thresh",
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
                "live_thresh":          r.get("live_thresh", ""),
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

    print(f"CSV report  -> {csv_path}")


if __name__ == "__main__":
    main()
