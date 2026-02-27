"""
mlp/mlp_collect_results.py
Aggregate all completed MLP hyperparameter search runs into a leaderboard.

Usage:
  python mlp/mlp_collect_results.py [--top 20]

Reads:   mlp_results/mlp*.json
Writes:  mlp_results/mlp_leaderboard.csv
         mlp_results/mlp_leaderboard.txt
"""

import argparse
import csv
import glob
import json
import os

RESULTS_DIR = "mlp_results"


def load_results():
    records = []
    pattern = os.path.join(RESULTS_DIR, "mlp*.json")
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
    cfg  = r.get("config", {})
    dims = '-'.join(str(d) for d in cfg.get("hidden_dims", []))
    key = (
        f"arch=[{dims}] "
        f"act={cfg.get('activation','?'):>10} "
        f"opt={cfg.get('optimizer','?'):>5} "
        f"lr={cfg.get('lr', 0):.0e} "
        f"sch={cfg.get('scheduler','?'):>18} "
        f"drop={cfg.get('dropout', 0):.2f} "
        f"bn={int(cfg.get('batch_norm', False))} "
        f"bs={cfg.get('batch_size','?'):>3}"
    )
    return (
        f"{rank:>4}  "
        f"mlp{r.get('config_idx', 0):04d}  "
        f"{r.get('val_f1', 0):.4f}   "
        f"{r.get('test_f1', 0):.4f}   "
        f"{r.get('n_params', 0):>8,}  "
        f"{key}"
    )


def main():
    # Run from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Collect MLP hyperparameter search results")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of top configs to display (default: 20)")
    args = parser.parse_args()

    records = load_results()
    if not records:
        print(f"No results found in {RESULTS_DIR}/. "
              "Run mlp/mlp_train_single.py jobs first.")
        return

    # Sort by test F1 desc, break ties by val F1
    records.sort(key=lambda r: (r.get("test_f1", 0), r.get("val_f1", 0)),
                 reverse=True)

    total = len(records)
    show  = min(args.top, total)

    # ---- Console leaderboard ----
    sep    = "=" * 110
    header = (f"{'Rank':>4}  {'Config':>8}  {'Val F1':>7}   "
              f"{'Test F1':>7}   {'#Params':>8}  Key params")
    lines  = [sep,
              f"  MLP Hyperparameter Search — {total} completed runs",
              sep, header, "-" * 110]
    for rank, r in enumerate(records[:show], 1):
        lines.append(format_row(rank, r))
    lines += [sep,
              f"  Best config : mlp_configs/config_{records[0].get('config_idx', 0):04d}.json",
              f"  Best model  : {records[0].get('model_file', '(see mlp_models/)')}",
              sep]

    report = "\n".join(lines)
    print(report)

    # ---- Write leaderboard.txt ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    txt_path = os.path.join(RESULTS_DIR, "mlp_leaderboard.txt")
    with open(txt_path, 'w') as fh:
        fh.write(report + "\n")
    print(f"\nText report → {txt_path}")

    # ---- Write leaderboard.csv ----
    csv_path = os.path.join(RESULTS_DIR, "mlp_leaderboard.csv")
    fieldnames = ["rank", "config_idx", "val_f1", "test_f1", "n_params",
                  "elapsed_s", "hidden_dims", "activation", "optimizer", "lr",
                  "weight_decay", "scheduler", "batch_size", "dropout",
                  "batch_norm", "label_smoothing", "seed",
                  "result_file", "model_file"]

    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rank, r in enumerate(records, 1):
            cfg = r.get("config", {})
            row = {
                "rank":          rank,
                "config_idx":    r.get("config_idx", ""),
                "val_f1":        r.get("val_f1", ""),
                "test_f1":       r.get("test_f1", ""),
                "n_params":      r.get("n_params", ""),
                "elapsed_s":     r.get("elapsed_s", ""),
                "hidden_dims":   str(cfg.get("hidden_dims", "")),
                "activation":    cfg.get("activation", ""),
                "optimizer":     cfg.get("optimizer", ""),
                "lr":            cfg.get("lr", ""),
                "weight_decay":  cfg.get("weight_decay", ""),
                "scheduler":     cfg.get("scheduler", ""),
                "batch_size":    cfg.get("batch_size", ""),
                "dropout":       cfg.get("dropout", ""),
                "batch_norm":    cfg.get("batch_norm", ""),
                "label_smoothing": cfg.get("label_smoothing", ""),
                "seed":          cfg.get("seed", ""),
                "result_file":   r.get("result_file", ""),
                "model_file":    r.get("model_file", ""),
            }
            writer.writerow(row)

    print(f"CSV report  → {csv_path}")


if __name__ == "__main__":
    main()
