"""
htm_distance_stats/htm_distance_stats_test_model.py
Test a trained HTM-Distance-Stats model on multiple evaluation modes.

Modes:
  orig            — original val/test splits from split.pkl
  all_non_train   — all cached files not in the training set (default)
  all_other_files — raw .txt files outside the original splits + bot files
                    (parsed on-the-fly; no pre-computed cache needed)

Usage (from project root):
  python htm_distance_stats/htm_distance_stats_test_model.py \\
      --model ds_models/<slug>.pkl [--mode all_non_train]

The reference pool (ref_dwells, ref_flights, ref_dists) is loaded from the
model pkl and used to compute features for any new files in all_other_files mode.
"""

import multiprocessing as mp
import os
import pickle
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_dist_stats_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_dist_stats_dir not in sys.path:
    sys.path.insert(0, _htm_dist_stats_dir)

os.chdir(_project_root)

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_distance_stats_common import (
    get_file_feature_seq, apply_detection, parse_file_distance,
)

SPLIT_FILE = "split.pkl"
CACHE_FILE = "stats_cache.pkl"
PLOTS_DIR  = "ds_plots"
LOGS_DIR   = "logs"

for _d in (PLOTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model_data, stats_cache, file_list, is_bot,
                  desc="Inference"):
    """
    Run HTM-Distance-Stats inference on a list of files.

    Extracts feature sequences from stats_cache using the model's window_size,
    window_step, and reference pool; then runs SP+TM+AL inference.

    Returns (mean_scores, labels, score_seqs).
    """
    sp          = model_data["sp"]
    tm          = model_data["tm"]
    encoder     = model_data["encoder"]
    al          = model_data.get("al")
    input_width = model_data["input_width"]
    warmup      = model_data.get("warmup_steps", 2)
    window_size = model_data.get("window_size", 5)
    window_step = model_data.get("window_step", 1)
    ref_dwells  = model_data["ref_dwells"]
    ref_flights = model_data["ref_flights"]
    ref_dists   = model_data["ref_dists"]

    active_cols = SDR(sp.getColumnDimensions())

    scores, labels, seqs = [], [], []
    print(f"  {desc}: {len(file_list)} files")

    for fp in file_list:
        events = stats_cache.get(fp)
        if not events:
            continue
        feat_seq = get_file_feature_seq(events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists)
        tm.reset()
        raw = []
        for feat in feat_seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(feat)
            sp.compute(enc_sdr, False, active_cols)
            tm.compute(active_cols, learn=False)
            raw_anomaly = float(tm.anomaly)
            score = (al.compute(raw_anomaly) if al is not None else raw_anomaly)
            raw.append(float(score))

        label = 1 if is_bot else 0
        valid = raw[warmup:]
        scores.append(float(np.mean(valid)) if valid else 0.0)
        labels.append(label)
        seqs.append(raw)

    return scores, labels, seqs


# ── Decision log (per-window) ─────────────────────────────────────────────────
def write_decision_log(filepath, events, model_data, true_label,
                       output_path, thresh=None):
    sp          = model_data["sp"]
    tm          = model_data["tm"]
    encoder     = model_data["encoder"]
    al          = model_data.get("al")
    input_width = model_data["input_width"]
    warmup      = model_data.get("warmup_steps", 2)
    window_size = model_data.get("window_size", 5)
    window_step = model_data.get("window_step", 1)
    ref_dwells  = model_data["ref_dwells"]
    ref_flights = model_data["ref_flights"]
    ref_dists   = model_data["ref_dists"]
    if thresh is None:
        thresh = model_data["best_thresh"]

    feat_seq = get_file_feature_seq(events, window_size, window_step,
                                    ref_dwells, ref_flights, ref_dists)

    active_cols      = SDR(sp.getColumnDimensions())
    tm.reset()
    rows             = []
    bot_triggered_at = None

    for step, feat in enumerate(feat_seq):
        enc_sdr       = SDR(input_width)
        enc_sdr.dense = encoder.encode(feat)
        sp.compute(enc_sdr, False, active_cols)
        tm.compute(active_cols, learn=False)
        raw_anomaly = float(tm.anomaly)
        score = (al.compute(raw_anomaly) if al is not None else raw_anomaly)

        if step < warmup:
            status = "WARMUP"
            reason = f"warmup ({step + 1}/{warmup})"
        elif bot_triggered_at is not None:
            status = "BOT"
            reason = f"already triggered at window {bot_triggered_at}"
        elif score >= thresh:
            bot_triggered_at = step
            status = "BOT"
            reason = f"first crossing: {score:.4f} >= {thresh:.4f}"
        else:
            status = "no crossing"
            reason = f"score {score:.4f} < {thresh:.4f}"

        rows.append((step, score, status, reason))

    short_file = len(feat_seq) <= warmup
    if short_file:
        final_pred = 1 - true_label
    else:
        final_pred = 1 if bot_triggered_at is not None else 0
    verdict = "CORRECT" if final_pred == true_label else "WRONG"

    with open(output_path, "w") as fh:
        fh.write(f"# File:           {filepath}\n")
        fh.write(f"# True label:     {'BOT' if true_label else 'HUMAN'}\n")
        fh.write(f"# Window size:    {window_size} keystrokes  step={window_step}\n")
        fh.write(f"# Total windows:  {len(feat_seq)}\n")
        if short_file:
            fh.write(f"# SHORT FILE:     {len(feat_seq)} windows <= warmup={warmup}"
                     " — auto-misclassified\n")
        fh.write(f"# Final decision: {'BOT' if final_pred else 'HUMAN'}  [{verdict}]\n")
        fh.write(f"# Threshold:      {thresh:.4f}\n")
        fh.write(f"# Warmup windows: {warmup}\n#\n")
        fh.write(f"{'Window':>7}  {'Score':>7}  {'Status':>14}  Reason\n")
        fh.write(f"{'-'*7}  {'-'*7}  {'-'*14}  {'-'*50}\n")
        for step, sc, st, rs in rows:
            fh.write(f"{step:>7}  {sc:>7.4f}  {st:>14}  {rs}\n")

    print(f"  Decision log -> {output_path}")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_results(h_seqs, b_seqs, h_scores, b_scores,
                 all_seqs, all_labels, thresh, f1_val,
                 fname_slug, title, warmup):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=7)

    ax = axes[0]
    n = min(3, len(h_seqs), len(b_seqs))
    for i in range(n):
        ax.plot(h_seqs[i], alpha=0.6, color="steelblue",
                label="Human" if i == 0 else "_nolegend_")
    for i in range(n):
        ax.plot(b_seqs[i], alpha=0.6, color="tomato",
                label="Bot" if i == 0 else "_nolegend_")
    ax.axhline(thresh, color="green", linestyle="--", linewidth=1.5,
               label=f"Thresh={thresh:.3f}")
    ax.axvline(warmup, color="gray", linestyle=":", linewidth=1,
               label=f"Warmup={warmup}")
    ax.set_title("Anomaly Score Over Time\n(sample files)")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Anomaly score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True)

    ax = axes[1]
    ax.hist(h_scores, bins=15, alpha=0.65, color="steelblue", label="Human")
    ax.hist(b_scores, bins=15, alpha=0.65, color="tomato",    label="Bot")
    ax.axvline(thresh, color="green", linestyle="--", linewidth=2,
               label=f"Thresh={thresh:.3f}")
    ax.set_title("Per-File Mean Anomaly Score")
    ax.set_xlabel("Mean anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ths = np.linspace(0, 1, 500)
    f1s = [
        f1_score(all_labels,
                 apply_detection(all_seqs, "first_crossing", t, warmup,
                                 labels=all_labels),
                 zero_division=0)
        for t in ths
    ]
    ax.plot(ths, f1s, color="blue", linewidth=2, label="F1")
    ax.axvline(thresh, color="red", linestyle="--", linewidth=2,
               label=f"Best={thresh:.3f} (F1={f1_val:.4f})")
    ax.set_title("F1 vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_results.png")
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot  -> {fpath}")


def plot_confusion(all_labels, all_preds, subset_name, fname_slug):
    unique = sorted(set(all_labels))
    names  = ["Human", "Bot"]
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=unique)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{subset_name} (F1={f1:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([i + 0.5 for i in range(len(unique))], [names[i] for i in unique])
    plt.yticks([i + 0.5 for i in range(len(unique))], [names[i] for i in unique])
    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_confusion.png")
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot  -> {fpath}")


def report(all_labels, all_seqs, thresh, warmup, subset_name):
    preds  = apply_detection(all_seqs, "first_crossing", thresh, warmup,
                             labels=all_labels)
    unique = sorted(set(all_labels))
    f1     = f1_score(all_labels, preds, labels=unique, zero_division=0)
    print(f"\n{'='*45}")
    print(f"  {subset_name}  F1={f1:.4f}  (thresh={thresh:.4f})")
    print(f"{'='*45}")
    print(classification_report(all_labels, preds,
                                 labels=unique,
                                 target_names=[["Human", "Bot"][i] for i in unique],
                                 zero_division=0))
    return f1, preds


# ── Parse raw files in parallel (all_other_files mode) ────────────────────────
def _parse_worker(fp):
    return fp, parse_file_distance(fp)


def _list_txt(root):
    paths = []
    for r, _, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith(".txt"):
                paths.append(os.path.abspath(os.path.join(r, fname)))
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test a trained HTM-Distance-Stats model")
    parser.add_argument("--model", required=True,
                        help="Path to ds_models/<slug>.pkl")
    parser.add_argument("--split",  default=SPLIT_FILE)
    parser.add_argument("--cache",  default=CACHE_FILE)
    parser.add_argument("--mode",
                        choices=["orig", "all_non_train", "all_other_files"],
                        default="all_non_train")
    parser.add_argument("--data_root", default="../UB_keystroke_dataset/",
                        help="Human .txt root for all_other_files mode")
    parser.add_argument("--bots_root", default="../BadUSBdataset",
                        help="Bot .txt root for all_other_files mode")
    parser.add_argument("--write_decision_log", action="store_true",
                        help="Write per-window decision log for one human + one bot file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override saved threshold")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)
    with open(args.model, "rb") as fh:
        md = pickle.load(fh)

    thresh      = args.threshold if args.threshold is not None else md["best_thresh"]
    warmup      = md.get("warmup_steps", 2)
    window_size = md.get("window_size", 5)
    window_step = md.get("window_step", 1)
    slug        = os.path.splitext(os.path.basename(args.model))[0]

    print(f"\nModel      : {args.model}")
    if args.threshold is not None:
        print(f"Thresh     : {thresh:.4f}  "
              f"[MANUAL OVERRIDE — model default was {md['best_thresh']:.4f}]")
    else:
        print(f"Thresh     : {thresh:.4f}")
    print(f"Warmup     : {warmup} windows   Window: {window_size}×{window_step}")
    print(f"Mode       : {args.mode}\n")

    # ── Load split ────────────────────────────────────────────────
    if not os.path.exists(args.split):
        print(f"ERROR: {args.split} not found.  Run htm/htm_prepare_data.py.")
        sys.exit(1)
    with open(args.split, "rb") as fh:
        split = pickle.load(fh)
    train_human = split["train_human"]
    val_human   = split["val_human"]
    test_human  = split["test_human"]
    val_bots    = split["val_bots"]
    test_bots   = split["test_bots"]

    # ── MODE: orig ────────────────────────────────────────────────
    if args.mode == "orig":
        if not os.path.exists(args.cache):
            print(f"ERROR: {args.cache} not found.")
            sys.exit(1)
        with open(args.cache, "rb") as fh:
            cache = pickle.load(fh)

        vh_sc, vh_lb, vh_sq = run_inference(md, cache, val_human,  False, "Val human")
        vb_sc, vb_lb, vb_sq = run_inference(md, cache, val_bots,   True,  "Val bot")
        th_sc, th_lb, th_sq = run_inference(md, cache, test_human, False, "Test human")
        tb_sc, tb_lb, tb_sq = run_inference(md, cache, test_bots,  True,  "Test bot")

        all_v_lb = vh_lb + vb_lb
        all_v_sq = vh_sq + vb_sq
        all_t_lb = th_lb + tb_lb
        all_t_sq = th_sq + tb_sq

        val_f1,  val_preds  = report(all_v_lb, all_v_sq, thresh, warmup, "Validation")
        test_f1, test_preds = report(all_t_lb, all_t_sq, thresh, warmup, "Test")

        plot_results(vh_sq, vb_sq, vh_sc, vb_sc,
                     all_v_sq, all_v_lb, thresh, val_f1,
                     f"{slug}_orig", f"{slug} (orig)", warmup)
        plot_confusion(all_v_lb, val_preds,  "Validation", f"{slug}_orig_val")
        plot_confusion(all_t_lb, test_preds, "Test",       f"{slug}_orig_test")

        if args.write_decision_log:
            lh = next((f for f in val_human if cache.get(f)), None)
            lb = next((f for f in val_bots  if cache.get(f)), None)
            if lh:
                write_decision_log(lh, cache[lh], md, 0,
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), thresh)
            if lb:
                write_decision_log(lb, cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), thresh)

    # ── MODE: all_non_train ───────────────────────────────────────
    elif args.mode == "all_non_train":
        if not os.path.exists(args.cache):
            print(f"ERROR: {args.cache} not found.")
            sys.exit(1)
        with open(args.cache, "rb") as fh:
            cache = pickle.load(fh)

        train_set   = set(train_human)
        known_human = set(val_human) | set(test_human)
        known_bot   = set(val_bots)  | set(test_bots)

        nt_human = [f for f in cache if f in known_human and f not in train_set]
        nt_bot   = [f for f in cache if f in known_bot]
        print(f"Non-train human: {len(nt_human)}   Non-train bot: {len(nt_bot)}")

        h_sc, h_lb, h_sq = run_inference(md, cache, nt_human, False, "Human")
        b_sc, b_lb, b_sq = run_inference(md, cache, nt_bot,   True,  "Bot")
        all_lb = h_lb + b_lb
        all_sq = h_sq + b_sq

        f1, preds = report(all_lb, all_sq, thresh, warmup, "All Non-Train")
        plot_results(h_sq, b_sq, h_sc, b_sc, all_sq, all_lb,
                     thresh, f1, f"{slug}_non_train",
                     f"{slug} (all_non_train)", warmup)
        plot_confusion(all_lb, preds, "All Non-Train", f"{slug}_non_train")

        if args.write_decision_log:
            lh = next((f for f in nt_human if cache.get(f)), None)
            lb = next((f for f in nt_bot   if cache.get(f)), None)
            if lh:
                write_decision_log(lh, cache[lh], md, 0,
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), thresh)
            if lb:
                write_decision_log(lb, cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), thresh)

    # ── MODE: all_other_files ─────────────────────────────────────
    elif args.mode == "all_other_files":
        known_names = set(
            os.path.basename(f)
            for f in train_human + val_human + test_human + val_bots + test_bots
        )

        print(f"Walking human root: {os.path.abspath(args.data_root)}")
        human_cands = [p for p in _list_txt(args.data_root)
                       if os.path.basename(p) not in known_names]
        print(f"Walking bot root:   {os.path.abspath(args.bots_root)}")
        bot_cands   = _list_txt(args.bots_root)
        print(f"Human candidates: {len(human_cands)}   "
              f"Bot candidates: {len(bot_cands)}")

        all_fps    = human_cands + bot_cands
        labels_map = {p: 0 for p in human_cands}
        labels_map.update({p: 1 for p in bot_cands})

        print(f"\nParsing {len(all_fps)} files (parallel)...")
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        with mp.Pool(processes=workers) as pool:
            parsed = dict(tqdm(pool.imap_unordered(_parse_worker, all_fps),
                               total=len(all_fps), desc="Parsing"))

        other_cache = {fp: evs for fp, evs in parsed.items() if evs}
        print(f"Files with events: {len(other_cache)}")

        h_fps = [p for p in other_cache if labels_map[p] == 0]
        b_fps = [p for p in other_cache if labels_map[p] == 1]

        h_sc, h_lb, h_sq = run_inference(md, other_cache, h_fps, False, "Human")
        b_sc, b_lb, b_sq = run_inference(md, other_cache, b_fps, True,  "Bot")
        all_lb = h_lb + b_lb
        all_sq = h_sq + b_sq

        f1, preds = report(all_lb, all_sq, thresh, warmup, "All Other Files")
        plot_results(h_sq, b_sq, h_sc, b_sc, all_sq, all_lb,
                     thresh, f1, f"{slug}_other",
                     f"{slug} (all_other_files)", warmup)
        plot_confusion(all_lb, preds, "All Other Files", f"{slug}_other")

        if args.write_decision_log:
            lh = next((f for f in h_fps if other_cache.get(f)), None)
            lb = next((f for f in b_fps if other_cache.get(f)), None)
            if lh:
                write_decision_log(lh, other_cache[lh], md, 0,
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), thresh)
            if lb:
                write_decision_log(lb, other_cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), thresh)


if __name__ == "__main__":
    mp.freeze_support()
    main()
