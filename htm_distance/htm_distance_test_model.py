"""
htm_distance/htm_distance_test_model.py
Test a trained HTM-Distance model on multiple evaluation modes.

Modes:
  orig            — original val/test splits from split.pkl
  all_non_train   — all cached files not in the training set (default)
  all_other_files — raw .txt files outside the original splits + bot files

Usage (from project root):
  python htm_distance/htm_distance_test_model.py \\
      --model dist_models/<slug>.pkl [--mode all_non_train]

Key difference from htm/htm_test_model.py:
  No reference pool or feature windows — each keystroke is one HTM step.
  The all_other_files mode runs parse_file_distance on-the-fly (no extra cache).
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_dist_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_dist_dir not in sys.path:
    sys.path.insert(0, _htm_dist_dir)

os.chdir(_project_root)

import argparse
import multiprocessing as mp
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_distance_common import parse_file_distance, apply_detection

SPLIT_FILE = "split.pkl"
CACHE_FILE = "dist_cache.pkl"
PLOTS_DIR  = "dist_plots"
LOGS_DIR   = "logs"
for _d in (PLOTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Inference ─────────────────────────────────────────────────────
def run_inference(model_data, dist_cache, file_list, is_bot,
                  desc="Inference", warmup=50):
    """
    Run HTM-Distance inference on a list of files.
    Uses AnomalyLikelihood (if saved in model) to convert raw TM anomaly →
    likelihood score relative to the human distribution seen during training.
    Returns (mean_scores, labels, likelihood_seqs).
    """
    sp          = model_data["sp"]
    tm          = model_data["tm"]
    encoder     = model_data["encoder"]
    al          = model_data.get("al")        # may be None for old models
    input_width = model_data["input_width"]
    active_cols = SDR(sp.getColumnDimensions())

    scores, labels, seqs = [], [], []
    print(f"  {desc}: {len(file_list)} files")

    for fp in file_list:
        events = dist_cache.get(fp)
        if not events:
            continue
        tm.reset()
        raw = []
        for key_idx, dwell, flight, dist in events:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, False, active_cols)
            tm.compute(active_cols, learn=False)
            raw_anomaly = float(tm.anomaly)
            score = (al.compute(raw_anomaly)
                     if al is not None else raw_anomaly)
            raw.append(float(score))

        label = 1 if is_bot else 0
        valid = raw[warmup:]
        # Always include — short files are always misclassified by apply_detection
        scores.append(float(np.mean(valid)) if valid else 0.0)
        labels.append(label)
        seqs.append(raw)

    return scores, labels, seqs


# ── Decision log (per-keystroke) ──────────────────────────────────
def write_decision_log(filepath, events, model_data, true_label,
                       output_path, warmup=50):
    sp          = model_data["sp"]
    tm          = model_data["tm"]
    encoder     = model_data["encoder"]
    al          = model_data.get("al")
    input_width = model_data["input_width"]
    thresh      = model_data["best_thresh"]
    active_cols = SDR(sp.getColumnDimensions())

    tm.reset()
    rows             = []
    bot_triggered_at = None

    for step, (key_idx, dwell, flight, dist) in enumerate(events):
        enc_sdr       = SDR(input_width)
        enc_sdr.dense = encoder.encode(key_idx, dwell, flight, dist)
        sp.compute(enc_sdr, False, active_cols)
        tm.compute(active_cols, learn=False)
        raw_anomaly = float(tm.anomaly)
        score = (al.compute(raw_anomaly)
                 if al is not None else raw_anomaly)

        ch = chr(key_idx + 32)

        if step < warmup:
            status = "WARMUP"
            reason = f"warmup ({step + 1}/{warmup})"
        elif bot_triggered_at is not None:
            status = "BOT"
            reason = f"already triggered at step {bot_triggered_at}"
        elif score >= thresh:
            bot_triggered_at = step
            status = "BOT"
            reason = f"first crossing: {score:.4f} >= {thresh:.4f}"
        else:
            status = "no crossing"
            reason = f"score {score:.4f} < {thresh:.4f}"

        rows.append((step, ch, dwell, flight, dist, score, status, reason))

    short_file = len(events) <= warmup
    if short_file:
        final_pred = 1 - true_label   # always misclassified
    else:
        final_pred = 1 if bot_triggered_at is not None else 0
    verdict    = "CORRECT" if final_pred == true_label else "WRONG"

    with open(output_path, "w") as fh:
        fh.write(f"# File:           {filepath}\n")
        fh.write(f"# True label:     {'BOT' if true_label else 'HUMAN'}\n")
        if short_file:
            fh.write(f"# SHORT FILE:     {len(events)} keystrokes < warmup={warmup}"
                     f" — auto-misclassified\n")
        fh.write(f"# Final decision: {'BOT' if final_pred else 'HUMAN'}  [{verdict}]\n")
        fh.write(f"# Threshold:      {thresh:.4f}\n")
        fh.write(f"# Warmup steps:   {warmup}\n")
        fh.write(f"# Total steps:    {len(events)}\n#\n")
        fh.write(f"{'Step':>6}  {'Key':>3}  {'Dwell':>7}  {'Flight':>7}  "
                 f"{'Dist':>6}  {'Score':>7}  {'Status':>14}  Reason\n")
        fh.write(f"{'-'*6}  {'-'*3}  {'-'*7}  {'-'*7}  "
                 f"{'-'*6}  {'-'*7}  {'-'*14}  {'-'*50}\n")
        for s, ch, dw, fl, di, sc, st, rs in rows:
            fh.write(f"{s:>6}  {ch!r:>3}  {dw:>7.1f}  {fl:>7.1f}  "
                     f"{di:>6.3f}  {sc:>7.4f}  {st:>14}  {rs}\n")

    print(f"  Decision log -> {output_path}")


# ── Plots ──────────────────────────────────────────────────────────
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
    ax.set_xlabel("Keystroke index"); ax.set_ylabel("Anomaly score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(True)

    ax = axes[1]
    ax.hist(h_scores, bins=15, alpha=0.65, color="steelblue", label="Human")
    ax.hist(b_scores, bins=15, alpha=0.65, color="tomato",    label="Bot")
    ax.axvline(thresh, color="green", linestyle="--", linewidth=2,
               label=f"Thresh={thresh:.3f}")
    ax.set_title("Per-File Mean Anomaly Score")
    ax.set_xlabel("Mean anomaly score"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True)

    ax = axes[2]
    ths = np.linspace(0, 1, 200)
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
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True)

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
    plt.xlabel("Predicted"); plt.ylabel("Actual")
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
                                 target_names=[["Human","Bot"][i] for i in unique],
                                 zero_division=0))
    return f1, preds


# ── Parse raw files in parallel (all_other_files mode) ───────────
def _parse_worker(fp):
    return fp, parse_file_distance(fp)


def _list_txt(root):
    paths = []
    for r, _, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith(".txt"):
                paths.append(os.path.abspath(os.path.join(r, fname)))
    return paths


# ── Main ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test a trained HTM-Distance model")
    parser.add_argument("--model", required=True,
                        help="Path to dist_models/<slug>.pkl")
    parser.add_argument("--split",   default=SPLIT_FILE)
    parser.add_argument("--cache",   default=CACHE_FILE)
    parser.add_argument("--mode",
                        choices=["orig", "all_non_train", "all_other_files"],
                        default="all_non_train",
                        help=(
                            "orig           — original val/test splits\n"
                            "all_non_train  — all cached files not in training set\n"
                            "all_other_files— raw .txt files outside original splits"
                        ))
    parser.add_argument("--data_root",
                        default="../UB_keystroke_dataset/",
                        help="Root for 'other' human .txt files (all_other_files mode)")
    parser.add_argument("--bots_root",
                        default="../BadUSBdataset",
                        help="Root for bot .txt files (all_other_files mode)")
    parser.add_argument("--write_decision_log", action="store_true",
                        help="Write per-keystroke decision log for one human + one bot")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}"); sys.exit(1)
    with open(args.model, "rb") as fh:
        md = pickle.load(fh)

    thresh = md["best_thresh"]
    warmup = md.get("warmup_steps", 50)
    slug   = os.path.splitext(os.path.basename(args.model))[0]
    print(f"\nModel : {args.model}")
    print(f"Thresh: {thresh:.4f}   Warmup: {warmup}   Mode: {args.mode}\n")

    # ── Load split ────────────────────────────────────────────────
    if not os.path.exists(args.split):
        print(f"ERROR: {args.split} not found. Run htm/htm_prepare_data.py"); sys.exit(1)
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
            print(f"ERROR: {args.cache} not found."); sys.exit(1)
        with open(args.cache, "rb") as fh:
            cache = pickle.load(fh)

        print("--- Validation ---")
        vh_sc, vh_lb, vh_sq = run_inference(md, cache, val_human, False, "Val human",  warmup)
        vb_sc, vb_lb, vb_sq = run_inference(md, cache, val_bots,  True,  "Val bot",    warmup)
        all_v_sc = vh_sc + vb_sc
        all_v_lb = vh_lb + vb_lb
        all_v_sq = vh_sq + vb_sq

        print("--- Test ---")
        th_sc, th_lb, th_sq = run_inference(md, cache, test_human, False, "Test human", warmup)
        tb_sc, tb_lb, tb_sq = run_inference(md, cache, test_bots,  True,  "Test bot",   warmup)
        all_t_sc = th_sc + tb_sc
        all_t_lb = th_lb + tb_lb
        all_t_sq = th_sq + tb_sq

        val_f1, val_preds  = report(all_v_lb, all_v_sq, thresh, warmup, "Validation")
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
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), warmup)
            if lb:
                write_decision_log(lb, cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), warmup)

    # ── MODE: all_non_train ───────────────────────────────────────
    elif args.mode == "all_non_train":
        if not os.path.exists(args.cache):
            print(f"ERROR: {args.cache} not found."); sys.exit(1)
        with open(args.cache, "rb") as fh:
            cache = pickle.load(fh)

        train_set   = set(train_human)
        known_human = set(val_human) | set(test_human)
        known_bot   = set(val_bots)  | set(test_bots)

        nt_human = [f for f in cache if f in known_human and f not in train_set]
        nt_bot   = [f for f in cache if f in known_bot]
        print(f"Non-train human: {len(nt_human)}   Non-train bot: {len(nt_bot)}")

        h_sc, h_lb, h_sq = run_inference(md, cache, nt_human, False, "Human", warmup)
        b_sc, b_lb, b_sq = run_inference(md, cache, nt_bot,   True,  "Bot",   warmup)
        all_sc = h_sc + b_sc
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
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), warmup)
            if lb:
                write_decision_log(lb, cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), warmup)

    # ── MODE: all_other_files ─────────────────────────────────────
    elif args.mode == "all_other_files":
        known_names = set(
            os.path.basename(f)
            for f in train_human + val_human + test_human + val_bots + test_bots
        )

        # Collect candidate files
        print(f"Walking human root: {os.path.abspath(args.data_root)}")
        all_human_cands = [
            p for p in _list_txt(args.data_root)
            if os.path.basename(p) not in known_names
        ]
        print(f"Walking bot root:   {os.path.abspath(args.bots_root)}")
        all_bot_cands = _list_txt(args.bots_root)
        print(f"Human candidates: {len(all_human_cands)}"
              f"   Bot candidates: {len(all_bot_cands)}")

        # Parse in parallel (no reference pool needed — parse_file_distance is self-contained)
        all_cands = [(p, False) for p in all_human_cands] + \
                    [(p, True)  for p in all_bot_cands]
        fps    = [p for p, _ in all_cands]
        labels_map = {p: int(is_bot) for p, is_bot in all_cands}

        print(f"\nParsing {len(fps)} files (parallel)...")
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        with mp.Pool(processes=workers) as pool:
            parsed = dict(tqdm(pool.imap_unordered(_parse_worker, fps),
                               total=len(fps), desc="Parsing"))

        other_cache = {fp: evs for fp, evs in parsed.items() if evs}
        print(f"Files with events: {len(other_cache)}")

        h_fps = [p for p in other_cache if labels_map[p] == 0]
        b_fps = [p for p in other_cache if labels_map[p] == 1]

        h_sc, h_lb, h_sq = run_inference(md, other_cache, h_fps, False, "Human", warmup)
        b_sc, b_lb, b_sq = run_inference(md, other_cache, b_fps, True,  "Bot",   warmup)
        all_sc = h_sc + b_sc
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
                    os.path.join(LOGS_DIR, f"{slug}_human_log.txt"), warmup)
            if lb:
                write_decision_log(lb, other_cache[lb], md, 1,
                    os.path.join(LOGS_DIR, f"{slug}_bot_log.txt"), warmup)


if __name__ == "__main__":
    mp.freeze_support()
    main()
