"""
htm/htm_test_model.py
Test a trained HTM model on multiple evaluation modes.

Modes:
  orig            — original val/test splits from split.pkl
  all_non_train   — all cached files not in the training set (default)
  all_other_files — raw .txt files outside the original splits + synthetic bots

Usage:
  python htm/htm_test_model.py --model models/<slug>.pkl [options]
"""

import os
import sys

# Allow imports from project root (common/, htm/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import pickle
import multiprocessing as mp
import warnings
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("HTM libraries not found. Please install htm.core.")
    sys.exit(1)

try:
    from htm.bindings.algorithms import AnomalyLikelihood
    _HAS_AL = True
except ImportError:
    _HAS_AL = False

from common.keystroke_features import (
    RANDOM_SEED, DEFAULT_WINDOW_SIZE, NUM_REFERENCES,
    parse_file, extract_features, create_reference_pool,
)
from htm_common import WARMUP_STEPS, apply_detection

# Match training step sizes
WINDOW_SIZE     = DEFAULT_WINDOW_SIZE
STEP_SIZE_HUMAN = 1
STEP_SIZE_BOT   = 1

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Feature computation for raw txt files (for all_other_files mode)
# ------------------------------------------------------------------
def compute_features_for_raw_file(filepath, ref_dwells, ref_flights, step_size):
    """
    Parse a raw keystroke .txt file and compute the feature sequence used
    during training (dwell + flight windows concatenated).
    Returns np.array of shape [num_windows, 14] or empty array.
    """
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))
    features_seq = []
    if min_len < WINDOW_SIZE:
        return np.array([])

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i: i + WINDOW_SIZE]
        w_f = f[i: i + WINDOW_SIZE]
        ft_d = extract_features(w_d, ref_dwells, WINDOW_SIZE)
        ft_f = extract_features(w_f, ref_flights, WINDOW_SIZE)
        if ft_d and ft_f:
            features_seq.append(ft_f + ft_d)

    return np.array(features_seq)


def _compute_features_wrapper(args):
    filepath, ref_dwells, ref_flights, step_size = args
    return filepath, compute_features_for_raw_file(filepath, ref_dwells, ref_flights, step_size)


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
def run_inference(model_data, file_features, file_list, is_bot, desc="Inference",
                  use_al=False, al_period=20, effective_warmup=WARMUP_STEPS):
    """
    Run HTM inference on a list of files using cached feature sequences.
    Returns (mean_scores, labels, raw_seqs):
      mean_scores — post-warmup mean per file (used for histogram plots)
      raw_seqs    — full per-window score sequences (used by apply_detection)
    If use_al=True, scores are converted through AnomalyLikelihood (reset per file).
    """
    sp          = model_data['sp']
    tm          = model_data['tm']
    encoder     = model_data['encoder']
    input_width = model_data['input_width']
    active_columns = SDR(sp.getColumnDimensions())

    scores, labels, seqs = [], [], []
    print(f"Running {desc} on {len(file_list)} files...")

    for filepath in file_list:
        if filepath not in file_features:
            continue
        tm.reset()
        al = AnomalyLikelihood(learningPeriod=al_period) if (use_al and _HAS_AL) else None
        file_scores = []
        for step, feats in enumerate(file_features[filepath]):
            dense_input = encoder.encode(feats)
            enc_sdr = SDR(input_width)
            enc_sdr.dense = dense_input
            sp.compute(enc_sdr, False, active_columns)
            tm.compute(active_columns, learn=False)
            score = tm.anomaly
            if al is not None:
                score = al.anomalyProbability(score, score, step)
            file_scores.append(score)

        if file_scores:
            valid = file_scores[effective_warmup:] if len(file_scores) > effective_warmup else file_scores
            scores.append(float(np.mean(valid)) if valid else 0.0)
            labels.append(1 if is_bot else 0)
            seqs.append(file_scores)

    return scores, labels, seqs


# ------------------------------------------------------------------
# Decision log
# ------------------------------------------------------------------
def write_decision_log(filepath, features_seq, model_data, true_label, output_path,
                       use_al=False, al_period=20, effective_warmup=WARMUP_STEPS):
    """
    Run inference on a single file and write a per-window decision log.
    Each line records: window index, anomaly score, status, and the reason
    behind the status (warmup, score vs. threshold, running mean, etc.).
    """
    sp             = model_data['sp']
    tm             = model_data['tm']
    encoder        = model_data['encoder']
    input_width    = model_data['input_width']
    best_thresh    = model_data.get('best_thresh', 0.5)
    detection_mode = model_data.get('detection_mode', 'mean')

    active_columns   = SDR(sp.getColumnDimensions())
    tm.reset()
    al = AnomalyLikelihood(learningPeriod=al_period) if (use_al and _HAS_AL) else None

    window_scores    = []
    rows             = []
    bot_triggered_at = None   # only used in first_crossing mode

    n_windows = len(features_seq)

    for step, feats in enumerate(features_seq):
        enc_sdr       = SDR(input_width)
        enc_sdr.dense = encoder.encode(feats)
        sp.compute(enc_sdr, False, active_columns)
        tm.compute(active_columns, learn=False)
        score = tm.anomaly
        if al is not None:
            score = al.anomalyProbability(score, score, step)

        window_scores.append(score)

        # ---- per-window status + reason ----
        if step < effective_warmup:
            status = "WARMUP"
            reason = f"warmup period ({step + 1}/{effective_warmup})"

        elif detection_mode == 'first_crossing':
            if bot_triggered_at is not None:
                status = "BOT"
                reason = f"already triggered at window {bot_triggered_at}"
            elif score >= best_thresh:
                bot_triggered_at = step
                status = "BOT"
                reason = f"first crossing: {score:.4f} >= threshold {best_thresh:.4f}"
            else:
                status = "no crossing"
                reason = f"score {score:.4f} < threshold {best_thresh:.4f}"

        else:  # mean mode
            post_so_far  = window_scores[effective_warmup:]
            running_mean = float(np.mean(post_so_far)) if post_so_far else 0.0
            is_last      = (step == n_windows - 1)
            cmp_str      = ">=" if running_mean >= best_thresh else "<"
            if is_last:
                if running_mean >= best_thresh:
                    status = "BOT"
                    reason = f"final mean {running_mean:.4f} >= threshold {best_thresh:.4f}"
                else:
                    status = "HUMAN"
                    reason = f"final mean {running_mean:.4f} < threshold {best_thresh:.4f}"
            else:
                status = "pending"
                reason = (f"running mean {running_mean:.4f} {cmp_str} "
                          f"threshold {best_thresh:.4f}")

        rows.append((step, score, status, reason))

    # ---- file-level verdict ----
    post = window_scores[effective_warmup:] if len(window_scores) > effective_warmup else window_scores
    if detection_mode == 'first_crossing':
        final_pred = 1 if bot_triggered_at is not None else 0
    else:
        final_pred = 1 if (post and float(np.mean(post)) >= best_thresh) else 0

    final_str  = "BOT"     if final_pred  == 1 else "HUMAN"
    true_str   = "BOT"     if true_label  == 1 else "HUMAN"
    verdict    = "CORRECT" if final_pred  == true_label else "WRONG"

    with open(output_path, 'w') as fh:
        fh.write(f"# File:           {filepath}\n")
        fh.write(f"# True label:     {true_str}\n")
        fh.write(f"# Final decision: {final_str}  [{verdict}]\n")
        fh.write(f"# Detection mode: {detection_mode}\n")
        fh.write(f"# Threshold:      {best_thresh:.4f}\n")
        fh.write(f"# Warmup steps:   {effective_warmup}\n")
        fh.write(f"# Total windows:  {n_windows}\n")
        fh.write("#\n")
        fh.write(f"{'Window':>8}  {'Score':>8}  {'Status':>14}  Reason\n")
        fh.write(f"{'-'*8}  {'-'*8}  {'-'*14}  {'-'*55}\n")
        for widx, sc, st, rs in rows:
            fh.write(f"{widx:>8}  {sc:>8.4f}  {st:>14}  {rs}\n")

    print(f"  Decision log → {output_path}")


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_results(val_h_seqs, val_b_seqs,
                 all_val_scores, all_val_labels,
                 all_val_seqs,
                 val_h_scores, val_b_scores,
                 best_thresh, val_f1,
                 fname_slug, title_prefix="",
                 detection_mode="mean", effective_warmup=WARMUP_STEPS):
    """
    3-panel plot:
      1) anomaly score over time for sample human/bot files
      2) per-file mean score distribution + threshold
      3) F1 vs threshold curve (using apply_detection for both modes)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title = f"{title_prefix} Val F1={val_f1:.4f}"
    fig.suptitle(title, fontsize=7)

    ax = axes[0]
    n = min(3, len(val_h_seqs), len(val_b_seqs))
    for i in range(n):
        ax.plot(val_h_seqs[i], alpha=0.7, color='steelblue',
                label='Human' if i == 0 else '_nolegend_')
    for i in range(n):
        ax.plot(val_b_seqs[i], alpha=0.7, color='tomato',
                label='Bot' if i == 0 else '_nolegend_')
    if detection_mode == 'first_crossing':
        ax.axhline(best_thresh, color='green', linestyle='--', linewidth=1.5,
                   label=f'Cross-thresh={best_thresh:.3f}', alpha=0.8)
    ax.set_title('Anomaly Score Over Time\n(sample files)')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.hist(val_h_scores, bins=15, alpha=0.65, color='steelblue', label='Human')
    ax.hist(val_b_scores, bins=15, alpha=0.65, color='tomato', label='Bot')
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2,
               label=f'Thresh={best_thresh:.3f}')
    ax.set_title('Per-File Mean Anomaly Score')
    ax.set_xlabel('Mean anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ths = np.linspace(0, 1, 200)
    f1s = [
        f1_score(all_val_labels,
                 apply_detection(all_val_seqs, detection_mode, t, effective_warmup),
                 zero_division=0)
        for t in ths
    ]
    ax.plot(ths, f1s, color='blue', linewidth=2, label='Val F1')
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Best={best_thresh:.3f} (F1={val_f1:.4f})')
    ax.set_title('F1 Score vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_results.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Plot  → {fpath}")


def eval_and_report(all_scores, all_labels, best_thresh, subset_name="Subset",
                    do_plot=False, plot_prefix=None,
                    all_seqs=None, detection_mode="mean",
                    effective_warmup=WARMUP_STEPS):
    """
    Compute metrics and print a classification report.
    Uses apply_detection when seqs are available; falls back to mean threshold otherwise.
    """
    if all_seqs is not None:
        preds = apply_detection(all_seqs, detection_mode, best_thresh, effective_warmup)
    else:
        preds = [1 if s >= best_thresh else 0 for s in all_scores]

    unique_labels = sorted(set(all_labels))
    label_names   = ['Human', 'Bot']
    present_target_names = [label_names[i] for i in unique_labels]

    f1 = f1_score(all_labels, preds, labels=unique_labels, zero_division=0)

    print("\n" + "=" * 40)
    print(f"{subset_name} RESULTS")
    print("=" * 40)
    print(f"{subset_name} F1: {f1:.4f}")
    print("-" * 40)
    print(f"{subset_name} Classification Report:")
    print(classification_report(
        all_labels, preds,
        labels=unique_labels,
        target_names=present_target_names,
        zero_division=0
    ))

    if do_plot and plot_prefix is not None:
        print(f"Generating confusion matrix plot for {subset_name}...")
        cm = confusion_matrix(all_labels, preds, labels=unique_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        tick_names = [label_names[i] for i in unique_labels]
        plt.title(f"{subset_name} (F1={f1:.4f})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(np.arange(len(unique_labels)) + 0.5, tick_names)
        plt.yticks(np.arange(len(unique_labels)) + 0.5, tick_names)
        plt.tight_layout()
        plot_path = os.path.join(
            PLOTS_DIR,
            f"{plot_prefix}_{subset_name.replace(' ', '_').lower()}_confusion.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"{subset_name} confusion plot saved to {plot_path}")

    return f1


def list_all_files_recursive(root_dir):
    """Return a list of all files under root_dir (recursive)."""
    paths = []
    for r, _, files in os.walk(root_dir):
        for fname in files:
            paths.append(os.path.abspath(os.path.join(r, fname)))
    return paths


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test a trained HTM model on Test/Val sets, all non-train, or other files."
    )
    parser.add_argument(
        "--model",
        default="models/cfg0007_sp40_enc16w7_tm16_act13_vf11.0000_tf11.0000.pkl",
        help="Path to the .pkl model file"
    )
    parser.add_argument(
        "--split",
        default="split.pkl",
        help="Path to split.pkl"
    )
    parser.add_argument(
        "--features",
        default="features_cache.pkl",
        help="Path to features_cache.pkl"
    )
    parser.add_argument(
        "--mode",
        choices=["orig", "all_non_train", "all_other_files"],
        default="all_non_train",
        help=(
            "Evaluation mode: "
            "'orig' uses original val/test splits. "
            "'all_non_train' evaluates all cached files not in the training set. "
            "'all_other_files' walks the filesystem under --data_root and evaluates "
            "files not present in the original splits."
        )
    )
    parser.add_argument(
        "--data_root",
        default="../UB_keystroke_dataset/",
        help="Root directory to search for 'other' human files when mode=all_other_files."
    )
    parser.add_argument(
        "--synthetic_bots_root",
        default="../bots_synt_dataset/Synthetic_Bots",
        help="Root directory for synthetic bot txt files when mode=all_other_files."
    )
    parser.add_argument(
        "--skip_orig_sets",
        action="store_true",
        help="If set, skip evaluation of original val/test sets (only relevant for mode=orig)."
    )
    parser.add_argument(
        "--write_decision_log",
        action="store_true",
        help="Write a per-window decision log for one human file and one bot file."
    )

    args = parser.parse_args()

    for path, label in [(args.model, "Model"), (args.split, "Split"),
                        (args.features, "Features")]:
        if not os.path.exists(path):
            print(f"Error: {label} file '{path}' not found.")
            sys.exit(1)

    # Load data
    print("Loading data...")
    with open(args.split, 'rb') as f:
        split = pickle.load(f)
    with open(args.features, 'rb') as f:
        features_cache = pickle.load(f)

    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)

    best_thresh      = model_data.get('best_thresh', 0.5)
    detection_mode   = model_data.get('detection_mode', 'mean')
    use_al           = model_data.get('use_anomaly_likelihood', False)
    al_period        = model_data.get('al_learning_period', 20)
    effective_warmup = model_data.get('effective_warmup',
                                      max(WARMUP_STEPS, al_period) if use_al else WARMUP_STEPS)

    if use_al and not _HAS_AL:
        print("WARNING: AnomalyLikelihood not available; scores will differ from training.",
              file=sys.stderr)
        use_al = False

    print(f"Model loaded. detection_mode={detection_mode}  use_al={use_al}  "
          f"thresh={best_thresh:.4f}  warmup={effective_warmup}")

    base_model_name = os.path.splitext(os.path.basename(args.model))[0]

    train_human = split.get('train_human', [])
    val_human   = split.get('val_human', [])
    test_human  = split.get('test_human', [])
    val_bots    = split.get('val_bots', [])
    test_bots   = split.get('test_bots', [])

    # Shared kwargs for run_inference
    infer_kw = dict(use_al=use_al, al_period=al_period, effective_warmup=effective_warmup)

    # ------------------------------------------------------------------
    # MODE: orig
    # ------------------------------------------------------------------
    if args.mode == "orig" and not args.skip_orig_sets:
        val_h_scores, val_h_labels, val_h_seqs = run_inference(
            model_data, features_cache, val_human, False,
            "Validation (Human)", **infer_kw
        )
        val_b_scores, val_b_labels, val_b_seqs = run_inference(
            model_data, features_cache, val_bots, True,
            "Validation (Bot)", **infer_kw
        )
        all_val_scores = val_h_scores + val_b_scores
        all_val_labels = val_h_labels + val_b_labels
        all_val_seqs   = val_h_seqs   + val_b_seqs

        test_h_scores, test_h_labels, test_h_seqs = run_inference(
            model_data, features_cache, test_human, False, "Test (Human)", **infer_kw
        )
        test_b_scores, test_b_labels, test_b_seqs = run_inference(
            model_data, features_cache, test_bots, True, "Test (Bot)", **infer_kw
        )
        all_test_scores = test_h_scores + test_b_scores
        all_test_labels = test_h_labels + test_b_labels
        all_test_seqs   = test_h_seqs   + test_b_seqs

        print("\n" + "=" * 40)
        print("RESULTS (Original Splits)")
        print("=" * 40)

        eval_kw = dict(detection_mode=detection_mode, effective_warmup=effective_warmup)

        val_f1 = eval_and_report(
            all_val_scores, all_val_labels, best_thresh,
            subset_name="Validation", do_plot=False,
            all_seqs=all_val_seqs, **eval_kw
        )
        eval_and_report(
            all_test_scores, all_test_labels, best_thresh,
            subset_name="Test", do_plot=False,
            all_seqs=all_test_seqs, **eval_kw
        )

        plot_results(
            val_h_seqs, val_b_seqs,
            all_val_scores, all_val_labels,
            all_val_seqs,
            val_h_scores, val_b_scores,
            best_thresh, val_f1,
            fname_slug=f"{base_model_name}_orig",
            title_prefix=f"{base_model_name} (orig)",
            detection_mode=detection_mode,
            effective_warmup=effective_warmup,
        )

        print("Generating combined confusion matrices plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (lbl, sc, seqs_set, title) in zip(axes, [
            (all_val_labels,  all_val_scores,  all_val_seqs,  "Validation Set"),
            (all_test_labels, all_test_scores, all_test_seqs, "Test Set"),
        ]):
            preds = apply_detection(seqs_set, detection_mode, best_thresh, effective_warmup)
            f1    = f1_score(lbl, preds, zero_division=0)
            cm    = confusion_matrix(lbl, preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{title} (F1={f1:.4f})")
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_xticklabels(['Human', 'Bot'])
            ax.set_yticklabels(['Human', 'Bot'])
        plt.tight_layout()
        orig_plot_path = os.path.join(PLOTS_DIR, f"{base_model_name}_orig_confusion.png")
        plt.savefig(orig_plot_path)
        plt.close()
        print(f"Original splits confusion plot saved to {orig_plot_path}")

        if args.write_decision_log:
            log_kw = dict(use_al=use_al, al_period=al_period, effective_warmup=effective_warmup)
            log_human = next((f for f in val_human  if f in features_cache), None)
            log_bot   = next((f for f in val_bots   if f in features_cache), None)
            if log_human:
                write_decision_log(log_human, features_cache[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_human_log.txt"),
                                   **log_kw)
            if log_bot:
                write_decision_log(log_bot, features_cache[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_bot_log.txt"),
                                   **log_kw)

    # ------------------------------------------------------------------
    # MODE: all_non_train
    # ------------------------------------------------------------------
    if args.mode == "all_non_train":
        print("\n" + "=" * 40)
        print("Evaluating ALL NON-TRAIN FILES (based on split/features_cache)")
        print("=" * 40)

        train_human_set = set(train_human)
        all_files       = list(features_cache.keys())
        known_human     = set(train_human) | set(val_human) | set(test_human)
        known_bot       = set(val_bots) | set(test_bots)

        non_train_human_files = [
            f for f in all_files
            if f in known_human and f not in train_human_set
        ]
        non_train_bot_files = [
            f for f in all_files
            if f in known_bot and f not in train_human_set
        ]

        print(f"Non-train human files: {len(non_train_human_files)}")
        print(f"Non-train bot files:   {len(non_train_bot_files)}")

        nt_h_scores, nt_h_labels, nt_h_seqs = run_inference(
            model_data, features_cache, non_train_human_files, False,
            "All Non-Train (Human)", **infer_kw
        )
        nt_b_scores, nt_b_labels, nt_b_seqs = run_inference(
            model_data, features_cache, non_train_bot_files, True,
            "All Non-Train (Bot)", **infer_kw
        )

        all_nt_scores = nt_h_scores + nt_b_scores
        all_nt_labels = nt_h_labels + nt_b_labels
        all_nt_seqs   = nt_h_seqs   + nt_b_seqs

        eval_kw = dict(detection_mode=detection_mode, effective_warmup=effective_warmup)

        nt_f1 = eval_and_report(
            all_nt_scores, all_nt_labels, best_thresh,
            subset_name="All Non-Train", do_plot=True,
            plot_prefix=f"{base_model_name}_all_non_train",
            all_seqs=all_nt_seqs, **eval_kw
        )

        plot_results(
            nt_h_seqs, nt_b_seqs,
            all_nt_scores, all_nt_labels,
            all_nt_seqs,
            nt_h_scores, nt_b_scores,
            best_thresh, nt_f1,
            fname_slug=f"{base_model_name}_all_non_train",
            title_prefix=f"{base_model_name} (all_non_train)",
            detection_mode=detection_mode,
            effective_warmup=effective_warmup,
        )

        if args.write_decision_log:
            log_kw = dict(use_al=use_al, al_period=al_period, effective_warmup=effective_warmup)
            log_human = next((f for f in non_train_human_files if f in features_cache), None)
            log_bot   = next((f for f in non_train_bot_files   if f in features_cache), None)
            if log_human:
                write_decision_log(log_human, features_cache[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_human_log.txt"),
                                   **log_kw)
            if log_bot:
                write_decision_log(log_bot, features_cache[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_bot_log.txt"),
                                   **log_kw)

    # ------------------------------------------------------------------
    # MODE: all_other_files
    # ------------------------------------------------------------------
    if args.mode == "all_other_files":
        print("\n" + "=" * 40)
        print("Evaluating ALL OTHER RAW TXT FILES (filesystem, excluding original splits)")
        print("=" * 40)

        split_exclude_names = set(
            os.path.basename(f) for f in (
                list(train_human) + list(val_human) + list(test_human) +
                list(val_bots) + list(test_bots)
            )
        )

        # ---------- Other human files ----------
        data_root_abs = os.path.abspath(args.data_root)
        print(f"[1/5] Walking data_root={data_root_abs} ...")
        all_fs_files = list_all_files_recursive(data_root_abs)
        print(f"    Total files found: {len(all_fs_files)}")

        print("[2/5] Filtering for .txt files (other humans)...")
        txt_files = [p for p in all_fs_files if p.lower().endswith(".txt")]
        print(f"    .txt files found: {len(txt_files)}")

        print("[3/5] Excluding original split filenames from other humans...")
        candidate_humans = [
            p for p in txt_files
            if os.path.basename(p) not in split_exclude_names
        ]
        print(f"    Human candidates after exclusions: {len(candidate_humans)}")

        # ---------- Synthetic bot files ----------
        synt_bots_root = os.path.abspath(args.synthetic_bots_root)
        print(f"[4/5] Walking synthetic bots root={synt_bots_root} ...")
        if os.path.isdir(synt_bots_root):
            synt_files = list_all_files_recursive(synt_bots_root)
            synt_txt   = [p for p in synt_files if p.lower().endswith(".txt")]
            candidate_bots = [
                p for p in synt_txt
                if os.path.basename(p) not in split_exclude_names
            ]
            print(f"    Bot candidates after exclusions: {len(candidate_bots)}")
        else:
            candidate_bots = []
            print("    Synthetic bots directory not found; skipping.")

        if not candidate_humans and not candidate_bots:
            print("No candidate 'other' txt files after exclusions.")
            return

        # ---------- Reference pool ----------
        print("[5/5] Building reference pool from train_human files...")
        ref_dwells, ref_flights = create_reference_pool(train_human)
        print("    Reference pool built.")

        # ---------- Stage A: parallel feature extraction ----------
        print("\n[Stage A] Computing feature sequences (multiprocessing)...")
        worker_args = (
            [(p, ref_dwells, ref_flights, STEP_SIZE_HUMAN) for p in candidate_humans]
            + [(p, ref_dwells, ref_flights, STEP_SIZE_BOT) for p in candidate_bots]
        )

        other_feature_seqs = {}
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        print(f"Using {num_workers} worker processes.")

        with mp.Pool(processes=num_workers) as pool:
            for filepath, seq in tqdm(
                pool.imap_unordered(_compute_features_wrapper, worker_args),
                total=len(worker_args), desc="Feature extraction", unit="file"
            ):
                if seq.size > 0:
                    other_feature_seqs[filepath] = seq

        print(f"    Files with usable features: {len(other_feature_seqs)}")

        if not other_feature_seqs:
            print("No usable 'other' txt files. Nothing to evaluate.")
            return

        # ---------- Stage B: inference ----------
        print("\n[Stage B] Running inference...")
        sp          = model_data['sp']
        tm          = model_data['tm']
        encoder     = model_data['encoder']
        input_width = model_data['input_width']
        active_columns = SDR(sp.getColumnDimensions())

        scores, labels = [], []
        raw_seqs = []
        seqs_human, seqs_bot = [], []
        humans_set = set(candidate_humans)
        bots_set   = set(candidate_bots)

        for filepath, feats_seq in tqdm(other_feature_seqs.items(),
                                        desc="Inference", unit="file"):
            tm.reset()
            al = AnomalyLikelihood(learningPeriod=al_period) if (use_al and _HAS_AL) else None
            file_scores = []
            for step, feats in enumerate(feats_seq):
                enc_sdr = SDR(input_width)
                enc_sdr.dense = encoder.encode(feats)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                score = tm.anomaly
                if al is not None:
                    score = al.anomalyProbability(score, score, step)
                file_scores.append(score)

            if file_scores:
                valid = file_scores[effective_warmup:] if len(file_scores) > effective_warmup else file_scores
                mean_score = float(np.mean(valid)) if valid else 0.0

                if filepath in bots_set:
                    labels.append(1)
                    seqs_bot.append(file_scores)
                else:
                    labels.append(0)
                    seqs_human.append(file_scores)

                scores.append(mean_score)
                raw_seqs.append(file_scores)

        if not scores:
            print("No scores produced for 'other' files.")
            return

        # ---------- Stage C: metrics + plots ----------
        print("\n[Stage C] Computing metrics and plots...")
        eval_kw = dict(detection_mode=detection_mode, effective_warmup=effective_warmup)

        other_f1 = eval_and_report(
            scores, labels, best_thresh,
            subset_name="All Other Files", do_plot=True,
            plot_prefix=f"{base_model_name}_all_other_files",
            all_seqs=raw_seqs, **eval_kw
        )

        human_scores = [s for s, l in zip(scores, labels) if l == 0]
        bot_scores   = [s for s, l in zip(scores, labels) if l == 1]

        plot_results(
            val_h_seqs=seqs_human,
            val_b_seqs=seqs_bot,
            all_val_scores=scores,
            all_val_labels=labels,
            all_val_seqs=raw_seqs,
            val_h_scores=human_scores,
            val_b_scores=bot_scores,
            best_thresh=best_thresh,
            val_f1=other_f1,
            fname_slug=f"{base_model_name}_all_other_files",
            title_prefix=f"{base_model_name} (all_other_files)",
            detection_mode=detection_mode,
            effective_warmup=effective_warmup,
        )

        if args.write_decision_log:
            log_kw = dict(use_al=use_al, al_period=al_period, effective_warmup=effective_warmup)
            log_human = next(
                (f for f in candidate_humans if f in other_feature_seqs), None)
            log_bot   = next(
                (f for f in candidate_bots   if f in other_feature_seqs), None)
            if log_human:
                write_decision_log(log_human, other_feature_seqs[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_human_log.txt"),
                                   **log_kw)
            if log_bot:
                write_decision_log(log_bot, other_feature_seqs[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_model_name}_bot_log.txt"),
                                   **log_kw)


if __name__ == "__main__":
    main()
