"""
htm_combined/htm_combined_test_model.py
Test a trained HTM-Combined model on multiple evaluation modes.

Modes:
  orig            — original val/test splits from split.pkl
  all_non_train   — all files not in the training set (uses windows cache if available)
  all_other_files — raw .txt files outside the original splits; parses on-the-fly
                    using parse_file_distance + ref pool stored in the model

Usage (from project root):
  python htm_combined/htm_combined_test_model.py \\
      --model hc_models/<slug>.pkl [--mode all_non_train] [--write_decision_log]

Running on Newton SLURM (for all_other_files parallel feature extraction):
  sbatch slurm/hc_test_model.sh
  (Edit MODEL= and MODE= in slurm/hc_test_model.sh before submitting.)
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_combined_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_combined_dir not in sys.path:
    sys.path.insert(0, _htm_combined_dir)

os.chdir(_project_root)

import argparse
import multiprocessing as mp
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_combined_common import (
    apply_detection, get_file_combined_seq,
    make_anomaly_likelihood, WARMUP_STEPS,
    parse_file_distance,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLIT_FILE  = "split.pkl"
CACHE_FILE  = "stats_cache.pkl"
DIST_CACHE  = "dist_cache.pkl"
WINDOWS_DIR = "windows_cache"
PLOTS_DIR   = "hc_plots"
LOGS_DIR    = "logs"

for _d in (PLOTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (for all_other_files mode — no cache)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_worker(args):
    filepath, window_size, window_step, ref_dwells, ref_flights, ref_dists = args
    events = parse_file_distance(filepath)
    if not events:
        return filepath, []
    seq = get_file_combined_seq(events, window_size, window_step,
                                ref_dwells, ref_flights, ref_dists)
    return filepath, seq


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def get_scores(file_list, is_bot, model_data, seqs_dict, warmup, al_period):
    """
    Run SP+TM+AL on each file.
    seqs_dict maps filepath → [(stats_21, key_idx, dwell, flight, dist), …]
    Returns (mean_scores, labels, raw_seqs).
    """
    sp          = model_data['sp']
    tm          = model_data['tm']
    encoder     = model_data['encoder']
    input_width = model_data['input_width']
    active_columns = SDR(sp.getColumnDimensions())

    scores, labels, seqs = [], [], []
    for fp in file_list:
        seq = seqs_dict.get(fp)
        if not seq:
            continue
        al = make_anomaly_likelihood(al_period)
        tm.reset()
        raw = []
        for stats, key_idx, dwell, flight, dist in seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, False, active_columns)
            tm.compute(active_columns, learn=False)
            raw.append(al.compute(float(tm.anomaly)))

        valid = raw[warmup:]
        scores.append(float(np.mean(valid)) if valid else 0.0)
        labels.append(1 if is_bot else 0)
        seqs.append(raw)

    return scores, labels, seqs


# ──────────────────────────────────────────────────────────────────────────────
# Decision log
# ──────────────────────────────────────────────────────────────────────────────
def write_decision_log(filepath, seq, model_data, true_label, out_path,
                       warmup, al_period):
    sp          = model_data['sp']
    tm          = model_data['tm']
    encoder     = model_data['encoder']
    input_width = model_data['input_width']
    best_thresh = model_data['best_thresh']
    active_columns = SDR(sp.getColumnDimensions())

    al = make_anomaly_likelihood(al_period)
    tm.reset()
    rows = []
    bot_triggered_at = None
    n_windows = len(seq)

    for step, (stats, key_idx, dwell, flight, dist) in enumerate(seq):
        enc_sdr       = SDR(input_width)
        enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
        sp.compute(enc_sdr, False, active_columns)
        tm.compute(active_columns, learn=False)
        score = al.compute(float(tm.anomaly))

        if step < warmup:
            status = "WARMUP"
            reason = f"warmup period ({step+1}/{warmup})"
        elif bot_triggered_at is not None:
            status = "BOT"
            reason = f"already triggered at window {bot_triggered_at}"
        elif score >= best_thresh:
            bot_triggered_at = step
            status = "BOT"
            reason = f"first crossing: {score:.4f} >= threshold {best_thresh:.4f}"
        else:
            status = "no crossing"
            reason = f"score {score:.4f} < threshold {best_thresh:.4f}"

        rows.append((step, score, status, reason))

    final_pred = 1 if bot_triggered_at is not None else 0
    final_str  = "BOT"     if final_pred  == 1 else "HUMAN"
    true_str   = "BOT"     if true_label  == 1 else "HUMAN"
    verdict    = "CORRECT" if final_pred  == true_label else "WRONG"

    with open(out_path, 'w') as fh:
        fh.write(f"# File:           {filepath}\n")
        fh.write(f"# True label:     {true_str}\n")
        fh.write(f"# Final decision: {final_str}  [{verdict}]\n")
        fh.write(f"# Detection mode: first_crossing\n")
        fh.write(f"# Threshold:      {best_thresh:.4f}\n")
        fh.write(f"# Warmup steps:   {warmup}\n")
        fh.write(f"# Total windows:  {n_windows}\n")
        fh.write("#\n")
        fh.write(f"{'Window':>8}  {'Score':>8}  {'Status':>14}  Reason\n")
        fh.write(f"{'-'*8}  {'-'*8}  {'-'*14}  {'-'*55}\n")
        for widx, sc, st, rs in rows:
            fh.write(f"{widx:>8}  {sc:>8.4f}  {st:>14}  {rs}\n")

    print(f"  Decision log -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(h_seqs, b_seqs, h_scores, b_scores,
                 all_seqs, all_labels, best_thresh, best_bacc,
                 fname_slug, title, warmup):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=7)

    ax = axes[0]
    n = min(3, len(h_seqs), len(b_seqs))
    for i in range(n):
        ax.plot(h_seqs[i], alpha=0.6, color='steelblue',
                label='Human' if i == 0 else '_nolegend_')
    for i in range(n):
        ax.plot(b_seqs[i], alpha=0.6, color='tomato',
                label='Bot' if i == 0 else '_nolegend_')
    ax.axhline(best_thresh, color='green', linestyle='--', linewidth=1.5,
               label=f'Thresh={best_thresh:.3f}')
    ax.axvline(warmup, color='gray', linestyle=':', linewidth=1,
               label=f'Warmup={warmup}')
    ax.set_title('Anomaly Score Over Time (window steps)\nsample files')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True)

    ax = axes[1]
    ax.hist(h_scores, bins=15, alpha=0.65, color='steelblue', label='Human')
    ax.hist(b_scores, bins=15, alpha=0.65, color='tomato',    label='Bot')
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2,
               label=f'Thresh={best_thresh:.3f}')
    ax.set_title('Per-File Mean Anomaly Score')
    ax.set_xlabel('Mean anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ths = np.linspace(0, 1, 500)
    baccs = [
        balanced_accuracy_score(
            all_labels,
            apply_detection(all_seqs, 'first_crossing', t, warmup, labels=all_labels))
        for t in ths
    ]
    ax.plot(ths, baccs, color='blue', linewidth=2, label='Balanced Acc')
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Best={best_thresh:.3f} (BAcc={best_bacc:.4f})')
    ax.set_title('Balanced Accuracy vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_results.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Plot  -> {fpath}")


def plot_confusion(h_labels, h_preds_labels, t_labels, t_preds_labels,
                   fname_slug, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=7)
    for ax, (labels, preds, subset) in zip(axes, [
        (h_labels, h_preds_labels, "Validation Set"),
        (t_labels, t_preds_labels, "Test Set"),
    ]):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(subset)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Human', 'Bot'])
        ax.set_yticklabels(['Human', 'Bot'])
    plt.tight_layout()
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_confusion.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Plot  -> {fpath}")


def eval_and_report(all_scores, all_labels, all_seqs, best_thresh, warmup,
                    subset_name="Subset", do_plot=False, plot_prefix=None,
                    h_seqs=None, b_seqs=None, h_scores=None, b_scores=None):
    preds = apply_detection(all_seqs, 'first_crossing', best_thresh, warmup,
                            labels=all_labels)
    f1   = f1_score(all_labels, preds, zero_division=0)
    bacc = balanced_accuracy_score(all_labels, preds)

    print(f"\n{'='*40}")
    print(f"{subset_name} RESULTS")
    print(f"{'='*40}")
    print(f"  F1   = {f1:.4f}")
    print(f"  BAcc = {bacc:.4f}")
    print(classification_report(all_labels, preds,
                                target_names=['Human', 'Bot'],
                                zero_division=0))

    if do_plot and plot_prefix is not None:
        cm = confusion_matrix(all_labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"{subset_name} (F1={f1:.4f}  BAcc={bacc:.4f})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Human', 'Bot'])
        plt.yticks([0.5, 1.5], ['Human', 'Bot'])
        plt.tight_layout()
        fpath = os.path.join(PLOTS_DIR,
                             f"{plot_prefix}_{subset_name.replace(' ', '_').lower()}_confusion.png")
        plt.savefig(fpath)
        plt.close()
        print(f"  Confusion plot -> {fpath}")

        if h_seqs is not None and b_seqs is not None:
            plot_results(h_seqs, b_seqs, h_scores, b_scores,
                         all_seqs, all_labels, best_thresh, bacc,
                         fname_slug=plot_prefix,
                         title=f"{plot_prefix}  {subset_name}  F1={f1:.4f}  BAcc={bacc:.4f}",
                         warmup=warmup)

    return f1, bacc


# ──────────────────────────────────────────────────────────────────────────────
# Load sequences from cache or windows_cache
# ──────────────────────────────────────────────────────────────────────────────
def _load_seqs_from_cache(file_list, wcache, cache, window_size, window_step,
                           ref_dwells, ref_flights, ref_dists):
    result = {}
    for fp in file_list:
        if wcache is not None:
            seq = wcache['sequences'].get(fp)
        else:
            events = cache.get(fp)
            seq = get_file_combined_seq(events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists) if events else None
        if seq:
            result[fp] = seq
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test a trained HTM-Combined model")
    parser.add_argument("--model", required=True,
                        help="Path to the .pkl model file (hc_models/<slug>.pkl)")
    parser.add_argument("--split",    default=SPLIT_FILE,
                        help=f"Path to split.pkl (default: {SPLIT_FILE})")
    parser.add_argument("--mode",
                        choices=["orig", "all_non_train", "all_other_files"],
                        default="all_non_train",
                        help=(
                            "'orig' — original val/test from split.pkl; "
                            "'all_non_train' — all non-training files from split.pkl (default); "
                            "'all_other_files' — walk filesystem for new .txt files"
                        ))
    parser.add_argument("--data_root",
                        default="../UB_keystroke_dataset/",
                        help="Human .txt files root (all_other_files mode)")
    parser.add_argument("--bots_root",
                        default="../BadUSBdataset",
                        help="Bot .txt files root (all_other_files mode)")
    parser.add_argument("--write_decision_log", action="store_true",
                        help="Write a per-window decision log for one human and one bot file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file '{args.model}' not found.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    with open(args.model, 'rb') as fh:
        model_data = pickle.load(fh)

    best_thresh  = model_data['best_thresh']
    warmup       = model_data.get('warmup_steps', WARMUP_STEPS)
    al_period    = model_data.get('al_period', 10)
    window_size  = model_data.get('window_size', 10)
    window_step  = model_data.get('window_step', 1)
    ref_dwells   = model_data['ref_dwells']
    ref_flights  = model_data['ref_flights']
    ref_dists    = model_data['ref_dists']
    config_idx   = model_data.get('config_idx', 0)

    print(f"  Config: hc{config_idx:04d}  thresh={best_thresh:.4f}  "
          f"warmup={warmup}  al_period={al_period}  "
          f"window={window_size}x{window_step}")

    base_name = os.path.splitext(os.path.basename(args.model))[0]

    # ── Load split ────────────────────────────────────────────────
    if not os.path.exists(args.split):
        print(f"ERROR: split.pkl not found at '{args.split}'.")
        sys.exit(1)

    with open(args.split, 'rb') as fh:
        split = pickle.load(fh)

    train_human = split['train_human']
    val_human   = split['val_human']
    test_human  = split['test_human']
    val_bots    = split['val_bots']
    test_bots   = split['test_bots']

    # ── Try windows cache first, then raw cache ───────────────────
    wcache_path = os.path.join(WINDOWS_DIR, f"ws{window_size}s{window_step}.pkl")
    wcache = None
    cache  = None

    if os.path.exists(wcache_path):
        print(f"Loading windows cache: {wcache_path}")
        with open(wcache_path, 'rb') as fh:
            wcache = pickle.load(fh)
    else:
        print(f"Windows cache not found ({wcache_path}); falling back to raw cache.")
        cache_path = CACHE_FILE if os.path.exists(CACHE_FILE) else DIST_CACHE
        if os.path.exists(cache_path):
            print(f"  Loading {cache_path}")
            with open(cache_path, 'rb') as fh:
                cache = pickle.load(fh)
        else:
            print(f"  WARNING: No cache found — orig/all_non_train modes will have no data.")

    def load_seqs(file_list):
        return _load_seqs_from_cache(file_list, wcache, cache,
                                     window_size, window_step,
                                     ref_dwells, ref_flights, ref_dists)

    infer_kw = dict(model_data=model_data, warmup=warmup, al_period=al_period)

    # ──────────────────────────────────────────────────────────────
    # MODE: orig
    # ──────────────────────────────────────────────────────────────
    if args.mode == "orig":
        print("\nMode: orig (val + test splits)")

        val_seqs  = load_seqs(val_human  + val_bots)
        test_seqs = load_seqs(test_human + test_bots)

        val_h_sc, val_h_lb, val_h_sq   = get_scores(val_human,  False, seqs_dict=val_seqs,  **infer_kw)
        val_b_sc, val_b_lb, val_b_sq   = get_scores(val_bots,   True,  seqs_dict=val_seqs,  **infer_kw)
        test_h_sc, test_h_lb, test_h_sq = get_scores(test_human, False, seqs_dict=test_seqs, **infer_kw)
        test_b_sc, test_b_lb, test_b_sq = get_scores(test_bots,  True,  seqs_dict=test_seqs, **infer_kw)

        all_val_seqs    = val_h_sq  + val_b_sq
        all_val_labels  = val_h_lb  + val_b_lb
        all_val_scores  = val_h_sc  + val_b_sc
        all_test_seqs   = test_h_sq + test_b_sq
        all_test_labels = test_h_lb + test_b_lb
        all_test_scores = test_h_sc + test_b_sc

        val_f1,  val_bacc  = eval_and_report(all_val_scores,  all_val_labels,  all_val_seqs,
                                              best_thresh, warmup, "Validation")
        test_f1, test_bacc = eval_and_report(all_test_scores, all_test_labels, all_test_seqs,
                                              best_thresh, warmup, "Test")

        val_preds  = apply_detection(all_val_seqs,  'first_crossing', best_thresh, warmup,
                                     labels=all_val_labels)
        test_preds = apply_detection(all_test_seqs, 'first_crossing', best_thresh, warmup,
                                     labels=all_test_labels)

        plot_results(val_h_sq, val_b_sq, val_h_sc, val_b_sc,
                     all_val_seqs, all_val_labels, best_thresh, val_bacc,
                     fname_slug=f"{base_name}_orig",
                     title=f"{base_name} (orig)  Val F1={val_f1:.4f}  Test F1={test_f1:.4f}",
                     warmup=warmup)
        plot_confusion(all_val_labels, val_preds, all_test_labels, test_preds,
                       f"{base_name}_orig", f"{base_name} (orig)")

        if args.write_decision_log:
            log_human = next((f for f in val_human if f in val_seqs), None)
            log_bot   = next((f for f in val_bots  if f in val_seqs), None)
            if log_human:
                write_decision_log(log_human, val_seqs[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_name}_human_log.txt"),
                                   warmup, al_period)
            if log_bot:
                write_decision_log(log_bot, val_seqs[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_name}_bot_log.txt"),
                                   warmup, al_period)

    # ──────────────────────────────────────────────────────────────
    # MODE: all_non_train
    # ──────────────────────────────────────────────────────────────
    elif args.mode == "all_non_train":
        print("\nMode: all_non_train")

        train_set   = set(train_human)
        known_human = set(train_human) | set(val_human) | set(test_human)
        known_bot   = set(val_bots) | set(test_bots)

        nt_human = [f for f in known_human if f not in train_set]
        nt_bots  = list(known_bot)

        print(f"  Non-train human: {len(nt_human)}")
        print(f"  Non-train bots:  {len(nt_bots)}")

        nt_seqs = load_seqs(nt_human + nt_bots)

        nt_h_sc, nt_h_lb, nt_h_sq = get_scores(nt_human, False, seqs_dict=nt_seqs, **infer_kw)
        nt_b_sc, nt_b_lb, nt_b_sq = get_scores(nt_bots,  True,  seqs_dict=nt_seqs, **infer_kw)

        all_nt_seqs   = nt_h_sq + nt_b_sq
        all_nt_labels = nt_h_lb + nt_b_lb
        all_nt_scores = nt_h_sc + nt_b_sc

        nt_f1, nt_bacc = eval_and_report(
            all_nt_scores, all_nt_labels, all_nt_seqs,
            best_thresh, warmup, "All Non-Train",
            do_plot=True, plot_prefix=f"{base_name}_all_non_train",
            h_seqs=nt_h_sq, b_seqs=nt_b_sq,
            h_scores=nt_h_sc, b_scores=nt_b_sc)

        if args.write_decision_log:
            log_human = next((f for f in nt_human if f in nt_seqs), None)
            log_bot   = next((f for f in nt_bots  if f in nt_seqs), None)
            if log_human:
                write_decision_log(log_human, nt_seqs[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_name}_human_log.txt"),
                                   warmup, al_period)
            if log_bot:
                write_decision_log(log_bot, nt_seqs[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_name}_bot_log.txt"),
                                   warmup, al_period)

    # ──────────────────────────────────────────────────────────────
    # MODE: all_other_files
    # ──────────────────────────────────────────────────────────────
    elif args.mode == "all_other_files":
        print("\nMode: all_other_files (filesystem walk + on-the-fly parsing)")

        split_names = set(
            os.path.basename(f)
            for f in list(train_human) + list(val_human) + list(test_human)
                     + list(val_bots) + list(test_bots)
        )

        def walk_txt(root):
            files = []
            if not os.path.isdir(root):
                print(f"  WARNING: directory not found: {root}")
                return files
            for r, _, names in os.walk(root):
                for name in names:
                    if name.lower().endswith('.txt'):
                        files.append(os.path.abspath(os.path.join(r, name)))
            return files

        candidate_humans = [p for p in walk_txt(os.path.abspath(args.data_root))
                            if os.path.basename(p) not in split_names]
        candidate_bots   = [p for p in walk_txt(os.path.abspath(args.bots_root))
                            if os.path.basename(p) not in split_names]

        print(f"  Human candidates: {len(candidate_humans)}")
        print(f"  Bot candidates:   {len(candidate_bots)}")

        if not candidate_humans and not candidate_bots:
            print("No candidate files found.")
            sys.exit(0)

        # Parallel feature extraction
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
        print(f"  Extracting features with {num_workers} workers...")

        worker_args = [
            (fp, window_size, window_step, ref_dwells, ref_flights, ref_dists)
            for fp in candidate_humans + candidate_bots
        ]

        other_seqs = {}
        with mp.Pool(processes=num_workers) as pool:
            for fp, seq in tqdm(
                pool.imap_unordered(_extract_worker, worker_args),
                total=len(worker_args), desc="Feature extraction", unit="file"
            ):
                if seq:
                    other_seqs[fp] = seq

        print(f"  Files with usable sequences: {len(other_seqs)}")
        if not other_seqs:
            print("No usable files after feature extraction.")
            sys.exit(0)

        other_h_sc, other_h_lb, other_h_sq = get_scores(
            candidate_humans, False, seqs_dict=other_seqs, **infer_kw)
        other_b_sc, other_b_lb, other_b_sq = get_scores(
            candidate_bots,   True,  seqs_dict=other_seqs, **infer_kw)

        all_other_seqs   = other_h_sq + other_b_sq
        all_other_labels = other_h_lb + other_b_lb
        all_other_scores = other_h_sc + other_b_sc

        other_f1, other_bacc = eval_and_report(
            all_other_scores, all_other_labels, all_other_seqs,
            best_thresh, warmup, "All Other Files",
            do_plot=True, plot_prefix=f"{base_name}_all_other_files",
            h_seqs=other_h_sq, b_seqs=other_b_sq,
            h_scores=other_h_sc, b_scores=other_b_sc)

        if args.write_decision_log:
            log_human = next((f for f in candidate_humans if f in other_seqs), None)
            log_bot   = next((f for f in candidate_bots   if f in other_seqs), None)
            if log_human:
                write_decision_log(log_human, other_seqs[log_human], model_data, 0,
                                   os.path.join(LOGS_DIR, f"{base_name}_human_log.txt"),
                                   warmup, al_period)
            if log_bot:
                write_decision_log(log_bot, other_seqs[log_bot], model_data, 1,
                                   os.path.join(LOGS_DIR, f"{base_name}_bot_log.txt"),
                                   warmup, al_period)


if __name__ == "__main__":
    main()
