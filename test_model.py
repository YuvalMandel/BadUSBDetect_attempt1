import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import multiprocessing as mp
from tqdm import tqdm
import glob
from scipy import stats
import warnings
import random

warnings.filterwarnings('ignore')

# Match training constants
WINDOW_SIZE = 15
STEP_SIZE_HUMAN = 1
STEP_SIZE_BOT = 1
NUM_REFERENCES = 50
RANDOM_SEED = 42


# Ensure htm_common is available
try:
    import htm_common
except ImportError:
    print("htm_common.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# HTM Imports
try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("HTM libraries not found. Please install htm.core.")
    sys.exit(1)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def parse_file(filepath):
    dwells = []
    flights = []
    active_keys = {}
    last_keyup = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return np.array([]), np.array([])

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key, action = parts[0], parts[1]
        try:
            ts = int(parts[2])
        except ValueError:
            continue

        scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup is not None:
                delta = (ts - last_keyup) / scale
                if 0 < delta < 5000:
                    flights.append(delta)
        elif action == "KeyUp":
            last_keyup = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts) / scale
                if 0 < delta < 3000:
                    dwells.append(delta)

    return np.array(dwells), np.array(flights)


def extract_features(window_data, reference_pool):
    if len(window_data) < WINDOW_SIZE:
        return None
    if np.isnan(window_data).any():
        return None

    feat_mean = np.mean(window_data)
    feat_med  = np.median(window_data)
    feat_std  = np.std(window_data)

    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0, 10
    else:
        feat_skew = stats.skew(window_data)
        feat_kurt = stats.kurtosis(window_data)

    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))

    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt,
            np.min(ks_scores), np.min(w_scores)]


def create_reference_pool(train_human_files):
    """Rebuild the reference pools exactly like training, using train_human paths from split.pkl."""
    print("--- Collecting Reference Pool (train humans only) ---")
    all_human_dwells  = []
    all_human_flights = []

    sample_files = list(train_human_files)
    random.shuffle(sample_files)

    for filepath in sample_files[:50]:
        d, fl = parse_file(filepath)
        if len(d)  >= WINDOW_SIZE: all_human_dwells.extend(d)
        if len(fl) >= WINDOW_SIZE: all_human_flights.extend(fl)

    ref_dwells, ref_flights = [], []

    if len(all_human_dwells) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_dwells) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_dwells.append(np.array(all_human_dwells[start: start + WINDOW_SIZE]))

    if len(all_human_flights) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_flights) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_flights.append(np.array(all_human_flights[start: start + WINDOW_SIZE]))

    return ref_dwells, ref_flights



def load_model(model_path):
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def compute_features_for_raw_file(filepath, ref_dwells, ref_flights, step_size):
    """
    Parse a raw keystroke .txt file and compute the same feature sequences
    used during training (dwells + flights windows).
    Returns np.array of shape [num_windows, feature_dim] or empty array.
    """
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))

    features_seq = []
    if min_len < WINDOW_SIZE:
        return np.array([])

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i: i + WINDOW_SIZE]
        w_f = f[i: i + WINDOW_SIZE]

        ft_d = extract_features(w_d, ref_dwells)
        ft_f = extract_features(w_f, ref_flights)

        if ft_d and ft_f:
            features_seq.append(ft_f + ft_d)

    return np.array(features_seq)

def _compute_features_wrapper(args):
    filepath, ref_dwells, ref_flights, step_size = args
    return filepath, compute_features_for_raw_file(filepath, ref_dwells, ref_flights, step_size)

def run_inference(model_data, file_features, file_list, is_bot, desc="Inference",
                  collect_seqs=False):
    """
    If collect_seqs=True, also return the raw anomaly sequences per file
    (for plotting anomaly-over-time).
    """
    sp = model_data['sp']
    tm = model_data['tm']
    encoder = model_data['encoder']
    input_width = model_data['input_width']

    # Create SDR object for input
    active_columns = SDR(sp.getColumnDimensions())

    scores = []
    labels = []
    seqs = []

    print(f"Running {desc} on {len(file_list)} files...")

    for filepath in file_list:
        if filepath not in file_features:
            continue

        # Reset Temporal Memory for each new sequence
        tm.reset()

        file_scores = []
        for feats in file_features[filepath]:
            # Encode
            dense_input = encoder.encode(feats)
            enc_sdr = SDR(input_width)
            enc_sdr.dense = dense_input

            # Spatial Pooler (learning disabled)
            sp.compute(enc_sdr, False, active_columns)

            # Temporal Memory (learning disabled)
            tm.compute(active_columns, learn=False)

            file_scores.append(tm.anomaly)

        if file_scores:
            # Skip warmup (first 5 steps)
            valid_scores = file_scores[5:] if len(file_scores) > 5 else file_scores
            scores.append(np.mean(valid_scores))
            labels.append(1 if is_bot else 0)
            if collect_seqs:
                seqs.append(file_scores)

    if collect_seqs:
        return scores, labels, seqs
    else:
        return scores, labels


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_results(val_h_seqs, val_b_seqs,
                 all_val_scores, all_val_labels,
                 val_h_scores, val_b_scores,
                 best_thresh, val_f1,
                 fname_slug, title_prefix=""):
    """
    3-panel plot:
      1) anomaly score over time for sample human/bot files
      2) per-file mean score distribution + threshold
      3) F1 vs threshold curve, with best threshold marked
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title = f"{title_prefix} Val F1={val_f1:.4f}"
    fig.suptitle(title, fontsize=7)

    # Panel 1 — anomaly score over time
    ax = axes[0]
    n = min(3, len(val_h_seqs), len(val_b_seqs))
    for i in range(n):
        ax.plot(val_h_seqs[i], alpha=0.7, color='steelblue',
                label='Human' if i == 0 else '_nolegend_')
    for i in range(n):
        ax.plot(val_b_seqs[i], alpha=0.7, color='tomato',
                label='Bot' if i == 0 else '_nolegend_')
    ax.set_title('Anomaly Score Over Time\n(sample files)')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    # Panel 2 — per-file mean score distribution
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

    # Panel 3 — F1 vs threshold
    ax = axes[2]
    ths = np.linspace(0, 1, 200)
    f1s = [
        f1_score(all_val_labels,
                 [1 if s >= t else 0 for s in all_val_scores],
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
                    do_plot=False, plot_prefix=None):
    preds = [1 if s >= best_thresh else 0 for s in all_scores]
    f1 = f1_score(all_labels, preds)
    print("\n" + "=" * 40)
    print(f"{subset_name} RESULTS")
    print("=" * 40)
    print(f"{subset_name} F1: {f1:.4f}")
    print("-" * 40)
    print(f"{subset_name} Classification Report:")
    print(classification_report(all_labels, preds, target_names=['Human', 'Bot']))

    if do_plot and plot_prefix is not None:
        print(f"Generating confusion matrix plot for {subset_name}...")
        cm = confusion_matrix(all_labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"{subset_name} (F1={f1:.4f})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0.5, 1.5], ['Human', 'Bot'])
        plt.yticks([0.5, 1.5], ['Human', 'Bot'])
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{plot_prefix}_{subset_name.replace(' ', '_').lower()}_confusion.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"{subset_name} confusion plot saved to {plot_path}")

    return f1

def validate_txt_file(path, min_cols=1):
    """
    Basic sanity check for a .txt file.
    - Tries to read lines.
    - Splits on whitespace or comma.
    - Ensures at least one non-empty line.
    - Ensures all non-empty lines have the same number of numeric columns.
    Returns True if the file 'makes sense', False otherwise.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_cols = None
            any_valid = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on comma first, fallback to whitespace
                if "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = line.split()

                # Try to parse as floats
                vals = []
                for p in parts:
                    if not p:
                        continue
                    try:
                        vals.append(float(p))
                    except ValueError:
                        # Not numeric, reject file
                        return False

                if not vals:
                    continue

                if first_cols is None:
                    first_cols = len(vals)
                    if first_cols < min_cols:
                        return False
                else:
                    if len(vals) != first_cols:
                        return False

                any_valid = True

            return bool(any_valid)
    except Exception as e:
        print(f"Warning: failed to read/validate {path}: {e}")
        return False


def list_all_files_recursive(root_dir):
    """Return a list of all files under root_dir (recursive)."""
    paths = []
    for r, _, files in os.walk(root_dir):
        for fname in files:
            # print(fname)
            full = os.path.join(r, fname)
            paths.append(os.path.abspath(full))
    return paths


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
        default="all_other_files",
        help=(
            "Evaluation mode: "
            "'orig' uses original val/test splits. "
            "'all_non_train' evaluates all files not in the original train set "
            "(using split/feature keys). "
            "'all_other_files' walks the filesystem under --data_root, then "
            "filters out any files that are part of the original train/val/test sets."
        )
    )
    parser.add_argument(
        "--data_root",
        default="../UB_keystroke_dataset/",
        help="Root directory to search for 'other' files when mode=all_other_files."
    )
    parser.add_argument(
        "--skip_orig_sets",
        action="store_true",
        help="If set, skip evaluation and plots for original val/test sets (only relevant for mode=orig)."
    )

    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
    if not os.path.exists(args.split):
        print(f"Error: Split file '{args.split}' not found.")
        sys.exit(1)
    if not os.path.exists(args.features):
        print(f"Error: Features file '{args.features}' not found.")
        sys.exit(1)

    # Load Data
    print("Loading data...")
    with open(args.split, 'rb') as f:
        split = pickle.load(f)
    with open(args.features, 'rb') as f:
        features_cache = pickle.load(f)

    # Load Model
    model_data = load_model(args.model)
    best_thresh = model_data.get('best_thresh', 0.5)
    print(f"Model loaded. Using threshold: {best_thresh:.4f}")

    # Model-based slug for plots
    base_model_name = os.path.splitext(os.path.basename(args.model))[0]

    # Original splits (paths as stored in split/features_cache)
    train_human = split.get('train_human', [])
    val_human = split.get('val_human', [])
    test_human = split.get('test_human', [])
    val_bots = split.get('val_bots', [])
    test_bots = split.get('test_bots', [])

    # ORIG MODE: behave as before (optionally skip printing/orig plots)
    if args.mode == "orig" and not args.skip_orig_sets:
        # Validation Set (collect sequences for the 3-panel plot)
        val_h_scores, val_h_labels, val_h_seqs = run_inference(
            model_data, features_cache, val_human, False,
            "Validation (Human)", collect_seqs=True
        )
        val_b_scores, val_b_labels, val_b_seqs = run_inference(
            model_data, features_cache, val_bots, True,
            "Validation (Bot)", collect_seqs=True
        )

        all_val_scores = val_h_scores + val_b_scores
        all_val_labels = val_h_labels + val_b_labels

        # Test Set
        test_h_scores, test_h_labels = run_inference(
            model_data, features_cache, test_human, False, "Test (Human)"
        )
        test_b_scores, test_b_labels = run_inference(
            model_data, features_cache, test_bots, True, "Test (Bot)"
        )

        all_test_scores = test_h_scores + test_b_scores
        all_test_labels = test_h_labels + test_b_labels

        # Calculate Metrics + Plots
        print("\n" + "=" * 40)
        print("RESULTS (Original Splits)")
        print("=" * 40)

        val_f1 = eval_and_report(
            all_val_scores, all_val_labels, best_thresh,
            subset_name="Validation", do_plot=False
        )
        test_f1 = eval_and_report(
            all_test_scores, all_test_labels, best_thresh,
            subset_name="Test", do_plot=False
        )

        # 3-panel results plot (using validation data)
        plot_results(
            val_h_seqs, val_b_seqs,
            all_val_scores, all_val_labels,
            val_h_scores, val_b_scores,
            best_thresh, val_f1,
            fname_slug=f"{base_model_name}_orig",
            title_prefix=f"{base_model_name} (orig)"
        )

        # Combined confusion matrices plot
        print("Generating combined confusion matrices plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        cm_val = confusion_matrix(all_val_labels, [1 if s >= best_thresh else 0 for s in all_val_scores])
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title(f"Validation Set (F1={val_f1:.4f})")
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticklabels(['Human', 'Bot'])
        axes[0].set_yticklabels(['Human', 'Bot'])

        cm_test = confusion_matrix(all_test_labels, [1 if s >= best_thresh else 0 for s in all_test_scores])
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
        axes[1].set_title(f"Test Set (F1={test_f1:.4f})")
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_xticklabels(['Human', 'Bot'])
        axes[1].set_yticklabels(['Human', 'Bot'])

        plt.tight_layout()
        orig_plot_path = os.path.join(PLOTS_DIR, f"{base_model_name}_orig_confusion.png")
        plt.savefig(orig_plot_path)
        plt.close()
        print(f"Original splits confusion plot saved to {orig_plot_path}")

    # ALL_NON_TRAIN MODE: evaluate all files not in train set (using split/feature keys)
    if args.mode == "all_non_train":
        print("\n" + "=" * 40)
        print("Evaluating ALL NON-TRAIN FILES (based on split/features_cache)")
        print("=" * 40)

        train_human_set = set(train_human)
        all_files = list(features_cache.keys())

        known_human = set(train_human) | set(val_human) | set(test_human)
        known_bot = set(val_bots) | set(test_bots)

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
            "All Non-Train (Human)", collect_seqs=True
        )
        nt_b_scores, nt_b_labels, nt_b_seqs = run_inference(
            model_data, features_cache, non_train_bot_files, True,
            "All Non-Train (Bot)", collect_seqs=True
        )

        all_nt_scores = nt_h_scores + nt_b_scores
        all_nt_labels = nt_h_labels + nt_b_labels

        nt_f1 = eval_and_report(
            all_nt_scores, all_nt_labels, best_thresh,
            subset_name="All Non-Train", do_plot=True,
            plot_prefix=f"{base_model_name}_all_non_train"
        )

        plot_results(
            nt_h_seqs, nt_b_seqs,
            all_nt_scores, all_nt_labels,
            nt_h_scores, nt_b_scores,
            best_thresh, nt_f1,
            fname_slug=f"{base_model_name}_all_non_train",
            title_prefix=f"{base_model_name} (all_non_train)"
        )

    # ALL_OTHER_FILES MODE: recursively list pre-processing txt files, exclude orig splits by filename,
    # then parse + feature-extract on the fly using the same logic as training.
    # ALL_OTHER_FILES MODE: recursively list pre-processing txt files, exclude orig splits by filename,
    # then parse + feature-extract on the fly using the same logic as training.
    if args.mode == "all_other_files":
        print("\n" + "=" * 40)
        print("Evaluating ALL OTHER RAW TXT FILES (pre-processing, excluding orig train/val/test by filename)")
        print("=" * 40)

        data_root_abs = os.path.abspath(args.data_root)
        print(f"[1/4] Walking data_root={data_root_abs} ...")
        all_fs_files = list_all_files_recursive(data_root_abs)
        print(f"    Total files found on filesystem: {len(all_fs_files)}")

        # Only .txt files
        print("[2/4] Filtering for .txt files...")
        txt_files = [p for p in all_fs_files if p.lower().endswith(".txt")]
        print(f"    .txt files found: {len(txt_files)}")

        # Build exclusion set from filenames of all original (post-processed) split lists
        split_exclude_names = set(
            os.path.basename(f) for f in (
                list(train_human) +
                list(val_human) +
                list(test_human) +
                list(val_bots) +
                list(test_bots)
            )
        )

        # Filter out any .txt whose basename matches original splits
        print("[3/4] Excluding original split filenames...")
        txt_non_split = [
            p for p in txt_files
            if os.path.basename(p) not in split_exclude_names
        ]
        print(f"    .txt files after excluding original splits by filename: {len(txt_non_split)}")

        if not txt_non_split:
            print("No candidate 'other' txt files after exclusions.")
        else:
            # Build reference pool using the original train_human raw txt paths
            print("[4/4] Building reference pool from train_human files...")
            ref_dwells, ref_flights = create_reference_pool(train_human)
            print("    Reference pool built.")

            print("\n[Stage A] Computing feature sequences for 'other' txt files (multiprocessing)...")

            worker_args = [
                (p, ref_dwells, ref_flights, STEP_SIZE_HUMAN)
                for p in txt_non_split
            ]

            other_feature_seqs = {}
            num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
            print(f"Using {num_workers} worker processes.")

            with mp.Pool(processes=num_workers) as pool:
                for filepath, seq in tqdm(
                        pool.imap_unordered(_compute_features_wrapper, worker_args),
                        total=len(worker_args),
                        desc="Feature extraction",
                        unit="file"
                ):
                    if seq.size > 0:
                        other_feature_seqs[filepath] = seq

            print(f"    Other txt files with usable feature sequences: {len(other_feature_seqs)}")


if __name__ == "__main__":
    main()
