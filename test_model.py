import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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


def load_model(model_path):
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


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
    Reproduces the 3-panel plot:
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


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained HTM model on Test/Val sets or all non-train files."
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
        choices=["orig", "all_non_train"],
        default="all_non_train",
        help=(
            "Evaluation mode: "
            "'orig' uses original val/test splits (default). "
            "'all_non_train' evaluates on all files that are not in the original train set."
        )
    )
    parser.add_argument(
        "--skip_orig_sets",
        action="store_true",
        help="If set, skip evaluation and plots for original val/test sets."
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

    # Original splits
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

        # Combined confusion matrices plot (like original script)
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

    # ALL_NON_TRAIN MODE: evaluate all files not in train set
    if args.mode == "all_non_train":
        print("\n" + "=" * 40)
        print("Evaluating ALL NON-TRAIN FILES")
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

        # Collect sequences here as well so we can reuse the same 3-panel plot
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

        # 3-panel results plot for all non-train
        plot_results(
            nt_h_seqs, nt_b_seqs,
            all_nt_scores, all_nt_labels,
            nt_h_scores, nt_b_scores,
            best_thresh, nt_f1,
            fname_slug=f"{base_model_name}_all_non_train",
            title_prefix=f"{base_model_name} (all_non_train)"
        )

    if args.mode == "orig" and args.skip_orig_sets:
        print("Mode is 'orig' but --skip_orig_sets is set; no evaluation was performed.")


if __name__ == "__main__":
    main()
