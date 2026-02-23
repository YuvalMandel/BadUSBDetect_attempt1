"""
htm/htm_train_interactive.py
Interactive HTM training (single fixed default configuration).

  RETRAIN = False (default): loads the latest htm_model_*.pkl and skips training.
  RETRAIN = True:            trains from scratch, saves a new timestamped pkl.

Outputs (relative to project root):
  htm_model_YYYYMMDD_HHMMSS.pkl
  htm_results.png
  htm_confusion_matrices.png

Usage:
  python htm/htm_train_interactive.py
"""

import os
import sys

# Allow imports from project root (common/, htm/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import glob
import random
import datetime
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# HTM imports
try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
    from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
except ImportError:
    print("HTM libraries not found. Please install htm.core.")
    class SDR:
        def __init__(self, n): self.dense = np.zeros(n)
    class SpatialPooler:
        def __init__(self, **kwargs): pass
        def compute(self, input_sdr, learn, active): pass
        def getColumnDimensions(self): return (2048,)
    class TemporalMemory:
        def __init__(self, **kwargs): self.anomaly = 0.0
        def compute(self, active, learn): pass
        def reset(self): pass
    class AnomalyLikelihood: pass

from common.keystroke_features import (
    FOLDERS, RANDOM_SEED, DEFAULT_WINDOW_SIZE, DEFAULT_STEP_HUMAN, DEFAULT_STEP_BOT,
    NUM_REFERENCES, is_task1, get_person_id, parse_file, extract_features,
    create_reference_pool, process_file_worker,
)
from htm_common import MultiAttributeEncoder

# ==============================================================================
# CONFIGURATION
# ==============================================================================
WINDOW_SIZE     = DEFAULT_WINDOW_SIZE
STEP_SIZE_HUMAN = DEFAULT_STEP_HUMAN
STEP_SIZE_BOT   = DEFAULT_STEP_BOT
CACHE_FILE      = "features_cache.pkl"
RETRAIN         = False   # Set to True to force a new training run


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_htm_results(val_human_seqs, val_bot_seqs,
                     all_val_scores, all_val_labels,
                     val_human_scores, val_bot_scores,
                     best_thresh, best_f1):
    """Three-panel plot: anomaly-score time series, score distribution, F1 vs threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HTM Training Results", fontsize=12)

    # --- Panel 1: Anomaly score over time (sample files) ---
    ax = axes[0]
    n_samples = min(3, len(val_human_seqs), len(val_bot_seqs))
    for i in range(n_samples):
        ax.plot(val_human_seqs[i], alpha=0.7, color='steelblue',
                label='Human' if i == 0 else '_nolegend_')
    for i in range(n_samples):
        ax.plot(val_bot_seqs[i], alpha=0.7, color='tomato',
                label='Bot' if i == 0 else '_nolegend_')
    ax.set_title('Anomaly Score Over Time\n(sample val files)')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    # --- Panel 2: Per-file mean score distribution ---
    ax = axes[1]
    ax.hist(val_human_scores, bins=20, alpha=0.65, color='steelblue', label='Human')
    ax.hist(val_bot_scores,   bins=20, alpha=0.65, color='tomato',    label='Bot')
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2,
               label=f'Threshold={best_thresh:.3f}')
    ax.set_title('Per-File Mean Anomaly Score\n(validation set)')
    ax.set_xlabel('Mean anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    # --- Panel 3: F1 vs threshold ---
    ax = axes[2]
    thresholds = np.linspace(0, 1, 200)
    f1_scores = []
    for th in thresholds:
        preds = [1 if s >= th else 0 for s in all_val_scores]
        f1_scores.append(f1_score(all_val_labels, preds, zero_division=0))
    ax.plot(thresholds, f1_scores, color='blue', linewidth=2, label='Val F1')
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Best thresh={best_thresh:.3f}\n(F1={best_f1:.4f})')
    ax.set_title('F1 Score vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 score')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fname = "htm_results.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Results plot saved → {fname}")


def plot_confusion_matrices(val_labels, val_preds, test_labels, test_preds):
    """Side-by-side confusion matrices for validation and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HTM Confusion Matrices", fontsize=12)

    for ax, (labels, preds, title) in zip(axes, [
        (val_labels,  val_preds,  "Validation Set"),
        (test_labels, test_preds, "Test Set"),
    ]):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Human', 'Bot'])
        ax.set_yticklabels(['Human', 'Bot'])

        if title == "Test Set":
            print(f"\n--- Classification Report ({title}) ---")
            print(classification_report(labels, preds, target_names=['Human', 'Bot']))

    plt.tight_layout()
    fname = "htm_confusion_matrices.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices plot saved → {fname}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Find Files
    print("Locating files...")
    human_files = []
    for folder in FOLDERS["Humans"]:
        files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        human_files.extend([f for f in files if is_task1(f)])

    bot_files = []
    for folder in FOLDERS["Bots"]:
        bot_files.extend(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))

    print(f"Found {len(human_files)} Human files and {len(bot_files)} Bot files.")

    # 2. Split
    person_ids = sorted(set(get_person_id(f) for f in human_files))
    random.shuffle(person_ids)
    n = len(person_ids)
    n_val = 15
    n_train = n - n_val - (n - n_val) // 5   # ~80 % train, rest test

    train_persons = set(person_ids[:n_train])
    val_persons   = set(person_ids[n_train:n_train + n_val])
    test_persons  = set(person_ids[n_train + n_val:])

    train_human = [f for f in human_files if get_person_id(f) in train_persons]
    val_human   = [f for f in human_files if get_person_id(f) in val_persons]
    test_human  = [f for f in human_files if get_person_id(f) in test_persons]

    random.shuffle(bot_files)
    nb = len(bot_files)
    nb_val = max(1, nb // 5)   # ~20 % bots for val threshold-tuning

    val_bots  = bot_files[:nb_val]
    test_bots = bot_files[nb_val:]   # ALL remaining bots go to test

    print(f"Split: Train Human={len(train_human)}, Val Human={len(val_human)}, "
          f"Val Bot={len(val_bots)}, Test Bot={len(test_bots)}")

    # 3. Reference Pool
    ref_dwells, ref_flights = create_reference_pool(train_human)

    # 4. Pre-process Features (Multiprocessing with Incremental Caching)
    file_features = {}

    if os.path.exists(CACHE_FILE):
        print(f"Loading features from cache: {CACHE_FILE}")
        try:
            with open(CACHE_FILE, 'rb') as f:
                file_features = pickle.load(f)
            print(f"Loaded {len(file_features)} files from cache.")
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing...")
            file_features = {}

    # Build the full list of files we need, then only process the ones not cached
    all_files_needed = (
        [(f, STEP_SIZE_HUMAN, ref_dwells, ref_flights, WINDOW_SIZE)
         for f in train_human + val_human + test_human]
        + [(f, STEP_SIZE_BOT, ref_dwells, ref_flights, WINDOW_SIZE)
           for f in val_bots + test_bots]
    )
    missing = [args for args in all_files_needed if args[0] not in file_features]

    if missing:
        print(f"Extracting features for {len(missing)} files...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file_worker, args) for args in missing]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Processing files"):
                fpath, feats = future.result()
                if len(feats) > 0:
                    file_features[fpath] = feats

        print(f"Saving features to cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(file_features, f)
    else:
        print("All files found in cache, skipping extraction.")

    # Collect Train Features for Min/Max
    train_feats_list = []
    for f in train_human:
        if f in file_features:
            train_feats_list.append(file_features[f])

    if not train_feats_list:
        print("No training features extracted. Exiting.")
        return

    concat_feats = np.vstack(train_feats_list)
    min_vals = np.min(concat_feats, axis=0)
    max_vals = np.max(concat_feats, axis=0)

    # Add buffer
    min_vals -= (np.abs(min_vals) * 0.1)
    max_vals += (np.abs(max_vals) * 0.1)

    print("Feature ranges computed.")

    # 5. Load saved model or train from scratch
    saved_models = sorted(glob.glob("htm_model_*.pkl"), reverse=True)
    best_thresh = None

    if not RETRAIN and saved_models:
        model_path = saved_models[0]
        print(f"Loading model: {model_path}  (set RETRAIN=True to retrain)")
        with open(model_path, 'rb') as f:
            md = pickle.load(f)
        sp          = md['sp']
        tm          = md['tm']
        encoder     = md['encoder']
        input_width = md['input_width']
        best_thresh = md['best_thresh']
        print(f"  Loaded. Threshold={best_thresh:.4f}")
        active_columns = SDR(sp.getColumnDimensions())

    else:
        if not RETRAIN:
            print("No saved model found — training from scratch.")

        # 5a. Initialize HTM
        encoder = MultiAttributeEncoder(14, min_vals, max_vals,
                                        bits_per_feature=32, w=5)
        input_width = encoder.total_bits

        sp = SpatialPooler(
            inputDimensions=(input_width,),
            columnDimensions=(2048,),
            potentialPct=0.8,
            globalInhibition=True,
            numActiveColumnsPerInhArea=40,
            localAreaDensity=0.0,
            synPermActiveInc=0.05,
            synPermConnected=0.1,
            synPermInactiveDec=0.005,
            boostStrength=1.0,
            seed=42
        )

        tm = TemporalMemory(
            columnDimensions=(2048,),
            cellsPerColumn=32,
            activationThreshold=13,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=10,
            maxNewSynapseCount=20,
            permanenceIncrement=0.1,
            permanenceDecrement=0.1,
            seed=42
        )

        # 5b. Train
        print("Training HTM...")
        active_columns = SDR(sp.getColumnDimensions())

        random.shuffle(train_human)
        for f in tqdm(train_human, desc="Training"):
            if f not in file_features:
                continue
            seq = file_features[f]

            tm.reset()
            for feats in seq:
                dense_input = encoder.encode(feats)
                enc_sdr = SDR(input_width)
                enc_sdr.dense = dense_input

                sp.compute(enc_sdr, True, active_columns)
                tm.compute(active_columns, learn=True)

    # 7. Validation
    print("Validating...")

    def get_scores(file_list, is_bot):
        scores = []
        labels = []
        sequences = []  # full per-file anomaly score sequences (for plotting)
        for f in file_list:
            if f not in file_features:
                continue
            seq = file_features[f]

            tm.reset()
            file_scores = []
            for feats in seq:
                dense_input = encoder.encode(feats)
                enc_sdr = SDR(input_width)
                enc_sdr.dense = dense_input

                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)

                file_scores.append(tm.anomaly)

            if file_scores:
                # Skip warmup
                valid_scores = file_scores[5:] if len(file_scores) > 5 else file_scores
                scores.append(np.mean(valid_scores))
                labels.append(1 if is_bot else 0)
                sequences.append(file_scores)
        return scores, labels, sequences

    val_human_scores, val_human_labels, val_human_seqs = get_scores(val_human, False)
    val_bot_scores, val_bot_labels, val_bot_seqs       = get_scores(val_bots,  True)

    all_val_scores = val_human_scores + val_bot_scores
    all_val_labels = val_human_labels + val_bot_labels

    if best_thresh is None:
        # Threshold tuning (only when we just trained)
        best_f1 = 0
        best_thresh = 0
        for th in np.linspace(0, 1, 100):
            preds = [1 if s >= th else 0 for s in all_val_scores]
            f1 = f1_score(all_val_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th

        # Save model now that we have best_thresh
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"htm_model_{ts}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'sp': sp, 'tm': tm, 'encoder': encoder,
                         'input_width': input_width, 'best_thresh': best_thresh}, f)
        print(f"  Model saved → {model_path}")

    best_f1 = f1_score(all_val_labels,
                       [1 if s >= best_thresh else 0 for s in all_val_scores],
                       zero_division=0)
    print(f"Best Validation F1: {best_f1:.4f} at Threshold: {best_thresh:.4f}")

    # 8. Test
    print("Testing...")
    test_human_scores, test_human_labels, _ = get_scores(test_human, False)
    test_bot_scores, test_bot_labels, _     = get_scores(test_bots, True)

    all_test_scores = test_human_scores + test_bot_scores
    all_test_labels = test_human_labels + test_bot_labels

    val_preds  = [1 if s >= best_thresh else 0 for s in all_val_scores]
    test_preds = [1 if s >= best_thresh else 0 for s in all_test_scores]

    # 9. Plots
    print("\nGenerating plots...")
    plot_htm_results(
        val_human_seqs, val_bot_seqs,
        all_val_scores, all_val_labels,
        val_human_scores, val_bot_scores,
        best_thresh, best_f1,
    )
    plot_confusion_matrices(all_val_labels, val_preds, all_test_labels, test_preds)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
