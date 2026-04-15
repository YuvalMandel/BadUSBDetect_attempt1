"""
BadUSBv0/BadUSB/HTM/htm_train.py
Train a single HTM-Combined model configuration.

This script loads the pre-computed feature windows and the data splits,
trains the HTM model, and saves the trained model, results, and plots.

Usage (from BadUSBv0/BadUSB/):
  python HTM/htm_train.py [--config config.json]
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

# --- Path Setup ---
_here = os.path.dirname(os.path.abspath(__file__))
_badusb_root = os.path.dirname(_here)
_project_root = os.path.dirname(os.path.dirname(_badusb_root))

for p in [_project_root, os.path.join(_project_root, "common"), os.path.join(_project_root, "htm_distance"), _here]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from htm_combined_common import (
    CombinedEncoder, get_file_combined_seq, apply_detection,
    make_anomaly_likelihood, WARMUP_STEPS,
)
from keystroke_features import RANDOM_SEED

# --- Configuration ---
SPLIT_JSON = os.path.join(_badusb_root, "data_split.json")
WINDOWS_CACHE = os.path.join(_here, "windows_cache.pkl")
MODELS_DIR = os.path.join(_badusb_root, "results", "HTM", "models")
PLOTS_DIR = os.path.join(_badusb_root, "results", "HTM", "plots")
RESULTS_DIR = os.path.join(_badusb_root, "results", "HTM", "results")

def plot_results(val_h_seqs, val_b_seqs, val_h_scores, val_b_scores,
                 all_val_seqs, all_val_labels, best_thresh, best_bacc,
                 fname_slug, title, warmup):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=9)
    
    ax = axes[0]
    n = min(3, len(val_h_seqs), len(val_b_seqs))
    for i in range(n):
        ax.plot(val_h_seqs[i], alpha=0.6, color='steelblue', label='Human' if i == 0 else '_nolegend_')
    for i in range(n):
        ax.plot(val_b_seqs[i], alpha=0.6, color='tomato', label='Bot' if i == 0 else '_nolegend_')
    ax.axhline(best_thresh, color='green', linestyle='--', linewidth=1.5, label=f'Thresh={best_thresh:.3f}')
    ax.axvline(warmup, color='gray', linestyle=':', linewidth=1, label=f'Warmup={warmup}')
    ax.set_title('Anomaly Score Over Time')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True)

    ax = axes[1]
    ax.hist(val_h_scores, bins=15, alpha=0.65, color='steelblue', label='Human')
    ax.hist(val_b_scores, bins=15, alpha=0.65, color='tomato', label='Bot')
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2, label=f'Thresh={best_thresh:.3f}')
    ax.set_title('Per-File Mean Anomaly Score')
    ax.set_xlabel('Mean anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ths = np.linspace(0, 1, 500)
    baccs = [
        balanced_accuracy_score(all_val_labels, apply_detection(all_val_seqs, 'first_crossing', t, warmup, labels=all_val_labels))
        for t in ths
    ]
    ax.plot(ths, baccs, color='blue', linewidth=2, label='Val Balanced Acc')
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2, label=f'Best={best_thresh:.3f}')
    ax.set_title('Balanced Accuracy vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_results.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Plot  -> {fpath}")

def plot_confusion(val_labels, val_preds, test_labels, test_preds, fname_slug, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=9)
    
    for ax, (labels, preds, subset) in zip(axes, [(val_labels, val_preds, "Validation Set"), (test_labels, test_preds, "Test Set")]):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(subset)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['Human', 'Bot'])
        ax.set_yticklabels(['Human', 'Bot'])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fpath = os.path.join(PLOTS_DIR, f"{fname_slug}_confusion.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Plot  -> {fpath}")

def main():
    parser = argparse.ArgumentParser(description="Train one HTM-Combined config")
    parser.add_argument("--config", help="Path to JSON config (optional)")
    args = parser.parse_args()

    for d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)
        config_idx = int(os.path.splitext(os.path.basename(args.config))[0].split("_")[-1])
    else:
        # Hardcoded default configuration if no config file is provided
        cfg = {
            "seed": 42,
            "sp_columnDimensions": 2048,
            "sp_potentialPct": 0.8,
            "sp_synPermActiveInc": 0.05,
            "sp_synPermConnected": 0.2,
            "sp_synPermInactiveDec": 0.0005,
            "sp_numActiveColumns": 40,
            "tm_cellsPerColumn": 16,
            "tm_activationThreshold": 13,
            "tm_initialPermanence": 0.21,
            "tm_connectedPermanence": 0.5,
            "tm_minThreshold": 10,
            "tm_maxNewSynapseCount": 20,
            "tm_permanenceIncrement": 0.1,
            "tm_permanenceDecrement": 0.1,
            "dwell_stats_enc_bits": 16, "dwell_stats_enc_w": 5,
            "flight_stats_enc_bits": 16, "flight_stats_enc_w": 5,
            "dist_stats_enc_bits": 16, "dist_stats_enc_w": 5,
            "dwell_scalar_enc_bits": 8, "dwell_scalar_enc_w": 7,
            "flight_scalar_enc_bits": 8, "flight_scalar_enc_w": 7,
            "dist_scalar_enc_bits": 8, "dist_scalar_enc_w": 7,
            "warmup_steps": 0,
            "al_period": 10,
        }
        config_idx = 0

    seed = cfg.get("seed", RANDOM_SEED)
    warmup = cfg.get("warmup_steps", WARMUP_STEPS)
    al_period = cfg.get("al_period", 10)

    random.seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    if not os.path.exists(WINDOWS_CACHE):
        print(f"ERROR: {WINDOWS_CACHE} not found. Run htm_prepare_data.py first.")
        sys.exit(1)
    with open(WINDOWS_CACHE, 'rb') as f:
        wcache = pickle.load(f)

    window_size = wcache['window_size']
    window_step = wcache['window_step']
    
    train_human = split['train']['humans']
    val_human = split['val']['humans']
    val_bots = split['val']['bots']
    test_human = split['test']['humans']
    test_bots = split['test']['bots']

    print("Loading training sequences from cache...")
    train_seqs = {fp: wcache['sequences'][fp] for fp in train_human if fp in wcache['sequences']}

    all_stats = [item[0] for seq in train_seqs.values() for item in seq]
    if not all_stats:
        print("ERROR: No training features extracted.")
        sys.exit(1)

    feat_arr = np.array(all_stats)
    min_v, max_v = np.min(feat_arr, axis=0), np.max(feat_arr, axis=0)
    range_v = max_v - min_v
    min_v -= 0.1 * np.where(range_v > 0, range_v, 0.1)
    max_v += 0.1 * np.where(range_v > 0, range_v, 0.1)

    print("Building HTM...")
    encoder = CombinedEncoder(min_v, max_v, **{k: v for k, v in cfg.items()
                                               if k.endswith('_enc_bits') or k.endswith('_enc_w')})
    input_width = encoder.total_bits
    n_columns = cfg.get('sp_columnDimensions', 2048)

    SP_KEYS = {
        'sp_potentialPct':       'potentialPct',
        'sp_synPermActiveInc':   'synPermActiveInc',
        'sp_synPermConnected':   'synPermConnected',
        'sp_synPermInactiveDec': 'synPermInactiveDec',
        'sp_numActiveColumns':   'numActiveColumnsPerInhArea',
    }
    TM_KEYS = {
        'tm_cellsPerColumn':        'cellsPerColumn',
        'tm_activationThreshold':   'activationThreshold',
        'tm_initialPermanence':     'initialPermanence',
        'tm_connectedPermanence':   'connectedPermanence',
        'tm_minThreshold':          'minThreshold',
        'tm_maxNewSynapseCount':    'maxNewSynapseCount',
        'tm_permanenceIncrement':   'permanenceIncrement',
        'tm_permanenceDecrement':   'permanenceDecrement',
    }
    sp = SpatialPooler(
        inputDimensions=(input_width,),
        columnDimensions=(n_columns,),
        globalInhibition=True,
        localAreaDensity=0,   # mutex with numActiveColumnsPerInhArea; default is 0.05 which conflicts
        seed=seed,
        **{v: cfg[k] for k, v in SP_KEYS.items() if k in cfg},
    )
    tm = TemporalMemory(
        columnDimensions=(n_columns,),
        seed=seed,
        **{v: cfg[k] for k, v in TM_KEYS.items() if k in cfg},
    )
    active_columns = SDR(sp.getColumnDimensions())
    al = make_anomaly_likelihood(al_period)

    print("Training...")
    shuffled_train = list(train_seqs.keys())
    random.shuffle(shuffled_train)
    for fp in tqdm(shuffled_train, desc="Train"):
        seq = train_seqs[fp]
        tm.reset()
        for stats, key_idx, dwell, flight, dist in seq:
            enc_sdr = SDR(input_width); enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, True, active_columns)
            tm.compute(active_columns, learn=True)
            al.compute(float(tm.anomaly))

    al_live = make_anomaly_likelihood(al_period)
    for fp in tqdm(shuffled_train, desc="Calibrate Live AL"):
        seq = train_seqs[fp]
        tm.reset()
        for stats, key_idx, dwell, flight, dist in seq:
            enc_sdr = SDR(input_width); enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, False, active_columns)
            tm.compute(active_columns, learn=False)
            al_live.compute(float(tm.anomaly))
    al_live_bytes = pickle.dumps(al_live)

    def get_scores(file_list, is_bot):
        scores, labels, seqs = [], [], []
        for fp in file_list:
            seq = wcache['sequences'].get(fp)
            if not seq: continue
            tm.reset()
            raw = []
            for stats, key_idx, dwell, flight, dist in seq:
                enc_sdr = SDR(input_width); enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(al.compute(float(tm.anomaly)))
            valid = raw[warmup:]
            scores.append(float(np.mean(valid)) if valid else 0.0)
            labels.append(1 if is_bot else 0)
            seqs.append(raw)
        return scores, labels, seqs

    print("Validating...")
    val_h_sc, val_h_lb, val_h_sq = get_scores(val_human, False)
    val_b_sc, val_b_lb, val_b_sq = get_scores(val_bots, True)
    all_val_seqs = val_h_sq + val_b_sq
    all_val_labels = val_h_lb + val_b_lb

    best_bacc, best_thresh = 0.0, 0.0
    for th in np.linspace(0, 1, 200):
        preds = apply_detection(all_val_seqs, 'first_crossing', th, warmup, labels=all_val_labels)
        bacc = balanced_accuracy_score(all_val_labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, th
    
    val_preds = apply_detection(all_val_seqs, 'first_crossing', best_thresh, warmup, labels=all_val_labels)
    best_f1 = f1_score(all_val_labels, val_preds, zero_division=0)

    print("Testing...")
    test_h_sc, test_h_lb, test_h_sq = get_scores(test_human, False)
    test_b_sc, test_b_lb, test_b_sq = get_scores(test_bots, True)
    all_test_seqs = test_h_sq + test_b_sq
    all_test_labels = test_h_lb + test_b_lb
    test_preds = apply_detection(all_test_seqs, 'first_crossing', best_thresh, warmup, labels=all_test_labels)
    test_f1 = f1_score(all_test_labels, test_preds, zero_division=0)

    print(f"\nVal BAcc={best_bacc:.4f} (thresh={best_thresh:.4f}) | Val F1={best_f1:.4f} | Test F1={test_f1:.4f}")
    
    fname_slug = f"htm_model_{config_idx:04d}_vf1{best_f1:.4f}_tf1{test_f1:.4f}"
    model_path = os.path.join(MODELS_DIR, f"{fname_slug}.pkl")
    with open(model_path, 'wb') as fh:
        pickle.dump({
            "sp": sp, "tm": tm, "encoder": encoder, "al": al, "al_live_bytes": al_live_bytes,
            "input_width": input_width, "best_thresh": best_thresh, "live_thresh": best_thresh,
            "detection_mode": "first_crossing", "warmup_steps": warmup, "al_period": al_period,
            "window_size": window_size, "window_step": window_step,
            "ref_dwells": wcache['ref_dwells'], "ref_flights": wcache['ref_flights'], "ref_dists": wcache['ref_dists'],
            "config": cfg, "config_idx": config_idx, "val_bacc": float(best_bacc), "val_f1": float(best_f1), "test_f1": float(test_f1),
        }, fh)
    print(f"  Model -> {model_path}")
    
    result = {
        "config_idx": config_idx,
        "config": cfg,
        "val_bacc": float(best_bacc),
        "val_f1": float(best_f1),
        "test_f1": float(test_f1),
        "best_thresh": float(best_thresh),
        "live_thresh": float(best_thresh),
        "model_file": model_path,
    }
    results_path = os.path.join(RESULTS_DIR, f"htm_model_{config_idx:04d}.json")
    with open(results_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  Result-> {results_path}")

    title = f"HTM Config {config_idx:04d} | Val BAcc={best_bacc:.4f} F1={best_f1:.4f} | Test F1={test_f1:.4f}"
    plot_results(val_h_sq, val_b_sq, val_h_sc, val_b_sc, all_val_seqs, all_val_labels, best_thresh, best_bacc, fname_slug, title, warmup)
    plot_confusion(all_val_labels, val_preds, all_test_labels, test_preds, fname_slug, title)

if __name__ == "__main__":
    main()
