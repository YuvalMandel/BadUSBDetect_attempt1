"""
htm_distance/htm_distance_train_single.py
Train one HTM-distance configuration. Designed for SLURM array jobs.

Input per timestep: (key_idx 0-94, dwell_ms, flight_ms, qwerty_dist)
No windowing, no statistical features — raw keystroke events fed directly.

Usage:
  python htm_distance/htm_distance_train_single.py --config dist_configs/config_0023.json

Outputs (all relative to project root):
  dist_models/<slug>.pkl          — SP + TM + encoder + threshold
  dist_plots/<slug>_results.png   — anomaly over time / score dist / F1 curve
  dist_plots/<slug>_confusion.png — val + test confusion matrices
  dist_results/<slug>.json        — scalar metrics + full config
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
import json
import pickle
import random

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

from common.keystroke_features import RANDOM_SEED
from htm_distance_common import (
    DistanceEncoder, apply_detection, WARMUP_STEPS,
)

# ── Output directories ────────────────────────────────────────────
SPLIT_FILE  = "split.pkl"
CACHE_FILE  = "dist_cache.pkl"
MODELS_DIR  = "dist_models"
PLOTS_DIR   = "dist_plots"
RESULTS_DIR = "dist_results"

for _d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ── Slug helpers ──────────────────────────────────────────────────
def _base_slug(cfg, idx):
    return (
        f"dist{idx:04d}"
        f"_sp{cfg['sp_numActiveColumns']}"
        f"_enc{cfg['enc_bits_per_feature']}w{cfg['enc_w']}"
        f"_tm{cfg['tm_cellsPerColumn']}"
        f"_act{cfg['tm_activationThreshold']}"
        f"_wu{cfg.get('warmup_steps', WARMUP_STEPS)}"
    )


def _full_slug(base, val_f1, test_f1):
    return f"{base}_vf1{val_f1:.4f}_tf1{test_f1:.4f}"


def _plot_title(cfg, idx, val_f1, test_f1):
    return (
        f"Dist-HTM {idx:04d}  "
        f"SP: act={cfg['sp_numActiveColumns']} pct={cfg['sp_potentialPct']} "
        f"synInc={cfg['sp_synPermActiveInc']}\n"
        f"TM: cells={cfg['tm_cellsPerColumn']} actThr={cfg['tm_activationThreshold']} "
        f"minThr={cfg['tm_minThreshold']} newSyn={cfg['tm_maxNewSynapseCount']}\n"
        f"Enc: bits={cfg['enc_bits_per_feature']} w={cfg['enc_w']}  "
        f"warmup={cfg.get('warmup_steps', WARMUP_STEPS)}  |  "
        f"Val F1={val_f1:.4f}   Test F1={test_f1:.4f}"
    )


# ── Plots ─────────────────────────────────────────────────────────
def plot_results(val_h_seqs, val_b_seqs,
                 val_h_scores, val_b_scores,
                 all_val_seqs, all_val_labels,
                 best_thresh, val_f1,
                 fname_slug, title, warmup):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=7)

    # Panel 1 — anomaly score over time
    ax = axes[0]
    n = min(3, len(val_h_seqs), len(val_b_seqs))
    for i in range(n):
        ax.plot(val_h_seqs[i], alpha=0.6, color='steelblue',
                label='Human' if i == 0 else '_nolegend_')
    for i in range(n):
        ax.plot(val_b_seqs[i], alpha=0.6, color='tomato',
                label='Bot' if i == 0 else '_nolegend_')
    ax.axhline(best_thresh, color='green', linestyle='--', linewidth=1.5,
               label=f'Thresh={best_thresh:.3f}')
    ax.axvline(warmup, color='gray', linestyle=':', linewidth=1,
               label=f'Warmup={warmup}')
    ax.set_title('Anomaly Score Over Time (keystroke steps)\nsample val files')
    ax.set_xlabel('Keystroke index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True)

    # Panel 2 — per-file mean score distribution
    ax = axes[1]
    ax.hist(val_h_scores, bins=15, alpha=0.65, color='steelblue', label='Human')
    ax.hist(val_b_scores, bins=15, alpha=0.65, color='tomato',    label='Bot')
    ax.axvline(best_thresh, color='green', linestyle='--', linewidth=2,
               label=f'Thresh={best_thresh:.3f}')
    ax.set_title('Per-File Mean Anomaly Score\n(validation set)')
    ax.set_xlabel('Mean anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True)

    # Panel 3 — F1 vs threshold
    ax = axes[2]
    ths = np.linspace(0, 1, 200)
    f1s = [
        f1_score(all_val_labels,
                 apply_detection(all_val_seqs, 'first_crossing', t, warmup),
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


def plot_confusion(val_labels, val_preds,
                   test_labels, test_preds,
                   fname_slug, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=7)
    for ax, (labels, preds, subset) in zip(axes, [
        (val_labels,  val_preds,  "Validation Set"),
        (test_labels, test_preds, "Test Set"),
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
    print(f"  Plot  → {fpath}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train one HTM-distance config")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config (e.g. dist_configs/config_0023.json)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    config_idx = int(
        os.path.splitext(os.path.basename(args.config))[0].split("_")[-1]
    )
    seed    = cfg.get("seed", RANDOM_SEED)
    warmup  = cfg.get("warmup_steps", WARMUP_STEPS)
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  HTM-Distance Config {config_idx:04d}   (seed={seed}  warmup={warmup})")
    for k, v in cfg.items():
        if k != "seed":
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────
    for path, label in ((SPLIT_FILE, "htm_prepare_data.py"),
                        (CACHE_FILE, "htm_distance_prepare_data.py")):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run {label} first.")
            sys.exit(1)

    with open(SPLIT_FILE, 'rb') as fh:
        split = pickle.load(fh)
    with open(CACHE_FILE, 'rb') as fh:
        dist_cache = pickle.load(fh)

    train_human = split['train_human']
    val_human   = split['val_human']
    test_human  = split['test_human']
    val_bots    = split['val_bots']
    test_bots   = split['test_bots']

    # ── Build HTM ─────────────────────────────────────────────────
    encoder    = DistanceEncoder(bits_per_feature=cfg['enc_bits_per_feature'],
                                 w=cfg['enc_w'])
    input_width = encoder.total_bits
    col_dims    = cfg['sp_columnDimensions']

    sp = SpatialPooler(
        inputDimensions           = (input_width,),
        columnDimensions          = (col_dims,),
        potentialPct              = cfg['sp_potentialPct'],
        globalInhibition          = True,
        numActiveColumnsPerInhArea= cfg['sp_numActiveColumns'],
        localAreaDensity          = 0.0,
        synPermActiveInc          = cfg['sp_synPermActiveInc'],
        synPermConnected          = cfg['sp_synPermConnected'],
        synPermInactiveDec        = cfg['sp_synPermInactiveDec'],
        boostStrength             = 0.0,
        seed                      = seed,
    )
    tm = TemporalMemory(
        columnDimensions     = (col_dims,),
        cellsPerColumn       = cfg['tm_cellsPerColumn'],
        activationThreshold  = cfg['tm_activationThreshold'],
        initialPermanence    = cfg['tm_initialPermanence'],
        connectedPermanence  = cfg['tm_connectedPermanence'],
        minThreshold         = cfg['tm_minThreshold'],
        maxNewSynapseCount   = cfg['tm_maxNewSynapseCount'],
        permanenceIncrement  = cfg['tm_permanenceIncrement'],
        permanenceDecrement  = cfg['tm_permanenceDecrement'],
        seed                 = seed,
    )
    active_columns = SDR(sp.getColumnDimensions())

    # ── Training ──────────────────────────────────────────────────
    print("Training...")
    shuffled_train = list(train_human)
    random.shuffle(shuffled_train)

    for fp in tqdm(shuffled_train, desc=f"Train dist{config_idx:04d}"):
        events = dist_cache.get(fp)
        if not events:
            continue
        tm.reset()
        for key_idx, dwell, flight, dist in events:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, True, active_columns)
            tm.compute(active_columns, learn=True)

    # ── Evaluation helper ─────────────────────────────────────────
    def get_scores(file_list, is_bot):
        scores, labels, seqs = [], [], []
        for fp in file_list:
            events = dist_cache.get(fp)
            if not events:
                continue
            tm.reset()
            raw = []
            for key_idx, dwell, flight, dist in events:
                enc_sdr       = SDR(input_width)
                enc_sdr.dense = encoder.encode(key_idx, dwell, flight, dist)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(tm.anomaly)
            if raw:
                valid = raw[warmup:] if len(raw) > warmup else raw
                scores.append(float(np.mean(valid)) if valid else 0.0)
                labels.append(1 if is_bot else 0)
                seqs.append(raw)
        return scores, labels, seqs

    # ── Validation + threshold sweep ──────────────────────────────
    print("Validating...")
    val_h_sc, val_h_lb, val_h_sq = get_scores(val_human, False)
    val_b_sc, val_b_lb, val_b_sq = get_scores(val_bots,  True)
    all_val_seqs   = val_h_sq + val_b_sq
    all_val_labels = val_h_lb + val_b_lb

    best_f1, best_thresh = 0.0, 0.0
    for th in np.linspace(0, 1, 100):
        preds = apply_detection(all_val_seqs, 'first_crossing', th, warmup)
        f1    = f1_score(all_val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, th

    # ── Test ──────────────────────────────────────────────────────
    print("Testing...")
    test_h_sc, test_h_lb, test_h_sq = get_scores(test_human, False)
    test_b_sc, test_b_lb, test_b_sq = get_scores(test_bots,  True)
    all_test_seqs   = test_h_sq + test_b_sq
    all_test_labels = test_h_lb + test_b_lb

    val_preds  = apply_detection(all_val_seqs,  'first_crossing', best_thresh, warmup)
    test_preds = apply_detection(all_test_seqs, 'first_crossing', best_thresh, warmup)
    test_f1    = f1_score(all_test_labels, test_preds, zero_division=0)

    print(f"\n  Val  F1 = {best_f1:.4f}  (thresh={best_thresh:.4f})")
    print(f"  Test F1 = {test_f1:.4f}")
    print(f"\n--- Test Classification Report ---")
    print(classification_report(all_test_labels, test_preds,
                                target_names=["Human", "Bot"]))

    # ── Save outputs ──────────────────────────────────────────────
    base_slug  = _base_slug(cfg, config_idx)
    fname_slug = _full_slug(base_slug, best_f1, test_f1)
    title      = _plot_title(cfg, config_idx, best_f1, test_f1)

    model_path = os.path.join(MODELS_DIR, f"{fname_slug}.pkl")
    with open(model_path, 'wb') as fh:
        pickle.dump({
            "sp": sp, "tm": tm, "encoder": encoder,
            "input_width": input_width,
            "best_thresh": best_thresh,
            "detection_mode": "first_crossing",
            "warmup_steps": warmup,
            "config": cfg,
            "config_idx": config_idx,
            "val_f1": float(best_f1),
            "test_f1": float(test_f1),
        }, fh)
    print(f"  Model → {model_path}")

    result = {
        "config_idx": config_idx,
        "config":     cfg,
        "val_f1":     float(best_f1),
        "test_f1":    float(test_f1),
        "best_thresh":float(best_thresh),
        "model_file": model_path,
    }
    results_path = os.path.join(RESULTS_DIR, f"{fname_slug}.json")
    with open(results_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  Result→ {results_path}")

    plot_results(val_h_sq, val_b_sq,
                 val_h_sc, val_b_sc,
                 all_val_seqs, all_val_labels,
                 best_thresh, best_f1,
                 fname_slug, title, warmup)
    plot_confusion(all_val_labels, val_preds,
                   all_test_labels, test_preds,
                   fname_slug, title)

    print(f"\n[Config {config_idx:04d}] DONE — "
          f"Val F1={best_f1:.4f}  Test F1={test_f1:.4f}\n")


if __name__ == "__main__":
    main()
