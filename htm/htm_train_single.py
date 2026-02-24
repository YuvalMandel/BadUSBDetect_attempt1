"""
htm/htm_train_single.py
Train one HTM configuration. Designed to be called by a SLURM array task.

Usage:
  python htm/htm_train_single.py --config configs/config_0023.json

Outputs (all relative to project root):
  models/  <slug>.pkl                — trained SP + TM + encoder + threshold
  plots/   <slug>_results.png        — anomaly-over-time / score dist / F1 curve
  plots/   <slug>_confusion.png      — val + test confusion matrices
  results/ <slug>.json               — scalar metrics + full config
"""

import os
import sys

# Allow imports from project root (common/, htm/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import pickle
import random

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for SLURM nodes
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

try:
    from htm.bindings.algorithms import AnomalyLikelihood
    _HAS_AL = True
except ImportError:
    _HAS_AL = False

from common.keystroke_features import RANDOM_SEED
from htm_common import MultiAttributeEncoder, WARMUP_STEPS, apply_detection

# ------------------------------------------------------------------
# Paths (relative to project root / CWD)
# ------------------------------------------------------------------
SPLIT_FILE  = "split.pkl"
CACHE_FILE  = "features_cache.pkl"
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"
RESULTS_DIR = "results"

for _d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ------------------------------------------------------------------
# Naming helpers
# ------------------------------------------------------------------
def _base_slug(cfg, config_idx):
    """Short, filesystem-safe key-param summary — no scores yet."""
    mode_tag = "_fc" if cfg.get("detection_mode", "mean") == "first_crossing" else ""
    al_tag   = "_al" if cfg.get("use_anomaly_likelihood", False) else ""
    return (
        f"cfg{config_idx:04d}"
        f"_sp{cfg['sp_numActiveColumns']}"
        f"_enc{cfg['enc_bits_per_feature']}w{cfg['enc_w']}"
        f"_tm{cfg['tm_cellsPerColumn']}"
        f"_act{cfg['tm_activationThreshold']}"
        f"{mode_tag}{al_tag}"
    )


def _full_slug(base, val_f1, test_f1):
    return f"{base}_vf1{val_f1:.4f}_tf1{test_f1:.4f}"


def _plot_title(cfg, config_idx, val_f1, test_f1):
    mode_str = cfg.get("detection_mode", "mean")
    al_str   = " AL" if cfg.get("use_anomaly_likelihood", False) else ""
    return (
        f"Config {config_idx:04d}  "
        f"SP: act={cfg['sp_numActiveColumns']} pct={cfg['sp_potentialPct']} "
        f"boost={cfg['sp_boostStrength']} synInc={cfg['sp_synPermActiveInc']}\n"
        f"TM: cells={cfg['tm_cellsPerColumn']} actThr={cfg['tm_activationThreshold']} "
        f"minThr={cfg['tm_minThreshold']} newSyn={cfg['tm_maxNewSynapseCount']} "
        f"permInc={cfg['tm_permanenceIncrement']}\n"
        f"Enc: bits={cfg['enc_bits_per_feature']} w={cfg['enc_w']}  "
        f"Det: {mode_str}{al_str}  |  "
        f"Val F1={val_f1:.4f}   Test F1={test_f1:.4f}"
    )


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_results(val_h_seqs, val_b_seqs,
                 all_val_scores, all_val_labels,
                 all_val_seqs,
                 val_h_scores, val_b_scores,
                 best_thresh, val_f1,
                 fname_slug, title,
                 detection_mode="mean", effective_warmup=WARMUP_STEPS):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
    if detection_mode == 'first_crossing':
        ax.axhline(best_thresh, color='green', linestyle='--', linewidth=1.5,
                   label=f'Cross-thresh={best_thresh:.3f}', alpha=0.8)
    ax.set_title('Anomaly Score Over Time\n(sample val files)')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend()
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

    # Panel 3 — F1 vs threshold  (uses apply_detection for both modes)
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train one HTM config")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config (e.g. configs/config_0023.json)")
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config) as fh:
        cfg = json.load(fh)
    # Derive index from filename: config_0023.json → 23
    config_idx = int(
        os.path.splitext(os.path.basename(args.config))[0].split("_")[-1]
    )
    seed = cfg.get("seed", RANDOM_SEED)
    random.seed(seed)
    np.random.seed(seed)

    # ---- Detection strategy ----
    detection_mode = cfg.get("detection_mode", "mean")
    use_al         = cfg.get("use_anomaly_likelihood", False)
    al_period      = cfg.get("al_learning_period", 20)

    if use_al and not _HAS_AL:
        print("WARNING: AnomalyLikelihood not available in this htm.core build; "
              "falling back to raw anomaly scores.", file=sys.stderr)
        use_al = False

    effective_warmup = max(WARMUP_STEPS, al_period) if use_al else WARMUP_STEPS

    print(f"\n{'='*60}")
    print(f"  HTM Config {config_idx:04d}   (seed={seed})")
    print(f"  detection_mode={detection_mode}  use_al={use_al}  warmup={effective_warmup}")
    for k, v in cfg.items():
        if k != "seed":
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # ---- Load split + feature cache ----
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: {SPLIT_FILE} not found. Run htm/htm_prepare_data.py first.")
        sys.exit(1)
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run htm/htm_prepare_data.py first.")
        sys.exit(1)

    with open(SPLIT_FILE, 'rb') as fh:
        split = pickle.load(fh)
    with open(CACHE_FILE, 'rb') as fh:
        file_features = pickle.load(fh)

    train_human = split['train_human']
    val_human   = split['val_human']
    test_human  = split['test_human']
    val_bots    = split['val_bots']
    test_bots   = split['test_bots']

    # ---- Feature ranges (train only, no leakage) ----
    train_feats = [file_features[f] for f in train_human if f in file_features]
    if not train_feats:
        print("ERROR: No training features in cache. Run htm/htm_prepare_data.py.")
        sys.exit(1)
    concat = np.vstack(train_feats)
    min_v  = np.min(concat, axis=0) - np.abs(np.min(concat, axis=0)) * 0.1
    max_v  = np.max(concat, axis=0) + np.abs(np.max(concat, axis=0)) * 0.1

    # ---- Build HTM from config ----
    col_dims    = cfg["sp_columnDimensions"]
    encoder     = MultiAttributeEncoder(14, min_v, max_v,
                                        bits_per_feature=cfg["enc_bits_per_feature"],
                                        w=cfg["enc_w"])
    input_width = encoder.total_bits

    sp = SpatialPooler(
        inputDimensions          =(input_width,),
        columnDimensions         =(col_dims,),
        potentialPct             =cfg["sp_potentialPct"],
        globalInhibition         =True,
        numActiveColumnsPerInhArea=cfg["sp_numActiveColumns"],
        localAreaDensity         =0.0,
        synPermActiveInc         =cfg["sp_synPermActiveInc"],
        synPermConnected         =cfg["sp_synPermConnected"],
        synPermInactiveDec       =cfg["sp_synPermInactiveDec"],
        boostStrength            =cfg["sp_boostStrength"],
        seed                     =seed,
    )
    tm = TemporalMemory(
        columnDimensions    =(col_dims,),
        cellsPerColumn      =cfg["tm_cellsPerColumn"],
        activationThreshold =cfg["tm_activationThreshold"],
        initialPermanence   =cfg["tm_initialPermanence"],
        connectedPermanence =cfg["tm_connectedPermanence"],
        minThreshold        =cfg["tm_minThreshold"],
        maxNewSynapseCount  =cfg["tm_maxNewSynapseCount"],
        permanenceIncrement =cfg["tm_permanenceIncrement"],
        permanenceDecrement =cfg["tm_permanenceDecrement"],
        seed                =seed,
    )
    active_columns = SDR(sp.getColumnDimensions())

    # ---- Train ----
    print("Training...")
    shuffled_train = list(train_human)
    random.shuffle(shuffled_train)

    for fp in tqdm(shuffled_train, desc=f"Train cfg{config_idx:04d}"):
        if fp not in file_features:
            continue
        tm.reset()
        for feats in file_features[fp]:
            enc_sdr = SDR(input_width)
            enc_sdr.dense = encoder.encode(feats)
            sp.compute(enc_sdr, True, active_columns)
            tm.compute(active_columns, learn=True)

    # ---- Evaluation helper ----
    def get_scores(file_list, is_bot):
        """
        Returns (mean_scores, labels, raw_seqs).
        mean_scores — post-warmup mean per file (for histogram plot).
        raw_seqs    — full per-window score sequence per file (for apply_detection).
        """
        scores, labels, seqs = [], [], []
        for fp in file_list:
            if fp not in file_features:
                continue
            tm.reset()
            al = AnomalyLikelihood(learningPeriod=al_period) if use_al else None
            raw = []
            for step, feats in enumerate(file_features[fp]):
                enc_sdr = SDR(input_width)
                enc_sdr.dense = encoder.encode(feats)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                score = tm.anomaly
                if al is not None:
                    score = al.anomalyProbability(score, score, step)
                raw.append(score)
            if raw:
                valid = raw[effective_warmup:] if len(raw) > effective_warmup else raw
                scores.append(float(np.mean(valid)) if valid else 0.0)
                labels.append(1 if is_bot else 0)
                seqs.append(raw)
        return scores, labels, seqs

    # ---- Validation + threshold tuning ----
    print("Validating...")
    val_h_sc, val_h_lb, val_h_sq = get_scores(val_human, False)
    val_b_sc, val_b_lb, val_b_sq = get_scores(val_bots,  True)
    all_val_sc  = val_h_sc + val_b_sc
    all_val_lb  = val_h_lb + val_b_lb
    all_val_seqs = val_h_sq + val_b_sq

    best_f1, best_thresh = 0.0, 0.0
    for th in np.linspace(0, 1, 100):
        preds = apply_detection(all_val_seqs, detection_mode, th, effective_warmup)
        f1    = f1_score(all_val_lb, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, th

    # ---- Test ----
    print("Testing...")
    test_h_sc, test_h_lb, test_h_sq = get_scores(test_human, False)
    test_b_sc, test_b_lb, test_b_sq = get_scores(test_bots,  True)
    all_test_sc   = test_h_sc + test_b_sc
    all_test_lb   = test_h_lb + test_b_lb
    all_test_seqs = test_h_sq + test_b_sq

    val_preds  = apply_detection(all_val_seqs,  detection_mode, best_thresh, effective_warmup)
    test_preds = apply_detection(all_test_seqs, detection_mode, best_thresh, effective_warmup)
    test_f1    = f1_score(all_test_lb, test_preds, zero_division=0)

    print(f"\n  Val  F1 = {best_f1:.4f}  (thresh={best_thresh:.4f}  mode={detection_mode})")
    print(f"  Test F1 = {test_f1:.4f}")
    print(f"\n--- Test Classification Report ---")
    print(classification_report(all_test_lb, test_preds,
                                 target_names=["Human", "Bot"]))

    # ---- Build filename slug ----
    base_slug  = _base_slug(cfg, config_idx)
    fname_slug = _full_slug(base_slug, best_f1, test_f1)
    title      = _plot_title(cfg, config_idx, best_f1, test_f1)

    # ---- Save model ----
    model_path = os.path.join(MODELS_DIR, f"{fname_slug}.pkl")
    with open(model_path, 'wb') as fh:
        pickle.dump({
            "sp":                    sp,
            "tm":                    tm,
            "encoder":               encoder,
            "input_width":           input_width,
            "best_thresh":           best_thresh,
            "detection_mode":        detection_mode,
            "use_anomaly_likelihood":use_al,
            "al_learning_period":    al_period,
            "effective_warmup":      effective_warmup,
            "config":                cfg,
            "config_idx":            config_idx,
            "val_f1":                float(best_f1),
            "test_f1":               float(test_f1),
        }, fh)
    print(f"  Model → {model_path}")

    # ---- Save results JSON ----
    result = {
        "config_idx":            config_idx,
        "config":                cfg,
        "val_f1":                float(best_f1),
        "test_f1":               float(test_f1),
        "best_thresh":           float(best_thresh),
        "detection_mode":        detection_mode,
        "use_anomaly_likelihood":use_al,
        "model_file":            model_path,
    }
    results_path = os.path.join(RESULTS_DIR, f"{fname_slug}.json")
    with open(results_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  Result→ {results_path}")

    # ---- Plots ----
    plot_results(val_h_sq, val_b_sq,
                 all_val_sc, all_val_lb,
                 all_val_seqs,
                 val_h_sc, val_b_sc,
                 best_thresh, best_f1,
                 fname_slug, title,
                 detection_mode=detection_mode,
                 effective_warmup=effective_warmup)
    plot_confusion(all_val_lb, val_preds,
                   all_test_lb, test_preds,
                   fname_slug, title)

    print(f"\n[Config {config_idx:04d}] DONE — "
          f"Val F1={best_f1:.4f}  Test F1={test_f1:.4f}\n")


if __name__ == "__main__":
    main()
