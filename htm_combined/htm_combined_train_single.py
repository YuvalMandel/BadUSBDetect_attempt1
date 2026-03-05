"""
htm_combined/htm_combined_train_single.py
Train one HTM-Combined configuration.  Designed for SLURM array jobs.

Each HTM timestep is one sliding window of printable keystrokes, encoded as:
  - 21-dim statistical features over the whole window (dwell, flight, QWERTY dist)
  - Last-keystroke identity: key type (one-hot, 95 bits) + dwell + flight + dist

See htm_combined_common.py for the full SDR layout.

Usage (from project root):
  python htm_combined/htm_combined_train_single.py \\
      --config hc_configs/config_0000.json

Outputs (all relative to project root):
  hc_models/<slug>.pkl          -- SP + TM + encoder + AL + ref_pool + threshold
  hc_plots/<slug>_results.png   -- anomaly over time / score dist / BAcc curve
  hc_plots/<slug>_confusion.png -- val + test confusion matrices
  hc_results/<slug>.json        -- scalar metrics + full config
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
import json
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from tqdm import tqdm

try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
except ImportError:
    print("ERROR: htm.core not installed.", file=sys.stderr)
    sys.exit(1)

from common.keystroke_features import RANDOM_SEED
from htm_combined_common import (
    CombinedEncoder, create_reference_pool,
    get_file_combined_seq, apply_detection,
    make_anomaly_likelihood, WARMUP_STEPS,
)

# ── Output directories ────────────────────────────────────────────────────────
SPLIT_FILE   = "split.pkl"
CACHE_FILE   = "stats_cache.pkl"
DIST_CACHE   = "dist_cache.pkl"    # fallback (same format)
WINDOWS_DIR  = "windows_cache"     # pre-computed per-window feature sequences
MODELS_DIR   = "hc_models"
PLOTS_DIR    = "hc_plots"
RESULTS_DIR  = "hc_results"

for _d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible encoder parameter extraction
#   Round 1: shared enc_bits_per_feature / enc_w
#   Round 2: split stats_enc_bits/stats_enc_w + scalar_enc_bits/scalar_enc_w
#   Round 3: fully independent 6-pair system (dwell/flight/dist × stats/scalar)
# ──────────────────────────────────────────────────────────────────────────────
def _get_enc_params(cfg: dict) -> dict:
    """Return the 12 encoder hyperparams for any config generation."""
    if "dwell_stats_enc_bits" in cfg:
        # Round 3: fully per-channel / per-scalar
        return {k: cfg[k] for k in (
            "dwell_stats_enc_bits",  "dwell_stats_enc_w",
            "flight_stats_enc_bits", "flight_stats_enc_w",
            "dist_stats_enc_bits",   "dist_stats_enc_w",
            "dwell_scalar_enc_bits", "dwell_scalar_enc_w",
            "flight_scalar_enc_bits","flight_scalar_enc_w",
            "dist_scalar_enc_bits",  "dist_scalar_enc_w",
        )}
    elif "stats_enc_bits" in cfg:
        # Round 2: split stats/scalar (same value across channels)
        sb, sw = cfg["stats_enc_bits"], cfg["stats_enc_w"]
        kb, kw = cfg["scalar_enc_bits"], cfg["scalar_enc_w"]
        return dict(
            dwell_stats_enc_bits=sb,  dwell_stats_enc_w=sw,
            flight_stats_enc_bits=sb, flight_stats_enc_w=sw,
            dist_stats_enc_bits=sb,   dist_stats_enc_w=sw,
            dwell_scalar_enc_bits=kb, dwell_scalar_enc_w=kw,
            flight_scalar_enc_bits=kb,flight_scalar_enc_w=kw,
            dist_scalar_enc_bits=kb,  dist_scalar_enc_w=kw,
        )
    else:
        # Round 1: single shared encoder for all 24 parameters
        b, w = cfg["enc_bits_per_feature"], cfg["enc_w"]
        return dict(
            dwell_stats_enc_bits=b,  dwell_stats_enc_w=w,
            flight_stats_enc_bits=b, flight_stats_enc_w=w,
            dist_stats_enc_bits=b,   dist_stats_enc_w=w,
            dwell_scalar_enc_bits=b, dwell_scalar_enc_w=w,
            flight_scalar_enc_bits=b,flight_scalar_enc_w=w,
            dist_scalar_enc_bits=b,  dist_scalar_enc_w=w,
        )


# ── Slug / title helpers ──────────────────────────────────────────────────────
def _base_slug(cfg, idx):
    ep = _get_enc_params(cfg)
    return (
        f"hc{idx:04d}"
        f"_sp{cfg['sp_numActiveColumns']}"
        f"_d{ep['dwell_stats_enc_bits']}w{ep['dwell_stats_enc_w']}"
        f"_f{ep['flight_stats_enc_bits']}w{ep['flight_stats_enc_w']}"
        f"_q{ep['dist_stats_enc_bits']}w{ep['dist_stats_enc_w']}"
        f"_dk{ep['dwell_scalar_enc_bits']}w{ep['dwell_scalar_enc_w']}"
        f"_fk{ep['flight_scalar_enc_bits']}w{ep['flight_scalar_enc_w']}"
        f"_qk{ep['dist_scalar_enc_bits']}w{ep['dist_scalar_enc_w']}"
        f"_ws{cfg['window_size']}s{cfg.get('window_step', 1)}"
        f"_tm{cfg['tm_cellsPerColumn']}"
        f"_act{cfg['tm_activationThreshold']}"
        f"_wu{cfg.get('warmup_steps', 0)}"
        f"_al{cfg.get('al_period', 10)}"
    )


def _full_slug(base, val_f1, test_f1):
    return f"{base}_vf1{val_f1:.4f}_tf1{test_f1:.4f}"


def _plot_title(cfg, idx, val_bacc, val_f1, test_f1):
    ep = _get_enc_params(cfg)
    enc_str = (
        f"dS={ep['dwell_stats_enc_bits']}w{ep['dwell_stats_enc_w']} "
        f"fS={ep['flight_stats_enc_bits']}w{ep['flight_stats_enc_w']} "
        f"qS={ep['dist_stats_enc_bits']}w{ep['dist_stats_enc_w']}  "
        f"dK={ep['dwell_scalar_enc_bits']}w{ep['dwell_scalar_enc_w']} "
        f"fK={ep['flight_scalar_enc_bits']}w{ep['flight_scalar_enc_w']} "
        f"qK={ep['dist_scalar_enc_bits']}w{ep['dist_scalar_enc_w']}"
    )
    return (
        f"HTM-Combined {idx:04d}  "
        f"SP: act={cfg['sp_numActiveColumns']} pct={cfg['sp_potentialPct']} "
        f"synInc={cfg['sp_synPermActiveInc']}\n"
        f"TM: cells={cfg['tm_cellsPerColumn']} actThr={cfg['tm_activationThreshold']} "
        f"minThr={cfg['tm_minThreshold']} newSyn={cfg['tm_maxNewSynapseCount']}\n"
        f"{enc_str}  "
        f"win={cfg['window_size']}x{cfg.get('window_step', 1)}  "
        f"al={cfg.get('al_period', 10)}  warmup={cfg.get('warmup_steps', 0)}  |  "
        f"Val BAcc={val_bacc:.4f}  Val F1={val_f1:.4f}   Test F1={test_f1:.4f}"
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(val_h_seqs, val_b_seqs,
                 val_h_scores, val_b_scores,
                 all_val_seqs, all_val_labels,
                 best_thresh, best_bacc,
                 fname_slug, title, warmup):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=7)

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
    ax.set_title('Anomaly Score Over Time (window steps)\nsample val files')
    ax.set_xlabel('Window index')
    ax.set_ylabel('Anomaly score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True)

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

    ax = axes[2]
    ths = np.linspace(0, 1, 500)
    baccs = [
        balanced_accuracy_score(
            all_val_labels,
            apply_detection(all_val_seqs, 'first_crossing', t, warmup,
                            labels=all_val_labels))
        for t in ths
    ]
    ax.plot(ths, baccs, color='blue', linewidth=2, label='Val Balanced Acc')
    ax.axvline(best_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Best={best_thresh:.3f} (BAcc={best_bacc:.4f})')
    ax.set_title('Balanced Accuracy vs Threshold\n(maximized during training)')
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
    print(f"  Plot  -> {fpath}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train one HTM-Combined config")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config (e.g. hc_configs/config_0000.json)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    config_idx  = int(
        os.path.splitext(os.path.basename(args.config))[0].split("_")[-1]
    )
    seed        = cfg.get("seed", RANDOM_SEED)
    warmup      = cfg.get("warmup_steps", WARMUP_STEPS)
    al_period   = cfg.get("al_period", 10)
    window_size = cfg.get("window_size", 10)
    window_step = cfg.get("window_step", 1)

    random.seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  HTM-Combined Config {config_idx:04d}   "
          f"(seed={seed}  warmup={warmup}  al_period={al_period}  "
          f"window={window_size}x{window_step})")
    for k, v in cfg.items():
        if k != "seed":
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: {SPLIT_FILE} not found. Run htm/htm_prepare_data.py first.")
        sys.exit(1)

    # Accept stats_cache.pkl or dist_cache.pkl (identical format)
    cache_path = CACHE_FILE if os.path.exists(CACHE_FILE) else DIST_CACHE
    if not os.path.exists(cache_path):
        print(f"ERROR: neither {CACHE_FILE} nor {DIST_CACHE} found. "
              "Run htm_distance_stats/htm_distance_stats_prepare_data.py first.")
        sys.exit(1)
    if cache_path == DIST_CACHE:
        print(f"  Note: {CACHE_FILE} not found; using {DIST_CACHE} (same format).")

    with open(SPLIT_FILE, 'rb') as fh:
        split = pickle.load(fh)
    with open(cache_path, 'rb') as fh:
        cache = pickle.load(fh)

    train_human = split['train_human']
    val_human   = split['val_human']
    test_human  = split['test_human']
    val_bots    = split['val_bots']
    test_bots   = split['test_bots']

    # ── Try loading pre-computed windows cache ─────────────────────
    wcache = None
    windows_cache_path = os.path.join(
        WINDOWS_DIR, f"ws{window_size}s{window_step}.pkl")

    if os.path.exists(windows_cache_path):
        print(f"Loading windows cache: {windows_cache_path}")
        with open(windows_cache_path, 'rb') as fh:
            wcache = pickle.load(fh)
        ref_dwells  = wcache['ref_dwells']
        ref_flights = wcache['ref_flights']
        ref_dists   = wcache['ref_dists']
        print(f"  Ref pool: {len(ref_dwells)} dwell / "
              f"{len(ref_flights)} flight / {len(ref_dists)} dist windows")
    else:
        print(f"Windows cache not found at {windows_cache_path}.")
        print(f"  Hint: run htm_combined/htm_combined_prepare_windows.py first.")
        print(f"  Falling back to on-the-fly computation (slow)...")

    # ── Build reference pool (only when cache is unavailable) ─────
    if wcache is None:
        print("Building reference pool...")
        ref_dwells, ref_flights, ref_dists = create_reference_pool(
            train_human, cache, window_size=window_size, seed=seed)
        if not ref_dwells or not ref_flights or not ref_dists:
            print("ERROR: Could not build reference pool.")
            sys.exit(1)

    # ── Load / compute training feature sequences ──────────────────
    if wcache is not None:
        print("Loading training sequences from windows cache...")
        train_seqs = {fp: wcache['sequences'][fp]
                      for fp in train_human if fp in wcache['sequences']}
        print(f"  {len(train_seqs)}/{len(train_human)} training files in cache")
    else:
        print("Pre-computing training features (for encoder range)...")
        train_seqs = {}
        for fp in tqdm(train_human, desc="  Features"):
            events = cache.get(fp)
            if not events:
                continue
            seq = get_file_combined_seq(events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists)
            if seq:
                train_seqs[fp] = seq

    # Fit min/max for the stats block over all training windows
    all_stats = [item[0] for seq in train_seqs.values() for item in seq]
    if not all_stats:
        print(f"ERROR: No training features extracted. "
              f"window_size={window_size} may exceed all training files.")
        sys.exit(1)

    feat_arr = np.array(all_stats)    # shape [N, 21]
    min_v    = np.min(feat_arr, axis=0)
    max_v    = np.max(feat_arr, axis=0)
    range_v  = max_v - min_v
    # 10% padding so boundary values are encoded, not clipped
    min_v   -= 0.1 * np.where(range_v > 0, range_v, 0.1)
    max_v   += 0.1 * np.where(range_v > 0, range_v, 0.1)

    n_windows = sum(len(seq) for seq in train_seqs.values())
    print(f"  Training windows: {n_windows} from {len(train_seqs)} files")

    # ── Build HTM ─────────────────────────────────────────────────
    ep       = _get_enc_params(cfg)
    encoder  = CombinedEncoder(min_v, max_v, **ep)
    input_width = encoder.total_bits
    col_dims    = cfg['sp_columnDimensions']

    stats_total  = 7 * (ep['dwell_stats_enc_bits'] + ep['flight_stats_enc_bits']
                        + ep['dist_stats_enc_bits'])
    scalar_total = (ep['dwell_scalar_enc_bits'] + ep['flight_scalar_enc_bits']
                    + ep['dist_scalar_enc_bits'])
    print(f"  SDR: {input_width} total bits  "
          f"(7×[{ep['dwell_stats_enc_bits']}+{ep['flight_stats_enc_bits']}+"
          f"{ep['dist_stats_enc_bits']}]={stats_total} stats "
          f"+ 95 key + [{ep['dwell_scalar_enc_bits']}+{ep['flight_scalar_enc_bits']}+"
          f"{ep['dist_scalar_enc_bits']}]={scalar_total} scalars)")

    sp = SpatialPooler(
        inputDimensions           =(input_width,),
        columnDimensions          =(col_dims,),
        potentialPct              =cfg['sp_potentialPct'],
        globalInhibition          =True,
        numActiveColumnsPerInhArea=cfg['sp_numActiveColumns'],
        localAreaDensity          =0.0,
        synPermActiveInc          =cfg['sp_synPermActiveInc'],
        synPermConnected          =cfg['sp_synPermConnected'],
        synPermInactiveDec        =cfg['sp_synPermInactiveDec'],
        boostStrength             =0.0,
        seed                      =seed,
    )
    tm = TemporalMemory(
        columnDimensions     =(col_dims,),
        cellsPerColumn       =cfg['tm_cellsPerColumn'],
        activationThreshold  =cfg['tm_activationThreshold'],
        initialPermanence    =cfg['tm_initialPermanence'],
        connectedPermanence  =cfg['tm_connectedPermanence'],
        minThreshold         =cfg['tm_minThreshold'],
        maxNewSynapseCount   =cfg['tm_maxNewSynapseCount'],
        permanenceIncrement  =cfg['tm_permanenceIncrement'],
        permanenceDecrement  =cfg['tm_permanenceDecrement'],
        seed                 =seed,
    )
    active_columns = SDR(sp.getColumnDimensions())

    al = make_anomaly_likelihood(al_period)

    # ── Training ──────────────────────────────────────────────────
    print("Training...")
    shuffled_train = list(train_seqs.keys())
    random.shuffle(shuffled_train)

    for fp in tqdm(shuffled_train, desc=f"Train hc{config_idx:04d}"):
        seq = train_seqs[fp]
        tm.reset()
        for stats, key_idx, dwell, flight, dist in seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
            sp.compute(enc_sdr, True, active_columns)
            tm.compute(active_columns, learn=True)
            al.compute(float(tm.anomaly))   # train AL on human distribution

    # ── Evaluation helper ─────────────────────────────────────────
    def get_scores(file_list, is_bot):
        """Run SP+TM+AL on each file.  Returns (mean_scores, labels, score_seqs)."""
        scores, labels, seqs = [], [], []
        for fp in file_list:
            # Prefer pre-loaded windows cache; fall back to on-the-fly
            if wcache is not None:
                seq = wcache['sequences'].get(fp)
            else:
                events = cache.get(fp)
                if not events:
                    continue
                seq = get_file_combined_seq(events, window_size, window_step,
                                            ref_dwells, ref_flights, ref_dists)
            if not seq:
                continue
            tm.reset()
            raw = []
            for stats, key_idx, dwell, flight, dist in seq:
                enc_sdr       = SDR(input_width)
                enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(al.compute(float(tm.anomaly)))

            label = 1 if is_bot else 0
            valid = raw[warmup:]
            scores.append(float(np.mean(valid)) if valid else 0.0)
            labels.append(label)
            seqs.append(raw)
        return scores, labels, seqs

    # ── Validation + two-stage threshold sweep (maximise balanced accuracy) ──
    print("Validating...")
    val_h_sc, val_h_lb, val_h_sq = get_scores(val_human, False)
    val_b_sc, val_b_lb, val_b_sq = get_scores(val_bots,  True)
    all_val_seqs   = val_h_sq + val_b_sq
    all_val_labels = val_h_lb + val_b_lb

    # Stage 1: coarse sweep (200 points) -- maximise balanced accuracy
    best_bacc, best_thresh = 0.0, 0.0
    coarse_pts = np.linspace(0, 1, 200)
    for th in coarse_pts:
        preds = apply_detection(all_val_seqs, 'first_crossing', th, warmup,
                                labels=all_val_labels)
        bacc = balanced_accuracy_score(all_val_labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, th

    # Stage 2: fine sweep (1000 points) in ±2 coarse steps around best
    coarse_step = 1.0 / (len(coarse_pts) - 1)
    fine_lo = max(0.0, best_thresh - 2 * coarse_step)
    fine_hi = min(1.0, best_thresh + 2 * coarse_step)
    for th in np.linspace(fine_lo, fine_hi, 1000):
        preds = apply_detection(all_val_seqs, 'first_crossing', th, warmup,
                                labels=all_val_labels)
        bacc = balanced_accuracy_score(all_val_labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, th

    best_f1 = f1_score(all_val_labels,
                       apply_detection(all_val_seqs, 'first_crossing',
                                       best_thresh, warmup, labels=all_val_labels),
                       zero_division=0)

    # ── Test ──────────────────────────────────────────────────────
    print("Testing...")
    test_h_sc, test_h_lb, test_h_sq = get_scores(test_human, False)
    test_b_sc, test_b_lb, test_b_sq = get_scores(test_bots,  True)
    all_test_seqs   = test_h_sq + test_b_sq
    all_test_labels = test_h_lb + test_b_lb

    val_preds  = apply_detection(all_val_seqs,  'first_crossing', best_thresh,
                                 warmup, labels=all_val_labels)
    test_preds = apply_detection(all_test_seqs, 'first_crossing', best_thresh,
                                 warmup, labels=all_test_labels)
    test_f1    = f1_score(all_test_labels, test_preds, zero_division=0)

    print(f"\n  Val  Balanced Acc = {best_bacc:.4f}  (thresh={best_thresh:.4f})")
    print(f"  Val  F1 (at thresh) = {best_f1:.4f}")
    print(f"  Test F1 = {test_f1:.4f}")
    print(f"\n--- Test Classification Report ---")
    print(classification_report(all_test_labels, test_preds,
                                target_names=["Human", "Bot"]))

    # ── Bot detection step printout ────────────────────────────────
    print(f"\n--- Bot Detection Steps (test set, {len(test_b_sq)} bots, "
          f"warmup={warmup} windows, thresh={best_thresh:.4f}) ---")
    caught_steps, missed_bot_lens = [], []
    for i, (seq, label) in enumerate(zip(all_test_seqs, all_test_labels)):
        if label != 1:
            continue
        post = seq[warmup:]
        if not post:
            missed_bot_lens.append(len(seq))
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"NOT DETECTED  (short file: {len(seq)} windows ≤ warmup={warmup})")
            continue
        detect_step = next(
            (warmup + j for j, s in enumerate(post) if s >= best_thresh), -1)
        if detect_step >= 0:
            caught_steps.append(detect_step)
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"detected at window {detect_step:4d}  "
                  f"(score={seq[detect_step]:.3f}, file_windows={len(seq)})")
        else:
            missed_bot_lens.append(len(seq))
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"NOT DETECTED  (max_score={max(post):.3f} < {best_thresh:.4f}, "
                  f"file_windows={len(seq)})")

    mean_detect = float(np.mean(caught_steps)) if caught_steps else -1.0
    n_caught    = len(caught_steps)
    n_bots      = len(test_b_sq)
    det_str     = f"{mean_detect:.1f}" if caught_steps else "N/A"
    print(f"  Caught {n_caught}/{n_bots} bots  |  Mean detection window: {det_str}")
    b_sc_str = f"{float(np.mean(test_b_sc)):.4f}" if test_b_sc else "N/A"
    h_sc_str = f"{float(np.mean(test_h_sc)):.4f}" if test_h_sc else "N/A"
    print(f"  Mean bot score   (post-warmup): {b_sc_str}")
    print(f"  Mean human score (post-warmup): {h_sc_str}")

    # ── Save outputs ──────────────────────────────────────────────
    base_slug  = _base_slug(cfg, config_idx)
    fname_slug = _full_slug(base_slug, best_f1, test_f1)
    title      = _plot_title(cfg, config_idx, best_bacc, best_f1, test_f1)

    model_path = os.path.join(MODELS_DIR, f"{fname_slug}.pkl")
    with open(model_path, 'wb') as fh:
        pickle.dump({
            "sp":             sp,
            "tm":             tm,
            "encoder":        encoder,
            "al":             al,
            "input_width":    input_width,
            "best_thresh":    best_thresh,
            "detection_mode": "first_crossing",
            "warmup_steps":   warmup,
            "al_period":      al_period,
            "window_size":    window_size,
            "window_step":    window_step,
            # Reference pool for inference on new files
            "ref_dwells":     ref_dwells,
            "ref_flights":    ref_flights,
            "ref_dists":      ref_dists,
            "config":         cfg,
            "config_idx":     config_idx,
            "val_bacc":       float(best_bacc),
            "val_f1":         float(best_f1),
            "test_f1":        float(test_f1),
        }, fh)
    print(f"  Model -> {model_path}")

    result = {
        "config_idx":            config_idx,
        "config":                cfg,
        "val_bacc":              float(best_bacc),
        "val_f1":                float(best_f1),
        "test_f1":               float(test_f1),
        "best_thresh":           float(best_thresh),
        "mean_bot_detect_step":  mean_detect,
        "n_bots_caught":         n_caught,
        "n_bots_total":          n_bots,
        "mean_bot_score":        float(np.mean(test_b_sc)) if test_b_sc else 0.0,
        "mean_human_score":      float(np.mean(test_h_sc)) if test_h_sc else 0.0,
        "model_file":            model_path,
    }
    results_path = os.path.join(RESULTS_DIR, f"{fname_slug}.json")
    with open(results_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f"  Result-> {results_path}")

    plot_results(val_h_sq, val_b_sq,
                 val_h_sc, val_b_sc,
                 all_val_seqs, all_val_labels,
                 best_thresh, best_bacc,
                 fname_slug, title, warmup)
    plot_confusion(all_val_labels, val_preds,
                   all_test_labels, test_preds,
                   fname_slug, title)

    print(f"\n[Config {config_idx:04d}] DONE -- "
          f"Val BAcc={best_bacc:.4f}  Val F1={best_f1:.4f}  Test F1={test_f1:.4f}\n")


if __name__ == "__main__":
    main()
