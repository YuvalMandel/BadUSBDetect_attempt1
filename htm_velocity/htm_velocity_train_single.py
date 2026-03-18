"""
htm_velocity/htm_velocity_train_single.py
Train one HTM-Velocity configuration.  Designed for SLURM array jobs.

Extends HTM-Combined: each HTM timestep is encoded as
  - 21-dim statistical features over the keystroke window (dwell, flight, dist)
  - Last-keystroke identity: key type (one-hot 95) + dwell + flight + dist
  - 3 polynomial Fitts'-Law error channels: poly_mean, poly_med, poly_std

See htm_velocity_common.py for the full SDR layout.

Usage (from project root):
  python htm_velocity/htm_velocity_train_single.py \\
      --config hv_configs/config_0000.json

Outputs (all relative to project root):
  hv_models/<slug>.pkl          -- SP + TM + encoder + AL + ref_pool + threshold
  hv_plots/<slug>_results.png   -- anomaly over time / score dist / BAcc curve
  hv_plots/<slug>_confusion.png -- val + test confusion matrices
  hv_results/<slug>.json        -- scalar metrics + full config
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_htm_vel_dir = os.path.dirname(os.path.abspath(__file__))
if _htm_vel_dir not in sys.path:
    sys.path.insert(0, _htm_vel_dir)

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
from htm_combined_common import create_reference_pool, apply_detection, WARMUP_STEPS
from htm_velocity_common import (
    VelocityEncoder, get_file_velocity_seq, make_anomaly_likelihood,
)

# ── Output directories ────────────────────────────────────────────────────────
SPLIT_FILE    = "split.pkl"
CACHE_FILE    = "stats_cache.pkl"
DIST_CACHE    = "dist_cache.pkl"        # fallback (same format)
VEL_CACHE     = "velocity_cache.pkl"
WINDOWS_DIR   = "hv_windows_cache"
MODELS_DIR    = "hv_models"
PLOTS_DIR     = "hv_plots"
RESULTS_DIR   = "hv_results"

for _d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder parameter extraction  (supports all htm_combined rounds + velocity)
# ──────────────────────────────────────────────────────────────────────────────
def _get_combined_enc_params(cfg: dict) -> dict:
    """Return the 12 combined-encoder hyperparams (backward compat Rounds 1/2/3)."""
    if "dwell_stats_enc_bits" in cfg:
        return {k: cfg[k] for k in (
            "dwell_stats_enc_bits",  "dwell_stats_enc_w",
            "flight_stats_enc_bits", "flight_stats_enc_w",
            "dist_stats_enc_bits",   "dist_stats_enc_w",
            "dwell_scalar_enc_bits", "dwell_scalar_enc_w",
            "flight_scalar_enc_bits","flight_scalar_enc_w",
            "dist_scalar_enc_bits",  "dist_scalar_enc_w",
        )}
    elif "stats_enc_bits" in cfg:
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
        b, w = cfg["enc_bits_per_feature"], cfg["enc_w"]
        return dict(
            dwell_stats_enc_bits=b,  dwell_stats_enc_w=w,
            flight_stats_enc_bits=b, flight_stats_enc_w=w,
            dist_stats_enc_bits=b,   dist_stats_enc_w=w,
            dwell_scalar_enc_bits=b, dwell_scalar_enc_w=w,
            flight_scalar_enc_bits=b,flight_scalar_enc_w=w,
            dist_scalar_enc_bits=b,  dist_scalar_enc_w=w,
        )


def _get_enc_params(cfg: dict) -> dict:
    """Return all 18 velocity encoder hyperparams (12 combined + 6 poly)."""
    ep = _get_combined_enc_params(cfg)
    ep["poly_mean_enc_bits"] = cfg.get("poly_mean_enc_bits", 8)
    ep["poly_mean_enc_w"]    = cfg.get("poly_mean_enc_w",    3)
    ep["poly_med_enc_bits"]  = cfg.get("poly_med_enc_bits",  8)
    ep["poly_med_enc_w"]     = cfg.get("poly_med_enc_w",     3)
    ep["poly_std_enc_bits"]  = cfg.get("poly_std_enc_bits",  8)
    ep["poly_std_enc_w"]     = cfg.get("poly_std_enc_w",     3)
    return ep


# ── Slug / title helpers ──────────────────────────────────────────────────────
def _base_slug(cfg, idx):
    ep = _get_enc_params(cfg)
    return (
        f"hv{idx:04d}"
        f"_sp{cfg['sp_numActiveColumns']}"
        f"_d{ep['dwell_stats_enc_bits']}w{ep['dwell_stats_enc_w']}"
        f"_f{ep['flight_stats_enc_bits']}w{ep['flight_stats_enc_w']}"
        f"_q{ep['dist_stats_enc_bits']}w{ep['dist_stats_enc_w']}"
        f"_dk{ep['dwell_scalar_enc_bits']}w{ep['dwell_scalar_enc_w']}"
        f"_fk{ep['flight_scalar_enc_bits']}w{ep['flight_scalar_enc_w']}"
        f"_qk{ep['dist_scalar_enc_bits']}w{ep['dist_scalar_enc_w']}"
        f"_pm{ep['poly_mean_enc_bits']}w{ep['poly_mean_enc_w']}"
        f"_pd{ep['poly_med_enc_bits']}w{ep['poly_med_enc_w']}"
        f"_ps{ep['poly_std_enc_bits']}w{ep['poly_std_enc_w']}"
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
    return (
        f"HTM-Velocity {idx:04d}  "
        f"SP: act={cfg['sp_numActiveColumns']} pct={cfg['sp_potentialPct']}\n"
        f"TM: cells={cfg['tm_cellsPerColumn']} actThr={cfg['tm_activationThreshold']} "
        f"wu={cfg.get('warmup_steps',0)} al={cfg.get('al_period',10)}\n"
        f"dS={ep['dwell_stats_enc_bits']}w{ep['dwell_stats_enc_w']} "
        f"fS={ep['flight_stats_enc_bits']}w{ep['flight_stats_enc_w']} "
        f"qS={ep['dist_stats_enc_bits']}w{ep['dist_stats_enc_w']} | "
        f"dK={ep['dwell_scalar_enc_bits']}w{ep['dwell_scalar_enc_w']} "
        f"fK={ep['flight_scalar_enc_bits']}w{ep['flight_scalar_enc_w']} "
        f"qK={ep['dist_scalar_enc_bits']}w{ep['dist_scalar_enc_w']} | "
        f"pm={ep['poly_mean_enc_bits']}w{ep['poly_mean_enc_w']} "
        f"pd={ep['poly_med_enc_bits']}w{ep['poly_med_enc_w']} "
        f"ps={ep['poly_std_enc_bits']}w{ep['poly_std_enc_w']}  "
        f"win={cfg['window_size']}x{cfg.get('window_step',1)}  |  "
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
    ax.set_title('Anomaly Score Over Time (window steps)')
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
    ax.set_title('Per-File Mean Anomaly Score (validation set)')
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
    parser = argparse.ArgumentParser(description="Train one HTM-Velocity config")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config (e.g. hv_configs/config_0000.json)")
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
    print(f"  HTM-Velocity Config {config_idx:04d}   "
          f"(seed={seed}  warmup={warmup}  al_period={al_period}  "
          f"window={window_size}x{window_step})")
    for k, v in cfg.items():
        if k != "seed":
            print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────
    if not os.path.exists(SPLIT_FILE):
        print(f"ERROR: {SPLIT_FILE} not found.  Run htm/htm_prepare_data.py first.")
        sys.exit(1)

    cache_path = CACHE_FILE if os.path.exists(CACHE_FILE) else DIST_CACHE
    if not os.path.exists(cache_path):
        print(f"ERROR: neither {CACHE_FILE} nor {DIST_CACHE} found.")
        sys.exit(1)
    if cache_path == DIST_CACHE:
        print(f"  Note: {CACHE_FILE} not found; using {DIST_CACHE}.")

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
    windows_cache_path = os.path.join(WINDOWS_DIR, f"ws{window_size}s{window_step}.pkl")

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
        print(f"  Hint: run htm_velocity/htm_velocity_prepare_windows.py first.")
        print(f"  Falling back to on-the-fly computation (slow)...")

    # ── Load velocity cache for fallback mode ─────────────────────
    vel_cache = None
    if wcache is None:
        if not os.path.exists(VEL_CACHE):
            print(f"ERROR: {VEL_CACHE} not found.")
            print("  Run: python htm_velocity/htm_velocity_prepare_data.py")
            sys.exit(1)
        print(f"Loading velocity cache: {VEL_CACHE}")
        with open(VEL_CACHE, 'rb') as fh:
            vel_cache = pickle.load(fh)

    # ── Build reference pool (only when windows cache is unavailable) ─
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
            events     = cache.get(fp)
            vel_events = vel_cache.get(fp)
            if not events or not vel_events:
                continue
            seq = get_file_velocity_seq(events, vel_events, window_size, window_step,
                                        ref_dwells, ref_flights, ref_dists)
            if seq:
                train_seqs[fp] = seq

    # Fit min/max for the stats block over all training windows
    all_stats = [item[0] for seq in train_seqs.values() for item in seq]
    if not all_stats:
        print(f"ERROR: No training features extracted.  "
              f"window_size={window_size} may exceed all training files.")
        sys.exit(1)

    feat_arr = np.array(all_stats)    # shape [N, 21]
    min_v    = np.min(feat_arr, axis=0)
    max_v    = np.max(feat_arr, axis=0)
    range_v  = max_v - min_v
    min_v   -= 0.1 * np.where(range_v > 0, range_v, 0.1)
    max_v   += 0.1 * np.where(range_v > 0, range_v, 0.1)

    n_windows = sum(len(seq) for seq in train_seqs.values())
    print(f"  Training windows: {n_windows} from {len(train_seqs)} files")

    # ── Build HTM ─────────────────────────────────────────────────
    ep       = _get_enc_params(cfg)
    encoder  = VelocityEncoder(min_v, max_v, **ep)
    input_width = encoder.total_bits
    col_dims    = cfg['sp_columnDimensions']

    combined_bits = encoder._combined.total_bits
    poly_bits     = ep['poly_mean_enc_bits'] + ep['poly_med_enc_bits'] + ep['poly_std_enc_bits']
    print(f"  SDR: {input_width} total bits  "
          f"({combined_bits} combined + {poly_bits} poly errors)")

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
    al             = make_anomaly_likelihood(al_period)

    # ── Training ──────────────────────────────────────────────────
    print("Training...")
    shuffled_train = list(train_seqs.keys())
    random.shuffle(shuffled_train)

    for fp in tqdm(shuffled_train, desc=f"Train hv{config_idx:04d}"):
        seq = train_seqs[fp]
        tm.reset()
        for stats, key_idx, dwell, flight, dist, pm, pmd, ps in seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist, pm, pmd, ps)
            sp.compute(enc_sdr, True, active_columns)
            tm.compute(active_columns, learn=True)
            al.compute(float(tm.anomaly))

    # Calibrate live-mode AL on the fully-trained TM (learn=False)
    al_live = make_anomaly_likelihood(al_period)
    for fp in tqdm(shuffled_train, desc=f"Calibrate live AL hv{config_idx:04d}"):
        seq = train_seqs[fp]
        tm.reset()
        for stats, key_idx, dwell, flight, dist, pm, pmd, ps in seq:
            enc_sdr       = SDR(input_width)
            enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist, pm, pmd, ps)
            sp.compute(enc_sdr, False, active_columns)
            tm.compute(active_columns, learn=False)
            al_live.compute(float(tm.anomaly))
    al_live_bytes = pickle.dumps(al_live)

    # ── Evaluation helper ─────────────────────────────────────────
    def _get_seq(fp):
        if wcache is not None:
            return wcache['sequences'].get(fp)
        events     = cache.get(fp)
        vel_events = vel_cache.get(fp) if vel_cache else None
        if not events or not vel_events:
            return None
        return get_file_velocity_seq(events, vel_events, window_size, window_step,
                                     ref_dwells, ref_flights, ref_dists)

    def get_scores(file_list, is_bot):
        scores, labels, seqs = [], [], []
        for fp in file_list:
            seq = _get_seq(fp)
            if not seq:
                continue
            tm.reset()
            raw = []
            for stats, key_idx, dwell, flight, dist, pm, pmd, ps in seq:
                enc_sdr       = SDR(input_width)
                enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist, pm, pmd, ps)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(al.compute(float(tm.anomaly)))
            label  = 1 if is_bot else 0
            valid  = raw[warmup:]
            scores.append(float(np.mean(valid)) if valid else 0.0)
            labels.append(label)
            seqs.append(raw)
        return scores, labels, seqs

    def get_scores_live(file_list, is_bot):
        scores, labels, seqs = [], [], []
        for fp in file_list:
            seq = _get_seq(fp)
            if not seq:
                continue
            al_fresh = pickle.loads(al_live_bytes)
            tm.reset()
            raw = []
            for stats, key_idx, dwell, flight, dist, pm, pmd, ps in seq:
                enc_sdr       = SDR(input_width)
                enc_sdr.dense = encoder.encode(stats, key_idx, dwell, flight, dist, pm, pmd, ps)
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                raw.append(al_fresh.compute(float(tm.anomaly)))
            label  = 1 if is_bot else 0
            valid  = raw[warmup:]
            scores.append(float(np.mean(valid)) if valid else 0.0)
            labels.append(label)
            seqs.append(raw)
        return scores, labels, seqs

    # ── Validation + two-stage threshold sweep ────────────────────
    print("Validating...")
    val_h_sc, val_h_lb, val_h_sq = get_scores(val_human, False)
    val_b_sc, val_b_lb, val_b_sq = get_scores(val_bots,  True)
    all_val_seqs   = val_h_sq + val_b_sq
    all_val_labels = val_h_lb + val_b_lb

    best_bacc, best_thresh = 0.0, 0.0
    coarse_pts = np.linspace(0, 1, 200)
    for th in coarse_pts:
        preds = apply_detection(all_val_seqs, 'first_crossing', th, warmup,
                                labels=all_val_labels)
        bacc = balanced_accuracy_score(all_val_labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, th

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

    # ── Live-mode threshold (per-file independent AL) ─────────────
    print("Computing live-mode threshold...")
    lv_h_sc, lv_h_lb, lv_h_sq = get_scores_live(val_human, False)
    lv_b_sc, lv_b_lb, lv_b_sq = get_scores_live(val_bots,  True)
    lv_val_seqs   = lv_h_sq + lv_b_sq
    lv_val_labels = lv_h_lb + lv_b_lb

    live_bacc, live_thresh = 0.0, 0.0
    for th in coarse_pts:
        preds = apply_detection(lv_val_seqs, 'first_crossing', th, warmup,
                                labels=lv_val_labels)
        bacc = balanced_accuracy_score(lv_val_labels, preds)
        if bacc > live_bacc:
            live_bacc, live_thresh = bacc, th

    fine_lo = max(0.0, live_thresh - 2 * coarse_step)
    fine_hi = min(1.0, live_thresh + 2 * coarse_step)
    for th in np.linspace(fine_lo, fine_hi, 1000):
        preds = apply_detection(lv_val_seqs, 'first_crossing', th, warmup,
                                labels=lv_val_labels)
        bacc = balanced_accuracy_score(lv_val_labels, preds)
        if bacc > live_bacc:
            live_bacc, live_thresh = bacc, th

    live_f1 = f1_score(lv_val_labels,
                       apply_detection(lv_val_seqs, 'first_crossing',
                                       live_thresh, warmup, labels=lv_val_labels),
                       zero_division=0)
    print(f"  Live thresh = {live_thresh:.4f}  (BAcc={live_bacc:.4f}  F1={live_f1:.4f})")

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
          f"warmup={warmup}, thresh={best_thresh:.4f}) ---")
    caught_steps, missed_bot_lens = [], []
    for i, (seq, label) in enumerate(zip(all_test_seqs, all_test_labels)):
        if label != 1:
            continue
        post = seq[warmup:]
        if not post:
            missed_bot_lens.append(len(seq))
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"NOT DETECTED  (short: {len(seq)} windows)")
            continue
        detect_step = next(
            (warmup + j for j, s in enumerate(post) if s >= best_thresh), -1)
        if detect_step >= 0:
            caught_steps.append(detect_step)
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"detected at window {detect_step:4d}  "
                  f"(score={seq[detect_step]:.3f})")
        else:
            missed_bot_lens.append(len(seq))
            print(f"  Bot {len(caught_steps)+len(missed_bot_lens):3d}: "
                  f"NOT DETECTED  (max={max(post):.3f} < {best_thresh:.4f})")

    mean_detect = float(np.mean(caught_steps)) if caught_steps else -1.0
    n_caught    = len(caught_steps)
    n_bots      = len(test_b_sq)
    det_str     = f"{mean_detect:.1f}" if caught_steps else "N/A"
    print(f"  Caught {n_caught}/{n_bots} bots  |  Mean detection window: {det_str}")

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
            "al_live_bytes":  al_live_bytes,
            "input_width":    input_width,
            "best_thresh":    best_thresh,
            "live_thresh":    live_thresh,
            "detection_mode": "first_crossing",
            "warmup_steps":   warmup,
            "al_period":      al_period,
            "window_size":    window_size,
            "window_step":    window_step,
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
        "config_idx":           config_idx,
        "config":               cfg,
        "val_bacc":             float(best_bacc),
        "val_f1":               float(best_f1),
        "test_f1":              float(test_f1),
        "best_thresh":          float(best_thresh),
        "live_thresh":          float(live_thresh),
        "mean_bot_detect_step": mean_detect,
        "n_bots_caught":        n_caught,
        "n_bots_total":         n_bots,
        "mean_bot_score":       float(np.mean(test_b_sc)) if test_b_sc else 0.0,
        "mean_human_score":     float(np.mean(test_h_sc)) if test_h_sc else 0.0,
        "model_file":           model_path,
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
