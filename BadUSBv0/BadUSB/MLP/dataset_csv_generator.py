import os
import glob
import json
import random
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- HYPERPARAMETERS ---
WINDOW_SIZE = 15
STEP_SIZE_HUMAN = 1
STEP_SIZE_BOT = 1
NUM_REFERENCES = 20       # diminishing returns beyond ~15-20 references
MAX_KEYSTROKES = None     # truncate every file to first N keystrokes; None = no limit

OUTPUT_REFS = "reference_pool.npz"
POLY_MODEL_PATH = "poly_regressor.pkl"

# ==============================================================================
# 0. KEYBOARD TOPOLOGY (X, Y coordinates)
# ==============================================================================
KEYBOARD_MAP = {
    'q': (0, 3, 'L'), 'w': (1, 3, 'L'), 'e': (2, 3, 'L'), 'r': (3, 3, 'L'), 't': (4, 3, 'L'),
    'y': (5, 3, 'R'), 'u': (6, 3, 'R'), 'i': (7, 3, 'R'), 'o': (8, 3, 'R'), 'p': (9, 3, 'R'),
    'a': (0.5, 2, 'L'), 's': (1.5, 2, 'L'), 'd': (2.5, 2, 'L'), 'f': (3.5, 2, 'L'), 'g': (4.5, 2, 'L'),
    'h': (5.5, 2, 'R'), 'j': (6.5, 2, 'R'), 'k': (7.5, 2, 'R'), 'l': (8.5, 2, 'R'),
    'z': (1, 1, 'L'), 'x': (2, 1, 'L'), 'c': (3, 1, 'L'), 'v': (4, 1, 'L'), 'b': (5, 1, 'L'),
    'n': (6, 1, 'R'), 'm': (7, 1, 'R'),
    'space': (4.5, 0, 'N')
}

# ==============================================================================
# 1. PARSER
# ==============================================================================
def parse_file(filepath):
    dwells = []
    flights_detailed = []
    active_keys = {}
    last_keyup_ts = None
    last_keyup_name = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return [], []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue

        key, action = parts[0].lower(), parts[1]
        try:
            ts = int(parts[2])
        except ValueError: continue

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts)
                if 0 < delta < 5000:
                    flights_detailed.append((delta, last_keyup_name, key))

        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts)
                if 0 < delta < 3000:
                    dwells.append(delta)

    return dwells[:MAX_KEYSTROKES], flights_detailed[:MAX_KEYSTROKES]

# ==============================================================================
# 2. FEATURE EXTRACTION (14 stats + 3 Poly Errors = 17 features)
# ==============================================================================
def extract_features(w_d, w_f_detailed, reference_pool_d, reference_pool_f, poly_model):
    if len(w_d) < WINDOW_SIZE or len(w_f_detailed) < WINDOW_SIZE: return None

    w_d_arr = np.array(w_d)
    d_std = np.std(w_d_arr)
    d_skew = stats.skew(w_d_arr) if d_std >= 0.0001 else 0
    d_kurt = stats.kurtosis(w_d_arr) if d_std >= 0.0001 else 10
    d_ks = min([stats.ks_2samp(w_d_arr, r, method='asymp')[0] for r in reference_pool_d])
    d_w = min([stats.wasserstein_distance(w_d_arr, r) for r in reference_pool_d])
    d_features = [np.mean(w_d_arr), np.median(w_d_arr), d_std, d_skew, d_kurt, d_ks, d_w]

    w_f_arr = np.array([item[0] for item in w_f_detailed])
    f_std = np.std(w_f_arr)
    f_skew = stats.skew(w_f_arr) if f_std >= 0.0001 else 0
    f_kurt = stats.kurtosis(w_f_arr) if f_std >= 0.0001 else 10
    f_ks = min([stats.ks_2samp(w_f_arr, r, method='asymp')[0] for r in reference_pool_f])
    f_w = min([stats.wasserstein_distance(w_f_arr, r) for r in reference_pool_f])
    f_features = [np.mean(w_f_arr), np.median(w_f_arr), f_std, f_skew, f_kurt, f_ks, f_w]

    X_poly = []
    y_true = []
    for t, k1, k2 in w_f_detailed:
        if 20 < t < 400:
            if k1 in KEYBOARD_MAP and k2 in KEYBOARD_MAP:
                x1, y1 = KEYBOARD_MAP[k1][:2]
                x2, y2 = KEYBOARD_MAP[k2][:2]
                X_poly.append([x1, y1, x2, y2])
                y_true.append(t)

    if len(X_poly) > 0:
        X_poly = np.array(X_poly)
        y_true = np.array(y_true)
        y_pred = poly_model.predict(X_poly)
        errors = np.abs(y_true - y_pred)
        err_mean, err_med, err_std = np.mean(errors), np.median(errors), np.std(errors)
    else:
        err_mean, err_med, err_std = 0.0, 0.0, 0.0

    return d_features + f_features + [err_mean, err_med, err_std]

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def process_single_file(filepath, label, step_size, ref_dwells, ref_flights, poly_model, file_id=0):
    d, f_det = parse_file(filepath)
    min_len = min(len(d), len(f_det))
    if min_len < WINDOW_SIZE: return []

    rows = []
    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i : i + WINDOW_SIZE]
        w_f = f_det[i : i + WINDOW_SIZE]
        feats = extract_features(w_d, w_f, ref_dwells, ref_flights, poly_model)
        if feats:
            rows.append(feats + [label, file_id])
    return rows

# ==============================================================================
# 4. REFERENCE POOL  (built from TRAIN humans only)
# ==============================================================================
def create_reference_pool(files):
    all_d, all_f = [], []
    for f in files[:50]:
        d, fl = parse_file(f)
        if len(d) >= WINDOW_SIZE: all_d.extend(d)
        if len(fl) >= WINDOW_SIZE: all_f.extend([item[0] for item in fl])

    ref_d, ref_f = [], []
    for _ in range(NUM_REFERENCES):
        start = random.randint(0, len(all_d) - WINDOW_SIZE)
        ref_d.append(np.array(all_d[start:start+WINDOW_SIZE]))
        start = random.randint(0, len(all_f) - WINDOW_SIZE)
        ref_f.append(np.array(all_f[start:start+WINDOW_SIZE]))
    return ref_d, ref_f

# ==============================================================================
# 5. PROCESS ONE SPLIT → balanced CSV
# ==============================================================================
COLS = [
    "D_Mean", "D_Med", "D_Std", "D_Skew", "D_Kurt", "D_MinKS", "D_MinW",
    "F_Mean", "F_Med", "F_Std", "F_Skew", "F_Kurt", "F_MinKS", "F_MinW",
    "Poly_Err_Mean", "Poly_Err_Med", "Poly_Err_Std",
    "Label", "FileID"
]

def process_split(split_name, human_files, bot_files, ref_d, ref_f, poly_model,
                  max_workers, mode="partial", tag=""):
    tasks = (
        [(f, 0, STEP_SIZE_HUMAN, ref_d, ref_f, poly_model, i)
         for i, f in enumerate(human_files)] +
        [(f, 1, STEP_SIZE_BOT,   ref_d, ref_f, poly_model, len(human_files) + i)
         for i, f in enumerate(bot_files)]
    )

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_single_file, *t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks),
                        desc=f"  {split_name}"):
            try:
                res = fut.result()
                if res: rows.extend(res)
            except Exception:
                pass

    humans_r = [r for r in rows if r[-2] == 0]  # r[-2]=label, r[-1]=file_id
    bots_r   = [r for r in rows if r[-2] == 1]

    if mode == "partial":
        n = min(len(humans_r), len(bots_r))
        balanced = random.sample(humans_r, n) + random.sample(bots_r, n)
        balance_note = f"→ {len(balanced)} balanced rows"
    else:  # full: all windows, no undersampling
        balanced = humans_r + bots_r
        balance_note = f"→ {len(balanced)} rows (imbalanced, use class weights)"

    df = pd.DataFrame(balanced, columns=COLS).sample(frac=1).reset_index(drop=True)
    tag_suffix = f"_{tag}" if tag else ""
    out_csv = f"{split_name}_dataset{tag_suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  ({len(humans_r)} human / {len(bots_r)} bot windows {balance_note})")
    return df

# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="../data_split.json",
                        help="Path to data_split.json (default: ../data_split.json)")
    parser.add_argument("--mode", choices=["partial", "full"], default="partial",
                        help="'partial' (default): balance by undersampling majority class. "
                             "'full': all windows, imbalanced — training scripts use class weights.")
    parser.add_argument("--tag", default="", help="Extra suffix for output CSV filenames (e.g. 'fullkey')")
    args = parser.parse_args()

    # Load split manifest
    split_json = os.path.abspath(args.split_json)
    if not os.path.exists(split_json):
        print(f"ERROR: {split_json} not found. Run split_persons.py first.")
        return
    with open(split_json) as f:
        split = json.load(f)

    print("--- 1. Loading poly regressor ---")
    try:
        poly_model = joblib.load(POLY_MODEL_PATH)
        print(f"Loaded: {POLY_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading polynomial model: {e}"); return

    print("\n--- 2. Building reference pool (train humans only) ---")
    random.seed(42)
    ref_d, ref_f = create_reference_pool(split["train"]["humans"])
    np.savez(OUTPUT_REFS, dwell=ref_d, flight=ref_f)
    print(f"Saved {OUTPUT_REFS}  ({len(ref_d)} references from "
          f"{len(split['train']['humans'])} train human files)")

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n--- 3. Feature extraction  (workers={max_workers}) ---")

    print(f"\nMode: {args.mode}")
    for name in ("train", "val", "test"):
        print(f"\n[{name}]  humans={len(split[name]['humans'])}, "
              f"bots={len(split[name]['bots'])}")
        process_split(name,
                      split[name]["humans"],
                      split[name]["bots"],
                      ref_d, ref_f, poly_model, max_workers, mode=args.mode, tag=args.tag)

    print("\nDone. Files created: train_dataset.csv, val_dataset.csv, test_dataset.csv")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
