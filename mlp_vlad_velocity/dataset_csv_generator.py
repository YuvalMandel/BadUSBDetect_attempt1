import os
import glob
import random
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- HYPERPARAMETERS ---
WINDOW_SIZE = 15        
STEP_SIZE_HUMAN = 10    
STEP_SIZE_BOT = 5       
NUM_REFERENCES = 50     

FOLDERS = {
    "Humans": ["s0"],             
    "Bots": ["Synthetic_Bots"]    
}

OUTPUT_CSV = "balanced_dataset.csv"
OUTPUT_REFS = "reference_pool.npz" 
POLY_MODEL_PATH = "poly_regressor.pkl" # Путь к вашей модели Тейлора

# ==============================================================================
# 0. KEYBOARD TOPOLOGY (X, Y coordinates)
# ==============================================================================
# Оставили 'L' и 'R' для совместимости, но брать будем только первые 2 значения (x, y)
KEYBOARD_MAP = {
    # Row 3 — QWERTY
    'q': (0, 3, 'L'), 'w': (1, 3, 'L'), 'e': (2, 3, 'L'), 'r': (3, 3, 'L'), 't': (4, 3, 'L'),
    'y': (5, 3, 'R'), 'u': (6, 3, 'R'), 'i': (7, 3, 'R'), 'o': (8, 3, 'R'), 'p': (9, 3, 'R'),
    'tab': (-0.5, 3, 'L'),
    'oemopenbrackets': (10, 3, 'R'), 'oemclosebrackets': (11, 3, 'R'),
    # Row 2 — ASDF
    'a': (0.5, 2, 'L'), 's': (1.5, 2, 'L'), 'd': (2.5, 2, 'L'), 'f': (3.5, 2, 'L'), 'g': (4.5, 2, 'L'),
    'h': (5.5, 2, 'R'), 'j': (6.5, 2, 'R'), 'k': (7.5, 2, 'R'), 'l': (8.5, 2, 'R'),
    'oemsemicolon': (9.5, 2, 'R'), 'oemquotes': (10.5, 2, 'R'), 'return': (12.5, 2, 'R'),
    # Row 1 — ZXCV
    'lshiftkey': (-1, 1, 'L'),
    'z': (1, 1, 'L'), 'x': (2, 1, 'L'), 'c': (3, 1, 'L'), 'v': (4, 1, 'L'), 'b': (5, 1, 'L'),
    'n': (6, 1, 'R'), 'm': (7, 1, 'R'),
    'oemcomma': (8, 1, 'R'), 'oemperiod': (9, 1, 'R'), 'oemquestion': (10, 1, 'R'),
    'rshiftkey': (12.5, 1, 'R'),
    # Row 0 — bottom
    'lcontrolkey': (0, 0, 'L'), 'space': (4.5, 0, 'N'), 'rcontrolkey': (13, 0, 'R'),
    # Row 4 — numbers
    'd1': (1, 4, 'L'), 'd2': (2, 4, 'L'), 'd3': (3, 4, 'L'), 'd4': (4, 4, 'L'), 'd5': (5, 4, 'L'),
    'd6': (6, 4, 'R'), 'd7': (7, 4, 'R'), 'd8': (8, 4, 'R'), 'd9': (9, 4, 'R'), 'd0': (10, 4, 'R'),
    'oemminus': (11, 4, 'R'), 'oemplus': (12, 4, 'R'), 'back': (13.5, 4, 'R'),
    # Arrow cluster
    'up': (15, 1, 'N'), 'left': (14, 0, 'N'), 'down': (15, 0, 'N'), 'right': (16, 0, 'N'),
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
    except: return [], []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        
        key, action = parts[0].lower(), parts[1]
        try:
            ts = int(parts[2])
        except ValueError: continue

        scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts) / scale
                if 0 < delta < 5000: 
                    flights_detailed.append((delta, last_keyup_name, key))
                    
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts) / scale
                if 0 < delta < 3000: 
                    dwells.append(delta)
                    
    return dwells, flights_detailed

# ==============================================================================
# 2. FEATURE EXTRACTION (14 old + 3 new Poly Errors)
# ==============================================================================
def extract_features(w_d, w_f_detailed, reference_pool_d, reference_pool_f, poly_model):
    if len(w_d) < WINDOW_SIZE or len(w_f_detailed) < WINDOW_SIZE: return None
    
    # --- 1. Dwell features ---
    w_d_arr = np.array(w_d)
    d_std = np.std(w_d_arr)
    d_skew = stats.skew(w_d_arr) if d_std >= 0.0001 else 0
    d_kurt = stats.kurtosis(w_d_arr) if d_std >= 0.0001 else 10
    d_ks = min([stats.ks_2samp(w_d_arr, r, method='asymp')[0] for r in reference_pool_d])
    d_w = min([stats.wasserstein_distance(w_d_arr, r) for r in reference_pool_d])
    d_features = [np.mean(w_d_arr), np.median(w_d_arr), d_std, d_skew, d_kurt, d_ks, d_w]

    # --- 2. Flight features ---
    w_f_arr = np.array([item[0] for item in w_f_detailed])
    f_std = np.std(w_f_arr)
    f_skew = stats.skew(w_f_arr) if f_std >= 0.0001 else 0
    f_kurt = stats.kurtosis(w_f_arr) if f_std >= 0.0001 else 10
    f_ks = min([stats.ks_2samp(w_f_arr, r, method='asymp')[0] for r in reference_pool_f])
    f_w = min([stats.wasserstein_distance(w_f_arr, r) for r in reference_pool_f])
    f_features = [np.mean(w_f_arr), np.median(w_f_arr), f_std, f_skew, f_kurt, f_ks, f_w]

    # --- 3. POLYNOMIAL FITT'S LAW FEATURES ---
    X_poly = []
    y_true = []

    for t, k1, k2 in w_f_detailed:
        if k1 in KEYBOARD_MAP and k2 in KEYBOARD_MAP:
            x1, y1 = KEYBOARD_MAP[k1][:2]
            x2, y2 = KEYBOARD_MAP[k2][:2]
            X_poly.append([x1, y1, x2, y2])
            y_true.append(t)

    # Вычисляем ошибку, только если в окне были известные переходы
    if len(X_poly) > 0:
        X_poly = np.array(X_poly)
        y_true = np.array(y_true)
        y_pred = poly_model.predict(X_poly)
        
        errors = np.abs(y_true - y_pred)
        err_mean = np.mean(errors)
        err_med = np.median(errors)
        err_std = np.std(errors)
    else:
        # Fallback (например, все 15 кнопок были Enter, Backspace, Ctrl)
        err_mean, err_med, err_std = 0.0, 0.0, 0.0

    # Итого: 7 + 7 + 3 = 17 фичей
    return d_features + f_features + [err_mean, err_med, err_std]

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def process_single_file(filepath, label, step_size, ref_dwells, ref_flights, poly_model):
    rows = []
    d, f_det = parse_file(filepath)
    min_len = min(len(d), len(f_det))
    if min_len < WINDOW_SIZE: return []

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i : i + WINDOW_SIZE]
        w_f = f_det[i : i + WINDOW_SIZE]
        
        feats = extract_features(w_d, w_f, ref_dwells, ref_flights, poly_model)
        if feats:
            rows.append(feats + [label])
    return rows

# ==============================================================================
# 4. REFERENCE COLLECTION
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
# 5. MAIN
# ==============================================================================
def main():
    print("--- 1. Initialization ---")
    try:
        poly_model = joblib.load(POLY_MODEL_PATH)
        print(f"✅ Loaded Fitts's Law Model: {POLY_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading polynomial model: {e}")
        return

    # Search for humans (only *1.txt)
    human_files = []
    for folder in FOLDERS["Humans"]:
        for f in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True):
            if f.endswith("1.txt"):
                human_files.append(f)
    random.shuffle(human_files)
    print(f"Humans (Task 1): {len(human_files)} files")

    # Search for bots
    bot_files = []
    for folder in FOLDERS["Bots"]:
        bot_files.extend(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    print(f"Bots: {len(bot_files)} files")

    if not human_files or not bot_files:
        print("Error: No human or bot files found.")
        return

    print("\n--- 2. Creating references ---")
    ref_dwells, ref_flights = create_reference_pool(human_files)
    np.savez(OUTPUT_REFS, dwell=ref_dwells, flight=ref_flights)

    # Prepare tasks
    tasks = []
    for f in human_files:
        tasks.append((f, 0, STEP_SIZE_HUMAN, ref_dwells, ref_flights, poly_model))
    for f in bot_files:
        tasks.append((f, 1, STEP_SIZE_BOT, ref_dwells, ref_flights, poly_model))

    print(f"\n--- 3. Feature extraction (Tasks: {len(tasks)}) ---")
    dataset_rows = []
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, *t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing files"):
            try:
                res = future.result()
                if res: dataset_rows.extend(res)
            except Exception as e: pass

    # --- 4. BALANCING ---
    humans_data = [r for r in dataset_rows if r[-1] == 0]
    bots_data = [r for r in dataset_rows if r[-1] == 1]
    
    min_count = min(len(humans_data), len(bots_data))
    print(f"\n--- 4. Balancing ---")
    print(f"Available windows: Humans={len(humans_data)}, Bots={len(bots_data)}")
    print(f"Trimming to {min_count} windows per class (50/50)...")
    
    balanced_data = random.sample(humans_data, min_count) + random.sample(bots_data, min_count)

    # --- 5. EXPORT ---
    cols = [
        "D_Mean", "D_Med", "D_Std", "D_Skew", "D_Kurt", "D_MinKS", "D_MinW",
        "F_Mean", "F_Med", "F_Std", "F_Skew", "F_Kurt", "F_MinKS", "F_MinW",
        "Poly_Err_Mean", "Poly_Err_Med", "Poly_Err_Std", # 3 Новые фичи
        "Label"
    ]
    
    df = pd.DataFrame(balanced_data, columns=cols)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Success! Dataset saved: {OUTPUT_CSV}")
    print(f"Final size: {len(df)} rows, 17 features + Label.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
