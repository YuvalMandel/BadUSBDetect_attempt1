import os
import glob
import random
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- HYPERPARAMETERS ---
WINDOW_SIZE = 15
STEP_SIZE_HUMAN = 10
STEP_SIZE_BOT = 1
NUM_REFERENCES = 50
SYNTHETIC_SAMPLES = 2000   # For the train split; val/test get proportional counts

# Person-based split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (the remainder)

RANDOM_SEED = 42

FOLDERS = {
    "Humans": [
        "../UB_keystroke_dataset/s0/rotation/",
        "../UB_keystroke_dataset/s1/rotation/",
        "../UB_keystroke_dataset/s2/rotation/",
    ],
    "Bots": ["../BadUSBdataset", "../only_timings_dataset"]
}

OUTPUT_TRAIN = "train_dataset.csv"
OUTPUT_VAL   = "val_dataset.csv"
OUTPUT_TEST  = "test_dataset.csv"
OUTPUT_REFS  = "reference_pool.npz"

# ==============================================================================
# 0. HELPERS — Task filtering and person-based splitting
# ==============================================================================

def is_task1(filepath):
    """
    Returns True if the file belongs to Task 1.
    File naming: UUUSKE.txt  (UUU=userID, S=session, K=keyboard, E=task)
    Task 1 → 6th character (index 5) == '1'.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return len(basename) == 6 and basename[5] == '1'


def get_person_id(filepath):
    """Extracts the 3-digit person ID (first 3 characters of the filename)."""
    return os.path.splitext(os.path.basename(filepath))[0][:3]


def split_by_person(all_files, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=RANDOM_SEED):
    """
    Splits a list of files into train/val/test sets such that ALL files
    belonging to the same person end up in the same split (no data leakage).

    Returns:
        (train_files, train_persons), (val_files, val_persons), (test_files, test_persons)
    """
    person_ids = sorted(set(get_person_id(f) for f in all_files))

    rng = random.Random(seed)
    rng.shuffle(person_ids)

    n = len(person_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_persons = set(person_ids[:n_train])
    val_persons   = set(person_ids[n_train : n_train + n_val])
    test_persons  = set(person_ids[n_train + n_val :])

    train_files = [f for f in all_files if get_person_id(f) in train_persons]
    val_files   = [f for f in all_files if get_person_id(f) in val_persons]
    test_files  = [f for f in all_files if get_person_id(f) in test_persons]

    return (train_files, train_persons), (val_files, val_persons), (test_files, test_persons)


# ==============================================================================
# 1. PARSER (logic unchanged, extracted for pickling)
# ==============================================================================

def parse_file(filepath):
    dwells = []
    flights = []
    active_keys = {}
    last_keyup = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except:
        return [], []

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


# ==============================================================================
# 2. FEATURE EXTRACTION
# ==============================================================================

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


# ==============================================================================
# 3. WORKER FUNCTION (processes a single file on a separate CPU core)
# ==============================================================================

def process_single_file(filepath, label, step_size, ref_dwells, ref_flights):
    rows = []
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))

    if min_len < WINDOW_SIZE:
        return []

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i : i + WINDOW_SIZE]
        w_f = f[i : i + WINDOW_SIZE]

        ft_d = extract_features(w_d, ref_dwells)
        ft_f = extract_features(w_f, ref_flights)

        if ft_d and ft_f:
            rows.append(ft_f + ft_d + [label])

    return rows


# ==============================================================================
# 4. REFERENCE POOL — built from TRAINING humans only (no leakage)
# ==============================================================================

def create_reference_pool(train_human_files):
    """
    Builds the KS/Wasserstein reference pool using only training-set human files
    so that validation and test distributions never influence feature extraction.
    """
    print("--- Collecting Reference Pool (train humans only) ---")
    all_human_dwells  = []
    all_human_flights = []

    sample_files = list(train_human_files)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(sample_files)

    for filepath in sample_files[:50]:
        d, fl = parse_file(filepath)
        if len(d)  >= WINDOW_SIZE: all_human_dwells.extend(d)
        if len(fl) >= WINDOW_SIZE: all_human_flights.extend(fl)

    ref_dwells, ref_flights = [], []
    rng2 = random.Random(RANDOM_SEED)

    if len(all_human_dwells) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_dwells) - WINDOW_SIZE
            start = rng2.randint(0, max_start)
            ref_dwells.append(np.array(all_human_dwells[start : start + WINDOW_SIZE]))
    else:
        print("!!! ERROR: Insufficient Dwell data for reference pool")

    if len(all_human_flights) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_flights) - WINDOW_SIZE
            start = rng2.randint(0, max_start)
            ref_flights.append(np.array(all_human_flights[start : start + WINDOW_SIZE]))
    else:
        print("!!! ERROR: Insufficient Flight data for reference pool")

    return ref_dwells, ref_flights


# ==============================================================================
# 5. BOT GENERATOR
# ==============================================================================

def generate_synthetic_bots(n_samples):
    synthetic_rows = []
    attack_types = ['machine_gun', 'gaussian', 'uniform', 'jitter']

    for _ in range(n_samples):
        attack_type = random.choice(attack_types)
        win_dwell, win_flight = None, None

        if attack_type == 'machine_gun':
            val = random.randint(4, 10)
            win_dwell  = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)
            win_flight = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)
        elif attack_type == 'gaussian':
            mean, std = random.randint(80, 150), random.randint(10, 40)
            win_dwell  = np.random.normal(mean, std, WINDOW_SIZE)
            win_flight = np.random.normal(mean, std, WINDOW_SIZE)
        elif attack_type == 'uniform':
            low, high = random.randint(10, 50), random.randint(60, 200)
            win_dwell  = np.random.uniform(low, high, WINDOW_SIZE)
            win_flight = np.random.uniform(low, high, WINDOW_SIZE)
        elif attack_type == 'jitter':
            win_dwell  = np.random.randint(5, 300, WINDOW_SIZE).astype(float)
            win_flight = np.random.randint(5, 300, WINDOW_SIZE).astype(float)

        if win_dwell is not None:
            synthetic_rows.append((np.abs(win_dwell), np.abs(win_flight)))

    return synthetic_rows


# ==============================================================================
# 6. PROCESS ONE SPLIT (parallel file processing + synthetic bots)
# ==============================================================================

def process_split(human_files, bot_files, ref_dwells, ref_flights, n_synthetic, split_name):
    dataset_rows = []

    tasks = []
    for f in human_files:
        tasks.append((f, 0, STEP_SIZE_HUMAN, ref_dwells, ref_flights))
    for f in bot_files:
        tasks.append((f, 1, STEP_SIZE_BOT, ref_dwells, ref_flights))

    if tasks:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_file, *t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks),
                               desc=f"Processing {split_name}"):
                try:
                    result_rows = future.result()
                    if result_rows:
                        dataset_rows.extend(result_rows)
                except Exception as e:
                    print(f"  Error in worker: {e}")

    # Synthetic bots (generated fresh, no leakage concern)
    if n_synthetic > 0:
        print(f"  Generating {n_synthetic} synthetic bots for {split_name}...")
        synth_data = generate_synthetic_bots(n_synthetic)
        for w_d, w_f in tqdm(synth_data, desc=f"  Synthetic ({split_name})"):
            ft_d = extract_features(w_d, ref_dwells)
            ft_f = extract_features(w_f, ref_flights)
            if ft_d and ft_f:
                dataset_rows.append(ft_f + ft_d + [1])

    return dataset_rows


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main():
    # ---- 1. Collect all Task 1 human files ----
    all_human_files = []
    for folder in FOLDERS["Humans"]:
        files = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        task1_files = [f for f in files if is_task1(f)]
        all_human_files.extend(task1_files)

    print(f"Task 1 human files found: {len(all_human_files)}")
    if not all_human_files:
        print("ERROR: No Task 1 human files found. Check FOLDERS paths.")
        return

    # ---- 2. Person-based split ----
    (train_human, train_persons), \
    (val_human,   val_persons),   \
    (test_human,  test_persons)   = split_by_person(all_human_files)

    print(f"\nPerson split — Train: {len(train_persons)}, "
          f"Val: {len(val_persons)}, Test: {len(test_persons)}")
    print(f"File split   — Train: {len(train_human)}, "
          f"Val: {len(val_human)}, Test: {len(test_human)}")

    # ---- 3. Reference pool from TRAIN humans only ----
    ref_dwells, ref_flights = create_reference_pool(train_human)
    if not ref_dwells:
        return
    np.savez(OUTPUT_REFS, dwell=ref_dwells, flight=ref_flights)
    print("Reference pool saved.")

    # ---- 4. Collect and split real bot files (random split, no person concept) ----
    all_bot_files = []
    for folder in FOLDERS["Bots"]:
        all_bot_files.extend(
            glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_bot_files)
    nb = len(all_bot_files)
    nb_train = int(nb * TRAIN_RATIO)
    nb_val   = int(nb * VAL_RATIO)
    train_bots = all_bot_files[:nb_train]
    val_bots   = all_bot_files[nb_train : nb_train + nb_val]
    test_bots  = all_bot_files[nb_train + nb_val :]
    print(f"\nReal bot files — Train: {len(train_bots)}, "
          f"Val: {len(val_bots)}, Test: {len(test_bots)}")

    # ---- 5. Synthetic bot counts proportional to split sizes ----
    n_synth_train = SYNTHETIC_SAMPLES
    n_synth_val   = int(SYNTHETIC_SAMPLES * VAL_RATIO / TRAIN_RATIO)
    n_synth_test  = int(SYNTHETIC_SAMPLES * (1 - TRAIN_RATIO - VAL_RATIO) / TRAIN_RATIO)

    # ---- 6. Process each split and save ----
    COLS = ["F_Mean", "F_Med", "F_Std", "F_Skew", "F_Kurt", "F_MinKS", "F_MinW",
            "D_Mean", "D_Med", "D_Std", "D_Skew", "D_Kurt", "D_MinKS", "D_MinW", "Label"]

    splits = [
        ("train", train_human, train_bots, n_synth_train, OUTPUT_TRAIN),
        ("val",   val_human,   val_bots,   n_synth_val,   OUTPUT_VAL),
        ("test",  test_human,  test_bots,  n_synth_test,  OUTPUT_TEST),
    ]

    for split_name, human_files, bot_files, n_synth, output_file in splits:
        print(f"\n{'='*60}")
        print(f" Processing {split_name.upper()} split")
        print(f"{'='*60}")

        rows = process_split(human_files, bot_files, ref_dwells, ref_flights,
                             n_synth, split_name)

        df = pd.DataFrame(rows, columns=COLS)
        df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        df.to_csv(output_file, index=False)

        print(f"\nSaved: {output_file}  ({len(df)} rows)")
        print(df["Label"].value_counts().to_string())

    print("\n✅ All splits saved.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()