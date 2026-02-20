import os
import glob
import random
import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# HTM Imports
try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler, TemporalMemory
    from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
except ImportError:
    print("HTM libraries not found. Please install htm.core.")
    # Mock classes for analysis/linting purposes
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

from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
WINDOW_SIZE = 15
STEP_SIZE_HUMAN = 1
STEP_SIZE_BOT = 1
NUM_REFERENCES = 50
RANDOM_SEED = 42
CACHE_FILE = "features_cache.pkl"

# Paths
FOLDERS = {
    "Humans": [
        "../UB_keystroke_dataset/s0/rotation/",
        "../UB_keystroke_dataset/s1/rotation/",
        "../UB_keystroke_dataset/s2/rotation/",
    ],
    "Bots": ["../BadUSBdataset", "../only_timings_dataset"]
}

# ==============================================================================
# DATA PROCESSING
# ==============================================================================

def is_task1(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return len(basename) == 6 and basename[5] == '1'

def get_person_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0][:3]

def parse_file(filepath):
    dwells = []
    flights = []
    active_keys = {}
    last_keyup = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
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

def create_reference_pool(train_human_files):
    print("--- Collecting Reference Pool (train humans only) ---")
    all_human_dwells  = []
    all_human_flights = []

    sample_files = list(train_human_files)
    random.shuffle(sample_files)

    # Use a subset to build pool
    for filepath in sample_files[:50]:
        d, fl = parse_file(filepath)
        if len(d)  >= WINDOW_SIZE: all_human_dwells.extend(d)
        if len(fl) >= WINDOW_SIZE: all_human_flights.extend(fl)

    ref_dwells, ref_flights = [], []
    
    if len(all_human_dwells) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_dwells) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_dwells.append(np.array(all_human_dwells[start : start + WINDOW_SIZE]))
    
    if len(all_human_flights) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_flights) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_flights.append(np.array(all_human_flights[start : start + WINDOW_SIZE]))

    return ref_dwells, ref_flights

def process_file_worker(args):
    filepath, step_size, ref_dwells, ref_flights = args
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))
    
    features_seq = []
    if min_len < WINDOW_SIZE:
        return filepath, np.array([])

    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i : i + WINDOW_SIZE]
        w_f = f[i : i + WINDOW_SIZE]

        ft_d = extract_features(w_d, ref_dwells)
        ft_f = extract_features(w_f, ref_flights)

        if ft_d and ft_f:
            features_seq.append(ft_f + ft_d)
            
    return filepath, np.array(features_seq)

# ==============================================================================
# ENCODER
# ==============================================================================

class SimpleScalarEncoder:
    def __init__(self, min_val, max_val, n_bits, w):
        self.min_val = min_val
        self.max_val = max_val
        self.n_bits = n_bits
        self.w = w
        self.range = max_val - min_val
        if self.range <= 0: self.range = 1.0

    def encode_into_array(self, value, dense_array, offset):
        if value < self.min_val: value = self.min_val
        if value > self.max_val: value = self.max_val
        
        pos = (value - self.min_val) / self.range
        idx = int(pos * (self.n_bits - self.w))
        
        # Clamp index
        if idx < 0: idx = 0
        if idx > (self.n_bits - self.w): idx = self.n_bits - self.w

        for i in range(self.w):
            dense_array[offset + idx + i] = 1

class MultiAttributeEncoder:
    def __init__(self, num_features, min_vals, max_vals, bits_per_feature=32, w=5):
        self.encoders = []
        self.total_bits = num_features * bits_per_feature
        self.bits_per_feature = bits_per_feature
        
        for i in range(num_features):
            self.encoders.append(
                SimpleScalarEncoder(min_vals[i], max_vals[i], bits_per_feature, w)
            )

    def encode(self, features):
        dense = np.zeros(self.total_bits, dtype=np.uint8)
        for i, val in enumerate(features):
            self.encoders[i].encode_into_array(val, dense, i * self.bits_per_feature)
        return dense

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
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_persons = set(person_ids[:n_train])
    val_persons = set(person_ids[n_train:n_train+n_val])
    test_persons = set(person_ids[n_train+n_val:])
    
    train_human = [f for f in human_files if get_person_id(f) in train_persons]
    val_human = [f for f in human_files if get_person_id(f) in val_persons]
    test_human = [f for f in human_files if get_person_id(f) in test_persons]
    
    random.shuffle(bot_files)
    nb = len(bot_files)
    nb_train = int(nb * 0.7) 
    nb_val = int(nb * 0.15)
    
    val_bots = bot_files[:nb_val]
    test_bots = bot_files[nb_val:nb_val+int(nb*0.15)]
    
    print(f"Split: Train Human={len(train_human)}, Val Human={len(val_human)}, Val Bot={len(val_bots)}")

    # 3. Reference Pool
    ref_dwells, ref_flights = create_reference_pool(train_human)
    
    # 4. Pre-process Features (Multiprocessing with Caching)
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
            
    if not file_features:
        print("Extracting features for all files...")
        
        all_files_to_process = []
        # Train Human
        for f in train_human: all_files_to_process.append((f, STEP_SIZE_HUMAN, ref_dwells, ref_flights))
        # Val Human
        for f in val_human: all_files_to_process.append((f, STEP_SIZE_HUMAN, ref_dwells, ref_flights))
        # Test Human
        for f in test_human: all_files_to_process.append((f, STEP_SIZE_HUMAN, ref_dwells, ref_flights))
        # Val Bot
        for f in val_bots: all_files_to_process.append((f, STEP_SIZE_BOT, ref_dwells, ref_flights))
        # Test Bot
        for f in test_bots: all_files_to_process.append((f, STEP_SIZE_BOT, ref_dwells, ref_flights))

        # Use max_workers=None (defaults to cpu_count)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file_worker, args) for args in all_files_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                fpath, feats = future.result()
                if len(feats) > 0:
                    file_features[fpath] = feats
        
        # Save to cache
        print(f"Saving features to cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(file_features, f)

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

    # 5. Initialize HTM
    encoder = MultiAttributeEncoder(14, min_vals, max_vals, bits_per_feature=32, w=5)
    input_width = encoder.total_bits
    
    sp = SpatialPooler(
        inputDimensions=(input_width,),
        columnDimensions=(2048,),
        potentialPct=0.8,
        globalInhibition=True,
        numActiveColumnsPerInhArea=40,
        localAreaDensity=0.0,  # Explicitly set to 0 to satisfy mutex check
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
    
    # 6. Train
    print("Training HTM...")
    active_columns = SDR(sp.getColumnDimensions())
    
    # Shuffle training order of files, but keep windows within file ordered
    random.shuffle(train_human)
    
    for f in tqdm(train_human, desc="Training"):
        if f not in file_features: continue
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
        for f in file_list:
            if f not in file_features: continue
            seq = file_features[f]
            
            tm.reset()
            file_scores = []
            for feats in seq:
                dense_input = encoder.encode(feats)
                enc_sdr = SDR(input_width)
                enc_sdr.dense = dense_input
                
                sp.compute(enc_sdr, False, active_columns)
                tm.compute(active_columns, learn=False)
                
                score = tm.anomaly
                file_scores.append(score)
            
            if file_scores:
                # Skip warmup
                valid_scores = file_scores[5:] if len(file_scores) > 5 else file_scores
                scores.append(np.mean(valid_scores))
                labels.append(1 if is_bot else 0)
        return scores, labels

    val_human_scores, val_human_labels = get_scores(val_human, False)
    val_bot_scores, val_bot_labels = get_scores(val_bots, True)
    
    all_val_scores = val_human_scores + val_bot_scores
    all_val_labels = val_human_labels + val_bot_labels
    
    best_f1 = 0
    best_thresh = 0
    
    thresholds = np.linspace(0, 1, 100)
    for th in thresholds:
        preds = [1 if s >= th else 0 for s in all_val_scores]
        f1 = f1_score(all_val_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            
    print(f"Best Validation F1: {best_f1:.4f} at Threshold: {best_thresh:.4f}")
    
    # 8. Test
    print("Testing...")
    test_human_scores, test_human_labels = get_scores(test_human, False)
    test_bot_scores, test_bot_labels = get_scores(test_bots, True)
    
    all_test_scores = test_human_scores + test_bot_scores
    all_test_labels = test_human_labels + test_bot_labels
    
    test_preds = [1 if s >= best_thresh else 0 for s in all_test_scores]
    
    print("\n--- Test Set Classification Report ---")
    print(classification_report(all_test_labels, test_preds, target_names=["Human", "Bot"]))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
