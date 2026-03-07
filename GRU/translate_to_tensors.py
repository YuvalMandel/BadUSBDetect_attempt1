import os
import glob
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- SETTINGS ---
SEQ_LEN = 15            # Sequence length (Cold start)
STEP_SIZE = 1           # Sliding window step

HUMAN_DIRS = ["s2/baseline", "s2/rotation"]  # Specify exact paths to human folders
BOT_DIRS = ["Synthetic_Bots"]                # Folder with synthetic data

OUTPUT_FILE = "rnn_dataset.pt"
SCALER_FILE = "rnn_scaler_params.npy"

# ==============================================================================
# 1. PARSER (Returns raw milliseconds)
# ==============================================================================
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
        if len(parts) < 3: continue
        key, action = parts[0], parts[1]
        try:
            ts = int(parts[2])
        except ValueError: continue

        scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup is not None:
                delta = (ts - last_keyup) / scale
                if 0 < delta < 5000: flights.append(delta)
        elif action == "KeyUp":
            last_keyup = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts) / scale
                if 0 < delta < 3000: dwells.append(delta)
                    
    # Return arrays of equal length
    min_len = min(len(dwells), len(flights))
    return np.array(dwells[:min_len]), np.array(flights[:min_len])

# ==============================================================================
# 2. SEQUENCE EXTRACTION
# ==============================================================================
def extract_sequences(directories):
    sequences = []
    files = []
    for d in directories:
        files.extend(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
        
    print(f"Searching in folders {directories}: found {len(files)} files")
    
    for filepath in tqdm(files, desc="Parsing files"):
        dwells, flights = parse_file(filepath)
        if len(dwells) < SEQ_LEN:
            continue
            
        # Slice into chunks of length SEQ_LEN
        for i in range(0, len(dwells) - SEQ_LEN, STEP_SIZE):
            window_d = dwells[i : i + SEQ_LEN]
            window_f = flights[i : i + SEQ_LEN]
            
            # Combine Dwell and Flight into matrix [SEQ_LEN, 2]
            # Format: [[D1, F1], [D2, F2], ..., [D20, F20]]
            seq = np.column_stack((window_d, window_f))
            sequences.append(seq)
            
    return sequences

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
def main():
    print("--- 1. Collecting raw sequences ---")
    human_seqs = extract_sequences(HUMAN_DIRS)
    bot_seqs = extract_sequences(BOT_DIRS)
    
    print(f"\nTotal found:")
    print(f"  Human (Label 0): {len(human_seqs)} windows")
    print(f"  Bot   (Label 1): {len(bot_seqs)} windows")
    
    if len(human_seqs) == 0 or len(bot_seqs) == 0:
        print("Error: Insufficient data for one of the categories.")
        return

    # --- 2. BALANCING (UNDERSAMPLING) ---
    print("\n--- 2. Class balancing ---")
    target_count = min(len(human_seqs), len(bot_seqs))
    
    # Randomly select required number of windows
    human_seqs_balanced = random.sample(human_seqs, target_count)
    bot_seqs_balanced = random.sample(bot_seqs, target_count)
    
    print(f"Balanced: {target_count} windows per class.")
    
    # Convert to Numpy: dimensions (Number_of_samples, SEQ_LEN, 2)
    X_human = np.array(human_seqs_balanced)
    X_bot = np.array(bot_seqs_balanced)
    
    y_human = np.zeros(target_count)
    y_bot = np.ones(target_count)
    
    X = np.concatenate((X_human, X_bot), axis=0)
    y = np.concatenate((y_human, y_bot), axis=0)
    
    # --- 3. TRAIN/VAL/TEST SPLIT ---
    print("\n--- 3. Train / Val / Test split ---")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --- 4. DATA SCALING FOR RNN ---
    print("\n--- 4. Normalization (StandardScaler) ---")
    # Scaler expects 2D array, but we have 3D (Samples, 20, 2).
    # We temporarily flatten data, train scaler, apply, and restore shape.
    
    scaler = StandardScaler()
    
    # Train ONLY on Train set (flatten: N * 20 rows, 2 columns)
    X_train_flat = X_train.reshape(-1, 2)
    scaler.fit(X_train_flat)
    
    # Apply to all sets
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 2)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 2)).reshape(X_test.shape)
    
    # Save scaler parameters
    np.save(SCALER_FILE, [scaler.mean_, scaler.scale_])
    print(f"Scaler saved to {SCALER_FILE}")

    # --- 5. EXPORT TO PYTORCH TENSORS ---
    print("\n--- 5. Saving tensors ---")
    dataset_dict = {
        "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        "X_val": torch.tensor(X_val_scaled, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        "X_test": torch.tensor(X_test_scaled, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    }
    
    torch.save(dataset_dict, OUTPUT_FILE)
    print(f"✅ Done! Tensor dataset saved to {OUTPUT_FILE}")
    print(f"X input format: {dataset_dict['X_train'].shape} -> [Batch_Size, Sequence_Length, Features]")

if __name__ == "__main__":
    main()