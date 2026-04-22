import os
import json
import argparse
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- SETTINGS ---
SEQ_LEN        = 15   # Sequence length
STEP_SIZE      = 1    # Sliding window step
MAX_KEYSTROKES = None  # truncate every file to first N keystrokes; None = no limit

# Set per mode after arg parsing (see main())
OUTPUT_FILE = None
SCALER_FILE = None

# ==============================================================================
# 1. PARSER (Returns raw milliseconds)
# ==============================================================================
def parse_file(filepath):
    dwells  = []
    flights = []
    active_keys = {}
    last_keyup  = None

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
                delta   = (ts - down_ts) / scale
                if 0 < delta < 3000: dwells.append(delta)

    min_len = min(len(dwells), len(flights)) if MAX_KEYSTROKES is None else min(len(dwells), len(flights), MAX_KEYSTROKES)
    return np.array(dwells[:min_len]), np.array(flights[:min_len])

# ==============================================================================
# 2. SEQUENCE EXTRACTION
# ==============================================================================
def files_to_sequences(file_list):
    seqs     = []
    file_ids = []
    for file_idx, filepath in enumerate(tqdm(file_list, desc="  Parsing")):
        d, f = parse_file(filepath)
        if len(d) < SEQ_LEN:
            continue
        for i in range(0, len(d) - SEQ_LEN, STEP_SIZE):
            seq = np.column_stack((d[i:i+SEQ_LEN], f[i:i+SEQ_LEN]))
            seqs.append(seq)
            file_ids.append(file_idx)
    return seqs, file_ids

# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="../data_split.json",
                        help="Path to data_split.json (default: ../data_split.json)")
    parser.add_argument("--mode", choices=["partial", "full"], default="partial",
                        help="'partial' (default): balance by undersampling. "
                             "'full': all windows, imbalanced — training script uses class weights.")
    parser.add_argument("--tag", default="", help="Extra suffix for output filenames (e.g. 'fullkey')")
    args = parser.parse_args()

    global OUTPUT_FILE, SCALER_FILE
    tag_suffix = f"_{args.tag}" if args.tag else ""
    OUTPUT_FILE = f"rnn_dataset_{args.mode}{tag_suffix}.pt"
    SCALER_FILE = f"rnn_scaler_params_{args.mode}{tag_suffix}.npy"

    split_json = os.path.abspath(args.split_json)
    if not os.path.exists(split_json):
        print(f"ERROR: {split_json} not found. Run split_persons.py first.")
        return
    with open(split_json) as fp:
        split = json.load(fp)

    # ── Extract sequences per split ───────────────────────────────────────────
    tensors = {}
    X_train_seqs, X_val_seqs, X_test_seqs = None, None, None
    fids_val = None

    for name in ("train", "val", "test"):
        print(f"\n[{name}]  humans={len(split[name]['humans'])}, "
              f"bots={len(split[name]['bots'])}")

        human_seqs, human_fids_raw = files_to_sequences(split[name]["humans"])
        bot_seqs,   bot_fids_raw   = files_to_sequences(split[name]["bots"])
        # offset bot IDs so they don't collide with human IDs within the split
        bot_offset   = len(split[name]["humans"])
        bot_fids_raw = [fid + bot_offset for fid in bot_fids_raw]

        print(f"  Windows: humans={len(human_seqs)}, bots={len(bot_seqs)}")

        if args.mode == "partial":
            n = min(len(human_seqs), len(bot_seqs))
            if n == 0:
                print(f"  WARNING: empty split '{name}', skipping.")
                continue
            import random
            random.seed(42)
            h_pairs = random.sample(list(zip(human_seqs, human_fids_raw)), n)
            b_pairs = random.sample(list(zip(bot_seqs,   bot_fids_raw)),   n)
            h_use  = [p[0] for p in h_pairs];  h_fids = [p[1] for p in h_pairs]
            b_use  = [p[0] for p in b_pairs];  b_fids = [p[1] for p in b_pairs]
            print(f"  Partial mode: balanced to {n} per class")
        else:  # full
            if len(human_seqs) == 0 or len(bot_seqs) == 0:
                print(f"  WARNING: empty split '{name}', skipping.")
                continue
            h_use  = human_seqs;  h_fids = human_fids_raw
            b_use  = bot_seqs;    b_fids = bot_fids_raw
            print(f"  Full mode: {len(h_use)} human + {len(b_use)} bot windows (imbalanced)")

        X    = np.array(h_use + b_use)
        y    = np.concatenate([np.zeros(len(h_use)), np.ones(len(b_use))])
        fids = np.array(h_fids + b_fids)

        # Shuffle
        idx = np.random.default_rng(42).permutation(len(X))
        X, y, fids = X[idx], y[idx], fids[idx]

        if name == "train":
            X_train_seqs = X
            y_train = y
        elif name == "val":
            X_val_seqs = X
            y_val = y
            fids_val = fids
        else:
            X_test_seqs = X
            y_test = y
            fids_test = fids

    if X_train_seqs is None:
        print("ERROR: no training data."); return

    # ── Scaling (fit on train only) ───────────────────────────────────────────
    print("\nFitting StandardScaler on train set...")
    scaler = StandardScaler()
    scaler.fit(X_train_seqs.reshape(-1, 2))
    np.save(SCALER_FILE, [scaler.mean_, scaler.scale_])

    def scale(X):
        return scaler.transform(X.reshape(-1, 2)).reshape(X.shape)

    X_train_s = scale(X_train_seqs)
    X_val_s   = scale(X_val_seqs)   if X_val_seqs  is not None else None
    X_test_s  = scale(X_test_seqs)  if X_test_seqs is not None else None

    # ── Save ──────────────────────────────────────────────────────────────────
    dataset = {
        "X_train": torch.tensor(X_train_s, dtype=torch.float32),
        "y_train": torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1),
    }
    if X_val_s is not None:
        dataset["X_val"] = torch.tensor(X_val_s, dtype=torch.float32)
        dataset["y_val"] = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
        if fids_val is not None:
            dataset["file_ids_val"] = torch.tensor(fids_val, dtype=torch.int64)
    if X_test_s is not None:
        dataset["X_test"] = torch.tensor(X_test_s, dtype=torch.float32)
        dataset["y_test"] = torch.tensor(y_test,   dtype=torch.float32).unsqueeze(1)
        if fids_test is not None:
            dataset["file_ids_test"] = torch.tensor(fids_test, dtype=torch.int64)

    torch.save(dataset, OUTPUT_FILE)

    print(f"\nTrain: {X_train_s.shape}  Val: "
          f"{X_val_s.shape if X_val_s is not None else 'n/a'}  "
          f"Test: {X_test_s.shape if X_test_s is not None else 'n/a'}")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
