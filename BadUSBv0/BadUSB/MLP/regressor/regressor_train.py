import os
import glob
import random
import argparse
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
# Polynomial degree (Taylor expansion). 
# 2 = 15 coefficients, 3 = 35 coefficients, 12 = 1820 coefficients. 
DEGREE = 12

KEYBOARD_MAP = {
    'q': (0, 3), 'w': (1, 3), 'e': (2, 3), 'r': (3, 3), 't': (4, 3),
    'y': (5, 3), 'u': (6, 3), 'i': (7, 3), 'o': (8, 3), 'p': (9, 3),
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'space': (4.5, 0)
}

def parse_and_extract_raw_coords(filepath):
    features = [] 
    active_keys = {} 
    last_keyup_ts = None
    last_keyup_name = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception: 
        return []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        key, action = parts[0].lower(), parts[1]
        try: 
            ts = int(parts[2])
        except ValueError: 
            continue

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts)
                
                # Learn only pure biomechanics (20ms to 400ms)
                if 20 < delta < 400: 
                    if last_keyup_name in KEYBOARD_MAP and key in KEYBOARD_MAP:
                        x1, y1 = KEYBOARD_MAP[last_keyup_name]
                        x2, y2 = KEYBOARD_MAP[key]
                        # Pass raw coordinates, the polynomial will find relationships itself
                        features.append([x1, y1, x2, y2, delta])
                        
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys: 
                active_keys.pop(key)
                
    return features

def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Train Polynomial Regressor for Fitts's Law approximation.")
    parser.add_argument("-hu", "--human_dir", type=str, required=True, 
                        help="Path to the directory containing human data (e.g., Balanced_Humans or s0)")
    parser.add_argument("-m", "--model_save_path", type=str, default="poly_regressor.pkl",
                        help="Path to save the trained model (default: poly_regressor.pkl)")
    parser.add_argument("-f", "--max_files", type=int, default=20,
                        help="Maximum number of files to use for training to prevent memory overload (default: 20)")
    args = parser.parse_args()

    print(f"--- 1. Collecting raw coordinates from '{args.human_dir}' ---")
    
    if not os.path.exists(args.human_dir):
        print(f"❌ ERROR: Data folder '{args.human_dir}' not found!")
        return

    # Find ALL *.txt files recursively in the provided directory (removed the *1.txt restriction)
    train_files = glob.glob(os.path.join(args.human_dir, "**", "*.txt"), recursive=True)
    
    if not train_files:
        print(f"❌ ERROR: No '.txt' files found in '{args.human_dir}'.")
        return

    # Shuffle and limit the number of files to match the original behavior (10 baseline + 10 rotation = 20)
    random.shuffle(train_files)
    train_files = train_files[:args.max_files]
    print(f"Selected {len(train_files)} files for training.")
    
    all_data = []
    for f in train_files:
        all_data.extend(parse_and_extract_raw_coords(f))
        
    all_data = np.array(all_data)
    
    if len(all_data) == 0:
        print("❌ ERROR: No valid transitions found in the selected files.")
        return
        
    print(f"Collected {len(all_data)} valid keystroke transitions.")

    X = all_data[:, :4] # x1, y1, x2, y2
    y = all_data[:, 4]  # delta

    # Create Pipeline: first raise to powers (Taylor), then find coefficients (Ridge)
    # Ridge (L2 regularization) is used to prevent coefficients from "exploding"
    model = make_pipeline(PolynomialFeatures(degree=DEGREE), Ridge(alpha=1.0))

    print("\n--- 2. Analytical calculation of coefficients... ---")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"✅ Equation found instantly! Average error on training set: {mae:.2f} ms")

    # Save the mathematical model
    joblib.dump(model, args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # (Optional) See how many coefficients were obtained
    poly_layer = model.named_steps['polynomialfeatures']
    print(f"Number of terms in polynomial: {poly_layer.n_output_features_}")

if __name__ == "__main__":
    main()
