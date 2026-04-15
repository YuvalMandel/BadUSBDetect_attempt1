import os
import glob
import argparse
import numpy as np
import joblib
import csv

# --- CONFIGURATION ---
KEYBOARD_MAP = {
    'q': (0, 3), 'w': (1, 3), 'e': (2, 3), 'r': (3, 3), 't': (4, 3),
    'y': (5, 3), 'u': (6, 3), 'i': (7, 3), 'o': (8, 3), 'p': (9, 3),
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'space': (4.5, 0)
}

# Output path for the results
RESULTS_DIR = os.path.join("..", "..", "results", "MLP", "regressor")
RESULTS_CSV = os.path.join(RESULTS_DIR, "regressor_test.csv")

def parse_and_extract_test(filepath):
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
                
                # During testing, take a wide range
                if 0 < delta < 2000: 
                    if last_keyup_name in KEYBOARD_MAP and key in KEYBOARD_MAP:
                        x1, y1 = KEYBOARD_MAP[last_keyup_name]
                        x2, y2 = KEYBOARD_MAP[key]
                        features.append([x1, y1, x2, y2, delta])
                        
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys: 
                active_keys.pop(key)
                
    return features

def evaluate_folder(folder_path, model, label_name):
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder {folder_path} not found.")
        return None

    # Просто берем ВСЕ .txt файлы, которые нам подготовил Makefile
    files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
        
    np.random.shuffle(files)
    files = files[:20] 
    
    all_errors = []
    
    for f in files:
        data = parse_and_extract_test(f)
        if not data: continue
        
        data = np.array(data)
        X = data[:, :4]
        y_true = data[:, 4]
        
        y_pred = model.predict(X)
        
        # Calculate error
        errors = np.abs(y_true - y_pred)
        all_errors.extend(errors)
        
    if not all_errors:
        print(f"{label_name}: No data for analysis.")
        return None
        
    all_errors = np.array(all_errors)
    mean_err = float(np.mean(all_errors))
    med_err = float(np.median(all_errors))
    std_err = float(np.std(all_errors))
    
    print(f"\n--- Results for {label_name} ---")
    print(f"Transitions processed: {len(all_errors)}")
    print(f"Mean error:   {mean_err:.2f} ms")
    print(f"Median error:  {med_err:.2f} ms")
    print(f"Error Std:  {std_err:.2f} ms")
    
    # Return dictionary for CSV export
    return {
        "Label": label_name,
        "Transitions_Processed": len(all_errors),
        "Mean_Error_ms": round(mean_err, 2),
        "Median_Error_ms": round(med_err, 2),
        "Std_Error_ms": round(std_err, 2)
    }

def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Test the trained Polynomial Regressor on human and bot datasets.")
    parser.add_argument("-hu", "--humans_dir", type=str, required=True, 
                        help="Path to the directory containing human test data (e.g., s2)")
    parser.add_argument("-b", "--bots_dir", type=str, required=True, 
                        help="Path to the directory containing bot test data (e.g., Synthetic_Bots_test)")
    parser.add_argument("-m", "--model_path", type=str, default="poly_regressor.pkl", 
                        help="Path to the trained model (default: poly_regressor.pkl)")
    
    args = parser.parse_args()

    print(f"Loading polynomial model (Taylor Series) from: {args.model_path}")
    try:
        model = joblib.load(args.model_path)
    except Exception as e:
        print(f"❌ Loading error: {e}")
        return
    
    # Evaluate both folders and collect results
    results = []
    
    human_results = evaluate_folder(args.humans_dir, model, "HUMANS")
    if human_results:
        results.append(human_results)
        
    bot_results = evaluate_folder(args.bots_dir, model, "BOTS")
    if bot_results:
        results.append(bot_results)
        
    # Save results to CSV if any data was processed
    if results:
        # Create directories if they do not exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Write to CSV
        headers = ["Label", "Transitions_Processed", "Mean_Error_ms", "Median_Error_ms", "Std_Error_ms"]
        with open(RESULTS_CSV, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n✅ Results successfully saved to: {os.path.abspath(RESULTS_CSV)}")

if __name__ == "__main__":
    main()
