import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Ensure htm_common is available
try:
    import htm_common
except ImportError:
    print("htm_common.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# HTM Imports
try:
    from htm.bindings.sdr import SDR
except ImportError:
    print("HTM libraries not found. Please install htm.core.")
    sys.exit(1)

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def run_inference(model_data, file_features, file_list, is_bot, desc="Inference"):
    sp = model_data['sp']
    tm = model_data['tm']
    encoder = model_data['encoder']
    input_width = model_data['input_width']
    
    # Create SDR object for input
    active_columns = SDR(sp.getColumnDimensions())
    
    scores = []
    labels = []
    
    print(f"Running {desc} on {len(file_list)} files...")
    
    for filepath in file_list:
        if filepath not in file_features:
            continue
            
        # Reset Temporal Memory for each new sequence
        tm.reset()
        
        file_scores = []
        for feats in file_features[filepath]:
            # Encode
            dense_input = encoder.encode(feats)
            enc_sdr = SDR(input_width)
            enc_sdr.dense = dense_input
            
            # Spatial Pooler (learning disabled)
            sp.compute(enc_sdr, False, active_columns)
            
            # Temporal Memory (learning disabled)
            tm.compute(active_columns, learn=False)
            
            file_scores.append(tm.anomaly)
            
        if file_scores:
            # Skip warmup (first 5 steps)
            valid_scores = file_scores[5:] if len(file_scores) > 5 else file_scores
            scores.append(np.mean(valid_scores))
            labels.append(1 if is_bot else 0)
            
    return scores, labels

def main():
    parser = argparse.ArgumentParser(description="Test a trained HTM model on Test/Val sets.")
    parser.add_argument("--model", default="models/cfg0007_sp40_enc16w7_tm16_act13_vf11.0000_tf11.0000.pkl", help="Path to the .pkl model file")
    parser.add_argument("--split", default="split.pkl", help="Path to split.pkl")
    parser.add_argument("--features", default="features_cache.pkl", help="Path to features_cache.pkl")
    
    args = parser.parse_args()
    
    # Check files
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
    if not os.path.exists(args.split):
        print(f"Error: Split file '{args.split}' not found.")
        sys.exit(1)
    if not os.path.exists(args.features):
        print(f"Error: Features file '{args.features}' not found.")
        sys.exit(1)
        
    # Load Data
    print("Loading data...")
    with open(args.split, 'rb') as f:
        split = pickle.load(f)
    with open(args.features, 'rb') as f:
        features_cache = pickle.load(f)
        
    # Load Model
    model_data = load_model(args.model)
    best_thresh = model_data.get('best_thresh', 0.5)
    print(f"Model loaded. Using threshold: {best_thresh:.4f}")
    
    # Run Inference
    # Validation Set
    val_human = split['val_human']
    val_bots = split['val_bots']
    
    val_h_scores, val_h_labels = run_inference(model_data, features_cache, val_human, False, "Validation (Human)")
    val_b_scores, val_b_labels = run_inference(model_data, features_cache, val_bots, True, "Validation (Bot)")
    
    all_val_scores = val_h_scores + val_b_scores
    all_val_labels = val_h_labels + val_b_labels
    
    # Test Set
    test_human = split['test_human']
    test_bots = split['test_bots']
    
    test_h_scores, test_h_labels = run_inference(model_data, features_cache, test_human, False, "Test (Human)")
    test_b_scores, test_b_labels = run_inference(model_data, features_cache, test_bots, True, "Test (Bot)")
    
    all_test_scores = test_h_scores + test_b_scores
    all_test_labels = test_h_labels + test_b_labels
    
    # Calculate Metrics
    val_preds = [1 if s >= best_thresh else 0 for s in all_val_scores]
    test_preds = [1 if s >= best_thresh else 0 for s in all_test_scores]
    
    val_f1 = f1_score(all_val_labels, val_preds)
    test_f1 = f1_score(all_test_labels, test_preds)
    
    print("\n" + "="*40)
    print(f"RESULTS")
    print("="*40)
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Test F1:       {test_f1:.4f}")
    print("-" * 40)
    print("Test Classification Report:")
    print(classification_report(all_test_labels, test_preds, target_names=['Human', 'Bot']))
    
    # Plotting
    print("Generating confusion matrices plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Validation Matrix
    cm_val = confusion_matrix(all_val_labels, val_preds)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f"Validation Set (F1={val_f1:.4f})")
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Human', 'Bot'])
    axes[0].set_yticklabels(['Human', 'Bot'])
    
    # Test Matrix
    cm_test = confusion_matrix(all_test_labels, test_preds)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title(f"Test Set (F1={test_f1:.4f})")
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['Human', 'Bot'])
    axes[1].set_yticklabels(['Human', 'Bot'])
    
    plt.tight_layout()
    plot_path = "test_model_confusion.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
