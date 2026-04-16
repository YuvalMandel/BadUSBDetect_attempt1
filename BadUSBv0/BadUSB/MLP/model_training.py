import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend to avoid CXXABI errors
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# --- CONFIGURATION ---
# dataset_csv_generator.py produces these three files (person-disjoint splits)
TRAIN_CSV = "train_dataset.csv"
VAL_CSV   = "val_dataset.csv"
TEST_CSV  = "test_dataset.csv"
MODEL_SAVE_PATH = "badusb_model.pth"
SCALER_SAVE_PATH = "scaler_params.npy"

# Directory to save all PNG plots
RESULTS_DIR = os.path.join("..", "results", "MLP")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
INPUT_SIZE = 17 # Number of features (F_Mean ... D_MinW)

# Device setup (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. DATA PREPARATION
# ==============================================================================
class BadUSBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [batch, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data():
    """Load pre-split person-disjoint CSVs produced by dataset_csv_generator.py."""
    def load_csv(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run dataset_csv_generator.py first.")
        df = pd.read_csv(path)
        return df.drop("Label", axis=1).values, df["Label"].values

    print("Loading person-disjoint splits...")
    X_train, y_train = load_csv(TRAIN_CSV)
    X_val,   y_val   = load_csv(VAL_CSV)
    X_test,  y_test  = load_csv(TEST_CSV)

    print(f"Dataset sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # Scale on train, apply to val/test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    np.save(SCALER_SAVE_PATH, [scaler.mean_, scaler.scale_])
    print("Scaler parameters saved.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

# ==============================================================================
# 2. MODEL (4 Layers, Fully Connected)
# ==============================================================================
class BadUSBClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BadUSBClassifier, self).__init__()
        # 4 layers, narrowing architecture
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1) # Binary output

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Weight initialization (Xavier Initialization)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# ==============================================================================
# 3. TRAINING FUNCTIONS
# ==============================================================================
def calculate_metrics(y_true, y_pred):
    # Round predictions (0.1 -> 0, 0.9 -> 1)
    y_pred_tag = torch.round(y_pred)
    
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum / y_true.shape[0]
    
    # Convert to CPU numpy for sklearn metrics
    y_t = y_true.cpu().detach().numpy()
    y_p = y_pred_tag.cpu().detach().numpy()
    f1 = f1_score(y_t, y_p, zero_division=0)
    
    return acc, f1

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        acc, f1 = calculate_metrics(y_batch, y_pred)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1
        
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            acc, f1 = calculate_metrics(y_batch, y_pred)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1
            
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    save_path = os.path.join(RESULTS_DIR, 'training_plot_functions.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"📊 Saved loss/accuracy plot to: {save_path}")
    plt.close() # Free memory

def plot_confusion_matrices(model, loaders):
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Train Set", "Validation Set", "Test Set"]
    
    with torch.no_grad():
        for i, (loader, title) in enumerate(zip(loaders, titles)):
            all_preds = []
            all_labels = []
            
            for X_b, y_b in loader:
                X_b = X_b.to(device)
                preds = model(X_b)
                all_preds.extend(torch.round(preds).cpu().numpy())
                all_labels.extend(y_b.numpy())
            
            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(title)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot'])
            axes[i].set_yticklabels(['Human', 'Bot'])
            
            # Print report for Test set
            if title == "Test Set":
                print(f"\n--- Classification Report for {title} ---")
                print(classification_report(all_labels, all_preds, target_names=['Human', 'Bot']))

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'training_plot_acc.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"📊 Saved confusion matrices to: {save_path}")
    plt.close() # Free memory

# ==============================================================================
# 5. FILE-LEVEL EVALUATION
# ==============================================================================
def evaluate_file_level(model, scaler, split_json_path):
    """Window-level + file-level (first-crossing) F1 on raw unbalanced test files."""
    import json, joblib
    _mlp_dir = os.path.dirname(os.path.abspath(__file__))
    if _mlp_dir not in sys.path:
        sys.path.insert(0, _mlp_dir)
    from dataset_csv_generator import (parse_file, extract_features,
                                       WINDOW_SIZE, STEP_SIZE_HUMAN)

    if not os.path.exists(split_json_path):
        print(f"  [file-level] split JSON not found: {split_json_path}")
        return
    with open(split_json_path) as fh:
        split = json.load(fh)

    try:
        poly_model = joblib.load("poly_regressor.pkl")
    except Exception as e:
        print(f"  [file-level] Cannot load poly_regressor.pkl: {e}")
        return

    try:
        refs = np.load("reference_pool.npz", allow_pickle=True)
        ref_d = list(refs['dwell'])
        ref_f = list(refs['flight'])
    except Exception as e:
        print(f"  [file-level] Cannot load reference_pool.npz: {e}")
        return

    model.eval()
    file_labels, file_preds = [], []
    win_labels,  win_preds  = [], []

    with torch.no_grad():
        for is_bot, file_list in [(0, split['test']['humans']),
                                  (1, split['test']['bots'])]:
            for fp in file_list:
                d, f_det = parse_file(fp)
                min_len = min(len(d), len(f_det))
                if min_len < WINDOW_SIZE:
                    continue
                file_win_scores = []
                for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE_HUMAN):
                    feats = extract_features(d[i:i+WINDOW_SIZE],
                                             f_det[i:i+WINDOW_SIZE],
                                             ref_d, ref_f, poly_model)
                    if feats is None:
                        continue
                    x = scaler.transform([feats])
                    prob = model(torch.tensor(x, dtype=torch.float32).to(device)).item()
                    file_win_scores.append(prob)
                    win_labels.append(is_bot)
                    win_preds.append(1 if prob >= 0.5 else 0)
                if not file_win_scores:
                    continue
                file_pred = 1 if any(s >= 0.5 for s in file_win_scores) else 0
                file_labels.append(is_bot)
                file_preds.append(file_pred)

    if not file_labels:
        print("  [file-level] No test files processed.")
        return

    n_human = sum(1 for l in file_labels if l == 0)
    n_bot   = sum(1 for l in file_labels if l == 1)
    win_f1  = f1_score(win_labels,  win_preds,  zero_division=0)
    file_f1 = f1_score(file_labels, file_preds, zero_division=0)

    print(f"\n=== MLP WINDOW-LEVEL F1 (test, unbalanced, all files) ===")
    print(f"  F1: {win_f1:.4f}   ({len(win_labels)} windows)")
    print(f"=== MLP FILE-LEVEL F1 (test, first_crossing) ===")
    print(f"  F1: {file_f1:.4f}   ({n_human} human + {n_bot} bot files)")


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="../data_split.json",
                        help="Path to data_split.json for file-level evaluation")
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_data()
    
    train_dataset = BadUSBDataset(X_train, y_train)
    val_dataset = BadUSBDataset(X_val, y_val)
    test_dataset = BadUSBDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 2. Model
    model = BadUSBClassifier(INPUT_SIZE).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    # 3. Training loop
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Acc: {train_acc:.4f} (Val: {val_acc:.4f}) | "
              f"F1: {train_f1:.4f} (Val: {val_f1:.4f})")
        
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
    # 4. Plots
    plot_history(history)
    
    # 5. Confusion matrices
    # Load best model version before testing
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    plot_confusion_matrices(model, [train_loader, val_loader, test_loader])

    # 6. File-level + window-level evaluation on raw test files
    evaluate_file_level(model, scaler, args.split_json)

if __name__ == "__main__":
    main()
