"""
BadUSBv0/BadUSB/MLP/model_training.py
Train MLP with optional Optuna HP search and full/partial data mode.

Usage (from BadUSBv0/BadUSB/MLP/):
  python model_training.py --split-json ../data_split.json [--mode full] [--search] [--n-trials 50]

Requires optuna for --search:  pip install optuna
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# --- CONFIGURATION ---
TRAIN_CSV        = "train_dataset.csv"
VAL_CSV          = "val_dataset.csv"
TEST_CSV         = "test_dataset.csv"
MODEL_SAVE_PATH  = "badusb_model.pth"
SCALER_SAVE_PATH = "scaler_params.npy"

RESULTS_DIR   = os.path.join("..", "results", "MLP")

BATCH_SIZE    = 32
LEARNING_RATE = 0.001
EPOCHS        = 50
INPUT_SIZE    = 17   # D_Mean…Poly_Err_Std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. DATA
# ==============================================================================
class BadUSBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):   return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def prepare_data():
    """Load pre-split CSVs from dataset_csv_generator.py."""
    def load_csv(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run dataset_csv_generator.py first.")
        df = pd.read_csv(path)
        return df.drop("Label", axis=1).values, df["Label"].values

    print("Loading person-disjoint splits...")
    X_train, y_train = load_csv(TRAIN_CSV)
    X_val,   y_val   = load_csv(VAL_CSV)
    X_test,  y_test  = load_csv(TEST_CSV)

    print(f"  Train: {len(X_train)}  (pos={int(y_train.sum())}, neg={int((y_train==0).sum())})")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    np.save(SCALER_SAVE_PATH, [scaler.mean_, scaler.scale_])
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

# ==============================================================================
# 2. MODEL — flexible hidden layers + optional dropout
# ==============================================================================
class BadUSBClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 3. LOSS — weighted BCE for full (imbalanced) mode
# ==============================================================================
class WeightedBCELoss(nn.Module):
    """Drop-in replacement for BCELoss; scales positive-class loss by pos_weight."""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight  # scalar 1-D tensor

    def forward(self, pred, target):
        if self.pos_weight is None:
            return F.binary_cross_entropy(pred, target)
        w = torch.where(target == 1,
                        self.pos_weight.to(pred.device),
                        torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=w)

# ==============================================================================
# 4. TRAINING HELPERS
# ==============================================================================
def _metrics(y_true, y_pred):
    tag = torch.round(y_pred.detach())
    acc = (tag == y_true).float().mean().item()
    f1  = f1_score(y_true.cpu().detach().numpy(), tag.cpu().numpy(), zero_division=0)
    return acc, f1

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = acc_sum = f1_sum = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward(); optimizer.step()
        acc, f1 = _metrics(yb, pred)
        loss_sum += loss.item(); acc_sum += acc; f1_sum += f1
    n = len(loader)
    return loss_sum / n, acc_sum / n, f1_sum / n

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = acc_sum = f1_sum = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            acc, f1 = _metrics(yb, pred)
            loss_sum += loss.item(); acc_sum += acc; f1_sum += f1
    n = len(loader)
    return loss_sum / n, acc_sum / n, f1_sum / n

# ==============================================================================
# 5. OPTUNA HP SEARCH  (val F1 objective — test never touched)
# ==============================================================================
def hp_search(X_train, y_train, X_val, y_val, n_trials, pos_weight):
    try:
        import optuna
    except ImportError:
        print("  [hp_search] optuna not installed. Run: pip install optuna")
        print("  [hp_search] Falling back to default HPs.")
        return (64, 32, 16), 0.0, LEARNING_RATE, BATCH_SIZE

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    criterion = WeightedBCELoss(pos_weight)
    val_loader = DataLoader(BadUSBDataset(X_val, y_val), batch_size=512)

    def objective(trial):
        n_hidden    = trial.suggest_int("n_hidden", 1, 3)
        # Always suggest 3 layer widths; use only the first n_hidden
        all_widths  = [trial.suggest_categorical(f"h{i}", [32, 64, 128, 256]) for i in range(3)]
        hidden_dims = tuple(all_widths[:n_hidden])
        dropout     = trial.suggest_float("dropout",    0.0,  0.4,  step=0.05)
        lr          = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        m   = BadUSBClassifier(INPUT_SIZE, hidden_dims, dropout).to(device)
        opt = optim.Adam(m.parameters(), lr=lr)
        ldr = DataLoader(BadUSBDataset(X_train, y_train),
                         batch_size=batch_size, shuffle=True)

        for _ in range(30):   # 30-epoch quick eval per trial
            train_epoch(m, ldr, criterion, opt)

        m.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = torch.round(m(Xb.to(device))).cpu().numpy().flatten()
                all_p.extend(p); all_y.extend(yb.numpy().flatten())
        return f1_score(all_y, all_p, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    bp = study.best_params
    print(f"\nHP search done — best val F1: {study.best_value:.4f}")
    n_h = bp["n_hidden"]
    print(f"  n_hidden={n_h}, dims={[bp[f'h{i}'] for i in range(n_h)]}, "
          f"dropout={bp['dropout']:.2f}, lr={bp['lr']:.5f}, batch={bp['batch_size']}")

    hidden_dims = tuple(bp[f"h{i}"] for i in range(n_h))
    return hidden_dims, bp["dropout"], bp["lr"], bp["batch_size"]

# ==============================================================================
# 6. VISUALISATION
# ==============================================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'],  label='Train'); ax2.plot(history['val_acc'],  label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'training_plot_functions.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"Loss/accuracy plot -> {p}")

def plot_confusion_matrices(model, loaders):
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Train Set", "Validation Set", "Test Set"]
    with torch.no_grad():
        for i, (loader, title) in enumerate(zip(loaders, titles)):
            all_p, all_y = [], []
            for Xb, yb in loader:
                all_p.extend(torch.round(model(Xb.to(device))).cpu().numpy())
                all_y.extend(yb.numpy())
            sns.heatmap(confusion_matrix(all_y, all_p), annot=True, fmt='d',
                        cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(title); axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot']); axes[i].set_yticklabels(['Human', 'Bot'])
            if title == "Test Set":
                print(f"\n--- Classification Report ({title}) ---")
                print(classification_report(all_y, all_p, target_names=['Human', 'Bot']))
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'training_plot_acc.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"Window-level confusion matrices -> {p}")

# ==============================================================================
# 7. FILE-LEVEL EVALUATION (all splits, 2×3 confusion matrix)
# ==============================================================================
def evaluate_file_level(model, scaler, split_json_path):
    """Window-level + file-level (first-crossing) F1 on raw unbalanced files, all splits."""
    import joblib
    _mlp_dir = os.path.dirname(os.path.abspath(__file__))
    if _mlp_dir not in sys.path:
        sys.path.insert(0, _mlp_dir)
    from dataset_csv_generator import (parse_file, extract_features,
                                       WINDOW_SIZE, STEP_SIZE_HUMAN)

    if not os.path.exists(split_json_path):
        print(f"  [file-level] split JSON not found: {split_json_path}"); return
    with open(split_json_path) as fh:
        split = json.load(fh)

    try:
        poly_model = joblib.load("poly_regressor.pkl")
    except Exception as e:
        print(f"  [file-level] Cannot load poly_regressor.pkl: {e}"); return

    try:
        refs  = np.load("reference_pool.npz", allow_pickle=True)
        ref_d = list(refs['dwell']); ref_f = list(refs['flight'])
    except Exception as e:
        print(f"  [file-level] Cannot load reference_pool.npz: {e}"); return

    model.eval()
    results = {}
    print("\n=== MLP File-level + Window-level evaluation (all splits) ===")

    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            file_labels, file_preds = [], []
            win_labels,  win_preds  = [], []
            for is_bot, flist in [(0, split[split_name]['humans']),
                                  (1, split[split_name]['bots'])]:
                for fp in flist:
                    d, f_det = parse_file(fp)
                    min_len  = min(len(d), len(f_det))
                    if min_len < WINDOW_SIZE: continue
                    scores = []
                    for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE_HUMAN):
                        feats = extract_features(d[i:i+WINDOW_SIZE], f_det[i:i+WINDOW_SIZE],
                                                 ref_d, ref_f, poly_model)
                        if feats is None: continue
                        prob = model(torch.tensor(scaler.transform([feats]),
                                                  dtype=torch.float32).to(device)).item()
                        scores.append(prob)
                        win_labels.append(is_bot)
                        win_preds.append(1 if prob >= 0.5 else 0)
                    if not scores: continue
                    file_labels.append(is_bot)
                    file_preds.append(1 if any(s >= 0.5 for s in scores) else 0)

            results[split_name] = dict(wl=win_labels, wp=win_preds,
                                       fl=file_labels, fp=file_preds)
            if file_labels:
                print(f"  {split_name.upper():5s}: Win-F1={f1_score(win_labels,  win_preds,  zero_division=0):.4f} "
                      f"({len(win_labels)} windows) | "
                      f"File-F1={f1_score(file_labels, file_preds, zero_division=0):.4f} "
                      f"({len(file_labels)} files)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("MLP — All Splits Confusion Matrices (Window / File level)", fontsize=14)
    for col, sn in enumerate(("train", "val", "test")):
        r = results[sn]
        for row, (labels, preds, title) in enumerate([
            (r['wl'], r['wp'], "Window-level"),
            (r['fl'], r['fp'], "File-level (first_crossing)"),
        ]):
            ax = axes[row, col]
            if labels:
                sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d',
                            cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{sn.capitalize()} — {title}\n"
                         f"F1={f1_score(labels, preds, zero_division=0):.4f}")
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_xticklabels(['Human', 'Bot']); ax.set_yticklabels(['Human', 'Bot'])
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'mlp_all_splits_confusion.png')
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Confusion matrices -> {p}")

# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="../data_split.json")
    parser.add_argument("--mode", choices=["partial", "full"], default="partial",
                        help="'partial': balanced CSVs (default). "
                             "'full': imbalanced CSVs, uses pos_weight in loss.")
    parser.add_argument("--search", action="store_true",
                        help="Run Optuna HP search (val F1 objective). "
                             "Requires: pip install optuna")
    parser.add_argument("--n-configs", type=int, default=50,
                        help="Number of HP configurations to try (default 50).")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_data()

    # 2. Class weight for imbalanced (full) mode
    pos_weight = None
    if args.mode == "full":
        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        print(f"Full mode — pos_weight: {pos_weight.item():.2f}  "
              f"(neg={n_neg}, pos={n_pos})")

    criterion = WeightedBCELoss(pos_weight)

    # 3. HP search or defaults
    if args.search:
        print(f"\nRunning Optuna HP search ({args.n_configs} configs, objective: val F1)...")
        hidden_dims, dropout, lr, batch_size = hp_search(
            X_train, y_train, X_val, y_val, args.n_configs, pos_weight
        )
        best_hps = {"hidden_dims": list(hidden_dims), "dropout": dropout,
                    "lr": lr, "batch_size": batch_size}
        hp_path = os.path.join(RESULTS_DIR, "mlp_best_hps.json")
        with open(hp_path, "w") as fh:
            json.dump(best_hps, fh, indent=2)
        print(f"Best HPs -> {hp_path}")
    else:
        hidden_dims = (64, 32, 16)
        dropout, lr, batch_size = 0.0, LEARNING_RATE, BATCH_SIZE

    # 4. Data loaders
    train_loader = DataLoader(BadUSBDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(BadUSBDataset(X_val,  y_val),  batch_size=batch_size)
    test_loader  = DataLoader(BadUSBDataset(X_test, y_test), batch_size=batch_size)

    # 5. Train final model
    model     = BadUSBClassifier(INPUT_SIZE, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining: hidden={hidden_dims}, dropout={dropout:.2f}, "
          f"lr={lr:.5f}, batch={batch_size}, epochs={EPOCHS}")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        tl, ta, tf = train_epoch(model, train_loader, criterion, optimizer)
        vl, va, vf = evaluate(model, val_loader, criterion)
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl);   history['val_acc'].append(va)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss {tl:.4f} (Val {vl:.4f}) | "
              f"Acc {ta:.4f} (Val {va:.4f}) | "
              f"F1 {tf:.4f} (Val {vf:.4f})")

    print(f"\nModel saved -> {MODEL_SAVE_PATH}")

    # 6. Load best checkpoint, plot, evaluate
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    plot_history(history)
    plot_confusion_matrices(model, [train_loader, val_loader, test_loader])
    evaluate_file_level(model, scaler, args.split_json)


if __name__ == "__main__":
    main()
