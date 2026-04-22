"""
BadUSBv0/BadUSB/GRU/train_gru.py
Train GRU with optional Optuna HP search and full/partial data mode.

Usage (from BadUSBv0/BadUSB/GRU/):
  python train_gru.py --split-json ../data_split.json [--mode full] [--search] [--n-trials 50]

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
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# --- CONFIGURATION ---
# Set per mode after arg parsing (see main())
DATASET_FILE    = None
MODEL_SAVE_PATH = None
RESULTS_DIR     = os.path.join("..", "results", "GRU")

BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 50

INPUT_DIM  = 2    # dwell + flight
HIDDEN_DIM = 64
NUM_LAYERS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. MODEL — GRU with optional dropout
# ==============================================================================
class GRUBotDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # dropout only applied between layers (ignored when num_layers=1)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc      = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)  # applied before fc

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])   # last time step
        return self.sigmoid(self.fc(out))

# ==============================================================================
# 2. LOSS — weighted BCE for full (imbalanced) mode
# ==============================================================================
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        if self.pos_weight is None:
            return F.binary_cross_entropy(pred, target)
        w = torch.where(target == 1,
                        self.pos_weight.to(pred.device),
                        torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=w)

# ==============================================================================
# 3. TRAINING HELPERS
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
# 4. OPTUNA HP SEARCH  (val F1 objective — test never touched)
# ==============================================================================
def hp_search(X_train, y_train, X_val, y_val, file_ids_val, n_trials, pos_weight):
    try:
        import optuna
    except ImportError:
        print("  [hp_search] optuna not installed. Run: pip install optuna")
        print("  [hp_search] Falling back to default HPs.")
        return HIDDEN_DIM, NUM_LAYERS, 0.0, LEARNING_RATE, BATCH_SIZE, 0.5

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    criterion  = WeightedBCELoss(pos_weight)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # CUDA context + fork (used by Optuna n_jobs>1) don't mix; run serial on GPU
        n_jobs = 1
        print(f"  GPU detected ({torch.cuda.get_device_name(0)}): serial trials on CUDA")
    else:
        n_jobs = max(1, min(n_trials, (os.cpu_count() or 1) // 2))
        print(f"  CPU mode: n_jobs={n_jobs} (CPUs={os.cpu_count()})")

    def objective(trial):
        if not use_gpu:
            torch.set_num_threads(1)  # prevent over-subscription with parallel CPU trials
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = 2  # fixed: 2 layers was optimal in both partial and full mode
        dropout    = trial.suggest_float("dropout",   0.1,  0.4,  step=0.05)
        lr         = trial.suggest_float("lr",        5e-5, 2e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        threshold  = trial.suggest_float("threshold", 0.3,  0.85, step=0.05)

        m   = GRUBotDetector(INPUT_DIM, hidden_dim, num_layers, dropout).to(device)
        opt = optim.Adam(m.parameters(), lr=lr)
        ldr = DataLoader(TensorDataset(X_train, y_train),
                         batch_size=batch_size, shuffle=True)

        for _ in range(30):   # 30-epoch quick eval per trial
            train_epoch(m, ldr, criterion, opt)

        # File-level first-crossing F1 (real deployment metric)
        m.eval()
        all_probs = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                all_probs.extend(m(Xb.to(device)).cpu().numpy().flatten())
        all_probs = np.array(all_probs)

        file_probs  = {}
        file_labels = {}
        for prob, label, fid in zip(all_probs, y_val.numpy().flatten(),
                                    file_ids_val.numpy()):
            fid = int(fid)
            if fid not in file_probs:
                file_probs[fid]  = []
                file_labels[fid] = int(label)
            file_probs[fid].append(prob)

        ftrue, fpred = [], []
        for fid in file_probs:
            ftrue.append(file_labels[fid])
            fpred.append(1 if any(p >= threshold for p in file_probs[fid]) else 0)
        return f1_score(ftrue, fpred, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    def trial_callback(study, trial):
        print(f"  Trial {trial.number+1:>3}/{n_trials} | file_f1={trial.value:.4f} | "
              f"best={study.best_value:.4f} | {trial.params}", flush=True)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=False, callbacks=[trial_callback])

    bp = study.best_params
    print(f"\nHP search done — best val file-F1: {study.best_value:.4f}")
    print(f"  hidden={bp['hidden_dim']}, layers=2, "
          f"dropout={bp['dropout']:.2f}, lr={bp['lr']:.5f}, batch={bp['batch_size']}, "
          f"threshold={bp['threshold']:.2f}")
    return bp["hidden_dim"], 2, bp["dropout"], bp["lr"], bp["batch_size"], bp["threshold"]

# ==============================================================================
# 5. VISUALISATION
# ==============================================================================
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('GRU Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'],  label='Train'); ax2.plot(history['val_acc'],  label='Val')
    ax2.set_title('GRU Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'gru_training_history.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"Training history -> {p}")

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
            sns.heatmap(confusion_matrix(all_y, all_p), annot=True, fmt='g',
                        cmap='Purples', ax=axes[i], cbar=False)
            axes[i].set_title(title); axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot']); axes[i].set_yticklabels(['Human', 'Bot'])
            if title == "Test Set":
                print(f"\n--- Classification Report ({title}) ---")
                print(classification_report(all_y, all_p, target_names=['Human', 'Bot']))
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'gru_confusion_matrices.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"Window-level confusion matrices -> {p}")

# ==============================================================================
# 6. FILE-LEVEL EVALUATION (all splits, 2×3 confusion matrix)
# ==============================================================================
def evaluate_file_level(model, split_json_path, mode="partial", threshold=0.5, tag=""):
    """Window-level + file-level (first-crossing) F1 on raw unbalanced files, all splits."""
    _gru_dir = os.path.dirname(os.path.abspath(__file__))
    if _gru_dir not in sys.path:
        sys.path.insert(0, _gru_dir)
    from translate_to_tensors import parse_file, SEQ_LEN, STEP_SIZE

    if not os.path.exists(split_json_path):
        print(f"  [file-level] split JSON not found: {split_json_path}"); return
    with open(split_json_path) as fh:
        split = json.load(fh)

    tag_suffix = f"_{tag}" if tag else ""
    try:
        sp = np.load(f"rnn_scaler_params_{mode}{tag_suffix}.npy", allow_pickle=True)
        mean, scale = sp[0], sp[1]
    except Exception as e:
        print(f"  [file-level] Cannot load rnn_scaler_params_{mode}{tag_suffix}.npy: {e}"); return

    model.eval()
    results = {}
    print("\n=== GRU File-level + Window-level evaluation (all splits) ===")

    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            file_labels, file_preds = [], []
            win_labels,  win_preds  = [], []
            for is_bot, flist in [(0, split[split_name]['humans']),
                                  (1, split[split_name]['bots'])]:
                for fp in flist:
                    d, f = parse_file(fp)
                    if len(d) < SEQ_LEN: continue
                    scores = []
                    for i in range(0, len(d) - SEQ_LEN, STEP_SIZE):
                        seq = np.column_stack((d[i:i+SEQ_LEN], f[i:i+SEQ_LEN]))
                        seq = (seq - mean) / scale
                        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                        prob = model(x).item()
                        scores.append(prob)
                        win_labels.append(is_bot)
                        win_preds.append(1 if prob >= threshold else 0)
                    if not scores: continue
                    file_labels.append(is_bot)
                    file_preds.append(1 if any(s >= threshold for s in scores) else 0)

            results[split_name] = dict(wl=win_labels, wp=win_preds,
                                       fl=file_labels, fp=file_preds)
            if file_labels:
                print(f"  {split_name.upper():5s}: Win-F1={f1_score(win_labels,  win_preds,  zero_division=0):.4f} "
                      f"({len(win_labels)} windows) | "
                      f"File-F1={f1_score(file_labels, file_preds, zero_division=0):.4f} "
                      f"({len(file_labels)} files)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GRU — All Splits Confusion Matrices (Window / File level)", fontsize=14)
    for col, sn in enumerate(("train", "val", "test")):
        r = results[sn]
        for row, (labels, preds, title) in enumerate([
            (r['wl'], r['wp'], "Window-level"),
            (r['fl'], r['fp'], "File-level (first_crossing)"),
        ]):
            ax = axes[row, col]
            if labels:
                sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d',
                            cmap='Purples', ax=ax, cbar=False)
            ax.set_title(f"{sn.capitalize()} — {title}\n"
                         f"F1={f1_score(labels, preds, zero_division=0):.4f}")
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_xticklabels(['Human', 'Bot']); ax.set_yticklabels(['Human', 'Bot'])
    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, 'gru_all_splits_confusion.png')
    plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
    print(f"Confusion matrices -> {p}")

# ==============================================================================
# 7. THRESHOLD TUNING (no retraining)
# ==============================================================================
def tune_threshold_mode(args):
    tag_suffix   = f"_{args.tag}" if args.tag else ""
    dataset_file = f"rnn_dataset_{args.mode}{tag_suffix}.pt"
    model_path   = f"gru_model_{args.mode}{tag_suffix}.pth"

    print(f"Loading tensors from {dataset_file}...")
    data = torch.load(dataset_file, weights_only=False)
    X_val, y_val     = data["X_val"], data["y_val"]
    file_ids_val     = data.get("file_ids_val", torch.arange(len(y_val)))

    if args.hps_json:
        with open(args.hps_json) as fh:
            hps = json.load(fh)
        hidden_dim, num_layers, dropout = hps["hidden_dim"], hps["num_layers"], hps["dropout"]
    else:
        hidden_dim, num_layers, dropout = HIDDEN_DIM, NUM_LAYERS, 0.0

    model = GRUBotDetector(INPUT_DIM, hidden_dim, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=1024)
    probs = []
    with torch.no_grad():
        for Xb, _ in val_loader:
            probs.extend(model(Xb.to(device)).cpu().numpy().flatten())
    probs  = np.array(probs)
    y_np   = y_val.numpy().flatten()
    fids   = file_ids_val.numpy()

    file_probs, file_labels = {}, {}
    for p, lbl, fid in zip(probs, y_np, fids):
        fid = int(fid)
        file_probs.setdefault(fid, []).append(float(p))
        file_labels.setdefault(fid, int(lbl))

    print(f"\nThreshold sweep on val ({len(file_probs)} files, "
          f"{sum(v for v in file_labels.values())} bot files):")
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        ftrue = [file_labels[fid] for fid in file_probs]
        fpred = [1 if any(p >= t for p in file_probs[fid]) else 0 for fid in file_probs]
        f1 = f1_score(ftrue, fpred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
            print(f"  thresh={t:.2f}  val_file_F1={f1:.4f}  ***")

    print(f"\nBest threshold: {best_t:.2f}  val_file_F1={best_f1:.4f}")

    if args.hps_json and os.path.exists(args.hps_json):
        with open(args.hps_json) as fh:
            hps = json.load(fh)
        hps["threshold"] = best_t
        with open(args.hps_json, "w") as fh:
            json.dump(hps, fh, indent=2)
        print(f"Updated threshold in {args.hps_json}")

    # Fast evaluation from stored tensors (val has file_ids; test uses sequential fid)
    X_test = data.get("X_test")
    y_test = data.get("y_test")
    for split_name, X_s, y_s, fids_s in [
        ("val",  X_val,  y_val,  file_ids_val),
        ("test", X_test, y_test, None),
    ]:
        if X_s is None: continue
        loader_s = DataLoader(TensorDataset(X_s, y_s), batch_size=1024)
        p_s = []
        with torch.no_grad():
            for Xb, _ in loader_s:
                p_s.extend(model(Xb.to(device)).cpu().numpy().flatten())
        p_s   = np.array(p_s)
        y_np  = y_s.numpy().flatten()
        if fids_s is None:
            fids_s = torch.arange(len(y_np))
        fp, fl = {}, {}
        for p, lbl, fid in zip(p_s, y_np, fids_s.numpy()):
            fid = int(fid)
            fp.setdefault(fid, []).append(float(p))
            fl.setdefault(fid, int(lbl))
        ftrue   = [fl[fid] for fid in fp]
        fpred   = [1 if any(p >= best_t for p in fp[fid]) else 0 for fid in fp]
        win_f1  = f1_score(y_np, (p_s >= best_t).astype(int), zero_division=0)
        file_f1 = f1_score(ftrue, fpred, zero_division=0)
        n_bot   = sum(fl.values())
        print(f"  {split_name.upper():5s}: Win-F1={win_f1:.4f} ({len(y_np)} windows) | "
              f"File-F1={file_f1:.4f} ({len(fp)} files, {n_bot} bot)")


# ==============================================================================
# 8. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="../data_split.json")
    parser.add_argument("--mode", choices=["partial", "full"], default="partial",
                        help="'partial': balanced tensors (default). "
                             "'full': imbalanced, uses pos_weight in loss.")
    parser.add_argument("--tag", default="", help="Extra suffix for output filenames (e.g. 'fullkey')")
    parser.add_argument("--search", action="store_true",
                        help="Run Optuna HP search (val F1 objective). "
                             "Requires: pip install optuna")
    parser.add_argument("--n-configs", type=int, default=50,
                        help="Number of HP configurations to try (default 50).")
    parser.add_argument("--hps-json", default=None,
                        help="Load best HPs from a JSON file instead of searching or using defaults.")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Skip training; sweep val thresholds on saved model and report test results.")
    args = parser.parse_args()

    global DATASET_FILE, MODEL_SAVE_PATH
    tag_suffix = f"_{args.tag}" if args.tag else ""
    DATASET_FILE    = f"rnn_dataset_{args.mode}{tag_suffix}.pt"
    MODEL_SAVE_PATH = f"gru_model_{args.mode}{tag_suffix}.pth"

    if args.tune_threshold:
        tune_threshold_mode(args)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load tensors
    print("Loading tensors...")
    try:
        data = torch.load(DATASET_FILE, weights_only=False)
    except FileNotFoundError:
        print(f"{DATASET_FILE} not found. Run translate_to_tensors.py first."); return

    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    file_ids_val     = data.get("file_ids_val", torch.arange(len(y_val)))

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # 2. Class weight for full mode
    pos_weight = None
    if args.mode == "full":
        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        print(f"Full mode — pos_weight: {pos_weight.item():.2f}  "
              f"(neg={n_neg}, pos={n_pos})")

    criterion = WeightedBCELoss(pos_weight)

    # 3. HP search or defaults
    best_threshold = 0.5
    if args.hps_json:
        with open(args.hps_json) as fh:
            hps = json.load(fh)
        hidden_dim     = hps["hidden_dim"]
        num_layers     = hps["num_layers"]
        dropout        = hps["dropout"]
        lr             = hps["lr"]
        batch_size     = hps["batch_size"]
        best_threshold = hps.get("threshold", 0.5)
        print(f"Loaded HPs from {args.hps_json}: hidden={hidden_dim}, layers={num_layers}, "
              f"dropout={dropout:.2f}, lr={lr:.5f}, batch={batch_size}, thresh={best_threshold}")
    elif args.search:
        print(f"\nRunning Optuna HP search ({args.n_configs} configs, objective: val file-F1)...")
        hidden_dim, num_layers, dropout, lr, batch_size, best_threshold = hp_search(
            X_train, y_train, X_val, y_val, file_ids_val, args.n_configs, pos_weight
        )
        best_hps = {"hidden_dim": hidden_dim, "num_layers": num_layers,
                    "dropout": dropout, "lr": lr, "batch_size": batch_size,
                    "threshold": best_threshold}
        hp_path = os.path.join(RESULTS_DIR, "gru_best_hps.json")
        with open(hp_path, "w") as fh:
            json.dump(best_hps, fh, indent=2)
        print(f"Best HPs -> {hp_path}")
    else:
        hidden_dim, num_layers, dropout, lr, batch_size = \
            HIDDEN_DIM, NUM_LAYERS, 0.0, LEARNING_RATE, BATCH_SIZE

    # 4. Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,  y_val),  batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 5. Train final model
    model     = GRUBotDetector(INPUT_DIM, hidden_dim, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining: hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.2f}, "
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
              f"Acc {ta:.4f} (Val {va:.4f})")

    print(f"\nModel saved -> {MODEL_SAVE_PATH}")

    # 6. Load best checkpoint, plot, evaluate
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    plot_history(history)
    plot_confusion_matrices(model, [train_loader, val_loader, test_loader])
    evaluate_file_level(model, args.split_json, args.mode, threshold=best_threshold, tag=args.tag)


if __name__ == "__main__":
    main()
