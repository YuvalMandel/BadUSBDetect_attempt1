"""
mlp/mlp_train.py
MLP (supervised) classifier for BadUSB detection.
Runs an Optuna hyperparameter search then a final training run.

Prerequisites: run mlp_prepare_data.py then mlp_balance_train.py first.

Usage:
  python mlp/mlp_train.py

Outputs (relative to project root):
  badusb_model.pth           — best model weights
  scaler_params.npy          — StandardScaler mean and scale
  history_<slug>.png         — training history plot
  confusion_<slug>.png       — confusion matrices (train / val / test)
"""

import os
import sys

# Run from project root so output files land there
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CSV_TRAIN = "train_final.csv"
CSV_VAL   = "val_dataset.csv"
CSV_TEST  = "test_dataset.csv"

MODEL_SAVE_PATH  = "badusb_model.pth"
SCALER_SAVE_PATH = "scaler_params.npy"
INPUT_SIZE = 14

# --- Optuna search budget ---
N_TRIALS        = 50
OPTUNA_EPOCHS   = 40
OPTUNA_PATIENCE = 8

# --- Final training budget ---
FINAL_EPOCHS   = 150
FINAL_PATIENCE = 20

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. DATASET
# ==============================================================================
class BadUSBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 2. FLEXIBLE MODEL
# ==============================================================================
def _make_activation(name: str) -> nn.Module:
    return {
        'relu':       nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.1),
        'elu':        nn.ELU(),
        'gelu':       nn.GELU(),
    }[name]


class FlexibleBadUSBClassifier(nn.Module):
    """
    Fully-connected binary classifier with configurable depth, width,
    activation, dropout, and optional BatchNorm.
    """
    def __init__(self, input_dim: int, hidden_dims: list,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(_make_activation(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev = dim
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 3. OPTIMIZER + SCHEDULER FACTORY
# ==============================================================================
def build_optimizer_and_scheduler(params: dict, model: nn.Module,
                                  steps_per_epoch: int, n_epochs: int):
    opt_name   = params['optimizer']
    lr         = params['lr']
    wd         = params['weight_decay']
    sched_name = params['scheduler']

    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    step_per_batch = False

    if sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7)
    elif sched_name == 'cosine_warm':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    elif sched_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    else:  # one_cycle
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            pct_start=0.3,
            anneal_strategy='cos',
        )
        step_per_batch = True

    return optimizer, scheduler, sched_name, step_per_batch


# ==============================================================================
# 4. TRAINING / EVALUATION
# ==============================================================================
def train_epoch(model, loader, optimizer, scheduler=None,
                step_per_batch=False, label_smoothing=0.0, clip_grad=1.0):
    model.train()
    tot_loss = tot_acc = tot_f1 = 0.0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)

        target = y_b * (1.0 - label_smoothing) + 0.5 * label_smoothing \
                 if label_smoothing > 0 else y_b
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()

        if clip_grad > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        if step_per_batch and scheduler is not None:
            scheduler.step()

        tag = torch.round(pred)
        tot_acc  += (tag == y_b).float().mean().item()
        tot_f1   += f1_score(y_b.cpu().numpy(), tag.cpu().detach().numpy(), zero_division=0)
        tot_loss += loss.item()

    n = len(loader)
    return tot_loss / n, tot_acc / n, tot_f1 / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss = tot_acc = tot_f1 = 0.0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        loss = F.binary_cross_entropy(pred, y_b)

        tag = torch.round(pred)
        tot_acc  += (tag == y_b).float().mean().item()
        tot_f1   += f1_score(y_b.cpu().numpy(), tag.cpu().numpy(), zero_division=0)
        tot_loss += loss.item()

    n = len(loader)
    return tot_loss / n, tot_acc / n, tot_f1 / n


# ==============================================================================
# 5. OPTUNA OBJECTIVE
# ==============================================================================
def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    seed_everything(trial.number)

    n_layers    = trial.suggest_int('n_layers', 2, 5)
    hidden_dims = [
        trial.suggest_categorical(f'h{i}', [32, 64, 128, 256, 512])
        for i in range(n_layers)
    ]
    dropout    = trial.suggest_float('dropout',         0.0, 0.5)
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    activation = trial.suggest_categorical('activation',
                     ['relu', 'leaky_relu', 'elu', 'gelu'])
    label_smooth = trial.suggest_float('label_smoothing', 0.0, 0.15)

    params = {
        'optimizer':    trial.suggest_categorical('optimizer',  ['adam', 'adamw']),
        'lr':           trial.suggest_float('lr',          1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'scheduler':    trial.suggest_categorical('scheduler',
                            ['cosine', 'cosine_warm', 'reduce_on_plateau', 'one_cycle']),
    }
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    train_loader = DataLoader(BadUSBDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(BadUSBDataset(X_val, y_val), batch_size=batch_size)

    model = FlexibleBadUSBClassifier(
        INPUT_SIZE, hidden_dims, dropout, batch_norm, activation
    ).to(device)

    optimizer, scheduler, sched_name, step_per_batch = build_optimizer_and_scheduler(
        params, model, len(train_loader), OPTUNA_EPOCHS
    )

    best_val_f1  = 0.0
    patience_cnt = 0

    for epoch in range(OPTUNA_EPOCHS):
        train_epoch(model, train_loader, optimizer, scheduler,
                    step_per_batch=step_per_batch, label_smoothing=label_smooth)
        val_loss, _, val_f1 = evaluate(model, val_loader)

        if not step_per_batch:
            if sched_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= OPTUNA_PATIENCE:
                break

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1


# ==============================================================================
# 6. FINAL TRAINING
# ==============================================================================
def train_final_model(best_params: dict, X_train, y_train, X_val, y_val):
    seed_everything()

    n_layers     = best_params['n_layers']
    hidden_dims  = [best_params[f'h{i}'] for i in range(n_layers)]
    dropout      = best_params['dropout']
    batch_norm   = best_params['batch_norm']
    activation   = best_params['activation']
    label_smooth = best_params['label_smoothing']
    batch_size   = best_params['batch_size']

    train_loader = DataLoader(BadUSBDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(BadUSBDataset(X_val, y_val), batch_size=batch_size)

    model = FlexibleBadUSBClassifier(
        INPUT_SIZE, hidden_dims, dropout, batch_norm, activation
    ).to(device)

    optimizer, scheduler, sched_name, step_per_batch = build_optimizer_and_scheduler(
        best_params, model, len(train_loader), FINAL_EPOCHS
    )

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'lr']}
    best_val_loss = float('inf')
    patience_cnt  = 0
    best_state    = None

    print(f"\nFinal training — up to {FINAL_EPOCHS} epochs (patience={FINAL_PATIENCE})")
    print(f"  Architecture : {hidden_dims}  act={activation}  bn={batch_norm}  drop={dropout:.2f}")
    print(f"  Optimizer    : {best_params['optimizer']}  lr={best_params['lr']:.2e}  "
          f"wd={best_params['weight_decay']:.2e}")
    print(f"  Scheduler    : {sched_name}  |  batch={batch_size}  smooth={label_smooth:.3f}\n")

    for epoch in range(FINAL_EPOCHS):
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, scheduler,
            step_per_batch=step_per_batch, label_smoothing=label_smooth,
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        if not step_per_batch:
            if sched_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_SAVE_PATH)
        else:
            patience_cnt += 1
            if patience_cnt >= FINAL_PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}  (best val loss: {best_val_loss:.4f})")
                break

        print(f"  Ep {epoch+1:3d}/{FINAL_EPOCHS} | "
              f"L {tr_loss:.4f} (v{val_loss:.4f}) | "
              f"Acc {tr_acc:.4f} (v{val_acc:.4f}) | "
              f"F1 {tr_f1:.4f} (v{val_f1:.4f}) | "
              f"LR {current_lr:.2e}")

    model.load_state_dict(best_state)
    print(f"\nBest model saved → {MODEL_SAVE_PATH}")
    return model, history


# ==============================================================================
# 7. VISUALIZATION
# ==============================================================================
def _params_to_slug(best_params: dict) -> str:
    n = best_params.get('n_layers', '?')
    dims = [best_params.get(f'h{i}', '?') for i in range(n)]
    dims_str = '-'.join(str(d) for d in dims)
    act  = best_params.get('activation', '?')
    opt  = best_params.get('optimizer', '?')
    lr   = best_params.get('lr', 0)
    sch  = best_params.get('scheduler', '?')
    drop = best_params.get('dropout', 0)
    bn   = best_params.get('batch_norm', False)
    bs   = best_params.get('batch_size', '?')
    slug = (f"arch{dims_str}_act{act}_opt{opt}_lr{lr:.0e}"
            f"_sch{sch}_drop{drop:.2f}_bn{int(bn)}_bs{bs}")
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        slug = slug.replace(ch, '_')
    return slug


def _params_to_title(best_params: dict) -> str:
    n = best_params.get('n_layers', '?')
    dims = [best_params.get(f'h{i}', '?') for i in range(n)]
    return (
        f"arch={dims}  act={best_params.get('activation','?')}  "
        f"opt={best_params.get('optimizer','?')}  lr={best_params.get('lr',0):.1e}\n"
        f"sched={best_params.get('scheduler','?')}  "
        f"drop={best_params.get('dropout',0):.2f}  "
        f"bn={best_params.get('batch_norm',False)}  "
        f"bs={best_params.get('batch_size','?')}"
    )


def plot_history(history: dict, best_params: dict):
    slug  = _params_to_slug(best_params)
    hpstr = _params_to_title(best_params)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Training History\n{hpstr}", fontsize=9)

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    ax_f1 = axes[2]
    ax_f1.plot(history['val_f1'], label='Val F1', color='green')
    ax_f1.set_title('Val F1  &  Learning Rate')
    ax_f1.set_xlabel('Epoch'); ax_f1.legend(loc='upper left'); ax_f1.grid(True)

    ax_lr = ax_f1.twinx()
    ax_lr.plot(history['lr'], color='orange', alpha=0.6, linestyle='--', label='LR')
    ax_lr.set_ylabel('Learning Rate', color='orange')
    ax_lr.legend(loc='upper right')

    plt.tight_layout()
    fname = f"history_{slug}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  History plot saved → {fname}")


def plot_confusion_matrices(model, loaders, best_params: dict):
    slug  = _params_to_slug(best_params)
    hpstr = _params_to_title(best_params)

    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Confusion Matrices\n{hpstr}", fontsize=9)
    titles = ["Train Set", "Validation Set", "Test Set"]

    with torch.no_grad():
        for i, (loader, title) in enumerate(zip(loaders, titles)):
            all_preds, all_labels = [], []
            for X_b, y_b in loader:
                preds = model(X_b.to(device))
                all_preds.extend(torch.round(preds).cpu().numpy())
                all_labels.extend(y_b.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=axes[i], cbar=False)
            axes[i].set_title(title)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Human', 'Bot'])
            axes[i].set_yticklabels(['Human', 'Bot'])

            if title == "Test Set":
                print(f"\n--- Classification Report ({title}) ---")
                print(classification_report(all_labels, all_preds,
                                            target_names=['Human', 'Bot']))

    plt.tight_layout()
    fname = f"confusion_{slug}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix plot saved → {fname}")


# ==============================================================================
# 8. DATA LOADING
# ==============================================================================
def prepare_data():
    print("Loading pre-split data...")
    for path in (CSV_TRAIN, CSV_VAL, CSV_TEST):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found.\n"
                "Run mlp/mlp_prepare_data.py then mlp/mlp_balance_train.py first."
            )

    train_df = pd.read_csv(CSV_TRAIN)
    val_df   = pd.read_csv(CSV_VAL)
    test_df  = pd.read_csv(CSV_TEST)

    X_train = train_df.drop("Label", axis=1).values; y_train = train_df["Label"].values
    X_val   = val_df.drop("Label", axis=1).values;   y_val   = val_df["Label"].values
    X_test  = test_df.drop("Label", axis=1).values;  y_test  = test_df["Label"].values

    print(f"\nDataset sizes (person-disjoint splits):")
    print(f"  Train : {len(X_train):6d}  "
          f"(humans: {(y_train==0).sum()}, bots: {(y_train==1).sum()})")
    print(f"  Val   : {len(X_val):6d}  "
          f"(humans: {(y_val==0).sum()}, bots: {(y_val==1).sum()})")
    print(f"  Test  : {len(X_test):6d}  "
          f"(humans: {(y_test==0).sum()}, bots: {(y_test==1).sum()})")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    np.save(SCALER_SAVE_PATH, [scaler.mean_, scaler.scale_])
    print("Scaler parameters saved.")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==============================================================================
# 9. MAIN
# ==============================================================================
def main():
    seed_everything()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    print(f"\nStarting Optuna search: {N_TRIALS} trials × up to {OPTUNA_EPOCHS} epochs ...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n" + "="*60)
    print("  Optuna search complete")
    print(f"  Best Val F1 : {study.best_value:.4f}")
    print(f"  Best params : {study.best_params}")
    print("="*60)

    model, history = train_final_model(
        study.best_params, X_train, y_train, X_val, y_val
    )

    plot_history(history, study.best_params)

    bs = study.best_params.get('batch_size', 32)
    train_loader = DataLoader(BadUSBDataset(X_train, y_train), batch_size=bs)
    val_loader   = DataLoader(BadUSBDataset(X_val,   y_val),   batch_size=bs)
    test_loader  = DataLoader(BadUSBDataset(X_test,  y_test),  batch_size=bs)

    plot_confusion_matrices(model, [train_loader, val_loader, test_loader], study.best_params)


if __name__ == "__main__":
    main()
