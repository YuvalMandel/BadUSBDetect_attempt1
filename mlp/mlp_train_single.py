"""
mlp/mlp_train_single.py
Train one MLP configuration. Designed to be called by a SLURM array task.

Usage:
  python mlp/mlp_train_single.py --config mlp_configs/config_0023.json

Outputs (all relative to project root):
  mlp_models/ <slug>_vf1{val:.4f}_tf1{test:.4f}.pth  — state_dict + config + scaler
  mlp_results/<slug>.json                              — val_f1, test_f1, n_params, elapsed_s, config
  mlp_plots/  <slug>_confusion.png                     — val + test confusion matrices
"""

import os
import sys
import time

# Allow imports from project root (common/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Run from project root so all relative paths work
os.chdir(_project_root)

import argparse
import copy
import json
import random

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for SLURM nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CSV_TRAIN     = "train_final.csv"
CSV_VAL       = "val_dataset.csv"
CSV_TEST      = "test_dataset.csv"
INPUT_SIZE    = 14
SEARCH_EPOCHS = 100
PATIENCE      = 15

MODELS_DIR  = "mlp_models"
RESULTS_DIR = "mlp_results"
PLOTS_DIR   = "mlp_plots"

device = torch.device("cpu")   # CPU sufficient for this dataset size


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class BadUSBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def _make_activation(name: str) -> nn.Module:
    return {
        'relu':       nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.1),
        'elu':        nn.ELU(),
        'gelu':       nn.GELU(),
    }[name]


class FlexibleMLP(nn.Module):
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


# ------------------------------------------------------------------
# Optimizer + scheduler
# ------------------------------------------------------------------
def build_optimizer_and_scheduler(cfg, model, steps_per_epoch, n_epochs):
    opt_name   = cfg['optimizer']
    lr         = cfg['lr']
    wd         = cfg['weight_decay']
    sched_name = cfg['scheduler']

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


# ------------------------------------------------------------------
# Train / eval
# ------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler,
                step_per_batch, label_smoothing, clip_grad=1.0):
    model.train()
    tot_loss = tot_f1 = 0.0

    for X_b, y_b in loader:
        optimizer.zero_grad()
        pred = model(X_b)

        target = (y_b * (1.0 - label_smoothing) + 0.5 * label_smoothing
                  if label_smoothing > 0 else y_b)
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()

        if clip_grad > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        if step_per_batch and scheduler is not None:
            scheduler.step()

        tag = torch.round(pred)
        tot_f1   += f1_score(y_b.numpy(), tag.detach().numpy(), zero_division=0)
        tot_loss += loss.item()

    n = len(loader)
    return tot_loss / n, tot_f1 / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss = tot_f1 = 0.0

    for X_b, y_b in loader:
        pred = model(X_b)
        loss = F.binary_cross_entropy(pred, y_b)
        tag  = torch.round(pred)
        tot_f1   += f1_score(y_b.numpy(), tag.numpy(), zero_division=0)
        tot_loss += loss.item()

    n = len(loader)
    return tot_loss / n, tot_f1 / n


# ------------------------------------------------------------------
# Slug
# ------------------------------------------------------------------
def make_slug(idx, cfg):
    dims    = '-'.join(str(d) for d in cfg['hidden_dims'])
    act     = cfg['activation']
    opt     = cfg['optimizer']
    lr      = cfg['lr']
    sch     = cfg['scheduler']
    drop    = cfg['dropout']
    bn      = int(cfg['batch_norm'])
    bs      = cfg['batch_size']
    slug = (f"mlp{idx:04d}_arch{dims}_act{act}_opt{opt}_lr{lr:.0e}"
            f"_sch{sch}_drop{drop:.2f}_bn{bn}_bs{bs}")
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        slug = slug.replace(ch, '_')
    return slug


# ------------------------------------------------------------------
# Confusion matrix plot
# ------------------------------------------------------------------
def plot_confusion(model, val_loader, test_loader, slug):
    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    loaders = [(val_loader, "Validation Set"), (test_loader, "Test Set")]

    with torch.no_grad():
        for ax, (loader, title) in zip(axes, loaders):
            all_preds, all_labels = [], []
            for X_b, y_b in loader:
                preds = model(X_b)
                all_preds.extend(torch.round(preds).numpy())
                all_labels.extend(y_b.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        ax=ax, cbar=False)
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Human', 'Bot'])
            ax.set_yticklabels(['Human', 'Bot'])

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fname = os.path.join(PLOTS_DIR, f"{slug}_confusion.png")
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    return fname


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train one MLP config (SLURM worker)")
    parser.add_argument("--config", required=True,
                        help="Path to mlp_configs/config_NNNN.json")
    args = parser.parse_args()

    # Derive config index from filename
    config_name = os.path.basename(args.config)          # config_0023.json
    config_idx  = int(config_name.split('_')[1].split('.')[0])

    with open(args.config) as fh:
        cfg = json.load(fh)

    # Reproducibility
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Config {config_idx:04d}: {cfg}")

    # ---- Load data ----
    for path in (CSV_TRAIN, CSV_VAL, CSV_TEST):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. "
                "Run mlp/mlp_prepare_data.py then mlp/mlp_balance_train.py first.")

    train_df = pd.read_csv(CSV_TRAIN)
    val_df   = pd.read_csv(CSV_VAL)
    test_df  = pd.read_csv(CSV_TEST)

    X_train = train_df.drop("Label", axis=1).values;  y_train = train_df["Label"].values
    X_val   = val_df.drop("Label", axis=1).values;    y_val   = val_df["Label"].values
    X_test  = test_df.drop("Label", axis=1).values;   y_test  = test_df["Label"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    bs = cfg['batch_size']
    train_loader = DataLoader(BadUSBDataset(X_train, y_train),
                              batch_size=bs, shuffle=True, drop_last=True)
    val_loader   = DataLoader(BadUSBDataset(X_val,   y_val),   batch_size=bs)
    test_loader  = DataLoader(BadUSBDataset(X_test,  y_test),  batch_size=bs)

    # ---- Build model ----
    model = FlexibleMLP(
        INPUT_SIZE,
        cfg['hidden_dims'],
        cfg['dropout'],
        cfg['batch_norm'],
        cfg['activation'],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer, scheduler, sched_name, step_per_batch = build_optimizer_and_scheduler(
        cfg, model, len(train_loader), SEARCH_EPOCHS
    )

    # ---- Training loop with early stopping on val F1 ----
    best_val_f1  = 0.0
    patience_cnt = 0
    best_state   = None
    t0 = time.time()

    for epoch in range(SEARCH_EPOCHS):
        train_epoch(model, train_loader, optimizer, scheduler,
                    step_per_batch, cfg['label_smoothing'])

        val_loss, val_f1 = evaluate(model, val_loader)

        if not step_per_batch:
            if sched_name == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}  (best val F1: {best_val_f1:.4f})")
                break

    elapsed = time.time() - t0

    # ---- Evaluate best checkpoint ----
    model.load_state_dict(best_state)
    _, val_f1  = evaluate(model, val_loader)
    _, test_f1 = evaluate(model, test_loader)

    print(f"  Val F1: {val_f1:.4f}  |  Test F1: {test_f1:.4f}  |  "
          f"Params: {n_params:,}  |  Time: {elapsed:.1f}s")

    slug = make_slug(config_idx, cfg)

    # ---- Save model ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(
        MODELS_DIR, f"{slug}_vf1{val_f1:.4f}_tf1{test_f1:.4f}.pth")
    torch.save({
        'state_dict':  model.state_dict(),
        'config':      cfg,
        'config_idx':  config_idx,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'val_f1':      val_f1,
        'test_f1':     test_f1,
    }, model_path)

    # ---- Save results JSON ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'config_idx': config_idx,
        'val_f1':     val_f1,
        'test_f1':    test_f1,
        'n_params':   n_params,
        'elapsed_s':  round(elapsed, 2),
        'config':     cfg,
        'model_file': model_path,
    }
    result_path = os.path.join(RESULTS_DIR, f"{slug}.json")
    with open(result_path, 'w') as fh:
        json.dump(result, fh, indent=2)

    # ---- Confusion matrix plot ----
    plot_confusion(model, val_loader, test_loader, slug)

    print(f"  Model   → {model_path}")
    print(f"  Results → {result_path}")


if __name__ == "__main__":
    main()
