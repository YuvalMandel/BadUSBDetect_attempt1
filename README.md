# BadUSB Detection

Keystroke-dynamics based BadUSB (malicious USB keyboard) detector.
Two models are implemented:

| Model | Folder | Approach |
|-------|--------|----------|
| HTM (anomaly)    | `htm/` | Unsupervised; learns normal typing, flags deviations |
| MLP (supervised) | `mlp/` | Binary classifier trained on labelled windows |

---

## Prerequisites

```bash
pip install htm.core scikit-learn scipy numpy matplotlib seaborn tqdm optuna torch
```

On Technion Newton, activate your virtual environment before running anything:

```bash
source /path/to/.venv/bin/activate
```

---

## Directory structure

```
.
├── common/
│   └── keystroke_features.py  Shared parsing & feature extraction (used by both models)
│
├── htm/
│   ├── htm_common.py          HTM-specific: SDR encoders + data splitting
│   ├── htm_train_interactive.py  Interactive HTM training (single fixed config)
│   ├── htm_train_single.py    SLURM worker — train one HTM config
│   ├── htm_prepare_data.py    One-time data prep  →  split.pkl, features_cache.pkl
│   ├── htm_generate_configs.py  Generate configs/ + slurm/submit_array.sh
│   ├── htm_test_model.py      Test a trained HTM model (3 evaluation modes)
│   └── htm_collect_results.py Aggregate results/  →  leaderboard
│
├── mlp/
│   ├── mlp_prepare_data.py    Parse raw files → train/val/test CSV + reference pool
│   ├── mlp_balance_train.py   Balance the training CSV (trim human majority)
│   ├── mlp_train.py           Optuna hyperparameter search + final MLP training
│   ├── mlp_generate_configs.py  Generate mlp_configs/ + slurm/mlp_submit_array.sh
│   ├── mlp_train_single.py    SLURM worker — train one MLP config
│   └── mlp_collect_results.py Aggregate mlp_results/  →  leaderboard
│
├── configs/               JSON HTM configs for current SLURM batch  [gitignored per-run]
│   └── config_NNNN.json   Globally-unique indices (continue across runs)
│
├── mlp_configs/           JSON MLP configs for current SLURM batch  [gitignored per-run]
│   └── config_NNNN.json
│
├── models/                Trained HTM model pkl files  [gitignored]
├── plots/                 HTM PNG plots                [gitignored]
├── results/               HTM per-run JSON + leaderboard  [gitignored]
├── mlp_models/            Trained MLP model pth files  [gitignored]
├── mlp_plots/             MLP PNG plots                [gitignored]
├── mlp_results/           MLP per-run JSON + leaderboard  [gitignored]
├── logs/                  SLURM stdout/stderr          [gitignored]
└── slurm/
    ├── submit_array.sh        Generated HTM SLURM submission script
    └── mlp_submit_array.sh    Generated MLP SLURM submission script
```

> All scripts are run from the **project root** directory.
> Output files (`split.pkl`, `features_cache.pkl`, CSV files, model weights, plots)
> are written to the project root or its subdirectories.

---

## Running the MLP model (interactive)

### Step 1 — Prepare data

```bash
python mlp/mlp_prepare_data.py
```

Loads the human train/val/test split from `split.pkl` (same split as HTM, so F1
scores are directly comparable). Extracts 14-dim feature windows, generates
synthetic bot samples, and writes `train_dataset.csv`, `val_dataset.csv`,
`test_dataset.csv`, `reference_pool.npz`.

> Run `htm/htm_prepare_data.py` first to create `split.pkl`.

### Step 2 — Balance training set

```bash
python mlp/mlp_balance_train.py
```

Trims the human majority class to match the bot count.
Writes `train_final.csv`. Val and test CSVs are left unbalanced.

### Step 3 — Train (Optuna search)

```bash
python mlp/mlp_train.py
```

Runs an Optuna hyperparameter search (50 trials) then a final training run.
Outputs: `badusb_model.pth`, `scaler_params.npy`, `history_*.png`, `confusion_*.png`.

---

## MLP hyperparameter search on Newton SLURM

### Step 1 — Prepare data (once)

```bash
python htm/htm_prepare_data.py       # creates split.pkl (if not already done)
python mlp/mlp_prepare_data.py       # uses split.pkl for humans
python mlp/mlp_balance_train.py      # creates train_final.csv
```

### Step 2 — Generate configs and SLURM script

```bash
python mlp/mlp_generate_configs.py --n-configs 128 --seed 0
```

Creates `mlp_configs/config_NNNN.json` files and `slurm/mlp_submit_array.sh`.

### Step 3 — Submit

```bash
sbatch slurm/mlp_submit_array.sh
```

Each job: 2 CPUs, 4 GB RAM, 20 min wall time. No GPU needed (14-dim input, ~50 K rows).

### Step 4 — Collect results

```bash
python mlp/mlp_collect_results.py --top 20
```

Reads `mlp_results/mlp*.json`, prints ranked leaderboard, writes
`mlp_results/mlp_leaderboard.csv` and `mlp_results/mlp_leaderboard.txt`.

### Clean before re-run

```bash
rm mlp_configs/config_*.json
rm mlp_results/mlp*.json mlp_results/mlp_leaderboard.csv mlp_results/mlp_leaderboard.txt
rm mlp_models/mlp*.pth
rm mlp_plots/mlp*_confusion.png
```

---

## Running the HTM model (interactive)

```bash
python htm/htm_train_interactive.py
```

- `RETRAIN = False` (default): loads the latest `htm_model_*.pkl` and skips training.
- `RETRAIN = True`: trains from scratch, saves a new timestamped pkl.

Outputs: `htm_model_YYYYMMDD_HHMMSS.pkl`, `htm_results.png`, `htm_confusion_matrices.png`.

---

## HTM hyperparameter search on Newton SLURM

### Step 1 — Prepare data (run once, on login node or locally)

```bash
python htm/htm_prepare_data.py
```

Creates:
- `split.pkl` — deterministic train/val/test file split (shared by HTM and MLP)
- `features_cache.pkl` — pre-computed feature sequences for every file

This step uses all CPU cores and takes a few minutes.
If either file already exists it is reused; only missing entries are extracted.

### Step 2 — Generate configs and SLURM script

```bash
python htm/htm_generate_configs.py --n-configs 128 --seed 0
```

**Cumulative workflow** — results are never deleted:
- Scans `results/cfg*.json` to find the highest completed `config_idx`
- Deletes old `configs/config_*.json` (already-run; results are preserved)
- Writes new configs numbered `last_idx + 1` … `last_idx + N`
- Updates `slurm/submit_array.sh`

First run starts at index 0. Second run (after jobs finish) continues from
wherever the first run ended.

### Step 3 — Submit to Newton

```bash
# Edit slurm/submit_array.sh to set --partition, --account, and activate your env.
sbatch slurm/submit_array.sh
```

Each array task:
- Uses exactly **1 CPU** and ≤ 8 GB RAM
- Runs ≈ 20–90 minutes
- All tasks read `split.pkl` and `features_cache.pkl` (read-only, no conflicts)
- Each task writes to its own uniquely named files under `models/`, `plots/`, `results/`

Monitor progress:

```bash
squeue -u $USER
ls results/ | wc -l          # how many jobs finished
```

### Step 4 — Collect results

```bash
python htm/htm_collect_results.py --top 20
```

Reads **all** `results/cfg*.json` files (cumulative across all runs) and prints
a ranked leaderboard. Writes `results/leaderboard.csv` and `results/leaderboard.txt`.

---

## Testing a trained HTM model

```bash
python htm/htm_test_model.py --model models/<slug>.pkl --mode all_non_train
```

### Evaluation modes (`--mode`)

| Mode | Description |
|------|-------------|
| `orig` | Evaluate on the original val/test splits from `split.pkl` |
| `all_non_train` | Evaluate all files in `features_cache.pkl` that are not in the training set (default) |
| `all_other_files` | Walk `--data_root` for new human `.txt` files + `--synthetic_bots_root` for synthetic bots; extract features on-the-fly |

### Additional flags

| Flag | Default | Effect |
|------|---------|--------|
| `--data_root` | `../UB_keystroke_dataset/` | Root to search for other human files (`all_other_files` mode) |
| `--synthetic_bots_root` | `../bots_synt_dataset/Synthetic_Bots` | Root for synthetic bot files (`all_other_files` mode) |
| `--skip_orig_sets` | off | Skip val/test evaluation in `orig` mode |
| `--write_decision_log` | off | Write a per-window decision log for one human and one bot file |

#### Decision logs (`--write_decision_log`)

When enabled, two files are written to `logs/`:

```
logs/<model_name>_human_log.txt
logs/<model_name>_bot_log.txt
```

Each file has one row per sliding window:

```
  Window     Score        Status  Reason
--------  --------  ------------  ---------------------------------------------------
       0    0.8321        WARMUP  warmup period (1/5)
       5    0.3210   no crossing  score 0.3210 < threshold 0.4545
       6    0.9812           BOT  first crossing: 0.9812 >= threshold 0.4545
       7    0.7631           BOT  already triggered at window 6
```

The model always uses `first_crossing` mode:
`WARMUP` → `no crossing` → `BOT` (once any post-warmup window crosses the threshold, all subsequent windows are marked `BOT`).

### Running on Newton SLURM

Use `run_test_model.sh` from the project root:

```bash
sbatch run_test_model.sh
```

Before submitting, edit the `MODEL=` line in `run_test_model.sh` to point at the `.pkl` you want to evaluate.

Key settings in that script:
```bash
#SBATCH --cpus-per-task=256
#SBATCH --mem=32G
#SBATCH --time=12:00:00
```
The script automatically uses `SLURM_CPUS_PER_TASK` workers for parallel feature extraction.

---

## Output files and naming conventions

All outputs from `htm/htm_train_single.py` use a slug encoding the key hyperparameters
and the scores achieved:

```
cfg{idx:04d}_sp{numActiveColumns}_enc{bits_per_feature}w{w}_tm{cellsPerColumn}_act{activationThreshold}_vf1{val_f1:.4f}_tf1{test_f1:.4f}
```

Example:
```
cfg0083_sp40_enc16w5_tm16_act13_fc_vf10.6667_tf10.6207
```

| File | Location | Contents |
|------|----------|----------|
| `<slug>.pkl` | `models/` | SP, TM, encoder, threshold, config, scores |
| `<slug>_results.png` | `plots/` | Anomaly-score time series / score distribution / F1 vs threshold |
| `<slug>_confusion.png` | `plots/` | Confusion matrices (val + test) |
| `<slug>.json` | `results/` | Scalar metrics + full config |

The **plot title** contains the complete hyperparameter configuration and both F1 scores,
so every image is self-documenting.

---

## Interpreting results

### `*_results.png` (3 panels)

| Panel | What it shows |
|-------|---------------|
| **Anomaly score over time** | Raw per-window anomaly score for 3 sample human files (blue) and 3 bot files (red). A well-trained model separates the two: bots should sit near 1.0, humans near 0.0. |
| **Per-file mean score distribution** | Histogram of mean anomaly scores (one bar per file). The green dashed line is the chosen threshold. Human and bot distributions should be clearly separated. |
| **F1 vs threshold** | Full threshold-sweep curve. The best threshold (red dashed) maximises val-set F1. A sharp, high peak indicates a robust decision boundary. |

### `*_confusion.png` (2 panels)

Standard confusion matrices for validation and test sets.
Rows = actual class, columns = predicted class.
Labels: Human (0) / Bot (1).

### Leaderboard (`results/leaderboard.csv`)

Sorted by **test F1** descending (val F1 as tiebreaker).
The top row identifies the best-performing config; load it with:

```python
import pickle
with open("models/<slug>.pkl", "rb") as f:
    m = pickle.load(f)
sp, tm, encoder, thresh = m["sp"], m["tm"], m["encoder"], m["best_thresh"]
```

---

## Shared data processing (`common/`)

`common/keystroke_features.py` is used by **both** the HTM and MLP pipelines.
It provides:

- `FOLDERS` — paths to human and bot dataset directories
- `parse_file(filepath)` — keystroke log parser → (dwells, flights) arrays
- `extract_features(window, ref_pool)` — 7-dim feature vector per window
- `create_reference_pool(train_human_files)` — KS/Wasserstein reference windows
- `process_file_worker(args)` — multiprocessing worker for HTM feature caching
- `is_task1(filepath)`, `get_person_id(filepath)` — file naming helpers

To change dataset paths, edit `FOLDERS` in `common/keystroke_features.py`.

---

## Hyperparameter search space (HTM)

The search varies the following parameters. Fixed parameters (`boostStrength`,
`use_anomaly_likelihood`) were eliminated by empirical evidence across search rounds.

| Group | Parameter | Values searched | Notes |
|-------|-----------|-----------------|-------|
| SP | `numActiveColumns` | **20, 25, 30, 35, 40, 45** | Round 6: removed 10,15 (never top-6); added 25,35,45 around hot zone 30–40 |
| SP | `potentialPct` | 0.5, 0.60, **0.65**, 0.70, 0.75, **0.80** | Round 6: added 0.65, 0.80 to bracket the 0.7–0.75 sweet spot |
| SP | `boostStrength` | ~~0, 1, 2, 3~~ → **fixed 0** | boost>0 killed all configs round 1 |
| SP | `synPermActiveInc` | 0.02, 0.05, 0.10 | |
| SP | `synPermConnected` | 0.10, **0.15**, 0.20 | Round 6: added 0.15 |
| SP | `synPermInactiveDec` | 0.003, 0.005, 0.010 | |
| TM | `cellsPerColumn` | 8, 16 | Round 6: removed 32 (never top-6); both 8 and 16 appear in top-6 |
| TM | `activationThreshold` | 13, 14, 15, 18, 20 | Full range valid — all values appear in top-6 |
| TM | `minThreshold` | 10, 11, 12 | 10 dominant (4/6 top) |
| TM | `maxNewSynapseCount` | 15, 20, **25**, 30 | Round 6: added 25 |
| TM | `initialPermanence` | 0.21, 0.31, 0.40 | |
| TM | `connectedPermanence` | 0.30, 0.50 | |
| TM | `permanenceIncrement` | 0.05, 0.10, 0.20 | |
| TM | `permanenceDecrement` | 0.03, 0.05, 0.10 | |
| Enc | `bits_per_feature` | ~~32~~ → **fixed 16** | Round 6: removed 32 (never top-6) |
| Enc | `w` (active bits) | 5, 6, 7 | Round 6: removed 4 (never top-6) |
| Det | `use_anomaly_likelihood` | ~~True, False~~ → **fixed False** | AL never helped in any run |
| Det | `warmup_steps` | 3, 5, 8, 12, 20 | Initial windows skipped before alarm can fire |

**Bold** = added or changed in round 6. ~~Strikethrough~~ = eliminated by evidence.

### Best config so far

`cfg0083` (254-run search, `first_crossing` detection):
`sp_act=40, pct=0.70, enc=16w5, tm=16, act=13, min=10` → **val F1=0.6667, test F1=0.6207**

```python
# Load and use:
import pickle
with open("models/cfg0083_sp40_enc16w5_tm16_act13_fc_vf10.6667_tf10.6207.pkl", "rb") as f:
    m = pickle.load(f)
```
