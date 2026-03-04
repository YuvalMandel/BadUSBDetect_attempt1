# BadUSB Detection

Keystroke-dynamics based BadUSB (malicious USB keyboard) detector.
Two models are implemented:

| Model | Folder | Approach |
|-------|--------|----------|
| HTM (anomaly)               | `htm/`                | Unsupervised; learns normal typing, flags deviations |
| HTM-Distance (anomaly)      | `htm_distance/`       | Same as HTM but raw per-keystroke events: key type + dwell + flight + QWERTY distance |
| HTM-Distance-Stats (anomaly)| `htm_distance_stats/` | Hybrid: sliding windows + 21-dim stats (dwell, flight, QWERTY dist) fed into HTM |
| HTM-Combined (anomaly)      | `htm_combined/`       | Full hybrid: 21-dim window stats **+** last-keystroke identity (key one-hot + dwell + flight + dist) in a single SDR; stats and scalar encoders have independent resolution settings |
| MLP (supervised)            | `mlp/`                | Binary classifier trained on labelled windows |

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
├── htm_distance/
│   ├── htm_distance_common.py         QWERTY layout, one-hot key encoder, scalar encoders, parser
│   ├── htm_distance_prepare_data.py   Parse files → dist_cache.pkl  (key_idx, dwell, flight, dist)
│   ├── htm_distance_train_single.py      SLURM worker — train one HTM-distance config
│   ├── htm_distance_generate_configs.py  Generate dist_configs/ + slurm/dist_submit_array.sh
│   ├── htm_distance_collect_results.py   Aggregate dist_results/  →  leaderboard
│   └── htm_distance_test_model.py        Test a saved model (3 evaluation modes)
│
├── htm_distance_stats/
│   ├── htm_distance_stats_common.py         StatsEncoder, reference pool, feature extraction, detection
│   ├── htm_distance_stats_prepare_data.py   Parse files → stats_cache.pkl  (reuses dist_cache.pkl if present)
│   ├── htm_distance_stats_train_single.py   SLURM worker — train one HTM-distance-stats config
│   ├── htm_distance_stats_generate_configs.py  Generate ds_configs/ + slurm/ds_submit_array.sh
│   ├── htm_distance_stats_collect_results.py   Aggregate ds_results/  →  leaderboard
│   └── htm_distance_stats_test_model.py        Test a saved model (3 evaluation modes)
│
├── htm_combined/
│   ├── htm_combined_common.py         CombinedEncoder (split stats/scalar settings), reference pool, feature extraction
│   ├── htm_combined_train_single.py   SLURM worker — train one HTM-combined config
│   ├── htm_combined_generate_configs.py  Generate hc_configs/ + slurm/hc_submit_array.sh
│   └── htm_combined_collect_results.py   Aggregate hc_results/  →  leaderboard
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
├── models/                Trained HTM model pkl files        [gitignored]
├── plots/                 HTM PNG plots                      [gitignored]
├── results/               HTM per-run JSON + leaderboard     [gitignored]
├── dist_configs/          JSON dist configs for current batch[gitignored per-run]
│   └── config_NNNN.json
├── dist_models/           Trained HTM-distance pkl files     [gitignored]
├── dist_plots/            HTM-distance PNG plots             [gitignored]
├── dist_results/          HTM-distance per-run JSON + leaderboard  [gitignored]
├── ds_configs/            JSON ds configs for current batch  [gitignored per-run]
│   └── config_NNNN.json
├── ds_models/             Trained HTM-distance-stats pkl files [gitignored]
├── ds_plots/              HTM-distance-stats PNG plots         [gitignored]
├── ds_results/            HTM-distance-stats per-run JSON + leaderboard [gitignored]
├── hc_configs/            JSON HTM-combined configs for current batch [gitignored per-run]
│   └── config_NNNN.json
├── hc_models/             Trained HTM-combined pkl files      [gitignored]
├── hc_plots/              HTM-combined PNG plots              [gitignored]
├── hc_results/            HTM-combined per-run JSON + leaderboard [gitignored]
├── mlp_models/            Trained MLP model pth files        [gitignored]
├── mlp_plots/             MLP PNG plots                      [gitignored]
├── mlp_results/           MLP per-run JSON + leaderboard     [gitignored]
├── logs/                  SLURM stdout/stderr                [gitignored]
└── slurm/
    ├── submit_array.sh        Generated HTM SLURM submission script
    ├── dist_submit_array.sh   Generated HTM-distance SLURM submission script
    ├── ds_submit_array.sh     Generated HTM-distance-stats SLURM submission script
    ├── hc_submit_array.sh     Generated HTM-combined SLURM submission script
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

## How the detection threshold is chosen

The HTM and HTM-Distance models produce a score per timestep (0–1; raw anomaly
or AnomalyLikelihood). A file is flagged as a bot when the first post-warmup
score exceeds a threshold (`first_crossing` mode). The threshold is **not** a
pre-training hyperparameter because its optimal value is unknowable before
training — it depends on where the model's output distribution lands.

Instead, after training, the threshold is found by a **two-stage search**:

1. Running the trained model (no learning) on the **validation set**.
2. **Coarse pass**: sweep 200 evenly-spaced candidate thresholds from 0 to 1.
3. **Fine pass**: sweep 1000 points in a ±2-step window around the coarse best.
   This gives an effective precision of ~0.00001.
4. Picking the threshold that maximises the selection criterion across both passes:
   - `htm/` and `htm_distance/`: **validation F1** (binary, positive = bot)
   - `htm_distance_stats/` (Round 2+): **validation balanced accuracy** = (TPR + TNR) / 2.
     Balanced accuracy is used because the val set has many more human files than bot files;
     optimising F1 in that setting degenerates to thresh = 0 (predict everything as bot).
5. Applying that threshold to the **test set** to get the reported test F1.

This is standard post-hoc threshold calibration. The threshold is saved in the
model `.pkl` alongside the SP and TM objects so it can be used at inference time
without re-calibration.

### Short-file misclassification rule

Files with fewer printable keystrokes than `warmup_steps` never enter the
post-warmup detection zone. Such files are **always counted as misclassified**,
regardless of the true label:

- A bot file that is too short → the model predicts *human* (false negative)
- A human file that is too short → the model predicts *bot* (false positive)

This is the conservative choice: a file that provides no evidence should not
inflate accuracy. It also penalises configs whose `warmup_steps` is too large
for the typical file length in the dataset.

> **Note for BadUSB files**: BadUSB attacks use modifier / function keys
> (Ctrl, Win, Arrow, Enter) which are filtered out during parsing. The effective
> keystroke count is often 3–10. Therefore `warmup_steps` must be ≤ 5 to avoid
> all bot files being classified as short-file misses.

---

## HTM-Distance-Stats hyperparameter search on Newton SLURM

Hybrid variant that combines:
- The **QWERTY-distance parser** from `htm_distance/` (only printable ASCII keys, computes Euclidean keyboard distance between successive keys)
- The **statistical-window approach** from `htm/` (sliding windows; 7-dim stats per channel)

Each HTM timestep is a sliding window of `window_size` printable keystrokes.
Three 7-dimensional feature vectors are extracted per window and concatenated into a **21-dim SDR input**:

| Feature group | Stats computed |
|---------------|----------------|
| Dwell times (ms) | mean, median, std, skew, kurtosis, min\_KS, min\_Wasserstein |
| Flight times (ms) | same 7 statistics |
| QWERTY distances (units) | same 7 statistics |

> **Key design choices**: `window_size` ranges from 10–30 keystrokes (same as `htm/`). Bot files shorter than `window_size` produce zero windows and are handled by the short-file rule (auto-misclassified, same as `htm/`). From Round 2 onwards, `warmup_steps` (0–3 windows) and `window_step` (1–2) are explicit hyperparameters. Threshold selection uses **balanced accuracy** to avoid the degenerate thresh = 0 solution that F1 produces when the val set is heavily imbalanced.

The reference pool for KS/Wasserstein comparisons is built from training human files, stored in the model pkl, and used at inference time for `all_other_files` mode.

### Step 1 — Prepare data (run once; requires `split.pkl`)

```bash
python htm/htm_prepare_data.py                              # creates split.pkl if needed
python htm_distance_stats/htm_distance_stats_prepare_data.py  # creates stats_cache.pkl
```

`stats_cache.pkl` stores `{filepath: [(key_idx, dwell_ms, flight_ms, dist_units), ...]}` — the same format as `dist_cache.pkl`. If `dist_cache.pkl` already exists it is used as a seed to avoid re-parsing.

> Feature extraction (windowing + statistics) is **not** performed in prepare_data — it depends on `window_size` which is a per-config hyperparameter and is computed on-the-fly in each SLURM job.

### Step 2 — Generate configs and SLURM script

```bash
python htm_distance_stats/htm_distance_stats_generate_configs.py --n-configs 128 --seed 0
```

Writes `ds_configs/config_NNNN.json` and `slurm/ds_submit_array.sh`.

### Step 3 — Submit to Newton

```bash
sbatch slurm/ds_submit_array.sh
```

Monitor:

```bash
squeue -u $USER
ls ds_results/ | wc -l   # jobs finished so far
```

### Step 4 — Collect results

```bash
python htm_distance_stats/htm_distance_stats_collect_results.py --top 20
```

Reads `ds_results/ds*.json` and writes `ds_results/leaderboard.txt` and `.csv`.

### Step 5 — Test a saved model

```bash
python htm_distance_stats/htm_distance_stats_test_model.py \
    --model ds_models/<slug>.pkl \
    --mode all_non_train
```

Evaluation modes are the same as `htm_distance/`: `orig`, `all_non_train`, `all_other_files`.

The `all_other_files` mode parses new `.txt` files on-the-fly using `parse_file_distance` and computes features using the reference pool stored inside the model pkl — no extra cache needed.

### HTM-Distance-Stats search space

#### Round 1 (configs 0114–0127, 14 runs)

| Group | Parameter | Values |
|-------|-----------|--------|
| SP | `numActiveColumns` | 20, 25, 30, 35, **40** |
| SP | `potentialPct` | 0.65, 0.80 |
| SP | `synPermActiveInc` | 0.02, 0.05, **0.10** |
| SP | `synPermConnected` | 0.10, 0.20 |
| SP | `synPermInactiveDec` | 0.003, 0.005, 0.010 |
| TM | `cellsPerColumn` | 16, 32 |
| TM | `activationThreshold` | 10, 13 |
| TM | `minThreshold` | 8, 10 |
| TM | `maxNewSynapseCount` | 15, 20, 25, 30 |
| TM | `initialPermanence` | 0.21, 0.31, 0.40 |
| TM | `connectedPermanence` | 0.30, 0.50 |
| TM | `permanenceIncrement` | 0.05, 0.10 |
| TM | `permanenceDecrement` | 0.05, 0.10 |
| Enc | `enc_bits_per_feature` | **16**, 24, 32 |
| Enc | `enc_w` | 5, 7, **9** |
| Win | `window_size` | 10, 15, 20, 25, 30 |
| Win | `window_step` | 1 |
| Det | `warmup_steps` | **fixed 0** |
| Det | `al_period` | 5, 10, 15, **20** |

Best Round 1 result: `ds0121` — val BAcc N/A (metric not yet tracked), test F1 = 0.5217 (18/18 bots caught). Threshold selection used F1, which collapsed to thresh = 0 for most configs (degenerate: predict everything as bot due to val-set class imbalance).

#### Round 2 (configs 0128+)

Changes from Round 1 (struck-through values removed, underlined values added):

| Group | Parameter | Values | Change |
|-------|-----------|--------|--------|
| SP | `numActiveColumns` | 20, 25, 30, 35 | removed 40 |
| SP | `synPermActiveInc` | 0.02, 0.05 | removed 0.10 |
| Enc | `enc_bits_per_feature` | 24, 32 | removed 16 (rank 12/14 in R1) |
| Enc | `enc_w` | 5, 7 | removed 9 (marginal; top-3 all used w=5) |
| Win | `window_step` | 1, **2** | added stride-2 for more independent windows |
| Det | `warmup_steps` | **0, 1, 2, 3** | now a hyperparameter |
| Det | `al_period` | 5, 10, 15 | removed 20 |

Threshold selection: **balanced accuracy** = (TPR + TNR) / 2 — avoids the thresh = 0 collapse. Default config anchored to Round 1 best (ds0121: sp_act=25, enc=24w5, ws=20, al=15).

### Clean before re-run (parser or encoder changed)

```bash
rm stats_cache.pkl
rm -rf ds_models/ ds_results/ ds_plots/ ds_configs/
```

---

## HTM-Combined hyperparameter search on Newton SLURM

Full hybrid variant. Each HTM timestep encodes a sliding window of keystrokes as:
- **21-dim window statistics** (7 stats × 3 channels: dwell times, flight times, QWERTY distances)
- **Last-keystroke identity**: key one-hot (95 bits) + dwell + flight + dist scalars

### SDR layout (Round 3 — per-channel / per-scalar encoders)

```
[ dwell_stats | flight_stats | dist_stats | key_one_hot | dwell_key | flight_key | dist_key ]
  7×d_sb        7×f_sb        7×q_sb       95 bits       d_kb        f_kb         q_kb
```

| Block | Bits | Active | Range |
|-------|------|--------|-------|
| Dwell statistics (7 features) | `7 × dwell_stats_enc_bits` | `7 × dwell_stats_enc_w` | Data-derived |
| Flight statistics (7 features) | `7 × flight_stats_enc_bits` | `7 × flight_stats_enc_w` | Data-derived |
| QWERTY-dist statistics (7 features) | `7 × dist_stats_enc_bits` | `7 × dist_stats_enc_w` | Data-derived |
| Key one-hot (95 printable ASCII) | 95 | 1 | Fixed |
| Last-key dwell time | `dwell_scalar_enc_bits` | `dwell_scalar_enc_w` | 0–400 ms |
| Last-key flight time | `flight_scalar_enc_bits` | `flight_scalar_enc_w` | 0–500 ms |
| Last-key QWERTY dist | `dist_scalar_enc_bits` | `dist_scalar_enc_w` | 0–12 units |

All 24 scalar-encoded parameters have **independent** `enc_bits` / `enc_w`.

---

### Before running Round 3 — what to delete

The encoder structure changed between rounds. Here is exactly what to clean and why:

| Directory | Delete? | Reason |
|-----------|---------|--------|
| `hc_configs/` | Auto-deleted | `generate_configs.py` removes old configs automatically |
| `hc_models/` | **Yes — delete** | Old `.pkl` files store a `CombinedEncoder` with the old API; they will not load correctly with the new class |
| `hc_plots/` | Yes (recommended) | Stale plots from old configs; takes disk space |
| `hc_results/` | **No — keep** | Old result JSONs are backward-compatible; the leaderboard script handles all 3 config formats. Keeping them preserves Round 1/2 comparisons and the cumulative config index |
| `stats_cache.pkl` | **No** | Unchanged — reused directly |
| `split.pkl` | **No** | Unchanged |

```bash
# Run this before generating Round 3 configs:
rm -rf hc_models/ hc_plots/
# hc_configs/ is cleaned automatically; hc_results/ should be kept
```

---

### Step 1 — Prepare data (run once, already done if stats_cache.pkl exists)

```bash
python htm/htm_prepare_data.py                               # creates split.pkl if needed
python htm_distance_stats/htm_distance_stats_prepare_data.py # creates stats_cache.pkl
```

### Step 2 — Generate configs and SLURM script

```bash
python htm_combined/htm_combined_generate_configs.py --n-configs 128 --seed 0
```

- Scans `hc_results/hc*.json` to find the highest completed `config_idx` (continues cumulatively)
- **Auto-deletes** old `hc_configs/config_*.json` (already-run; results preserved)
- Writes new configs numbered `last_idx + 1` … `last_idx + N`
- Writes `slurm/hc_submit_array.sh`

### Step 3 — Submit to Newton

```bash
sbatch slurm/hc_submit_array.sh
```

```bash
squeue -u $USER
ls hc_results/ | wc -l   # jobs finished so far
```

### Step 4 — Collect results

```bash
python htm_combined/htm_combined_collect_results.py --top 30
```

Reads all `hc_results/hc*.json` (all rounds), prints ranked leaderboard, writes
`hc_results/leaderboard.txt` and `hc_results/leaderboard.csv`.

> The leaderboard script handles all config generations automatically:
> Round 1 (shared `enc_bits_per_feature`), Round 2 (split `stats_enc_bits`/`scalar_enc_bits`),
> Round 3 (per-channel `dwell_stats_enc_bits` etc.) all display correctly.

### Clean for a fully fresh start (discard all results)

Only if you want to restart from config index 0:

```bash
rm -rf hc_configs/ hc_models/ hc_results/ hc_plots/
```

> Never delete `stats_cache.pkl`, `split.pkl`, `dist_cache.pkl`, or any `ds_*`/`dist_*` dirs.

### HTM-Combined search space

#### Round 1 (configs 0000–0121, 122 runs)

Shared encoder for all 24 scalar parameters (`enc_bits_per_feature` / `enc_w`).

| Group | Parameter | Values |
|-------|-----------|--------|
| SP | `numActiveColumns` | 20, 25, 30, 35, 40 |
| SP | `potentialPct` | 0.65, 0.80 |
| TM | `cellsPerColumn` | 16, 32 |
| TM | `activationThreshold` | 10, 13 |
| TM | `minThreshold` | 8, 10 |
| Enc (shared) | `enc_bits_per_feature` | 16, 24, 32 |
| Enc (shared) | `enc_w` | 5, 7, 9 |
| Win | `window_size` | 5, 10, 15, 20 |
| Win | `window_step` | 1, 2 |
| Det | `warmup_steps` | 0, 1, 2, 3 (windows) |
| Det | `al_period` | 5, 10, 15 |

Best Round 1: `hc0021` — val BAcc=0.8194, **test F1=0.8947**, caught 17/18 bots,
mean detection window=2.0. Key params: `sp=30, enc=16w9, ws=5s1, tm=16, act=13, wu=2, al=15`.

#### Round 2 (configs 0122+)

Key changes from Round 1:

| Group | Parameter | Change |
|-------|-----------|--------|
| SP | `numActiveColumns` | Removed 20, 40 (never in top-5) |
| Enc (stats) | `stats_enc_bits` | **Split from scalar**: 16, 24, 32 |
| Enc (stats) | `stats_enc_w` | 5, 7, 9 |
| Enc (scalar) | `scalar_enc_bits` | **New param**: 8, 16, 24 |
| Enc (scalar) | `scalar_enc_w` | **New param**: 3, 5, 7 |
| Det | `warmup_steps` | Narrowed to 2, 3 (0/1 → degenerate thresh=1.0 in Round 1) |

Best Round 2: `hc0036` — val BAcc=0.8750, **test F1=0.9189**, caught 17/18 bots,
mean detection window=2.0. Key params: `sp=30 pct=0.8, se=32w5, ke=8w5, ws=5s1, tm=16, act=10, wu=2, al=15`.

#### Round 3 (configs 0248+)

All 24 scalar-encoded parameters now have **independent** `enc_bits` / `enc_w`:

| Group | Parameters | Values | Notes |
|-------|-----------|--------|-------|
| Dwell stats (7 features) | `dwell_stats_enc_bits` / `dwell_stats_enc_w` | bits: 16, 24, 32 / w: 5, 7, 9 | 7 statistics of dwell-time distribution |
| Flight stats (7 features) | `flight_stats_enc_bits` / `flight_stats_enc_w` | bits: 16, 24, 32 / w: 5, 7, 9 | 7 statistics of flight-time distribution |
| Dist stats (7 features) | `dist_stats_enc_bits` / `dist_stats_enc_w` | bits: 16, 24, 32 / w: 5, 7, 9 | 7 statistics of QWERTY-distance distribution |
| Dwell scalar (1 feature) | `dwell_scalar_enc_bits` / `dwell_scalar_enc_w` | bits: 8, 16 / w: 3, 5, 7 | Last-keystroke dwell time (0–400 ms) |
| Flight scalar (1 feature) | `flight_scalar_enc_bits` / `flight_scalar_enc_w` | bits: 8, 16 / w: 3, 5, 7 | Last-keystroke flight time (0–500 ms) |
| Dist scalar (1 feature) | `dist_scalar_enc_bits` / `dist_scalar_enc_w` | bits: 8, 16 / w: 3, 5, 7 | Last-keystroke QWERTY distance (0–12 u) |

Round-2 analysis (top-20 configs) motivating the changes:
- `scalar_enc_bits=8` appeared in 75% of top-20; 24 narrowed out, range narrowed to 8/16
- `stats_enc_bits` evenly distributed across 16/24/32 → all values kept per channel
- `tm_maxNewSynapseCount=30` dominant (40%); 15 dropped
- `scalar_enc_w=7` dominant (65%)

Config slug format (Round 3):
```
hc{idx}_sp{act}_d{dsb}w{dsw}_f{fsb}w{fsw}_q{qsb}w{qsw}_dk{dkb}w{dkw}_fk{fkb}w{fkw}_qk{qkb}w{qkw}_ws{ws}s{step}_tm{cells}_act{act}_wu{wu}_al{al}
```

**Backward compatibility**: `htm_combined_train_single.py` supports all 3 config formats via `_get_enc_params(cfg)`. Old result files continue to load and run correctly.

Default anchored on Round-2 winner hc0036: `sp=30 pct=0.8, all_stats=32w5, all_scalars=8w5, ws=5s1, tm=16, act=10, wu=2, al=15`.

---

## HTM-Distance hyperparameter search on Newton SLURM

Variant of the HTM that operates on raw per-keystroke events instead of
statistical windows. Each timestep fed to the HTM encodes:

| Feature | Encoding | Bits | Active |
|---------|----------|------|--------|
| Key type (1 of 95 printable ASCII keys) | One-hot | 95 | 1 |
| Dwell time (key hold duration, ms) | Scalar | `enc_bits_per_feature` | `enc_w` |
| Flight time (gap since previous key-up, ms) | Scalar | `enc_bits_per_feature` | `enc_w` |
| QWERTY distance from previous key (units) | Scalar | `enc_bits_per_feature` | `enc_w` |
| **Total** | | **95 + 3 × enc_bits** | **1 + 3 × enc_w** |

> The key one-hot block is fixed at 95 bits regardless of `enc_bits_per_feature`.
> Best so far: `enc_bits=16, enc_w=9` → SDR = 95+48 = **143 bits, 28 active (19.6% sparsity)**.

### Step 1 — Prepare data (run once; requires `split.pkl`)

```bash
python htm/htm_prepare_data.py          # creates split.pkl if not already done
python htm_distance/htm_distance_prepare_data.py   # creates dist_cache.pkl
```

`dist_cache.pkl` maps every file path to its list of `(key_idx, dwell_ms, flight_ms, dist_units)` tuples.

### Step 2 — Generate configs and SLURM script

```bash
python htm_distance/htm_distance_generate_configs.py --n-configs 128 --seed 0
```

- Scans `dist_results/dist*.json` to find the highest completed index
- Deletes old `dist_configs/config_*.json` (already-run; results are preserved)
- Writes new configs numbered `last_idx + 1` … `last_idx + N`
- Writes `slurm/dist_submit_array.sh`

### Step 3 — Submit to Newton

```bash
# Edit slurm/dist_submit_array.sh to set --partition, --account, and activate your env.
sbatch slurm/dist_submit_array.sh
```

Monitor:

```bash
squeue -u $USER
ls dist_results/ | wc -l          # jobs finished so far
```

### Step 4 — Collect results

```bash
python htm_distance/htm_distance_collect_results.py --top 20
```

Reads all `dist_results/dist*.json` and writes `dist_results/leaderboard.txt` and `.csv`.

Leaderboard columns include:

| Column | Description |
|--------|-------------|
| `Val F1` / `Test F1` | F1 score on validation / test split |
| `Thresh` | Chosen detection threshold |
| `det=` | Mean keystroke index at which the first bot was detected (post-warmup) |
| `caught=X/Y` | Number of bot files correctly flagged / total bot files |
| `bLik=` | Mean AnomalyLikelihood score for bot files |
| `hLik=` | Mean AnomalyLikelihood score for human files |

The SLURM `.out` log for each job also prints per-bot detection details, e.g.:
```
  Bot 0: detected at keystroke 5  (likelihood=0.842, file_len=8)
  Bot 1: NOT DETECTED  (max_lik=0.021 < 0.0281, file_len=6)
  Bot 2: NOT DETECTED  (short file: 3 ≤ warmup=5)
```

### Step 5 — Test a saved model

```bash
python htm_distance/htm_distance_test_model.py \
    --model dist_models/<slug>.pkl \
    --mode all_non_train
```

#### Evaluation modes (`--mode`)

| Mode | Description |
|------|-------------|
| `orig` | Val and test splits from `split.pkl` (same files used during training) |
| `all_non_train` | All files in `dist_cache.pkl` that were not in the training set (default) |
| `all_other_files` | Walk `--data_root` for new human `.txt` files + `--bots_root` for bot files; parsed on the fly — no cache needed |

#### Additional flags

| Flag | Default | Effect |
|------|---------|--------|
| `--data_root` | `../UB_keystroke_dataset/` | Human files root for `all_other_files` mode |
| `--bots_root` | `../BadUSBdataset` | Bot files root for `all_other_files` mode |
| `--write_decision_log` | off | Write a per-keystroke log for one human + one bot file |

#### Decision log (`--write_decision_log`)

Written to `logs/<slug>_human_log.txt` and `logs/<slug>_bot_log.txt`.
One row per keystroke (score column shows AnomalyLikelihood, or raw anomaly if AL
is not available in the pkl):

```
 Step  Key    Dwell   Flight    Dist  Likelihood          Status  Reason
------  ---  -------  -------  ------  ----------  --------------  --------------------------------------------------
     0  'I'    141.0      0.0   0.000      0.0000          WARMUP  warmup (1/5)
     5  'H'     78.0    110.0   2.693      0.0210      no crossing  score 0.0210 < 0.0281
     6  'E'    110.0      0.0   2.000      0.8420             BOT  first crossing: 0.8420 >= 0.0281
     7  ' '    109.0      0.0   4.243      0.8420             BOT  already triggered at step 6
```

Short files (fewer keystrokes than `warmup_steps`) are noted in the log header
and automatically counted as misclassified (see [Short-file rule](#short-file-misclassification-rule)).

> Key difference from the original HTM tester: no reference pool or feature windows
> are needed. `parse_file_distance` runs directly on raw `.txt` files, so
> `all_other_files` mode is fast and requires no pre-computation step.

### Clean before re-run (encoder or parser changed)

If you change the encoder (`DistanceEncoder`) or the key parser (`parse_file_distance` /
`key_to_char`), the cache and all old results are stale and must be wiped:

```bash
rm dist_cache.pkl
rm -rf dist_models/ dist_results/ dist_plots/ dist_configs/
```

Then start from Step 1 above.

> `split.pkl` and `features_cache.pkl` are **not** affected and do not need to be deleted.

### Best config so far (Round 5, 83 runs)

`dist0121`: `sp_act=40, pct=0.8, enc=16w9, tm=16, act=13, min=8, wu=5, al=15`
→ **val F1=0.8571, test F1=0.9189**, mean detection keystroke=5.0, caught 17/18 bots

```python
import pickle
with open("dist_models/dist0121_sp40_enc16w9_tm16_act13_wu5_al15_vf10.8571_tf10.9189.pkl", "rb") as f:
    m = pickle.load(f)
sp, tm, encoder, thresh, al = m["sp"], m["tm"], m["encoder"], m["best_thresh"], m["al"]
```

### HTM-Distance search space (Round 5)

| Group | Parameter | Values | Notes |
|-------|-----------|--------|-------|
| SP | `numActiveColumns` | 20, 25, 30, 35, 40 | |
| SP | `potentialPct` | 0.65, 0.80 | |
| SP | `synPermActiveInc` | 0.02, 0.05, 0.10 | |
| SP | `synPermConnected` | 0.10, 0.20 | |
| SP | `synPermInactiveDec` | 0.003, 0.005, 0.010 | |
| TM | `cellsPerColumn` | 16, 32 | |
| TM | `activationThreshold` | 10, 13 | |
| TM | `minThreshold` | 8, 10 | |
| TM | `maxNewSynapseCount` | 15, 20, 25, 30 | |
| TM | `initialPermanence` | 0.21, 0.31, 0.40 | |
| TM | `connectedPermanence` | 0.30, 0.50 | |
| TM | `permanenceIncrement` | 0.05, 0.10 | |
| TM | `permanenceDecrement` | 0.05, 0.10 | |
| Enc | `enc_bits_per_feature` | 16, 24, 32 | |
| Enc | `enc_w` | 5, 7, 9 | |
| Det | `warmup_steps` | **1, 2, 3, 5** | Must be ≤ 5 — BadUSB files have 3–10 printable keystrokes |
| Det | `al_period` | **5, 10, 15, 20** | AnomalyLikelihood history window; **new in Round 5** |

**Bold** = added/changed in Round 5. Warmup reduced from 150–300 → 1–5 because
BadUSB attack files contain very few printable keystrokes after filtering modifiers.

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
