"""
BadUSBv0 — Live Detector (fullkey models)

Three switchable engines:
  MLP  — 17-feature window (dwell stats + flight stats + poly error), threshold=0.01, 3 strikes
  GRU  — (dwell, flight) sequence, 2-layer, threshold=0.92, 1 strike
  HTM  — CombinedEncoder (stats21 + last-keystroke), anomaly likelihood, 1 strike

Run from project root or BadUSBv0/BadUSB/BadUSBDetector/:
  python detector_v0.py
"""

import os
import sys
import json
import pickle
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
import joblib
import tkinter as tk
from tkinter import ttk
from pynput import keyboard as pynput_keyboard

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_here        = os.path.dirname(os.path.abspath(__file__))
_badusb_root = os.path.dirname(_here)                              # BadUSBv0/BadUSB/
_v0_root     = os.path.dirname(_badusb_root)                       # BadUSBv0/
_project_root = os.path.dirname(_v0_root)                          # repo root

MLP_MODEL_PATH   = os.path.join(_badusb_root, "MLP", "badusb_model_fullkey.pth")
MLP_SCALER_PATH  = os.path.join(_badusb_root, "MLP", "scaler_params_fullkey.npy")
MLP_REFS_PATH    = os.path.join(_badusb_root, "MLP", "reference_pool.npz")
MLP_POLY_PATH    = os.path.join(_badusb_root, "MLP", "poly_regressor.pkl")
MLP_HPS_PATH     = os.path.join(_badusb_root, "results", "MLP", "mlp_best_hps_fullkey.json")

GRU_MODEL_PATH   = os.path.join(_badusb_root, "GRU", "gru_model_full_fullkey.pth")
GRU_SCALER_PATH  = os.path.join(_badusb_root, "GRU", "rnn_scaler_params_full_fullkey.npy")
GRU_HPS_PATH     = os.path.join(_badusb_root, "results", "GRU", "gru_best_hps_fullkey.json")

HTM_MODEL_PATH   = os.path.join(_badusb_root, "results", "HTM", "models",
                                 "htm_model_0449_vf10.7333_fullkey.pkl")

# Make BadUSBv0 HTM modules importable (CombinedEncoder lives here)
for _p in [
    os.path.join(_badusb_root, "HTM"),
    os.path.join(_project_root, "htm_distance"),
    os.path.join(_project_root, "common"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 15   # MLP and GRU window (keystrokes), matches training
HTM_APP_WARMUP  = 5    # skip first N windows after reset (TM starts blank → all anomalous)

# QWERTY keyboard map used for poly-error feature (MLP) — same as dataset_csv_generator.py
_MLP_KMAP = {
    'q': (0, 3), 'w': (1, 3), 'e': (2, 3), 'r': (3, 3), 't': (4, 3),
    'y': (5, 3), 'u': (6, 3), 'i': (7, 3), 'o': (8, 3), 'p': (9, 3),
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'space': (4.5, 0),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = "MLP"

# ──────────────────────────────────────────────────────────────────────────────
# Model architectures
# ──────────────────────────────────────────────────────────────────────────────
class _MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _GRUDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc      = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        return self.sigmoid(self.fc(self.dropout(out[:, -1, :])))


# ──────────────────────────────────────────────────────────────────────────────
# Load MLP
# ──────────────────────────────────────────────────────────────────────────────
print("Loading MLP...")
with open(MLP_HPS_PATH) as f:
    _mlp_hps = json.load(f)
MLP_THRESHOLD   = _mlp_hps.get("threshold", 0.01)
_mlp_hidden     = tuple(_mlp_hps.get("hidden_dims", [256]))
_mlp_dropout    = _mlp_hps.get("dropout", 0.10)

mlp_model = _MLPClassifier(17, _mlp_hidden, _mlp_dropout).to(device)
mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device))
mlp_model.eval()

_mlp_scaler = np.load(MLP_SCALER_PATH)
mlp_scaler_mean, mlp_scaler_scale = _mlp_scaler[0], _mlp_scaler[1]

_mlp_refs     = np.load(MLP_REFS_PATH)
mlp_ref_dwells  = list(_mlp_refs['dwell'])
mlp_ref_flights = list(_mlp_refs['flight'])

poly_model = joblib.load(MLP_POLY_PATH)
print(f"  MLP loaded (hidden={_mlp_hidden}, dropout={_mlp_dropout}, threshold={MLP_THRESHOLD})")

# ──────────────────────────────────────────────────────────────────────────────
# Load GRU
# ──────────────────────────────────────────────────────────────────────────────
print("Loading GRU...")
with open(GRU_HPS_PATH) as f:
    _gru_hps = json.load(f)
GRU_THRESHOLD  = _gru_hps.get("threshold", 0.92)
_gru_hidden    = _gru_hps.get("hidden_dim", 64)
_gru_layers    = _gru_hps.get("num_layers", 2)
_gru_dropout   = _gru_hps.get("dropout", 0.15)

gru_model = _GRUDetector(2, _gru_hidden, _gru_layers, _gru_dropout).to(device)
gru_model.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=device))
gru_model.eval()

_gru_scaler    = np.load(GRU_SCALER_PATH)
gru_scaler_mean, gru_scaler_scale = _gru_scaler[0], _gru_scaler[1]
print(f"  GRU loaded (hidden={_gru_hidden}, layers={_gru_layers}, threshold={GRU_THRESHOLD})")

# ──────────────────────────────────────────────────────────────────────────────
# Load HTM (optional)
# ──────────────────────────────────────────────────────────────────────────────
_htm_available = False
htm_model_data = None
htm_sp = htm_tm = htm_encoder = htm_al = htm_active_cols_sdr = None
htm_input_width = htm_warmup = 0
htm_window_size = 10
HTM_THRESHOLD   = 0.5
htm_ref_dwells = htm_ref_flights = htm_ref_dists = None

try:
    from htm.bindings.sdr import SDR as _SDR
    from htm_combined_common import extract_combined_window, make_anomaly_likelihood
    from htm_distance_common import qwerty_distance, char_to_idx, key_to_char

    with open(HTM_MODEL_PATH, 'rb') as f:
        htm_model_data = pickle.load(f)

    htm_sp          = htm_model_data['sp']
    htm_tm          = htm_model_data['tm']
    htm_encoder     = htm_model_data['encoder']
    htm_input_width = htm_model_data['input_width']
    htm_warmup      = htm_model_data.get('warmup_steps', 0)
    htm_window_size = htm_model_data.get('window_size', 10)
    htm_ref_dwells  = htm_model_data['ref_dwells']
    htm_ref_flights = htm_model_data['ref_flights']
    htm_ref_dists   = htm_model_data['ref_dists']
    htm_active_cols_sdr = _SDR(htm_sp.getColumnDimensions())
    HTM_THRESHOLD   = htm_model_data.get('live_thresh', htm_model_data['best_thresh'])

    if 'al_live_bytes' in htm_model_data:
        htm_al = pickle.loads(htm_model_data['al_live_bytes'])
    else:
        htm_al = htm_model_data.get('al', make_anomaly_likelihood(
            htm_model_data.get('al_period', 10)))

    _htm_available = True
    print(f"  HTM loaded (cfg_0449, window={htm_window_size}, threshold={HTM_THRESHOLD:.4f})")
except Exception as _e:
    print(f"  HTM not available: {_e}")

# ──────────────────────────────────────────────────────────────────────────────
# Queues
# ──────────────────────────────────────────────────────────────────────────────
raw_events_queue = queue.Queue()
gui_update_queue = queue.Queue()
control_queue    = queue.Queue()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pynput_to_mlp_key(key_str):
    """Map pynput str(key) to MLP keyboard-map name, or None if not mapped."""
    if key_str == 'Key.space':
        return 'space'
    s = key_str.strip("'")
    if len(s) == 1 and s.lower() in _MLP_KMAP:
        return s.lower()
    return None


def _extract_mlp_features(w_d, w_f_detailed):
    """17-feature extraction matching dataset_csv_generator.py."""
    d_arr = np.array(w_d, dtype=float)
    d_std = float(np.std(d_arr))
    d_features = [
        float(np.mean(d_arr)), float(np.median(d_arr)), d_std,
        float(scipy_stats.skew(d_arr))     if d_std >= 0.0001 else 0.0,
        float(scipy_stats.kurtosis(d_arr)) if d_std >= 0.0001 else 10.0,
        float(min(scipy_stats.ks_2samp(d_arr, r, method='asymp')[0] for r in mlp_ref_dwells)),
        float(min(scipy_stats.wasserstein_distance(d_arr, r)         for r in mlp_ref_dwells)),
    ]

    f_arr = np.array([item[0] for item in w_f_detailed], dtype=float)
    f_std = float(np.std(f_arr))
    f_features = [
        float(np.mean(f_arr)), float(np.median(f_arr)), f_std,
        float(scipy_stats.skew(f_arr))     if f_std >= 0.0001 else 0.0,
        float(scipy_stats.kurtosis(f_arr)) if f_std >= 0.0001 else 10.0,
        float(min(scipy_stats.ks_2samp(f_arr, r, method='asymp')[0] for r in mlp_ref_flights)),
        float(min(scipy_stats.wasserstein_distance(f_arr, r)         for r in mlp_ref_flights)),
    ]

    X_poly, y_true = [], []
    for flight_ms, k1, k2 in w_f_detailed:
        if 20 < flight_ms < 400 and k1 in _MLP_KMAP and k2 in _MLP_KMAP:
            x1, y1 = _MLP_KMAP[k1]
            x2, y2 = _MLP_KMAP[k2]
            X_poly.append([x1, y1, x2, y2])
            y_true.append(flight_ms)

    if X_poly:
        errors = np.abs(np.array(y_true) - poly_model.predict(np.array(X_poly)))
        poly_feats = [float(np.mean(errors)), float(np.median(errors)), float(np.std(errors))]
    else:
        poly_feats = [0.0, 0.0, 0.0]

    return d_features + f_features + poly_feats  # 17 floats


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard listener thread
# ──────────────────────────────────────────────────────────────────────────────
def input_listener_thread():
    def on_press(key):
        raw_events_queue.put(("KeyDown", str(key), int(time.time() * 1000)))

    def on_release(key):
        raw_events_queue.put(("KeyUp", str(key), int(time.time() * 1000)))

    with pynput_keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ──────────────────────────────────────────────────────────────────────────────
# Processing thread
# ──────────────────────────────────────────────────────────────────────────────
def processing_thread():
    # Shared timing state
    active_keys    = {}
    last_keyup_ts  = None
    last_key_name  = None   # MLP: normalized key name of last released key

    # MLP buffers
    dwell_buf          = []   # float ms
    flight_detailed_buf = []  # (float ms, prev_key_name, curr_key_name)

    # GRU buffers (simple dwell + flight pairs)
    gru_dwell_buf  = []
    gru_flight_buf = []

    # HTM state
    htm_buffer    = []   # (key_idx, dwell, flight, dist)
    htm_prev_char = None
    htm_prev_keyup_ts = None
    htm_win_count = 0
    htm_app_warmup_count = 0   # app-level warmup: skip first HTM_APP_WARMUP windows
    htm_al_state  = pickle.loads(htm_model_data['al_live_bytes']) if (
        htm_model_data and 'al_live_bytes' in htm_model_data) else htm_al

    consecutive_strikes = 0

    def _reset():
        nonlocal last_keyup_ts, last_key_name, consecutive_strikes
        nonlocal htm_prev_char, htm_prev_keyup_ts, htm_win_count, htm_app_warmup_count, htm_al_state
        consecutive_strikes = 0
        last_keyup_ts = None
        last_key_name = None
        dwell_buf.clear(); flight_detailed_buf.clear()
        gru_dwell_buf.clear(); gru_flight_buf.clear()
        htm_buffer.clear()
        htm_prev_char = None; htm_prev_keyup_ts = None; htm_win_count = 0; htm_app_warmup_count = 0
        if htm_model_data and 'al_live_bytes' in htm_model_data:
            htm_al_state = pickle.loads(htm_model_data['al_live_bytes'])
        if htm_model_data:
            htm_tm.reset()
        while not raw_events_queue.empty():
            raw_events_queue.get()

    while True:
        # ── Reset ──
        while not control_queue.empty():
            if control_queue.get() == "RESET":
                _reset()

        # ── Receive event ──
        try:
            action, key_str, ts = raw_events_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # ── KeyDown ──
        if action == "KeyDown":
            active_keys[key_str] = ts

            if last_keyup_ts is not None:
                flight_ms = float(ts - last_keyup_ts)
                if 0 < flight_ms < 2000:
                    gru_flight_buf.append(flight_ms)
                    curr_key = _pynput_to_mlp_key(key_str)
                    if last_key_name is not None and curr_key is not None:
                        flight_detailed_buf.append((flight_ms, last_key_name, curr_key))
                    elif len(flight_detailed_buf) < len(gru_flight_buf):
                        # keep buffers aligned even for unmapped keys
                        flight_detailed_buf.append((flight_ms, None, None))

            # HTM
            if _htm_available:
                char = key_to_char(key_str)
                if char and htm_prev_keyup_ts is not None:
                    # flight stored on KeyDown; assembled into buffer on KeyUp
                    pass  # flight computed from KeyUp timing below

        # ── KeyUp ──
        elif action == "KeyUp":
            if key_str not in active_keys:
                continue
            down_ts = active_keys.pop(key_str)
            dwell_ms = float(ts - down_ts)
            if not (0 < dwell_ms < 2000):
                continue

            last_key_name = _pynput_to_mlp_key(key_str)
            last_keyup_ts = ts
            dwell_buf.append(dwell_ms)
            gru_dwell_buf.append(dwell_ms)

            # HTM
            if _htm_available:
                char = key_to_char(key_str)
                if char:
                    flight = float(down_ts - htm_prev_keyup_ts) if htm_prev_keyup_ts is not None else 0.0
                    flight = max(0.0, min(flight, 5000.0))
                    dist   = qwerty_distance(htm_prev_char, char) if htm_prev_char is not None else 0.0
                    htm_buffer.append((char_to_idx(char), dwell_ms, flight, dist))
                    htm_prev_char     = char
                    htm_prev_keyup_ts = ts

        # ── Inference ──
        score     = None
        avg_flight = 0.0

        if CURRENT_MODEL == "MLP":
            if (len(dwell_buf) >= WINDOW_SIZE and
                    len(flight_detailed_buf) >= WINDOW_SIZE):
                w_d = dwell_buf[:WINDOW_SIZE]
                w_f = flight_detailed_buf[:WINDOW_SIZE]
                avg_flight = float(np.mean([x[0] for x in w_f]))
                try:
                    feat = np.array(_extract_mlp_features(w_d, w_f), dtype=np.float32)
                    feat = ((feat - mlp_scaler_mean) / mlp_scaler_scale).astype(np.float32)
                    t    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        score = mlp_model(t).item()
                except Exception as ex:
                    print(f"MLP inference error: {ex}")
                dwell_buf.pop(0); flight_detailed_buf.pop(0)
                if gru_dwell_buf:  gru_dwell_buf.pop(0)
                if gru_flight_buf: gru_flight_buf.pop(0)

        elif CURRENT_MODEL == "GRU":
            if (len(gru_dwell_buf) >= WINDOW_SIZE and
                    len(gru_flight_buf) >= WINDOW_SIZE):
                w_d = np.array(gru_dwell_buf[:WINDOW_SIZE], dtype=np.float32)
                w_f = np.array(gru_flight_buf[:WINDOW_SIZE], dtype=np.float32)
                avg_flight = float(np.mean(w_f))
                seq = np.column_stack((w_d, w_f))
                seq = (seq - gru_scaler_mean) / gru_scaler_scale
                try:
                    t = torch.tensor(seq.reshape(1, WINDOW_SIZE, 2),
                                     dtype=torch.float32).to(device)
                    with torch.no_grad():
                        score = gru_model(t).item()
                except Exception as ex:
                    print(f"GRU inference error: {ex}")
                gru_dwell_buf.pop(0); gru_flight_buf.pop(0)
                if dwell_buf:          dwell_buf.pop(0)
                if flight_detailed_buf: flight_detailed_buf.pop(0)

        elif CURRENT_MODEL == "HTM" and _htm_available:
            if len(htm_buffer) >= htm_window_size:
                result = extract_combined_window(
                    htm_buffer, 0, htm_window_size,
                    htm_ref_dwells, htm_ref_flights, htm_ref_dists)
                if result is not None:
                    stats_21, kid, dw, fl, di = result
                    avg_flight = float(np.mean([e[2] for e in htm_buffer[:htm_window_size]]))
                    try:
                        enc_sdr = _SDR(htm_input_width)
                        enc_sdr.dense = htm_encoder.encode(stats_21, kid, dw, fl, di)
                        htm_sp.compute(enc_sdr, False, htm_active_cols_sdr)
                        htm_tm.compute(htm_active_cols_sdr, learn=False)
                        al_score = htm_al_state.compute(float(htm_tm.anomaly))
                        htm_win_count += 1
                        if htm_win_count > htm_warmup:
                            htm_app_warmup_count += 1
                            if htm_app_warmup_count > HTM_APP_WARMUP:
                                score = al_score
                            else:
                                print(f"HTM app warmup {htm_app_warmup_count}/{HTM_APP_WARMUP} "
                                      f"AL={al_score:.3f}")
                    except Exception as ex:
                        print(f"HTM inference error: {ex}")
                htm_buffer.pop(0)

        if score is None:
            continue

        # ── Strike logic ──
        if CURRENT_MODEL == "MLP":
            alarm_thresh     = MLP_THRESHOLD
            required_strikes = 3
        elif CURRENT_MODEL == "GRU":
            alarm_thresh     = GRU_THRESHOLD
            required_strikes = 1
        else:
            alarm_thresh     = HTM_THRESHOLD
            required_strikes = 1

        if score >= alarm_thresh:
            consecutive_strikes += 1
        else:
            consecutive_strikes = 0

        if consecutive_strikes >= required_strikes:
            gui_update_queue.put({
                "type": "ALARM", "avg": avg_flight, "prob": score,
                "status": f"!!! BAD USB DETECTED ({CURRENT_MODEL}) !!!"
            })
        else:
            # "Suspicious" only when meaningfully elevated:
            # MLP/GRU use 0-1 probability → suspicious above 0.5
            # HTM uses anomaly likelihood → suspicious above 80% of threshold
            if CURRENT_MODEL in ("MLP", "GRU"):
                suspicious = score > 0.5
            else:
                suspicious = score > alarm_thresh * 0.8
            status = "Suspicious..." if suspicious else "Human (Safe)"
            gui_update_queue.put({
                "type": "update_stats", "avg": avg_flight,
                "prob": score, "status": status
            })


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────
class BadUSBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BadUSB Detector v0 (Fullkey Models)")
        self.root.geometry("420x580")
        self.is_alarm = False

        frame = ttk.Frame(root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, width=150, height=150)
        self.canvas.pack(pady=10)
        self.light = self.canvas.create_oval(10, 10, 140, 140,
                                             fill="green", outline="black", width=3)

        self.lbl_status = ttk.Label(frame, text="STATUS: MONITORING",
                                    font=("Arial", 14, "bold"))
        self.lbl_status.pack(pady=10)

        self.lbl_stats = ttk.Label(frame,
                                   text="Avg flight: -- ms\nScore: --",
                                   font=("Courier", 10))
        self.lbl_stats.pack(pady=5)

        ttk.Label(frame, text="Select AI Engine:", font=("Arial", 9, "bold")).pack(pady=(10, 0))

        self.model_var = tk.StringVar(value="MLP")
        ttk.Radiobutton(frame, text=f"MLP (17-feat, thresh={MLP_THRESHOLD:.2f}, 3 strikes)",
                        variable=self.model_var, value="MLP",
                        command=self.change_model).pack(anchor="w", padx=40)
        ttk.Radiobutton(frame, text=f"GRU (2-layer, thresh={GRU_THRESHOLD:.2f}, 1 strike)",
                        variable=self.model_var, value="GRU",
                        command=self.change_model).pack(anchor="w", padx=40)

        htm_rb = ttk.Radiobutton(
            frame,
            text=f"HTM cfg_0449 (combined, thresh={HTM_THRESHOLD:.4f}, 1 strike)"
                 if _htm_available else "HTM (not available)",
            variable=self.model_var, value="HTM",
            command=self.change_model)
        htm_rb.pack(anchor="w", padx=40)
        if not _htm_available:
            htm_rb.config(state="disabled")

        ttk.Button(frame, text="RESET ALARM", command=self.reset_system).pack(pady=15)

        self.root.after(100, self.process_gui_queue)

    def change_model(self):
        global CURRENT_MODEL
        new = self.model_var.get()
        if new != CURRENT_MODEL:
            CURRENT_MODEL = new
            print(f"Switched to: {CURRENT_MODEL}")
            self.reset_system()

    def reset_system(self):
        self.is_alarm = False
        self.canvas.itemconfig(self.light, fill="green")
        self.lbl_status.config(text="STATUS: MONITORING", foreground="black")
        self.lbl_stats.config(text="Avg flight: -- ms\nScore: --")
        control_queue.put("RESET")

    def process_gui_queue(self):
        try:
            while True:
                msg = gui_update_queue.get_nowait()
                if msg["type"] == "ALARM":
                    if not self.is_alarm:
                        self.is_alarm = True
                        self.canvas.itemconfig(self.light, fill="red")
                        self.lbl_status.config(text=msg["status"], foreground="red")
                        self.lbl_stats.config(
                            text=f"Avg flight: {msg['avg']:.1f} ms\nScore: {msg['prob']:.4f}")
                elif msg["type"] == "update_stats" and not self.is_alarm:
                    color = "yellow" if "Suspicious" in msg["status"] else "green"
                    self.canvas.itemconfig(self.light, fill=color)
                    self.lbl_status.config(text=msg["status"], foreground="black")
                    self.lbl_stats.config(
                        text=f"Avg flight: {msg['avg']:.1f} ms\nScore: {msg['prob']:.4f}")
        except queue.Empty:
            pass
        self.root.after(100, self.process_gui_queue)


if __name__ == "__main__":
    print(f"Device: {device}")
    t1 = threading.Thread(target=input_listener_thread, daemon=True)
    t2 = threading.Thread(target=processing_thread,    daemon=True)
    t1.start(); t2.start()
    root = tk.Tk()
    BadUSBApp(root)
    root.mainloop()
