import threading
import queue
import time
import os
import sys
import pickle
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# --- SETTINGS ---
WINDOW_SIZE = 15  # MLP / GRU window size (keystrokes)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_MLP_PATH  = os.path.join(_project_root, "badusb_model.pth")
SCALER_MLP_PATH = os.path.join(_project_root, "scaler_params.npy")
REF_POOL_PATH   = os.path.join(_project_root, "reference_pool.npz")

MODEL_GRU_PATH  = os.path.join(_project_root, "gru_model.pth")
SCALER_GRU_PATH = os.path.join(_project_root, "rnn_scaler_params.npy")

MODEL_HTM_PATH  = os.path.join(
    _project_root, "hc_models",
    "hc0499_sp35_d16w7_f24w5_q24w9_dk16w7_fk8w7_qk8w7_ws10s1_tm16_act13_wu2_al15_vf10.8571_tf10.9189.pkl"
)
HTM_APP_WARMUP       = 5   # windows to skip before alarming (covers TM-reset spike)
HTM_CALIB_SKIP_FIRST = 5   # skip first N warmup windows from adaptive calibration (TM reset spike)
HTM_DEBUG      = True  # print per-window anomaly details for HTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = "MLP"

# --- HTM IMPORTS (optional — disabled gracefully if htm.core not installed) ---
_htm_available = False
try:
    for _d in (_project_root,
               os.path.join(_project_root, "htm_combined"),
               os.path.join(_project_root, "htm_distance")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from htm.bindings.sdr import SDR as _SDR
    from htm_distance_common import key_to_char, char_to_idx, qwerty_distance
    from htm_combined_common import extract_combined_window
    _htm_available = True
except ImportError as _e:
    print(f"HTM not available ({_e}); HTM mode disabled.")

# --- QUEUES ---
raw_events_queue = queue.Queue()
gui_update_queue = queue.Queue()
control_queue    = queue.Queue()

# ==============================================================================
# 1. MODEL ARCHITECTURES
# ==============================================================================
class BadUSBClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BadUSBClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),    # net.0
            nn.BatchNorm1d(32),          # net.1
            nn.ReLU(),                   # net.2
            nn.Dropout(0.3),             # net.3
            nn.Linear(32, 64),           # net.4
            nn.BatchNorm1d(64),          # net.5
            nn.ReLU(),                   # net.6
            nn.Dropout(0.3),             # net.7
            nn.Linear(64, 512),          # net.8
            nn.BatchNorm1d(512),         # net.9
            nn.ReLU(),                   # net.10
            nn.Dropout(0.3),             # net.11
            nn.Linear(512, 1),           # net.12
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class GRUBotDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(GRUBotDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)


# ==============================================================================
# 2. INITIALIZATION AND WEIGHT LOADING
# ==============================================================================
print("Loading models...")
try:
    mlp_model = BadUSBClassifier(input_dim=14).to(device)
    mlp_model.load_state_dict(torch.load(MODEL_MLP_PATH, map_location=device))
    mlp_model.eval()
    mlp_scaler_mean, mlp_scaler_scale = np.load(SCALER_MLP_PATH)

    gru_model = GRUBotDetector(input_dim=2, hidden_dim=64, num_layers=1).to(device)
    gru_model.load_state_dict(torch.load(MODEL_GRU_PATH, map_location=device))
    gru_model.eval()
    gru_scaler_mean, gru_scaler_scale = np.load(SCALER_GRU_PATH)

    refs = np.load(REF_POOL_PATH)
    ref_dwells, ref_flights = refs['dwell'], refs['flight']
    print("✅ MLP + GRU models loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# HTM model (optional)
htm_model_data      = None
htm_sp              = None
htm_tm              = None
htm_encoder         = None
htm_al              = None
htm_input_width     = 0
htm_best_thresh     = 0.5
htm_warmup          = 2
htm_window_size     = 10
htm_ref_dwells      = None
htm_ref_flights     = None
htm_ref_dists       = None
htm_active_cols_sdr  = None
htm_has_live_thresh  = False   # True when model has a properly calibrated live_thresh

if _htm_available:
    try:
        with open(MODEL_HTM_PATH, 'rb') as _f:
            htm_model_data = pickle.load(_f)
        htm_sp          = htm_model_data['sp']
        htm_tm          = htm_model_data['tm']
        htm_encoder     = htm_model_data['encoder']
        htm_input_width = htm_model_data['input_width']
        htm_warmup      = htm_model_data.get('warmup_steps', 2)
        htm_window_size = htm_model_data.get('window_size', 10)
        htm_ref_dwells  = htm_model_data['ref_dwells']
        htm_ref_flights = htm_model_data['ref_flights']
        htm_ref_dists   = htm_model_data['ref_dists']
        htm_active_cols_sdr = _SDR(htm_sp.getColumnDimensions())

        # Prefer live_thresh (calibrated in per-file mode matching live deployment).
        # Fall back to best_thresh (calibrated in shared-AL mode) + adaptive.
        htm_has_live_thresh = 'live_thresh' in htm_model_data
        htm_best_thresh = htm_model_data.get('live_thresh',
                                             htm_model_data['best_thresh'])
        _thresh_source = 'live' if htm_has_live_thresh else 'shared-AL (adaptive fallback)'

        # Restore AL from post-training frozen state if available;
        # otherwise fall back to the post-evaluation state.
        if 'al_live_bytes' in htm_model_data:
            htm_al = pickle.loads(htm_model_data['al_live_bytes'])
            _al_source = 'frozen post-training'
        else:
            htm_al = htm_model_data['al']
            _al_source = 'post-evaluation (old model)'

        print(f"✅ HTM model loaded  (window={htm_window_size}, "
              f"thresh={htm_best_thresh:.3f} [{_thresh_source}], "
              f"AL: {_al_source})")
    except Exception as _e:
        print(f"⚠️  HTM model failed to load: {_e}")
        htm_model_data = None


# ==============================================================================
# 3. HELPERS
# ==============================================================================
def get_timestamp_ms():
    return int(time.time() * 1000)


def extract_features_mlp(window_data, reference_pool):
    """7-dim feature extractor for MLP inference."""
    feat_mean = np.mean(window_data)
    feat_med  = np.median(window_data)
    feat_std  = np.std(window_data)
    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0.0, 10.0
    else:
        feat_skew = float(stats.skew(window_data))
        feat_kurt = float(stats.kurtosis(window_data))
    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))
    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt,
            float(np.min(ks_scores)), float(np.min(w_scores))]


# ==============================================================================
# 4. DATA COLLECTION THREAD
# ==============================================================================
def input_listener_thread():
    def on_press(key):
        raw_events_queue.put(("KeyDown", str(key), get_timestamp_ms()))

    def on_release(key):
        raw_events_queue.put(("KeyUp", str(key), get_timestamp_ms()))

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ==============================================================================
# 5. ANALYSIS THREAD
# ==============================================================================
def processing_thread():
    # MLP / GRU state
    active_keys   = {}
    last_keyup_ts = None
    dwell_buffer  = []
    flight_buffer = []

    # HTM state
    htm_active_keys      = {}       # key_str → (char, down_ts)
    htm_prev_char        = None
    htm_prev_keyup_ts    = None
    htm_buffer           = []       # list of (key_idx, dwell_ms, flight_ms, dist)
    htm_win_count        = 0        # windows processed since last reset
    htm_warmup_al_scores = []       # AL values collected during warmup
    htm_effective_thresh = None     # adaptive threshold set after warmup

    consecutive_strikes = 0

    while True:
        # ── 1. RESET ───────────────────────────────────────────────────────────
        while not control_queue.empty():
            cmd = control_queue.get()
            if cmd == "RESET":
                consecutive_strikes = 0
                dwell_buffer.clear()
                flight_buffer.clear()
                active_keys.clear()
                last_keyup_ts = None
                htm_buffer.clear()
                htm_active_keys.clear()
                htm_prev_char        = None
                htm_prev_keyup_ts    = None
                htm_win_count        = 0
                htm_warmup_al_scores = []
                htm_effective_thresh = None
                if htm_model_data is not None:
                    htm_tm.reset()
                while not raw_events_queue.empty():
                    raw_events_queue.get()

        # ── 2. RECEIVE EVENT ───────────────────────────────────────────────────
        try:
            action, key, ts = raw_events_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # ── 3. UPDATE BUFFERS ──────────────────────────────────────────────────
        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup_ts is not None:
                delta = ts - last_keyup_ts
                if 0 < delta < 2000:
                    flight_buffer.append(float(delta))
            # HTM: track printable key presses
            if _htm_available:
                char = key_to_char(key)
                if char is not None:
                    htm_active_keys[key] = (char, ts)

        elif action == "KeyUp":
            last_keyup_ts = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = ts - down_ts
                if 0 < delta < 2000:
                    dwell_buffer.append(float(delta))
            # HTM: complete a printable keystroke
            if _htm_available and key in htm_active_keys:
                char, down_ts = htm_active_keys.pop(key)
                dwell = float(ts - down_ts)
                if 0 < dwell < 3000:
                    flight = float(down_ts - htm_prev_keyup_ts) if htm_prev_keyup_ts is not None else 0.0
                    flight = max(0.0, min(flight, 5000.0))
                    dist   = qwerty_distance(htm_prev_char, char) if htm_prev_char is not None else 0.0
                    htm_buffer.append((char_to_idx(char), dwell, flight, dist))
                    htm_prev_char     = char
                    htm_prev_keyup_ts = ts

        # ── 4. INFERENCE ───────────────────────────────────────────────────────
        score      = None
        avg_flight = 0.0

        if CURRENT_MODEL in ("MLP", "GRU"):
            if len(dwell_buffer) >= WINDOW_SIZE and len(flight_buffer) >= WINDOW_SIZE:
                w_d = np.array(dwell_buffer[:WINDOW_SIZE])
                w_f = np.array(flight_buffer[:WINDOW_SIZE])
                avg_flight = float(np.mean(w_f))
                try:
                    if CURRENT_MODEL == "MLP":
                        ft_d = extract_features_mlp(w_d, ref_dwells)
                        ft_f = extract_features_mlp(w_f, ref_flights)
                        raw  = np.array(ft_f + ft_d)
                        sc   = (raw - mlp_scaler_mean) / mlp_scaler_scale
                        t    = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            score = mlp_model(t).item()
                    else:  # GRU
                        seq = np.column_stack((w_d, w_f))
                        seq = (seq - gru_scaler_mean) / gru_scaler_scale
                        t   = torch.tensor(seq.reshape(1, WINDOW_SIZE, 2),
                                           dtype=torch.float32).to(device)
                        with torch.no_grad():
                            score = gru_model(t).item()
                except Exception as ex:
                    print(f"Inference error: {ex}")
                    continue
                dwell_buffer.pop(0)
                flight_buffer.pop(0)

        elif CURRENT_MODEL == "HTM" and htm_model_data is not None:
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
                        raw_anomaly = float(htm_tm.anomaly)
                        al_score    = htm_al.compute(raw_anomaly)
                        htm_win_count += 1
                        in_warmup = htm_win_count <= HTM_APP_WARMUP
                        # For new models with live_thresh: threshold is already correct,
                        # no adaptive calibration needed. Warmup only suppresses TM-reset spike.
                        # For old models (best_thresh only): adaptively calibrate from warmup.
                        if in_warmup and not htm_has_live_thresh:
                            if htm_win_count > HTM_CALIB_SKIP_FIRST:
                                htm_warmup_al_scores.append(al_score)
                            if htm_win_count == HTM_APP_WARMUP:
                                if htm_warmup_al_scores:
                                    floor = float(np.percentile(htm_warmup_al_scores, 95))
                                    # Cap at 0.99; floor*2 puts threshold well above the human AL floor
                                    htm_effective_thresh = min(max(floor * 2.0, htm_best_thresh, floor + 0.05), 0.99)
                                    print(f"HTM adaptive threshold: {htm_effective_thresh:.4f}  "
                                          f"(warmup floor p95={floor:.4f}, trained={htm_best_thresh:.3f})")
                                else:
                                    htm_effective_thresh = htm_best_thresh
                                    print(f"HTM adaptive threshold: fallback to trained={htm_best_thresh:.3f}")
                        effective_thresh = htm_effective_thresh if htm_effective_thresh is not None else htm_best_thresh
                        if HTM_DEBUG:
                            d_mean = float(np.mean([e[1] for e in htm_buffer[:htm_window_size]]))
                            f_mean = float(np.mean([e[2] for e in htm_buffer[:htm_window_size]]))
                            q_mean = float(np.mean([e[3] for e in htm_buffer[:htm_window_size]]))
                            flag = " <<< ALARM" if (not in_warmup and al_score >= effective_thresh) else ""
                            warmup_tag = f"[warmup {htm_win_count}/{HTM_APP_WARMUP}]" if in_warmup else ""
                            print(f"HTM win={htm_win_count:>4} {warmup_tag:<18} "
                                  f"raw={raw_anomaly:.3f}  AL={al_score:.3f}  thresh={effective_thresh:.3f}  "
                                  f"dwell={d_mean:.0f}ms  flight={f_mean:.0f}ms  dist={q_mean:.2f}{flag}")
                        if not in_warmup:
                            score = al_score
                    except Exception as ex:
                        print(f"HTM inference error: {ex}")
                        htm_buffer.pop(0)
                        continue
                htm_buffer.pop(0)

        # ── 5. STRIKE LOGIC & GUI UPDATE ───────────────────────────────────────
        if score is None:
            continue

        if CURRENT_MODEL == "HTM":
            alarm_thresh = htm_effective_thresh if htm_effective_thresh is not None else htm_best_thresh
        else:
            alarm_thresh = 0.8
        required_strikes = 1 if CURRENT_MODEL in ("GRU", "HTM") else 3

        if score >= alarm_thresh:
            consecutive_strikes += 1
        else:
            consecutive_strikes = 0

        if consecutive_strikes >= required_strikes:
            gui_update_queue.put({
                "type":   "ALARM",
                "avg":    avg_flight,
                "prob":   score,
                "status": f"!!! BAD USB DETECTED ({CURRENT_MODEL}) !!!"
            })
        else:
            status_txt = "Suspicious..." if score > 0.5 else "Human (Safe)"
            gui_update_queue.put({
                "type":   "update_stats",
                "avg":    avg_flight,
                "prob":   score,
                "status": status_txt
            })


# ==============================================================================
# 6. GUI (MAIN THREAD)
# ==============================================================================
class BadUSBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BadUSB Detector")
        self.root.geometry("400x560")
        self.is_alarm_triggered = False

        self.frame = ttk.Frame(root, padding="20")
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, width=150, height=150)
        self.canvas.pack(pady=10)
        self.light = self.canvas.create_oval(10, 10, 140, 140,
                                             fill="green", outline="black", width=3)

        self.lbl_status = ttk.Label(self.frame, text="STATUS: MONITORING",
                                    font=("Arial", 14, "bold"))
        self.lbl_status.pack(pady=10)

        self.lbl_stats = ttk.Label(self.frame,
                                   text="Last Batch Avg: -- ms\nBot Probability: --%",
                                   font=("Courier", 10))
        self.lbl_stats.pack(pady=5)

        self.model_var = tk.StringVar(value="MLP")
        ttk.Label(self.frame, text="Select AI Engine:",
                  font=("Arial", 9, "bold")).pack(pady=(10, 0))
        ttk.Radiobutton(self.frame, text="Statistic + MLP (3 Strikes)",
                        variable=self.model_var, value="MLP",
                        command=self.change_model).pack()
        ttk.Radiobutton(self.frame, text="Sequence + GRU (Instant)",
                        variable=self.model_var, value="GRU",
                        command=self.change_model).pack()
        htm_rb = ttk.Radiobutton(self.frame, text="HTM Combined (1 Strike)",
                                  variable=self.model_var, value="HTM",
                                  command=self.change_model)
        htm_rb.pack()
        if htm_model_data is None:
            htm_rb.config(state="disabled")

        self.btn_reset = ttk.Button(self.frame, text="RESET ALARM",
                                    command=self.reset_system)
        self.btn_reset.pack(pady=15)

        self.root.after(100, self.process_gui_queue)

    def change_model(self):
        global CURRENT_MODEL
        new_model = self.model_var.get()
        if new_model == CURRENT_MODEL:
            return
        CURRENT_MODEL = new_model
        print(f"Model switched to: {CURRENT_MODEL}")
        self.reset_system()

    def reset_system(self):
        self.is_alarm_triggered = False
        self.canvas.itemconfig(self.light, fill="green")
        self.lbl_status.config(text="STATUS: MONITORING", foreground="black")
        self.lbl_stats.config(text="Last Batch Avg: -- ms\nBot Probability: --%")
        control_queue.put("RESET")
        print("System reset.")

    def set_red_alert(self, avg, prob):
        if not self.is_alarm_triggered:
            self.is_alarm_triggered = True
            self.canvas.itemconfig(self.light, fill="red")
            self.lbl_status.config(text="⚠️ ATTACK DETECTED ⚠️", foreground="red")
            self.lbl_stats.config(text=f"Last Batch Avg: {avg:.2f} ms\nBot Probability: {prob*100:.1f}%")

    def update_stats(self, avg, prob, status_text):
        if not self.is_alarm_triggered:
            self.lbl_stats.config(text=f"Last Batch Avg: {avg:.2f} ms\nBot Probability: {prob*100:.1f}%")
            self.lbl_status.config(text=status_text)
            color = "yellow" if "Suspicious" in status_text else "green"
            self.canvas.itemconfig(self.light, fill=color)

    def process_gui_queue(self):
        try:
            while True:
                msg = gui_update_queue.get_nowait()
                if msg["type"] == "ALARM":
                    self.set_red_alert(msg["avg"], msg["prob"])
                elif msg["type"] == "update_stats":
                    self.update_stats(msg["avg"], msg["prob"], msg["status"])
        except queue.Empty:
            pass
        self.root.after(100, self.process_gui_queue)


if __name__ == "__main__":
    t_listener  = threading.Thread(target=input_listener_thread, daemon=True)
    t_processor = threading.Thread(target=processing_thread, daemon=True)
    t_listener.start()
    t_processor.start()

    root = tk.Tk()
    app  = BadUSBApp(root)
    root.mainloop()
