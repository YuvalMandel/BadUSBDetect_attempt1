import threading
import queue
import time
import os
import tkinter as tk
from tkinter import ttk
from pynput import keyboard
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

# --- SETTINGS ---
WINDOW_SIZE = 15

MODEL_MLP_PATH = "badusb_model.pth"
SCALER_MLP_PATH = "scaler_params.npy"
REF_POOL_PATH = "reference_pool.npz"

MODEL_GRU_PATH = "gru_model.pth"
SCALER_GRU_PATH = "rnn_scaler_params.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_MODEL = "MLP" 

# --- QUEUES ---
raw_events_queue = queue.Queue()
gui_update_queue = queue.Queue()
control_queue = queue.Queue() # NEW QUEUE FOR RESET BUTTON

# ==============================================================================
# 1. MODEL ARCHITECTURES
# ==============================================================================
class BadUSBClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BadUSBClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.sigmoid(self.output(x))

class GRUBotDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(GRUBotDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
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
    print("✅ Models loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

def get_timestamp_ms():
    return int(time.time() * 1000)

def extract_features(window_data, reference_pool):
    feat_mean = np.mean(window_data)
    feat_med = np.median(window_data)
    feat_std = np.std(window_data)
    
    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0, 10
    else:
        feat_skew = float(stats.skew(window_data))
        feat_kurt = float(stats.kurtosis(window_data))

    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))
    
    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt, np.min(ks_scores), np.min(w_scores)]

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
# 5. ANALYSIS THREAD (FIXED STEP AND LOGIC)
# ==============================================================================
def processing_thread():
    active_keys = {}
    last_keyup_ts = None
    
    dwell_buffer = []
    flight_buffer = []
    consecutive_strikes = 0

    while True:
        # --- 1. RESET COMMAND PROCESSING ---
        while not control_queue.empty():
            cmd = control_queue.get()
            if cmd == "RESET":
                consecutive_strikes = 0
                dwell_buffer.clear()
                flight_buffer.clear()
                active_keys.clear()
                last_keyup_ts = None
                # Clear raw events queue to avoid processing old attack data
                while not raw_events_queue.empty():
                    raw_events_queue.get()

        # --- 2. RECEIVE NEW EVENT ---
        try:
            # Use timeout so loop can regularly check control_queue for Reset commands
            action, key, ts = raw_events_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup_ts is not None:
                delta = ts - last_keyup_ts
                if 0 < delta < 2000:
                    flight_buffer.append(delta)
                    
        elif action == "KeyUp":
            last_keyup_ts = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = ts - down_ts
                if 0 < delta < 2000:
                    dwell_buffer.append(delta)

        # --- 3. SLIDING WINDOW CHECK ---
        if len(dwell_buffer) >= WINDOW_SIZE and len(flight_buffer) >= WINDOW_SIZE:
            
            w_d = np.array(dwell_buffer[:WINDOW_SIZE])
            w_f = np.array(flight_buffer[:WINDOW_SIZE])
            
            avg_flight = np.mean(w_f)
            prob = 0.0
            
            try:
                if CURRENT_MODEL == "MLP":
                    ft_d = extract_features(w_d, ref_dwells)
                    ft_f = extract_features(w_f, ref_flights)
                    raw_features = np.array(ft_f + ft_d)
                    scaled_features = (raw_features - mlp_scaler_mean) / mlp_scaler_scale
                    tensor_x = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prob = mlp_model(tensor_x).item()
                        
                elif CURRENT_MODEL == "GRU":
                    seq = np.column_stack((w_d, w_f))
                    seq_flat = seq.reshape(-1, 2)
                    seq_scaled = (seq_flat - gru_scaler_mean) / gru_scaler_scale
                    seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, 2)
                    
                    tensor_x = torch.tensor(seq_scaled, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        prob = gru_model(tensor_x).item()
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            # --- STRIKE LOGIC (MODIFIED) ---
            # MLP requires 3 consecutive strikes, GRU requires only 1.
            required_strikes = 3 if CURRENT_MODEL == "MLP" else 1

            if prob > 0.8:
                consecutive_strikes += 1
            else:
                consecutive_strikes = 0

            if consecutive_strikes >= required_strikes:
                gui_update_queue.put({
                    "type": "ALARM",
                    "avg": avg_flight,
                    "prob": prob,
                    "status": f"!!! BAD USB DETECTED ({CURRENT_MODEL}) !!!"
                })
            else:
                status_txt = "Suspicious..." if prob > 0.5 else "Human (Safe)"
                gui_update_queue.put({
                    "type": "update_stats",
                    "avg": avg_flight,
                    "prob": prob,
                    "status": status_txt
                })

            # --- TRUE SLIDING WINDOW (STEP = 1) ---
            # Remove the oldest keystroke, leaving 14 fresh ones for the next pass
            dwell_buffer.pop(0)
            flight_buffer.pop(0)

# ==============================================================================
# 6. GUI (MAIN THREAD)
# ==============================================================================
class BadUSBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BadUSB Detector")
        self.root.geometry("400x500") # Slightly enlarged window for button
        self.is_alarm_triggered = False

        self.frame = ttk.Frame(root, padding="20")
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, width=150, height=150)
        self.canvas.pack(pady=10)
        self.light = self.canvas.create_oval(10, 10, 140, 140, fill="green", outline="black", width=3)

        self.lbl_status = ttk.Label(self.frame, text="STATUS: MONITORING", font=("Arial", 14, "bold"))
        self.lbl_status.pack(pady=10)

        self.lbl_stats = ttk.Label(self.frame, text="Last Batch Avg: -- ms\nBot Probability: --%", font=("Courier", 10))
        self.lbl_stats.pack(pady=5)
        
        self.model_var = tk.StringVar(value="MLP")
        ttk.Label(self.frame, text="Select AI Engine:", font=("Arial", 9, "bold")).pack(pady=(10, 0))
        ttk.Radiobutton(self.frame, text="Statistic + MLP (3 Strikes)", variable=self.model_var, value="MLP", command=self.change_model).pack()
        ttk.Radiobutton(self.frame, text="Sequence + GRU (Instant)", variable=self.model_var, value="GRU", command=self.change_model).pack()

        # --- RESET BUTTON ---
        self.btn_reset = ttk.Button(self.frame, text="RESET ALARM", command=self.reset_system)
        self.btn_reset.pack(pady=15)

        self.root.after(100, self.process_gui_queue)

    def change_model(self):
        global CURRENT_MODEL
        CURRENT_MODEL = self.model_var.get()
        print(f"Model switched to: {CURRENT_MODEL}")
        self.reset_system() # Reset when switching models - good practice

    def reset_system(self):
        """Reset GUI and send cleanup command to background thread"""
        self.is_alarm_triggered = False
        self.canvas.itemconfig(self.light, fill="green")
        self.lbl_status.config(text="STATUS: MONITORING", foreground="black")
        self.lbl_stats.config(text="Last Batch Avg: -- ms\nBot Probability: --%")
        
        # Send command to background thread
        control_queue.put("RESET")
        print("System reset to initial state.")

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
            
            if "Suspicious" in status_text:
                self.canvas.itemconfig(self.light, fill="yellow")
            else:
                self.canvas.itemconfig(self.light, fill="green")

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
    t_listener = threading.Thread(target=input_listener_thread, daemon=True)
    t_listener.start()

    t_processor = threading.Thread(target=processing_thread, daemon=True)
    t_processor.start()

    root = tk.Tk()
    app = BadUSBApp(root)
    root.mainloop()