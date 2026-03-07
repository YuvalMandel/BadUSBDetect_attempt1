import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import joblib # Добавлено для загрузки полиномиальной модели

# --- КОНФИГУРАЦИЯ ---
TEST_FOLDER = "test_dataset" # Папка для проверки
MODEL_PATH = "badusb_model.pth"
SCALER_PATH = "scaler_params.npy"
REF_POOL_PATH = "reference_pool.npz"
POLY_MODEL_PATH = "poly_regressor.pkl" # Путь к вашей модели Тейлора

WINDOW_SIZE = 15
STEP_SIZE = 10 
THRESHOLD = 0.5

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 0. ТОПОЛОГИЯ КЛАВИАТУРЫ (Координаты X, Y)
# ==============================================================================
KEYBOARD_MAP = {
    'q': (0, 3, 'L'), 'w': (1, 3, 'L'), 'e': (2, 3, 'L'), 'r': (3, 3, 'L'), 't': (4, 3, 'L'),
    'y': (5, 3, 'R'), 'u': (6, 3, 'R'), 'i': (7, 3, 'R'), 'o': (8, 3, 'R'), 'p': (9, 3, 'R'),
    'a': (0.5, 2, 'L'), 's': (1.5, 2, 'L'), 'd': (2.5, 2, 'L'), 'f': (3.5, 2, 'L'), 'g': (4.5, 2, 'L'),
    'h': (5.5, 2, 'R'), 'j': (6.5, 2, 'R'), 'k': (7.5, 2, 'R'), 'l': (8.5, 2, 'R'),
    'z': (1, 1, 'L'), 'x': (2, 1, 'L'), 'c': (3, 1, 'L'), 'v': (4, 1, 'L'), 'b': (5, 1, 'L'),
    'n': (6, 1, 'R'), 'm': (7, 1, 'R'),
    'space': (4.5, 0, 'N')
}

# ==============================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ
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
        x = self.sigmoid(self.output(x))
        return x

# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def parse_file(filepath):
    dwells = []
    flights_detailed = [] 
    active_keys = {} 
    last_keyup_ts = None
    last_keyup_name = None

    if not os.path.exists(filepath):
        return [], []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return [], []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        key, action = parts[0].lower(), parts[1]
        try:
            ts = int(parts[2])
        except ValueError: continue

        scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts) / scale
                if 0 < delta < 5000: 
                    flights_detailed.append((delta, last_keyup_name, key))
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts) / scale
                if 0 < delta < 3000: 
                    dwells.append(delta)
                    
    return np.array(dwells), flights_detailed


def extract_features(w_d, w_f_detailed, reference_pool_d, reference_pool_f, poly_model):
    if len(w_d) < WINDOW_SIZE or len(w_f_detailed) < WINDOW_SIZE: return None
    
    # --- Dwell (7 старых фичей) ---
    w_d_arr = np.array(w_d)
    d_std = np.std(w_d_arr)
    d_skew = stats.skew(w_d_arr) if d_std >= 0.0001 else 0
    d_kurt = stats.kurtosis(w_d_arr) if d_std >= 0.0001 else 10
    d_ks = min([stats.ks_2samp(w_d_arr, r, method='asymp')[0] for r in reference_pool_d])
    d_w = min([stats.wasserstein_distance(w_d_arr, r) for r in reference_pool_d])
    d_features = [np.mean(w_d_arr), np.median(w_d_arr), d_std, d_skew, d_kurt, d_ks, d_w]

    # --- Flight (7 старых фичей) ---
    w_f_arr = np.array([item[0] for item in w_f_detailed])
    f_std = np.std(w_f_arr)
    f_skew = stats.skew(w_f_arr) if f_std >= 0.0001 else 0
    f_kurt = stats.kurtosis(w_f_arr) if f_std >= 0.0001 else 10
    f_ks = min([stats.ks_2samp(w_f_arr, r, method='asymp')[0] for r in reference_pool_f])
    f_w = min([stats.wasserstein_distance(w_f_arr, r) for r in reference_pool_f])
    f_features = [np.mean(w_f_arr), np.median(w_f_arr), f_std, f_skew, f_kurt, f_ks, f_w]

    # --- Полиномиальные ошибки Закона Фиттса (3 новые фичи) ---
    X_poly = []
    y_true = []

    for t, k1, k2 in w_f_detailed:
        if k1 in KEYBOARD_MAP and k2 in KEYBOARD_MAP:
            x1, y1 = KEYBOARD_MAP[k1][:2]
            x2, y2 = KEYBOARD_MAP[k2][:2]
            X_poly.append([x1, y1, x2, y2])
            y_true.append(t)

    # Защита: если в окне были только "неизвестные" кнопки вроде Enter или Ctrl
    if len(X_poly) > 0:
        X_poly = np.array(X_poly)
        y_true = np.array(y_true)
        # Получаем предсказания модели
        y_pred = poly_model.predict(X_poly)
        
        # Вычисляем абсолютные ошибки
        errors = np.abs(y_true - y_pred)
        err_mean = np.mean(errors)
        err_med = np.median(errors)
        err_std = np.std(errors)
    else:
        err_mean, err_med, err_std = 0.0, 0.0, 0.0

    # Возвращаем ровно 17 фичей
    return d_features + f_features + [err_mean, err_med, err_std]

# ==============================================================================
# 3. ОСНОВНОЙ СКРИПТ ПРОВЕРКИ 
# ==============================================================================
def main():
    print(f"--- ЗАПУСК ТЕСТИРОВАНИЯ (Папка: {TEST_FOLDER}) ---")

    # 1. Загрузка полиномиальной модели (Ряд Тейлора)
    try:
        poly_model = joblib.load(POLY_MODEL_PATH)
        print("✅ Полиномиальный регрессор успешно загружен.")
    except Exception as e:
        print(f"❌ Ошибка загрузки полиномиальной модели: {e}")
        return

    if not os.path.exists(REF_POOL_PATH):
        print(f"Ошибка: Не найден {REF_POOL_PATH}")
        return
    
    refs = np.load(REF_POOL_PATH)
    ref_dwells, ref_flights = refs['dwell'], refs['flight']

    if not os.path.exists(SCALER_PATH):
        print(f"Ошибка: Не найден {SCALER_PATH}")
        return
    
    scaler_mean, scaler_scale = np.load(SCALER_PATH)

    # ИЗМЕНЕНИЕ: input_dim = 17
    model = BadUSBClassifier(input_dim=17).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    files = glob.glob(os.path.join(TEST_FOLDER, "*.txt"))
    if not files:
        print(f"Папка {TEST_FOLDER} пуста или не существует.")
        return

    total_files_tested = 0
    total_bots_detected = 0

    print(f"\nНайдено файлов для проверки: {len(files)}")
    
    print("\n" + "="*75)
    print(f"{'FILENAME':<30} | {'VERDICT':<10} | {'AVG':<5} | {'BAD WINDOWS':<15}")
    print("="*75)

    for filepath in files:
        filename = os.path.basename(filepath)
        dwells, flights = parse_file(filepath)
        min_len = min(len(dwells), len(flights))
        
        if min_len < WINDOW_SIZE:
            print(f"{filename[:28]:<30} | {'SKIPPED':<10} | {'-':<5} | Not enough data ({min_len})")
            continue

        file_features = []
        
        # СОБИРАЕМ ПРИЗНАКИ
        for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
            w_d = dwells[i : i + WINDOW_SIZE]
            w_f = flights[i : i + WINDOW_SIZE]
            
            # Передаем poly_model в функцию извлечения признаков
            feats = extract_features(w_d, w_f, ref_dwells, ref_flights, poly_model)
            
            if feats:
                file_features.append(feats)

        if not file_features:
            continue

        # МАССОВАЯ ОБРАБОТКА
        raw_features = np.array(file_features) 
        scaled_features = (raw_features - scaler_mean) / scaler_scale 
        tensor_x = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs = model(tensor_x).squeeze(-1).cpu().numpy() 
            
        if probs.ndim == 0:
            probs = [float(probs)]

        # АНАЛИЗ РЕЗУЛЬТАТОВ
        consecutive_strikes = 0     
        STRIKES_REQUIRED = 3        
        alarm_triggered = False     
        bot_windows_count = 0  
        
        for prob in probs:
            if prob > 0.5:
                consecutive_strikes += 1
                bot_windows_count += 1
            else:
                consecutive_strikes = 0 

            if consecutive_strikes >= STRIKES_REQUIRED:
                alarm_triggered = True

        avg_prob = np.mean(probs)
        
        if alarm_triggered or avg_prob > 0.8:
            verdict = "🔴 BOT"
            total_bots_detected += 1
        else:
            verdict = "🟢 HUMAN"
            
        total_files_tested += 1
            
        bad_win_str = f"{bot_windows_count}/{len(probs)}"
        print(f"{filename[:28]:<30} | {verdict:<10} | {avg_prob:<5.2f} | {bad_win_str:<15}")

    print("="*75)
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего файлов проверено: {total_files_tested}")
    print(f"Распознано как БОТ (False Positive): {total_bots_detected}")
    print(f"Распознано как ЧЕЛОВЕК (True Negative): {total_files_tested - total_bots_detected}")
    if total_files_tested > 0:
        accuracy = ((total_files_tested - total_bots_detected) / total_files_tested) * 100
        bot_accuracy = ((total_bots_detected) / total_files_tested) * 100
        print(f"Точность на людях:     {accuracy:.2f}%")
        print(f"Bot acc:     {bot_accuracy:.2f}%")
        
    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    main()
