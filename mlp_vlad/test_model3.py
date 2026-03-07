import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
import os
import glob

# --- КОНФИГУРАЦИЯ ---
TEST_FOLDER = "s0/rotation"  # Папка для проверки
MODEL_PATH = "badusb_model.pth"
SCALER_PATH = "scaler_params.npy"
REF_POOL_PATH = "reference_pool.npz"

WINDOW_SIZE = 15
STEP_SIZE = 1  # Шаг проверки (1 для максимальной детальности)
THRESHOLD = 0.5

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    flights = []
    active_keys = {} 
    last_keyup = None

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
        key, action = parts[0], parts[1]
        try:
            ts = int(parts[2])
        except ValueError: continue

        scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown":
            active_keys[key] = ts
            if last_keyup is not None:
                delta = (ts - last_keyup) / scale
                if 0 < delta < 5000: flights.append(delta)
        elif action == "KeyUp":
            last_keyup = ts
            if key in active_keys:
                down_ts = active_keys.pop(key)
                delta = (ts - down_ts) / scale
                if 0 < delta < 3000: dwells.append(delta)
                    
    return np.array(dwells), np.array(flights)

def extract_features(window_data, reference_pool):
    if len(window_data) < WINDOW_SIZE: return None
    
    feat_mean = np.mean(window_data)
    feat_med = np.median(window_data)
    feat_std = np.std(window_data)
    
    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0, 10
    else:
        feat_skew = stats.skew(window_data)
        feat_kurt = stats.kurtosis(window_data)

    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))
    
    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt, np.min(ks_scores), np.min(w_scores)]

# ==============================================================================
# 3. ОСНОВНОЙ СКРИПТ ПРОВЕРКИ (ОПТИМИЗИРОВАННЫЙ С ВЫВОДОМ ON-THE-FLY)
# ==============================================================================
def main():
    print(f"--- ЗАПУСК ТЕСТИРОВАНИЯ (Папка: {TEST_FOLDER}) ---")

    if not os.path.exists(REF_POOL_PATH):
        print(f"Ошибка: Не найден {REF_POOL_PATH}")
        return
    
    refs = np.load(REF_POOL_PATH)
    ref_dwells, ref_flights = refs['dwell'], refs['flight']

    if not os.path.exists(SCALER_PATH):
        print(f"Ошибка: Не найден {SCALER_PATH}")
        return
    
    scaler_mean, scaler_scale = np.load(SCALER_PATH)

    model = BadUSBClassifier(input_dim=14).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    files = glob.glob(os.path.join(TEST_FOLDER, "*.txt"))
    if not files:
        print(f"Папка {TEST_FOLDER} пуста или не существует.")
        return

    total_files_tested = 0
    total_bots_detected = 0

    print(f"\nНайдено файлов для проверки: {len(files)}")
    
    # ПЕЧАТАЕМ ШАПКУ ТАБЛИЦЫ ДО ЦИКЛА
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
        
        # 1. СОБИРАЕМ ПРИЗНАКИ
        for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
            w_d = dwells[i : i + WINDOW_SIZE]
            w_f = flights[i : i + WINDOW_SIZE]
            
            ft_d = extract_features(w_d, ref_dwells)
            ft_f = extract_features(w_f, ref_flights)
            
            if ft_d and ft_f:
                file_features.append(ft_f + ft_d)

        if not file_features:
            continue

        # 2. МАССОВАЯ ОБРАБОТКА
        raw_features = np.array(file_features) 
        scaled_features = (raw_features - scaler_mean) / scaler_scale 
        tensor_x = torch.tensor(scaled_features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs = model(tensor_x).squeeze(-1).cpu().numpy() 
            
        if probs.ndim == 0:
            probs = [float(probs)]

        # 3. АНАЛИЗ РЕЗУЛЬТАТОВ
        consecutive_strikes = 0     
        STRIKES_REQUIRED = 3        
        alarm_triggered = False     
        bot_windows_count = 0  
        
        for prob in probs:
            if prob > 0.8:
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
            
        # ПЕЧАТАЕМ РЕЗУЛЬТАТ ПО ФАЙЛУ СРАЗУ ЖЕ
        bad_win_str = f"{bot_windows_count}/{len(probs)}"
        print(f"{filename[:28]:<30} | {verdict:<10} | {avg_prob:<5.2f} | {bad_win_str:<15}")

    # ИТОГОВАЯ СТАТИСТИКА ПОСЛЕ ОКОНЧАНИЯ ВСЕХ ФАЙЛОВ
    print("="*75)
    print("\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего файлов проверено: {total_files_tested}")
    print(f"Распознано как БОТ (False Positive): {total_bots_detected}")
    print(f"Распознано как ЧЕЛОВЕК (True Negative): {total_files_tested - total_bots_detected}")
    if total_files_tested > 0:
        accuracy = ((total_files_tested - total_bots_detected) / total_files_tested) * 100
        print(f"Точность на людях:     {accuracy:.2f}%")
        
    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    main()
