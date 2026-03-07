import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
import os
import glob

# --- КОНФИГУРАЦИЯ ---
TEST_FOLDER = "s2/rotation/"
MODEL_PATH = "badusb_model.pth"
SCALER_PATH = "scaler_params.npy"
REF_POOL_PATH = "reference_pool.npz"

WINDOW_SIZE = 30
STEP_SIZE = 10  # Шаг проверки (можно 1 для детальности, 10 для скорости)
THRESHOLD = 0.5 # Порог уверенности (если > 0.5, то Бот)

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. АРХИТЕКТУРА МОДЕЛИ (Должна точь-в-точь совпадать с training скриптом)
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
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Парсер и Фичи)
# ==============================================================================
def parse_file(filepath):
    dwells = []
    flights = []
    active_keys = {} 
    last_keyup = None

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return [], []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading: {e}")
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
# 3. ОСНОВНОЙ СКРИПТ ПРОВЕРКИ
# ==============================================================================
# ... (весь код до функции main остается без изменений) ...

# ==============================================================================
# 3. ОСНОВНОЙ СКРИПТ ПРОВЕРКИ
# ==============================================================================
def main():
    print("--- ЗАПУСК СИСТЕМЫ ДЕТЕКЦИИ (S.T.R.I.K.E. SYSTEM) ---")

    # 1. Загрузка Эталонов
    if not os.path.exists(REF_POOL_PATH):
        print(f"Ошибка: Не найден {REF_POOL_PATH}")
        return
    refs = np.load(REF_POOL_PATH)
    ref_dwells = refs['dwell']
    ref_flights = refs['flight']
    print("Эталоны загружены.")

    # 2. Загрузка Скейлера
    if not os.path.exists(SCALER_PATH):
        print(f"Ошибка: Не найден {SCALER_PATH}")
        return
    scaler_mean, scaler_scale = np.load(SCALER_PATH)
    print("Параметры скейлера загружены.")

    # 3. Загрузка Модели
    model = BadUSBClassifier(input_dim=14).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Нейросеть загружена.")
    
    # 4. Сканирование папки
    files = glob.glob(os.path.join(TEST_FOLDER, "*.txt"))
    if not files:
        print(f"Папка {TEST_FOLDER} пуста или не существует.")
        return

    results_table = []

    print(f"\nНайдено файлов для проверки: {len(files)}")
    print("-" * 60)

    for filepath in files:
        filename = os.path.basename(filepath)
        dwells, flights = parse_file(filepath)
        
        min_len = min(len(dwells), len(flights))
        
        if min_len < WINDOW_SIZE:
            print(f"Skipping {filename}: Not enough data ({min_len} events)")
            continue

        # --- !!! ИЗМЕНЕНИЕ: ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ СТРАЙКОВ !!! ---
        file_probs = [] 
        consecutive_strikes = 0     # Счетчик страйков подряд
        STRIKES_REQUIRED = 3        # Порог срабатывания (нужно 3 окна подряд)
        alarm_triggered = False     # Сработала ли сирена
        bot_windows_count = 0       # Статистика "плохих" окон
        # ---------------------------------------------------------
        
        for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
            w_d = dwells[i : i + WINDOW_SIZE]
            w_f = flights[i : i + WINDOW_SIZE]
            
            ft_d = extract_features(w_d, ref_dwells)
            ft_f = extract_features(w_f, ref_flights)
            
            if ft_d and ft_f:
                raw_features = np.array(ft_f + ft_d)
                scaled_features = (raw_features - scaler_mean) / scaler_scale
                tensor_x = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prob = model(tensor_x).item()
                    file_probs.append(prob)

                # --- !!! ИЗМЕНЕНИЕ: ЛОГИКА ПРОВЕРКИ !!! ---
                # Если сеть уверена > 80%, что это бот
                if prob > 0.8:
                    consecutive_strikes += 1
                    bot_windows_count += 1
                else:
                    # Если попалось нормальное окно - сбрасываем счетчик!
                    # Это спасет человека, который сделал 1 случайную ошибку.
                    consecutive_strikes = 0 

                # Если накопили 3 страйка подряд — поднимаем тревогу
                if consecutive_strikes >= STRIKES_REQUIRED:
                    alarm_triggered = True
                # ------------------------------------------

        if not file_probs:
            continue

        # АНАЛИЗ РЕЗУЛЬТАТОВ ПО ФАЙЛУ
        avg_prob = np.mean(file_probs)
        max_prob = np.max(file_probs)
        
        # --- !!! ИЗМЕНЕНИЕ: ФИНАЛЬНЫЙ ВЕРДИКТ !!! ---
        # Бот, если: Сработали СТРАЙКИ  ИЛИ  Среднее значение очень высокое
        if alarm_triggered or avg_prob > 0.6:
            verdict = "🔴 BOT (ATTACK)"
        else:
            verdict = "🟢 HUMAN (SAFE)"
        # -------------------------------------------
            
        results_table.append({
            "File": filename,
            "Verdict": verdict,
            "Avg_Score": f"{avg_prob:.4f}",
            "Max_Score": f"{max_prob:.4f}",
            "Bad_Windows": f"{bot_windows_count}/{len(file_probs)}" # Добавил статистику
        })

    # ВЫВОД ТАБЛИЦЫ
    print("\n" + "="*95)
    print(f"{'FILENAME':<30} | {'VERDICT':<15} | {'AVG':<8} | {'MAX':<8} | {'BAD WIN':<10}")
    print("="*95)
    
    for row in results_table:
        print(f"{row['File']:<30} | {row['Verdict']:<15} | {row['Avg_Score']:<8} | {row['Max_Score']:<8} | {row['Bad_Windows']:<10}")
    print("="*95)
    print(f"Logic: Alarm if {STRIKES_REQUIRED} consecutive windows > 0.8 prob OR Avg > 0.6")
    a = input()
if __name__ == "__main__":
    main()