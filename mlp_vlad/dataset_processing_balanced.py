import os
import glob
import random
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- ГИПЕРПАРАМЕТРЫ ---
WINDOW_SIZE = 15        
STEP_SIZE_HUMAN = 10    
STEP_SIZE_BOT = 1       
NUM_REFERENCES = 50     
SYNTHETIC_SAMPLES = 3000 

FOLDERS = {
    "Humans": ["s2"],  
    "Bots": ["only_timings_dataset", "BadUSBdataset"]                  
}

OUTPUT_CSV = "balanced_dataset.csv"
OUTPUT_REFS = "reference_pool.npz" 

# ==============================================================================
# 1. ПАРСЕР (Без изменений логики, но вынесен для pickling)
# ==============================================================================
def parse_file(filepath):
    dwells = []
    flights = []
    active_keys = {} 
    last_keyup = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except: return [], []

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

# ==============================================================================
# 2. FEATURE EXTRACTION
# ==============================================================================
def extract_features(window_data, reference_pool):
    if len(window_data) < WINDOW_SIZE: return None
    if np.isnan(window_data).any(): return None

    # Stats
    feat_mean = np.mean(window_data)
    feat_med = np.median(window_data)
    feat_std = np.std(window_data)
    
    if feat_std < 0.0001:
        feat_skew, feat_kurt = 0, 10
    else:
        feat_skew = stats.skew(window_data)
        feat_kurt = stats.kurtosis(window_data)

    # Distances (Самая тяжелая часть)
    ks_scores, w_scores = [], []
    for ref_win in reference_pool:
        ks, _ = stats.ks_2samp(window_data, ref_win)
        ks_scores.append(ks)
        w_scores.append(stats.wasserstein_distance(window_data, ref_win))
    
    return [feat_mean, feat_med, feat_std, feat_skew, feat_kurt, np.min(ks_scores), np.min(w_scores)]

# ==============================================================================
# 3. ФУНКЦИЯ-ВОРКЕР (Обрабатывает один файл целиком)
# ==============================================================================
def process_single_file(filepath, label, step_size, ref_dwells, ref_flights):
    """
    Эта функция будет запускаться на отдельном ядре процессора.
    Она берет файл, парсит его и возвращает СПИСОК готовых строк для датасета.
    """
    rows = []
    d, f = parse_file(filepath)
    min_len = min(len(d), len(f))
    
    if min_len < WINDOW_SIZE:
        return []

    # Скользящее окно
    for i in range(0, min_len - WINDOW_SIZE, step_size):
        w_d = d[i : i + WINDOW_SIZE]
        w_f = f[i : i + WINDOW_SIZE]
        
        ft_d = extract_features(w_d, ref_dwells)
        ft_f = extract_features(w_f, ref_flights)
        
        if ft_d and ft_f:
            rows.append(ft_f + ft_d + [label])
            
    return rows

# ==============================================================================
# 4. СБОР ЭТАЛОНОВ (В один поток, это быстро)
# ==============================================================================
def create_reference_pool():
    print("--- Сбор эталонов (Reference Pool) ---")
    all_human_dwells = []
    all_human_flights = []
    
    # Берем первые 50 файлов для скорости генерации эталонов
    human_files = []
    for folder in FOLDERS["Humans"]:
        human_files.extend(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    
    # Чтобы эталоны были разнообразными, перемешаем файлы перед выборкой
    random.shuffle(human_files)

    for f in human_files[:50]: 
        d, fl = parse_file(f)
        if len(d) >= WINDOW_SIZE: all_human_dwells.extend(d)
        if len(fl) >= WINDOW_SIZE: all_human_flights.extend(fl)

    ref_dwells, ref_flights = [], []
    
    if len(all_human_dwells) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_dwells) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_dwells.append(np.array(all_human_dwells[start:start+WINDOW_SIZE]))
    else:
        print("!!! ОШИБКА: Мало данных Dwell")

    if len(all_human_flights) > WINDOW_SIZE:
        for _ in range(NUM_REFERENCES):
            max_start = len(all_human_flights) - WINDOW_SIZE
            start = random.randint(0, max_start)
            ref_flights.append(np.array(all_human_flights[start:start+WINDOW_SIZE]))
    else:
         print("!!! ОШИБКА: Мало данных Flight")
            
    return ref_dwells, ref_flights

# ==============================================================================
# 5. ГЕНЕРАТОР БОТОВ (Исправленный)
# ==============================================================================
def generate_synthetic_bots(n_samples):
    synthetic_rows = [] # Тут сразу будем хранить окна (dwell, flight)
    print(f"--- Генерация {n_samples} синтетических атак ---")
    
    attack_types = ['machine_gun', 'gaussian', 'uniform', 'jitter']
    
    for _ in range(n_samples):
        attack_type = random.choice(attack_types)
        win_dwell, win_flight = None, None

        if attack_type == 'machine_gun':
            val = random.randint(4, 10)
            win_dwell = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)
            win_flight = np.full(WINDOW_SIZE, val) + np.random.normal(0, 0.1, WINDOW_SIZE)
        elif attack_type == 'gaussian':
            mean, std = random.randint(80, 150), random.randint(10, 40)
            win_dwell = np.random.normal(mean, std, WINDOW_SIZE)
            win_flight = np.random.normal(mean, std, WINDOW_SIZE)
        elif attack_type == 'uniform':
            low, high = random.randint(10, 50), random.randint(60, 200)
            win_dwell = np.random.uniform(low, high, WINDOW_SIZE)
            win_flight = np.random.uniform(low, high, WINDOW_SIZE)
        elif attack_type == 'jitter':
            win_dwell = np.random.randint(5, 300, WINDOW_SIZE).astype(float)
            win_flight = np.random.randint(5, 300, WINDOW_SIZE).astype(float)

        if win_dwell is not None:
            synthetic_rows.append((np.abs(win_dwell), np.abs(win_flight)))
            
    return synthetic_rows

# ==============================================================================
# 6. MAIN (С ПАРАЛЛЕЛЬНОЙ ОБРАБОТКОЙ)
# ==============================================================================
def main():
    # 1. Эталоны
    ref_dwells, ref_flights = create_reference_pool()
    if not ref_dwells: return
    np.savez(OUTPUT_REFS, dwell=ref_dwells, flight=ref_flights)

    dataset_rows = []
    
    # 2. Собираем список задач (Tasks) для процессоров
    tasks = []
    
    for label_name, folders in FOLDERS.items():
        label = 0 if label_name == "Humans" else 1
        step = STEP_SIZE_BOT if label == 1 else STEP_SIZE_HUMAN
        
        all_files = []
        for folder in folders:
            all_files.extend(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
        
        print(f"Найден класс {label_name}: {len(all_files)} файлов. Подготовка задач...")
        
        # Создаем задачу: (путь, метка, шаг, эталоны...)
        for f in all_files:
            tasks.append((f, label, step, ref_dwells, ref_flights))

    # 3. ЗАПУСК PARALLEL PROCESSING
    # Используем столько воркеров, сколько ядер у CPU (минус 1, чтобы система дышала)
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"\n🚀 Запуск обработки на {max_workers} ядрах. Пожалуйста, подождите...")

    # ProcessPoolExecutor магия
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем задачи
        # future_to_file хранит состояние задачи
        futures = [executor.submit(process_single_file, *t) for t in tasks]
        
        # tqdm показывает прогресс бар по мере завершения файлов
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Обработка файлов"):
            try:
                result_rows = future.result()
                if result_rows:
                    dataset_rows.extend(result_rows)
            except Exception as e:
                print(f"Ошибка в процессе: {e}")

    # 4. СИНТЕТИКА (Быстро, в один поток)
    synth_data = generate_synthetic_bots(SYNTHETIC_SAMPLES)
    print("Расчет признаков для синтетических ботов...")
    for w_d, w_f in tqdm(synth_data):
        ft_d = extract_features(w_d, ref_dwells)
        ft_f = extract_features(w_f, ref_flights)
        if ft_d and ft_f:
            dataset_rows.append(ft_f + ft_d + [1]) # Label 1

    # 5. ЭКСПОРТ
    print("Сохранение CSV...")
    cols = ["F_Mean", "F_Med", "F_Std", "F_Skew", "F_Kurt", "F_MinKS", "F_MinW",
            "D_Mean", "D_Med", "D_Std", "D_Skew", "D_Kurt", "D_MinKS", "D_MinW", "Label"]
    
    df = pd.DataFrame(dataset_rows, columns=cols)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Готово! Файл: {OUTPUT_CSV}")
    print("Размер датасета:", len(df))
    print(df["Label"].value_counts())

if __name__ == "__main__":
    # В Windows для мультипроцессинга обязателен этот блок
    multiprocessing.freeze_support()
    main()