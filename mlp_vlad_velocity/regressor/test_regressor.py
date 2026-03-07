import os
import glob
import numpy as np
import joblib

# --- КОНФИГУРАЦИЯ ---
HUMANS_TEST_FOLDER = "s2"
BOTS_TEST_FOLDER = "Synthetic_Bots_test" # Убедитесь, что папка называется именно так
MODEL_PATH = "poly_regressor.pkl"

KEYBOARD_MAP = {
    'q': (0, 3), 'w': (1, 3), 'e': (2, 3), 'r': (3, 3), 't': (4, 3),
    'y': (5, 3), 'u': (6, 3), 'i': (7, 3), 'o': (8, 3), 'p': (9, 3),
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'space': (4.5, 0)
}

def parse_and_extract_test(filepath):
    features = [] 
    active_keys = {} 
    last_keyup_ts = None
    last_keyup_name = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception: return []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        key, action = parts[0].lower(), parts[1]
        try: ts = int(parts[2])
        except ValueError: continue

        #scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts) #/ scale
                
                # При тестировании берем широкий диапазон
                if 0 < delta < 2000: 
                    if last_keyup_name in KEYBOARD_MAP and key in KEYBOARD_MAP:
                        x1, y1 = KEYBOARD_MAP[last_keyup_name]
                        x2, y2 = KEYBOARD_MAP[key]
                        features.append([x1, y1, x2, y2, delta])
                        
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys: active_keys.pop(key)
                
    return features

def evaluate_folder(folder_path, model, label_name):
    if not os.path.exists(folder_path):
        print(f"❌ ОШИБКА: Папка {folder_path} не найдена.")
        return

    files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
    if "s2" in folder_path:
        files = [f for f in files if f.endswith("1.txt")]
        
    np.random.shuffle(files)
    files = files[:20] 
    
    all_errors = []
    
    for f in files:
        data = parse_and_extract_test(f)
        if not data: continue
        
        data = np.array(data)
        X = data[:, :4]
        y_true = data[:, 4]
        
        y_pred = model.predict(X)
        
        # Вычисляем ошибку
        errors = np.abs(y_true - y_pred)
        all_errors.extend(errors)
        
    if not all_errors:
        print(f"{label_name}: Нет данных для анализа.")
        return
        
    all_errors = np.array(all_errors)
    mean_err = np.mean(all_errors)
    med_err = np.median(all_errors)
    std_err = np.std(all_errors)
    
    print(f"\n--- Результаты для {label_name} ---")
    print(f"Обработано переходов: {len(all_errors)}")
    print(f"Средняя ошибка (Mean):   {mean_err:.2f} ms")
    print(f"Медианная ошибка (Med):  {med_err:.2f} ms")
    print(f"Дисперсия ошибки (Std):  {std_err:.2f} ms")

def main():
    print("Загрузка полиномиальной модели (Ряд Тейлора)...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    evaluate_folder(HUMANS_TEST_FOLDER, model, "ЛЮДИ (s2 Test Set)")
    evaluate_folder(BOTS_TEST_FOLDER, model, "БОТЫ (Synthetic)")
    a = input("Press enter")

if __name__ == "__main__":
    main()
