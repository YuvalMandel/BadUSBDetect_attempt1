import os
import glob
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

# --- КОНФИГУРАЦИЯ ---
DATA_FOLDER_BASELINE = "s0/baseline"
DATA_FOLDER_ROTATION = "s0/rotation"
MODEL_SAVE_PATH = "poly_regressor.pkl"

# Степень полинома (разложение Тейлора). 
# 2 = 15 коэффициентов, 3 = 35 коэффициентов. 
# Начнем с 2, чтобы избежать переобучения (Overfitting).
DEGREE = 12#10 

KEYBOARD_MAP = {
    'q': (0, 3), 'w': (1, 3), 'e': (2, 3), 'r': (3, 3), 't': (4, 3),
    'y': (5, 3), 'u': (6, 3), 'i': (7, 3), 'o': (8, 3), 'p': (9, 3),
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2),
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2),
    'z': (1, 1), 'x': (2, 1), 'c': (3, 1), 'v': (4, 1), 'b': (5, 1),
    'n': (6, 1), 'm': (7, 1),
    'space': (4.5, 0)
}

def parse_and_extract_raw_coords(filepath):
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

      #  scale = 10000.0 if ts > 10**15 else 1.0

        if action == "KeyDown" or action == "keydown":
            active_keys[key] = ts
            if last_keyup_ts is not None and last_keyup_name is not None:
                delta = (ts - last_keyup_ts) #/ scale
                
                # Учим только чистую биомеханику!
                if 20 < delta < 400: 
                    if last_keyup_name in KEYBOARD_MAP and key in KEYBOARD_MAP:
                        x1, y1 = KEYBOARD_MAP[last_keyup_name]
                        x2, y2 = KEYBOARD_MAP[key]
                        # Подаем голые координаты, полином сам найдет связи
                        features.append([x1, y1, x2, y2, delta])
                        
        elif action == "KeyUp" or action == "keyup":
            last_keyup_ts = ts
            last_keyup_name = key
            if key in active_keys: active_keys.pop(key)
                
    return features

def main():
    print("--- 1. Сбор сырых координат для аппроксимации ---")
    
    if not os.path.exists(DATA_FOLDER_BASELINE) or not os.path.exists(DATA_FOLDER_ROTATION):
        print("❌ ОШИБКА: Папки с данными не найдены!")
        return

    base_files = glob.glob(os.path.join(DATA_FOLDER_BASELINE, "**", "*1.txt"), recursive=True)[:10]
    rot_files = glob.glob(os.path.join(DATA_FOLDER_ROTATION, "**", "*1.txt"), recursive=True)[:10]
    train_files = base_files + rot_files
    
    all_data = []
    for f in train_files:
        all_data.extend(parse_and_extract_raw_coords(f))
        
    all_data = np.array(all_data)
    print(f"Собрано {len(all_data)} переходов.")

    X = all_data[:, :4] # x1, y1, x2, y2
    y = all_data[:, 4]  # delta

    # Создаем Pipeline: сначала возводим в степени (Taylor), затем ищем коэффициенты (Ridge)
    # Ridge (L2 регуляризация) используется, чтобы коэффициенты не "взрывались"
    model = make_pipeline(PolynomialFeatures(degree=DEGREE), Ridge(alpha=1.0))

    print("\n--- 2. Аналитическое вычисление коэффициентов... ---")
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"✅ Уравнение найдено мгновенно! Средняя ошибка на обучающей выборке: {mae:.2f} мс")

    # Сохраняем математическую модель
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Модель сохранена в {MODEL_SAVE_PATH}")

    # (Опционально) Посмотреть, сколько коэффициентов получилось
    poly_layer = model.named_steps['polynomialfeatures']
    print(f"Количество членов в полиноме: {poly_layer.n_output_features_}")
    a = input("Press enter")

if __name__ == "__main__":
    main()
