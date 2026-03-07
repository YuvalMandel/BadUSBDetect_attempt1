import pandas as pd
import os

# --- НАСТРОЙКИ ---
INPUT_FILE = "balanced_dataset.csv"   # Тот файл, который получился после генерации (где 75к людей)
OUTPUT_FILE = "final_train_data.csv"  # Итоговый файл для нейросети
TARGET_HUMAN_COUNT = 3000             # Сколько людей оставить (чтобы было примерно поровну с ботами)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Ошибка: Файл {INPUT_FILE} не найден.")
        return

    print(f"Загрузка {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Смотрим, что было
    print("\n--- ИСХОДНАЯ СТАТИСТИКА ---")
    counts = df['Label'].value_counts()
    print(counts)
    
    n_bots = counts.get(1, 0)
    n_humans = counts.get(0, 0)

    # 2. Разделяем на два датафрейма
    df_bots = df[df['Label'] == 1]
    df_humans = df[df['Label'] == 0]

    # 3. Обрезаем людей
    # Берем первые N строк, как вы и просили
    if len(df_humans) > TARGET_HUMAN_COUNT:
        print(f"\nОбрезаем людей: было {len(df_humans)}, берем первые {TARGET_HUMAN_COUNT}...")
        df_humans_cut = df_humans.iloc[:TARGET_HUMAN_COUNT]
    else:
        print(f"\nЛюдей меньше, чем {TARGET_HUMAN_COUNT}, берем всех, что есть.")
        df_humans_cut = df_humans

    # 4. Собираем обратно
    # concat соединяет списки
    final_df = pd.concat([df_bots, df_humans_cut])

    # 5. ВАЖНО: ПЕРЕМЕШИВАЕМ (SHUFFLE)
    # frac=1 означает "взять 100% выборки", но в случайном порядке
    # reset_index сбрасывает старые индексы строк
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Сохраняем
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- ГОТОВО ---")
    print(f"Файл сохранен: {OUTPUT_FILE}")
    print("Финальная статистика:")
    print(final_df['Label'].value_counts())

if __name__ == "__main__":
    main()