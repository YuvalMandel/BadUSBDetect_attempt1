import pandas as pd
import os

# --- SETTINGS ---
# Only the TRAIN split is balanced; val/test keep their natural distribution
INPUT_TRAIN  = "train_dataset.csv"
OUTPUT_TRAIN = "train_final.csv"

def main():
    if not os.path.exists(INPUT_TRAIN):
        print(f"Error: File '{INPUT_TRAIN}' not found. Run dataset_processing_balanced.py first.")
        return

    print(f"Loading {INPUT_TRAIN}...")
    df = pd.read_csv(INPUT_TRAIN)

    print("\n--- ORIGINAL TRAIN STATISTICS ---")
    counts = df['Label'].value_counts()
    print(counts)

    n_bots   = counts.get(1, 0)
    n_humans = counts.get(0, 0)

    df_bots   = df[df['Label'] == 1]
    df_humans = df[df['Label'] == 0]

    # Balance: trim humans to match bot count so the classifier isn't biased
    target = n_bots
    if len(df_humans) > target:
        print(f"\nTrimming humans: {len(df_humans)} -> {target} (to match bot count)")
        df_humans_cut = df_humans.iloc[:target]
    else:
        print(f"\nHumans ({len(df_humans)}) ≤ bots ({n_bots}), keeping all humans.")
        df_humans_cut = df_humans

    final_df = pd.concat([df_bots, df_humans_cut])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(OUTPUT_TRAIN, index=False)

    print("\n--- BALANCED TRAIN STATISTICS ---")
    print(final_df['Label'].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_TRAIN}  ({len(final_df)} rows)")
    print("\nNote: val_dataset.csv and test_dataset.csv are left unbalanced for honest evaluation.")

if __name__ == "__main__":
    main()
