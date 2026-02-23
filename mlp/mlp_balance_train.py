"""
mlp/mlp_balance_train.py
Balance the MLP training set by trimming the majority class (humans)
to match the minority class (bots). Val and test sets are left unbalanced
for honest evaluation.

Run after mlp_prepare_data.py and before mlp_train.py:
  python mlp/mlp_balance_train.py

Reads:  train_dataset.csv
Writes: train_final.csv
"""

import os
import sys
import pandas as pd

INPUT_TRAIN  = "train_dataset.csv"
OUTPUT_TRAIN = "train_final.csv"


def main():
    if not os.path.exists(INPUT_TRAIN):
        print(f"Error: File '{INPUT_TRAIN}' not found. Run mlp/mlp_prepare_data.py first.")
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

    target = n_bots
    if len(df_humans) > target:
        print(f"\nTrimming humans: {len(df_humans)} -> {target} (to match bot count)")
        df_humans_cut = df_humans.iloc[:target]
    else:
        print(f"\nHumans ({len(df_humans)}) <= bots ({n_bots}), keeping all humans.")
        df_humans_cut = df_humans

    final_df = pd.concat([df_bots, df_humans_cut])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df.to_csv(OUTPUT_TRAIN, index=False)

    print("\n--- BALANCED TRAIN STATISTICS ---")
    print(final_df['Label'].value_counts().to_string())
    print(f"\nSaved: {OUTPUT_TRAIN}  ({len(final_df)} rows)")
    print("\nNote: val_dataset.csv and test_dataset.csv are left unbalanced for honest evaluation.")
    print("Next: run mlp/mlp_train.py")


if __name__ == "__main__":
    main()
