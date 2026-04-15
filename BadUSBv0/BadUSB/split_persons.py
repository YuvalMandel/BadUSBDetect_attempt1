"""
Person-disjoint train / val / test split generator.

Assigns participants (identified by first 3 chars of s2 filename) to
train / val / test at PERSON level, then randomly splits bot files.
All three models (MLP, GRU, HTM) load data_split.json so they always
use identical splits.

Run from BadUSBv0/BadUSB/:
    python split_persons.py [--bots-dir dataset_generator/Synthetic_Bots]
"""
import os, glob, json, random, argparse
from collections import defaultdict

RANDOM_SEED = 42
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.15
# test gets the remainder (~5%)

S2_DIRS  = ["dataset_generator/s2/baseline", "dataset_generator/s2/rotation"]
OUTPUT   = "data_split.json"


def person_id(path):
    """First 3 characters of the filename are the participant ID."""
    return os.path.basename(path)[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots-dir", default="dataset_generator/Synthetic_Bots",
                        help="Path to bot files (relative to this script)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    here = os.path.dirname(os.path.abspath(__file__))

    # ── Collect human files grouped by person ────────────────────────────────
    all_human = []
    for d in S2_DIRS:
        full = os.path.join(here, d)
        all_human.extend(sorted(glob.glob(os.path.join(full, "*.txt"))))

    if not all_human:
        raise FileNotFoundError(f"No .txt files found in {S2_DIRS}. "
                                "Run from BadUSBv0/BadUSB/ with s2/ present.")

    by_person = defaultdict(list)
    for f in all_human:
        by_person[person_id(f)].append(f)

    persons = sorted(by_person)
    random.shuffle(persons)

    n        = len(persons)
    n_train  = round(n * TRAIN_FRAC)
    n_val    = round(n * VAL_FRAC)
    # n_test = n - n_train - n_val

    train_p = persons[:n_train]
    val_p   = persons[n_train : n_train + n_val]
    test_p  = persons[n_train + n_val :]

    train_h = [f for p in train_p for f in by_person[p]]
    val_h   = [f for p in val_p   for f in by_person[p]]
    test_h  = [f for p in test_p  for f in by_person[p]]

    # ── Collect bot files ─────────────────────────────────────────────────────
    bots_full = os.path.join(here, args.bots_dir)
    all_bots  = sorted(glob.glob(os.path.join(bots_full, "*.txt")))

    if not all_bots:
        print(f"WARNING: No bot files found in {bots_full}. "
              "Run `make generate_dataset` first.")

    random.shuffle(all_bots)
    nb       = len(all_bots)
    nb_train = round(nb * TRAIN_FRAC)
    nb_val   = round(nb * VAL_FRAC)

    train_b = all_bots[:nb_train]
    val_b   = all_bots[nb_train : nb_train + nb_val]
    test_b  = all_bots[nb_train + nb_val :]

    # ── Save manifest ─────────────────────────────────────────────────────────
    split = {
        "train": {"humans": train_h, "bots": train_b},
        "val":   {"humans": val_h,   "bots": val_b},
        "test":  {"humans": test_h,  "bots": test_b},
    }

    out_path = os.path.join(here, OUTPUT)
    with open(out_path, "w") as fp:
        json.dump(split, fp, indent=2)

    print(f"Persons : {n} total  →  train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")
    print(f"Human files : {len(train_h)} / {len(val_h)} / {len(test_h)}")
    print(f"Bot files   : {len(train_b)} / {len(val_b)} / {len(test_b)}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
