"""
Person-disjoint train / val / test split generator.

Assigns participants (identified by first 3 chars of filename) to
train / val / test at PERSON level, then randomly splits bot files.
All three models (MLP, GRU, HTM) load data_split.json so they always
use identical splits.

IID design choices:
  - Multiple sessions (s0, s1, s2): rotation users get 3 different
    keyboards across sessions, giving a rich keyboard-type mix.
    Baseline users get 3 sessions of the same keyboard (more data).
  - Stratified by subset: baseline (001-075) and rotation (076-148)
    are each split 80/15/5% independently and then merged, so every
    split has the same keyboard-type ratio.
  - Only task 1 (free-text) files are used by default. Task 0 is a
    fixed transcription of the same Steve Jobs speech for everyone —
    its n-gram distribution is not representative of real typing.
    Pass --tasks 01 to include both.

Run from BadUSBv0/BadUSB/:
    python split_persons.py

On Newton (all sessions, pointing at UB dataset root):
    python split_persons.py \\
        --ub-dir /home/yuval.mandel/HWSecurity/UB_keystroke_dataset \\
        --sessions s0 s1 s2
"""
import os, glob, json, random, argparse
from collections import defaultdict

RANDOM_SEED    = 42
TRAIN_FRAC     = 0.80
VAL_FRAC       = 0.15
BASELINE_MAX_ID = "075"   # persons 001-075 are baseline; 076-148 are rotation

OUTPUT = "data_split.json"


def person_id(path):
    """First 3 characters of the filename are the participant ID."""
    return os.path.basename(path)[:3]


def task_id(path):
    """Last character before .txt is the task code (0=transcription, 1=free-text)."""
    return os.path.basename(path)[5]


def stratified_split(persons, frac_train, frac_val):
    """Split a list (already shuffled) into (train, val, test)."""
    random.shuffle(persons)
    n       = len(persons)
    n_train = round(n * frac_train)
    n_val   = round(n * frac_val)
    return persons[:n_train], persons[n_train:n_train + n_val], persons[n_train + n_val:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots-dir", default="dataset_generator/Synthetic_Bots",
                        help="Path to bot files (relative to this script)")
    parser.add_argument("--ub-dir", default=None,
                        help="Path to UB dataset root (contains s0/, s1/, s2/ subdirs). "
                             "Default: dataset_generator/ relative to this script.")
    parser.add_argument("--sessions", nargs="+", default=["s2"],
                        choices=["s0", "s1", "s2"],
                        help="Which sessions to include (default: s2). "
                             "Use 's0 s1 s2' for all sessions.")
    parser.add_argument("--tasks", default="1", choices=["0", "1", "01", "10"],
                        help="Task files to include: '1'=free-text only (default), "
                             "'0'=transcription only, '01'=both")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    here = os.path.dirname(os.path.abspath(__file__))

    # ── Resolve UB dataset root ───────────────────────────────────────────────
    if args.ub_dir:
        ub_root = os.path.abspath(args.ub_dir)
    else:
        ub_root = os.path.join(here, "dataset_generator")

    # ── Collect human files across sessions, filtered by task ─────────────────
    all_human = []
    allowed_tasks = set(args.tasks)

    for session in args.sessions:
        for subset in ("baseline", "rotation"):
            d = os.path.join(ub_root, session, subset)
            files = sorted(glob.glob(os.path.join(d, "*.txt")))
            files = [f for f in files if task_id(f) in allowed_tasks]
            all_human.extend(files)

    if not all_human:
        raise FileNotFoundError(
            f"No matching files found under {ub_root} "
            f"(sessions={args.sessions}, tasks={args.tasks}).\n"
            "On Newton, pass: --ub-dir ~/HWSecurity/UB_keystroke_dataset")

    # ── Group by person ───────────────────────────────────────────────────────
    by_person = defaultdict(list)
    for f in all_human:
        by_person[person_id(f)].append(f)

    # ── Stratified split: baseline and rotation separately ───────────────────
    baseline = sorted(p for p in by_person if p <= BASELINE_MAX_ID)
    rotation = sorted(p for p in by_person if p >  BASELINE_MAX_ID)

    b_train, b_val, b_test = stratified_split(baseline, TRAIN_FRAC, VAL_FRAC)
    r_train, r_val, r_test = stratified_split(rotation, TRAIN_FRAC, VAL_FRAC)

    train_p = b_train + r_train
    val_p   = b_val   + r_val
    test_p  = b_test  + r_test

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

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total = len(by_person)
    files_per_person = len(all_human) / n_total if n_total else 0
    print(f"Sessions     : {args.sessions}")
    print(f"Tasks        : {args.tasks}  ({'free-text only' if args.tasks=='1' else 'both' if '0' in args.tasks and '1' in args.tasks else 'transcription only'})")
    print(f"Persons      : {n_total}  (baseline={len(baseline)}, rotation={len(rotation)})")
    print(f"Files/person : {files_per_person:.1f}")
    print(f"  Train      : {len(train_p)} persons  →  {len(train_h)} files  (baseline={len(b_train)}, rotation={len(r_train)})")
    print(f"  Val        : {len(val_p)} persons  →  {len(val_h)} files  (baseline={len(b_val)}, rotation={len(r_val)})")
    print(f"  Test       : {len(test_p)} persons  →  {len(test_h)} files  (baseline={len(b_test)}, rotation={len(r_test)})")
    print(f"Bot files    : {len(train_b)} / {len(val_b)} / {len(test_b)}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
