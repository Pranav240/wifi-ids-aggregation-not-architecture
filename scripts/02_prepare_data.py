"""
Stage 2 — Streaming, session-aware data preparation (resubmission rewrite).

Run 01_select_features.py FIRST — this script loads selected_features.pkl.

WHAT THIS DOES
1. Streams through each attack folder's raw CSV files, one at a time,
   reading only until that folder's row target is hit (avoids loading
   the full ~44GB dataset into memory).
2. Groups packets into sessions using wlan.sa (source MAC), falling back
   to wlan.ta / wlan.bssid when wlan.sa is missing — this is the real
   session grouping the earlier label-block approach was a proxy for.
3. Sorts each session by frame.time_relative, builds NON-OVERLAPPING
   windows of WINDOW_SIZE consecutive packets within each session only.
4. Sanitizes inf/NaN BEFORE any float32 cast (fixes the earlier overflow
   crash).
5. Applies capped stratified sampling per class (not full equalization).
6. Splits by window into train/val/test, scales using train only.
7. Saves three formats: windowed (Transformer), flattened (ablation
   baselines), single-packet (primary baseline comparison).
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================== CONFIG ==============================
RAW_DIR = Path(r"D:\awid3_attacks")   # adjust to your actual path
OUTPUT_DIR = Path("data/prepared")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "Label"
TIME_COL = "frame.time_relative"
SESSION_COLS_PRIORITY = ["wlan.sa", "wlan.ta", "wlan.bssid"]  # first non-null used

WINDOW_SIZE = 10
STEP = WINDOW_SIZE  # non-overlapping

ATTACK_ROWS_TARGET_PER_FOLDER = 20_000   # aim high; scarce folders (e.g.
                                          # Rogue_AP) will contribute less —
                                          # watch console for the "real
                                          # ceiling" notes
NORMAL_ROWS_TARGET_PER_FOLDER = 5_000    # background Normal rows per folder;
                                          # Normal windows accumulate across
                                          # all 13 folders combined
READ_CHUNK_ROWS = 20_000            # per-file read chunk while scanning
MAX_FILES_TO_SCAN = 200             # generous; early-exit once target hit

CAP_MULTIPLIER = 8
MIN_SAMPLES_PER_CLASS = 200

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# ======================================================================


def load_selected_features():
    with open(OUTPUT_DIR / "selected_features.pkl", "rb") as f:
        return pickle.load(f)


def canonical_label_from_folder(folder_name):
    """Strip a leading 'N.' index prefix from a folder name to get the
    canonical attack label, e.g. '6.Kr00k' -> 'Kr00k'. Used to override
    inconsistent raw label spelling/casing within a folder's own files
    (observed: 'Kr00k' vs 'Kr00K', 'Rogue_AP' folder vs 'RogueAP' label)."""
    return re.sub(r"^\d+\.", "", folder_name)


def stream_folder(folder_path, feature_cols, attack_target, normal_target, folder_name):
    """Scan CSV files in a folder, actively collecting non-Normal (attack)
    rows up to attack_target, plus a modest Normal background pool, instead
    of blindly reading the first N rows (which are ~100% Normal, since
    attacks are injected partway through each file). Scans as many files
    as needed (up to MAX_FILES_TO_SCAN) to find attack rows; reports the
    true per-folder ceiling when a folder's entire content is exhausted
    before hitting the target."""
    needed_cols = list(set(feature_cols) | set(SESSION_COLS_PRIORITY) |
                        {TIME_COL, LABEL_COL})
    canonical_attack_label = canonical_label_from_folder(folder_name)

    csvs = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    attack_chunks, normal_chunks = [], []
    attack_rows_found, normal_rows_found = 0, 0
    files_scanned = 0

    for fname in csvs[:MAX_FILES_TO_SCAN]:
        if attack_rows_found >= attack_target:
            break
        fpath = folder_path / fname
        files_scanned += 1
        if files_scanned % 10 == 0:
            print(f"    ...scanned {files_scanned} files, "
                  f"{attack_rows_found} attack rows found so far")

        try:
            header = pd.read_csv(fpath, nrows=0).columns.tolist()
            usecols = [c for c in needed_cols if c in header]
        except Exception as e:
            print(f"    skipping {fpath}: {e}")
            continue

        try:
            reader = pd.read_csv(fpath, usecols=usecols, chunksize=READ_CHUNK_ROWS,
                                  low_memory=False)
        except Exception as e:
            print(f"    skipping {fpath}: {e}")
            continue

        for chunk in reader:
            if LABEL_COL not in chunk.columns:
                break
            for c in feature_cols:
                if c in chunk.columns:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
                else:
                    chunk[c] = 0.0

            chunk = chunk.dropna(subset=[LABEL_COL])
            chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.strip()

            # case-insensitive Normal detection — raw data has shown
            # casing/spelling drift (e.g. 'Kr00k' vs 'Kr00K', 'RogueAP' vs
            # 'Rogue_AP'); don't trust exact string match for either side
            is_attack = chunk[LABEL_COL].str.lower() != "normal"
            attack_part = chunk[is_attack].copy()
            if len(attack_part) > 0:
                # override with the canonical, folder-derived label instead
                # of trusting the raw file's spelling/casing for this row
                attack_part[LABEL_COL] = canonical_attack_label
                attack_chunks.append(attack_part)
                attack_rows_found += len(attack_part)

            if normal_rows_found < normal_target:
                normal_part = chunk[~is_attack].copy()
                take = min(len(normal_part), normal_target - normal_rows_found)
                if take > 0:
                    normal_part = normal_part.iloc[:take]
                    normal_part[LABEL_COL] = "Normal"  # canonicalize here too
                    normal_chunks.append(normal_part)
                    normal_rows_found += take

            if attack_rows_found >= attack_target:
                break

    attack_df = (pd.concat(attack_chunks, axis=0, ignore_index=True).iloc[:attack_target]
                 if attack_chunks else None)
    normal_df = (pd.concat(normal_chunks, axis=0, ignore_index=True)
                 if normal_chunks else None)

    n_attack = len(attack_df) if attack_df is not None else 0
    n_normal = len(normal_df) if normal_df is not None else 0
    print(f"  scanned {files_scanned}/{len(csvs)} file(s) -> "
          f"{n_attack} attack rows + {n_normal} normal rows")
    if n_attack < attack_target and files_scanned >= len(csvs):
        print(f"    NOTE: entire folder scanned ({len(csvs)} files) and only "
              f"{n_attack} attack rows exist in total — this is the "
              f"real ceiling for this class, not a scan-depth limit.")

    pieces = [d for d in (attack_df, normal_df) if d is not None and len(d) > 0]
    if not pieces:
        return pd.DataFrame(columns=needed_cols)
    combined = pd.concat(pieces, axis=0, ignore_index=True)
    return combined


def get_session_key(df):
    key = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col in SESSION_COLS_PRIORITY:
        if col in df.columns:
            key = key.fillna(df[col])
    key = key.fillna("UNKNOWN_SESSION")
    return key


def build_windows_for_folder(df, feature_cols):
    if len(df) == 0:
        return None, None, None

    df = df.copy()
    df["_session"] = get_session_key(df)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").fillna(0)
        df = df.sort_values(["_session", TIME_COL])
    else:
        df = df.sort_values(["_session"])

    windows = []
    labels = []
    for _, sess_df in df.groupby("_session", sort=False):
        feats = sess_df[feature_cols].values
        # sanitize inf/-inf BEFORE any float32 cast
        feats = np.where(np.isinf(feats), np.nan, feats)
        n_rows = len(feats)
        n_windows = n_rows // WINDOW_SIZE
        if n_windows == 0:
            continue
        usable = n_windows * WINDOW_SIZE
        feats = feats[:usable].reshape(n_windows, WINDOW_SIZE, len(feature_cols))
        windows.append(feats)

        sess_labels = sess_df[LABEL_COL].values[:usable].reshape(n_windows, WINDOW_SIZE)
        # a session window may straddle a normal->attack transition; use
        # the majority label within the window, ties broken by last packet
        for row in sess_labels:
            vals, counts = np.unique(row, return_counts=True)
            majority = vals[np.argmax(counts)]
            labels.append(majority)

    if not windows:
        return None, None, None

    X = np.concatenate(windows, axis=0)
    # impute remaining NaN (from inf-sanitization or missing columns) with
    # per-feature median computed on this folder's data — final imputation
    # happens again properly at the scaling stage using train-only stats;
    # this is just to keep intermediate arrays finite. Columns that are
    # entirely NaN in this folder (e.g. a feature never populated for this
    # attack type) fall back to 0 explicitly, rather than letting
    # nanmedian warn and return NaN for them.
    flat = X.reshape(-1, X.shape[-1])
    all_nan_cols = np.all(np.isnan(flat), axis=0)
    col_median = np.zeros(flat.shape[1], dtype=np.float64)
    if not all_nan_cols.all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid_medians = np.nanmedian(flat[:, ~all_nan_cols], axis=0)
        col_median[~all_nan_cols] = valid_medians
    # all_nan_cols entries stay at 0.0 by initialization
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_median, inds[2])

    y = np.array(labels)
    return X.astype(np.float32), y, feature_cols


def process_all_folders(feature_cols):
    all_X, all_y = [], []
    folders = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(RAW_DIR / d)])

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        folder_df = stream_folder(RAW_DIR / folder, feature_cols,
                                   ATTACK_ROWS_TARGET_PER_FOLDER,
                                   NORMAL_ROWS_TARGET_PER_FOLDER,
                                   folder)
        X, y, _ = build_windows_for_folder(folder_df, feature_cols)
        if X is None:
            print(f"  no usable windows from {folder}")
            continue
        print(f"  -> {X.shape[0]} windows built")
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\nTotal windows across all folders: {X_all.shape[0]:,}")
    print("Label distribution (windows, pre-sampling):")
    print(pd.Series(y_all).value_counts())
    return X_all, y_all


def stratified_cap_sample(X, y, cap_multiplier, min_per_class, seed):
    rng = np.random.RandomState(seed)
    counts = pd.Series(y).value_counts()
    smallest = counts.min()
    cap = max(smallest * cap_multiplier, min_per_class)
    print(f"\nSmallest class: {smallest} windows | cap per class: {cap:.0f}")

    keep_idx = []
    for label, count in counts.items():
        idx = np.where(y == label)[0]
        if count < min_per_class:
            print(f"  WARNING: '{label}' has only {count} windows (< {min_per_class})")
            keep_idx.append(idx)
        elif count > cap:
            keep_idx.append(rng.choice(idx, size=int(cap), replace=False))
        else:
            keep_idx.append(idx)

    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    print("\nFinal sampled distribution:")
    print(pd.Series(y[keep_idx]).value_counts())
    return X[keep_idx], y[keep_idx]


def split_and_save(X_windowed, y, feature_cols):
    n_feat = len(feature_cols)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    idx = np.arange(len(y_enc))
    idx_train, idx_temp = train_test_split(
        idx, test_size=(1 - TRAIN_FRAC), random_state=RANDOM_SEED, stratify=y_enc
    )
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1 - rel_val), random_state=RANDOM_SEED,
        stratify=y_enc[idx_temp]
    )
    print(f"\nSplit -> train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)}")

    scaler = StandardScaler()
    scaler.fit(X_windowed[idx_train].reshape(-1, n_feat))

    def scale(X_win):
        shape = X_win.shape
        flat = X_win.reshape(-1, n_feat)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        return scaler.transform(flat).reshape(shape).astype(np.float32)

    X_train_win = scale(X_windowed[idx_train])
    X_val_win = scale(X_windowed[idx_val])
    X_test_win = scale(X_windowed[idx_test])

    def flatten(X_win):
        return X_win.reshape(X_win.shape[0], -1)

    def unwindow(X_win, y_win):
        n, w, f = X_win.shape
        return X_win.reshape(n * w, f), np.repeat(y_win, w)

    X_train_single, y_train_single = unwindow(X_train_win, y_enc[idx_train])
    X_val_single, y_val_single = unwindow(X_val_win, y_enc[idx_val])
    X_test_single, y_test_single = unwindow(X_test_win, y_enc[idx_test])

    np.savez_compressed(
        OUTPUT_DIR / "windowed_transformer.npz",
        X_train=X_train_win, y_train=y_enc[idx_train],
        X_val=X_val_win, y_val=y_enc[idx_val],
        X_test=X_test_win, y_test=y_enc[idx_test],
    )
    np.savez_compressed(
        OUTPUT_DIR / "flattened_ablation.npz",
        X_train=flatten(X_train_win), y_train=y_enc[idx_train],
        X_val=flatten(X_val_win), y_val=y_enc[idx_val],
        X_test=flatten(X_test_win), y_test=y_enc[idx_test],
    )
    np.savez_compressed(
        OUTPUT_DIR / "single_packet_baseline.npz",
        X_train=X_train_single, y_train=y_train_single,
        X_val=X_val_single, y_val=y_val_single,
        X_test=X_test_single, y_test=y_test_single,
    )

    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    info = {
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "n_features": n_feat,
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "attack_rows_target_per_folder": ATTACK_ROWS_TARGET_PER_FOLDER,
        "normal_rows_target_per_folder": NORMAL_ROWS_TARGET_PER_FOLDER,
        "cap_multiplier": CAP_MULTIPLIER,
        "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        "train_windows": int(len(idx_train)),
        "val_windows": int(len(idx_val)),
        "test_windows": int(len(idx_test)),
        "session_grouping": "wlan.sa (fallback wlan.ta / wlan.bssid), "
                             "sorted by frame.time_relative within session, "
                             "non-overlapping windows of size 10",
        "window_label_rule": "majority label within window (ties -> first "
                              "in np.unique order); sessions may straddle "
                              "normal->attack transitions",
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/: windowed_transformer.npz, "
          f"flattened_ablation.npz, single_packet_baseline.npz, "
          f"scaler.pkl, label_encoder.pkl, dataset_info.json")


def main():
    feature_cols = load_selected_features()
    print(f"Using {len(feature_cols)} selected features")

    X_all, y_all = process_all_folders(feature_cols)
    X_all, y_all = stratified_cap_sample(
        X_all, y_all, CAP_MULTIPLIER, MIN_SAMPLES_PER_CLASS, RANDOM_SEED
    )
    split_and_save(X_all, y_all, feature_cols)


if __name__ == "__main__":
    main()
