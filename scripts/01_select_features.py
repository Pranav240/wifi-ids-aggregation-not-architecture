"""
Stage 1 — Feature selection (resubmission rewrite).

Replaces the old undocumented "91 features" with a justified, reproducible
selection process run on a pooled sample from the raw AWID3 folders:

  1. Load a modest sample from each attack folder (fast, fits in RAM).
  2. Keep only numeric feature columns (drop identifiers/MAC/text fields —
     those are used for session grouping later, not as model inputs).
  3. Drop features with near-zero variance or excessive missingness.
  4. Rank remaining features by importance using a quick multiclass
     XGBoost model.
  5. Drop redundant features (correlation > 0.95 with a higher-importance
     feature).
  6. Keep the smallest feature set that covers ~95% cumulative importance.

Output: selected_features.pkl + a text report you can cite/paste into the
paper's methodology section (feature_selection_report.txt).
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ============================== CONFIG ==============================
RAW_DIR = Path(r"D:\awid3_attacks")   # adjust to your actual path
OUTPUT_DIR = Path("data/prepared")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_ROWS_PER_FOLDER = 3000     # total rows pulled per attack folder for this stage
ATTACK_ROWS_TARGET = 1500         # of which, aim for this many non-Normal rows
CHUNK_SIZE = 20_000                # rows read per chunk while scanning for attack rows
MAX_FILES_TO_SCAN = 200            # generous cap; early-exit stops as soon as
                                    # ATTACK_ROWS_TARGET is hit, so this only
                                    # costs time for folders where attacks
                                    # sit in a later file
LABEL_COL = "Label"

# Columns that are identifiers / grouping keys / raw timestamps — never
# used as model features, but some are needed later for session windowing.
ID_LIKE_COLS = {
    "frame.time", "frame.time_epoch", "frame.time_delta_displayed",
    "frame.number", "frame.time_relative",
    "wlan.sa", "wlan.da", "wlan.bssid", "wlan.ta", "wlan.ra", "wlan.ssid",
    "wlan.country_info.code", "ip.src", "ip.dst",
    LABEL_COL,
}

# Capture-session artifacts: hardware clocks and fixed radio-channel/
# frequency fields. These can trivially fingerprint WHICH capture session
# a row came from (each attack scenario was recorded as its own
# session/channel), rather than encoding anything about attack behavior.
# If left in, the model can appear to "detect" attacks by recognizing the
# capture file instead of the traffic pattern. Excluded from candidate
# features entirely, not just deprioritized.
LEAKY_SESSION_COLS = {
    "radiotap.mactime", "radiotap.timestamp.ts",
    "wlan.fixed.timestamp", "wlan_radio.timestamp",
    "wlan_radio.start_tsf", "wlan_radio.end_tsf",
    "wlan_radio.channel", "wlan_radio.frequency", "radiotap.channel.freq",
}
ID_LIKE_COLS = ID_LIKE_COLS | LEAKY_SESSION_COLS

MISSING_THRESHOLD = 0.95    # drop feature if >95% missing/zero-filled-NaN
VARIANCE_THRESHOLD = 1e-6   # drop near-constant features
CORR_THRESHOLD = 0.95       # drop redundant features above this correlation
CUM_IMPORTANCE_TARGET = 0.95
RANDOM_SEED = 42
# ======================================================================


def load_pooled_sample():
    frames = []
    folders = sorted([d for d in os.listdir(RAW_DIR)
                       if os.path.isdir(RAW_DIR / d)])
    for folder in folders:
        folder_path = RAW_DIR / folder
        csvs = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
        if not csvs:
            continue

        attack_chunks = []
        normal_chunks = []
        attack_rows_found = 0
        files_scanned = 0
        labels_seen = set()

        for fname in csvs[:MAX_FILES_TO_SCAN]:
            if attack_rows_found >= ATTACK_ROWS_TARGET:
                break
            fpath = folder_path / fname
            files_scanned += 1
            if files_scanned % 10 == 0:
                print(f"    ...scanned {files_scanned} files so far, "
                      f"{attack_rows_found} attack rows found")
            try:
                reader = pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False)
            except Exception as e:
                print(f"  skipping {fpath}: {e}")
                continue

            for chunk in reader:
                if LABEL_COL not in chunk.columns:
                    break
                is_attack = chunk[LABEL_COL].astype(str) != "Normal"
                labels_seen.update(chunk[LABEL_COL].astype(str).unique().tolist())
                attack_part = chunk[is_attack]
                if len(attack_part) > 0:
                    attack_chunks.append(attack_part)
                    attack_rows_found += len(attack_part)
                if len(normal_chunks) == 0 or sum(len(c) for c in normal_chunks) < 500:
                    normal_chunks.append(chunk[~is_attack].head(500))
                if attack_rows_found >= ATTACK_ROWS_TARGET:
                    break

        if attack_chunks:
            attack_df = pd.concat(attack_chunks, axis=0, ignore_index=True)
            attack_df = attack_df.head(ATTACK_ROWS_TARGET)
        else:
            attack_df = pd.DataFrame()

        normal_target = SAMPLE_ROWS_PER_FOLDER - len(attack_df)
        normal_df = (pd.concat(normal_chunks, axis=0, ignore_index=True).head(normal_target)
                     if normal_chunks else pd.DataFrame())

        combined = pd.concat([attack_df, normal_df], axis=0, ignore_index=True)
        frames.append(combined)
        print(f"  {folder}: scanned {files_scanned} file(s) -> "
              f"{len(attack_df)} attack rows + {len(normal_df)} normal rows "
              f"= {len(combined)} total")
        if len(attack_df) == 0:
            print(f"    WARNING: no non-Normal rows found for '{folder}' "
                  f"after scanning {files_scanned} file(s). "
                  f"Distinct label values actually seen: {labels_seen}. "
                  f"If that set is just {{'Normal'}}, this folder's attack "
                  f"traffic is likely concentrated in a file beyond "
                  f"MAX_FILES_TO_SCAN, or its label string doesn't match "
                  f"'Normal' as expected — verify manually.")

    pooled = pd.concat(frames, axis=0, ignore_index=True)
    print(f"\nPooled sample: {len(pooled):,} rows, {pooled.shape[1]} columns")
    print("Pooled label distribution:")
    print(pooled[LABEL_COL].astype(str).value_counts())

    # Catch stray label variants (whitespace, casing, encoding artifacts,
    # or actual missing values) that value_counts() can visually merge but
    # LabelEncoder would treat as distinct classes downstream. Built to be
    # robust regardless of the column's underlying dtype/mixed content.
    label_strings = pooled[LABEL_COL].map(
        lambda v: str(v) if pd.notna(v) else "__MISSING__"
    )
    unique_labels = sorted(label_strings.unique().tolist(), key=str)
    print(f"\nDistinct label strings ({len(unique_labels)}): {unique_labels}")

    stripped = label_strings.str.strip()
    if stripped.nunique() != label_strings.nunique():
        print(f"  WARNING: {label_strings.nunique()} raw label variants but "
              f"only {stripped.nunique()} after stripping whitespace — "
              f"labels will be auto-stripped below to avoid fragmenting a class.")
    if "__MISSING__" in unique_labels:
        n_missing = (label_strings == "__MISSING__").sum()
        print(f"  WARNING: {n_missing} rows had a missing/unparseable "
              f"Label value — these will be dropped.")
        pooled = pooled[label_strings != "__MISSING__"].copy()
        label_strings = label_strings[label_strings != "__MISSING__"]

    pooled[LABEL_COL] = label_strings.str.strip()

    return pooled


def coerce_numeric_features(df):
    feature_cols = [c for c in df.columns if c not in ID_LIKE_COLS]
    numeric_cols = []
    for c in feature_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        # keep only columns that are actually numeric-representable
        # (i.e. not almost entirely NaN just from failed coercion of text)
        non_null_after = coerced.notna().sum()
        non_null_before = df[c].notna().sum()
        if non_null_before > 0 and (non_null_after / max(non_null_before, 1)) > 0.5:
            df[c] = coerced
            numeric_cols.append(c)
    print(f"Numeric-coercible feature columns: {len(numeric_cols)} / {len(feature_cols)}")
    return df, numeric_cols


def filter_missing_and_variance(df, numeric_cols):
    kept = []
    for c in numeric_cols:
        col = df[c]
        missing_frac = col.isna().mean()
        # protocol-absent fields are typically 0/NaN — treat NaN as "absent"
        filled = col.fillna(0)
        var = filled.var()
        if missing_frac > MISSING_THRESHOLD:
            continue
        if var is None or var < VARIANCE_THRESHOLD:
            continue
        kept.append(c)
    print(f"Kept after missingness/variance filter: {len(kept)} / {len(numeric_cols)}")
    return kept


def rank_by_importance(df, feature_cols, label_col):
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    le = LabelEncoder()
    y_raw = df[label_col].astype(str)

    print(f"\n  Label distribution in pooled sample BEFORE encoding:")
    print(df[label_col].astype(str).value_counts())

    y = le.fit_transform(y_raw)
    print(f"  Unique classes found: {len(le.classes_)}")
    if len(le.classes_) < 2:
        raise ValueError(
            f"Only {len(le.classes_)} class(es) present in the pooled "
            f"sample — XGBoost cannot learn anything. The sample is "
            f"almost certainly dominated by a single label (likely "
            f"'Normal', since attacks are injected partway through each "
            f"raw file). Increase SAMPLE_ROWS_PER_FOLDER or sample from "
            f"later in each file / multiple files per folder instead of "
            f"just the first N rows of file 0."
        )

    print(f"  X shape: {X.shape} | any NaN: {X.isna().any().any()} | "
          f"any inf: {np.isinf(X.values).any()}")
    print(f"  X column variance (min/max): "
          f"{X.var().min():.6f} / {X.var().max():.6f}")

    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, random_state=RANDOM_SEED,
        eval_metric="mlogloss", tree_method="hist",
    )
    model.fit(X, y)
    raw_importances = model.feature_importances_
    n_nan = np.isnan(raw_importances).sum()
    if n_nan > 0:
        print(f"  NOTE: {n_nan} features had NaN importance "
              f"(never used in any tree split) — treated as 0")
    importances = pd.Series(np.nan_to_num(raw_importances, nan=0.0), index=feature_cols)
    importances = importances.sort_values(ascending=False)

    total = importances.sum()
    print(f"  Importance sum check: {total:.6f} (should be ~1.0)")
    print(f"  Top feature: {importances.index[0]} = {importances.iloc[0]:.4f}")
    if len(importances) > 1:
        print(f"  2nd feature: {importances.index[1]} = {importances.iloc[1]:.4f}")

    return importances


def prune_correlated(df, ranked_features):
    X = df[ranked_features].fillna(0).replace([np.inf, -np.inf], 0)
    corr = X.corr().abs()

    dropped = set()
    kept = []
    for feat in ranked_features:  # already sorted by importance, descending
        if feat in dropped:
            continue
        kept.append(feat)
        # drop anything highly correlated with this (higher-importance) feature
        correlated_with = corr.index[(corr[feat] > CORR_THRESHOLD) & (corr.index != feat)]
        for other in correlated_with:
            if other not in kept:
                dropped.add(other)

    print(f"Dropped {len(dropped)} redundant features (corr > {CORR_THRESHOLD})")
    return kept


def select_by_cumulative_importance(importances, kept_features):
    ranked_kept = importances.loc[kept_features].sort_values(ascending=False)
    total = ranked_kept.sum()
    if total <= 0 or np.isnan(total):
        raise ValueError(
            f"Cumulative importance sum is {total} — something is wrong "
            f"upstream (all-zero or NaN importances). Inspect the raw "
            f"XGBoost importances before proceeding; do not trust a "
            f"downstream feature count computed from this."
        )
    cum = ranked_kept.cumsum() / total

    top_share = ranked_kept.iloc[0] / total
    if top_share > 0.5:
        print(f"  WARNING: single top feature ('{ranked_kept.index[0]}') "
              f"holds {top_share*100:.1f}% of total importance on its own. "
              f"This is a red flag for leakage (e.g. a feature that "
              f"fingerprints the capture session rather than reflecting "
              f"attack behavior) — inspect this feature manually before "
              f"trusting the selection.")

    n_needed = (cum <= CUM_IMPORTANCE_TARGET).sum() + 1
    n_needed = min(n_needed, len(ranked_kept))
    final_features = ranked_kept.index[:n_needed].tolist()
    print(f"\nSelected {len(final_features)} features covering "
          f"~{CUM_IMPORTANCE_TARGET*100:.0f}% cumulative importance")
    return final_features, ranked_kept


def main():
    print("Loading pooled sample across attack folders...")
    pooled = load_pooled_sample()

    print("\nCoercing numeric feature columns...")
    pooled, numeric_cols = coerce_numeric_features(pooled)

    print("\nFiltering by missingness / variance...")
    survivors = filter_missing_and_variance(pooled, numeric_cols)

    print("\nRanking by XGBoost importance...")
    importances = rank_by_importance(pooled, survivors, LABEL_COL)

    print("\nPruning correlated features...")
    kept = prune_correlated(pooled, importances.index.tolist())

    final_features, ranked_kept = select_by_cumulative_importance(importances, kept)

    with open(OUTPUT_DIR / "selected_features.pkl", "wb") as f:
        pickle.dump(final_features, f)

    report_lines = [
        "Feature Selection Report",
        "=========================",
        f"Pooled sample size: {len(pooled):,} rows across "
        f"{len(os.listdir(RAW_DIR))} attack folders",
        f"Raw columns: {pooled.shape[1]}",
        f"Numeric-coercible: {len(numeric_cols)}",
        f"After missingness (>{MISSING_THRESHOLD*100:.0f}%) / "
        f"variance (<{VARIANCE_THRESHOLD}) filter: {len(survivors)}",
        f"After correlation pruning (>{CORR_THRESHOLD}): {len(kept)}",
        f"Final feature count (~{CUM_IMPORTANCE_TARGET*100:.0f}% cumulative "
        f"XGBoost importance): {len(final_features)}",
        "",
        "Top 20 features by importance:",
    ]
    for feat, imp in ranked_kept.head(20).items():
        report_lines.append(f"  {feat}: {imp:.6f}")

    report = "\n".join(report_lines)
    with open(OUTPUT_DIR / "feature_selection_report.txt", "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nSaved: {OUTPUT_DIR}/selected_features.pkl")
    print(f"Saved: {OUTPUT_DIR}/feature_selection_report.txt")


if __name__ == "__main__":
    main()
