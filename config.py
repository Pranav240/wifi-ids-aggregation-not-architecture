"""
Central configuration for the Wi-Fi IDS pipeline.

All paths, seeds, and shared hyperparameters live here so that:
  1. There is exactly ONE place to change the raw-data location.
  2. Noise parameters cannot silently drift apart between the script that
     BUILDS the noisy test set (02b) and the scripts that apply train-time
     augmentation (03/04/05/06/08/09/10) - a mismatch there would
     reintroduce the train/test noise leakage this project exists to fix.
  3. The exact configuration used for a run can be dumped alongside
     results for reproducibility.

Environment overrides are supported so the pipeline can be run on a
different machine without editing this file:

    # PowerShell
    $env:WIFI_IDS_RAW_DIR = "E:\\awid3_attacks"
    python scripts\01_select_features.py

    # bash
    export WIFI_IDS_RAW_DIR=/data/awid3_attacks
    python scripts/01_select_features.py
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
# Raw AWID3 capture root: a directory of per-attack-category folders,
# each containing multiple CSV files.
RAW_DIR = Path(os.environ.get("WIFI_IDS_RAW_DIR", r"D:\awid3_attacks"))

PROJECT_ROOT = Path(os.environ.get("WIFI_IDS_PROJECT_ROOT", ".")).resolve()

DATA_DIR = PROJECT_ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared"
RESULTS_DIR = DATA_DIR / "results"

BASELINES_DIR = RESULTS_DIR / "baselines"
TRANSFORMER_DIR = RESULTS_DIR / "transformer"
FLATTENED_DIR = RESULTS_DIR / "flattened_ablation"
MULTISEED_DIR = RESULTS_DIR / "multiseed"
SHUFFLE_DIR = RESULTS_DIR / "shuffle_test"
MEANPOOL_DIR = RESULTS_DIR / "mean_pooling"
SPLITVAR_DIR = RESULTS_DIR / "split_variance"

FIGURES_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------
# DATA SCHEMA
# ---------------------------------------------------------------------
LABEL_COL = "Label"
TIME_COL = "frame.time_relative"
SESSION_COLS_PRIORITY = ["wlan.sa", "wlan.ta", "wlan.bssid"]
NORMAL_LABEL = "Normal"

# Identifier / grouping columns - never used as model features.
ID_LIKE_COLS = {
    "frame.time", "frame.time_epoch", "frame.time_delta_displayed",
    "frame.number", "frame.time_relative",
    "wlan.sa", "wlan.da", "wlan.bssid", "wlan.ta", "wlan.ra", "wlan.ssid",
    "wlan.country_info.code", "ip.src", "ip.dst",
    LABEL_COL,
}

# Capture-session artefacts: hardware clocks and fixed radio channel /
# frequency fields. These can fingerprint WHICH capture a row came from
# rather than encoding attack behaviour, so they are excluded outright.
LEAKY_SESSION_COLS = {
    "radiotap.mactime", "radiotap.timestamp.ts",
    "wlan.fixed.timestamp", "wlan_radio.timestamp",
    "wlan_radio.start_tsf", "wlan_radio.end_tsf",
    "wlan_radio.channel", "wlan_radio.frequency", "radiotap.channel.freq",
}

EXCLUDED_COLS = ID_LIKE_COLS | LEAKY_SESSION_COLS

# ---------------------------------------------------------------------
# WINDOWING & SAMPLING
# ---------------------------------------------------------------------
WINDOW_SIZE = 10
STEP = WINDOW_SIZE          # non-overlapping

ATTACK_ROWS_TARGET_PER_FOLDER = 20_000
NORMAL_ROWS_TARGET_PER_FOLDER = 5_000
READ_CHUNK_ROWS = 20_000
MAX_FILES_TO_SCAN = 200

CAP_MULTIPLIER = 8          # majority class capped at 8x smallest
MIN_SAMPLES_PER_CLASS = 200

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# ---------------------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------------------
SAMPLE_ROWS_PER_FOLDER = 3000
ATTACK_ROWS_TARGET = 1500
MISSING_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 1e-6
CORR_THRESHOLD = 0.95
CUM_IMPORTANCE_TARGET = 0.95

# ---------------------------------------------------------------------
# NOISE PROTOCOL
# ---------------------------------------------------------------------
# Train-time augmentation and test-time evaluation noise MUST stay
# distinct. If these ever become equal, noisy-test accuracy stops
# measuring generalisation and starts measuring memorisation of the
# augmentation process - the exact defect this pipeline was rebuilt to
# eliminate. validate_config() enforces the separation.
TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08
TRAIN_NOISE_SEED = 7

TEST_NOISE_LEVEL = 0.25
TEST_MISSING_RATE = 0.15
TEST_SIGN_FLIP_RATE = 0.03      # not present in train-time augmentation
TEST_NOISE_SEED = 123

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------
RANDOM_SEED = 42
MODEL_SEEDS = [42, 123, 2024]
SPLIT_SEEDS = [42, 7, 2718]

EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5

N_OPTUNA_TRIALS = 30
OPTUNA_SEARCH_EPOCHS = 10
OPTUNA_SEARCH_PATIENCE = 3
FINAL_EPOCHS = 40
FINAL_PATIENCE = 6

DEFAULT_TRANSFORMER_PARAMS = {
    "num_layers": 2,
    "num_heads": 4,
    "d_model": 64,
    "ff_dim": 128,
    "dropout": 0.1,
    "learning_rate": 1e-3,
}

SHUFFLE_SEED = 9999
WINDOW_SIZES_SWEEP = [1, 2, 5, 10]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def ensure_dirs():
    """Create every output directory the pipeline writes to."""
    for d in (PREPARED_DIR, BASELINES_DIR, TRANSFORMER_DIR, FLATTENED_DIR,
               MULTISEED_DIR, SHUFFLE_DIR, MEANPOOL_DIR, SPLITVAR_DIR,
               FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate_config():
    """Fail loudly on configurations that would silently invalidate results."""
    errors = []

    if TEST_NOISE_LEVEL <= TRAIN_NOISE_LEVEL:
        errors.append(
            f"TEST_NOISE_LEVEL ({TEST_NOISE_LEVEL}) must exceed "
            f"TRAIN_NOISE_LEVEL ({TRAIN_NOISE_LEVEL}); otherwise the test set "
            f"is no harder than training augmentation and robustness results "
            f"are not meaningful.")
    if TEST_MISSING_RATE <= TRAIN_MISSING_RATE:
        errors.append(
            f"TEST_MISSING_RATE ({TEST_MISSING_RATE}) must exceed "
            f"TRAIN_MISSING_RATE ({TRAIN_MISSING_RATE}).")
    if TEST_SIGN_FLIP_RATE <= 0:
        errors.append(
            "TEST_SIGN_FLIP_RATE must be > 0 so that test-time corruption "
            "uses a mechanism absent from training augmentation.")
    if TRAIN_NOISE_SEED == TEST_NOISE_SEED:
        errors.append(
            "TRAIN_NOISE_SEED and TEST_NOISE_SEED must differ.")
    if not (0.99 < TRAIN_FRAC + VAL_FRAC + TEST_FRAC < 1.01):
        errors.append(
            f"TRAIN/VAL/TEST fractions sum to "
            f"{TRAIN_FRAC + VAL_FRAC + TEST_FRAC}, expected 1.0.")
    if STEP < WINDOW_SIZE:
        errors.append(
            f"STEP ({STEP}) < WINDOW_SIZE ({WINDOW_SIZE}) produces "
            f"overlapping windows, which leaks data across the train/test "
            f"split. Set STEP == WINDOW_SIZE unless you have added "
            f"split-aware overlap handling.")

    if errors:
        raise ValueError(
            "Invalid configuration:\n  - " + "\n  - ".join(errors))


def snapshot(path=None):
    """Dump the active configuration next to results, so any output can be
    traced back to the exact settings that produced it."""
    cfg = {
        k: (str(v) if isinstance(v, Path) else
            sorted(v) if isinstance(v, set) else v)
        for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }
    path = Path(path) if path else (PREPARED_DIR / "config_snapshot.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return path


def describe():
    print("=" * 60)
    print("WIFI-IDS PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  raw data        : {RAW_DIR}")
    print(f"  raw dir exists  : {RAW_DIR.exists()}")
    print(f"  project root    : {PROJECT_ROOT}")
    print(f"  window size     : {WINDOW_SIZE} (step {STEP}, non-overlapping)")
    print(f"  train noise     : level={TRAIN_NOISE_LEVEL} "
          f"missing={TRAIN_MISSING_RATE}")
    print(f"  test  noise     : level={TEST_NOISE_LEVEL} "
          f"missing={TEST_MISSING_RATE} sign_flip={TEST_SIGN_FLIP_RATE}")
    print(f"  model seeds     : {MODEL_SEEDS}")
    print(f"  split seeds     : {SPLIT_SEEDS}")
    print("=" * 60)


if __name__ == "__main__":
    describe()
    validate_config()
    print("Configuration valid.")
    if not RAW_DIR.exists():
        print(f"\nWARNING: raw data directory not found at {RAW_DIR}")
        print("Set WIFI_IDS_RAW_DIR or edit RAW_DIR in this file. "
              "Stages 01-02 need it; later stages read only from data/prepared.")
