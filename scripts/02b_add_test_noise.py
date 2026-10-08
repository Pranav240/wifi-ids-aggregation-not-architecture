"""
Stage 2b — Noisy evaluation set generation.

Run this AFTER 02_prepare_data.py.

Builds a NOISY TEST SET from the existing clean X_test, using different
noise parameters than what training-time augmentation will use (see
train_noise_augment() below, used inside training scripts). This is the
fix for the original review complaint: "unrealistic near-perfect results
require stronger validation" — traced back to train-time and test-time
noise previously being generated from the IDENTICAL distribution, so
models were evaluated on noise they'd effectively already seen.

Two distinct noise processes, deliberately different in both magnitude
and mechanism:

  TRAIN-TIME (used inside 03_train_baselines.py / transformer script,
  applied on-the-fly, not materialized here):
    - Gaussian noise, moderate level
    - Moderate missing-value rate (set to 0 = "missing")

  TEST-TIME (built here, once, saved to disk):
    - Gaussian noise at a HIGHER level than training
    - Missing-value rate at a DIFFERENT rate than training
    - Additionally: a small fraction of features get their sign flipped /
      clipped to simulate a structurally different corruption, not just
      "the same formula turned up" — this is what makes it a genuine
      generalization test rather than an interpolation test.

Produces X_test_noisy for both single_packet_baseline.npz and
windowed_transformer.npz (and flattened_ablation.npz, derived from the
same windowed noisy array).
"""

import numpy as np
import json
from pathlib import Path

# ============================== CONFIG ==============================
DATA_DIR = Path("data/prepared")

# Train-time augmentation noise (reference values — actually applied
# on-the-fly inside training scripts, documented here so the two
# processes are visibly different at a glance)
TRAIN_NOISE_LEVEL = 0.10        # gaussian std, as a fraction of each
                                 # feature's train-set std
TRAIN_MISSING_RATE = 0.08       # fraction of values zeroed out

# Test-time evaluation noise — deliberately different from training
TEST_NOISE_LEVEL = 0.25         # higher magnitude than training
TEST_MISSING_RATE = 0.15        # higher missing rate than training
TEST_SIGN_FLIP_RATE = 0.03      # structurally different corruption:
                                 # randomly negate a small fraction of
                                 # values, simulating measurement/parsing
                                 # errors rather than just "more of the
                                 # same" gaussian degradation

RANDOM_SEED = 123   # different seed than training-time augmentation
# ======================================================================


def add_test_noise(X, feature_std, rng):
    """Apply the test-time noise process to an array of shape (..., n_features).
    Works for both (N, F) single-packet and (N, W, F) windowed arrays."""
    X_noisy = X.copy()

    # 1. Gaussian noise scaled per-feature by that feature's train-set std
    gaussian = rng.normal(0, TEST_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian

    # 2. Missing values (zeroed out)
    missing_mask = rng.random(X.shape) < TEST_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)

    # 3. Structurally different corruption: sign flip on a small fraction
    flip_mask = rng.random(X.shape) < TEST_SIGN_FLIP_RATE
    X_noisy = np.where(flip_mask, -X_noisy, X_noisy)

    return X_noisy.astype(np.float32)


def main():
    rng = np.random.RandomState(RANDOM_SEED)

    # ---- single-packet baseline ----
    d = np.load(DATA_DIR / "single_packet_baseline.npz")
    X_train, X_test = d["X_train"], d["X_test"]
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)  # avoid 0-std features

    X_test_noisy = add_test_noise(X_test, feature_std, rng)

    out = {k: d[k] for k in d.files}
    out["X_test_noisy"] = X_test_noisy
    np.savez_compressed(DATA_DIR / "single_packet_baseline.npz", **out)
    print(f"single_packet_baseline.npz: added X_test_noisy {X_test_noisy.shape}")

    # ---- windowed transformer ----
    d2 = np.load(DATA_DIR / "windowed_transformer.npz")
    X_train_w, X_test_w = d2["X_train"], d2["X_test"]
    # std computed per-feature across all packets/windows in train
    feature_std_w = X_train_w.reshape(-1, X_train_w.shape[-1]).std(axis=0)
    feature_std_w = np.where(feature_std_w == 0, 1.0, feature_std_w)

    X_test_w_noisy = add_test_noise(X_test_w, feature_std_w, rng)

    out2 = {k: d2[k] for k in d2.files}
    out2["X_test_noisy"] = X_test_w_noisy
    np.savez_compressed(DATA_DIR / "windowed_transformer.npz", **out2)
    print(f"windowed_transformer.npz: added X_test_noisy {X_test_w_noisy.shape}")

    # ---- flattened ablation (derived from the same noisy windowed array) ----
    d3 = np.load(DATA_DIR / "flattened_ablation.npz")
    n_test = d3["X_test"].shape[0]
    n_feat_flat = d3["X_test"].shape[1]
    X_test_flat_noisy = X_test_w_noisy.reshape(n_test, n_feat_flat)

    out3 = {k: d3[k] for k in d3.files}
    out3["X_test_noisy"] = X_test_flat_noisy
    np.savez_compressed(DATA_DIR / "flattened_ablation.npz", **out3)
    print(f"flattened_ablation.npz: added X_test_noisy {X_test_flat_noisy.shape}")

    # document the two noise processes for the paper's methodology section
    info = {
        "train_time_augmentation": {
            "gaussian_noise_level": TRAIN_NOISE_LEVEL,
            "missing_rate": TRAIN_MISSING_RATE,
            "applied": "on-the-fly during training, regenerated each epoch/pass",
        },
        "test_time_evaluation_noise": {
            "gaussian_noise_level": TEST_NOISE_LEVEL,
            "missing_rate": TEST_MISSING_RATE,
            "sign_flip_rate": TEST_SIGN_FLIP_RATE,
            "applied": "once, to the held-out test split only",
        },
        "note": "Test-time noise uses a higher magnitude, a different "
                "missing rate, AND an additional corruption mechanism "
                "(sign flip) not present in training augmentation. This "
                "is a deliberate distributional gap, not a scaled-up "
                "version of the same process, so that noisy-test "
                "performance reflects generalization rather than "
                "memorization of the augmentation noise.",
    }
    with open(DATA_DIR / "noise_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved noise process documentation to {DATA_DIR}/noise_info.json")


if __name__ == "__main__":
    main()
