"""
Stage 10 — Multi-Split Variance Test.

WHY THIS EXISTS
Every result reported so far uses ONE fixed train/val/test split
(RANDOM_SEED = 42 in 02_prepare_data.py). The multi-seed analysis in
stage 06 varied model initialisation and noise draws, but NOT the split.

That matters because the paper makes a variance-based claim: Random
Forest is presented as both the most accurate AND the most stable model
(+/-0.16% vs the Transformer's +/-1.54%). Split variance is typically
the LARGER source of uncertainty in a comparison like this, and none of
it is currently measured. A reviewer can reasonably object that the
reported error bars understate true uncertainty, and that RF's apparent
stability advantage is partly an artefact of RF having fewer stochastic
degrees of freedom than a neural network.

WHAT THIS DOES
Re-splits the SAME pool of windows with several different split seeds,
retrains, and decomposes the resulting variance:

  - within-split  variance: spread across model seeds, split held fixed
  - across-split  variance: spread of split means

If across-split variance is comparable to or larger than within-split
variance, the stage 06 error bars are too narrow and the paper must
report the wider figure.

Models: Random Forest (flattened) and the Optuna-tuned Transformer -
the two the paper's central comparison rests on.

IMPORTANT SCOPE NOTE
This re-splits the already-windowed, already-scaled pool. It does NOT
re-run windowing or re-fit the scaler per split, so a small amount of
scaler information leaks across splits. That makes this a LOWER BOUND on
true split variance. Doing it fully cleanly would require re-running
stage 02 per split; this is the affordable approximation and should be
described as such in the paper rather than overstated.

Run AFTER 02_prepare_data.py, 02b_add_test_noise.py, 04_train_transformer.py.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================== CONFIG ==============================
DATA_DIR = Path("data/prepared")
TRANSFORMER_RESULTS_DIR = Path("data/results/transformer")
OUTPUT_DIR = Path("data/results/split_variance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_SEEDS = [42, 7, 2718]      # 42 reproduces the original split
MODEL_SEEDS = [42, 123]           # kept small: runs = splits x model seeds

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5

TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08

# must match 02b_add_test_noise.py
TEST_NOISE_LEVEL = 0.25
TEST_MISSING_RATE = 0.15
TEST_SIGN_FLIP_RATE = 0.03
TEST_NOISE_SEED = 123
# ======================================================================


def load_pool():
    """Reassemble the full window pool from the saved splits so it can be
    re-split with different seeds."""
    d = np.load(DATA_DIR / "windowed_transformer.npz")
    X = np.concatenate([d["X_train"], d["X_val"], d["X_test"]], axis=0)
    y = np.concatenate([d["y_train"], d["y_val"], d["y_test"]], axis=0)
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(TRANSFORMER_RESULTS_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)
    return X, y, le, best_params


def make_split(X, y, split_seed):
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx, test_size=(1 - TRAIN_FRAC), random_state=split_seed, stratify=y)
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1 - rel_val), random_state=split_seed,
        stratify=y[idx_temp])
    return idx_train, idx_val, idx_test


def augment_with_noise(X, feature_std, rng):
    X_noisy = X.copy()
    gaussian = rng.normal(0, TRAIN_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian
    missing_mask = rng.random(X.shape) < TRAIN_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)
    return X_noisy.astype(np.float32)


def apply_test_noise(X, feature_std, rng):
    X_noisy = X.copy()
    gaussian = rng.normal(0, TEST_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian
    missing_mask = rng.random(X.shape) < TEST_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)
    flip_mask = rng.random(X.shape) < TEST_SIGN_FLIP_RATE
    X_noisy = np.where(flip_mask, -X_noisy, X_noisy)
    return X_noisy.astype(np.float32)


class NoiseAugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, feature_std, batch_size, seed):
        super().__init__()
        self.X, self.y = X, y
        self.feature_std = feature_std
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        s, e = idx * self.batch_size, (idx + 1) * self.batch_size
        Xb, yb = self.X[s:e], self.y[s:e]
        Xn = augment_with_noise(Xb, self.feature_std, self.rng)
        mask = self.rng.random(len(Xb)) < 0.5
        return np.where(mask[:, None, None], Xn, Xb), yb

    def on_epoch_end(self):
        p = self.rng.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]


def quick_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    return acc, f1


def positional_encoding(window_size, d_model):
    positions = np.arange(window_size)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    ang = positions * rates
    ang[:, 0::2] = np.sin(ang[:, 0::2])
    ang[:, 1::2] = np.cos(ang[:, 1::2])
    return tf.constant(ang[np.newaxis, ...], dtype=tf.float32)


def encoder_block(x, num_heads, d_model, ff_dim, dropout):
    a = layers.MultiHeadAttention(num_heads=num_heads,
                                   key_dim=d_model // num_heads)(x, x)
    a = layers.Dropout(dropout)(a)
    x = layers.LayerNormalization(epsilon=1e-6)(x + a)
    f = layers.Dense(ff_dim, activation="relu")(x)
    f = layers.Dense(d_model)(f)
    f = layers.Dropout(dropout)(f)
    return layers.LayerNormalization(epsilon=1e-6)(x + f)


def build_transformer(W, F, n_classes, p):
    inp = layers.Input(shape=(W, F))
    x = layers.Dense(p["d_model"])(inp)
    x = x + positional_encoding(W, p["d_model"])
    for _ in range(p["num_layers"]):
        x = encoder_block(x, p["num_heads"], p["d_model"], p["ff_dim"], p["dropout"])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(p["dropout"])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def run_rf(model_seed, Xtr, ytr, Xte, yte, Xte_noisy):
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(len(Xte), -1)
    Xten_f = Xte_noisy.reshape(len(Xte_noisy), -1)

    rng = np.random.RandomState(model_seed + 1000)
    fstd = Xtr_f.std(axis=0)
    fstd = np.where(fstd == 0, 1.0, fstd)
    Xtr_aug = np.concatenate([Xtr_f, augment_with_noise(Xtr_f, fstd, rng)], axis=0)
    ytr_aug = np.concatenate([ytr, ytr], axis=0)

    m = RandomForestClassifier(n_estimators=200, max_depth=20,
                                random_state=model_seed, n_jobs=-1)
    m.fit(Xtr_aug, ytr_aug)
    acc_c, f1_c = quick_metrics(yte, m.predict(Xte_f))
    acc_n, f1_n = quick_metrics(yte, m.predict(Xten_f))
    return acc_c, f1_c, acc_n, f1_n


def run_transformer(model_seed, params, Xtr, ytr, Xv, yv, Xte, yte, Xte_noisy,
                     n_classes):
    tf.random.set_seed(model_seed)
    W, F = Xtr.shape[1], Xtr.shape[2]
    fstd = Xtr.reshape(-1, F).std(axis=0)
    fstd = np.where(fstd == 0, 1.0, fstd)
    seq = NoiseAugmentedSequence(Xtr, ytr, fstd, BATCH_SIZE, model_seed + 2000)
    m = build_transformer(W, F, n_classes, params)
    m.fit(seq, validation_data=(Xv, yv), epochs=EPOCHS,
          callbacks=[callbacks.EarlyStopping(patience=PATIENCE,
                                              restore_best_weights=True)],
          verbose=0)
    acc_c, f1_c = quick_metrics(yte, np.argmax(m.predict(Xte, verbose=0), axis=1))
    acc_n, f1_n = quick_metrics(yte, np.argmax(m.predict(Xte_noisy, verbose=0), axis=1))
    return acc_c, f1_c, acc_n, f1_n


def main():
    X, y, le, best_params = load_pool()
    n_classes = len(le.classes_)
    F = X.shape[2]
    print(f"Window pool: {X.shape}, {n_classes} classes")
    print(f"Split seeds: {SPLIT_SEEDS} | model seeds: {MODEL_SEEDS}")
    print(f"Total runs per model: {len(SPLIT_SEEDS) * len(MODEL_SEEDS)}")

    rows = []
    for split_seed in SPLIT_SEEDS:
        idx_tr, idx_v, idx_te = make_split(X, y, split_seed)
        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xv, yv = X[idx_v], y[idx_v]
        Xte, yte = X[idx_te], y[idx_te]

        # regenerate test noise for THIS split's test set
        noise_rng = np.random.RandomState(TEST_NOISE_SEED)
        fstd = Xtr.reshape(-1, F).std(axis=0)
        fstd = np.where(fstd == 0, 1.0, fstd)
        Xte_noisy = apply_test_noise(Xte, fstd, noise_rng)

        print(f"\n{'=' * 60}\nSPLIT SEED {split_seed}  "
              f"(train {len(Xtr)} / val {len(Xv)} / test {len(Xte)})\n{'=' * 60}")

        for ms in MODEL_SEEDS:
            t0 = time.time()
            acc_c, f1_c, acc_n, f1_n = run_rf(ms, Xtr, ytr, Xte, yte, Xte_noisy)
            print(f"  RandomForest  [model seed {ms}] clean={acc_c:.4f} "
                  f"noisy={acc_n:.4f}  ({time.time() - t0:.1f}s)")
            rows.append({"model": "RandomForest", "split_seed": split_seed,
                          "model_seed": ms, "accuracy_clean": acc_c,
                          "f1_clean": f1_c, "accuracy_noisy": acc_n,
                          "f1_noisy": f1_n})

            t0 = time.time()
            acc_c, f1_c, acc_n, f1_n = run_transformer(
                ms, best_params, Xtr, ytr, Xv, yv, Xte, yte, Xte_noisy, n_classes)
            print(f"  Transformer   [model seed {ms}] clean={acc_c:.4f} "
                  f"noisy={acc_n:.4f}  ({time.time() - t0:.1f}s)")
            rows.append({"model": "Transformer", "split_seed": split_seed,
                          "model_seed": ms, "accuracy_clean": acc_c,
                          "f1_clean": f1_c, "accuracy_noisy": acc_n,
                          "f1_noisy": f1_n})

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "split_variance_raw.csv", index=False)

    print("\n" + "=" * 70)
    print("VARIANCE DECOMPOSITION (noisy-test accuracy)")
    print("=" * 70)

    verdict = {}
    for model in raw["model"].unique():
        sub = raw[raw["model"] == model]
        split_means = sub.groupby("split_seed")["accuracy_noisy"].mean()
        # spread across model seeds with the split held fixed, averaged
        within = sub.groupby("split_seed")["accuracy_noisy"].std().mean()
        across = split_means.std()
        overall = sub["accuracy_noisy"].mean()

        print(f"\n{model}")
        print(f"  overall mean noisy accuracy : {overall * 100:.2f}%")
        print(f"  per-split means             : "
              f"{', '.join(f'{s}={v*100:.2f}%' for s, v in split_means.items())}")
        if np.isnan(within):
            print(f"  within-split std (model seed): n/a "
                  f"(needs >=2 model seeds; only {len(MODEL_SEEDS)} configured)")
        else:
            print(f"  within-split std (model seed): {within * 100:.3f} pts")
        print(f"  across-split std (data split): {across * 100:.3f} pts")

        if np.isnan(within):
            note = (f"Cannot compare variance sources: only {len(MODEL_SEEDS)} "
                     f"model seed(s) configured, so within-split variance is "
                     f"undefined. Across-split std is "
                     f"{across * 100:.3f} pts - report this, but re-run with "
                     f">=2 model seeds to decompose the sources.")
        elif across > within:
            note = ("Split variance EXCEEDS model-seed variance. The stage 06 "
                     "error bars understate true uncertainty; report the wider "
                     "across-split figure in the paper.")
        else:
            note = ("Split variance is within model-seed variance. The stage 06 "
                     "error bars are a reasonable, if incomplete, uncertainty "
                     "estimate.")
        print(f"  -> {note}")

        verdict[model] = {
            "overall_mean_noisy": float(overall),
            "within_split_std": None if np.isnan(within) else float(within),
            "across_split_std": float(across),
            "per_split_means": {int(k): float(v) for k, v in split_means.items()},
            "note": note,
        }

    # does the RF-over-Transformer gap survive split variation?
    rf = verdict.get("RandomForest")
    tr = verdict.get("Transformer")
    if rf and tr:
        gap = (rf["overall_mean_noisy"] - tr["overall_mean_noisy"]) * 100
        combined = np.sqrt(rf["across_split_std"] ** 2
                            + tr["across_split_std"] ** 2) * 100
        print("\n" + "-" * 70)
        print(f"RF - Transformer gap: {gap:+.2f} pts | combined across-split "
              f"std: {combined:.2f} pts")
        if abs(gap) > 2 * combined:
            concl = ("Gap survives split variation - the RF advantage is robust "
                      "to how the data is partitioned.")
        else:
            concl = ("Gap is NOT clearly larger than split-induced variation - "
                      "the paper should soften any claim that RF definitively "
                      "outperforms, and report the overlap honestly.")
        print(f"-> {concl}")
        verdict["gap_analysis"] = {"gap_pts": float(gap),
                                    "combined_across_split_std_pts": float(combined),
                                    "conclusion": concl}

    with open(OUTPUT_DIR / "split_variance_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
