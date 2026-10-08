"""
Stage 6 — Multi-seed evaluation of the close-cluster models.

Single-seed noisy-test results left RandomForest_Flattened, CNN_Flattened,
LSTM_Flattened, and Transformer_Tuned within ~1.3 points of each other
(except RF's clear lead). This script reruns just those four models
across multiple seeds to check whether that ranking is stable or just
single-run noise.

Does NOT re-run the Optuna search — that's a one-time hyperparameter
decision, not something that needs its own seed variance check. Loads
the best params already found and saved by 04_train_transformer.py.

All four models train on windowed_transformer.npz. RandomForest needs a
flat vector, so it's derived by reshaping the same windowed arrays —
NOT loaded from a separate file — to guarantee it's the exact same
underlying data as the other three, just reshaped.

Run this AFTER 02_prepare_data.py, 02b_add_test_noise.py, and
04_train_transformer.py (needs optuna_best_params.json).
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================== CONFIG ==============================
DATA_DIR = Path("data/prepared")
TRANSFORMER_RESULTS_DIR = Path("data/results/transformer")
OUTPUT_DIR = Path("data/results/multiseed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]

EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5

TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08
# ======================================================================


def load_data():
    d = np.load(DATA_DIR / "windowed_transformer.npz")
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(TRANSFORMER_RESULTS_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            d["X_test"], d["y_test"], d["X_test_noisy"], le, best_params)


def augment_with_noise(X, feature_std, rng):
    X_noisy = X.copy()
    gaussian = rng.normal(0, TRAIN_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian
    missing_mask = rng.random(X.shape) < TRAIN_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)
    return X_noisy.astype(np.float32)


class NoiseAugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, feature_std, batch_size, seed):
        self.X = X
        self.y = y
        self.feature_std = feature_std
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        X_noisy = augment_with_noise(X_batch, self.feature_std, self.rng)
        mask = self.rng.random(len(X_batch)) < 0.5
        X_final = np.where(mask[:, None, None], X_noisy, X_batch)
        return X_final, y_batch

    def on_epoch_end(self):
        perm = self.rng.permutation(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]


def quick_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return acc, f1


def run_rf(seed, X_train_flat, y_train, X_test_flat, y_test, X_test_noisy_flat):
    rng = np.random.RandomState(seed + 1000)  # distinct from model's own seed
    feature_std = X_train_flat.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_train_noisy_copy = augment_with_noise(X_train_flat, feature_std, rng)
    X_train_aug = np.concatenate([X_train_flat, X_train_noisy_copy], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1
    )
    model.fit(X_train_aug, y_train_aug)

    acc_clean, f1_clean = quick_metrics(y_test, model.predict(X_test_flat))
    acc_noisy, f1_noisy = quick_metrics(y_test, model.predict(X_test_noisy_flat))
    return acc_clean, f1_clean, acc_noisy, f1_noisy


def build_cnn(window_size, n_features, n_classes):
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_lstm(window_size, n_features, n_classes):
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def positional_encoding(window_size, d_model):
    positions = np.arange(window_size)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles = positions * angle_rates
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(angles[np.newaxis, ...], dtype=tf.float32)


def transformer_encoder_block(x, num_heads, d_model, ff_dim, dropout):
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff_out = layers.Dense(ff_dim, activation="relu")(x)
    ff_out = layers.Dense(d_model)(ff_out)
    ff_out = layers.Dropout(dropout)(ff_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)
    return x


def build_transformer(window_size, n_features, n_classes, params):
    d_model = params["d_model"]
    num_heads = params["num_heads"]
    if d_model % num_heads != 0:
        # shouldn't happen since Optuna pruned invalid combos, but guard anyway
        raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")

    inputs = layers.Input(shape=(window_size, n_features))
    x = layers.Dense(d_model)(inputs)
    x = x + positional_encoding(window_size, d_model)
    for _ in range(params["num_layers"]):
        x = transformer_encoder_block(x, num_heads, d_model,
                                       params["ff_dim"], params["dropout"])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(params["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def run_keras_model(build_fn, seed, window_size, n_features, n_classes,
                     X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy,
                     extra_build_args=None):
    tf.random.set_seed(seed)
    feature_std = X_train.reshape(-1, n_features).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    train_seq = NoiseAugmentedSequence(X_train, y_train, feature_std,
                                        BATCH_SIZE, seed + 2000)

    args = [window_size, n_features, n_classes]
    if extra_build_args is not None:
        args.append(extra_build_args)
    model = build_fn(*args)

    model.fit(
        train_seq, validation_data=(X_val, y_val), epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=PATIENCE,
                                            restore_best_weights=True)],
        verbose=0,
    )

    acc_clean, f1_clean = quick_metrics(
        y_test, np.argmax(model.predict(X_test, verbose=0), axis=1))
    acc_noisy, f1_noisy = quick_metrics(
        y_test, np.argmax(model.predict(X_test_noisy, verbose=0), axis=1))
    return acc_clean, f1_clean, acc_noisy, f1_noisy


def main():
    print("Loading windowed data + Optuna best params...")
    (X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy,
     le, best_params) = load_data()
    window_size, n_features = X_train.shape[1], X_train.shape[2]
    n_classes = len(le.classes_)
    print(f"Window size: {window_size}, features: {n_features}, "
          f"classes: {n_classes}")
    print(f"Optuna best params: {best_params}")
    print(f"Seeds: {SEEDS}\n")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_noisy_flat = X_test_noisy.reshape(X_test_noisy.shape[0], -1)

    rows = []

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        print("  RandomForest_Flattened...")
        t0 = time.time()
        acc_c, f1_c, acc_n, f1_n = run_rf(
            seed, X_train_flat, y_train, X_test_flat, y_test, X_test_noisy_flat
        )
        print(f"    clean acc={acc_c:.4f} f1={f1_c:.4f} | "
              f"noisy acc={acc_n:.4f} f1={f1_n:.4f} ({time.time()-t0:.1f}s)")
        rows.append({"model": "RandomForest_Flattened", "seed": seed,
                     "accuracy_clean": acc_c, "f1_clean": f1_c,
                     "accuracy_noisy": acc_n, "f1_noisy": f1_n})

        print("  CNN_Flattened...")
        t0 = time.time()
        acc_c, f1_c, acc_n, f1_n = run_keras_model(
            build_cnn, seed, window_size, n_features, n_classes,
            X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy
        )
        print(f"    clean acc={acc_c:.4f} f1={f1_c:.4f} | "
              f"noisy acc={acc_n:.4f} f1={f1_n:.4f} ({time.time()-t0:.1f}s)")
        rows.append({"model": "CNN_Flattened", "seed": seed,
                     "accuracy_clean": acc_c, "f1_clean": f1_c,
                     "accuracy_noisy": acc_n, "f1_noisy": f1_n})

        print("  LSTM_Flattened...")
        t0 = time.time()
        acc_c, f1_c, acc_n, f1_n = run_keras_model(
            build_lstm, seed, window_size, n_features, n_classes,
            X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy
        )
        print(f"    clean acc={acc_c:.4f} f1={f1_c:.4f} | "
              f"noisy acc={acc_n:.4f} f1={f1_n:.4f} ({time.time()-t0:.1f}s)")
        rows.append({"model": "LSTM_Flattened", "seed": seed,
                     "accuracy_clean": acc_c, "f1_clean": f1_c,
                     "accuracy_noisy": acc_n, "f1_noisy": f1_n})

        print("  Transformer_Tuned...")
        t0 = time.time()
        acc_c, f1_c, acc_n, f1_n = run_keras_model(
            build_transformer, seed, window_size, n_features, n_classes,
            X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy,
            extra_build_args=best_params
        )
        print(f"    clean acc={acc_c:.4f} f1={f1_c:.4f} | "
              f"noisy acc={acc_n:.4f} f1={f1_n:.4f} ({time.time()-t0:.1f}s)")
        rows.append({"model": "Transformer_Tuned", "seed": seed,
                     "accuracy_clean": acc_c, "f1_clean": f1_c,
                     "accuracy_noisy": acc_n, "f1_noisy": f1_n})

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(OUTPUT_DIR / "multiseed_raw.csv", index=False)

    summary = raw_df.groupby("model").agg(
        accuracy_clean_mean=("accuracy_clean", "mean"),
        accuracy_clean_std=("accuracy_clean", "std"),
        f1_clean_mean=("f1_clean", "mean"),
        f1_clean_std=("f1_clean", "std"),
        accuracy_noisy_mean=("accuracy_noisy", "mean"),
        accuracy_noisy_std=("accuracy_noisy", "std"),
        f1_noisy_mean=("f1_noisy", "mean"),
        f1_noisy_std=("f1_noisy", "std"),
    ).sort_values("accuracy_noisy_mean", ascending=False)

    print("\n" + "="*70)
    print(f"MULTI-SEED SUMMARY (n={len(SEEDS)} seeds) — ranked by noisy accuracy")
    print("="*70)
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "multiseed_summary.csv")

    print(f"\nSaved raw per-seed results and summary to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
