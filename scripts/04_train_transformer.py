"""
Stage 4 — Transformer + Optuna-tuned BBOT.

Trains on windowed_transformer.npz — real (N, 10, 18) packet sequences,
with positional encoding and multi-head self-attention.

Run this AFTER 02_prepare_data.py and 02b_add_test_noise.py.

Produces TWO trained models for the ablation table:
  1. "Transformer_Tuned"   — hyperparameters selected by Optuna (TPE)
  2. "Transformer_Default" — fixed, untuned hyperparameters
Both trained identically otherwise (same data, same noise augmentation,
same epoch budget for the final fit) so the comparison isolates the
effect of tuning, not incidental differences in setup.

Both are evaluated on clean AND noisy test sets, same as the baselines,
so results are directly comparable in the same summary table format.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import optuna

# ============================== CONFIG ==============================
DATA_DIR = Path("data/prepared")
OUTPUT_DIR = Path("data/results/transformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

N_OPTUNA_TRIALS = 30
OPTUNA_SEARCH_EPOCHS = 10     # short budget per trial to keep search fast
OPTUNA_SEARCH_PATIENCE = 3

FINAL_EPOCHS = 40
FINAL_PATIENCE = 6
BATCH_SIZE = 64

# same noise parameters as 03_train_baselines.py — kept identical so the
# Transformer and baselines see the same train-time noise process
TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08
TRAIN_NOISE_SEED = 7

DEFAULT_HYPERPARAMS = {
    "num_layers": 2,
    "num_heads": 4,
    "d_model": 64,
    "ff_dim": 128,
    "dropout": 0.1,
    "learning_rate": 1e-3,
}
# ======================================================================


def load_data():
    d = np.load(DATA_DIR / "windowed_transformer.npz")
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            d["X_test"], d["y_test"], d["X_test_noisy"], le)


def augment_with_noise(X, feature_std, rng):
    """Same process as 03_train_baselines.py's train-time augmentation,
    generalized to the (N, W, F) windowed shape via broadcasting."""
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


def build_transformer(window_size, n_features, n_classes, num_layers,
                       num_heads, d_model, ff_dim, dropout, learning_rate):
    inputs = layers.Input(shape=(window_size, n_features))
    x = layers.Dense(d_model)(inputs)  # project raw features to d_model
    pos_enc = positional_encoding(window_size, d_model)
    x = x + pos_enc

    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads, d_model, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_optuna_search(X_train, y_train, X_val, y_val, window_size, n_features, n_classes):
    def objective(trial):
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        # key_dim = d_model / num_heads must be a positive integer
        if d_model % num_heads != 0:
            raise optuna.TrialPruned()
        ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)

        tf.random.set_seed(RANDOM_SEED)
        model = build_transformer(
            window_size, n_features, n_classes,
            num_layers, num_heads, d_model, ff_dim, dropout, learning_rate
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=OPTUNA_SEARCH_EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(patience=OPTUNA_SEARCH_PATIENCE,
                                         restore_best_weights=True)
            ],
            verbose=0,
        )
        val_acc = max(history.history["val_accuracy"])
        return val_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\nBest trial: val_accuracy={study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)
    with open(OUTPUT_DIR / "optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    return study.best_params


def evaluate(model_name, condition, y_true, y_pred, le, elapsed_s, results_log):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"\n{'='*60}")
    print(f"{model_name} [{condition}] — accuracy: {acc:.4f} | "
          f"weighted P: {precision:.4f} R: {recall:.4f} F1: {f1:.4f}"
          + (f" | train time: {elapsed_s:.1f}s" if elapsed_s is not None else ""))
    print(f"{'='*60}")

    report = classification_report(
        y_true, y_pred, target_names=le.classes_, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    with open(OUTPUT_DIR / f"{model_name}_{condition}_report.txt", "w") as f:
        f.write(f"{model_name} [{condition}]\n")
        f.write(f"Accuracy: {acc:.4f}\nWeighted P: {precision:.4f}\n"
                f"Weighted R: {recall:.4f}\nWeighted F1: {f1:.4f}\n")
        if elapsed_s is not None:
            f.write(f"Train time (s): {elapsed_s:.1f}\n")
        f.write("\n" + report)
    np.savetxt(OUTPUT_DIR / f"{model_name}_{condition}_confusion_matrix.csv",
               cm, delimiter=",", fmt="%d")

    key = f"{model_name}_{condition}"
    results_log[key] = {
        "model": model_name, "condition": condition,
        "accuracy": float(acc), "precision_weighted": float(precision),
        "recall_weighted": float(recall), "f1_weighted": float(f1),
        "train_time_s": float(elapsed_s) if elapsed_s is not None else None,
    }
    return results_log


def train_and_evaluate(model_name, hyperparams, X_train, y_train, X_val, y_val,
                        X_test, y_test, X_test_noisy, window_size, n_features,
                        n_classes, le, results_log):
    tf.random.set_seed(RANDOM_SEED)
    model = build_transformer(
        window_size, n_features, n_classes,
        hyperparams["num_layers"], hyperparams["num_heads"],
        hyperparams["d_model"], hyperparams["ff_dim"],
        hyperparams["dropout"], hyperparams["learning_rate"],
    )

    feature_std = X_train.reshape(-1, n_features).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    train_seq = NoiseAugmentedSequence(
        X_train, y_train, feature_std, BATCH_SIZE, TRAIN_NOISE_SEED
    )

    start = time.time()
    model.fit(
        train_seq,
        validation_data=(X_val, y_val),
        epochs=FINAL_EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=FINAL_PATIENCE,
                                            restore_best_weights=True)],
        verbose=1,
    )
    elapsed = time.time() - start

    y_pred_clean = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_pred_noisy = np.argmax(model.predict(X_test_noisy, verbose=0), axis=1)

    results_log = evaluate(model_name, "clean", y_test, y_pred_clean, le,
                            elapsed, results_log)
    results_log = evaluate(model_name, "noisy", y_test, y_pred_noisy, le,
                            None, results_log)

    model.save(OUTPUT_DIR / f"{model_name}.keras")
    return results_log


def main():
    print("Loading windowed transformer data (clean + noisy test)...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy, le = load_data()
    window_size, n_features = X_train.shape[1], X_train.shape[2]
    n_classes = len(le.classes_)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, "
          f"Test: {X_test.shape}, Test (noisy): {X_test_noisy.shape}")
    print(f"Window size: {window_size}, Features: {n_features}, Classes: {n_classes}")

    results_log = {}

    print(f"\n{'='*60}\nOptuna hyperparameter search ({N_OPTUNA_TRIALS} trials)\n{'='*60}")
    best_params = run_optuna_search(X_train, y_train, X_val, y_val,
                                     window_size, n_features, n_classes)

    print(f"\n{'='*60}\nTraining Transformer_Tuned (Optuna best params)\n{'='*60}")
    results_log = train_and_evaluate(
        "Transformer_Tuned", best_params, X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, window_size, n_features, n_classes,
        le, results_log
    )

    print(f"\n{'='*60}\nTraining Transformer_Default (fixed hyperparams, for ablation)\n{'='*60}")
    results_log = train_and_evaluate(
        "Transformer_Default", DEFAULT_HYPERPARAMS, X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, window_size, n_features, n_classes,
        le, results_log
    )

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_log, f, indent=2)

    summary_df = pd.DataFrame(results_log).T
    summary_df.to_csv(OUTPUT_DIR / "summary_raw.csv")

    pivot = summary_df.pivot_table(
        index="model", columns="condition",
        values=["accuracy", "f1_weighted"], aggfunc="first"
    )
    pivot["accuracy_drop"] = pivot[("accuracy", "clean")] - pivot[("accuracy", "noisy")]
    pivot["f1_drop"] = pivot[("f1_weighted", "clean")] - pivot[("f1_weighted", "noisy")]

    print("\n" + "="*70)
    print("SUMMARY — Transformer: tuned vs. default, clean vs. noisy")
    print("="*70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "summary_clean_vs_noisy.csv")

    print(f"\nSaved all reports, models, and summaries to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
