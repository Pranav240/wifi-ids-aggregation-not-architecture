"""
Stage 3a — Baseline model training (DT, RF, XGBoost, CNN, LSTM).

Trains on single_packet_baseline.npz — this is the PRIMARY comparison
against the Transformer, matching how these models would actually be
deployed (classify one packet at a time, no buffering).

Run this AFTER 02_prepare_data.py.

Produces per-class precision/recall/F1 for every model (not just
aggregate accuracy) and saves everything to data/results/baselines/.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================== CONFIG ==============================
DATA_DIR = Path("data/prepared")
OUTPUT_DIR = Path("data/results/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 64

# Train-time noise augmentation — MUST match noise_info.json's
# "train_time_augmentation" values from 02b_add_test_noise.py, so the
# documented gap between train and test noise is accurate.
TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08
TRAIN_NOISE_SEED = 7   # different from the test-noise seed (123)
# ======================================================================


def augment_with_noise(X, feature_std, rng):
    """Train-time noise augmentation — Gaussian + missing values only,
    deliberately simpler/milder than the test-time noise process (which
    also includes a sign-flip corruption). Returns a NEW array; caller
    decides whether to concatenate with clean data or replace it."""
    X_noisy = X.copy()
    gaussian = rng.normal(0, TRAIN_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian
    missing_mask = rng.random(X.shape) < TRAIN_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)
    return X_noisy.astype(np.float32)


def load_data():
    d = np.load(DATA_DIR / "single_packet_baseline.npz")
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            d["X_test"], d["y_test"], d["X_test_noisy"], le)


def evaluate(model_name, condition, y_true, y_pred, le, elapsed_s, results_log):
    """condition: 'clean' or 'noisy' — kept separate in the results log so
    the paper can report both side by side, not just one blended number."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
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
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted precision: {precision:.4f}\n")
        f.write(f"Weighted recall: {recall:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n")
        if elapsed_s is not None:
            f.write(f"Train time (s): {elapsed_s:.1f}\n")
        f.write("\n")
        f.write(report)

    np.savetxt(OUTPUT_DIR / f"{model_name}_{condition}_confusion_matrix.csv", cm,
               delimiter=",", fmt="%d")

    key = f"{model_name}_{condition}"
    results_log[key] = {
        "model": model_name,
        "condition": condition,
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "train_time_s": float(elapsed_s) if elapsed_s is not None else None,
    }
    return results_log


def train_decision_tree(X_train, y_train, X_test, y_test, X_test_noisy, le, results_log):
    rng = np.random.RandomState(TRAIN_NOISE_SEED)
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    # trees have no epoch loop to inject noise into each pass, so augment
    # once by concatenating a noisy copy alongside the clean training data
    X_train_noisy_copy = augment_with_noise(X_train, feature_std, rng)
    X_train_aug = np.concatenate([X_train, X_train_noisy_copy], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)

    start = time.time()
    model = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=20)
    model.fit(X_train_aug, y_train_aug)
    elapsed = time.time() - start

    results_log = evaluate("DecisionTree", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("DecisionTree", "noisy", y_test,
                            model.predict(X_test_noisy), le, None, results_log)
    return results_log


def train_random_forest(X_train, y_train, X_test, y_test, X_test_noisy, le, results_log):
    rng = np.random.RandomState(TRAIN_NOISE_SEED)
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_train_noisy_copy = augment_with_noise(X_train, feature_std, rng)
    X_train_aug = np.concatenate([X_train, X_train_noisy_copy], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)

    start = time.time()
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train_aug, y_train_aug)
    elapsed = time.time() - start

    results_log = evaluate("RandomForest", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("RandomForest", "noisy", y_test,
                            model.predict(X_test_noisy), le, None, results_log)
    return results_log


def train_xgboost(X_train, y_train, X_test, y_test, X_test_noisy, le, results_log):
    rng = np.random.RandomState(TRAIN_NOISE_SEED)
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_train_noisy_copy = augment_with_noise(X_train, feature_std, rng)
    X_train_aug = np.concatenate([X_train, X_train_noisy_copy], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)

    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        random_state=RANDOM_SEED, eval_metric="mlogloss", tree_method="hist"
    )
    model.fit(X_train_aug, y_train_aug)
    elapsed = time.time() - start

    results_log = evaluate("XGBoost", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("XGBoost", "noisy", y_test,
                            model.predict(X_test_noisy), le, None, results_log)
    return results_log


class NoiseAugmentedSequence(tf.keras.utils.Sequence):
    """Feeds a FRESH noisy version of the training data each epoch, rather
    than a single fixed noisy copy — this is what "on-the-fly" augmentation
    means for the Keras models, unlike the tree models above which only
    get one static noisy copy concatenated in."""
    def __init__(self, X, y, feature_std, batch_size, seed):
        self.X = X
        self.y = y
        self.feature_std = feature_std
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.input_dim = X.shape[1]

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        X_noisy = augment_with_noise(X_batch, self.feature_std, self.rng)
        # mix clean and noisy within the batch (50/50) rather than
        # replacing clean data entirely, so the model doesn't lose
        # calibration on clean inputs
        mask = self.rng.random(len(X_batch)) < 0.5
        X_final = np.where(mask[:, None], X_noisy, X_batch)
        return X_final.reshape(-1, self.input_dim, 1), y_batch

    def on_epoch_end(self):
        perm = self.rng.permutation(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]


def build_cnn(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
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


def build_lstm(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_keras_model(build_fn, model_name, X_train, y_train, X_val, y_val,
                       X_test, y_test, X_test_noisy, le, results_log):
    tf.random.set_seed(RANDOM_SEED)
    n_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    X_val_r = X_val.reshape(-1, input_dim, 1)
    X_test_r = X_test.reshape(-1, input_dim, 1)
    X_test_noisy_r = X_test_noisy.reshape(-1, input_dim, 1)

    train_seq = NoiseAugmentedSequence(
        X_train, y_train, feature_std, BATCH_SIZE, TRAIN_NOISE_SEED
    )

    model = build_fn(input_dim, n_classes)
    start = time.time()
    model.fit(
        train_seq,
        validation_data=(X_val_r, y_val),
        epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )
    elapsed = time.time() - start

    y_pred_clean = np.argmax(model.predict(X_test_r, verbose=0), axis=1)
    y_pred_noisy = np.argmax(model.predict(X_test_noisy_r, verbose=0), axis=1)

    results_log = evaluate(model_name, "clean", y_test, y_pred_clean, le,
                            elapsed, results_log)
    results_log = evaluate(model_name, "noisy", y_test, y_pred_noisy, le,
                            None, results_log)
    return results_log


def main():
    print("Loading single-packet baseline data (clean + noisy test)...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy, le = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, "
          f"Test: {X_test.shape}, Test (noisy): {X_test_noisy.shape}")
    print(f"Classes: {le.classes_.tolist()}")

    results_log = {}

    print("\n--- Decision Tree ---")
    results_log = train_decision_tree(X_train, y_train, X_test, y_test,
                                       X_test_noisy, le, results_log)

    print("\n--- Random Forest ---")
    results_log = train_random_forest(X_train, y_train, X_test, y_test,
                                       X_test_noisy, le, results_log)

    print("\n--- XGBoost ---")
    results_log = train_xgboost(X_train, y_train, X_test, y_test,
                                 X_test_noisy, le, results_log)

    print("\n--- CNN ---")
    results_log = train_keras_model(
        build_cnn, "CNN", X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, le, results_log
    )

    print("\n--- LSTM ---")
    results_log = train_keras_model(
        build_lstm, "LSTM", X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, le, results_log
    )

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(results_log, f, indent=2)

    summary_df = pd.DataFrame(results_log).T
    summary_df.to_csv(OUTPUT_DIR / "summary_raw.csv")

    # pivot into a clean-vs-noisy comparison table with a degradation column
    pivot = summary_df.pivot_table(
        index="model", columns="condition",
        values=["accuracy", "f1_weighted"], aggfunc="first"
    )
    pivot["accuracy_drop"] = pivot[("accuracy", "clean")] - pivot[("accuracy", "noisy")]
    pivot["f1_drop"] = pivot[("f1_weighted", "clean")] - pivot[("f1_weighted", "noisy")]

    print("\n" + "="*70)
    print("SUMMARY — clean vs. noisy test performance")
    print("="*70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "summary_clean_vs_noisy.csv")

    print(f"\nSaved all reports, confusion matrices, and summaries to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
