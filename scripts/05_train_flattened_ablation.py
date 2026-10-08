"""
Stage 5 — Flattened-window ablation (DT, RF, XGBoost, CNN, LSTM).

Trains the SAME baseline models as 03_train_baselines.py, but on
flattened_ablation.npz instead of single_packet_baseline.npz — i.e. each
model now sees the SAME 10-packet window context the Transformer gets,
just flattened into one long vector (10*18=180 features) instead of kept
as a sequence.

Purpose: isolate whether the Transformer's noise robustness comes from
its ATTENTION MECHANISM specifically, or simply from having more context
(10 packets) per prediction — something any model would benefit from.
If these flattened baselines ALSO recover much of their noisy-test
accuracy (compared to their single-packet scores in
data/results/baselines/), that tells you the robustness gain is mostly
"more context." If they stay closer to their single-packet noisy scores
despite having the same context as the Transformer, that's stronger
evidence the attention architecture itself is doing the real work.

Run this AFTER 02_prepare_data.py and 02b_add_test_noise.py.
Compare these results directly against data/results/baselines/ (primary,
single-packet) and data/results/transformer/ (windowed, attention-based).
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
OUTPUT_DIR = Path("data/results/flattened_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 64

# Same train-time noise parameters as 03_train_baselines.py and
# 04_train_transformer.py, so all three result sets are directly
# comparable — nobody gets an easier or harder training regime.
TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08
TRAIN_NOISE_SEED = 7
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
    with open(DATA_DIR / "selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)
    n_features = len(selected_features)

    d = np.load(DATA_DIR / "flattened_ablation.npz")
    total_dim = d["X_train"].shape[1]
    window_size = total_dim // n_features
    assert window_size * n_features == total_dim, (
        f"flattened dim {total_dim} doesn't divide evenly by "
        f"n_features {n_features} — window_size inference is wrong, "
        f"check selected_features.pkl matches what built this file"
    )
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            d["X_test"], d["y_test"], d["X_test_noisy"], le,
            window_size, n_features)


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

    results_log = evaluate("DecisionTree_Flattened", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("DecisionTree_Flattened", "noisy", y_test,
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

    results_log = evaluate("RandomForest_Flattened", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("RandomForest_Flattened", "noisy", y_test,
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

    results_log = evaluate("XGBoost_Flattened", "clean", y_test,
                            model.predict(X_test), le, elapsed, results_log)
    results_log = evaluate("XGBoost_Flattened", "noisy", y_test,
                            model.predict(X_test_noisy), le, None, results_log)
    return results_log


class NoiseAugmentedSequence(tf.keras.utils.Sequence):
    """Feeds a FRESH noisy version of the training data each epoch. Reshapes
    the flat (N, window_size*n_features) input back to its real windowed
    shape (N, window_size, n_features) before returning each batch — the
    flattening was only ever needed for the tree models, which have no
    concept of sequence structure. CNN/LSTM are sequence-capable, so they
    get the real window shape back, same as the Transformer sees, minus
    the attention mechanism."""
    def __init__(self, X, y, feature_std, batch_size, seed, window_size, n_features):
        self.X = X
        self.y = y
        self.feature_std = feature_std
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.window_size = window_size
        self.n_features = n_features

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        X_noisy = augment_with_noise(X_batch, self.feature_std, self.rng)
        mask = self.rng.random(len(X_batch)) < 0.5
        X_final = np.where(mask[:, None], X_noisy, X_batch)
        X_final = X_final.reshape(-1, self.window_size, self.n_features)
        return X_final, y_batch

    def on_epoch_end(self):
        perm = self.rng.permutation(len(self.X))
        self.X = self.X[perm]
        self.y = self.y[perm]


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


def train_keras_model(build_fn, model_name, X_train, y_train, X_val, y_val,
                       X_test, y_test, X_test_noisy, le, results_log,
                       window_size, n_features):
    tf.random.set_seed(RANDOM_SEED)
    n_classes = len(le.classes_)

    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    X_val_r = X_val.reshape(-1, window_size, n_features)
    X_test_r = X_test.reshape(-1, window_size, n_features)
    X_test_noisy_r = X_test_noisy.reshape(-1, window_size, n_features)

    train_seq = NoiseAugmentedSequence(
        X_train, y_train, feature_std, BATCH_SIZE, TRAIN_NOISE_SEED,
        window_size, n_features
    )

    model = build_fn(window_size, n_features, n_classes)
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
    print("Loading flattened-window ablation data (clean + noisy test)...")
    (X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy, le,
     window_size, n_features) = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, "
          f"Test: {X_test.shape}, Test (noisy): {X_test_noisy.shape}")
    print(f"Window size: {window_size}, features per packet: {n_features}")
    print(f"Classes: {le.classes_.tolist()}")
    print("\nNOTE: DT/RF/XGBoost see the truly flattened 180-length vector "
          "(they have no concept of sequence). CNN/LSTM get the REAL "
          "(10, 18) windowed shape back — same information as the "
          "Transformer, just without attention — since they're "
          "sequence-capable models and flattening them to (180,1) would "
          "be a meaningless artifact, not a fair comparison.")

    results_log = {}

    print("\n--- Decision Tree (flattened window) ---")
    results_log = train_decision_tree(X_train, y_train, X_test, y_test,
                                       X_test_noisy, le, results_log)

    print("\n--- Random Forest (flattened window) ---")
    results_log = train_random_forest(X_train, y_train, X_test, y_test,
                                       X_test_noisy, le, results_log)

    print("\n--- XGBoost (flattened window) ---")
    results_log = train_xgboost(X_train, y_train, X_test, y_test,
                                 X_test_noisy, le, results_log)

    print("\n--- CNN (real window shape, no attention) ---")
    results_log = train_keras_model(
        build_cnn, "CNN_Flattened", X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, le, results_log, window_size, n_features
    )

    print("\n--- LSTM (real window shape, no attention) ---")
    results_log = train_keras_model(
        build_lstm, "LSTM_Flattened", X_train, y_train, X_val, y_val,
        X_test, y_test, X_test_noisy, le, results_log, window_size, n_features
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
    print("SUMMARY — flattened-window ablation: clean vs. noisy")
    print("="*70)
    print(pivot.to_string())
    pivot.to_csv(OUTPUT_DIR / "summary_clean_vs_noisy.csv")

    print(f"\nSaved all reports, confusion matrices, and summaries to {OUTPUT_DIR}/")
    print("\nCompare noisy accuracy_drop here against data/results/baselines/"
          "summary_clean_vs_noisy.csv (single-packet, same models) and "
          "data/results/transformer/summary_clean_vs_noisy.csv (windowed, "
          "attention-based) to see whether context size alone explains "
          "the Transformer's robustness, or whether attention adds "
          "something beyond that.")


if __name__ == "__main__":
    main()
