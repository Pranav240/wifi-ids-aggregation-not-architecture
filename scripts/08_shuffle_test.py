"""
Stage 8 — Within-Window Shuffle Test (DECISIVE EXPERIMENT).

PURPOSE
Our context-equalized ablation showed that giving models a 10-packet
window dramatically improves noise robustness. Two mechanisms could
explain this, and they imply very different papers:

  (A) TEMPORAL LEARNING: the models exploit the ORDER of packets in the
      window, learning genuine sequential attack signatures.

  (B) SAMPLE REDUNDANCY / NOISE AVERAGING: consecutive packets within a
      session are highly correlated (often near-identical), and the
      test-time noise is applied independently per element. Averaging
      ~10 noisy copies of a near-identical vector reduces noise by
      roughly sqrt(10). Any model that can average across the window
      benefits, with no temporal pattern learned at all.

THE TEST
Randomly permute packet order WITHIN each window, in both train and
test, keeping window membership and labels identical. Only the ordering
information is destroyed; the set of packets and all noise-averaging
opportunity is preserved exactly.

  - If accuracy is UNCHANGED -> mechanism (B). No temporal information
    is being used. The paper's claim must be reframed around sample
    redundancy, not temporal context.
  - If accuracy DROPS meaningfully -> mechanism (A) contributes. Order
    carries real information and the temporal framing is supported.

Models tested: RandomForest (flattened, the overall winner) and the
Optuna-tuned Transformer (the architecture whose entire premise is
order-awareness via positional encoding). If the Transformer is
insensitive to shuffling, its positional encoding is contributing
nothing on this data - a finding worth reporting either way.

Run AFTER 02_prepare_data.py, 02b_add_test_noise.py, 04_train_transformer.py.
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
OUTPUT_DIR = Path("data/results/shuffle_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]

EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 5

TRAIN_NOISE_LEVEL = 0.10
TRAIN_MISSING_RATE = 0.08

SHUFFLE_SEED = 9999   # fixed, so the shuffled dataset is identical across
                       # model seeds and only model randomness varies
# ======================================================================


def load_data():
    d = np.load(DATA_DIR / "windowed_transformer.npz")
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(TRANSFORMER_RESULTS_DIR / "optuna_best_params.json") as f:
        best_params = json.load(f)
    return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
            d["X_test"], d["y_test"], d["X_test_noisy"], le, best_params)


def shuffle_within_windows(X, rng):
    """Independently permute the packet order inside every window.
    Window membership, labels, and the multiset of packets per window are
    all preserved exactly - ONLY ordering is destroyed."""
    X_shuf = X.copy()
    n_windows, window_size, _ = X.shape
    for i in range(n_windows):
        perm = rng.permutation(window_size)
        X_shuf[i] = X_shuf[i][perm]
    return X_shuf


def augment_with_noise(X, feature_std, rng):
    X_noisy = X.copy()
    gaussian = rng.normal(0, TRAIN_NOISE_LEVEL, size=X.shape) * feature_std
    X_noisy = X_noisy + gaussian
    missing_mask = rng.random(X.shape) < TRAIN_MISSING_RATE
    X_noisy = np.where(missing_mask, 0.0, X_noisy)
    return X_noisy.astype(np.float32)


class NoiseAugmentedSequence(tf.keras.utils.Sequence):
    def __init__(self, X, y, feature_std, batch_size, seed):
        super().__init__()
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
    inputs = layers.Input(shape=(window_size, n_features))
    x = layers.Dense(params["d_model"])(inputs)
    x = x + positional_encoding(window_size, params["d_model"])
    for _ in range(params["num_layers"]):
        x = transformer_encoder_block(x, params["num_heads"], params["d_model"],
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


def run_rf(seed, X_train_w, y_train, X_test_w, y_test, X_test_noisy_w):
    """RandomForest on the flattened view of whatever windows it's given."""
    X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1)
    X_test_flat = X_test_w.reshape(X_test_w.shape[0], -1)
    X_test_noisy_flat = X_test_noisy_w.reshape(X_test_noisy_w.shape[0], -1)

    rng = np.random.RandomState(seed + 1000)
    feature_std = X_train_flat.std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    X_train_noisy_copy = augment_with_noise(X_train_flat, feature_std, rng)
    X_train_aug = np.concatenate([X_train_flat, X_train_noisy_copy], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1
    )
    model.fit(X_train_aug, y_train_aug)

    acc_c, f1_c = quick_metrics(y_test, model.predict(X_test_flat))
    acc_n, f1_n = quick_metrics(y_test, model.predict(X_test_noisy_flat))
    return acc_c, f1_c, acc_n, f1_n


def run_transformer(seed, params, window_size, n_features, n_classes,
                     X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy):
    tf.random.set_seed(seed)
    feature_std = X_train.reshape(-1, n_features).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    train_seq = NoiseAugmentedSequence(X_train, y_train, feature_std,
                                        BATCH_SIZE, seed + 2000)
    model = build_transformer(window_size, n_features, n_classes, params)
    model.fit(
        train_seq, validation_data=(X_val, y_val), epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=PATIENCE,
                                            restore_best_weights=True)],
        verbose=0,
    )
    acc_c, f1_c = quick_metrics(
        y_test, np.argmax(model.predict(X_test, verbose=0), axis=1))
    acc_n, f1_n = quick_metrics(
        y_test, np.argmax(model.predict(X_test_noisy, verbose=0), axis=1))
    return acc_c, f1_c, acc_n, f1_n


def report_redundancy_diagnostic(X_train):
    """Quantify HOW redundant consecutive packets within a window are.
    If within-window variance is tiny relative to between-window variance,
    that is direct evidence for the noise-averaging explanation."""
    n_windows, window_size, n_features = X_train.shape
    # variance across packets WITHIN each window, averaged over windows
    within = X_train.var(axis=1).mean(axis=0)
    # variance across windows of the window-mean vector
    between = X_train.mean(axis=1).var(axis=0)

    ratio = within / np.where(between == 0, np.nan, between)
    print("\n" + "=" * 70)
    print("WINDOW REDUNDANCY DIAGNOSTIC")
    print("=" * 70)
    print(f"Mean within-window variance  (avg over features): {within.mean():.6f}")
    print(f"Mean between-window variance (avg over features): {between.mean():.6f}")
    print(f"Median per-feature ratio (within/between):        "
          f"{np.nanmedian(ratio):.4f}")
    print("\nInterpretation: a ratio well below 1.0 means packets inside a")
    print("window are far more similar to each other than windows are to each")
    print("other - i.e. the window is largely redundant copies, which is")
    print("exactly the condition under which noise-averaging (not temporal")
    print("learning) explains robustness gains.")
    return {
        "within_window_variance_mean": float(within.mean()),
        "between_window_variance_mean": float(between.mean()),
        "median_within_between_ratio": float(np.nanmedian(ratio)),
    }


def main():
    print("Loading windowed data + Optuna best params...")
    (X_train, y_train, X_val, y_val, X_test, y_test, X_test_noisy,
     le, best_params) = load_data()
    window_size, n_features = X_train.shape[1], X_train.shape[2]
    n_classes = len(le.classes_)
    print(f"Window size: {window_size}, features: {n_features}, classes: {n_classes}")

    diagnostic = report_redundancy_diagnostic(X_train)

    shuf_rng = np.random.RandomState(SHUFFLE_SEED)
    X_train_s = shuffle_within_windows(X_train, shuf_rng)
    X_val_s = shuffle_within_windows(X_val, shuf_rng)
    X_test_s = shuffle_within_windows(X_test, shuf_rng)
    X_test_noisy_s = shuffle_within_windows(X_test_noisy, shuf_rng)

    # sanity check: shuffling must preserve the per-window multiset of packets
    assert np.allclose(np.sort(X_train.reshape(-1), kind="stable"),
                        np.sort(X_train_s.reshape(-1), kind="stable")), \
        "shuffle altered values, not just order"
    print("\nSanity check passed: shuffling preserved all values, changed only order.")

    rows = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")

        for condition, (Xtr, Xv, Xte, Xten) in {
            "ordered": (X_train, X_val, X_test, X_test_noisy),
            "shuffled": (X_train_s, X_val_s, X_test_s, X_test_noisy_s),
        }.items():
            print(f"  RandomForest [{condition}]...")
            t0 = time.time()
            acc_c, f1_c, acc_n, f1_n = run_rf(seed, Xtr, y_train, Xte, y_test, Xten)
            print(f"    clean={acc_c:.4f}  noisy={acc_n:.4f}  ({time.time()-t0:.1f}s)")
            rows.append({"model": "RandomForest", "order": condition, "seed": seed,
                          "accuracy_clean": acc_c, "f1_clean": f1_c,
                          "accuracy_noisy": acc_n, "f1_noisy": f1_n})

            print(f"  Transformer [{condition}]...")
            t0 = time.time()
            acc_c, f1_c, acc_n, f1_n = run_transformer(
                seed, best_params, window_size, n_features, n_classes,
                Xtr, y_train, Xv, y_val, Xte, y_test, Xten)
            print(f"    clean={acc_c:.4f}  noisy={acc_n:.4f}  ({time.time()-t0:.1f}s)")
            rows.append({"model": "Transformer", "order": condition, "seed": seed,
                          "accuracy_clean": acc_c, "f1_clean": f1_c,
                          "accuracy_noisy": acc_n, "f1_noisy": f1_n})

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "shuffle_test_raw.csv", index=False)

    summary = raw.groupby(["model", "order"]).agg(
        acc_clean_mean=("accuracy_clean", "mean"),
        acc_clean_std=("accuracy_clean", "std"),
        acc_noisy_mean=("accuracy_noisy", "mean"),
        acc_noisy_std=("accuracy_noisy", "std"),
    )
    summary.to_csv(OUTPUT_DIR / "shuffle_test_summary.csv")

    print("\n" + "=" * 70)
    print("SHUFFLE TEST SUMMARY (n=3 seeds)")
    print("=" * 70)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    verdicts = {}
    for model in raw["model"].unique():
        o = summary.loc[(model, "ordered"), "acc_noisy_mean"]
        s = summary.loc[(model, "shuffled"), "acc_noisy_mean"]
        o_std = summary.loc[(model, "ordered"), "acc_noisy_std"]
        delta = (o - s) * 100
        print(f"\n{model}: ordered={o*100:.2f}%  shuffled={s*100:.2f}%  "
              f"delta={delta:+.2f} pts (ordered std={o_std*100:.2f})")
        threshold = 2 * o_std * 100
        if abs(delta) < threshold:
            verdict = ("NO significant order effect - robustness is explained by "
                        "sample redundancy / noise averaging, NOT temporal learning. "
                        "The paper's framing must be changed accordingly.")
        elif delta > 0:
            verdict = ("Order DOES matter - shuffling degrades performance beyond "
                        "seed variance, supporting a genuine temporal-learning "
                        "contribution.")
        else:
            verdict = ("ANOMALY: shuffling IMPROVED performance beyond seed "
                        "variance. This does not support temporal learning; it "
                        "suggests the ordered arrangement was actively unhelpful "
                        "(e.g. the model overfitting to positional artifacts, or "
                        "shuffling acting as regularization). Investigate before "
                        "drawing conclusions - do not report as evidence of "
                        "sequence learning.")
        print(f"  -> {verdict}")
        verdicts[model] = {"ordered": float(o), "shuffled": float(s),
                            "delta_pts": float(delta), "verdict": verdict}

    with open(OUTPUT_DIR / "shuffle_test_verdict.json", "w") as f:
        json.dump({"redundancy_diagnostic": diagnostic, "verdicts": verdicts},
                   f, indent=2)

    print(f"\nSaved results and verdict to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
