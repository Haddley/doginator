#!/usr/bin/env python3
"""
Train a bark/not-bark classifier on top of YAMNet embeddings.

Usage:
  python3 train_model.py

Saves: model/bark_classifier.keras and model/yamnet/
Deploy: sudo cp -r model/ /opt/doginator/model/
"""
import csv
import sys
from pathlib import Path

import numpy as np

LABELS_FILE = Path(__file__).parent / "labels.csv"
RECORDINGS_DIR = Path("/var/log/doginator/recordings")
MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "bark_classifier.keras"


def load_labels():
    labels = {}
    with open(LABELS_FILE) as f:
        for row in csv.DictReader(f):
            if row["filename"] != "filename":  # skip duplicate header rows
                labels[row["filename"]] = row["label"]
    return labels


def main():
    import tensorflow as tf
    import tensorflow_hub as hub
    from pydub import AudioSegment

    MODEL_DIR.mkdir(exist_ok=True)

    labels_dict = load_labels()

    paths, y = [], []
    for filename, label in labels_dict.items():
        path = RECORDINGS_DIR / filename
        if path.exists():
            paths.append(path)
            y.append(1 if label == "bark" else 0)

    n_bark = sum(y)
    print(f"Found {len(paths)} clips: {n_bark} bark, {len(y) - n_bark} not_bark")
    if n_bark < 10:
        print("Too few bark examples to train. Label more clips first.")
        sys.exit(1)

    yamnet_path = MODEL_DIR / "yamnet"
    if yamnet_path.exists():
        print("Loading YAMNet from local cache...")
        yamnet = tf.saved_model.load(str(yamnet_path))
    else:
        print("Loading YAMNet from TF Hub (downloads ~25MB)...")
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        print(f"Saving YAMNet locally for deployment...")
        tf.saved_model.save(yamnet, str(yamnet_path))
        print(f"YAMNet saved to {yamnet_path}")

    print(f"Extracting YAMNet embeddings from {len(paths)} clips...")
    X, y_clean = [], []
    for i, (path, label) in enumerate(zip(paths, y), 1):
        try:
            audio = (
                AudioSegment.from_mp3(str(path))
                .set_channels(1)
                .set_frame_rate(16000)
            )
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            _, emb, _ = yamnet(samples)
            X.append(tf.reduce_mean(emb, axis=0).numpy())
            y_clean.append(label)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    X = np.array(X)
    y_arr = np.array(y_clean)
    print(f"Extracted {len(X)} embeddings.")

    # Stratified 80/20 split
    rng = np.random.default_rng(42)
    bark_idx = np.where(y_arr == 1)[0]
    not_bark_idx = np.where(y_arr == 0)[0]
    rng.shuffle(bark_idx)
    rng.shuffle(not_bark_idx)

    n_val_bark = max(1, int(len(bark_idx) * 0.2))
    n_val_nb = max(1, int(len(not_bark_idx) * 0.2))
    val_idx = np.concatenate([bark_idx[:n_val_bark], not_bark_idx[:n_val_nb]])
    train_idx = np.concatenate([bark_idx[n_val_bark:], not_bark_idx[n_val_nb:]])

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_arr[train_idx], y_arr[val_idx]
    print(f"Train: {len(X_train)} ({int(sum(y_train))} bark)  Val: {len(X_val)} ({int(sum(y_val))} bark)")

    # Class weights to compensate for imbalance
    n_total = len(y_train)
    n_b = max(1, int(sum(y_train == 1)))
    n_nb = max(1, int(sum(y_train == 0)))
    class_weight = {0: n_total / (2 * n_nb), 1: n_total / (2 * n_b)}
    print(f"Class weights: not_bark={class_weight[0]:.2f}  bark={class_weight[1]:.2f}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    tp = int(np.sum((y_pred == 1) & (y_val == 1)))
    fp = int(np.sum((y_pred == 1) & (y_val == 0)))
    fn = int(np.sum((y_pred == 0) & (y_val == 1)))
    tn = int(np.sum((y_pred == 0) & (y_val == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\nValidation — accuracy: {(tp + tn) / len(y_val):.2f}")
    print(f"Bark       — precision: {precision:.2f}  recall: {recall:.2f}  F1: {f1:.2f}")
    print(f"Confusion  — TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}")

    model.save(MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    print("Deploy with: sudo cp -r model/ /opt/doginator/model/")


if __name__ == "__main__":
    main()
