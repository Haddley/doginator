#!/usr/bin/env python3
"""
Label bark recordings as 'bark' or 'not_bark'.
Results saved to labels.csv for training.

Scoring priority:
  1. Fine-tuned bark classifier (model/bark_classifier.keras) if available
  2. YAMNet bark class score as fallback
  3. Chronological order if neither is available

When the fine-tuned model confidence is ≥90% or ≤10%, a label is suggested
and you can confirm with Enter instead of typing B or N.

Install ML dependencies with: pip install tensorflow tensorflow-hub
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

RECORDINGS_DIR = Path(os.getenv("DOGINATOR_RECORDINGS_DIR", "/var/log/doginator/recordings"))
LABELS_FILE = Path(__file__).parent / "labels.csv"
MODEL_PATH = Path(__file__).parent / "model" / "bark_classifier.keras"

AUTO_SUGGEST_BARK = 0.90
AUTO_SUGGEST_NOT_BARK = 0.10


def load_existing_labels():
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            for row in csv.DictReader(f):
                if row["filename"] != "filename":  # skip duplicate header rows
                    labels[row["filename"]] = row["label"]
    return labels


def play(filepath):
    subprocess.run(["mpg123", "-q", "-o", "alsa", "-a", "hw:2,0", str(filepath)], check=False)


def _load_tf_deps():
    try:
        import numpy as np
        import tensorflow as tf
        from pydub import AudioSegment
        return np, tf, AudioSegment
    except ImportError:
        return None


def score_with_finetuned_model(paths):
    """Score using fine-tuned classifier. Returns {filename: bark_prob} or None if unavailable."""
    if not MODEL_PATH.exists():
        return None
    yamnet_path = MODEL_PATH.parent / "yamnet"
    if not yamnet_path.exists():
        print("YAMNet model not found — run train_model.py first.")
        return None
    libs = _load_tf_deps()
    if not libs:
        print("tensorflow not installed — cannot use fine-tuned model.")
        return None
    np, tf, AudioSegment = libs

    print("Loading fine-tuned bark classifier...")
    try:
        yamnet = tf.saved_model.load(str(yamnet_path))
        classifier = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    scores = {}
    print(f"Scoring {len(paths)} clips...", flush=True)
    for i, path in enumerate(paths, 1):
        try:
            audio = (
                AudioSegment.from_mp3(str(path))
                .set_channels(1)
                .set_frame_rate(16000)
            )
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            _, emb, _ = yamnet(samples)
            mean_emb = tf.reduce_mean(emb, axis=0).numpy()[np.newaxis]
            scores[path.name] = float(classifier(mean_emb, training=False).numpy()[0][0])
        except Exception:
            scores[path.name] = 0.5
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    print("Done scoring.\n")
    return scores


def score_with_yamnet(paths):
    """Fallback: score with raw YAMNet bark class. Returns {} if unavailable."""
    libs = _load_tf_deps()
    if not libs:
        print("tensorflow not installed — skipping YAMNet scoring.\n")
        return {}
    np, tf, AudioSegment = libs

    try:
        import tensorflow_hub as hub
    except ImportError:
        print("tensorflow-hub not installed — skipping YAMNet scoring.\n")
        return {}

    print("Loading YAMNet model (downloads ~25MB on first run)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode()
    bark_idx = None
    with open(class_map_path) as f:
        for row in csv.DictReader(f):
            if row["display_name"] == "Bark":
                bark_idx = int(row["index"])
                break

    if bark_idx is None:
        print("Could not find 'Bark' class in YAMNet map — skipping scoring.")
        return {}

    scores = {}
    print(f"Scoring {len(paths)} clips with YAMNet...", flush=True)
    for i, path in enumerate(paths, 1):
        try:
            audio = (
                AudioSegment.from_mp3(str(path))
                .set_channels(1)
                .set_frame_rate(16000)
            )
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            yamnet_scores, _, _ = model(samples)
            scores[path.name] = float(tf.reduce_mean(yamnet_scores[:, bark_idx]))
        except Exception:
            scores[path.name] = 0.0
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    print("Done scoring.\n")
    return scores


def main():
    recordings = sorted(RECORDINGS_DIR.glob("*.mp3"))
    if not recordings:
        print(f"No recordings found in {RECORDINGS_DIR}")
        sys.exit(0)

    existing = load_existing_labels()
    unlabelled = [r for r in recordings if r.name not in existing]

    print(f"{len(recordings)} total recordings, {len(unlabelled)} unlabelled.")
    if not unlabelled:
        print("All labelled!")
        sys.exit(0)

    using_finetuned = False
    scores = score_with_finetuned_model(unlabelled)
    if scores is not None:
        using_finetuned = True
        unlabelled.sort(key=lambda p: scores.get(p.name, 0.5), reverse=True)
        print(f"Clips sorted by fine-tuned model confidence (highest first).")
        print(f"Confidence ≥{AUTO_SUGGEST_BARK:.0%} or ≤{AUTO_SUGGEST_NOT_BARK:.0%}: Enter to confirm suggestion.\n")
    else:
        scores = score_with_yamnet(unlabelled)
        if scores:
            unlabelled.sort(key=lambda p: scores.get(p.name, 0.0), reverse=True)
            top_score = scores.get(unlabelled[0].name, 0.0)
            print(f"Using YAMNet (no fine-tuned model found). Top score: {top_score:.3f}")
            print("Tip: scores below ~0.05 are unlikely to be barks — press Q to stop early.\n")
        else:
            print("Presenting clips in chronological order.\n")

    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(["filename", "label"])

        for i, path in enumerate(unlabelled, 1):
            score = scores.get(path.name) if scores else None

            if using_finetuned and score is not None:
                score_str = f"  Model: {score:.0%} bark"
            elif score is not None:
                score_str = f"  YAMNet bark: {score:.3f}"
            else:
                score_str = ""

            print(f"\n[{i}/{len(unlabelled)}] {path.name}{score_str}")
            play(path)

            if using_finetuned and score is not None and score >= AUTO_SUGGEST_BARK:
                suggestion = "b"
                print(f"  → Suggested: BARK")
            elif using_finetuned and score is not None and score <= AUTO_SUGGEST_NOT_BARK:
                suggestion = "n"
                print(f"  → Suggested: NOT BARK")
            else:
                suggestion = None

            while True:
                if suggestion:
                    prompt = "  B=bark  N=not bark  R=replay  Q=quit  Enter=confirm: "
                else:
                    prompt = "  B=bark  N=not bark  R=replay  Q=quit: "
                key = input(prompt).strip().lower()
                if key == "" and suggestion:
                    key = suggestion
                if key == "b":
                    writer.writerow([path.name, "bark"])
                    f.flush()
                    break
                elif key == "n":
                    writer.writerow([path.name, "not_bark"])
                    f.flush()
                    break
                elif key == "r":
                    play(path)
                elif key == "q":
                    print("Saved progress. Run again to continue.")
                    sys.exit(0)

    print(f"\nDone! Labels saved to {LABELS_FILE}")


if __name__ == "__main__":
    main()
