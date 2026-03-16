#!/usr/bin/env python3
"""
Label bark recordings as 'bark' or 'not_bark'.
Results saved to labels.csv for training.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

RECORDINGS_DIR = Path(os.getenv("DOGINATOR_RECORDINGS_DIR", "/var/log/doginator/recordings"))
LABELS_FILE = Path(__file__).parent / "labels.csv"


def load_existing_labels():
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = row["label"]
    return labels


def play(filepath):
    subprocess.run(["mpg123", "-q", "-o", "alsa", "-a", "hw:2,0", str(filepath)], check=False)


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

    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not existing:
            writer.writerow(["filename", "label"])

        for i, path in enumerate(unlabelled, 1):
            print(f"\n[{i}/{len(unlabelled)}] {path.name}")
            play(path)

            while True:
                key = input("  B=bark  N=not bark  R=replay  Q=quit: ").strip().lower()
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
