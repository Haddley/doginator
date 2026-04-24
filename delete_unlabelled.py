#!/usr/bin/env python3
"""
Delete recordings that have not yet been labelled.
"""

import csv
import os
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


def main():
    recordings = sorted(RECORDINGS_DIR.glob("*.mp3"))
    if not recordings:
        print(f"No recordings found in {RECORDINGS_DIR}")
        sys.exit(0)

    existing = load_existing_labels()
    unlabelled = [r for r in recordings if r.name not in existing]

    if not unlabelled:
        print("No unlabelled recordings to delete.")
        sys.exit(0)

    print(f"{len(unlabelled)} unlabelled recording(s) will be deleted:")
    for r in unlabelled:
        print(f"  {r.name}")

    answer = input("\nDelete these files? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)

    for r in unlabelled:
        r.unlink()
        print(f"Deleted {r.name}")

    print(f"\nDone. {len(unlabelled)} file(s) deleted.")


if __name__ == "__main__":
    main()
