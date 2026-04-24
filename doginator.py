#!/usr/bin/env python3
"""
Doginator - Dog Bark Detection Service
Listens for dog barks, logs events to CSV, and saves MP3 recordings.
"""

import csv
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# --- Configuration ---
SAMPLE_RATE = 44100          # Hz
CHANNELS = 1                  # Mono
CHUNK_DURATION = 0.5          # Seconds per detection chunk (500ms)
BARK_THRESHOLD = 0.10         # RMS amplitude threshold (0.0–1.0); tune to your environment
MIN_BARK_DURATION = 0.5       # Minimum seconds above threshold to count as a bark
RECORD_DURATION = 3.0         # Seconds to record after bark detected
COOLDOWN = 10.0               # Seconds to wait before detecting next bark

DEBUG_RMS = os.getenv("DOGINATOR_DEBUG_RMS", "false").lower() == "true"

LOG_DIR = Path(os.getenv("DOGINATOR_LOG_DIR", "/var/log/doginator"))
RECORDINGS_DIR = Path(os.getenv("DOGINATOR_RECORDINGS_DIR", "/var/log/doginator/recordings"))
CSV_FILE = LOG_DIR / "barks.csv"
MODEL_PATH = Path(os.getenv("DOGINATOR_MODEL_PATH", "/opt/doginator/model/bark_classifier.keras"))
ML_FILTER_THRESHOLD = float(os.getenv("DOGINATOR_ML_THRESHOLD", "0.3"))

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("doginator")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "recording_file", "peak_rms"])


def log_bark(timestamp: datetime, filename: str, peak_rms: float):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            f"{peak_rms:.4f}",
        ])
    log.info(f"Logged bark: {filename} (peak RMS: {peak_rms:.4f})")


def save_mp3(audio_data: np.ndarray, timestamp: datetime) -> str:
    """Convert numpy audio array to MP3 and save. Returns filename."""
    filename = timestamp.strftime("bark_%Y%m%d_%H%M%S.mp3")
    filepath = RECORDINGS_DIR / filename

    # Convert float32 [-1, 1] to int16
    int_data = (audio_data * 32767).astype(np.int16)

    segment = AudioSegment(
        int_data.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,  # 16-bit = 2 bytes
        channels=CHANNELS,
    )
    segment.export(str(filepath), format="mp3", bitrate="128k")
    log.info(f"Saved recording: {filepath}")
    return filename


class BarkClassifier:
    """Lazy-loads YAMNet + fine-tuned classifier on first use. Silently disabled if model absent."""

    def __init__(self):
        self._yamnet = None
        self._classifier = None
        self._available = None  # None=untested, True/False=known

    def _try_load(self) -> bool:
        if self._available is not None:
            return self._available
        if not MODEL_PATH.exists():
            log.info(f"No fine-tuned model at {MODEL_PATH} — ML filtering disabled.")
            self._available = False
            return False
        try:
            import tensorflow as tf
            yamnet_path = MODEL_PATH.parent / "yamnet"
            if not yamnet_path.exists():
                log.warning(f"YAMNet model not found at {yamnet_path} — run train_model.py and redeploy.")
                self._available = False
                return False
            log.info("Loading bark classifier (first bark after startup)...")
            self._yamnet = tf.saved_model.load(str(yamnet_path))
            self._classifier = tf.keras.models.load_model(MODEL_PATH)
            log.info("Bark classifier ready.")
            self._available = True
            return True
        except Exception as e:
            log.warning(f"Could not load bark classifier: {e} — ML filtering disabled.")
            self._available = False
            return False

    def predict(self, filepath: Path) -> float | None:
        """Return bark probability [0, 1], or None if model unavailable."""
        if not self._try_load():
            return None
        try:
            import numpy as np
            import tensorflow as tf
            from pydub import AudioSegment
            audio = (
                AudioSegment.from_mp3(str(filepath))
                .set_channels(1)
                .set_frame_rate(16000)
            )
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            _, emb, _ = self._yamnet(samples)
            mean_emb = tf.reduce_mean(emb, axis=0).numpy()[np.newaxis]
            return float(self._classifier(mean_emb, training=False).numpy()[0][0])
        except Exception as e:
            log.warning(f"ML inference failed for {filepath.name}: {e}")
            return None


class BarkDetector:
    def __init__(self):
        self.running = True
        self.last_bark_time = 0.0
        self.classifier = BarkClassifier()

    def run(self):
        log.info("Doginator started. Listening for barks...")
        log.info(f"Threshold: {BARK_THRESHOLD} | Record duration: {RECORD_DURATION}s | Cooldown: {COOLDOWN}s")

        above_threshold_chunks = 0
        required_chunks = max(1, int(MIN_BARK_DURATION / CHUNK_DURATION))
        peak_rms = 0.0
        record_buffer = []

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            latency="high",
        ) as stream:
            while self.running:
                chunk, overflowed = stream.read(CHUNK_SAMPLES)
                if overflowed:
                    log.debug("Audio buffer overflowed")

                chunk = chunk.flatten()
                record_buffer.append(chunk)
                # Keep a rolling buffer of enough audio for one recording
                max_buffer = int(RECORD_DURATION / CHUNK_DURATION) + 2
                if len(record_buffer) > max_buffer:
                    record_buffer.pop(0)

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if DEBUG_RMS:
                    log.info(f"RMS={rms:.4f} (threshold={BARK_THRESHOLD})")

                if rms >= BARK_THRESHOLD:
                    above_threshold_chunks += 1
                    peak_rms = max(peak_rms, rms)
                else:
                    above_threshold_chunks = 0
                    peak_rms = 0.0

                now = datetime.now().timestamp()
                cooldown_ok = (now - self.last_bark_time) >= COOLDOWN

                if above_threshold_chunks >= required_chunks and cooldown_ok:
                    timestamp = datetime.now()
                    log.info(f"Bark detected! RMS={peak_rms:.4f}. Recording {RECORD_DURATION}s...")

                    # Collect remaining chunks to fill RECORD_DURATION
                    extra_chunks = int(RECORD_DURATION / CHUNK_DURATION) - len(record_buffer)
                    for _ in range(max(0, extra_chunks)):
                        c, _ = stream.read(CHUNK_SAMPLES)
                        record_buffer.append(c.flatten())

                    audio_data = np.concatenate(record_buffer[-int(RECORD_DURATION / CHUNK_DURATION):])
                    saved_peak = peak_rms
                    saved_ts = timestamp

                    def _process(audio=audio_data, ts=saved_ts, peak=saved_peak):
                        filename = save_mp3(audio, ts)
                        filepath = RECORDINGS_DIR / filename
                        bark_prob = self.classifier.predict(filepath)
                        if bark_prob is not None and bark_prob < ML_FILTER_THRESHOLD:
                            filepath.unlink(missing_ok=True)
                            log.info(f"ML filter rejected {filename} (bark_prob={bark_prob:.2f})")
                            return
                        if bark_prob is not None:
                            log.info(f"ML filter confirmed {filename} (bark_prob={bark_prob:.2f})")
                        log_bark(ts, filename, peak)

                    threading.Thread(target=_process, daemon=True).start()

                    self.last_bark_time = datetime.now().timestamp()
                    above_threshold_chunks = 0
                    peak_rms = 0.0
                    record_buffer.clear()

    def stop(self):
        self.running = False


def main():
    ensure_dirs()
    detector = BarkDetector()

    def handle_signal(signum, frame):
        log.info("Shutdown signal received.")
        detector.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        detector.run()
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
