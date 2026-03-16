#!/usr/bin/env python3
"""
Doginator - Dog Bark Detection Service
Listens for dog barks, logs events to CSV, and saves MP3 recordings.
"""

import csv
import logging
import os
import queue
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
CHUNK_DURATION = 0.2          # Seconds per detection chunk (200ms)
BARK_THRESHOLD = 0.15         # RMS amplitude threshold (0.0–1.0); tune to your environment
MIN_BARK_DURATION = 0.3       # Minimum seconds above threshold to count as a bark
RECORD_DURATION = 3.0         # Seconds to record after bark detected
COOLDOWN = 10.0               # Seconds to wait before detecting next bark

DEBUG_RMS = os.getenv("DOGINATOR_DEBUG_RMS", "false").lower() == "true"

LOG_DIR = Path(os.getenv("DOGINATOR_LOG_DIR", "/var/log/doginator"))
RECORDINGS_DIR = Path(os.getenv("DOGINATOR_RECORDINGS_DIR", "/var/log/doginator/recordings"))
CSV_FILE = LOG_DIR / "barks.csv"

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


class BarkDetector:
    def __init__(self):
        self.audio_queue: queue.Queue = queue.Queue()
        self.running = True
        self.last_bark_time = 0.0
        self._lock = threading.Lock()

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            log.warning(f"Audio stream status: {status}")
        self.audio_queue.put(indata.copy())

    def record_clip(self) -> np.ndarray:
        """Record RECORD_DURATION seconds of audio from the stream."""
        num_chunks = int(RECORD_DURATION / CHUNK_DURATION)
        chunks = []
        for _ in range(num_chunks):
            try:
                chunk = self.audio_queue.get(timeout=2.0)
                chunks.append(chunk)
            except queue.Empty:
                break
        return np.concatenate(chunks, axis=0).flatten() if chunks else np.array([])

    def run(self):
        log.info("Doginator started. Listening for barks...")
        log.info(f"Threshold: {BARK_THRESHOLD} | Record duration: {RECORD_DURATION}s | Cooldown: {COOLDOWN}s")

        above_threshold_chunks = 0
        required_chunks = int(MIN_BARK_DURATION / CHUNK_DURATION)
        in_bark = False
        peak_rms = 0.0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=self.audio_callback,
        ):
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

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

                if (
                    not in_bark
                    and above_threshold_chunks >= required_chunks
                    and cooldown_ok
                ):
                    in_bark = True
                    timestamp = datetime.now()
                    log.info(f"Bark detected! RMS={peak_rms:.4f}. Recording {RECORD_DURATION}s...")

                    # Drain the queue and record a fresh clip
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break

                    audio_data = self.record_clip()

                    if audio_data.size > 0:
                        saved_peak = peak_rms
                        saved_ts = timestamp
                        saved_audio = audio_data
                        threading.Thread(
                            target=lambda: log_bark(saved_ts, save_mp3(saved_audio, saved_ts), saved_peak),
                            daemon=True,
                        ).start()

                    self.last_bark_time = datetime.now().timestamp()
                    in_bark = False
                    above_threshold_chunks = 0
                    peak_rms = 0.0

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
