# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Service

```bash
# Run locally
pip install -r requirements.txt
python3 doginator.py

# Deploy and restart as systemd service
sudo cp doginator.py /opt/doginator/doginator.py
sudo systemctl restart doginator

# View live logs
sudo journalctl -u doginator -f

# Enable debug mode (logs RMS value for every 100ms audio chunk)
DOGINATOR_DEBUG_RMS=true python3 doginator.py
```

## Architecture

Single-file Python service (`doginator.py`) that detects dog barks via continuous audio monitoring.

**Detection pipeline:**
1. `sounddevice.InputStream` feeds 100ms audio chunks via callback into a `queue.Queue`
2. Main loop in `BarkDetector.run()` dequeues chunks, computes RMS, and accumulates time above threshold
3. When sustained sound exceeds `MIN_BARK_DURATION` (0.3s) with no active cooldown, a bark is confirmed
4. `record_clip()` returns the buffered 3-second window, `save_mp3()` encodes it, `log_bark()` appends a CSV row

**Key configuration constants** (lines 21–35, overridable via environment variables):
- `BARK_THRESHOLD` (0.11) — primary tuning knob; raise if false positives, lower if misses
- `COOLDOWN` (10.0s) — minimum gap between detections
- `LOG_DIR` / `RECORDINGS_DIR` — via `DOGINATOR_LOG_DIR` / `DOGINATOR_RECORDINGS_DIR`

**Outputs:**
- CSV log: `$LOG_DIR/barks.csv` — columns: `timestamp, recording_file, peak_rms`
- MP3 clips: `$RECORDINGS_DIR/bark_YYYYMMDD_HHMMSS.mp3` — 3s at 128 kbps
