---
phase: 10-field-data-collection
plan: 01
subsystem: recording
tags: [audio, wav, soundfile, scipy, pydantic, metadata, json]

requires:
  - phase: 01-audio-capture-beamforming-and-infrastructure
    provides: AudioRingBuffer and AudioCapture for raw 16-channel audio
  - phase: 08-pytorch-training-pipeline
    provides: collect_wav_files() directory scanner contract
provides:
  - RecordingConfig with ACOUSTIC_RECORDING_ env prefix
  - RecordingMetadata Pydantic model with D-10 sidecar JSON CRUD
  - RecordingSession for mono 16kHz WAV streaming write
  - RecordingManager orchestrating record-first/label-later workflow
affects: [10-02 REST API, 10-03 frontend recording UI, training pipeline]

tech-stack:
  added: [soundfile streaming write, scipy.signal.resample_poly]
  patterns: [sidecar JSON metadata, record-first/label-later workflow, feed_chunk passive observer]

key-files:
  created:
    - src/acoustic/recording/__init__.py
    - src/acoustic/recording/config.py
    - src/acoustic/recording/metadata.py
    - src/acoustic/recording/recorder.py
    - src/acoustic/recording/manager.py
    - tests/unit/test_recorder.py
    - tests/unit/test_recording_metadata.py
    - tests/unit/test_recording_manager.py
  modified: []

key-decisions:
  - "Mono downmix via channel mean + resample_poly(up=1,down=3) for 48kHz->16kHz"
  - "Record-first workflow: files start in _unlabeled/, label_recording moves to {label}/"
  - "feed_chunk passive observer pattern: pipeline thread forwards chunks, no ring buffer contention"
  - "Sidecar JSON with Pydantic BaseModel for type-safe metadata CRUD"

patterns-established:
  - "RecordingConfig: Pydantic BaseSettings with ACOUSTIC_RECORDING_ env prefix"
  - "Sidecar JSON: wav_path.with_suffix('.json') convention for metadata"
  - "Thread-safe manager: threading.Lock for start/stop, atomic reference read for feed_chunk"

requirements-completed: [COL-01, COL-02, COL-03]

duration: 4min
completed: 2026-04-02
---

# Phase 10 Plan 01: Recording Backend Summary

**Recording engine with mono 16kHz WAV capture, sidecar JSON metadata, auto-stop, and label-based directory organization compatible with collect_wav_files()**

## What Was Built

### RecordingConfig (`src/acoustic/recording/config.py`)
Pydantic BaseSettings with `ACOUSTIC_RECORDING_` env prefix. Configures data_root (`data/field`), max_duration_s (300s), target_sample_rate (16kHz), source_sample_rate (48kHz), and top_labels (drone/background/other).

### RecordingMetadata (`src/acoustic/recording/metadata.py`)
Pydantic BaseModel with all D-10 fields: label (required), sub_label, distance_m, altitude_m, conditions, notes, recorded_at, duration_s, sample_rate, channels, original_sr, filename. Functions: write_metadata, read_metadata, update_metadata for sidecar JSON CRUD.

### RecordingSession (`src/acoustic/recording/recorder.py`)
Streaming WAV writer that accepts (samples, 16) float32 chunks, downmixes to mono via mean(axis=1), resamples 48kHz to 16kHz using scipy.signal.resample_poly(up=1, down=3), and writes via soundfile.SoundFile in streaming mode (WAV/FLOAT/1ch/16kHz). Tracks RMS level for live feedback.

### RecordingManager (`src/acoustic/recording/manager.py`)
Orchestrates the full record-first/label-later workflow:
- start_recording(): creates session in _unlabeled/ with timestamped filename
- feed_chunk(): passive observer pattern called by pipeline thread
- Auto-stop when duration >= max_duration_s
- label_recording(): validates label, moves WAV+JSON from _unlabeled/ to {label}/
- Full CRUD: list, get, update, delete recordings
- get_state(): returns status/elapsed/remaining/level_db for WebSocket

## Test Coverage

33 unit tests across 3 test files:
- `test_recorder.py` (10 tests): WAV creation, mono downmix, resample verification, duration, RMS level
- `test_recording_metadata.py` (7 tests): required fields, optional defaults, write/read roundtrip, update merge, JSON formatting
- `test_recording_manager.py` (16 tests): start/stop lifecycle, auto-stop, label workflow, CRUD, state, collect_wav_files compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functionality is fully wired.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 6c71e72 | RecordingConfig, RecordingMetadata, RecordingSession + 17 tests |
| 2 | 5a066b5 | RecordingManager with auto-stop, label workflow, CRUD + 16 tests |

## Self-Check: PASSED

All 8 files verified present. Both commits (6c71e72, 5a066b5) confirmed in git log.
