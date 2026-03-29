---
phase: 01-audio-capture-beamforming-and-infrastructure
plan: 01
subsystem: audio, infra
tags: [sounddevice, numpy, pydantic-settings, docker, ring-buffer, uma16v2]

# Dependency graph
requires: []
provides:
  - AudioRingBuffer for lock-free 16-channel chunk buffering
  - AudioCapture with callback-based sounddevice.InputStream
  - detect_uma16v2 device detection via sounddevice query
  - SimulatedAudioSource generating directional 16-channel audio
  - AcousticSettings loading 17 fields from ACOUSTIC_* env vars
  - Dockerfile with python:3.11-slim, ALSA, PortAudio, HEALTHCHECK
  - Shared types (DeviceInfo, PeakDetection, HealthStatus)
  - Test fixtures (settings, mic_positions, synthetic_audio, wav_audio_fixture)
affects: [01-02-PLAN, 01-03-PLAN]

# Tech tracking
tech-stack:
  added: [fastapi, uvicorn, sounddevice, soundfile, numpy, scipy, pydantic-settings, pyzmq, pytest, pytest-asyncio, httpx, ruff]
  patterns: [pydantic-settings env config, callback-based audio capture, numpy ring buffer, TDD]

key-files:
  created:
    - pyproject.toml
    - requirements.txt
    - requirements-dev.txt
    - Dockerfile
    - .dockerignore
    - docker-compose.yml
    - src/acoustic/__init__.py
    - src/acoustic/config.py
    - src/acoustic/types.py
    - src/acoustic/audio/__init__.py
    - src/acoustic/audio/device.py
    - src/acoustic/audio/capture.py
    - src/acoustic/audio/simulator.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/unit/__init__.py
    - tests/unit/test_config.py
    - tests/unit/test_device.py
    - tests/unit/test_ring_buffer.py
    - tests/unit/test_capture.py
  modified: []

key-decisions:
  - "Ring buffer uses one-slot-reserved circular buffer pattern for full/empty disambiguation"
  - "AudioCapture callback does only np.copyto + monotonic timestamp -- no logging in audio thread"
  - "SimulatedAudioSource generates plane waves with per-mic time delays for realistic DOA testing"

patterns-established:
  - "Pydantic Settings with ACOUSTIC_ prefix for all configuration"
  - "Callback-based audio capture (not blocking sd.rec)"
  - "Pre-allocated numpy ring buffer for zero-allocation audio streaming"
  - "Shared test fixtures in conftest.py with lazy imports for cross-task dependencies"

requirements-completed: [AUD-01, AUD-02, AUD-03, INF-01, INF-03]

# Metrics
duration: 8min
completed: 2026-03-29
---

# Phase 1 Plan 01: Project Scaffolding and Audio Capture Summary

**Callback-based 16-channel audio capture pipeline with numpy ring buffer, UMA-16v2 device detection, simulated audio source, and Docker infrastructure**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-29T22:06:33Z
- **Completed:** 2026-03-29T22:14:16Z
- **Tasks:** 2
- **Files modified:** 20

## Accomplishments
- AcousticSettings with 17 env-configurable fields (ACOUSTIC_* prefix), computed chunk_samples and ring_chunks properties
- AudioRingBuffer with pre-allocated numpy array, FIFO ordering, overflow detection, and available count
- AudioCapture using sounddevice.InputStream callback that writes to ring buffer with minimal GIL work
- UMA-16v2 device detection scanning sounddevice.query_devices() (case-insensitive)
- SimulatedAudioSource generating 16-channel plane waves with per-mic time delays from configurable direction
- Docker setup with python:3.11-slim, ALSA/PortAudio deps, HEALTHCHECK, and docker-compose with USB passthrough
- 34 passing unit tests covering config, ring buffer, device detection, capture, and simulator

## Task Commits

Each task was committed atomically:

1. **Task 1: Project scaffolding, configuration, and Docker setup** - `377480a` (feat)
2. **Task 2: Audio capture pipeline** - `bc0717c` (feat)

## Files Created/Modified
- `pyproject.toml` - Project metadata, pytest and ruff config
- `requirements.txt` - Runtime dependencies (fastapi, sounddevice, numpy, etc.)
- `requirements-dev.txt` - Dev dependencies (pytest, ruff, httpx)
- `Dockerfile` - python:3.11-slim with ALSA/PortAudio and HEALTHCHECK
- `.dockerignore` - Excludes .git, POC-code, audio-data, tests
- `docker-compose.yml` - Service with USB device passthrough
- `src/acoustic/__init__.py` - Package init with version
- `src/acoustic/config.py` - AcousticSettings(BaseSettings) with 17 fields
- `src/acoustic/types.py` - DeviceInfo, PeakDetection, HealthStatus dataclasses
- `src/acoustic/audio/__init__.py` - Audio module exports
- `src/acoustic/audio/device.py` - detect_uma16v2() device scanner
- `src/acoustic/audio/capture.py` - AudioRingBuffer and AudioCapture classes
- `src/acoustic/audio/simulator.py` - build_mic_positions(), generate_simulated_chunk(), SimulatedAudioSource
- `tests/conftest.py` - Shared fixtures: settings, mic_positions, synthetic_audio, wav_audio_fixture
- `tests/unit/test_config.py` - 22 tests for AcousticSettings defaults, overrides, properties
- `tests/unit/test_device.py` - 3 tests for UMA-16v2 detection (found, not found, case-insensitive)
- `tests/unit/test_ring_buffer.py` - 5 tests for ring buffer FIFO, overflow, availability
- `tests/unit/test_capture.py` - 4 tests for AudioCapture stream creation, callback, SimulatedAudioSource shape and direction

## Decisions Made
- Ring buffer uses one-slot-reserved pattern (capacity = num_chunks - 1) for full/empty disambiguation without locks
- AudioCapture._callback does only np.copyto and time.monotonic() -- no logging or GIL-heavy operations per Pitfall 1
- generate_simulated_chunk is both a standalone function (for test fixtures) and used by SimulatedAudioSource class

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all implementations are complete and functional.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Audio capture layer complete: ring buffer, device detection, callback-based capture, simulator all tested
- Ready for Plan 01-02: SRP-PHAT beamforming engine consumes ring buffer chunks via AudioCapture.ring
- build_mic_positions() and generate_simulated_chunk() available for beamforming validation tests

## Self-Check: PASSED

All 20 files verified present. Both task commits (377480a, bc0717c) verified in git log.

---
*Phase: 01-audio-capture-beamforming-and-infrastructure*
*Completed: 2026-03-29*
