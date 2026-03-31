# Quick Task 260331-myc: Fix Device Disconnect/Reconnect

**Status:** Complete
**Date:** 2026-03-31
**Commits:** 2f7e609, 6cb524d, 2b25331

## What Changed

### Task 1: Harden AudioCapture.stop() + Pipeline helpers
- `src/acoustic/audio/capture.py` — `stop()` wrapped in try/except for `sd.PortAudioError` (device may already be gone)
- `src/acoustic/pipeline.py` — Added `clear_state()` (nulls latest_map/peak) and `restart(ring_buffer)` (stop + clear + start with new buffer)

### Task 2: Make WS handlers resilient to pipeline swap
- `src/acoustic/api/websocket.py` — `ws_heatmap` and `ws_targets` re-fetch `app.state.pipeline` each iteration instead of caching at connection start. Inner try/except catches transient errors during pipeline swap.

### Task 3: Device lifecycle background task
- `src/acoustic/main.py` — Added `_device_lifecycle_task()` async background task that subscribes to DeviceMonitor events:
  - **On disconnect:** stops old AudioCapture, clears pipeline state, sets capture to None
  - **On reconnect:** calls `detect_uma16v2()` for new device index, creates new AudioCapture, restarts pipeline with new ring buffer
- Added `_create_hardware_capture()` helper to DRY up capture creation
- Health endpoint handles `capture=None` during disconnect window
- Lifecycle task started in `lifespan()`, cancelled on shutdown

## Root Cause
AudioCapture and BeamformingPipeline were created once at startup with no mechanism to recover from USB device removal. The sounddevice.InputStream dies on unplug, and nothing recreated it on replug.
