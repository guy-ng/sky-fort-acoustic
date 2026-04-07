---
id: 260407-twh
description: Backend falls back to first available input device when UMA-16v2 is not detected
mode: quick
---

# Quick Task 260407-twh — Fallback mic device selection

## Problem
`detect_uma16v2()` returns `None` when no UMA-16v2 is plugged in, leaving the backend with no audio device. A different mic is now connected and the backend should use it as a fallback so the pipeline can still start.

## Decisions (from user)
- **Fallback target:** first input device with `max_input_channels >= 1`
- **Reporting:** log warning, mark `DeviceStatus.detected=true` (no new fields)

## Approach
1. Add `detect_audio_device()` to `src/acoustic/audio/device.py`:
   - Calls `detect_uma16v2()` first.
   - If `None`, iterates `sd.query_devices()` and returns the first device with `max_input_channels >= 1` as a `DeviceInfo` (using its actual channel count).
   - Logs a `WARNING` when falling back.
   - Returns `None` only if no input device exists at all.
2. Update `_create_hardware_capture()` in `src/acoustic/main.py` to accept a `channels` argument (defaulted to `settings.num_channels`) and pass `device_info.channels` from the callers — required because sounddevice will refuse to open a 16-channel stream on a 2-channel device.
3. Replace the three `detect_uma16v2()` call sites in `main.py` with `detect_audio_device()` and pass `device_info.channels` into `_create_hardware_capture`.
4. Replace the two `detect_uma16v2()` call sites in `src/acoustic/audio/monitor.py` with `detect_audio_device()`.
5. Export `detect_audio_device` from `src/acoustic/audio/__init__.py`.
6. Add tests in `tests/unit/test_device.py`:
   - `test_detect_audio_device_prefers_uma16` — UMA-16 in list → returns UMA-16.
   - `test_detect_audio_device_fallback` — no UMA-16, list has 2-ch device → returns DeviceInfo with channels=2.
   - `test_detect_audio_device_none` — no input devices at all → returns None.

## Files
- `src/acoustic/audio/device.py` (add function)
- `src/acoustic/audio/__init__.py` (export)
- `src/acoustic/main.py` (signature + 3 call sites)
- `src/acoustic/audio/monitor.py` (2 call sites)
- `tests/unit/test_device.py` (3 new tests)

## Verify
- `pytest tests/unit/test_device.py -q` passes.
- `python -c "from acoustic.audio import detect_audio_device"` succeeds.
- `ruff check src/acoustic/audio/device.py src/acoustic/main.py src/acoustic/audio/monitor.py` clean.
