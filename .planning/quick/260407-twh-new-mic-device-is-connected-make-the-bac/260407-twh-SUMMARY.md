---
id: 260407-twh
description: Backend falls back to first available input device when UMA-16v2 is not detected
date: 2026-04-07
status: complete
---

# Quick Task 260407-twh — SUMMARY

## What changed
The backend used to start with no audio capture whenever the UMA-16v2 was unplugged. With a ReSpeaker Mic Array v3.0 (Seeed VID 0x2886, PID 0x0018) now connected to the dev machine, the user wanted the service to fall back to whatever input device is present so CNN-only detection still works without the 16-channel array.

## Approach (per user decisions)
- **Fallback target:** first input device with `max_input_channels >= 1` (no name filtering — device-agnostic).
- **Reporting:** log a `WARNING`; `DeviceStatus` still reports `detected=true` (no new UI fields).
- **Channel handling:** force `channels=1` (mono) on the fallback path. The CNN classifier already mono-downmixes via `chunk.mean(axis=1)`, so 1 channel is sufficient for detection. Beamforming results from a 1-mic stream are meaningless but are demand-gated by CNN confirmation, so they don't affect anything until detection fires.

## Files
- `src/acoustic/types.py` — added `DeviceInfo.is_fallback: bool = False`.
- `src/acoustic/audio/device.py` — new `detect_audio_device()` that calls `detect_uma16v2()` first, then returns the first input device with `max_input_channels >= 1` and `is_fallback=True`. Logs `WARNING` on fallback.
- `src/acoustic/audio/__init__.py` — export `detect_audio_device`.
- `src/acoustic/main.py`:
  - Import switched from `detect_uma16v2` → `detect_audio_device`.
  - `_create_hardware_capture()` now accepts an optional `channels` override (defaults to `settings.num_channels`) so PortAudio doesn't reject a 16-ch request on a 1-ch device.
  - 3 call sites (`_initial_scan_task`, `_reconnect_loop`, `lifespan`) use `detect_audio_device()` and pass `channels=1` when `device_info.is_fallback`.
  - Lifespan logs a distinct `WARNING` line ("FALLBACK hardware capture (mono, detection-only)") so the operator can tell at a glance which device the service is running on.
- `src/acoustic/audio/monitor.py` — `DeviceMonitor` poll loop and initial detection now use `detect_audio_device()`. `DeviceStatus` will report whichever input device is currently present.
- `tests/unit/test_device.py` — added `TestDetectAudioDevice` with 3 cases:
  - prefers UMA-16v2 when present (no fallback flag)
  - falls back to first input device with `is_fallback=True`
  - returns `None` when there are no input devices at all

## Verification
- `pytest tests/unit/test_device.py -q` → **6 passed** (3 existing + 3 new).
- `python -c "from acoustic.audio import detect_audio_device, detect_uma16v2"` → imports clean.
- `ruff check` on changed files → only pre-existing E501 violations on untouched lines (148, 175 of `monitor.py`); no new lint errors introduced.
- Live device probe (`sd.query_devices()` after `sd._terminate(); sd._initialize()`):
  ```
  [0] ReSpeaker 4 Mic Array (UAC1.0)  in=6  out=2  sr=16000  [INPUT]
  [1] MacBook Pro Microphone          in=1  out=0  sr=44100  [INPUT]
  [2] MacBook Pro Speakers            in=0  out=2  sr=44100
  [3] Microsoft Teams Audio           in=1  out=1  sr=48000  [INPUT]
  ```
  `detect_audio_device()` will pick `[0] ReSpeaker` and open it as mono. Core Audio will resample its native 16 kHz to `settings.sample_rate` (48 kHz) transparently.

## Out of scope (follow-up phase suggested)
- **ReSpeaker on-board DOA + LED ring.** Confirmed via USB descriptor that the connected device is `SEEED 0x2886:0018` (XMOS-based), which exposes DOA / VAD / LED control over a vendor USB control interface. Reading DOA and driving the LED ring requires `pyusb` + `libusb`, plus a new backend module, REST/WebSocket plumbing, and a UI compass widget. This is meaningful work and should land as a dedicated phase, not bolted onto a quick task.

## Notes / caveats
- The `BandpassFilter` self-resets on channel-count mismatch (`device.py:61`), so a 1-ch chunk doesn't crash the demand-gated beamforming path. `clear_state()` calls `BandpassFilter.reset(settings.num_channels=16)`, which is harmless because `apply()` will re-reset on the next chunk.
- `BeamformingPipeline.clear_state()` still references `settings.num_channels` for the bandpass reset — fine for now (pre-existing behavior), but worth revisiting if we ever want first-class non-UMA support beyond detection-only fallback.
