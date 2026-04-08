---
phase: 21
plan: 06
subsystem: rpi-edge
tags: [gpio, audio-alarm, detection-log, signals, rotating-file-handler]
dependency_graph:
  requires: [21-04, 21-05]
  provides: [LedAlarm, AudioAlarm, DetectionLogger, assets/alert.wav]
  affects: [21-07]
tech_stack:
  added: [gpiozero==2.0.1, "logging.handlers.RotatingFileHandler"]
  patterns:
    - "gpiozero MockFactory for host-side GPIO tests"
    - "signal.SIGTERM + signal.SIGINT + atexit for multi-path cleanup"
    - "dedicated logger with propagate=False for forensic isolation"
key_files:
  created:
    - apps/rpi-edge/skyfort_edge/gpio_led.py
    - apps/rpi-edge/skyfort_edge/audio_alarm.py
    - apps/rpi-edge/skyfort_edge/detection_log.py
    - apps/rpi-edge/assets/alert.wav
  modified:
    - apps/rpi-edge/tests/test_gpio_sigterm.py
    - apps/rpi-edge/tests/test_audio_alarm_degrades.py
    - apps/rpi-edge/tests/test_detection_log.py
decisions:
  - "D-16 extension point realised as LedAlarm(gpio_pin: int) ctor parameter so future output pins add a HardwareConfig field + second LedAlarm instance (no refactor)."
  - "D-18/D-19 silent-degrade: every sounddevice/soundfile failure path is caught and logged WARNING; AudioAlarm goes to an effectively-disabled state on load failure so play() becomes a no-op."
  - "D-21/T-21-14 forensic isolation: DetectionLogger uses its own Logger('skyfort_edge.detection') with propagate=False so the D-23 general log level never silences detection records."
  - "T-21-15 disk bound: RotatingFileHandler max_bytes + backup_count (defaults 10 MB x 5) give a predictable ~50 MB ceiling."
metrics:
  tasks_completed: 2
  tests_added: 13
  files_created: 4
  files_modified: 3
---

# Phase 21 Plan 06: Output Side-Effects (LED, Audio Alarm, Detection Log) Summary

One-liner: Three reliability-critical output modules — SIGTERM-safe GPIO LED driver, silent-degrade audio alarm, and structurally-always-on rotating JSONL detection log — plus a bundled 1 kHz alert tone.

## What Shipped

### LedAlarm (`apps/rpi-edge/skyfort_edge/gpio_led.py`)
- `LedAlarm(gpio_pin: int)` builds a `gpiozero.LED` bound to the currently-active `Device.pin_factory` (so `MockFactory` works in host tests, lgpio is used on-Pi).
- `on_event(StateEvent)` turns the LED on at `RISING_EDGE` and off at `FALLING_EDGE` — min-on hold is enforced upstream by `HysteresisStateMachine`.
- `close()` drives the pin low, closes the LED (releases the factory), is idempotent.
- SIGTERM + SIGINT handlers installed via `signal.signal`, plus `atexit.register(self.close)`, giving three redundant cleanup paths. Previous signal handlers are chained (never silently dropped).
- D-16 extension point documented inline: add a new `HardwareConfig` field and instantiate a second `LedAlarm`; no refactor.

### AudioAlarm (`apps/rpi-edge/skyfort_edge/audio_alarm.py`)
- `AudioAlarm(enabled, alert_wav_path, device=None)` — disabled-by-default (D-18).
- When enabled, loads the WAV via soundfile during `__init__`. Any read failure logs `WARNING` and flips the instance back to disabled state so downstream `play()` calls are safe no-ops.
- `play()` serializes rising-edge playback via an internal `_is_playing` flag guarded by a `threading.Lock` so repeated rising-edge calls inside the same latch cycle are silently suppressed (D-18).
- `reset()` clears the latch so the next rising edge can replay.
- Any `sounddevice.play()` exception is caught and logged as `"AudioAlarm play failed (degraded silently): ..."` — never propagated to the main loop (D-19, extends T-21-04 scope).

### DetectionLogger (`apps/rpi-edge/skyfort_edge/detection_log.py`)
- `DetectionLogger(cfg)` creates `logging.getLogger("skyfort_edge.detection")`, sets `propagate = False`, and attaches a `RotatingFileHandler(path, maxBytes, backupCount)` with a custom `_JsonLineFormatter` that serialises the `extra={"payload": ...}` dict as a compact JSON line.
- `write_latch(event, class_name, score, mel_stats=None)` writes one line containing `timestamp` (ISO-8601 UTC), `event`, `class`, `score`, `latch_duration_seconds`, and optional `mel_stats`.
- `close()` flushes and detaches the handler so tests can re-instantiate cleanly.
- `DetectionLogConfig` has no `enabled`/`disabled` field — structurally always-on (D-21/D-22).

### `assets/alert.wav`
- 1 s, 48 kHz, mono, 1 kHz sine wave shaped with a Hann window (peak 0.3). Generated inline via numpy + soundfile.

## Tests (all GREEN, 13 total)

`apps/rpi-edge/tests/test_gpio_sigterm.py` (3)
- `test_sigterm_drives_led_pin_low` — rising edge turns LED on; `close()` drives it low.
- `test_sigterm_releases_gpio_factory` — `close()` idempotent.
- `test_on_event_off_on_falling_edge` — falling edge turns LED off.

`apps/rpi-edge/tests/test_audio_alarm_degrades.py` (4)
- `test_missing_audio_device_logs_warning_and_continues` — monkeypatched `sd.play` → RuntimeError is caught and logged.
- `test_playback_failure_does_not_crash_pipeline` — raising `sd.play` must not propagate.
- `test_disabled_alarm_is_noop` — `enabled=False` means `play()` silently does nothing.
- `test_play_once_per_latch_cycle` — three `play()` calls = one `sd.play` invocation; `reset()` re-arms.

`apps/rpi-edge/tests/test_detection_log.py` (6)
- `test_one_jsonl_record_per_latch` — 5 writes = 5 parseable JSON lines.
- `test_required_fields_present` — timestamp/class/score/latch_duration_seconds all present.
- `test_size_based_rotation` — `maxBytes=200` forces `detections.jsonl.1+` backups.
- `test_cannot_be_disabled_via_config` — structural dataclass field-name check.
- `test_uses_dedicated_logger_with_propagate_false` — `propagate is False`, dedicated name.
- `test_general_log_level_does_not_affect_detection_log` — root logger forced to CRITICAL, detection log still writes.

Verification commands run:
```
cd apps/rpi-edge && python -m pytest tests/test_gpio_sigterm.py tests/test_audio_alarm_degrades.py tests/test_detection_log.py -q
→ 13 passed
```

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | `7d699cf` | feat(21-06): LedAlarm + AudioAlarm with SIGTERM-safe / silent-degrade semantics |
| 2 | `4b749cc` | feat(21-06): DetectionLogger always-on rotating JSONL (D-20/D-21/D-22) |

Base: `279052e` (wave 2 tip — 21-05 complete).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed `gpiozero==2.0.1` in the host Python environment.**
- **Found during:** Task 1 pre-test smoke check.
- **Issue:** `gpiozero` was listed in `apps/rpi-edge/requirements.txt` and `pyproject.toml` but not installed in the worktree's active Python environment, so `test_gpio_sigterm.py` would have errored at import.
- **Fix:** `pip install gpiozero==2.0.1` — matches pinned version, no file changes required.
- **Commit:** N/A (environment only; not a code change).

### Minor plan deviation

**2. `REPO_ROOT` path in `test_audio_alarm_degrades.py`**
- The PLAN literal used `Path(__file__).resolve().parents[2]` to reach the `apps/rpi-edge/` directory. In this worktree the test file is already at `apps/rpi-edge/tests/test_audio_alarm_degrades.py`, so `parents[1]` is the correct anchor (`apps/rpi-edge/`). Adjusted so that `ALERT_WAV` resolves to `apps/rpi-edge/assets/alert.wav`. Functionally equivalent; no semantic change.
- No separate commit — bundled into Task 1 commit.

**3. `test_play_once_per_latch_cycle` uses the pytest `monkeypatch` fixture directly** instead of the inline `pytest.MonkeyPatch()` pattern shown in the PLAN literal. Result is identical; simpler and idiomatic.

Everything else was executed exactly as written in the PLAN.

## Threat Model Status

| Threat | Disposition | Mitigation |
|--------|-------------|------------|
| T-21-04 GPIO pin stuck high after crash | mitigate | SIGTERM + SIGINT + atexit all call `LedAlarm.close()`; `test_sigterm_drives_led_pin_low` verifies pin state. |
| T-21-14 Detection dropped via general log level | mitigate | Dedicated `Logger("skyfort_edge.detection")` with `propagate=False`; `test_general_log_level_does_not_affect_detection_log` proves it. |
| T-21-15 Detection log fills disk | mitigate | `RotatingFileHandler(maxBytes=10 MB, backupCount=5)` default; `test_size_based_rotation` proves rotation. |

No new threat flags — modules are local side-effect outputs; no new network endpoints or trust boundaries.

## Known Stubs

None. All three modules are fully wired to the hysteresis `StateEvent` interface and are ready for 21-07 to compose into the main pipeline.

## Interfaces Provided to 21-07 (pipeline wiring)

```python
# 21-07 should instantiate once at startup and cleanup on shutdown.
led = LedAlarm(gpio_pin=cfg.hardware.led_gpio_pin)
audio = AudioAlarm(
    enabled=cfg.hardware.alarm_enabled,
    alert_wav_path=Path("apps/rpi-edge/assets/alert.wav"),  # or bundled /opt path
    device=cfg.hardware.alarm_audio_device,
)
detection_log = DetectionLogger(cfg.detection_log)

# Per hysteresis event:
led.on_event(event)
if event.type == EventType.RISING_EDGE:
    audio.play()
    detection_log.write_latch(event, class_name="drone", score=event.score)
elif event.type == EventType.FALLING_EDGE:
    audio.reset()
    detection_log.write_latch(event, class_name="drone", score=event.score)
```

## Self-Check: PASSED

- `apps/rpi-edge/skyfort_edge/gpio_led.py` — FOUND
- `apps/rpi-edge/skyfort_edge/audio_alarm.py` — FOUND
- `apps/rpi-edge/skyfort_edge/detection_log.py` — FOUND
- `apps/rpi-edge/assets/alert.wav` — FOUND (48 kHz, 48000 frames)
- `apps/rpi-edge/tests/test_gpio_sigterm.py` — FOUND (GREEN, 3 tests)
- `apps/rpi-edge/tests/test_audio_alarm_degrades.py` — FOUND (GREEN, 4 tests)
- `apps/rpi-edge/tests/test_detection_log.py` — FOUND (GREEN, 6 tests)
- Commit `7d699cf` — FOUND
- Commit `4b749cc` — FOUND
- All 13 tests pass on final run.
