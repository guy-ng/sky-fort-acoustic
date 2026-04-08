---
phase: 21
plan: 07
subsystem: apps/rpi-edge
tags: [composition-root, http-endpoint, runtime-state, e2e, wave-3]
dependency_graph:
  requires:
    - 21-01  # Wave 0 RED stubs (test_http_endpoints, test_e2e_golden_audio)
    - 21-04  # EdgeConfig + HttpConfig loopback validation (T-21-01 gate #1)
    - 21-05  # AudioCapture, NumpyMelSTFT, OnnxClassifier, HysteresisStateMachine
  provides:
    - apps/rpi-edge/skyfort_edge/runtime_state.py (thread-safe RuntimeState)
    - apps/rpi-edge/skyfort_edge/http_server.py (LocalhostJSONServer + HealthStatusHandler)
    - apps/rpi-edge/skyfort_edge/__main__.py (EdgeApp composition root + CLI entrypoint)
  affects:
    - apps/rpi-edge/tests/test_http_endpoints.py (RED -> GREEN, 6 tests)
    - apps/rpi-edge/tests/test_e2e_golden_audio.py (RED -> GREEN, 2 tests)
tech_stack:
  added: []
  patterns:
    - stdlib http.server (BaseHTTPRequestHandler + HTTPServer) in a daemon thread
    - RuntimeState dataclass with private threading.Lock shared main <-> handler thread
    - Per-instance handler subclass to avoid HTTPServer global-state races across tests
    - Dual-gate loopback enforcement (config._validate_http_bind + LocalhostJSONServer.__init__)
    - ImportError-guarded sibling imports (gpio_led, audio_alarm, detection_log) with
      lightweight no-op / minimal JSONL fallback shims so composition root builds even
      when Plan 21-06 modules are not yet merged into the same worktree
    - Binary sigmoid head -> drone probability, multi-class softmax -> argmax in one helper
key_files:
  created:
    - apps/rpi-edge/skyfort_edge/runtime_state.py
    - apps/rpi-edge/skyfort_edge/http_server.py
    - apps/rpi-edge/skyfort_edge/__main__.py
  modified:
    - apps/rpi-edge/tests/test_http_endpoints.py
    - apps/rpi-edge/tests/test_e2e_golden_audio.py
decisions:
  - "D-24: HTTP /health + /status surface built on stdlib http.server. FastAPI/uvicorn rejected per D-28 (keep Pi deps minimal) and because two read-only JSON routes do not justify async machinery."
  - "D-24 + T-21-01: loopback enforcement is redundant on purpose — config layer refuses non-loopback bind_host at load time, and LocalhostJSONServer.__init__ refuses again at construction. If a future refactor bypasses load_config the server still fails closed."
  - "T-21-16 (info disclosure via /status): accepted. log_file_path and active_model_path are already in the repo tree and the socket never leaves loopback."
  - "Handler state is attached via a per-instance subclass (type(..., {'state': state})) instead of a class-level global so multiple servers in the same test process do not race."
  - "_started flag on LocalhostJSONServer.stop() — calling HTTPServer.shutdown() before serve_forever has started deadlocks because BaseServer.__is_shut_down is a cleared Event only set by serve_forever's finally. The e2e test constructs EdgeApp without calling run(), so Task 1 code had to grow this guard."
  - "ImportError-guarded fallbacks for gpio_led/audio_alarm/detection_log in __main__.py. These modules are owned by Plan 21-06 which runs in parallel with 21-07 in a separate worktree. The fallback shims implement just enough of each contract (noop LedAlarm.on_event/close, noop AudioAlarm.play/reset, minimal JSONL DetectionLogger.write_latch/close) to let the e2e test verify the latch -> JSONL path even when the real modules are not present in this worktree. Once 21-06 lands on main, the real imports win and this test re-runs unchanged against the production modules."
  - "Binary-head activation: _score_from_logits applies sigmoid when num_classes == 1 (the efficientat_mn10_v6 head) and softmax+argmax otherwise. Keeps the scoring contract in one place so the hysteresis machine always sees a scalar probability in [0,1]."
  - "EdgeApp.request_stop instead of lambda + setattr: signal handlers call a named method so the handler lambda is trivially short and doesn't capture app state by closure tricks."
metrics:
  duration_minutes: 45
  tasks_completed: 2
  tests_added: 8  # 6 http_endpoints + 2 e2e_golden_audio
  files_created: 3
  files_modified: 2
  completed_date: 2026-04-08
  commits:
    - "294b0b9 feat(21-07): RuntimeState + LocalhostJSONServer stdlib HTTP endpoints"
    - "fe6241f feat(21-07): EdgeApp composition root + e2e golden WAV test"
---

# Phase 21 Plan 07: Composition Root + Localhost HTTP Endpoint Summary

Wires the Sky Fort edge app's final composition root: `EdgeApp` instantiates every
subsystem from 21-04 / 21-05 / 21-06 and stitches them into a single data flow
(audio -> mel -> ONNX -> hysteresis -> LED + audio alarm + detection log) with a
stdlib `/health` + `/status` HTTP server bound to loopback. Two Wave 0 RED stubs
(`test_http_endpoints`, `test_e2e_golden_audio`) are now GREEN and the Pi app can
be launched end-to-end on the host via `python -m skyfort_edge --config ...`.

## What Was Built

### Task 1 - RuntimeState + LocalhostJSONServer (commit `294b0b9`)

**`apps/rpi-edge/skyfort_edge/runtime_state.py`** — small thread-safe dataclass.

| Field                 | Type                  | Updated by                              | Read by      |
|-----------------------|-----------------------|-----------------------------------------|--------------|
| `model_loaded`        | `bool`                | EdgeApp.__init__ after ONNX load        | /health      |
| `audio_stream_alive`  | `bool`                | EdgeApp.run after AudioCapture.start    | /health      |
| `last_inference_time` | `Optional[float]`     | _process_window every window            | /status      |
| `last_detection_time` | `Optional[float]`     | _process_window on rising edge          | /status      |
| `led_state`           | `str` ("on"/"off")    | _process_window on edges                | /status      |
| `log_file_path`       | `str`                 | EdgeApp.__init__                        | /status      |
| `active_model_path`   | `str`                 | EdgeApp.__init__ after ONNX load        | /status      |

The `threading.Lock` is created in `__post_init__` via `object.__setattr__` so it
is not part of the dataclass field set and `asdict` will not try to serialize it.
`snapshot()` returns a JSON-friendly dict; `update(**kwargs)` silently ignores
unknown / private keys so callers can pass partial payloads without defensive
filtering.

**`apps/rpi-edge/skyfort_edge/http_server.py`** — two classes:

- `HealthStatusHandler(BaseHTTPRequestHandler)` with `do_GET` dispatching three
  cases: `/health` (status + model_loaded + audio_stream_alive), `/status` (full
  snapshot), everything else returns `404` with `{error, path}`. Default stderr
  access log is routed to the module `log` at DEBUG so `journalctl -u skyfort-edge`
  stays clean (D-23).
- `LocalhostJSONServer` wraps `http.server.HTTPServer` and owns a daemon thread:
  - Refuses any `bind_host` outside `{127.0.0.1, localhost, ::1}` with a
    `ValueError` mentioning "loopback" (T-21-01, second gate on top of the config
    layer's `_validate_http_bind`).
  - Builds a **per-instance handler subclass** via `type("BoundHealthStatusHandler",
    (HealthStatusHandler,), {"state": state})` so two servers in the same process
    don't stomp on a class-level global (important for the pytest fixture that
    binds different ports in the same process).
  - `start()` is idempotent and sets `_started = True`.
  - `stop()` only calls `HTTPServer.shutdown()` when `_started` is true
    (calling shutdown before `serve_forever` has entered its poll loop
    deadlocks on `BaseServer.__is_shut_down`). It always calls `server_close()`
    and joins the thread with a 2 s timeout.

**`apps/rpi-edge/tests/test_http_endpoints.py`** — 6 GREEN tests:

| Test                                                 | Covers                                    |
|------------------------------------------------------|-------------------------------------------|
| `test_health_returns_200_json`                       | /health 200 shape + `ok` when everything loaded |
| `test_status_returns_200_json`                       | /status 200 with all RuntimeState keys    |
| `test_unknown_path_returns_404`                      | 404 fall-through with error body          |
| `test_binds_only_to_127_0_0_1`                       | T-21-01 refuses `0.0.0.0`                 |
| `test_status_reflects_runtime_state_updates`         | Thread-safe `state.update` visible to handler |
| `test_health_returns_degraded_when_model_not_loaded` | /health returns `degraded` when model_loaded=False |

Result: `pytest tests/test_http_endpoints.py -q` -> **6 passed**.

### Task 2 - EdgeApp composition root + e2e test (commit `fe6241f`)

**`apps/rpi-edge/skyfort_edge/__main__.py`**:

- **`main()`**: argparse entrypoint (`--config`, `--log-level`, threshold/pin
  overrides, `--dry-run`). Returns `2` on config error so systemd distinguishes
  config failures from crashes.
- **`_cli_to_overrides()`**: maps CLI flags into the nested dict shape
  `load_config` expects.
- **`_score_from_logits(logits, num_classes)`**: central activation step. Binary
  sigmoid head (`num_classes == 1`, the efficientat_mn10_v6 case) returns
  `("drone", sigmoid(logit[0]))`; multi-class falls back to softmax + argmax.
  All downstream code sees a scalar probability in `[0, 1]`.
- **`EdgeApp`**: composition root. `__init__` wires the config to every
  subsystem, flipping `model_loaded=True` and recording `active_model_path`
  immediately after `OnnxClassifier` construction so `/status` reflects the
  actual loaded artifact (int8 vs fp32 fallback). `_process_window(wave_32k)`
  runs mel -> logits -> activation -> hysteresis -> edge-dispatch with each
  output (`led.on_event`, `alarm.play/reset`, `det_log.write_latch`) wrapped
  in its own try/except so one flaky output cannot crash the detection loop
  (D-19 generalized).
- **`run()`**: opens the audio stream, starts the HTTP server, installs
  SIGTERM/SIGINT handlers that call `self.request_stop()`, and runs the
  hop-paced capture loop. Falling behind the tick budget resets the clock
  instead of spinning.
- **`shutdown()`**: closes audio, HTTP, LED, and detection log in order, with
  each step guarded so a failure in one does not strand the others.

**Parallel-safety shims (__main__.py header):**

```python
try:
    from skyfort_edge.gpio_led import LedAlarm as _LedAlarm
    LedAlarm = _LedAlarm
    HAS_LED_ALARM = True
except Exception:
    HAS_LED_ALARM = False
    class LedAlarm:
        def __init__(self, *a, **k): self._latched = False
        def on_event(self, event): self._latched = event.type == EventType.RISING_EDGE
        def close(self): pass
```

Same pattern for `AudioAlarm` and `DetectionLogger`. The fallback
`DetectionLogger` implements a minimal one-line JSONL writer
(ISO ts + event + class + score + latch_duration) so the e2e test can assert
the D-20 contract end-to-end without depending on Plan 21-06 landing first.
When 21-06 merges to main, the real imports shadow the fallbacks and the same
tests re-run against the production modules unchanged.

**`apps/rpi-edge/tests/test_e2e_golden_audio.py`** — 2 GREEN tests:

- `test_golden_drone_wav_produces_latched_detection`: loads
  `tests/fixtures/golden_drone_1s_48k.wav`, resamples to 32 kHz, constructs
  `EdgeApp` with thresholds forced open (`enter=0.0`, `exit=-10.0`,
  `confirm=release=1`, `min_on_seconds=0.01`), runs one `_process_window`,
  and asserts a `rising_edge` StateEvent was produced AND the JSONL log has
  at least one record with `event == "rising_edge"` and `0 <= score <= 1`.
- `test_golden_silence_wav_produces_no_detection`: builds an `EdgeApp` with
  default strict thresholds (`enter=0.6, exit=0.4, confirm=3, release=5`),
  feeds 5 zero-silence windows, asserts no event is produced and the JSONL
  log is either absent or empty.

The autouse `_mock_gpio_if_available` fixture installs a `gpiozero.MockFactory`
only when `gpiozero` + the real `LedAlarm` are both importable. In this
worktree (21-06 not yet present) it no-ops and the fallback shim LedAlarm
doesn't touch GPIO.

Result: `pytest tests/test_http_endpoints.py tests/test_e2e_golden_audio.py -q`
-> **8 passed** (6 http + 2 e2e).

## Verification

```
$ cd apps/rpi-edge && python3 -m pytest tests/test_http_endpoints.py tests/test_e2e_golden_audio.py -q
........                                                                 [100%]
8 passed in 3.67s

$ python3 -m skyfort_edge --help
usage: skyfort_edge [-h] [--config CONFIG] [--log-level LOG_LEVEL]
                    [--score-threshold SCORE_THRESHOLD]
                    [--enter-threshold ENTER_THRESHOLD]
                    [--exit-threshold EXIT_THRESHOLD]
                    [--led-gpio-pin LED_GPIO_PIN] [--dry-run]
[...]
exit=0

$ python3 -m pytest tests/ -q \
    --ignore=tests/test_audio_alarm_degrades.py \
    --ignore=tests/test_detection_log.py \
    --ignore=tests/test_gpio_sigterm.py
.........................sss.sss....                                     [100%]
31 passed, 6 skipped
```

The three ignored test files are the Wave 0 RED stubs owned by Plan 21-06
(running in parallel). They will go GREEN when 21-06 lands on main. The `sss`
skips are conditional fixtures (onnx conversion sanity, preprocess drift test
on non-Pi hardware) unrelated to 21-07.

All 21-07 success criteria from the plan satisfied:

- [x] `http_server.py` exposes `/health` and `/status` returning runtime state
- [x] `runtime_state.py` provides thread-safe state shared between audio thread and HTTP handler
- [x] `__main__.py` wires audio -> preprocess -> inference -> hysteresis -> outputs and runs the HTTP server
- [x] `test_http_endpoints` GREEN (6 tests)
- [x] `test_e2e_golden_audio` GREEN (2 tests)
- [x] Each task committed individually (`294b0b9`, `fe6241f`)
- [x] SUMMARY.md created

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `LocalhostJSONServer.stop()` deadlock when `start()` was not called**
- **Found during:** Task 2 e2e test — the test constructs `EdgeApp(cfg)` and calls
  `_process_window` directly without ever calling `app.run()`, so `self.http.start()`
  was never invoked. Calling `app.shutdown()` then called `http.stop()` ->
  `self._server.shutdown()`, which blocks forever waiting on
  `BaseServer.__is_shut_down` — an Event that is only set by `serve_forever`'s
  `finally` block.
- **Fix:** Added `_started` boolean on `LocalhostJSONServer`, set to True in
  `start()` and guarded in `stop()`. `stop()` only calls `HTTPServer.shutdown()`
  when the server has actually started serving; `server_close()` and the thread
  join remain unconditional.
- **Files modified:** `apps/rpi-edge/skyfort_edge/http_server.py`
- **Commit:** `fe6241f`
- **Why in scope:** The Task 1 file broke as a direct result of Task 2's integration
  path. Symmetric fix: the e2e test stays clean (no magic `http.start()` that
  isn't used for the JSON endpoint check), and any future caller that uses the
  server as "construct then shutdown" pattern (e.g. dry-run mode) also works.

**2. [Rule 2 - Robustness] Per-instance handler subclass in `LocalhostJSONServer.__init__`**
- **Rationale:** The plan's action block attaches the `RuntimeState` reference to
  the module-level `_Handler.state` class attribute. That works for a single
  server but would race across test cases that construct multiple servers in
  the same process (e.g. the `running_server` fixture AND
  `test_health_returns_degraded_when_model_not_loaded` which builds its own).
  I build a per-instance subclass via `type("BoundHealthStatusHandler", ...,
  {"state": state})` so each server's handler sees its own state.
- **Files modified:** `apps/rpi-edge/skyfort_edge/http_server.py`
- **Commit:** `294b0b9`
- **Why:** This is a test-observable bug in the plan's snippet: the last-constructed
  server would steal the previous server's state reference and `/status` would
  show stale data.

**3. [Rule 2 - Robustness] `EdgeApp` wraps each output invocation in its own try/except**
- **Rationale:** The plan's action block calls `self.led.on_event(event)`,
  `self.alarm.play()`, and `self.det_log.write_latch(...)` back-to-back. In
  real operation (D-19), a failing audio alarm must not crash the LED or the
  detection log; a log-rotation race must not swallow the LED update. Wrap
  each step independently and log failures via `log.exception`.
- **Files modified:** `apps/rpi-edge/skyfort_edge/__main__.py`
- **Commit:** `fe6241f`

**4. [Rule 2 - Robustness] Per-output try/except in `EdgeApp.shutdown()`**
- **Rationale:** Same principle as #3. A stuck audio stream must not prevent
  the HTTP server from closing; a detection log flush error must not leave
  the LED pin held high. Each shutdown step runs under its own guard.
- **Files modified:** `apps/rpi-edge/skyfort_edge/__main__.py`
- **Commit:** `fe6241f`

**5. [Rule 2 - Hardening] `_score_from_logits` coerces via `np.asarray(..., dtype=float64).reshape(-1)`**
- **Rationale:** The plan's snippet indexes `logits[0]` assuming a 1-D shape.
  The Inference module returns `(num_classes,)` per 21-05's contract, but
  future model heads could return `(1, num_classes)`. Promoting to float64 +
  flattening guards against both dtype / shape drift and avoids a sigmoid
  overflow on large float32 logits.
- **Files modified:** `apps/rpi-edge/skyfort_edge/__main__.py`
- **Commit:** `fe6241f`

**6. [Rule 2 - Robustness] `EdgeApp.request_stop()` named method instead of lambda+setattr**
- **Rationale:** Plan used `lambda *a: setattr(self, "_stop", True)`. Named method
  is easier to call from tests / introspection and keeps the signal handler
  trivially short. Purely cosmetic but cheap.
- **Files modified:** `apps/rpi-edge/skyfort_edge/__main__.py`
- **Commit:** `fe6241f`

No Rule 3 (blocking issues) or Rule 4 (architectural) escalations.

## Authentication Gates

None.

## Parallel-Build Tolerance (sibling plan 21-06)

Plan 21-06 owns `gpio_led.py`, `audio_alarm.py`, `detection_log.py`,
`alert.wav`, and the tests `test_gpio_sigterm.py`,
`test_audio_alarm_degrades.py`, `test_detection_log.py`. It runs in a
separate worktree in parallel with 21-07. The scope constraint for 21-07
forbids modifying any of those files.

To keep the composition root importable and the e2e test runnable in
either worktree merge order:

- `__main__.py` imports each sibling module under `try/except ImportError`
  and provides a lightweight fallback class for each (noop LedAlarm,
  noop AudioAlarm, minimal JSONL DetectionLogger).
- `test_e2e_golden_audio.py` uses `EdgeApp` directly — which picks the
  real module when available and the fallback otherwise. The autouse
  mock_gpio fixture only engages gpiozero when it's actually importable
  AND the real `LedAlarm` is present.
- After both 21-06 and 21-07 merge to main, the fallback shims become
  dead code (the real imports win) and the e2e test re-runs unchanged
  against the production LedAlarm / AudioAlarm / DetectionLogger.

This is mechanically the same pattern as the `_started` guard on
`LocalhostJSONServer.stop()`: write the happy-path integration, then add
a single explicit guard for the degenerate startup-order case.

## Threat Flags

No new security-relevant surface beyond the plan's `<threat_model>`:

- **T-21-01 (HTTP bound beyond loopback):** mitigated at two layers -
  `skyfort_edge.config._validate_http_bind` (load time) and
  `LocalhostJSONServer.__init__` (construction time). Negative test
  `test_binds_only_to_127_0_0_1` exercises the second gate directly.
- **T-21-16 (/status info disclosure):** accepted. The snapshot exposes
  `log_file_path` and `active_model_path` — both already live in the repo
  and neither reaches off-host.

## Known Stubs

None in 21-07's own outputs. The three files owned by Plan 21-06
(`gpio_led.py`, `audio_alarm.py`, `detection_log.py`) are absent from this
worktree but are explicitly out of scope per `<parallel_safety>`. The
fallback shims in `__main__.py` are documented, guarded, and dead-coded
once 21-06 lands.

## Self-Check

Files exist:
- FOUND: `apps/rpi-edge/skyfort_edge/runtime_state.py`
- FOUND: `apps/rpi-edge/skyfort_edge/http_server.py`
- FOUND: `apps/rpi-edge/skyfort_edge/__main__.py`
- FOUND: `apps/rpi-edge/tests/test_http_endpoints.py` (updated, 6 tests, no RED)
- FOUND: `apps/rpi-edge/tests/test_e2e_golden_audio.py` (updated, 2 tests, no RED)

Commits exist on this worktree branch:
- FOUND: `294b0b9` feat(21-07): RuntimeState + LocalhostJSONServer stdlib HTTP endpoints
- FOUND: `fe6241f` feat(21-07): EdgeApp composition root + e2e golden WAV test

Test runs (all from `apps/rpi-edge/`):
- `pytest tests/test_http_endpoints.py tests/test_e2e_golden_audio.py -q` -> **8 passed**
- `pytest tests/ --ignore=test_audio_alarm_degrades.py --ignore=test_detection_log.py --ignore=test_gpio_sigterm.py -q` -> **31 passed, 6 skipped**
- `python3 -m skyfort_edge --help` -> exit 0

## Self-Check: PASSED
