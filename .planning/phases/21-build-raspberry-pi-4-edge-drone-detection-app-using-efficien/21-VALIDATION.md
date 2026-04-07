---
phase: 21
slug: build-raspberry-pi-4-edge-drone-detection-app-using-efficien
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 21-RESEARCH.md "Validation Architecture" section.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (existing repo infra) |
| **Config file** | `pyproject.toml` / `pytest.ini` (existing) |
| **Quick run command** | `pytest apps/rpi-edge/tests -x -q` |
| **Full suite command** | `pytest apps/rpi-edge/tests tests/edge -q` |
| **Estimated runtime** | ~30 seconds (host-side; no Pi required) |

---

## Sampling Rate

- **After every task commit:** Run `pytest apps/rpi-edge/tests -x -q`
- **After every plan wave:** Run `pytest apps/rpi-edge/tests tests/edge -q`
- **Before `/gsd-verify-work`:** Full suite must be green + manual on-device smoke test (see Manual-Only)
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

> Filled by gsd-planner when creating PLAN.md files. Each task must reference one of the test files listed in Wave 0.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 0 | CLS-01 | — | Wave 0 stubs installed | infra | `pytest apps/rpi-edge/tests -x -q --collect-only` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Wave 0 must install these test stubs (red tests — they fail until implementation lands):

- [ ] `apps/rpi-edge/tests/test_preprocess_parity.py` — golden-parity test: vendored `preprocess.py` output vs training-side `src/acoustic/classification/efficientat/preprocess.py` on a fixed audio fixture (assert np.allclose within 1e-5). Covers D-04.
- [ ] `apps/rpi-edge/tests/test_preprocess_drift.py` — byte-identity test for `preprocess.py` and `mel_banks_128_1024_32k.pt` between `apps/rpi-edge/` and `src/acoustic/classification/efficientat/`. Covers D-04 (CI drift guard).
- [ ] `apps/rpi-edge/tests/test_onnx_conversion_sanity.py` — runs `scripts/convert_efficientat_to_onnx.py` output against PyTorch reference on held-out samples; asserts top-1 agreement ≥ 95% and mean logit delta < 0.05. Covers D-06, D-07, D-08.
- [ ] `apps/rpi-edge/tests/test_hysteresis.py` — state-machine test for enter/exit thresholds + confirm/release hit counts; asserts rising-edge latch, min-on latch, release after cooldown. Covers D-12, D-14.
- [ ] `apps/rpi-edge/tests/test_gpio_sigterm.py` — uses `gpiozero.pins.mock.MockFactory` to assert LED pin is driven low on SIGTERM/SIGINT and released. Covers D-15.
- [ ] `apps/rpi-edge/tests/test_audio_alarm_degrades.py` — simulates missing audio device; asserts detection pipeline continues and a warning is logged. Covers D-19.
- [ ] `apps/rpi-edge/tests/test_detection_log.py` — rotating JSONL logger writes one record per latch, rotates at size limit, and cannot be disabled via config (assert AttributeError / no-op on disable attempt). Covers D-20, D-21, D-22.
- [ ] `apps/rpi-edge/tests/test_config_merge.py` — YAML file + CLI override merge precedence, unknown-key rejection, per-group validation. Covers D-09, D-10, D-11.
- [ ] `apps/rpi-edge/tests/test_http_endpoints.py` — /health + /status return JSON, bind only to 127.0.0.1, return 404 for other paths. Covers D-24.
- [ ] `apps/rpi-edge/tests/test_resample_48_to_32.py` — `scipy.signal.resample_poly(2, 3)` 48→32 kHz resample correctness + latency budget (<50 ms for 1 s window on host). Covers D-02.
- [ ] `apps/rpi-edge/tests/test_e2e_golden_audio.py` — end-to-end: golden drone WAV → pipeline → expected latched detection with mocked GPIO. Covers D-03 through D-20 integration.
- [ ] `apps/rpi-edge/tests/test_inference_latency_host.py` — ONNX session + MN10 int8 inference on 1 s mel input; asserts <150 ms host-side (proxy for Pi latency budget). Covers D-05, D-07.
- [ ] `apps/rpi-edge/tests/conftest.py` — shared fixtures: golden audio, mock GPIO factory, temp config dir, tmp JSONL log path.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| onnxruntime 1.18.1 loads on RPi 4 (Cortex-A72) | D-05 | Requires physical Pi 4 hardware — wheel architecture and illegal-instruction risk cannot be verified on x86 host | `scripts/install_edge_rpi.sh` on Pi → `python -c "import onnxruntime; print(onnxruntime.__version__)"` → run conversion sanity script on Pi |
| USB mic capture + LED blink on real hardware | D-01, D-13, D-15 | Needs physical mic + LED wiring | Wire LED to configured GPIO pin → play drone audio near USB mic → observe LED latches on detection → Ctrl-C → verify LED releases |
| systemd unit starts on boot, restarts on failure, logs to journald | D-26 | Requires systemd-enabled Pi | `systemctl enable --now skyfort-edge && systemctl restart skyfort-edge && journalctl -u skyfort-edge -f` → kill process, verify restart |
| End-to-end latency budget (~2 inferences/sec, no audio drops) | D-03 | Needs real audio stream under load | Run for 5+ minutes on Pi, monitor journald for "audio callback overrun" warnings → must be zero |
| Audio alarm plays once per latch cycle (if enabled) | D-17, D-18 | Needs actual speaker | Enable in config → trigger detection → verify alert.wav plays once, does not replay until release+re-latch |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
