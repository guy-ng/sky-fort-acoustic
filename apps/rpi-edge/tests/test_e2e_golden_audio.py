"""End-to-end pipeline test on the golden WAV fixtures (D-03/D-12/D-14/D-20).

Feeds the golden 1s@48kHz drone WAV through the full composition root
(`EdgeApp`) with the audio capture stubbed to replay the fixture, and
asserts:

- A rising-edge latch is produced and appended to the JSONL detection log.
- A silent window does not produce any latch.

Parallel-build note:
    Plan 21-06 owns gpio_led / audio_alarm / detection_log. If those
    sibling modules are not yet present in this worktree, `EdgeApp` falls
    back to lightweight no-op shims (see apps/rpi-edge/skyfort_edge/__main__.py)
    which still honor the output contracts exercised here. When the real
    modules land on main, this same test re-runs against them unchanged.

    The mock_gpio_factory fixture is only installed when gpiozero is
    importable AND real LedAlarm is present, so the test does not silently
    skip on a bare-Python CI box.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"
INT8 = REPO_ROOT / "models" / "efficientat_mn10_v6_int8.onnx"
FP32 = REPO_ROOT / "models" / "efficientat_mn10_v6_fp32.onnx"


@pytest.fixture(autouse=True)
def _mock_gpio_if_available() -> Iterator[None]:
    """If gpiozero + real LedAlarm are importable, install a MockFactory.

    Otherwise fall through: EdgeApp's fallback LedAlarm shim doesn't touch
    hardware and does not need a mock factory.
    """
    try:
        import gpiozero  # type: ignore
        from gpiozero.pins.mock import MockFactory  # type: ignore
    except Exception:
        yield
        return
    prev = gpiozero.Device.pin_factory
    gpiozero.Device.pin_factory = MockFactory()
    try:
        yield
    finally:
        gpiozero.Device.pin_factory = prev


def _write_yaml(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "config.yaml"
    f.write_text(body)
    return f


def _require_onnx_artifacts() -> None:
    if not INT8.exists() or not FP32.exists():
        pytest.skip(
            "ONNX artifacts missing under models/ — run Plan 21-03 conversion first"
        )


def test_golden_drone_wav_produces_latched_detection(tmp_path: Path) -> None:
    _require_onnx_artifacts()
    jsonl_path = tmp_path / "detections.jsonl"
    cfg_path = _write_yaml(
        tmp_path,
        f"""
thresholds:
  enter_threshold: 0.0
  exit_threshold: -10.0
  confirm_hits: 1
  release_hits: 1
timing:
  window_seconds: 1.0
  hop_seconds: 0.5
hardware:
  led_gpio_pin: 17
  min_on_seconds: 0.01
  alarm_enabled: false
model:
  onnx_path: {INT8}
  fallback_onnx_path: {FP32}
  prefer_int8: true
  num_threads: 1
detection_log:
  path: {jsonl_path}
  rotate_max_bytes: 1048576
  rotate_backup_count: 2
http:
  bind_host: 127.0.0.1
  bind_port: 18799
""",
    )
    from skyfort_edge.__main__ import EdgeApp
    from skyfort_edge.config import load_config

    cfg = load_config(cfg_path)

    wav, sr = sf.read(FIXTURES / "golden_drone_1s_48k.wav", dtype="float32")
    assert sr == 48000, f"golden fixture must be 48 kHz, got {sr}"
    wave_32k = resample_poly(wav, up=2, down=3).astype(np.float32)

    app = EdgeApp(cfg)
    try:
        event = app._process_window(wave_32k)
    finally:
        app.shutdown()

    # Thresholds are forced open, so this window MUST produce a rising-edge latch.
    assert event is not None, "expected a state event on the golden drone window"
    assert event.type.value == "rising_edge"

    assert jsonl_path.exists(), f"detection log not written at {jsonl_path}"
    lines = [line for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(lines) >= 1, "expected at least one JSONL record"
    record = json.loads(lines[0])
    assert record["event"] == "rising_edge"
    assert "score" in record
    assert 0.0 <= float(record["score"]) <= 1.0


def test_golden_silence_wav_produces_no_detection(tmp_path: Path) -> None:
    _require_onnx_artifacts()
    jsonl_path = tmp_path / "detections.jsonl"
    cfg_path = _write_yaml(
        tmp_path,
        f"""
thresholds:
  enter_threshold: 0.6
  exit_threshold: 0.4
  confirm_hits: 3
  release_hits: 5
timing:
  window_seconds: 1.0
  hop_seconds: 0.5
hardware:
  led_gpio_pin: 18
  min_on_seconds: 0.1
  alarm_enabled: false
model:
  onnx_path: {INT8}
  fallback_onnx_path: {FP32}
  prefer_int8: true
  num_threads: 1
detection_log:
  path: {jsonl_path}
  rotate_max_bytes: 1048576
  rotate_backup_count: 2
http:
  bind_host: 127.0.0.1
  bind_port: 18798
""",
    )
    from skyfort_edge.__main__ import EdgeApp
    from skyfort_edge.config import load_config

    cfg = load_config(cfg_path)

    silence_32k = np.zeros(32000, dtype=np.float32)

    app = EdgeApp(cfg)
    try:
        for _ in range(5):
            event = app._process_window(silence_32k)
            assert event is None, f"silence should never latch, got {event}"
    finally:
        app.shutdown()

    # With strict defaults, silence must not produce any JSONL records.
    if jsonl_path.exists():
        body = jsonl_path.read_text().strip()
        assert body == "", f"silence unexpectedly wrote detections: {body!r}"
