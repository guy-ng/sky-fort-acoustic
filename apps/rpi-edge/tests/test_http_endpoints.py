"""D-24 + T-21-01: loopback /health and /status endpoint tests.

Covers:
- GET /health returns 200 + JSON with ``status`` / ``model_loaded`` /
  ``audio_stream_alive`` fields and flips to ``degraded`` when the model is
  not loaded.
- GET /status returns 200 + JSON reflecting the shared RuntimeState, and
  thread-safe updates via ``RuntimeState.update`` are observable through
  the handler.
- Unknown paths return 404.
- Attempting to bind non-loopback hosts raises ValueError at the server
  constructor (defense in depth beyond config-level validation).
"""
from __future__ import annotations

import http.client
import json

import pytest

from skyfort_edge.config import HttpConfig
from skyfort_edge.http_server import LocalhostJSONServer
from skyfort_edge.runtime_state import RuntimeState


def _get(port: int, path: str) -> tuple[int, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


@pytest.fixture
def running_server():
    state = RuntimeState(
        model_loaded=True,
        audio_stream_alive=True,
        log_file_path="/tmp/x.jsonl",
        active_model_path="/tmp/model.onnx",
    )
    cfg = HttpConfig(bind_host="127.0.0.1", bind_port=18765)
    server = LocalhostJSONServer(cfg, state)
    server.start()
    try:
        yield server, state
    finally:
        server.stop()


def test_health_returns_200_json(running_server):
    status, body = _get(18765, "/health")
    assert status == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["audio_stream_alive"] is True


def test_status_returns_200_json(running_server):
    status, body = _get(18765, "/status")
    assert status == 200
    data = json.loads(body)
    # Full RuntimeState snapshot keys
    for key in (
        "model_loaded",
        "audio_stream_alive",
        "last_inference_time",
        "last_detection_time",
        "led_state",
        "log_file_path",
        "active_model_path",
    ):
        assert key in data, f"missing key {key!r} in /status body {data!r}"
    assert data["log_file_path"] == "/tmp/x.jsonl"
    assert data["active_model_path"] == "/tmp/model.onnx"


def test_unknown_path_returns_404(running_server):
    status, body = _get(18765, "/does_not_exist")
    assert status == 404
    data = json.loads(body)
    assert data["error"] == "not found"
    assert data["path"] == "/does_not_exist"


def test_binds_only_to_127_0_0_1():
    cfg = HttpConfig(bind_host="0.0.0.0", bind_port=18766)
    with pytest.raises(ValueError, match="loopback"):
        LocalhostJSONServer(cfg, RuntimeState())


def test_status_reflects_runtime_state_updates(running_server):
    server, state = running_server
    state.update(led_state="on", last_detection_time=12345.67)
    status, body = _get(18765, "/status")
    assert status == 200
    data = json.loads(body)
    assert data["led_state"] == "on"
    assert data["last_detection_time"] == pytest.approx(12345.67)


def test_health_returns_degraded_when_model_not_loaded():
    state = RuntimeState(model_loaded=False, audio_stream_alive=True)
    cfg = HttpConfig(bind_host="127.0.0.1", bind_port=18767)
    server = LocalhostJSONServer(cfg, state)
    server.start()
    try:
        status, body = _get(18767, "/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["audio_stream_alive"] is True
    finally:
        server.stop()
