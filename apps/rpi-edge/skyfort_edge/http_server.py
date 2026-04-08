"""Minimal loopback-only JSON /health + /status HTTP server (D-24 / T-21-01).

Uses the stdlib ``http.server`` module deliberately: D-28 calls out keeping
the Pi install lightweight, and this endpoint only needs two read-only JSON
routes without auth. FastAPI / uvicorn would add ~15 MB of wheels and a
startup dependency chain for zero functional benefit.

Security:
    - LocalhostJSONServer refuses any bind_host that is not one of
      ``127.0.0.1``, ``localhost``, or ``::1``. This is a second gate on top
      of the config-loader validation in ``skyfort_edge.config`` -- if a
      future refactor ever bypasses ``load_config``, the server constructor
      still fails closed (T-21-01).
    - No authentication is provided because the socket only binds to
      loopback.
    - /status exposes the detection log path and active model path; those
      are considered non-sensitive per T-21-16 (disposition: accept).
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from skyfort_edge.runtime_state import RuntimeState

log = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class HealthStatusHandler(BaseHTTPRequestHandler):
    """Request handler: GET /health, GET /status, everything else 404."""

    # Set by LocalhostJSONServer.__init__ before the server starts serving.
    state: Optional[RuntimeState] = None

    # Silence the default stderr access log; route through our logger at DEBUG.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("%s - %s", self.client_address[0], format % args)

    def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        if self.path == "/health":
            snap = self.state.snapshot() if self.state is not None else {}
            healthy = bool(snap.get("model_loaded")) and bool(
                snap.get("audio_stream_alive")
            )
            self._send_json(
                200,
                {
                    "status": "ok" if healthy else "degraded",
                    "model_loaded": bool(snap.get("model_loaded", False)),
                    "audio_stream_alive": bool(
                        snap.get("audio_stream_alive", False)
                    ),
                },
            )
            return
        if self.path == "/status":
            snap = self.state.snapshot() if self.state is not None else {}
            self._send_json(200, snap)
            return
        self._send_json(404, {"error": "not found", "path": self.path})


class LocalhostJSONServer:
    """Thin wrapper around ``http.server.HTTPServer`` running in a daemon thread."""

    def __init__(self, http_cfg: Any, state: RuntimeState) -> None:
        if http_cfg.bind_host not in LOOPBACK_HOSTS:
            raise ValueError(
                "T-21-01: bind_host must be loopback "
                f"(one of {sorted(LOOPBACK_HOSTS)}), got {http_cfg.bind_host!r}"
            )
        # Attach the shared state to the handler class. We install a fresh
        # subclass per-instance so that multiple servers in the same process
        # (e.g. in tests) do not race on a global class attribute.
        handler_cls = type(
            "BoundHealthStatusHandler",
            (HealthStatusHandler,),
            {"state": state},
        )
        self._state = state
        self._server = HTTPServer(
            (http_cfg.bind_host, int(http_cfg.bind_port)), handler_cls
        )
        self._thread: Optional[threading.Thread] = None
        self._started = False

    @property
    def server_address(self) -> tuple[str, int]:
        return self._server.server_address  # type: ignore[return-value]

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="skyfort-edge-http",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        log.info("HTTP server listening on %s", self._server.server_address)

    def stop(self) -> None:
        # HTTPServer.shutdown() blocks forever on an Event that is only set
        # by serve_forever(). If start() was never called, calling
        # _server.shutdown() deadlocks the caller. Guard with _started.
        if self._started:
            try:
                self._server.shutdown()
            except Exception:
                log.exception("HTTPServer.shutdown raised")
            self._started = False
        try:
            self._server.server_close()
        except Exception:
            log.exception("HTTPServer.server_close raised")
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
