"""Wave 0 RED stub — local HTTP /health and /status endpoints.

Covers: D-24 (loopback-only HTTP introspection).
Owner: Plan 21-04 (skyfort_edge/http_server.py).
"""
from __future__ import annotations

import pytest


def test_health_returns_200_json():
    pytest.fail(
        "not implemented — Plan 21-04: GET /health on the loopback HTTP server must "
        "return 200 + JSON {status: 'ok', ...}"
    )


def test_status_returns_200_json():
    pytest.fail(
        "not implemented — Plan 21-04: GET /status must return 200 + JSON with current "
        "hysteresis state, model_version, last_detection_ts"
    )


def test_binds_only_to_127_0_0_1():
    pytest.fail(
        "not implemented — Plan 21-04: HTTP server must bind to 127.0.0.1 only "
        "(connecting from a non-loopback addr must be refused)"
    )


def test_unknown_path_returns_404():
    pytest.fail(
        "not implemented — Plan 21-04: GET /does-not-exist must return 404"
    )
