"""Wave 0 RED stub — GPIO LED cleanup on SIGTERM/SIGINT.

Covers: D-15 (graceful GPIO release on shutdown).
Owner: Plan 21-04 (skyfort_edge/gpio.py + signal handlers).
"""
from __future__ import annotations

import pytest


def test_sigterm_drives_led_pin_low(mock_gpio_factory):
    pytest.fail(
        "not implemented — Plan 21-04 must install SIGTERM handler that drives the "
        "configured LED pin low before exit (verify via MockFactory pin state)"
    )


def test_sigterm_releases_gpio_factory(mock_gpio_factory):
    pytest.fail(
        "not implemented — Plan 21-04 must close gpiozero Device on SIGTERM and release "
        "the pin factory so a fresh process can re-acquire it"
    )
