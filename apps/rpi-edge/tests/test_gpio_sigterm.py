"""D-15: GPIO SIGTERM-safe cleanup (T-21-04 mitigation)."""
from __future__ import annotations

import pytest
from gpiozero import Device
from gpiozero.pins.mock import MockFactory

from skyfort_edge.gpio_led import LedAlarm
from skyfort_edge.hysteresis import EventType, StateEvent


@pytest.fixture(autouse=True)
def use_mock_factory():
    prev = Device.pin_factory
    Device.pin_factory = MockFactory()
    yield
    Device.pin_factory = prev


def test_sigterm_drives_led_pin_low():
    led = LedAlarm(gpio_pin=17)
    led.on_event(StateEvent(EventType.RISING_EDGE, 0.0, 0.9))
    assert led.is_on
    # Simulate SIGTERM → close()
    led.close()
    assert not led.is_on


def test_sigterm_releases_gpio_factory():
    led = LedAlarm(gpio_pin=18)
    led.close()
    # Calling close twice must be idempotent
    led.close()


def test_on_event_off_on_falling_edge():
    led = LedAlarm(gpio_pin=19)
    led.on_event(StateEvent(EventType.RISING_EDGE, 0.0, 0.9))
    assert led.is_on
    led.on_event(
        StateEvent(
            EventType.FALLING_EDGE,
            5.0,
            0.1,
            latch_duration_seconds=5.0,
        )
    )
    assert not led.is_on
    led.close()
