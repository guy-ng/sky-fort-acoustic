"""D-13/D-14/D-15/D-16: Single-LED alarm with SIGTERM-safe cleanup.

# D-16: structured for future output pins — add new pin as HardwareConfig field + LedAlarm param.
# The gpio_pin parameter on __init__ is the extension point: instantiate additional LedAlarm
# objects (or subclass) with new HardwareConfig fields. No refactor required.

Uses gpiozero with the default lgpio backend (Bookworm-native). Tests use
gpiozero.pins.mock.MockFactory and override Device.pin_factory.
"""
from __future__ import annotations

import atexit
import logging
import signal

from skyfort_edge.hysteresis import EventType, StateEvent

log = logging.getLogger(__name__)


class LedAlarm:
    """LED output driver with rising-edge latch + SIGTERM-safe cleanup.

    D-16 extension pattern: to add more output pins (e.g. a strobe or buzzer
    relay), add a new `HardwareConfig` field and instantiate a second
    `LedAlarm` (or a subclass) with it. The `gpio_pin` ctor parameter is the
    extension point — no refactor required.
    """

    def __init__(self, gpio_pin: int) -> None:
        from gpiozero import LED, Device

        self._pin = gpio_pin
        # Capture the currently active pin factory so tests that installed a
        # MockFactory keep working; on the Pi this will be the lgpio backend.
        self._factory = Device.pin_factory
        self._led = LED(gpio_pin, pin_factory=self._factory)
        self._closed = False
        self._install_signal_handlers()
        atexit.register(self.close)
        log.info(
            "LedAlarm initialized on GPIO pin %d (factory=%s)",
            gpio_pin,
            type(self._factory).__name__,
        )

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)

                def handler(signum, frame, _prev=prev):
                    log.info("Received signal %s — releasing GPIO", signum)
                    self.close()
                    if callable(_prev) and _prev not in (
                        signal.SIG_DFL,
                        signal.SIG_IGN,
                    ):
                        _prev(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
            except (ValueError, OSError) as e:
                log.warning("could not install %s handler: %s", sig, e)

    def on_event(self, event: StateEvent) -> None:
        if self._closed:
            return
        if event.type == EventType.RISING_EDGE:
            self._led.on()
            log.info("LED ON (rising edge, score=%.3f)", event.score)
        elif event.type == EventType.FALLING_EDGE:
            self._led.off()
            log.info(
                "LED OFF (falling edge, latch_duration=%.2fs)",
                event.latch_duration_seconds,
            )

    @property
    def is_on(self) -> bool:
        try:
            return bool(self._led.value)
        except Exception:
            return False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._led.off()
            self._led.close()
        except Exception as e:
            log.warning("LED close error: %s", e)
        self._closed = True
        log.info("LedAlarm closed; GPIO pin %d released", self._pin)
