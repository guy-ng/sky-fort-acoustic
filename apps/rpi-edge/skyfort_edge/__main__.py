"""Sky Fort Edge main entrypoint (composition root).

Usage:
    python -m skyfort_edge --config /etc/skyfort-edge/config.yaml [--log-level INFO]

Architecture (composition root, D-01..D-28):
    AudioCapture (48k) -> read_window_32k (scipy.signal.resample_poly 2/3)
        -> NumpyMelSTFT (128xT) -> OnnxClassifier (int8-preferred, FP32 fallback)
        -> sigmoid on logit[0] -> HysteresisStateMachine
        -> on rising/falling edge: LedAlarm + AudioAlarm + DetectionLogger
    LocalhostJSONServer exposes /health + /status on 127.0.0.1 in parallel.

Parallel-build tolerance:
    gpio_led, audio_alarm, and detection_log are delivered by Plan 21-06,
    which runs in parallel with this plan (21-07). When those modules are
    not yet available in a given worktree, lightweight no-op stand-ins keep
    the composition root importable and let Plans 21-04/05/07 integration
    tests run. Once 21-06 lands on main, the real implementations replace
    the stand-ins via normal imports.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from skyfort_edge.audio import AudioCapture
from skyfort_edge.config import EdgeConfig, load_config
from skyfort_edge.http_server import LocalhostJSONServer
from skyfort_edge.hysteresis import EventType, HysteresisStateMachine, StateEvent
from skyfort_edge.inference import OnnxClassifier
from skyfort_edge.preprocess import NumpyMelSTFT
from skyfort_edge.runtime_state import RuntimeState

log = logging.getLogger("skyfort_edge")

ALERT_WAV_PATH = Path(__file__).parent.parent / "assets" / "alert.wav"


# ---------------------------------------------------------------------------
# Sibling-plan imports with no-op fallbacks (see module docstring).
# ---------------------------------------------------------------------------

try:
    from skyfort_edge.gpio_led import LedAlarm as _LedAlarm  # type: ignore

    LedAlarm: Any = _LedAlarm
    HAS_LED_ALARM = True
except Exception:  # pragma: no cover - exercised only when 21-06 missing
    HAS_LED_ALARM = False

    class LedAlarm:  # type: ignore[no-redef]
        """Fallback: no-op LED alarm used when gpio_led is not yet available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._latched = False

        def on_event(self, event: StateEvent) -> None:
            self._latched = event.type == EventType.RISING_EDGE

        def close(self) -> None:
            pass


try:
    from skyfort_edge.audio_alarm import AudioAlarm as _AudioAlarm  # type: ignore

    AudioAlarm: Any = _AudioAlarm
    HAS_AUDIO_ALARM = True
except Exception:  # pragma: no cover - exercised only when 21-06 missing
    HAS_AUDIO_ALARM = False

    class AudioAlarm:  # type: ignore[no-redef]
        """Fallback: no-op audio alarm used when audio_alarm is not yet available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def play(self) -> None:
            pass

        def reset(self) -> None:
            pass


try:
    from skyfort_edge.detection_log import DetectionLogger as _DetectionLogger  # type: ignore

    DetectionLogger: Any = _DetectionLogger
    HAS_DETECTION_LOG = True
except Exception:  # pragma: no cover - exercised only when 21-06 missing
    import json
    from datetime import datetime, timezone

    HAS_DETECTION_LOG = False

    class DetectionLogger:  # type: ignore[no-redef]
        """Fallback JSONL writer: writes one latched-event record per line.

        This is a minimal shim so the e2e integration test can verify the
        log contract (D-20) even when the real rotating logger from 21-06
        is not available in the current worktree. The real implementation
        adds size-based rotation and richer fields.
        """

        def __init__(self, cfg: Any) -> None:
            self.path = Path(cfg.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", buffering=1)

        def write_latch(
            self,
            event: StateEvent,
            class_name: str = "drone",
            score: float = 0.0,
        ) -> None:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event.type.value,
                "class": class_name,
                "score": float(score),
                "latch_duration_seconds": float(event.latch_duration_seconds),
            }
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

        def close(self) -> None:
            try:
                self._fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI parsing / overrides
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="skyfort_edge")
    p.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    p.add_argument(
        "--log-level",
        default=None,
        help="Override model.log_level (DEBUG/INFO/WARNING/ERROR)",
    )
    p.add_argument("--score-threshold", type=float, default=None)
    p.add_argument("--enter-threshold", type=float, default=None)
    p.add_argument("--exit-threshold", type=float, default=None)
    p.add_argument("--led-gpio-pin", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct everything and exit without running the capture loop",
    )
    return p.parse_args(argv)


def _cli_to_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    o: dict[str, dict[str, Any]] = {}
    if args.score_threshold is not None:
        o.setdefault("thresholds", {})["score_threshold"] = args.score_threshold
    if args.enter_threshold is not None:
        o.setdefault("thresholds", {})["enter_threshold"] = args.enter_threshold
    if args.exit_threshold is not None:
        o.setdefault("thresholds", {})["exit_threshold"] = args.exit_threshold
    if args.led_gpio_pin is not None:
        o.setdefault("hardware", {})["led_gpio_pin"] = args.led_gpio_pin
    if args.log_level is not None:
        o.setdefault("model", {})["log_level"] = args.log_level
    return o


def _setup_general_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _score_from_logits(
    logits: np.ndarray, num_classes: Optional[int]
) -> tuple[str, float]:
    """Convert raw logits to (class_name, probability).

    Binary sigmoid head (efficientat_mn10_v6, num_classes=1): apply sigmoid
    to logit[0] to get the drone probability. Multi-class heads: softmax +
    argmax. Falls back to sigmoid when ``num_classes`` is None (dynamic
    output shape) but the logits vector has length 1.
    """
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if (num_classes == 1) or arr.shape[0] == 1:
        sig = 1.0 / (1.0 + float(np.exp(-arr[0])))
        return ("drone", float(sig))
    shifted = arr - float(np.max(arr))
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return (f"class_{idx}", float(probs[idx]))


# ---------------------------------------------------------------------------
# EdgeApp composition root
# ---------------------------------------------------------------------------


class EdgeApp:
    """Wires config -> audio -> mel -> onnx -> hysteresis -> outputs + http."""

    def __init__(self, cfg: EdgeConfig) -> None:
        self.cfg = cfg
        self.state = RuntimeState(log_file_path=cfg.detection_log.path)
        self.preprocess = NumpyMelSTFT()
        self.classifier = OnnxClassifier(cfg.model)
        self.state.update(
            model_loaded=True,
            active_model_path=str(self.classifier.active_model_path),
        )
        self.hysteresis = HysteresisStateMachine(
            enter_threshold=cfg.thresholds.enter_threshold,
            exit_threshold=cfg.thresholds.exit_threshold,
            confirm_hits=cfg.thresholds.confirm_hits,
            release_hits=cfg.thresholds.release_hits,
            min_on_seconds=cfg.hardware.min_on_seconds,
        )
        self.led = LedAlarm(gpio_pin=cfg.hardware.led_gpio_pin)
        self.alarm = AudioAlarm(
            enabled=cfg.hardware.alarm_enabled,
            alert_wav_path=ALERT_WAV_PATH,
            device=cfg.hardware.alarm_audio_device,
        )
        self.det_log = DetectionLogger(cfg.detection_log)
        self.http = LocalhostJSONServer(cfg.http, self.state)
        self.audio: Optional[AudioCapture] = None
        self._stop = False

    def _process_window(self, wave_32k: np.ndarray) -> Optional[StateEvent]:
        mel = self.preprocess.forward(wave_32k)
        logits = self.classifier.classify(mel)
        class_name, score = _score_from_logits(logits, self.classifier.num_classes)
        now = time.time()
        self.state.update(last_inference_time=now)
        event = self.hysteresis.update(score, now)
        if event is None:
            return None
        if event.type == EventType.RISING_EDGE:
            try:
                self.led.on_event(event)
            except Exception:
                log.exception("LED on_event failed")
            try:
                self.alarm.play()
            except Exception:
                log.exception("audio alarm play failed")
            try:
                self.det_log.write_latch(event, class_name=class_name, score=score)
            except Exception:
                log.exception("detection log write_latch failed")
            self.state.update(last_detection_time=now, led_state="on")
        elif event.type == EventType.FALLING_EDGE:
            try:
                self.led.on_event(event)
            except Exception:
                log.exception("LED on_event failed (falling)")
            try:
                self.alarm.reset()
            except Exception:
                log.exception("audio alarm reset failed")
            try:
                self.det_log.write_latch(event, class_name=class_name, score=score)
            except Exception:
                log.exception("detection log write_latch (falling) failed")
            self.state.update(led_state="off")
        return event

    def run(self) -> None:
        self.audio = AudioCapture(device=self.cfg.hardware.input_device)
        self.audio.start()
        self.state.update(audio_stream_alive=True)
        self.http.start()

        signal.signal(signal.SIGTERM, lambda *a: self.request_stop())
        signal.signal(signal.SIGINT, lambda *a: self.request_stop())

        hop = self.cfg.timing.hop_seconds
        window = self.cfg.timing.window_seconds
        next_tick = time.time()
        log.info(
            "EdgeApp running: window=%.2fs hop=%.2fs model=%s",
            window,
            hop,
            self.classifier.active_model_path,
        )
        while not self._stop:
            try:
                wave = self.audio.read_window_32k(window)
                self._process_window(wave)
            except Exception as exc:
                log.exception("process_window error: %s", exc)
            next_tick += hop
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # We're falling behind; reset the tick so we don't spin.
                next_tick = time.time()
        self.shutdown()

    def request_stop(self) -> None:
        self._stop = True

    def shutdown(self) -> None:
        log.info("EdgeApp shutting down")
        for closer_name, closer in (
            ("audio", lambda: self.audio.stop() if self.audio is not None else None),
            ("http", self.http.stop),
            ("led", self.led.close),
            ("det_log", self.det_log.close),
        ):
            try:
                closer()
            except Exception:
                log.exception("shutdown step %s failed", closer_name)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(yaml_path=args.config, cli_overrides=_cli_to_overrides(args))
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    _setup_general_logging(cfg.model.log_level)
    app = EdgeApp(cfg)
    if args.dry_run:
        app.shutdown()
        return 0
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
