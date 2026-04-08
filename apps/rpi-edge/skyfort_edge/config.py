"""Edge app config: YAML + CLI overrides, no hot reload (D-09/D-10/D-11).

Security: D-09 uses yaml.safe_load ONLY (T-21-03 mitigation). Never use the
non-safe YAML loaders — the safe loader rejects arbitrary Python tags.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ThresholdsConfig:
    score_threshold: float = 0.5
    enter_threshold: float = 0.6
    exit_threshold: float = 0.4
    confirm_hits: int = 3
    release_hits: int = 5
    class_allowlist: list[str] = field(default_factory=list)
    per_class_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class TimingConfig:
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    smoothing_window: int = 3
    ema_alpha: float = 0.0  # 0.0 = disabled
    post_release_cooldown_seconds: float = 1.0


@dataclass
class HardwareConfig:
    input_device: Optional[str] = None  # None = default USB mic
    capture_sample_rate: int = 48000
    led_gpio_pin: int = 17
    min_on_seconds: float = 2.0
    alarm_audio_device: Optional[str] = None
    alarm_enabled: bool = False


@dataclass
class ModelConfig:
    onnx_path: str = "/opt/skyfort-edge/models/efficientat_mn10_v6_int8.onnx"
    fallback_onnx_path: str = "/opt/skyfort-edge/models/efficientat_mn10_v6_fp32.onnx"
    prefer_int8: bool = True
    num_threads: int = 2
    execution_provider: str = "CPUExecutionProvider"
    log_level: str = "INFO"


@dataclass
class DetectionLogConfig:
    path: str = "/var/lib/skyfort-edge/detections.jsonl"
    rotate_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    rotate_backup_count: int = 5


@dataclass
class HttpConfig:
    bind_host: str = "127.0.0.1"  # T-21-01: localhost only, never 0.0.0.0
    bind_port: int = 8088


@dataclass
class EdgeConfig:
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection_log: DetectionLogConfig = field(default_factory=DetectionLogConfig)
    http: HttpConfig = field(default_factory=HttpConfig)


_GROUP_CLASSES = {
    "thresholds": ThresholdsConfig,
    "timing": TimingConfig,
    "hardware": HardwareConfig,
    "model": ModelConfig,
    "detection_log": DetectionLogConfig,
    "http": HttpConfig,
}


DEFAULT_CONFIG_PATHS = [
    Path("/etc/skyfort-edge/config.yaml"),
    Path("apps/rpi-edge/config.yaml"),
]


def _validate_http_bind(cfg: HttpConfig) -> None:
    # T-21-01: refuse non-loopback binds
    if cfg.bind_host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"http.bind_host must be loopback (127.0.0.1/localhost/::1); got {cfg.bind_host!r}. "
            f"Non-loopback binds are rejected to prevent remote exposure."
        )


def _validate_detection_log_path(cfg: DetectionLogConfig) -> None:
    # T-21-02: reject path traversal, require absolute path
    parts = Path(cfg.path).parts
    if ".." in parts:
        raise ValueError(f"detection_log.path must not contain '..': {cfg.path!r}")
    if not Path(cfg.path).is_absolute():
        raise ValueError(f"detection_log.path must be absolute: {cfg.path!r}")


def _merge_group(group_cls, yaml_dict: dict[str, Any], cli_overrides: dict[str, Any]):
    allowed = {f.name for f in fields(group_cls)}
    unknown = set(yaml_dict.keys()) - allowed
    if unknown:
        raise ValueError(
            f"unknown keys in {group_cls.__name__}: {sorted(unknown)}"
        )
    merged = {**yaml_dict}
    for k, v in cli_overrides.items():
        if k in allowed:
            merged[k] = v
    return group_cls(**merged)


def load_config(
    yaml_path: Optional[Path] = None,
    cli_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> EdgeConfig:
    """Load config with defaults -> YAML -> CLI precedence.

    Args:
        yaml_path: optional explicit path. If None, tries DEFAULT_CONFIG_PATHS in order.
        cli_overrides: nested dict {group_name: {field: value}} from argparse.

    Returns:
        EdgeConfig with all groups populated.

    Raises:
        ValueError: on unknown keys, non-loopback http bind, invalid log path.
        FileNotFoundError: if yaml_path is explicitly given and does not exist.
    """
    cli_overrides = cli_overrides or {}

    yaml_data: dict[str, Any] = {}
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"config file not found: {yaml_path}")
        with yaml_path.open("r") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                with candidate.open("r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                break

    if not isinstance(yaml_data, dict):
        raise ValueError(
            f"config YAML top-level must be a mapping, got {type(yaml_data).__name__}"
        )

    unknown_groups = set(yaml_data.keys()) - set(_GROUP_CLASSES.keys())
    if unknown_groups:
        raise ValueError(
            f"unknown top-level config groups: {sorted(unknown_groups)}"
        )

    groups = {}
    for name, cls in _GROUP_CLASSES.items():
        groups[name] = _merge_group(
            cls,
            yaml_data.get(name, {}) or {},
            cli_overrides.get(name, {}),
        )

    cfg = EdgeConfig(**groups)
    _validate_http_bind(cfg.http)
    _validate_detection_log_path(cfg.detection_log)
    return cfg
