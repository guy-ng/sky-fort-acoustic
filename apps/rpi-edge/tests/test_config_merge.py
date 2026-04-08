"""D-09/D-10/D-11 + T-21-01/T-21-02/T-21-03: config loader behavior tests.

Covers:
- D-09: yaml.safe_load only (no yaml.load / unsafe loader)
- D-10: CLI overrides > YAML > defaults
- D-11: All four param groups (thresholds, timing, hardware, model) + detection_log + http
- T-21-01: HTTP bind must be loopback (no 0.0.0.0)
- T-21-02: detection_log.path must not contain '..'
- T-21-03: YAML loads only with safe_load
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import skyfort_edge.config as config_module
from skyfort_edge.config import (
    DEFAULT_CONFIG_PATHS,
    EdgeConfig,
    load_config,
)


REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_yaml_loads_with_safe_load_only():
    src = Path(inspect.getfile(config_module)).read_text()
    assert "yaml.safe_load" in src
    assert "yaml.load(" not in src
    assert "unsafe_load" not in src


def test_default_paths_match_research():
    paths = [str(p) for p in DEFAULT_CONFIG_PATHS]
    assert "/etc/skyfort-edge/config.yaml" in paths
    cfg = load_config(REPO_CONFIG)
    assert cfg.detection_log.path == "/var/lib/skyfort-edge/detections.jsonl"
    assert cfg.model.onnx_path.endswith("efficientat_mn10_v6_int8.onnx")


def test_cli_overrides_yaml(tmp_path):
    yaml_text = """
thresholds:
  enter_threshold: 0.6
"""
    f = tmp_path / "c.yaml"
    f.write_text(yaml_text)
    cfg = load_config(
        f, cli_overrides={"thresholds": {"enter_threshold": 0.9}}
    )
    assert cfg.thresholds.enter_threshold == 0.9


def test_unknown_top_level_key_rejected(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("zzz: 1\n")
    with pytest.raises(ValueError, match="zzz"):
        load_config(f)


def test_unknown_field_in_group_rejected(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("thresholds:\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(f)


def test_http_bind_host_must_be_loopback(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("http:\n  bind_host: 0.0.0.0\n")
    with pytest.raises(ValueError, match="loopback"):
        load_config(f)


def test_detection_log_path_rejects_dotdot(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("detection_log:\n  path: /var/lib/../etc/passwd\n")
    with pytest.raises(ValueError, match=r"\.\."):
        load_config(f)


def test_all_four_param_groups_present():
    cfg = load_config(REPO_CONFIG)
    assert hasattr(cfg, "thresholds")
    assert hasattr(cfg, "timing")
    assert hasattr(cfg, "hardware")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "detection_log")
    assert hasattr(cfg, "http")
    assert isinstance(cfg, EdgeConfig)
