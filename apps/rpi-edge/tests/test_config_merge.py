"""Wave 0 RED stub — config loader merge precedence and validation.

Covers: D-09 (YAML safe_load), D-10 (CLI overrides YAML), D-11 (per_class_thresholds + strict schema).
Owner: Plan 21-04 (skyfort_edge/config.py).
"""
from __future__ import annotations

import pytest


def test_yaml_loads_with_safe_load_only(tmp_config_dir):
    pytest.fail(
        "not implemented — Plan 21-04: config loader must use yaml.safe_load (no "
        "yaml.load with arbitrary tags); inject a !!python/object tag and assert load fails"
    )


def test_cli_overrides_yaml(tmp_config_dir):
    pytest.fail(
        "not implemented — Plan 21-04: CLI flag must override matching YAML key; "
        "assert merged config reflects CLI value"
    )


def test_unknown_top_level_key_rejected(tmp_config_dir):
    pytest.fail(
        "not implemented — Plan 21-04: unknown top-level key in YAML must raise "
        "ValueError (strict schema validation)"
    )


def test_default_paths_match_research():
    pytest.fail(
        "not implemented — Plan 21-04: default model_path / log_path / config_path "
        "must match 21-RESEARCH.md 'Recommended Project Structure' section"
    )
