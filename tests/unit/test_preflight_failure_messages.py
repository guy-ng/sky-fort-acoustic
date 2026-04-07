"""D-15 meta-tests: preflight failure messages must contain the fix command."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "tests" / "integration" / "test_noise_corpora_present.py"


def _load_preflight_module():
    """Import the preflight module without triggering the env-var skip."""
    os.environ.pop("ACOUSTIC_SKIP_NOISE_PREFLIGHT", None)
    spec = importlib.util.spec_from_file_location(
        "_preflight_under_test", PREFLIGHT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_esc50_message_contains_fix_command() -> None:
    mod = _load_preflight_module()
    msg = mod._missing_data_message("esc50", 0, 2000)
    assert "python scripts/acquire_noise_corpora.py esc50" in msg
    assert "0 wav files" in msg
    assert "expected >= 2000" in msg


def test_urbansound8k_message_contains_fix_command() -> None:
    mod = _load_preflight_module()
    msg = mod._missing_data_message("urbansound8k", 5, 8000)
    assert "python scripts/acquire_noise_corpora.py urbansound8k" in msg
    assert "5 wav files" in msg


def test_fsd50k_subset_message_contains_fix_command() -> None:
    mod = _load_preflight_module()
    msg = mod._missing_data_message("fsd50k", 100, 1500)
    assert "python scripts/acquire_noise_corpora.py fsd50k" in msg


def test_fsd50k_class_message_names_class_and_force_flag() -> None:
    mod = _load_preflight_module()
    msg = mod._fsd50k_class_message("Mechanical_fan", 42)
    assert "Mechanical_fan" in msg
    assert "42 wav files" in msg
    assert "python scripts/acquire_noise_corpora.py fsd50k --force" in msg


def test_readability_message_points_at_force_rerun() -> None:
    mod = _load_preflight_module()
    msg = mod._readability_message(
        "esc50",
        Path("data/noise/esc50/audio/1-100032-A-0.wav"),
        "zero frames",
    )
    assert "esc50" in msg
    assert "Download may be truncated" in msg
    assert "python scripts/acquire_noise_corpora.py esc50 --force" in msg


def test_module_level_skip_env_var_name() -> None:
    """ACOUSTIC_SKIP_NOISE_PREFLIGHT must be the literal opt-out (D-12)."""
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    assert "ACOUSTIC_SKIP_NOISE_PREFLIGHT" in text
    assert "allow_module_level=True" in text


def test_six_fsd50k_classes_listed() -> None:
    mod = _load_preflight_module()
    assert mod.FSD50K_CLASSES == (
        "Wind",
        "Rain",
        "Traffic_noise_and_roadway_noise",
        "Mechanical_fan",
        "Engine",
        "Bird",
    )
