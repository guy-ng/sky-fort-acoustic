"""Unit tests for scripts/acquire_noise_corpora.py (Phase 20.1 Plan 01).

All tests monkeypatch soundata.initialize and the module-level NOISE_ROOT so no
test ever touches the real data/noise/ tree or the network. soundata itself is
stubbed via sys.modules at import time so the test suite runs even when the
host has not yet installed `soundata>=1.0,<2.0`.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# --- soundata stub: install BEFORE importing the script under test -----------
if "soundata" not in sys.modules:
    _stub = types.ModuleType("soundata")
    _stub.__version__ = "1.0.1-test-stub"
    _stub.initialize = MagicMock(name="soundata.initialize")
    sys.modules["soundata"] = _stub

# Make scripts/ importable as a package path (it has no __init__.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import acquire_noise_corpora as ans  # noqa: E402


# --- helpers -----------------------------------------------------------------


def _touch_wavs(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"clip_{i:04d}.wav").write_bytes(b"")


def _write_dev_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fname", "labels", "mids", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --- tests -------------------------------------------------------------------


def test_cli_subcommands_registered():
    parser = ans.build_parser()
    for name in ("esc50", "urbansound8k", "fsd50k", "all"):
        ns = parser.parse_args([name])
        assert ns.corpus == name


def test_marker_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    corpus_dir = tmp_path / "test"
    _touch_wavs(corpus_dir, 3)

    ans.write_marker(corpus_dir, "test", "unit-test:source")
    marker = corpus_dir / ".acquired.json"
    payload = json.loads(marker.read_text())

    assert payload["schema_version"] == 1
    assert payload["corpus"] == "test"
    assert payload["source"] == "unit-test:source"
    assert payload["file_count"] == 3
    # ISO-8601 UTC parses
    datetime.strptime(payload["acquired_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_is_already_acquired_true_when_count_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    corpus_dir = tmp_path / "esc50"
    _touch_wavs(corpus_dir, 5)
    ans.write_marker(corpus_dir, "esc50", "test")
    assert ans.is_already_acquired(corpus_dir) is True


def test_is_already_acquired_false_on_count_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    corpus_dir = tmp_path / "esc50"
    _touch_wavs(corpus_dir, 5)
    ans.write_marker(corpus_dir, "esc50", "test")
    # Delete one wav -> count drift
    next(corpus_dir.glob("*.wav")).unlink()
    assert ans.is_already_acquired(corpus_dir) is False


def test_disk_guard_aborts_on_low_space(tmp_path, monkeypatch):
    DiskUsage = type("DiskUsage", (), {})
    fake = DiskUsage()
    fake.total = 100 * 1024 ** 3
    fake.used = 99 * 1024 ** 3
    fake.free = 1 * 1024 ** 3
    monkeypatch.setattr(ans.shutil, "disk_usage", lambda _p: fake)
    with pytest.raises(RuntimeError) as excinfo:
        ans.check_disk_space(tmp_path, "fsd50k")
    msg = str(excinfo.value)
    assert "Insufficient disk space" in msg
    assert "fsd50k" in msg


def test_disk_guard_passes_on_sufficient_space(tmp_path, monkeypatch):
    DiskUsage = type("DiskUsage", (), {})
    fake = DiskUsage()
    fake.total = 200 * 1024 ** 3
    fake.used = 100 * 1024 ** 3
    fake.free = 100 * 1024 ** 3
    monkeypatch.setattr(ans.shutil, "disk_usage", lambda _p: fake)
    # Must not raise
    ans.check_disk_space(tmp_path, "fsd50k")


def test_primary_tag_rule(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for fname in ("1", "2", "3"):
        (audio / f"{fname}.wav").write_bytes(b"")

    csv_path = tmp_path / "dev.csv"
    _write_dev_csv(
        csv_path,
        [
            {"fname": "1", "labels": "Wind,Rain", "mids": "x", "split": "train"},
            {"fname": "2", "labels": "Bird", "mids": "x", "split": "train"},
            {"fname": "3", "labels": "OtherThing", "mids": "x", "split": "train"},
        ],
    )

    results = list(ans._iter_fsd50k_clips(csv_path, audio))
    assert len(results) == 2
    by_name = {wav.name: cls for wav, cls in results}
    assert by_name["1.wav"] == "Wind"
    assert by_name["2.wav"] == "Bird"
    assert "3.wav" not in by_name


def test_secondary_label_iter(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for fname in ("1", "2", "3"):
        (audio / f"{fname}.wav").write_bytes(b"")

    csv_path = tmp_path / "dev.csv"
    _write_dev_csv(
        csv_path,
        [
            {"fname": "1", "labels": "Wind,Rain", "mids": "x", "split": "train"},
            {"fname": "2", "labels": "Bird", "mids": "x", "split": "train"},
            {"fname": "3", "labels": "OtherThing", "mids": "x", "split": "train"},
        ],
    )

    results = list(ans._iter_fsd50k_clips_secondary(csv_path, audio, "Rain"))
    assert len(results) == 1
    assert results[0][0].name == "1.wav"


def test_safe_copy_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    src = tmp_path / "src.wav"
    src.write_bytes(b"")
    # Build a path that escapes NOISE_ROOT after resolve()
    bad_dst = tmp_path / ".." / "etc" / "passwd"
    with pytest.raises(RuntimeError) as excinfo:
        ans._safe_copy(src, bad_dst)
    assert "Refusing to copy outside NOISE_ROOT" in str(excinfo.value)


def test_safe_copy_writes_regular_file_not_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    src = tmp_path / "src.wav"
    src.write_bytes(b"hello")
    dst = tmp_path / "fsd50k_subset" / "Wind" / "src.wav"
    ans._safe_copy(src, dst)
    assert dst.is_file()
    assert not dst.is_symlink()
    assert dst.read_bytes() == b"hello"


def test_acquire_esc50_skips_when_marker_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ans, "NOISE_ROOT", tmp_path)
    corpus_dir = tmp_path / "esc50"
    # Use a smaller count for speed; idempotency is by file_count match, not absolute
    _touch_wavs(corpus_dir, 2000)
    ans.write_marker(corpus_dir, "esc50", "test")

    fake_initialize = MagicMock(name="soundata.initialize")
    monkeypatch.setattr(ans.soundata, "initialize", fake_initialize)

    result = ans.acquire_esc50(force=False)

    assert result == corpus_dir
    assert fake_initialize.call_count == 0
