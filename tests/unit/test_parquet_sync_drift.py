"""Meta-tests for parquet sync helpers — simulate drift in tmp_path (D-17 verification)."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "tests" / "integration" / "test_parquet_shards_in_sync.py"


def _load_preflight_module():
    os.environ.pop("ACOUSTIC_SKIP_NOISE_PREFLIGHT", None)
    spec = importlib.util.spec_from_file_location(
        "_parquet_preflight_under_test", PREFLIGHT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_fake_parquet(path: Path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"label": [0] * n_rows})
    pq.write_table(table, str(path))


def _write_fake_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # tiny stub; tests don't decode


def test_total_shard_rows_sums_across_glob(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    shard_dir = tmp_path / "shards"
    _write_fake_parquet(shard_dir / "train-0.parquet", 5)
    _write_fake_parquet(shard_dir / "train-1.parquet", 7)
    assert mod.total_shard_rows(shard_dir) == 12


def test_total_shard_rows_zero_when_dir_missing(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    assert mod.total_shard_rows(tmp_path / "nonexistent") == 0


def test_count_source_wavs_recursive(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    src = tmp_path / "src"
    _write_fake_wav(src / "a.wav")
    _write_fake_wav(src / "sub" / "b.wav")
    _write_fake_wav(src / "sub" / "deeper" / "c.wav")
    (src / "not_a_wav.txt").write_text("ignored")
    assert mod.count_source_wavs(src) == 3


def test_count_eval_label_entries_reads_labels_json(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([
        {"file": "a.wav", "label": "drone"},
        {"file": "b.wav", "label": "no_drone"},
    ]))
    assert mod.count_eval_label_entries(labels) == 2


def test_count_eval_label_entries_rejects_non_list(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(RuntimeError, match="not a JSON list"):
        mod.count_eval_label_entries(labels)


def test_mtime_gate_detects_newer_source(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    shard_dir = tmp_path / "shards"
    src_dir = tmp_path / "src"
    _write_fake_parquet(shard_dir / "train-0.parquet", 1)
    _write_fake_wav(src_dir / "a.wav")
    # Make the WAV strictly newer than the parquet shard.
    time.sleep(0.05)
    os.utime(src_dir / "a.wav", None)
    assert mod.newest_source_mtime(src_dir) > mod.newest_shard_mtime(shard_dir)


def test_mtime_gate_passes_when_shard_newer(tmp_path: Path) -> None:
    mod = _load_preflight_module()
    shard_dir = tmp_path / "shards"
    src_dir = tmp_path / "src"
    _write_fake_wav(src_dir / "a.wav")
    time.sleep(0.05)
    _write_fake_parquet(shard_dir / "train-0.parquet", 1)
    assert mod.newest_shard_mtime(shard_dir) >= mod.newest_source_mtime(src_dir)


def test_no_auto_regen_imports() -> None:
    """D-18: preflight must NOT import the exporter (no auto-regen path)."""
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    assert "import scripts.export_uma16_parquet" not in text
    assert "from scripts.export_uma16_parquet" not in text
    assert "from scripts import export_uma16_parquet" not in text


def test_no_writes_to_source_trees() -> None:
    """D-19: preflight must not call any write/modify ops on source dirs."""
    text = PREFLIGHT_PATH.read_text(encoding="utf-8")
    forbidden_ops = ["write_text", "write_bytes", "shutil.copy", "shutil.move",
                     "shutil.rmtree", "os.remove", "Path.unlink", ".unlink("]
    for op in forbidden_ops:
        assert op not in text, f"{op} found in preflight — D-19 violation"
