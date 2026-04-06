"""Phase 20.1 preflight: parquet shards must be in sync with WAV source trees (D-16..D-19).

The training Docker image bakes these parquet shards via `Dockerfile.vertex-base`.
A stale shard (rows out of sync with the WAV source tree, or older mtime than the
newest source WAV) means the trainer sees data that no longer matches the field/eval
captures on disk. This test fires before `docker build` to catch that.

D-18: this test does NOT auto-regenerate shards. It only reports drift. The fix is
documented in every failure message: `python scripts/export_uma16_parquet.py`.

D-19: this test does NOT modify `data/field/uma16_ambient/` or `data/eval/uma16_real/`
or any file under them. It reads `labels.json` and `*.wav` mtimes only.

The single auditable opt-out is the env var ACOUSTIC_SKIP_NOISE_PREFLIGHT=1 — same
gate as `tests/integration/test_noise_corpora_present.py`. One grep finds both.

D-17 NOTE: The CONTEXT.md decision text says
`parquet_row_count == len(rglob source_dir, "*.wav")` for both shards. This is
correct for the ambient branch. The eval branch uses `len(json.load(labels.json))`
because `scripts/export_uma16_parquet.py` populates eval rows from `labels.json`
entries (verified by reading the exporter source). This is a research-surfaced
implementation detail of D-17, not a scope change. Documented in 20.1-03-PLAN.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

if os.environ.get("ACOUSTIC_SKIP_NOISE_PREFLIGHT") == "1":
    pytest.skip(
        "ACOUSTIC_SKIP_NOISE_PREFLIGHT=1 — parquet sync preflight skipped",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]

# Mirror constants from the parquet exporter script (NOT imported per D-18).
# Source-of-truth file path: <repo>/scripts/export_uma16_parquet.py
AMBIENT_SRC = REPO_ROOT / "data" / "field" / "uma16_ambient"
EVAL_SRC = REPO_ROOT / "data" / "eval" / "uma16_real"
EVAL_LABELS_PATH = EVAL_SRC / "labels.json"
OUT_ROOT = REPO_ROOT / "data" / "parquet"
AMBIENT_SHARD_DIR = OUT_ROOT / "ambient"
EVAL_SHARD_DIR = OUT_ROOT / "eval"

REGEN_HINT = "python scripts/export_uma16_parquet.py"


# ---------- helpers (module-level so meta-tests can import them) ----------


def all_shard_paths(shard_dir: Path) -> list[Path]:
    """Return sorted list of train-*.parquet under shard_dir (forward-compatible)."""
    if not shard_dir.is_dir():
        return []
    return sorted(shard_dir.glob("train-*.parquet"))


def total_shard_rows(shard_dir: Path) -> int:
    return sum(pq.read_metadata(str(p)).num_rows for p in all_shard_paths(shard_dir))


def newest_shard_mtime(shard_dir: Path) -> float:
    paths = all_shard_paths(shard_dir)
    if not paths:
        return 0.0
    return max(os.stat(p).st_mtime for p in paths)


def count_source_wavs(src_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    return sum(1 for _ in src_dir.rglob("*.wav"))


def newest_source_mtime(src_dir: Path) -> float:
    if not src_dir.is_dir():
        return 0.0
    mtimes = [os.stat(p).st_mtime for p in src_dir.rglob("*.wav")]
    return max(mtimes) if mtimes else 0.0


def count_eval_label_entries(labels_path: Path) -> int:
    """D-17 corrected formula for eval shard: row count = labels.json entry count."""
    if not labels_path.is_file():
        return 0
    data = json.loads(labels_path.read_text())
    if not isinstance(data, list):
        raise RuntimeError(
            f"{labels_path} is not a JSON list — schema changed?"
        )
    return len(data)


# ---------- tests ----------


def test_ambient_shards_exist() -> None:
    paths = all_shard_paths(AMBIENT_SHARD_DIR)
    if not paths:
        pytest.fail(
            f"No parquet shards found at {AMBIENT_SHARD_DIR}/train-*.parquet. "
            f"Regenerate: python scripts/export_uma16_parquet.py"
        )


def test_ambient_shard_row_count_matches_source() -> None:
    if not AMBIENT_SRC.is_dir():
        pytest.fail(
            f"Source dir {AMBIENT_SRC} missing — cannot verify ambient shard sync. "
            f"Restore the WAV tree or skip via ACOUSTIC_SKIP_NOISE_PREFLIGHT=1. "
            f"Regenerate: python scripts/export_uma16_parquet.py"
        )
    src_count = count_source_wavs(AMBIENT_SRC)
    shard_rows = total_shard_rows(AMBIENT_SHARD_DIR)
    if shard_rows != src_count:
        pytest.fail(
            f"data/parquet/ambient/ has {shard_rows} parquet rows but "
            f"data/field/uma16_ambient/ has {src_count} wav files. "
            f"Shard is stale. Regenerate: python scripts/export_uma16_parquet.py"
        )


def test_eval_shards_exist() -> None:
    paths = all_shard_paths(EVAL_SHARD_DIR)
    if not paths:
        pytest.fail(
            f"No parquet shards found at {EVAL_SHARD_DIR}/train-*.parquet. "
            f"Regenerate: python scripts/export_uma16_parquet.py"
        )


def test_eval_shard_row_count_matches_labels() -> None:
    if not EVAL_LABELS_PATH.is_file():
        pytest.fail(
            f"{EVAL_LABELS_PATH} missing — cannot verify eval shard sync. "
            f"Restore labels.json or skip via ACOUSTIC_SKIP_NOISE_PREFLIGHT=1. "
            f"Regenerate: python scripts/export_uma16_parquet.py"
        )
    label_count = count_eval_label_entries(EVAL_LABELS_PATH)
    shard_rows = total_shard_rows(EVAL_SHARD_DIR)
    if shard_rows != label_count:
        pytest.fail(
            f"data/parquet/eval/ has {shard_rows} parquet rows but "
            f"data/eval/uma16_real/labels.json has {label_count} entries. "
            f"Shard is stale. Regenerate: python scripts/export_uma16_parquet.py"
        )


def test_shards_newer_than_sources() -> None:
    """D-17 mtime gate: every shard must be at least as new as its newest source WAV."""
    # Ambient
    if AMBIENT_SRC.is_dir() and all_shard_paths(AMBIENT_SHARD_DIR):
        ambient_shard_mt = newest_shard_mtime(AMBIENT_SHARD_DIR)
        ambient_src_mt = newest_source_mtime(AMBIENT_SRC)
        if ambient_shard_mt < ambient_src_mt:
            pytest.fail(
                f"data/parquet/ambient/ newest shard mtime {ambient_shard_mt} is older "
                f"than newest source WAV mtime {ambient_src_mt} in data/field/uma16_ambient/. "
                f"Regenerate: python scripts/export_uma16_parquet.py"
            )
    # Eval
    if EVAL_SRC.is_dir() and all_shard_paths(EVAL_SHARD_DIR):
        eval_shard_mt = newest_shard_mtime(EVAL_SHARD_DIR)
        eval_src_mt = newest_source_mtime(EVAL_SRC)
        if eval_shard_mt < eval_src_mt:
            pytest.fail(
                f"data/parquet/eval/ newest shard mtime {eval_shard_mt} is older "
                f"than newest source WAV mtime {eval_src_mt} in data/eval/uma16_real/. "
                f"Regenerate: python scripts/export_uma16_parquet.py"
            )
