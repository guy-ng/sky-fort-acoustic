"""Phase 20.1 preflight: noise corpora must exist on disk before training (D-11..D-15).

This is a HARD GATE. It MUST fail on missing data so Phase 20-05 can never
silently ship a Docker image without background-noise training data.

The single auditable opt-out is the env var ACOUSTIC_SKIP_NOISE_PREFLIGHT=1.
CI may set it explicitly when running in an environment without the corpora;
the training host MUST NOT set it. One `grep ACOUSTIC_SKIP_NOISE_PREFLIGHT`
finds every skip site in the repo.

D-11..D-15 -- see .planning/phases/20.1-.../20.1-CONTEXT.md
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import soundfile as sf

if os.environ.get("ACOUSTIC_SKIP_NOISE_PREFLIGHT") == "1":
    pytest.skip(
        "ACOUSTIC_SKIP_NOISE_PREFLIGHT=1 -- noise corpora preflight skipped",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
NOISE_ROOT = REPO_ROOT / "data" / "noise"

FSD50K_CLASSES: tuple[str, ...] = (
    "Wind",
    "Rain",
    "Traffic_noise_and_roadway_noise",
    "Mechanical_fan",
    "Engine",
    "Bird",
)


def _count_wavs(corpus_dir: Path) -> int:
    if not corpus_dir.is_dir():
        return 0
    return sum(1 for _ in corpus_dir.rglob("*.wav"))


def _missing_data_message(corpus: str, found: int, expected: int) -> str:
    """D-15: failure message points at the fix command."""
    return (
        f"data/noise/{corpus}/ has {found} wav files (expected >= {expected}). "
        f"Run: python scripts/acquire_noise_corpora.py {corpus}"
    )


def _fsd50k_class_message(cls: str, found: int) -> str:
    """D-15: per-class failure for FSD50K subset."""
    return (
        f"data/noise/fsd50k_subset/{cls}/ has {found} wav files (expected >= 100). "
        f"Run: python scripts/acquire_noise_corpora.py fsd50k --force"
    )


def _readability_message(corpus: str, path: Path, reason: str) -> str:
    """D-15: readability failure points at --force re-run."""
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    return (
        f"{corpus}: {rel} failed readability check ({reason}). "
        f"Download may be truncated. "
        f"Re-run: python scripts/acquire_noise_corpora.py {corpus} --force"
    )


def _spot_check_readable(corpus_dir: Path, corpus: str, n: int = 2) -> None:
    wavs = list(corpus_dir.rglob("*.wav"))
    if not wavs:
        return  # count check already failed in the corpus's test_*
    sample = random.sample(wavs, min(n, len(wavs)))
    for p in sample:
        try:
            info = sf.info(str(p))
        except Exception as e:
            pytest.fail(
                _readability_message(
                    corpus, p, f"sf.info raised {type(e).__name__}: {e}"
                )
            )
        if info.frames == 0:
            pytest.fail(_readability_message(corpus, p, "zero frames"))


def test_esc50_present() -> None:
    corpus_dir = NOISE_ROOT / "esc50"
    count = _count_wavs(corpus_dir)
    if count < 2000:
        pytest.fail(_missing_data_message("esc50", count, 2000))


def test_urbansound8k_present() -> None:
    corpus_dir = NOISE_ROOT / "urbansound8k"
    count = _count_wavs(corpus_dir)
    if count < 8000:
        pytest.fail(_missing_data_message("urbansound8k", count, 8000))


def test_fsd50k_subset_present() -> None:
    corpus_dir = NOISE_ROOT / "fsd50k_subset"
    total = _count_wavs(corpus_dir)
    if total < 1500:
        pytest.fail(_missing_data_message("fsd50k", total, 1500))
    for cls in FSD50K_CLASSES:
        cls_dir = corpus_dir / cls
        cls_count = _count_wavs(cls_dir)
        if cls_count < 100:
            pytest.fail(_fsd50k_class_message(cls, cls_count))


def test_readability_spot_check() -> None:
    """D-14: 1-2 random WAVs per corpus via sf.info to catch truncated downloads."""
    for corpus in ("esc50", "urbansound8k", "fsd50k_subset"):
        corpus_dir = NOISE_ROOT / corpus
        if corpus_dir.is_dir() and _count_wavs(corpus_dir) > 0:
            _spot_check_readable(corpus_dir, corpus)
