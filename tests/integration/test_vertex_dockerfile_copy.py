"""Integration RED stub: Dockerfile.vertex must COPY Phase 20 noise/ambient data (D-24).

Plan 20-05 will add COPY lines for ``data/noise`` and ``data/field/uma16_ambient``
into Dockerfile.vertex so the Vertex training container has access to the
expanded background-noise corpus and the UMA-16 ambient negatives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[2] / "Dockerfile.vertex"


def _read_dockerfile() -> str:
    if not DOCKERFILE.is_file():
        pytest.skip(f"Dockerfile.vertex not found at {DOCKERFILE}")
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_copies_noise_dir() -> None:
    """Dockerfile.vertex must COPY data/noise into the container (D-24)."""
    content = _read_dockerfile()
    assert "data/noise" in content, (
        "Dockerfile.vertex missing COPY for data/noise — Plan 20-05 must add it"
    )


def test_dockerfile_copies_uma16_ambient() -> None:
    """Dockerfile.vertex must COPY data/field/uma16_ambient (D-24)."""
    content = _read_dockerfile()
    assert "data/field/uma16_ambient" in content, (
        "Dockerfile.vertex missing COPY for data/field/uma16_ambient — Plan 20-05 must add it"
    )
