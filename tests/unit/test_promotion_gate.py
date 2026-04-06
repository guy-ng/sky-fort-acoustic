"""RED stubs for the v7 promotion gate (D-26, D-27, D-29).

Plan 20-06 implements ``promote_v7_if_gates_pass`` in
``acoustic.evaluation.promotion``. Thresholds are pinned by D-26/D-27:

  - DADS test accuracy >= 0.95
  - Real-capture TPR     >= 0.80
  - Real-capture FPR     <= 0.05

When all gates pass the function copies the trained checkpoint
(``models/efficientat_mn10_v7.pt``) to the production slot
(``models/efficientat_mn10.pt``).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_promote():
    from acoustic.evaluation.promotion import promote_v7_if_gates_pass

    return promote_v7_if_gates_pass


def test_promotion_blocked_when_dads_fails(tmp_path: Path) -> None:
    promote = _import_promote()
    src = tmp_path / "efficientat_mn10_v7.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "efficientat_mn10.pt"
    ok = promote(
        dads_acc=0.93,
        real_tpr=0.85,
        real_fpr=0.03,
        src_checkpoint=src,
        dst_checkpoint=dst,
    )
    assert ok is False
    assert not dst.exists()


def test_promotion_blocked_when_real_tpr_fails(tmp_path: Path) -> None:
    promote = _import_promote()
    src = tmp_path / "efficientat_mn10_v7.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "efficientat_mn10.pt"
    ok = promote(
        dads_acc=0.97,
        real_tpr=0.70,
        real_fpr=0.03,
        src_checkpoint=src,
        dst_checkpoint=dst,
    )
    assert ok is False
    assert not dst.exists()


def test_promotion_blocked_when_real_fpr_fails(tmp_path: Path) -> None:
    promote = _import_promote()
    src = tmp_path / "efficientat_mn10_v7.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "efficientat_mn10.pt"
    ok = promote(
        dads_acc=0.97,
        real_tpr=0.85,
        real_fpr=0.10,
        src_checkpoint=src,
        dst_checkpoint=dst,
    )
    assert ok is False
    assert not dst.exists()


def test_promotion_succeeds_when_both_pass(tmp_path: Path) -> None:
    promote = _import_promote()
    src = tmp_path / "efficientat_mn10_v7.pt"
    src.write_bytes(b"weights")
    dst = tmp_path / "efficientat_mn10.pt"
    ok = promote(
        dads_acc=0.97,
        real_tpr=0.85,
        real_fpr=0.03,
        src_checkpoint=src,
        dst_checkpoint=dst,
    )
    assert ok is True


def test_promotion_copies_checkpoint(tmp_path: Path) -> None:
    promote = _import_promote()
    src = tmp_path / "efficientat_mn10_v7.pt"
    src.write_bytes(b"v7-weights")
    dst = tmp_path / "efficientat_mn10.pt"
    ok = promote(
        dads_acc=0.97,
        real_tpr=0.85,
        real_fpr=0.03,
        src_checkpoint=src,
        dst_checkpoint=dst,
    )
    assert ok is True
    assert dst.exists()
    assert dst.read_bytes() == b"v7-weights"
