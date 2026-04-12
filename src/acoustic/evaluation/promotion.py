"""Phase 22: promotion gate for EfficientAT v8 (and any future version)."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .uma16_eval import _sha256

_log = logging.getLogger(__name__)

# D-27 promotion thresholds (from Phase 22 CONTEXT.md — USER-LOCKED)
REAL_TPR_MIN: float = 0.80
REAL_FPR_MAX: float = 0.05
# D-26 DADS accuracy threshold (carry-over from Phase 20)
DADS_ACC_MIN: float = 0.95


def promote_if_gates_pass(
    source_path: Path,
    target_path: Path,
    metrics: dict[str, Any],
    *,
    expected_sha256: str | None = None,
    tpr_min: float = REAL_TPR_MIN,
    fpr_max: float = REAL_FPR_MAX,
    dads_acc_min: float | None = DADS_ACC_MIN,
    metrics_out: Path | None = None,
) -> tuple[bool, list[str]]:
    """Promote source -> target if ALL gates pass.

    Returns (promoted, reasons). If any gate fails, source is NOT copied
    and the list of failing reasons is returned.
    """
    reasons: list[str] = []

    if not source_path.exists():
        reasons.append(f"source missing: {source_path}")
        return False, reasons

    if expected_sha256 is not None:
        got = _sha256(source_path)
        if got != expected_sha256:
            reasons.append(
                f"sha256 mismatch: expected {expected_sha256[:12]}... got {got[:12]}..."
            )

    real_tpr = float(metrics.get("real_TPR", 0.0))
    real_fpr = float(metrics.get("real_FPR", 1.0))
    if real_tpr < tpr_min:
        reasons.append(f"real_TPR {real_tpr:.3f} < {tpr_min}")
    if real_fpr > fpr_max:
        reasons.append(f"real_FPR {real_fpr:.3f} > {fpr_max}")

    if (
        dads_acc_min is not None
        and "dads_accuracy" in metrics
        and metrics["dads_accuracy"] is not None
    ):
        dads_acc = float(metrics["dads_accuracy"])
        if dads_acc < dads_acc_min:
            reasons.append(f"dads_accuracy {dads_acc:.3f} < {dads_acc_min}")

    if metrics_out is not None:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2, default=str))

    if reasons:
        _log.error("PROMOTION BLOCKED: %s", "; ".join(reasons))
        return False, reasons

    _log.info("PROMOTION GATES PASSED. Copying %s -> %s", source_path, target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(target_path))
    return True, []
