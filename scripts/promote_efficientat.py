#!/usr/bin/env python
"""Phase 22: EfficientAT promotion CLI.

Runs the D-27 real-device gate against a candidate checkpoint, and promotes
to models/efficientat_mn10.pt only if ALL gates pass.

Usage:
    python scripts/promote_efficientat.py \
        --version v8 \
        --checkpoint models/efficientat_mn10_v8.pt \
        --expected-sha256 <64-char hex> \
        --manifest data/eval/uma16_real_v8/manifest.json \
        --metrics-out .planning/phases/22-.../metrics_v8.json

Exit codes:
    0 — promoted (target file overwritten)
    1 — gate failed (target unchanged)
    2 — precondition failed (missing files, sha mismatch)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from acoustic.evaluation.promotion import (
    DADS_ACC_MIN,
    REAL_FPR_MAX,
    REAL_TPR_MIN,
    promote_if_gates_pass,
)
from acoustic.evaluation.uma16_eval import evaluate_on_uma16

_log = logging.getLogger("promote_efficientat")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 22 EfficientAT promotion gate")
    parser.add_argument("--version", required=True, help="e.g. v8")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("models/efficientat_mn10.pt"),
        help="Live-service symlink/file to overwrite on successful promotion",
    )
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval/uma16_real_v8/manifest.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tpr-min", type=float, default=REAL_TPR_MIN)
    parser.add_argument("--fpr-max", type=float, default=REAL_FPR_MAX)
    parser.add_argument(
        "--dads-acc-min",
        type=float,
        default=None,
        help="If set, additional DADS accuracy gate. Default: off for v8.",
    )
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print gate decision but do NOT copy",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.checkpoint.exists():
        _log.error("checkpoint missing: %s", args.checkpoint)
        return 2

    # Load classifier through the actual inference path
    # (so any train/serve drift surfaces here, not just in production)
    from acoustic.classification.efficientat.classifier import EfficientATClassifier

    _log.info("Loading classifier from %s", args.checkpoint)
    import acoustic.classification.efficientat  # noqa: F401 — register model
    from acoustic.classification.ensemble import load_model

    clf = load_model("efficientat_mn10", str(args.checkpoint))

    _log.info("Running UMA-16 evaluation on manifest %s", args.manifest)
    metrics = evaluate_on_uma16(
        clf,
        manifest_path=args.manifest,
        threshold=args.threshold,
    )

    print(json.dumps(metrics, indent=2, default=str))

    if args.dry_run:
        _log.info("DRY RUN — skipping promotion")
        return (
            0
            if (
                metrics["real_TPR"] >= args.tpr_min
                and metrics["real_FPR"] <= args.fpr_max
            )
            else 1
        )

    promoted, reasons = promote_if_gates_pass(
        source_path=args.checkpoint,
        target_path=args.target,
        metrics=metrics,
        expected_sha256=args.expected_sha256,
        tpr_min=args.tpr_min,
        fpr_max=args.fpr_max,
        dads_acc_min=args.dads_acc_min,
        metrics_out=args.metrics_out,
    )

    if not promoted:
        _log.error("PROMOTION FAILED: %s", "; ".join(reasons))
        return 1

    _log.info(
        "PROMOTED %s -> %s (version=%s)",
        args.checkpoint,
        args.target,
        args.version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
