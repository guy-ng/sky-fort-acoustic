"""Phase 22 Wave 0: eval harness produces metrics.json with D-27 gate keys. Green after Plan 07."""
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="Phase 22 Plan 07 creates promote_efficientat.py + uma16_eval.py",
)


def test_promote_if_gates_pass_is_importable():
    from acoustic.evaluation.promotion import promote_if_gates_pass  # noqa


def test_uma16_eval_is_importable():
    from acoustic.evaluation.uma16_eval import evaluate_on_uma16  # noqa


def test_d27_thresholds_are_defined_in_code():
    from acoustic.evaluation.promotion import REAL_TPR_MIN, REAL_FPR_MAX
    assert REAL_TPR_MIN == 0.80
    assert REAL_FPR_MAX == 0.05


def test_promote_efficientat_cli_exists():
    assert Path("scripts/promote_efficientat.py").exists()


def test_metrics_json_schema(tmp_path):
    """After running the eval harness on a fixture, metrics.json must have
    real_TPR, real_FPR, dads_accuracy keys."""
    pytest.skip("requires a tiny synthetic UMA-16 fixture set -- Plan 07 smoke")
