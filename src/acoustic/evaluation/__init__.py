"""Evaluation harness for model performance analysis."""

from .evaluator import Evaluator
from .models import DistributionStats, EvaluationResult, FileResult
from .promotion import (
    DADS_ACC_MIN,
    REAL_FPR_MAX,
    REAL_TPR_MIN,
    promote_if_gates_pass,
)
from .uma16_eval import evaluate_on_uma16

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "FileResult",
    "DistributionStats",
    "evaluate_on_uma16",
    "promote_if_gates_pass",
    "REAL_TPR_MIN",
    "REAL_FPR_MAX",
    "DADS_ACC_MIN",
]
