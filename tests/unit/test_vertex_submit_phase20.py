"""RED stubs for scripts/vertex_submit.py Phase 20 wiring (D-21..D-24).

Plan 20-05 will add ``build_env_vars_v7`` and ``check_l4_quota`` to
``scripts/vertex_submit.py``. These tests pin the contract.
"""

from __future__ import annotations

import pytest


def _import_build():
    from scripts.vertex_submit import build_env_vars_v7

    return build_env_vars_v7


def _import_quota():
    from scripts.vertex_submit import check_l4_quota

    return check_l4_quota


def test_v7_job_name() -> None:
    """The Phase 20 job name must contain the literal 'v7'."""
    build_env_vars_v7 = _import_build()
    payload = build_env_vars_v7()
    job_name = payload.get("job_name") or payload.get("display_name") or ""
    assert "v7" in job_name


def test_env_vars_include_phase20() -> None:
    """All Phase 20 ACOUSTIC_TRAINING_* keys must be propagated (D-23)."""
    build_env_vars_v7 = _import_build()
    payload = build_env_vars_v7()
    env = payload.get("env_vars") or payload
    required = {
        "ACOUSTIC_TRAINING_WIDE_GAIN_DB",
        "ACOUSTIC_TRAINING_RIR_ENABLED",
        "ACOUSTIC_TRAINING_RIR_PROBABILITY",
        "ACOUSTIC_TRAINING_NOISE_DIRS",
        "ACOUSTIC_TRAINING_WINDOW_OVERLAP_RATIO",
    }
    missing = required - set(env.keys())
    assert not missing, f"missing env vars: {missing}"


def test_l4_with_t4_fallback() -> None:
    """Phase 20 must request L4 (g2-standard-8) with a documented T4 fallback (D-22)."""
    build_env_vars_v7 = _import_build()
    payload = build_env_vars_v7()
    machine = payload.get("machine_type", "")
    accelerator = payload.get("accelerator_type", "")
    fallback = payload.get("fallback_accelerator_type", "")
    assert machine == "g2-standard-8"
    assert "L4" in accelerator
    assert "T4" in fallback


def test_preflight_quota_check_callable() -> None:
    """check_l4_quota(project, region) must exist and return a bool (D-22)."""
    check_l4_quota = _import_quota()
    result = check_l4_quota("interception-dashboard", "us-central1")
    assert isinstance(result, bool)
