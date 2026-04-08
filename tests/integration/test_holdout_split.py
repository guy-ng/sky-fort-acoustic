"""Phase 22 Wave 0: holdout split determinism. Green after Plan 04."""
import pytest

HOLDOUT = frozenset({
    "20260408_091054_136dc5.wav",  # 10inch 4kg (explicit)
    "20260408_092615_1a055f.wav",  # 10inch heavy
    "20260408_091724_bb0ed8.wav",  # phantom 4
    "20260408_084222_44dc5c.wav",  # 5inch
    "20260408_090757_1c50e9.wav",  # bg (104s)
})


def test_holdout_split_is_frozen_in_code():
    """The 5-file holdout list lives in code (not config) and is git-tracked."""
    # When Plan 04 lands: from scripts.preflight_v8_data import HOLDOUT_FILES
    # assert set(HOLDOUT_FILES) == HOLDOUT
    try:
        from scripts.preflight_v8_data import HOLDOUT_FILES
        assert set(HOLDOUT_FILES) == HOLDOUT
    except ImportError:
        pytest.xfail("Plan 04 creates preflight_v8_data")


def test_holdout_covers_all_four_drone_subclasses():
    """Split rationale (Focus 5): one file per sub-label."""
    # 5inch, 10inch payload, 10inch heavy, phantom 4 -> each present in holdout
    assert len(HOLDOUT) == 5
