"""Wave 0 RED stub — byte-identity drift guard for vendored preprocess artifacts.

Covers: D-04 (CI drift guard between apps/rpi-edge/ and src/acoustic/classification/efficientat/).
Owner: Plan 21-02.
"""
from __future__ import annotations

import pytest


def test_preprocess_py_byte_identical_to_training_source():
    pytest.fail(
        "not implemented — Plan 21-02 must hash apps/rpi-edge/skyfort_edge/preprocess.py "
        "and src/acoustic/classification/efficientat/preprocess.py and assert sha256 equality"
    )


def test_mel_banks_pt_byte_identical_to_training_source():
    pytest.fail(
        "not implemented — Plan 21-02 must hash apps/rpi-edge/skyfort_edge/mel_banks_128_1024_32k.pt "
        "vs the training-side artifact and assert sha256 equality"
    )
