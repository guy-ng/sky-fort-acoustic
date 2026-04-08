"""Phase 22 Wave 0: dataset cardinality matches preflight expectations. Green after Plan 04."""
import pytest

pytestmark = pytest.mark.xfail(
    strict=False,
    reason="Phase 22 Plan 04/06 wires field data into training dataset",
)


def test_concat_dataset_cardinality_matches_manifest():
    """After building the v8 train dataset, total windows = sum of per-file window counts."""
    pytest.skip("ConcatDataset wiring lands in Plan 06")


def test_no_holdout_file_appears_in_training_set():
    """Session-level isolation invariant: the 5 holdout filenames must never
    appear in train_ds file list."""
    pytest.skip("ConcatDataset wiring lands in Plan 06")
