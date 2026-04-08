# Phase 22 Test Fixtures

Holds synthetic audio fixtures for unit/integration tests that must not depend on
the real `data/field/` tree (which is large and not always checked in).

Fixtures (populated as tests need them):
- `parity_sample.wav` -- 16kHz mono 1s random noise for RMS parity test
- `uma16_tiny_eval/` -- synthetic 2-file eval set for eval-harness smoke (Plan 07)
