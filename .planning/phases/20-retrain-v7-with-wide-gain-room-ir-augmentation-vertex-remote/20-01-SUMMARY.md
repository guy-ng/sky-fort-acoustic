---
phase: 20
plan: 01
subsystem: training-augmentation
tags: [augmentation, pyroomacoustics, wide-gain, room-ir, phase-20]
requires:
  - 20-00 (Wave 0 test stubs — recreated locally as TDD RED since worktree was sparse)
provides:
  - WideGainAugmentation class (D-01..D-04)
  - RoomIRAugmentation class (D-05..D-08)
  - pyroomacoustics dependency pinned for local + Vertex
affects:
  - downstream Plan 20-04 (trainer wiring) consumes both classes via ComposedAugmentation
tech-stack:
  added:
    - "pyroomacoustics>=0.8,<0.11"
  patterns:
    - "Pickle-safe augmentation classes (DataLoader num_workers > 0 compatible)"
    - "Precomputed RIR pool sampled per-call (pyroomacoustics out of hot path)"
key-files:
  created:
    - tests/unit/test_wide_gain_augmentation.py
    - tests/unit/test_room_ir_augmentation.py
  modified:
    - src/acoustic/training/augmentation.py
    - requirements.txt
    - requirements-vertex.txt
decisions:
  - "Followed plan verbatim: ±40 dB wide gain, pool_size=500, max_order=10, RIR cap 1 s"
  - "Pickle excludes live RNG and pool; pool rebuilt from seed on unpickle (deterministic)"
metrics:
  duration: ~10 min
  completed: 2026-04-06
---

# Phase 20 Plan 01: New Augmentations Summary

Implemented WideGainAugmentation (±40 dB uniform gain with [-1,1] clipping) and RoomIRAugmentation (procedural ShoeBox RIR pool of 500, sampled per-call via scipy.signal.fftconvolve) in `src/acoustic/training/augmentation.py`, and pinned `pyroomacoustics>=0.8,<0.11` in both `requirements.txt` and `requirements-vertex.txt`. Both classes are pickle-safe for DataLoader workers.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 RED | failing tests for WideGainAugmentation | eb92958 | tests/unit/test_wide_gain_augmentation.py |
| 1 GREEN | implement WideGainAugmentation + pin dep | 43a5149 | src/acoustic/training/augmentation.py, requirements.txt, requirements-vertex.txt |
| 2 RED | failing tests for RoomIRAugmentation | 37132b4 | tests/unit/test_room_ir_augmentation.py |
| 2 GREEN | implement RoomIRAugmentation | 1529974 | src/acoustic/training/augmentation.py |

## Verification

```
PYTHONPATH=src python -m pytest \
  tests/unit/test_wide_gain_augmentation.py \
  tests/unit/test_room_ir_augmentation.py -x -q
# 11 passed in 2.72s
```

Acceptance criteria checks:
- `grep -c "class WideGainAugmentation\|class RoomIRAugmentation" src/acoustic/training/augmentation.py` -> 2
- `grep -c "pyroomacoustics" requirements.txt requirements-vertex.txt` -> 1 each
- `grep -n "np.clip(out, -1.0, 1.0)" src/acoustic/training/augmentation.py` -> match (Pitfall 2)
- `grep -n "max_len = self._sr" src/acoustic/training/augmentation.py` -> match (Pitfall 3)
- `grep -n "__getstate__" src/acoustic/training/augmentation.py` -> 2 hooks (pickle-safe)
- pickle round-trip via `python -c "from acoustic.training.augmentation import RoomIRAugmentation; import pickle; pickle.loads(pickle.dumps(RoomIRAugmentation(pool_size=4)))"` -> OK

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Wave 0 test stubs were missing from the worktree**
- **Found during:** Task 1 RED phase
- **Issue:** The plan assumes Wave 0 (`20-00`) created RED test stubs at `tests/unit/test_wide_gain_augmentation.py` and `tests/unit/test_room_ir_augmentation.py`. These files did not exist in the worktree (Wave 0 was not pre-applied to this branch).
- **Fix:** Created both test files locally as the TDD RED step for Tasks 1 and 2, matching the behavior contract from `20-00-wave0-test-stubs-and-data-acquisition-PLAN.md` (5 WideGain tests, 6 RoomIR tests). Each test file was committed before the corresponding GREEN implementation.
- **Files:** `tests/unit/test_wide_gain_augmentation.py`, `tests/unit/test_room_ir_augmentation.py`
- **Commits:** eb92958 (WideGain RED), 37132b4 (RoomIR RED)

**2. [Rule 3 - Blocking] Worktree initially sparse / index out of sync with HEAD**
- **Found during:** Initial verification of base commit
- **Issue:** The worktree was branched from a much later commit (b0d3e36) than the orchestrator-specified base (e01aa92). After `git reset --soft e01aa92` the index showed hundreds of stale staged deletions/additions because the working tree did not match HEAD.
- **Fix:** `git reset HEAD` to clear the index, then `git checkout HEAD -- .` to repopulate the working tree from e01aa92. Required source files (`src/acoustic/training/augmentation.py`, `requirements-vertex.txt`) were restored. Each subsequent commit then staged only the specific Task files.
- **Files:** N/A (git plumbing)
- **Commit:** N/A (corrected before any commit landed)

**3. [Rule 3 - Blocking] Editable install pointed at the main repo, not the worktree**
- **Found during:** Task 1 GREEN test run
- **Issue:** `python -c "import acoustic.training.augmentation as a; print(a.__file__)"` resolved to `/Users/guyelisha/Projects/sky-fort-acoustic/src/...` (main repo) instead of the worktree path. New code was not visible to pytest.
- **Fix:** Ran pytest with `PYTHONPATH=src` to force resolution against the worktree's `src/`. Documented in this SUMMARY so the orchestrator's hook stage can use the same flag if needed.
- **Files:** N/A
- **Commit:** N/A

## Threat Flags

None. New surface (`pyroomacoustics`) is mitigated per the plan's threat register: pinned to `>=0.8,<0.11`, runtime-bounded by `max_order=10` and 1-second RIR truncation.

## Self-Check: PASSED

- FOUND: src/acoustic/training/augmentation.py (WideGainAugmentation @ line 251, RoomIRAugmentation @ line 287)
- FOUND: tests/unit/test_wide_gain_augmentation.py
- FOUND: tests/unit/test_room_ir_augmentation.py
- FOUND: requirements.txt (pyroomacoustics line 7)
- FOUND: requirements-vertex.txt (pyroomacoustics line 7)
- FOUND: commit eb92958 (WideGain RED)
- FOUND: commit 43a5149 (WideGain GREEN + deps)
- FOUND: commit 37132b4 (RoomIR RED)
- FOUND: commit 1529974 (RoomIR GREEN)
- 11 tests passing
