"""Phase 22 data integrity preflight (REQ-22-D3).

Runs inside Vertex jobs BEFORE any DataLoader is constructed. Fails fast if
any 2026-04-08 field recording is missing, decodes to wrong SR/channels,
contains NaN/Inf, is empty, or if cardinality doesn't match the frozen
holdout split.

Usage:
    python scripts/preflight_v8_data.py

Exit codes:
    0 — all files present, decode cleanly, SR=16000 mono, cardinality matches
    1 — at least one integrity check failed (see logs)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

_log = logging.getLogger("preflight_v8_data")

EXPECTED_SR = 16000
EXPECTED_CHANNELS = 1
DRONE_DIR = Path("data/field/drone")
BG_DIR = Path("data/field/background")

# --- Frozen holdout split (Phase 22 Research Focus 5) ---------------------
# ONE file per drone sub-class + the longest background. Frozen here in code,
# NOT in config, so a PR reviewer sees any change to the split.
HOLDOUT_FILES: frozenset[str] = frozenset({
    "20260408_091054_136dc5.wav",  # 10inch 4kg — hardest payload condition
    "20260408_092615_1a055f.wav",  # 10inch heavy — longest drone clip
    "20260408_091724_bb0ed8.wav",  # phantom 4 — non-FPV diversity
    "20260408_084222_44dc5c.wav",  # 5inch — smallest prop class
    "20260408_090757_1c50e9.wav",  # background — only sizeable bg (104s)
})

# Cardinality expectations (from Research Focus 4 inventory)
TOTAL_DRONE_FILES = 13
TOTAL_BG_FILES = 4
EXPECTED_TRAIN_DRONE = TOTAL_DRONE_FILES - 4  # 9
EXPECTED_TRAIN_BG = TOTAL_BG_FILES - 1         # 3
EXPECTED_HOLDOUT_DRONE = 4
EXPECTED_HOLDOUT_BG = 1


def _check_wav(
    wav: Path, errors: list[str], *, label: str,
) -> tuple[int, float] | None:
    if not wav.exists():
        errors.append(f"missing file: {wav}")
        return None
    try:
        info = sf.info(str(wav))
    except Exception as exc:
        errors.append(f"sf.info failed for {wav}: {exc}")
        return None
    if info.samplerate != EXPECTED_SR:
        errors.append(f"{wav}: sr={info.samplerate} expected {EXPECTED_SR}")
        return None
    if info.channels != EXPECTED_CHANNELS:
        errors.append(f"{wav}: channels={info.channels} expected {EXPECTED_CHANNELS}")
        return None
    try:
        audio, sr = sf.read(str(wav), dtype="float32")
    except Exception as exc:
        errors.append(f"sf.read failed for {wav}: {exc}")
        return None
    if audio.size == 0:
        errors.append(f"{wav}: empty audio")
        return None
    if not np.isfinite(audio).all():
        errors.append(f"{wav}: NaN or Inf in audio")
        return None
    return int(audio.shape[0]), float(audio.shape[0] / sr)


def preflight_field_recordings(
    drone_dir: Path = DRONE_DIR,
    bg_dir: Path = BG_DIR,
    holdout_files: Iterable[str] = HOLDOUT_FILES,
) -> dict[str, list[tuple[Path, int, float]]]:
    """Validate field recordings and return per-class manifest (excluding holdout).

    Raises AssertionError on any integrity failure.
    """
    holdout_set = set(holdout_files)
    manifest: dict[str, list[tuple[Path, int, float]]] = {"drone": [], "background": []}
    errors: list[str] = []

    for label, subdir in [("drone", drone_dir), ("background", bg_dir)]:
        if not subdir.is_dir():
            errors.append(f"missing dir: {subdir}")
            continue
        for wav in sorted(subdir.glob("20260408_*.wav")):
            if wav.name in holdout_set:
                _log.info("HOLDOUT (excluded from training): %s", wav.name)
                continue
            result = _check_wav(wav, errors, label=label)
            if result is not None:
                manifest[label].append((wav, result[0], result[1]))

    # Cardinality after holdout removal
    if len(manifest["drone"]) != EXPECTED_TRAIN_DRONE:
        errors.append(
            f"drone training count: {len(manifest['drone'])} != {EXPECTED_TRAIN_DRONE}"
        )
    if len(manifest["background"]) != EXPECTED_TRAIN_BG:
        errors.append(
            f"bg training count: {len(manifest['background'])} != {EXPECTED_TRAIN_BG}"
        )

    if errors:
        for e in errors:
            _log.error("PREFLIGHT: %s", e)
        raise AssertionError(
            f"Phase 22 preflight failed with {len(errors)} errors: {errors}"
        )

    total_drone_s = sum(d for _, _, d in manifest["drone"])
    total_bg_s = sum(d for _, _, d in manifest["background"])
    _log.info(
        "PREFLIGHT OK: drone=%d files (%.1fs), bg=%d files (%.1fs)",
        len(manifest["drone"]), total_drone_s,
        len(manifest["background"]), total_bg_s,
    )
    return manifest


def preflight_holdout(
    drone_dir: Path = DRONE_DIR, bg_dir: Path = BG_DIR,
) -> dict[str, list[tuple[Path, int, float]]]:
    """Validate holdout files separately. Used by eval harness."""
    manifest: dict[str, list[tuple[Path, int, float]]] = {"drone": [], "background": []}
    errors: list[str] = []
    for name in HOLDOUT_FILES:
        drone_path = drone_dir / name
        bg_path = bg_dir / name
        if drone_path.exists():
            r = _check_wav(drone_path, errors, label="drone")
            if r is not None:
                manifest["drone"].append((drone_path, r[0], r[1]))
        elif bg_path.exists():
            r = _check_wav(bg_path, errors, label="background")
            if r is not None:
                manifest["background"].append((bg_path, r[0], r[1]))
        else:
            errors.append(f"holdout file not found in drone or bg dir: {name}")
    if errors:
        raise AssertionError(f"Phase 22 holdout preflight failed: {errors}")
    assert len(manifest["drone"]) == EXPECTED_HOLDOUT_DRONE, (
        f"holdout drone count: {len(manifest['drone'])} != {EXPECTED_HOLDOUT_DRONE}"
    )
    assert len(manifest["background"]) == EXPECTED_HOLDOUT_BG, (
        f"holdout bg count: {len(manifest['background'])} != {EXPECTED_HOLDOUT_BG}"
    )
    # Trimmed-file sentinel check
    trimmed = drone_dir / "20260408_091054_136dc5.wav"
    if trimmed.exists():
        info = sf.info(str(trimmed))
        dur = info.frames / info.samplerate
        assert 60.0 < dur < 62.0, (
            f"20260408_091054_136dc5.wav duration {dur:.2f}s out of trimmed range "
            f"(expected ~61.4s). Was the .bak restored?"
        )
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        train_manifest = preflight_field_recordings()
        holdout_manifest = preflight_holdout()
    except AssertionError as exc:
        _log.error("PREFLIGHT FAILED: %s", exc)
        return 1
    print(
        f"OK — training: {len(train_manifest['drone'])} drone + "
        f"{len(train_manifest['background'])} bg | "
        f"holdout: {len(holdout_manifest['drone'])} drone + "
        f"{len(holdout_manifest['background'])} bg"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
