"""Ingest the Acoustic-UAV-Identification dataset into Phase 20 data trees.

The source dataset (Acoustic-UAV-Identification-main-main/audio-data/) was
captured with the same UMA-16v2 hardware Phase 20 targets. Files are stored
**one mic per WAV file** at 48 kHz, mono PCM_16. We resample each file to
16 kHz mono and copy it into:

  - data/field/uma16_ambient/outdoor_quiet/  (D-09 ambient pool, training-time noise)
  - data/eval/uma16_real/audio/              (D-27 promotion eval set)

Why mic01 + mic03 only:
  Phase 20 forbids "joining" multiple UMA-16 channels into a single mono
  WAV — summing/averaging across the array creates comb filtering from
  per-channel time-of-arrival differences, which corrupts the spectrum.
  Each input file in the source dataset is already a single mic channel,
  so it is safe to use as-is. We pull mic01 (fullest coverage) and, for
  ambient, also mic03 to push the ambient pool past the D-09 30-min floor.
  We do NOT mix mic01 + mic03 samples together — they remain independent
  WAV files in the dataset.

Idempotent: re-running skips files whose target already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "Acoustic-UAV-Identification-main-main" / "audio-data" / "data"

AMBIENT_DST = REPO_ROOT / "data" / "field" / "uma16_ambient" / "outdoor_quiet"
EVAL_AUDIO_DST = REPO_ROOT / "data" / "eval" / "uma16_real" / "audio"
EVAL_LABELS = REPO_ROOT / "data" / "eval" / "uma16_real" / "labels.json"

SRC_SR = 48000
DST_SR = 16000


def resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Polyphase resample 48 kHz mono float32 -> 16 kHz mono float32."""
    if src_sr == DST_SR:
        return audio.astype(np.float32, copy=False)
    if src_sr != SRC_SR:
        raise ValueError(f"Unexpected source sample rate {src_sr}, expected {SRC_SR}")
    # 48000 / 16000 = 3 -> down=3, up=1
    out = resample_poly(audio.astype(np.float64), up=1, down=3)
    return out.astype(np.float32)


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        # Should never happen for this dataset, but be defensive: take channel 0
        audio = audio[:, 0]
    return audio, sr


def write_pcm16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, DST_SR, subtype="PCM_16")


def iter_mic_files(root: Path, mic_ids: list[str]) -> list[Path]:
    """Recursively find all WAV files matching one of the given _micNN suffixes."""
    out: list[Path] = []
    for p in root.rglob("*.wav"):
        name = p.name
        if "_mic" not in name:
            continue
        mic = name.split("_mic")[1][:2]
        if mic in mic_ids:
            out.append(p)
    return sorted(out)


def ingest_ambient(dry_run: bool, max_minutes: float | None = None) -> tuple[int, float]:
    """Resample mic01+mic03 background WAVs into the ambient outdoor_quiet pool."""
    src_files = iter_mic_files(SRC_ROOT / "background", mic_ids=["01", "03"])
    n_written = 0
    total_seconds = 0.0
    for src in src_files:
        if max_minutes is not None and total_seconds >= max_minutes * 60:
            break
        rel_name = src.name  # already unique by take + mic
        dst = AMBIENT_DST / rel_name
        if dst.exists():
            try:
                total_seconds += sf.info(str(dst)).duration
            except Exception:
                pass
            continue
        if dry_run:
            n_written += 1
            try:
                total_seconds += sf.info(str(src)).duration / 3.0  # rough
            except Exception:
                pass
            continue
        audio, sr = load_mono(src)
        audio_16k = resample_to_16k(audio, sr)
        write_pcm16(dst, audio_16k)
        n_written += 1
        total_seconds += len(audio_16k) / DST_SR
    return n_written, total_seconds


def ingest_eval(
    dry_run: bool,
    drone_minutes_target: float = 6.0,
    ambient_minutes_target: float = 16.0,
) -> tuple[list[dict], float, float]:
    """Resample mic01 drone+background into eval/uma16_real, generating labels.json entries.

    Each entry: {file, label, start_s, end_s, source_take}
    The contiguous-segment per WAV file approach: each input WAV becomes one
    entry whose start_s/end_s span its full duration.
    """
    entries: list[dict] = []
    drone_seconds = 0.0
    ambient_seconds = 0.0

    # Drone first (smaller target, easier to saturate)
    drone_src = iter_mic_files(SRC_ROOT / "drone", mic_ids=["01"])
    for src in drone_src:
        if drone_seconds >= drone_minutes_target * 60:
            break
        dst = EVAL_AUDIO_DST / "drone" / src.name
        if not dst.exists():
            if not dry_run:
                audio, sr = load_mono(src)
                audio_16k = resample_to_16k(audio, sr)
                write_pcm16(dst, audio_16k)
                dur_s = len(audio_16k) / DST_SR
            else:
                dur_s = sf.info(str(src)).duration / 3.0
        else:
            dur_s = sf.info(str(dst)).duration
        entries.append(
            {
                "file": str(dst.relative_to(EVAL_AUDIO_DST.parent)),
                "label": "drone",
                "start_s": 0.0,
                "end_s": float(dur_s),
                "source_take": src.stem,
            }
        )
        drone_seconds += dur_s

    # Then ambient
    ambient_src = iter_mic_files(SRC_ROOT / "background", mic_ids=["01"])
    for src in ambient_src:
        if ambient_seconds >= ambient_minutes_target * 60:
            break
        dst = EVAL_AUDIO_DST / "no_drone" / src.name
        if not dst.exists():
            if not dry_run:
                audio, sr = load_mono(src)
                audio_16k = resample_to_16k(audio, sr)
                write_pcm16(dst, audio_16k)
                dur_s = len(audio_16k) / DST_SR
            else:
                dur_s = sf.info(str(src)).duration / 3.0
        else:
            dur_s = sf.info(str(dst)).duration
        entries.append(
            {
                "file": str(dst.relative_to(EVAL_AUDIO_DST.parent)),
                "label": "no_drone",
                "start_s": 0.0,
                "end_s": float(dur_s),
                "source_take": src.stem,
            }
        )
        ambient_seconds += dur_s

    return entries, drone_seconds, ambient_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ambient-max-min",
        type=float,
        default=None,
        help="Cap ambient ingest at N minutes (default: ingest all mic01+mic03 background)",
    )
    parser.add_argument("--drone-eval-min", type=float, default=6.0)
    parser.add_argument("--ambient-eval-min", type=float, default=16.0)
    args = parser.parse_args()

    if not SRC_ROOT.exists():
        print(f"ERROR: source dataset not found at {SRC_ROOT}", file=sys.stderr)
        return 2

    print(f"Source: {SRC_ROOT}")
    print(f"Ambient dst: {AMBIENT_DST}")
    print(f"Eval dst:    {EVAL_AUDIO_DST}")
    print(f"Eval labels: {EVAL_LABELS}")
    print(f"Resample:    {SRC_SR} Hz -> {DST_SR} Hz (polyphase)")
    print()

    print("[1/2] Ingesting ambient (mic01 + mic03 background) ...")
    n_amb, sec_amb = ingest_ambient(dry_run=args.dry_run, max_minutes=args.ambient_max_min)
    print(f"  -> {n_amb} files, {sec_amb / 60:.1f} min")
    if sec_amb < 30 * 60:
        print(f"  WARNING: ambient pool is {sec_amb/60:.1f} min, below D-09 floor of 30 min")

    print()
    print("[2/2] Ingesting eval set (mic01 drone + mic01 background) ...")
    entries, sec_drone, sec_ambient_eval = ingest_eval(
        dry_run=args.dry_run,
        drone_minutes_target=args.drone_eval_min,
        ambient_minutes_target=args.ambient_eval_min,
    )
    print(f"  -> {len(entries)} entries; drone={sec_drone/60:.1f} min, no_drone={sec_ambient_eval/60:.1f} min")
    if sec_drone < 5 * 60:
        print(f"  WARNING: eval drone={sec_drone/60:.1f} min, below D-27 floor of 5 min")
    if sec_ambient_eval < 15 * 60:
        print(f"  WARNING: eval no_drone={sec_ambient_eval/60:.1f} min, below D-27 floor of 15 min")

    if not args.dry_run:
        EVAL_LABELS.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_LABELS.open("w") as fh:
            json.dump(entries, fh, indent=2)
        print(f"  -> wrote {EVAL_LABELS} ({len(entries)} entries)")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
