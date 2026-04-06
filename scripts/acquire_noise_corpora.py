"""Acquire noise corpora (ESC-50, UrbanSound8K, FSD50K subset) for Phase 20 training.

Phase 20.1 -- implements D-01..D-10 from
.planning/phases/20.1-acquire-noise-corpora-esc50-urbansound8k-fsd50k-subset-and-a/20.1-CONTEXT.md

Idempotent: re-running with data present is a no-op (checked via .acquired.json marker).

Licenses:
  ESC-50:        CC BY-NC 3.0 (per soundata loader; CONTEXT says 4.0 -- same family)
  UrbanSound8K:  CC BY-NC 4.0 (Zenodo)
  FSD50K:        CC-BY 4.0 (per-clip; see FSD50K.documentation for full attribution)

Requires `zip` on PATH for FSD50K multi-part zip reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# soundata is imported lazily inside acquire_*() so that `--help` and unit
# tests (which monkeypatch sys.modules['soundata']) do not require the dep
# to be installed. The operator installs it via `pip install -r requirements.txt`
# before running any acquisition subcommand.
try:
    import soundata  # type: ignore
except ImportError:  # pragma: no cover - exercised on hosts without the dep
    soundata = None  # type: ignore

LOG = logging.getLogger("acquire_noise_corpora")

REPO_ROOT = Path(__file__).resolve().parent.parent
NOISE_ROOT = REPO_ROOT / "data" / "noise"

REQUIRED_FREE_GB: dict[str, int] = {
    "esc50": 2,
    "urbansound8k": 14,
    "fsd50k": 30,
}

# D-09: 6 dominant FSD50K classes for the noise subset.
# Slugs MUST match FSD50K.ground_truth/vocabulary.csv. mids are informational.
FSD50K_TARGET_CLASSES: dict[str, str] = {
    "Wind": "/m/03m9d0z",
    "Rain": "/m/06mb1",
    "Traffic_noise_and_roadway_noise": "/m/0btp2",
    "Mechanical_fan": "/m/02x984l",
    "Engine": "/m/02mk9",
    "Bird": "/m/015p6",
}

MARKER_SCHEMA_VERSION = 1


def _require_soundata():
    if soundata is None:
        raise RuntimeError(
            "soundata is not installed. Run: pip install -r requirements.txt"
        )
    return soundata


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_wavs(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.wav"))


def _total_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.wav"))


def is_already_acquired(corpus_dir: Path) -> bool:
    marker = corpus_dir / ".acquired.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    return _count_wavs(corpus_dir) == data.get("file_count")


def write_marker(corpus_dir: Path, corpus: str, source: str) -> None:
    marker = corpus_dir / ".acquired.json"
    payload = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "corpus": corpus,
        "source": source,
        "soundata_version": (
            getattr(soundata, "__version__", "unknown") if soundata is not None else "unknown"
        ),
        "file_count": _count_wavs(corpus_dir),
        "total_bytes": _total_bytes(corpus_dir),
        "acquired_at": _utcnow_iso(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    LOG.info(
        "wrote marker %s (%d files, %d bytes)",
        marker,
        payload["file_count"],
        payload["total_bytes"],
    )


def check_disk_space(path: Path, corpus: str) -> None:
    required_gb = REQUIRED_FREE_GB[corpus]
    path.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space for {corpus}: "
            f"{free_gb:.1f} GB free at {path}, need >= {required_gb} GB. "
            f"Free some space and re-run."
        )


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy src to dst with path-traversal guard rooted at NOISE_ROOT."""
    dst_resolved = dst.resolve()
    if not dst_resolved.is_relative_to(NOISE_ROOT.resolve()):
        raise RuntimeError(
            f"Refusing to copy outside NOISE_ROOT: {dst_resolved}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)  # NOT symlink -- Docker COPY breaks symlinks


def acquire_esc50(force: bool = False) -> Path:
    corpus_dir = NOISE_ROOT / "esc50"
    if not force and is_already_acquired(corpus_dir):
        LOG.info("esc50 already acquired at %s -- skipping", corpus_dir)
        return corpus_dir
    sd = _require_soundata()
    check_disk_space(corpus_dir, "esc50")
    LOG.info("downloading esc50 -> %s", corpus_dir)
    ds = sd.initialize("esc50", data_home=str(corpus_dir))
    ds.download(cleanup=True, force_overwrite=force)
    missing, invalid = ds.validate()
    if missing or invalid:
        raise RuntimeError(
            f"soundata.validate() for esc50: {len(missing)} missing, "
            f"{len(invalid)} invalid checksums. Re-run with --force."
        )
    write_marker(corpus_dir, "esc50", "soundata:esc50@1.0:github")
    return corpus_dir


def acquire_urbansound8k(force: bool = False) -> Path:
    corpus_dir = NOISE_ROOT / "urbansound8k"
    if not force and is_already_acquired(corpus_dir):
        LOG.info("urbansound8k already acquired at %s -- skipping", corpus_dir)
        return corpus_dir
    sd = _require_soundata()
    check_disk_space(corpus_dir, "urbansound8k")
    LOG.info("downloading urbansound8k -> %s (Zenodo)", corpus_dir)
    ds = sd.initialize("urbansound8k", data_home=str(corpus_dir))
    ds.download(cleanup=True, force_overwrite=force)
    missing, invalid = ds.validate()
    if missing or invalid:
        raise RuntimeError(
            f"soundata.validate() for urbansound8k: {len(missing)} missing, "
            f"{len(invalid)} invalid checksums. Re-run with --force."
        )
    write_marker(corpus_dir, "urbansound8k", "soundata:urbansound8k@1.0:zenodo")
    return corpus_dir


def _iter_fsd50k_clips(gt_csv: Path, audio_dir: Path):
    """Yield (wav_path, primary_label) for clips whose primary label is in target set."""
    with gt_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            primary = row["labels"].split(",", 1)[0].strip()
            if primary in FSD50K_TARGET_CLASSES:
                wav = audio_dir / f"{row['fname']}.wav"
                if wav.is_file():
                    yield wav, primary


def _iter_fsd50k_clips_secondary(gt_csv: Path, audio_dir: Path, target_label: str):
    """Yield (wav_path, target_label) for clips whose labels list CONTAINS target_label."""
    with gt_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels = [s.strip() for s in row["labels"].split(",")]
            if target_label in labels:
                wav = audio_dir / f"{row['fname']}.wav"
                if wav.is_file():
                    yield wav, target_label


def acquire_fsd50k(force: bool = False) -> Path:
    final_dir = NOISE_ROOT / "fsd50k_subset"
    if not force and is_already_acquired(final_dir):
        LOG.info("fsd50k_subset already acquired at %s -- skipping", final_dir)
        return final_dir
    sd = _require_soundata()
    check_disk_space(NOISE_ROOT, "fsd50k")
    stage_dir = NOISE_ROOT / ".fsd50k_stage"
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("downloading FSD50K -> %s (staging)", stage_dir)
        ds = sd.initialize("fsd50k", data_home=str(stage_dir))
        ds.download(
            partial_download=["FSD50K.dev_audio", "FSD50K.eval_audio", "ground_truth"],
            cleanup=True,
            force_overwrite=force,
        )
        missing, invalid = ds.validate()
        if missing or invalid:
            raise RuntimeError(
                f"soundata.validate() for fsd50k: {len(missing)} missing, "
                f"{len(invalid)} invalid checksums. Re-run with --force."
            )

        gt_root = stage_dir / "FSD50K.ground_truth"
        dev_csv = gt_root / "dev.csv"
        eval_csv = gt_root / "eval.csv"
        vocab_csv = gt_root / "vocabulary.csv"

        # Validate vocabulary contains all 6 target slugs
        vocab_slugs = set()
        with vocab_csv.open() as fh:
            for row in csv.reader(fh):
                if len(row) >= 2:
                    vocab_slugs.add(row[1].strip())
        missing_slugs = set(FSD50K_TARGET_CLASSES) - vocab_slugs
        if missing_slugs:
            raise RuntimeError(
                f"FSD50K vocabulary.csv missing target slugs: {sorted(missing_slugs)} -- "
                "upstream rename suspected. Update FSD50K_TARGET_CLASSES."
            )

        dev_audio = stage_dir / "FSD50K.dev_audio"
        eval_audio = stage_dir / "FSD50K.eval_audio"

        # Pass 1: primary-label copy
        per_class_count: dict[str, int] = {k: 0 for k in FSD50K_TARGET_CLASSES}
        for csv_path, audio_dir in [(dev_csv, dev_audio), (eval_csv, eval_audio)]:
            for wav, cls in _iter_fsd50k_clips(csv_path, audio_dir):
                target = final_dir / cls / wav.name
                _safe_copy(wav, target)
                per_class_count[cls] += 1

        # Pass 2: D-09 fallback for under-populated classes (Mechanical_fan most likely)
        for cls, count in list(per_class_count.items()):
            if count >= 100:
                continue
            LOG.warning(
                "fsd50k class %s has %d primary-label clips (<100); "
                "falling back to secondary-label match",
                cls,
                count,
            )
            for csv_path, audio_dir in [(dev_csv, dev_audio), (eval_csv, eval_audio)]:
                for wav, _ in _iter_fsd50k_clips_secondary(csv_path, audio_dir, cls):
                    target = final_dir / cls / wav.name
                    if target.exists():
                        continue
                    _safe_copy(wav, target)
                    per_class_count[cls] += 1

        for cls, count in per_class_count.items():
            LOG.info("fsd50k class %s: %d files copied", cls, count)

        write_marker(
            final_dir,
            "fsd50k_subset",
            "soundata:fsd50k@1.0:zenodo+filter:6class-primary-with-secondary-fallback",
        )
    finally:
        if stage_dir.exists():
            LOG.info("cleaning up FSD50K staging dir %s", stage_dir)
            shutil.rmtree(stage_dir, ignore_errors=True)
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire noise corpora for Phase 20 training (D-01)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass idempotency marker and re-download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override NOISE_ROOT (default: data/noise/)",
    )
    sub = parser.add_subparsers(dest="corpus", required=True)
    sub.add_parser("esc50")
    sub.add_parser("urbansound8k")
    sub.add_parser("fsd50k")
    sub.add_parser("all")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    global NOISE_ROOT
    if args.data_dir is not None:
        NOISE_ROOT = args.data_dir.resolve()
    NOISE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        if args.corpus == "esc50":
            acquire_esc50(force=args.force)
        elif args.corpus == "urbansound8k":
            acquire_urbansound8k(force=args.force)
        elif args.corpus == "fsd50k":
            acquire_fsd50k(force=args.force)
        elif args.corpus == "all":
            acquire_esc50(force=args.force)
            acquire_urbansound8k(force=args.force)
            acquire_fsd50k(force=args.force)
    except Exception as exc:
        LOG.error("acquisition failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
