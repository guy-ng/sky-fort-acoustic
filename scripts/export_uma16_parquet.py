"""Export ingested UMA-16 WAVs to Parquet shards in the trainer's schema.

The existing ``acoustic.training.parquet_dataset.ParquetDataset`` consumer
expects shards named ``train-*.parquet`` with two columns:

  audio  : struct<bytes: binary, path: string>     # raw WAV bytes (44-byte hdr + PCM_16)
  label  : int64                                   # 0 = no_drone / ambient, 1 = drone

This script reads:
  data/field/uma16_ambient/  -> data/parquet/ambient/train-0.parquet (label=0)
  data/eval/uma16_real/      -> data/parquet/eval/train-0.parquet    (label=0|1)

Why parquet:
  - Single file per dataset = single Docker COPY layer = better build-cache reuse.
  - ~50% smaller than the int16 WAVs after columnar + zstd compression.
  - Loadable directly by ParquetDatasetBuilder without an additional file scan.

The original WAV trees are NOT deleted — BackgroundNoiseMixer in plan 20-02
may prefer per-file random access. The parquet shards are an additional
artifact for the Docker bake path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
AMBIENT_SRC = REPO_ROOT / "data" / "field" / "uma16_ambient"
EVAL_SRC = REPO_ROOT / "data" / "eval" / "uma16_real"
EVAL_LABELS_PATH = EVAL_SRC / "labels.json"

OUT_ROOT = REPO_ROOT / "data" / "parquet"


def read_wav_bytes(path: Path) -> bytes:
    return path.read_bytes()


def export_ambient() -> Path:
    """Export ambient WAVs as a single parquet shard with label=0."""
    files = sorted(AMBIENT_SRC.rglob("*.wav"))
    if not files:
        msg = f"No ambient WAVs found under {AMBIENT_SRC}"
        raise FileNotFoundError(msg)

    audio_structs = []
    labels = []
    for f in files:
        audio_structs.append(
            {
                "bytes": read_wav_bytes(f),
                "path": str(f.relative_to(AMBIENT_SRC)),
            }
        )
        labels.append(0)  # ambient = no_drone

    table = pa.table(
        {
            "audio": audio_structs,
            "label": pa.array(labels, type=pa.int64()),
        }
    )

    dst_dir = OUT_ROOT / "ambient"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "train-0.parquet"
    pq.write_table(table, dst, compression="zstd")
    return dst


def export_eval() -> Path:
    """Export eval WAVs (with drone/no_drone labels from labels.json)."""
    if not EVAL_LABELS_PATH.exists():
        msg = f"labels.json not found at {EVAL_LABELS_PATH}"
        raise FileNotFoundError(msg)

    with EVAL_LABELS_PATH.open() as fh:
        entries = json.load(fh)

    audio_structs = []
    labels = []
    for entry in entries:
        rel_audio = entry["file"]  # e.g. "audio/drone/foo.wav"
        wav_path = EVAL_SRC / rel_audio
        if not wav_path.exists():
            msg = f"missing eval WAV: {wav_path}"
            raise FileNotFoundError(msg)
        audio_structs.append(
            {
                "bytes": read_wav_bytes(wav_path),
                "path": rel_audio,
            }
        )
        labels.append(1 if entry["label"] == "drone" else 0)

    table = pa.table(
        {
            "audio": audio_structs,
            "label": pa.array(labels, type=pa.int64()),
        }
    )

    dst_dir = OUT_ROOT / "eval"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "train-0.parquet"
    pq.write_table(table, dst, compression="zstd")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ambient-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        out = export_ambient()
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"ambient -> {out} ({size_mb:.1f} MB)")

    if not args.ambient_only:
        out = export_eval()
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"eval    -> {out} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
