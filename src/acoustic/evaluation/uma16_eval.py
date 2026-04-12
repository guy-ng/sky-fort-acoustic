"""Phase 22: UMA-16 real-device evaluation harness.

Evaluates an EfficientAT classifier on the frozen holdout manifest at
data/eval/uma16_real_v8/manifest.json. Crucially, runs through the same
inference code path (EfficientATClassifier.predict) that the live service
uses — so any train/serve drift surfaces here.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F_audio

from acoustic.classification.efficientat.window_contract import (
    EFFICIENTAT_SEGMENT_SAMPLES,
    EFFICIENTAT_TARGET_SR,
)

_log = logging.getLogger(__name__)


@dataclass
class FileMetrics:
    name: str
    klass: str
    num_segments: int
    num_positive_predictions: int  # prediction >= threshold
    mean_prob: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_on_uma16(
    classifier,  # EfficientATClassifier
    manifest_path: Path = Path("data/eval/uma16_real_v8/manifest.json"),
    *,
    threshold: float = 0.5,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Evaluate classifier on the UMA-16 holdout manifest.

    For each manifest file:
      1. Verify sha256 (unless verify_sha256=False)
      2. Load at source SR (16 kHz), resample to 32 kHz
      3. Slice into non-overlapping 1-second segments (32000 samples each)
      4. Run classifier.predict on each segment
      5. Binary threshold at `threshold` to get per-segment drone/not-drone
    Aggregate into real_TPR and real_FPR over all segments.

    Returns a metrics dict with keys:
      real_TPR, real_FPR, num_drone_segments, num_bg_segments,
      threshold, per_file (list[FileMetrics as dict])
    """
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]

    per_file: list[dict[str, Any]] = []
    drone_tp = 0
    drone_total = 0
    bg_fp = 0
    bg_total = 0

    for entry in files:
        src = Path(entry["source_path"])
        if verify_sha256:
            got = _sha256(src)
            if got != entry["sha256"]:
                raise RuntimeError(
                    f"sha256 mismatch for {src}: manifest={entry['sha256'][:12]}... "
                    f"actual={got[:12]}..."
                )
        audio, sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Resample to inference SR
        t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if sr != EFFICIENTAT_TARGET_SR:
            t = F_audio.resample(t, sr, EFFICIENTAT_TARGET_SR)

        # Non-overlapping 1s segments
        seg = EFFICIENTAT_SEGMENT_SAMPLES
        num_segs = t.shape[-1] // seg
        if num_segs == 0:
            _log.warning("%s: shorter than 1s after resample, skipping", src.name)
            continue

        positives = 0
        probs: list[float] = []
        for i in range(num_segs):
            chunk = t[i * seg : (i + 1) * seg]
            # Go through the INFERENCE code path — this is the whole point
            p = classifier.predict(chunk)
            probs.append(float(p))
            if p >= threshold:
                positives += 1

        per_file.append({
            "name": entry["name"],
            "class": entry["class"],
            "num_segments": num_segs,
            "num_positive_predictions": positives,
            "mean_prob": float(np.mean(probs)) if probs else 0.0,
        })

        if entry["class"] == "drone":
            drone_tp += positives
            drone_total += num_segs
        elif entry["class"] == "background":
            bg_fp += positives
            bg_total += num_segs

    real_tpr = drone_tp / drone_total if drone_total > 0 else 0.0
    real_fpr = bg_fp / bg_total if bg_total > 0 else 0.0

    return {
        "real_TPR": real_tpr,
        "real_FPR": real_fpr,
        "num_drone_segments": drone_total,
        "num_bg_segments": bg_total,
        "threshold": threshold,
        "per_file": per_file,
    }
