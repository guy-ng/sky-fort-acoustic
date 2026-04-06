"""REST API endpoints for test pipeline: prepare and play DADS dataset samples."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test-pipeline", tags=["test-pipeline"])

SAMPLES_DIR = Path("data/test_samples")
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"
HF_REPO = "geronimobasso/drone-audio-detection-samples"

# DADS dataset: 16 kHz mono, int16 PCM
DADS_SR = 16000
TARGET_MIN_SEC = 3.0
TARGET_MAX_SEC = 5.0
NUM_DRONE = 20
NUM_BACKGROUND = 20


def _decode_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 array, skipping the 44-byte header."""
    return np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0


def _write_wav(path: Path, audio: np.ndarray, sr: int = DADS_SR) -> None:
    """Write float32 audio as 16-bit PCM WAV."""
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _prepare_samples() -> list[dict]:
    """Extract random drone/background samples from DADS via HuggingFace datasets.

    Uses streaming mode to avoid downloading the entire dataset. Iterates
    through shuffled rows, collecting samples that are >= 3 seconds long.
    """
    from datasets import Audio, load_dataset

    logger.info("Loading DADS dataset from HuggingFace (%s) in streaming mode...", HF_REPO)
    ds = load_dataset(HF_REPO, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=random.randint(0, 100000))

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    min_samples = int(TARGET_MIN_SEC * DADS_SR)
    manifest: list[dict] = []
    rng = random.Random()

    drone_collected = 0
    bg_collected = 0
    scanned = 0

    for row in ds:
        if drone_collected >= NUM_DRONE and bg_collected >= NUM_BACKGROUND:
            break

        scanned += 1
        label_val = row["label"]

        # Skip if we already have enough of this class
        if label_val == 1 and drone_collected >= NUM_DRONE:
            continue
        if label_val == 0 and bg_collected >= NUM_BACKGROUND:
            continue

        try:
            audio = _decode_wav_bytes(row["audio"]["bytes"])
        except Exception:
            continue

        if len(audio) < min_samples:
            continue

        max_possible = min(len(audio) / DADS_SR, TARGET_MAX_SEC)
        if max_possible < TARGET_MIN_SEC:
            continue

        duration = rng.uniform(TARGET_MIN_SEC, max_possible)
        n_samples = int(duration * DADS_SR)
        start = rng.randint(0, len(audio) - n_samples) if len(audio) > n_samples else 0
        segment = audio[start : start + n_samples]

        label_str = "drone" if label_val == 1 else "background"
        idx = drone_collected if label_val == 1 else bg_collected
        sample_id = f"{label_str}_{idx:03d}"
        wav_path = SAMPLES_DIR / f"{sample_id}.wav"
        _write_wav(wav_path, segment)

        manifest.append({
            "id": sample_id,
            "label": label_str,
            "duration_s": round(len(segment) / DADS_SR, 2),
            "filename": f"{sample_id}.wav",
        })

        if label_val == 1:
            drone_collected += 1
        else:
            bg_collected += 1

        logger.info("Prepared %s (%0.1fs) [scanned %d rows]", sample_id, len(segment) / DADS_SR, scanned)

    logger.info(
        "Done: %d drone + %d background samples from %d rows scanned",
        drone_collected, bg_collected, scanned,
    )

    if drone_collected < NUM_DRONE:
        logger.warning("Only found %d/%d drone samples with >= %.0fs audio", drone_collected, NUM_DRONE, TARGET_MIN_SEC)
    if bg_collected < NUM_BACKGROUND:
        logger.warning("Only found %d/%d background samples with >= %.0fs audio", bg_collected, NUM_BACKGROUND, TARGET_MIN_SEC)

    # Save manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


@router.post("/prepare")
async def prepare_test_samples() -> JSONResponse:
    """Prepare 20 drone + 20 background test samples from the DADS dataset.

    Extracts random 3-5 second clips and saves them as WAV files.
    Uses HuggingFace datasets streaming to avoid loading entire shards.
    """
    try:
        loop = asyncio.get_running_loop()
        manifest = await loop.run_in_executor(None, _prepare_samples)
    except Exception as exc:
        logger.exception("Failed to prepare test samples")
        return JSONResponse(status_code=500, content={"message": str(exc)})
    return JSONResponse(content={"samples": manifest, "count": len(manifest)})


def _known_good_samples() -> list[dict]:
    """Static "known good" samples that are always shown in the panel.

    These are clips from v6's training-similar distribution that we've
    verified DO fire detection. Useful as a baseline when DADS samples
    don't trigger and you need to confirm the live pipeline still works.
    Files live in data/test_samples/known_*.wav and ship with the repo
    rather than being generated by /prepare.
    """
    out: list[dict] = []
    for path in sorted(SAMPLES_DIR.glob("known_*.wav")):
        try:
            with wave.open(str(path), "rb") as wf:
                duration = round(wf.getnframes() / wf.getframerate(), 2)
        except Exception:
            duration = 0.0
        out.append({
            "id": path.stem,           # e.g. "known_dji_mini_2"
            "label": "drone",
            "duration_s": duration,
            "filename": path.name,
        })
    return out


@router.get("/samples")
async def list_test_samples() -> JSONResponse:
    """List prepared test samples (known-good clips first, then DADS samples)."""
    known = _known_good_samples()
    if not MANIFEST_PATH.exists():
        return JSONResponse(content={"samples": known, "count": len(known)})
    with open(MANIFEST_PATH) as f:
        dads = json.load(f)
    combined = known + dads
    return JSONResponse(content={"samples": combined, "count": len(combined)})


@router.get("/samples/{sample_id}/audio", response_model=None)
async def get_sample_audio(sample_id: str) -> FileResponse | JSONResponse:
    """Serve a test sample WAV file for browser playback."""
    wav_path = SAMPLES_DIR / f"{sample_id}.wav"
    if not wav_path.exists():
        return JSONResponse(status_code=404, content={"message": f"Sample {sample_id} not found"})
    return FileResponse(wav_path, media_type="audio/wav", filename=f"{sample_id}.wav")
