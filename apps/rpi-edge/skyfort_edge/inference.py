"""ONNX Runtime classifier with int8-preferred + FP32 fallback (D-05/D-07).

Model contract (from 21-03-SUMMARY):
  input  'mel'    float32 shape [batch, 1, 128, 100]
  output 'logits' float32 shape [batch, 1]   (binary sigmoid head, num_classes=1)

The Pi app consumes scalar drone probability via sigmoid(logits); the caller is
responsible for the activation step because some callers want raw logits for
calibration / metrics.

T-21-05 mitigation: _verify_checksum compares each candidate .onnx file against
the committed sha256 manifest (models/efficientat_mn10_v6_onnx.sha256) before
handing it to onnxruntime. Mismatch triggers fallback, and if nothing loads we
raise RuntimeError rather than silently serving a tampered model.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

CHECKSUM_FILENAME = "efficientat_mn10_v6_onnx.sha256"


def _load_expected_checksums(checksum_file: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-compatible manifest into {basename: hex}."""
    if not checksum_file.exists():
        return {}
    result: dict[str, str] = {}
    for raw in checksum_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].strip(), parts[1].strip()
        # sha256sum output may prefix the filename with '*' for binary mode.
        if name.startswith("*"):
            name = name[1:]
        result[name] = digest
    return result


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(onnx_path: Path) -> bool:
    """Return True if the file matches its recorded sha256, or there is no manifest.

    A missing manifest logs a warning and allows the load (dev workflows). A
    manifest that does not list this file also logs a warning and allows load.
    A recorded hash that disagrees with the on-disk bytes fails closed.
    """
    checksum_file = onnx_path.parent / CHECKSUM_FILENAME
    expected = _load_expected_checksums(checksum_file)
    if not expected:
        log.warning("no checksum file at %s -- skipping tamper check", checksum_file)
        return True
    name = onnx_path.name
    if name not in expected:
        log.warning("no checksum recorded for %s in %s", name, checksum_file)
        return True
    actual = _sha256_of(onnx_path)
    if actual != expected[name]:
        log.error(
            "SHA256 MISMATCH for %s: expected %s got %s",
            name,
            expected[name],
            actual,
        )
        return False
    return True


class OnnxClassifier:
    """Loads the int8 ONNX with FP32 fallback, runs single-window inference."""

    def __init__(self, model_cfg) -> None:
        import onnxruntime as ort

        self._session = None
        self._active_path: Optional[Path] = None

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(getattr(model_cfg, "num_threads", 2))
        sess_opts.inter_op_num_threads = 1
        providers = [getattr(model_cfg, "execution_provider", "CPUExecutionProvider")]

        prefer_int8 = bool(getattr(model_cfg, "prefer_int8", True))
        primary_path = Path(model_cfg.onnx_path) if prefer_int8 else Path(model_cfg.fallback_onnx_path)
        fallback_path = Path(model_cfg.fallback_onnx_path) if prefer_int8 else Path(model_cfg.onnx_path)

        candidates: list[tuple[Path, str]] = [(primary_path, "primary")]
        if fallback_path != primary_path:
            candidates.append((fallback_path, "fallback"))

        last_error: Optional[Exception] = None
        for candidate, tag in candidates:
            if not candidate.exists():
                log.warning("%s ONNX path does not exist: %s", tag, candidate)
                continue
            if not _verify_checksum(candidate):
                log.warning(
                    "%s ONNX checksum mismatch at %s -- refusing to load",
                    tag,
                    candidate,
                )
                continue
            try:
                self._session = ort.InferenceSession(
                    str(candidate), sess_options=sess_opts, providers=providers
                )
                self._active_path = candidate
                log.info(
                    "Loaded ONNX model: %s (provider=%s, tag=%s)",
                    candidate,
                    providers[0],
                    tag,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on ORT env
                last_error = exc
                log.warning("failed to load %s ONNX (%s): %s", tag, candidate, exc)

        if self._session is None:
            raise RuntimeError(
                f"Could not load any ONNX model. Tried primary={primary_path}, "
                f"fallback={fallback_path}. Last error: {last_error!r}"
            )

        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        self._input_name = inp.name
        self._output_name = out.name
        last_dim = out.shape[-1] if out.shape else None
        self._num_classes = last_dim if isinstance(last_dim, int) else None

    @property
    def active_model_path(self) -> Path:
        assert self._active_path is not None
        return self._active_path

    @property
    def num_classes(self) -> Optional[int]:
        return self._num_classes

    def classify(self, mel: np.ndarray) -> np.ndarray:
        """Run one inference on a (128, T) mel spectrogram.

        Returns a 1-D float32 array of logits of length ``num_classes`` (for the
        binary sigmoid head this is length 1). Callers applying sigmoid get the
        scalar drone probability.
        """
        if mel.ndim != 2 or mel.shape[0] != 128:
            raise ValueError(f"expected (128, T) mel, got shape {mel.shape}")
        batch = mel[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
        logits = self._session.run([self._output_name], {self._input_name: batch})[0]
        return np.asarray(logits[0], dtype=np.float32).reshape(-1)
