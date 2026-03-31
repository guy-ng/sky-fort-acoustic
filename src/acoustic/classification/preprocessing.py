"""Mel-spectrogram preprocessing pipeline for CNN drone classification.

Ports the POC's exact parameters to ensure model compatibility.
"""

from __future__ import annotations

import math

import librosa
import numpy as np
from scipy.signal import resample_poly

# POC-matching constants (from uma16_master_live_with_polar.py)
SR_CNN = 16000
CNN_SEGMENT_SECONDS = 2.0
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
MAX_FRAMES = 128


def fast_resample(y: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample audio from fs_in to fs_out using polyphase filtering.

    Returns float32 array. If fs_in == fs_out, returns input unchanged.
    """
    if fs_in == fs_out:
        return y
    g = math.gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(y, up, down).astype(np.float32)


def make_melspec(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel spectrogram in dB, transposed to (frames, n_mels).

    Matches POC parameters exactly: n_fft=1024, hop=256, 64 mels, power=2.0.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T  # (frames, n_mels)


def pad_or_trim(spec: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    """Pad or truncate spectrogram to exactly max_frames along axis 0."""
    n = spec.shape[0]
    if n < max_frames:
        pad_width = ((0, max_frames - n), (0, 0))
        return np.pad(spec, pad_width, mode="constant", constant_values=0.0)
    return spec[:max_frames]


def norm_spec(spec: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    mean = spec.mean()
    std = spec.std()
    return (spec - mean) / (std + 1e-8)


def preprocess_for_cnn(mono_audio: np.ndarray, fs_in: int) -> np.ndarray:
    """Full preprocessing pipeline: resample, segment, melspec, normalize, reshape.

    Args:
        mono_audio: 1-D float audio signal (any sample rate).
        fs_in: Input sample rate (Hz).

    Returns:
        Array of shape (1, 128, 64, 1) float32 ready for ONNX inference.
    """
    # Resample to CNN sample rate
    y = fast_resample(mono_audio, fs_in, SR_CNN)

    # Take or pad last CNN_SEGMENT_SECONDS
    n_samples = int(SR_CNN * CNN_SEGMENT_SECONDS)
    if len(y) >= n_samples:
        y = y[-n_samples:]
    else:
        y = np.pad(y, (n_samples - len(y), 0), mode="constant").astype(np.float32)

    # Mel spectrogram
    spec = make_melspec(y, SR_CNN)

    # Pad or trim to MAX_FRAMES
    spec = pad_or_trim(spec, MAX_FRAMES)

    # Normalize
    spec = norm_spec(spec)

    # Reshape for NHWC: (1, 128, 64, 1)
    return spec.reshape(1, MAX_FRAMES, N_MELS, 1).astype(np.float32)
