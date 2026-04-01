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

# Module-level cache for MelSpectrogram transforms, keyed by MelConfig (frozen/hashable).
_mel_spec_cache: dict[MelConfig, T.MelSpectrogram] = {}


def _get_mel_transform(config: MelConfig) -> T.MelSpectrogram:
    """Return a cached MelSpectrogram transform for the given config."""
    if config not in _mel_spec_cache:
        _mel_spec_cache[config] = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
            pad_mode="constant",
        )
    return _mel_spec_cache[config]


def mel_spectrogram_from_segment(
    segment: np.ndarray,
    config: MelConfig | None = None,
) -> torch.Tensor:
    """Convert a pre-extracted audio segment to a normalized mel-spectrogram.

    Produces identical output to ResearchPreprocessor.process() when given
    the same segment (of length config.segment_samples).

    Args:
        segment: 1-D float32 mono audio. Will be zero-padded if shorter
                 than config.segment_samples, trimmed if longer.
        config: MelConfig with preprocessing parameters.

    Returns:
        Tensor of shape (1, 1, max_frames, n_mels) with values in [0, 1].
    """
    c = config or MelConfig()
    n = c.segment_samples

    waveform = torch.from_numpy(segment).float()

    # Pad or trim to segment length (same logic as ResearchPreprocessor)
    if waveform.shape[0] >= n:
        waveform = waveform[-n:]
    else:
        waveform = torch.nn.functional.pad(waveform, (n - waveform.shape[0], 0))

    # Mel spectrogram (power)
    mel_transform = _get_mel_transform(c)
    S = mel_transform(waveform)  # (n_mels, frames)

    # power_to_db with ref=max
    S_db = _power_to_db(S, top_db=c.db_range)

    # Normalize to [0, 1]
    S_norm = (S_db + c.db_range) / c.db_range
    S_norm = S_norm.clamp(0.0, 1.0)

    # Transpose to (frames, n_mels) then pad/trim to max_frames
    spec = S_norm.T  # (frames, n_mels)
    frames = spec.shape[0]
    if frames < c.max_frames:
        pad_amount = c.max_frames - frames
        spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_amount))
    elif frames > c.max_frames:
        start = (frames - c.max_frames) // 2
        spec = spec[start : start + c.max_frames]

    # Shape: (1, 1, max_frames, n_mels) = (1, 1, 128, 64)
    return spec.unsqueeze(0).unsqueeze(0)


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


EFFICIENTNET_SIZE = 224

# RMS threshold below which audio is considered silence (skip CNN).
# -60 dBFS ≈ 0.001 RMS — well below any real drone signal.
SILENCE_RMS_THRESHOLD = 0.001


def preprocess_for_cnn(mono_audio: np.ndarray, fs_in: int) -> np.ndarray | None:
    """Full preprocessing pipeline: resample, segment, melspec, normalize, reshape.

    Args:
        mono_audio: 1-D float audio signal (any sample rate).
        fs_in: Input sample rate (Hz).

    Returns:
        Array of shape (1, 3, 224, 224) float32 ready for EfficientNet-B0 ONNX inference,
        or None if the audio is below the silence threshold.
    """
    from scipy.ndimage import zoom

    # Resample to CNN sample rate
    y = fast_resample(mono_audio, fs_in, SR_CNN)

    # Energy gate: skip CNN on silence to avoid false positives
    rms = np.sqrt(np.mean(y ** 2))
    if rms < SILENCE_RMS_THRESHOLD:
        return None

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

    # Resize to EfficientNet input: (128, 64) -> (224, 224)
    zoom_h = EFFICIENTNET_SIZE / spec.shape[0]
    zoom_w = EFFICIENTNET_SIZE / spec.shape[1]
    spec_resized = zoom(spec, (zoom_h, zoom_w), order=1).astype(np.float32)

    # NCHW with 3 channels (repeat grayscale): (1, 3, 224, 224)
    return np.stack([spec_resized] * 3)[np.newaxis].astype(np.float32)
