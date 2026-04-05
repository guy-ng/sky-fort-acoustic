"""Research-validated mel-spectrogram preprocessor using torchaudio.

Implements the Preprocessor protocol. All parameters come from MelConfig.
Replaces the EfficientNet preprocessing pipeline (D-01).
Uses torchaudio instead of librosa (D-06).
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from acoustic.classification.config import MelConfig


def pad_or_loop(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad short audio by looping (tiling) instead of zero-padding.

    If audio is shorter than target_len, tiles it to fill the segment.
    If audio is already long enough, returns it unchanged.
    """
    if len(audio) >= target_len:
        return audio
    repeats = (target_len // len(audio)) + 1
    return np.tile(audio, repeats)[:target_len]


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


class RawAudioPreprocessor:
    """Resampling preprocessor for models that handle their own mel/features.

    Resamples from the pipeline sample rate (e.g. 48kHz) to the model's
    expected rate (default 32kHz for EfficientAT) and returns a 1-D tensor.
    """

    def __init__(self, target_sr: int = 32000) -> None:
        self._target_sr = target_sr
        self._resampler: torchaudio.transforms.Resample | None = None
        self._cached_sr: int | None = None

    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        waveform = torch.from_numpy(audio).float()
        if sr != self._target_sr:
            if self._cached_sr != sr:
                self._resampler = torchaudio.transforms.Resample(sr, self._target_sr)
                self._cached_sr = sr
            waveform = self._resampler(waveform)
        return waveform


class ResearchPreprocessor:
    """Mel-spectrogram preprocessor matching the research code exactly.

    Produces (1, 1, max_frames, n_mels) tensors with (S_db+80)/80
    normalization clipped to [0, 1].

    Matches librosa.feature.melspectrogram + librosa.power_to_db(ref=np.max)
    exactly by using torchaudio MelSpectrogram with librosa-compatible defaults
    and per-spectrogram max-reference dB normalization.

    Satisfies the Preprocessor protocol.
    """

    def __init__(self, config: MelConfig | None = None) -> None:
        self._config = config or MelConfig()
        c = self._config
        # Match librosa defaults: norm="slaney", mel_scale="slaney", center=True
        self._mel_spec = T.MelSpectrogram(
            sample_rate=c.sample_rate,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            n_mels=c.n_mels,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
            center=True,
            pad_mode="constant",
        )

    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Process raw mono audio into a model-ready mel-spectrogram tensor.

        Args:
            audio: 1-D float32 mono audio array.
            sr: Sample rate of input audio.

        Returns:
            Tensor of shape (1, 1, max_frames, n_mels) with values in [0, 1].
        """
        c = self._config
        waveform = torch.from_numpy(audio).float()

        # Resample if needed
        if sr != c.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, c.sample_rate)

        # Take last segment_samples or zero-pad at the beginning
        n = c.segment_samples
        if waveform.shape[0] >= n:
            waveform = waveform[-n:]
        else:
            waveform = torch.nn.functional.pad(waveform, (n - waveform.shape[0], 0))

        # Mel spectrogram (power)
        S = self._mel_spec(waveform)  # (n_mels, frames)

        # power_to_db with ref=max (matches librosa.power_to_db(S, ref=np.max))
        S_db = _power_to_db(S, top_db=c.db_range)

        # Normalize to [0, 1]: (S_db + db_range) / db_range
        S_norm = (S_db + c.db_range) / c.db_range
        S_norm = S_norm.clamp(0.0, 1.0)

        # Transpose to (frames, n_mels) then pad/trim to max_frames
        spec = S_norm.T  # (frames, n_mels)
        frames = spec.shape[0]
        if frames < c.max_frames:
            pad_amount = c.max_frames - frames
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_amount))
        elif frames > c.max_frames:
            # Center-crop
            start = (frames - c.max_frames) // 2
            spec = spec[start : start + c.max_frames]

        # Shape: (1, 1, max_frames, n_mels) = (1, 1, 128, 64)
        return spec.unsqueeze(0).unsqueeze(0)


def _power_to_db(S: torch.Tensor, *, top_db: float = 80.0) -> torch.Tensor:
    """Convert power spectrogram to dB, matching librosa.power_to_db(ref=np.max).

    Uses per-spectrogram max as reference (ref=np.max in librosa terms),
    then clips to -top_db floor.
    """
    ref = S.max()
    # Avoid log(0)
    ref = torch.clamp(ref, min=1e-10)
    S = torch.clamp(S, min=1e-10)
    S_db = 10.0 * torch.log10(S / ref)
    S_db = torch.clamp(S_db, min=-top_db)
    return S_db
