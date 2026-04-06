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


def _rms_normalize(
    audio,  # np.ndarray | torch.Tensor
    target: float = 0.1,
    eps: float = 1e-6,
):
    """Scale a waveform so its RMS equals ``target``.

    Shared helper used on BOTH the training dataset path AND
    ``RawAudioPreprocessor.process()`` so the model sees the same amplitude
    distribution at train and inference time (D-34). Closes the train/inference
    domain shift surfaced by ``scripts/verify_rms_domain_mismatch.py``.

    Contract:
      - Accepts a 1-D float32 ``np.ndarray`` OR ``torch.Tensor``. Returns the
        same type it received.
      - If ``current_rms < eps`` (silence or near-silence), returns the input
        unchanged so we never amplify a noise floor.
      - Otherwise multiplies the waveform by ``target / current_rms`` so the
        output RMS equals ``target``.
      - Does not clip. Downstream code handles saturation if any; the default
        ``target=0.1`` is well below 1.0 so clipping is unlikely.
      - Idempotent within float32 precision.
    """
    if isinstance(audio, torch.Tensor):
        current_rms = torch.sqrt(torch.mean(audio * audio))
        if float(current_rms.item()) < eps:
            return audio
        return audio * (target / current_rms)
    # numpy path
    current_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if current_rms < eps:
        return audio
    scaled = audio * (target / current_rms)
    return scaled.astype(audio.dtype)


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

    Also applies a fixed input gain so the audio level matches what the model
    was trained on. This is mic-calibration, NOT AGC: every chunk gets the same
    multiplier so intra-chunk dynamics and inter-chunk loudness differences are
    preserved. The default (500x) is calibrated for the UMA-16v2 — its ambient
    floor sits at ~1e-4 RMS while the EfficientAT training data sits around
    ~0.05 RMS, a ~500x gap. Configure via AcousticSettings.cnn_input_gain.

    Debug capture: set env var ACOUSTIC_DEBUG_DUMP_DIR=/some/path to dump every
    processed chunk to a WAV file in that directory. Each call writes a numbered
    WAV (sequence_NNNNNN_rms_X.XXXX.wav) at the model's target sample rate. Use
    this to listen to / re-classify exactly what the model is seeing, without
    racing the live pipeline.
    """

    def __init__(self, target_sr: int = 32000, input_gain: float = 1.0) -> None:
        self._target_sr = target_sr
        self._resampler: torchaudio.transforms.Resample | None = None
        self._cached_sr: int | None = None
        self._input_gain = float(input_gain)
        # Debug dump (opt-in via env var so zero cost when off)
        import os
        dump_dir = os.environ.get("ACOUSTIC_DEBUG_DUMP_DIR")
        self._dump_dir: str | None = dump_dir
        self._dump_seq: int = 0
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            import logging
            logging.getLogger(__name__).warning(
                "RawAudioPreprocessor debug dump ENABLED → %s (every CNN input written to disk)",
                dump_dir,
            )

    def set_input_gain(self, gain: float) -> None:
        """Hot-update the calibration gain. Lets the Pipeline-tab gain field
        change effect without recreating the preprocessor — so the debug-dump
        sequence counter (and any other live state) survives."""
        self._input_gain = float(gain)

    def process(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        waveform = torch.from_numpy(audio).float()
        if sr != self._target_sr:
            if self._cached_sr != sr:
                self._resampler = torchaudio.transforms.Resample(sr, self._target_sr)
                self._cached_sr = sr
            waveform = self._resampler(waveform)
        if self._input_gain != 1.0:
            waveform = waveform * self._input_gain

        if self._dump_dir is not None:
            self._dump(waveform)

        return waveform

    def _dump(self, waveform: torch.Tensor) -> None:
        """Write the post-gain post-resample waveform to a WAV for offline analysis."""
        try:
            import os
            import soundfile as sf
            arr = waveform.detach().cpu().numpy().astype(np.float32)
            rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0
            self._dump_seq += 1
            fname = f"cnn_input_{self._dump_seq:06d}_rms_{rms:.4f}.wav"
            path = os.path.join(self._dump_dir, fname)  # type: ignore[arg-type]
            sf.write(path, arr, self._target_sr, subtype="FLOAT")
        except Exception:  # pragma: no cover - debug only
            import logging
            logging.getLogger(__name__).exception("Failed to dump CNN input")


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
