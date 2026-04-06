"""Data augmentation for drone audio training: waveform-level and spectrogram-level."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from scipy.signal import fftconvolve
try:
    from audiomentations import Compose, Gain, PitchShift, TimeStretch
except ImportError:
    Compose = Gain = PitchShift = TimeStretch = None  # type: ignore[assignment,misc]


class WaveformAugmentation:
    """Waveform-level augmentation: Gaussian noise injection + random gain.

    Applied to raw audio segments before mel-spectrogram conversion (D-10).
    """

    def __init__(
        self,
        snr_range: tuple[float, float] = (10.0, 40.0),
        gain_db: float = 6.0,
    ) -> None:
        self._snr_range = snr_range
        self._gain_db = gain_db

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise injection and gain scaling to a 1-D float32 audio segment.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Augmented audio as float32 array of same length.
        """
        rng = np.random.default_rng()
        out = audio.copy()

        # Gaussian noise at random SNR
        signal_power = float(np.mean(out**2))
        if signal_power > 1e-10:
            snr_db = rng.uniform(self._snr_range[0], self._snr_range[1])
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0, np.sqrt(noise_power), size=out.shape).astype(
                np.float32
            )
            out = out + noise

        # Random gain in +/- gain_db
        gain_db = rng.uniform(-self._gain_db, self._gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        out = out * gain_linear

        return out.astype(np.float32)


class SpecAugment:
    """Spectrogram-level augmentation: time and frequency masking (D-09).

    Uses torchaudio TimeMasking and FrequencyMasking transforms.
    """

    def __init__(
        self,
        time_mask_param: int = 20,
        freq_mask_param: int = 8,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        self._time_mask_param = time_mask_param
        self._freq_mask_param = freq_mask_param
        self._num_time_masks = num_time_masks
        self._num_freq_masks = num_freq_masks

        # Build mask transforms (only if params > 0)
        self._time_masks: list[T.TimeMasking] = []
        self._freq_masks: list[T.FrequencyMasking] = []
        if time_mask_param > 0:
            self._time_masks = [
                T.TimeMasking(time_mask_param=time_mask_param)
                for _ in range(num_time_masks)
            ]
        if freq_mask_param > 0:
            self._freq_masks = [
                T.FrequencyMasking(freq_mask_param=freq_mask_param)
                for _ in range(num_freq_masks)
            ]

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masking to a spectrogram.

        Args:
            spectrogram: Tensor of shape (1, time, freq) = (1, 128, 64).

        Returns:
            Masked spectrogram of same shape (1, 128, 64).
        """
        if not self._time_masks and not self._freq_masks:
            return spectrogram

        # torchaudio masks expect (..., freq, time) layout
        # Input is (1, time=128, freq=64) -> transpose to (1, freq=64, time=128)
        spec = spectrogram.transpose(-2, -1)

        for mask in self._freq_masks:
            spec = mask(spec)
        for mask in self._time_masks:
            spec = mask(spec)

        # Transpose back to (1, time=128, freq=64)
        return spec.transpose(-2, -1)


class BackgroundNoiseMixer:
    """Mix background noise from ESC-50/UrbanSound8K at random SNR (TRN-11).

    Scans noise directories for WAV files on construction but defers loading
    until ``warm_cache()`` is called.  During training, ``__call__`` mixes a
    random noise clip with the input audio at a uniformly-sampled SNR.
    """

    def __init__(
        self,
        noise_dirs: list[Path],
        snr_range: tuple[float, float] = (-10.0, 20.0),
        sample_rate: int = 16000,
        p: float = 0.5,
    ) -> None:
        self._snr_range = snr_range
        self._sample_rate = sample_rate
        self._p = p
        self._rng = np.random.default_rng()

        # Discover WAV files (lazy -- no audio loaded yet)
        self._noise_files: list[Path] = []
        for d in noise_dirs:
            d = Path(d)
            if d.is_dir():
                self._noise_files.extend(sorted(d.rglob("*.wav")))

        self._noise_cache: list[np.ndarray] = []

    def warm_cache(self) -> None:
        """Load all noise WAV files into memory, resampling to target SR."""
        self._noise_cache = []
        for fpath in self._noise_files:
            audio, sr = sf.read(str(fpath), dtype="float32")
            # Convert stereo to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample if needed
            if sr != self._sample_rate:
                audio_t = torch.from_numpy(audio).unsqueeze(0)
                audio_t = torchaudio.functional.resample(audio_t, sr, self._sample_rate)
                audio = audio_t.squeeze(0).numpy()
            self._noise_cache.append(audio.astype(np.float32))

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Mix noise at random SNR with probability *p*.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Audio with (possibly) added noise, float32, same length.
        """
        # Skip if no noise available or random skip
        if not self._noise_cache or self._rng.random() >= self._p:
            return audio

        # Pick random noise clip
        noise = self._noise_cache[self._rng.integers(len(self._noise_cache))]

        n = len(audio)
        if len(noise) > n:
            start = self._rng.integers(len(noise) - n + 1)
            noise_seg = noise[start : start + n]
        elif len(noise) < n:
            from acoustic.classification.preprocessing import pad_or_loop
            noise_seg = pad_or_loop(noise, n)
        else:
            noise_seg = noise

        # Compute scale factor for desired SNR
        snr_db = self._rng.uniform(self._snr_range[0], self._snr_range[1])
        signal_power = float(np.mean(audio**2))
        noise_power = float(np.mean(noise_seg**2))

        if noise_power > 1e-10 and signal_power > 1e-10:
            scale = np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0)))
            mixed = audio + scale * noise_seg
        else:
            mixed = audio

        return np.clip(mixed, -1.0, 1.0).astype(np.float32)


class AudiomentationsAugmentation:
    """Waveform augmentation using audiomentations library (TRN-12).

    Replaces WaveformAugmentation with PitchShift + TimeStretch + Gain.
    """

    def __init__(
        self,
        pitch_semitones: float = 3.0,
        time_stretch_range: tuple[float, float] = (0.85, 1.15),
        gain_db: float = 6.0,
        p: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self._sample_rate = sample_rate
        self._augment = Compose(
            [
                PitchShift(
                    min_semitones=-pitch_semitones,
                    max_semitones=pitch_semitones,
                    p=p,
                ),
                TimeStretch(
                    min_rate=time_stretch_range[0],
                    max_rate=time_stretch_range[1],
                    p=p,
                ),
                Gain(
                    min_gain_db=-gain_db,
                    max_gain_db=gain_db,
                    p=p,
                ),
            ]
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline.

        Args:
            audio: 1-D float32 mono audio array.

        Returns:
            Augmented 1-D float32 array of the same length.
        """
        return self._augment(samples=audio, sample_rate=self._sample_rate)


class WideGainAugmentation:
    """Wide ±wide_gain_db uniform gain (Phase 20 D-01..D-04).

    Replaces the WaveformAugmentation small-gain stage. Runs as a separate
    pre-stage in ComposedAugmentation. Clips to [-1, 1] before returning so
    downstream RIR convolution sees a bounded signal (Pitfall 2 in Phase 20
    research).
    """

    def __init__(self, wide_gain_db: float = 40.0, p: float = 1.0) -> None:
        self._wide_gain_db = float(wide_gain_db)
        self._p = float(p)
        self._rng = np.random.default_rng()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self._p:
            return audio
        gain_db = self._rng.uniform(-self._wide_gain_db, self._wide_gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        out = (audio * gain_linear).astype(np.float32)
        return np.clip(out, -1.0, 1.0)

    def __getstate__(self):
        # Pickle-safe: exclude live RNG (rebuilt on unpickle by worker)
        return {"wide_gain_db": self._wide_gain_db, "p": self._p}

    def __setstate__(self, state):
        self._wide_gain_db = state["wide_gain_db"]
        self._p = state["p"]
        self._rng = np.random.default_rng()


class RoomIRAugmentation:
    """Procedural ShoeBox RIR convolution (Phase 20 D-05..D-08).

    Pre-generates a pool of pool_size RIRs at construction (via
    pyroomacoustics.ShoeBox image source method). Each __call__ samples one
    RIR from the pool and convolves with the input via
    scipy.signal.fftconvolve. Faster than per-call generation (~5-15 ms per
    ShoeBox simulation -- see Pitfall 3 in 20-RESEARCH.md) and removes
    pyroomacoustics from the per-batch hot path.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pool_size: int = 500,
        room_dim_min: tuple[float, float, float] = (3.0, 3.0, 2.5),
        room_dim_max: tuple[float, float, float] = (12.0, 12.0, 4.0),
        absorption_range: tuple[float, float] = (0.2, 0.7),
        source_distance_range: tuple[float, float] = (1.0, 8.0),
        max_order: int = 10,
        p: float = 0.7,
        seed: int = 42,
    ) -> None:
        self._sr = int(sample_rate)
        self._pool_size = int(pool_size)
        self._room_dim_min = tuple(room_dim_min)
        self._room_dim_max = tuple(room_dim_max)
        self._absorption_range = tuple(absorption_range)
        self._source_distance_range = tuple(source_distance_range)
        self._max_order = int(max_order)
        self._p = float(p)
        self._seed = int(seed)
        init_rng = np.random.default_rng(self._seed)
        self._pool: list[np.ndarray] = [
            self._generate_one(init_rng) for _ in range(self._pool_size)
        ]
        self._call_rng = np.random.default_rng()

    def _generate_one(self, rng: np.random.Generator) -> np.ndarray:
        room_dim = rng.uniform(self._room_dim_min, self._room_dim_max)  # shape (3,)
        absorption = float(rng.uniform(*self._absorption_range))
        room = pra.ShoeBox(
            room_dim.tolist(),
            fs=self._sr,
            materials=pra.Material(absorption),
            max_order=self._max_order,
        )
        mic_pos = room_dim / 2.0
        src_pos = mic_pos + np.array([1.0, 0.0, 0.0])  # default fallback
        for _ in range(8):
            dist = float(rng.uniform(*self._source_distance_range))
            theta = float(rng.uniform(0, 2 * np.pi))
            phi = float(rng.uniform(np.pi / 4, 3 * np.pi / 4))
            offset = np.array([
                dist * np.sin(phi) * np.cos(theta),
                dist * np.sin(phi) * np.sin(theta),
                dist * np.cos(phi),
            ])
            candidate = mic_pos + offset
            margin = 0.3
            if np.all(candidate > margin) and np.all(candidate < room_dim - margin):
                src_pos = candidate
                break
        room.add_source(src_pos.tolist())
        room.add_microphone(mic_pos.tolist())
        room.compute_rir()
        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        max_len = self._sr  # 1 second cap (Pitfall 3)
        if len(rir) > max_len:
            rir = rir[:max_len]
        return rir

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if not self._pool or self._call_rng.random() >= self._p:
            return audio
        rir = self._pool[self._call_rng.integers(len(self._pool))]
        out = fftconvolve(audio, rir, mode="full")[: len(audio)]
        peak_in = float(np.abs(audio).max())
        peak_out = float(np.abs(out).max())
        if peak_out > 1e-8 and peak_in > 0:
            out = out * (peak_in / peak_out)
        return out.astype(np.float32)

    def __getstate__(self):
        # Pool is reproducible from seed -- exclude RNG and pool, rebuild on unpickle.
        return {
            "sample_rate": self._sr,
            "pool_size": self._pool_size,
            "room_dim_min": self._room_dim_min,
            "room_dim_max": self._room_dim_max,
            "absorption_range": self._absorption_range,
            "source_distance_range": self._source_distance_range,
            "max_order": self._max_order,
            "p": self._p,
            "seed": self._seed,
        }

    def __setstate__(self, state):
        self.__init__(**state)


class ComposedAugmentation:
    """Chain multiple waveform augmentations sequentially.

    Unlike a closure/lambda, this class is picklable so it works with
    DataLoader num_workers > 0.
    """

    def __init__(self, augmentations: list[Callable[[np.ndarray], np.ndarray]]) -> None:
        self._augmentations = augmentations

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for aug in self._augmentations:
            audio = aug(audio)
        return audio
