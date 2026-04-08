"""Pure-numpy port of EfficientAT AugmentMelSTFT (inference mode only).

Parity-locked against src/acoustic/classification/efficientat/preprocess.py via
apps/rpi-edge/tests/test_preprocess_parity.py (atol=1e-5). Do NOT edit without
updating the parity test expectations.

Intentionally no torch / torchaudio imports -- D-28 forbids PyTorch on the Pi.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MEL_BANKS_PATH = Path(__file__).parent / "mel_banks_128_1024_32k.npy"

SR = 32000
N_MELS = 128
WIN_LENGTH = 800
HOP_SIZE = 320
N_FFT = 1024


def _hann_window_symmetric(n: int) -> np.ndarray:
    """Matches torch.hann_window(n, periodic=False): 0.5 - 0.5*cos(2*pi*k/(n-1))."""
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))).astype(np.float32)


def _padded_window(win_length: int, n_fft: int) -> np.ndarray:
    """torch.stft pads win_length to n_fft by centering the window."""
    assert win_length <= n_fft
    w = _hann_window_symmetric(win_length)
    pad_left = (n_fft - win_length) // 2
    pad_right = n_fft - win_length - pad_left
    return np.concatenate(
        [
            np.zeros(pad_left, dtype=np.float32),
            w,
            np.zeros(pad_right, dtype=np.float32),
        ]
    )


class NumpyMelSTFT:
    """Pure-numpy equivalent of AugmentMelSTFT in eval mode.

    Parameters match the training-side defaults used in EfficientAT:
      sr=32000, n_mels=128, win_length=800, hopsize=320, n_fft=1024.
    """

    def __init__(self) -> None:
        self.window = _padded_window(WIN_LENGTH, N_FFT)
        mel_basis = np.load(_MEL_BANKS_PATH)
        assert mel_basis.shape == (N_MELS, N_FFT // 2 + 1), (
            f"mel_basis shape {mel_basis.shape} != ({N_MELS}, {N_FFT // 2 + 1})"
        )
        self.mel_basis = mel_basis.astype(np.float32)

    def _preemphasis(self, x: np.ndarray) -> np.ndarray:
        # torch conv1d kernel [-0.97, 1], no padding
        #   out[t] = -0.97 * x[t] + 1.0 * x[t+1]
        # Output length = len(x) - 1.
        return (-0.97 * x[:-1] + x[1:]).astype(np.float32)

    def _stft(self, x: np.ndarray) -> np.ndarray:
        # torch.stft(center=True) reflect-pads n_fft // 2 on both sides.
        pad = N_FFT // 2
        xp = np.pad(x, (pad, pad), mode="reflect")
        num_frames = 1 + (len(xp) - N_FFT) // HOP_SIZE
        # Build sliding-window frame matrix (T, N_FFT).
        # Use stride tricks for efficiency on the Pi.
        stride = xp.strides[0]
        frames = np.lib.stride_tricks.as_strided(
            xp,
            shape=(num_frames, N_FFT),
            strides=(stride * HOP_SIZE, stride),
            writeable=False,
        ).astype(np.float32)
        # Apply the (zero-padded, centered) window of length N_FFT.
        frames = frames * self.window  # broadcast over frames
        spec = np.fft.rfft(frames, n=N_FFT, axis=-1)  # (T, F)
        power = (spec.real.astype(np.float32) ** 2 + spec.imag.astype(np.float32) ** 2)
        return power.T.astype(np.float32)  # (F, T)

    def forward(self, wave: np.ndarray) -> np.ndarray:
        """Convert a 1-D float32 waveform at 32 kHz to a (128, T) float32 mel spectrogram.

        Matches AugmentMelSTFT(freqm=0, timem=0).eval() semantics: preemphasis +
        windowed STFT (center=True, reflect pad) + power magnitude + precomputed
        mel basis + log(. + 1e-5) + (x + 4.5) / 5.0 normalization.
        """
        if wave.ndim != 1:
            raise ValueError(f"expected 1-D wave, got shape {wave.shape}")
        wave = wave.astype(np.float32)
        x = self._preemphasis(wave)
        power = self._stft(x)  # (F, T)
        mel = self.mel_basis @ power  # (128, T)
        mel = np.log(mel + 1e-5)
        mel = (mel + 4.5) / 5.0
        return mel.astype(np.float32)
