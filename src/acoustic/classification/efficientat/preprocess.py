"""AugmentMelSTFT vendored from EfficientAT models/preprocess.py.

Source: https://github.com/fschmid56/EfficientAT
License: Apache-2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class AugmentMelSTFT(nn.Module):
    """Mel spectrogram with preemphasis and (log_mel + 4.5) / 5.0 normalization.

    Matches EfficientAT's preprocessing exactly for pretrained weight compatibility.
    Default params: 32kHz, 128 mels, win=800, hop=320, n_fft=1024.
    """

    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        freqm: int = 48,
        timem: int = 192,
        fmin: float = 0.0,
        fmax: float | None = None,
        fmin_aug_range: int = 10,
        fmax_aug_range: int = 2000,
    ) -> None:
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.hopsize = hopsize
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False), persistent=False
        )
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False
        )
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert waveform (batch, samples) to normalized mel spectrogram."""
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x ** 2).sum(dim=-1)  # power mag

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels, self.n_fft, self.sr,
            fmin, fmax, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.0  # fast normalization
        return melspec
