"""RecordingSession: captures 16-channel audio, downmixes to mono, resamples to 16kHz WAV."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


class RecordingSession:
    """Streaming WAV writer that accepts 16-channel 48kHz chunks and outputs mono 16kHz.

    Usage:
        session = RecordingSession(output_path=Path("rec.wav"))
        session.start()
        session.write_chunk(chunk)  # (samples, 16) float32
        duration = session.stop()
    """

    def __init__(
        self,
        output_path: Path,
        source_sr: int = 48000,
        target_sr: int = 16000,
        gain_db: float = 20.0,
    ) -> None:
        self._path = output_path
        self._source_sr = source_sr
        self._target_sr = target_sr
        self._gain_linear: float = 10.0 ** (gain_db / 20.0)
        self._file: sf.SoundFile | None = None
        self._samples_written = 0
        self._running = False
        self._last_rms_db: float = -100.0

    def start(self) -> None:
        """Open the WAV file for streaming write."""
        self._file = sf.SoundFile(
            str(self._path),
            mode="w",
            samplerate=self._target_sr,
            channels=1,
            format="WAV",
            subtype="FLOAT",
        )
        self._running = True

    def write_chunk(self, chunk: np.ndarray) -> None:
        """Accept a (samples, 16) chunk, downmix to mono, resample, and write.

        No-op if session is not running.
        """
        if not self._running or self._file is None:
            return

        # Mono downmix: average all channels (D-12)
        mono = chunk.mean(axis=1)

        # Apply gain amplification
        mono = mono * self._gain_linear

        # Resample 48kHz -> 16kHz (ratio 1:3)
        resampled = resample_poly(mono, up=1, down=3).astype(np.float32)

        self._file.write(resampled)
        self._samples_written += len(resampled)

        # RMS for level meter
        rms = np.sqrt(np.mean(resampled**2))
        self._last_rms_db = 20.0 * np.log10(max(rms, 1e-10))

    @property
    def duration_s(self) -> float:
        """Current recording duration in seconds."""
        return self._samples_written / self._target_sr

    @property
    def rms_db(self) -> float:
        """RMS level of the last written chunk in dB."""
        return self._last_rms_db

    @property
    def running(self) -> bool:
        """Whether the session is actively recording."""
        return self._running

    @property
    def path(self) -> Path:
        """Path to the output WAV file."""
        return self._path

    def stop(self) -> float:
        """Close the WAV file and return the total duration in seconds."""
        self._running = False
        if self._file is not None:
            self._file.close()
            self._file = None
        return self.duration_s

    def to_parquet(self, label: int) -> Path:
        """Convert the completed WAV recording to a single-row Parquet file.

        Must be called after stop(). Reads the WAV file, packages audio bytes
        with label into DADS-compatible Parquet schema.

        Args:
            label: Integer class label (1=drone, 0=no-drone).

        Returns:
            Path to the created .parquet file.
        """
        wav_path = self._path
        wav_bytes = wav_path.read_bytes()

        table = pa.table({
            "audio": [{"bytes": wav_bytes, "path": wav_path.name}],
            "label": [label],
        })

        parquet_path = wav_path.with_suffix(".parquet")
        pq.write_table(table, str(parquet_path))
        return parquet_path
