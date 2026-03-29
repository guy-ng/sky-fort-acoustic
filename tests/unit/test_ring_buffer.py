"""Tests for AudioRingBuffer."""

from __future__ import annotations

import numpy as np

from acoustic.audio.capture import AudioRingBuffer


class TestAudioRingBuffer:
    """Ring buffer FIFO, overflow, and availability tests."""

    def _make_buffer(self, num_chunks: int = 5) -> AudioRingBuffer:
        return AudioRingBuffer(num_chunks=num_chunks, chunk_samples=7200, num_channels=16)

    def test_write_read_single_chunk(self):
        buf = self._make_buffer()
        chunk = np.random.randn(7200, 16).astype(np.float32)
        assert buf.write(chunk) is True
        result = buf.read()
        assert result is not None
        np.testing.assert_array_equal(result, chunk)

    def test_fifo_order(self):
        buf = self._make_buffer()
        chunks = [np.full((7200, 16), i, dtype=np.float32) for i in range(3)]
        for c in chunks:
            buf.write(c)
        for i in range(3):
            result = buf.read()
            assert result is not None
            np.testing.assert_array_equal(result, chunks[i])

    def test_read_empty_returns_none(self):
        buf = self._make_buffer()
        assert buf.read() is None

    def test_overflow_detection(self):
        # Buffer of size 5 can hold 4 chunks (one slot reserved for full/empty disambiguation)
        buf = self._make_buffer(num_chunks=5)
        chunk = np.zeros((7200, 16), dtype=np.float32)
        # Fill to capacity
        written = 0
        for _ in range(10):
            if buf.write(chunk):
                written += 1
            else:
                break
        # Next write should fail
        assert buf.write(chunk) is False
        assert buf.overflow_count >= 1

    def test_available_property(self):
        buf = self._make_buffer(num_chunks=10)
        chunk = np.zeros((7200, 16), dtype=np.float32)
        for _ in range(3):
            buf.write(chunk)
        assert buf.available == 3
        buf.read()
        assert buf.available == 2
