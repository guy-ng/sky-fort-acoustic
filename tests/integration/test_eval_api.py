"""Integration tests for evaluation and model listing API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from httpx import ASGITransport, AsyncClient

from acoustic.classification.research_cnn import ResearchCNN


@pytest.mark.asyncio
class TestEvalRun:
    """Tests for POST /api/eval/run."""

    async def test_eval_missing_model_returns_404(self, running_app):
        """POST /api/eval/run with nonexistent model returns 404."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/eval/run",
                json={"model_path": "/nonexistent/model.pt"},
            )
            assert response.status_code == 404
            data = response.json()
            assert "Model file not found" in data["message"]

    async def test_eval_missing_datadir_returns_404(self, running_app):
        """POST /api/eval/run with nonexistent data_dir returns 404."""
        # Create a real model file first so model_path check passes
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ResearchCNN().state_dict(), f.name)
            model_path = f.name

        try:
            async with AsyncClient(
                transport=ASGITransport(app=running_app, raise_app_exceptions=False),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/eval/run",
                    json={"model_path": model_path, "data_dir": "/nonexistent/dir"},
                )
                assert response.status_code == 404
                data = response.json()
                assert "Directory not found" in data["message"]
        finally:
            Path(model_path).unlink(missing_ok=True)

    async def test_eval_with_valid_data_returns_200(self, running_app):
        """POST /api/eval/run with valid model and synthetic data returns 200."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create label subdirectories with synthetic WAVs
            drone_dir = Path(tmpdir) / "drone"
            drone_dir.mkdir()
            bg_dir = Path(tmpdir) / "no drone"
            bg_dir.mkdir()

            sr = 16000
            duration = 1.0
            samples = int(sr * duration)

            # Write synthetic WAV files
            for i in range(2):
                audio = np.random.randn(samples).astype(np.float32) * 0.1
                sf.write(str(drone_dir / f"drone_{i}.wav"), audio, sr)
                sf.write(str(bg_dir / f"bg_{i}.wav"), audio, sr)

            # Save untrained model
            model_path = Path(tmpdir) / "model.pt"
            torch.save(ResearchCNN().state_dict(), str(model_path))

            async with AsyncClient(
                transport=ASGITransport(app=running_app, raise_app_exceptions=False),
                base_url="http://test",
                timeout=30.0,
            ) as client:
                response = await client.post(
                    "/api/eval/run",
                    json={"model_path": str(model_path), "data_dir": tmpdir},
                )
                assert response.status_code == 200
                data = response.json()
                assert "summary" in data
                assert "accuracy" in data["summary"]
                assert "distribution" in data
                assert "drone" in data["distribution"]
                assert "per_file" in data
                assert len(data["per_file"]) == 4


@pytest.mark.asyncio
class TestModelList:
    """Tests for GET /api/models."""

    async def test_models_returns_200(self, running_app):
        """GET /api/models returns 200 with models list."""
        async with AsyncClient(
            transport=ASGITransport(app=running_app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/models")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert isinstance(data["models"], list)
