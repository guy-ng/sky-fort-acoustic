"""Shared fixtures for integration tests."""

from __future__ import annotations

import asyncio
import os

import pytest

# Ensure simulated mode for tests (no hardware needed)
os.environ["ACOUSTIC_AUDIO_SOURCE"] = "simulated"

from acoustic.main import app, lifespan  # noqa: E402


@pytest.fixture
async def running_app():
    """Start the app lifespan (audio capture + pipeline) and yield the app."""
    async with lifespan(app):
        # Give pipeline a moment to start processing
        await asyncio.sleep(0.3)
        yield app
