"""Integration tests for PATCH /api/settings endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from acoustic.api.routes import router
from acoustic.config import AcousticSettings


@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with settings on app.state."""
    app = FastAPI()
    app.include_router(router)
    app.state.settings = AcousticSettings(audio_source="simulated")
    return app


@pytest.fixture
async def client(test_app):
    """Async HTTP client bound to the test app."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_patch_bf_nu_valid(client, test_app):
    """PATCH with valid bf_nu returns 200 and updated value."""
    resp = await client.patch("/api/settings", json={"bf_nu": 50.0})
    assert resp.status_code == 200
    assert resp.json() == {"updated": {"bf_nu": 50.0}}
    # Verify settings object was mutated
    assert test_app.state.settings.bf_nu == 50.0


@pytest.mark.asyncio
async def test_patch_bf_nu_too_low(client):
    """bf_nu < 1.0 returns 422 validation error."""
    resp = await client.patch("/api/settings", json={"bf_nu": 0.5})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_patch_bf_nu_too_high(client):
    """bf_nu > 1000.0 returns 422 validation error."""
    resp = await client.patch("/api/settings", json={"bf_nu": 1500})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_patch_empty_body(client):
    """Empty body returns 200 with empty updated dict."""
    resp = await client.patch("/api/settings", json={})
    assert resp.status_code == 200
    assert resp.json() == {"updated": {}}


@pytest.mark.asyncio
async def test_patch_updates_persists(client, test_app):
    """After PATCH, settings value is updated for subsequent reads."""
    assert test_app.state.settings.bf_nu == 100.0  # default
    await client.patch("/api/settings", json={"bf_nu": 25.0})
    assert test_app.state.settings.bf_nu == 25.0
