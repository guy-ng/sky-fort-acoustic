"""SPA static file serving for the built React frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Project root -> web/dist (built React frontend)
_DIST_DIR = Path(__file__).parent.parent.parent.parent / "web" / "dist"


def mount_static(app: FastAPI) -> None:
    """Mount static files and SPA catch-all if the frontend is built.

    MUST be called AFTER all API and WebSocket routers are included,
    because the SPA catch-all route would shadow API routes otherwise.
    """
    if not _DIST_DIR.is_dir():
        logger.info("Frontend not built, static files not served (looked for %s)", _DIST_DIR)
        return

    assets_dir = _DIST_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static-assets")
        logger.info("Mounted static assets from %s", assets_dir)

    index_html = _DIST_DIR / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(request: Request, full_path: str) -> HTMLResponse | FileResponse:
        """Serve index.html for all non-API, non-WebSocket paths (SPA routing)."""
        # Serve specific files from dist if they exist (favicon, etc.)
        file_path = _DIST_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        # Otherwise serve index.html for SPA client-side routing
        return FileResponse(str(index_html))

    logger.info("SPA fallback route registered (serving %s)", index_html)
