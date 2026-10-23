"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend import lifecycle as presence
from backend.api import convert, logs, models, presets, stream, system, training
from backend.assets import AssetStore, legacy_asset_directories
from backend.config import ROOT, Settings
from backend.config import settings as default_settings
from backend.frontend_build import MISSING_UI_PAGE
from backend.lifecycle import Lifecycle
from backend.logbook import logbook
from backend.model_manager import ModelManager
from backend.presets import PresetStore
from backend.security import cors_origins
from backend.training.jobs import TrainingJobs
from backend.virtual_mic import VirtualMic

logger = logging.getLogger(__name__)


def app_version() -> str:
    try:
        return version("openvoicechanger")
    except PackageNotFoundError:
        return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def create_app(settings: Settings | None = None, *, manager: ModelManager | None = None,
               on_ready: Callable[[], None] | None = None, lifecycle: Lifecycle | None = None) -> FastAPI:
    settings = settings or default_settings
    assets = manager.assets if manager is not None else AssetStore(settings.ASSETS_DIR,
                                                                    legacy_asset_directories(ROOT))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(assets.prepare)
        assets.download_missing_in_background()
        # Probe the device up front (imports CUDA) so the first request is fast.
        await asyncio.to_thread(lambda: app.state.model_manager.compute)
        if settings.ALLOW_ANY_ORIGIN:
            logger.warning("OVC_ALLOW_ANY_ORIGIN is on: any website may use this API. Trusted networks only.")
        if settings.HOST not in {"127.0.0.1", "localhost", "::1"}:
            logger.warning("Listening on %s: anyone who can reach it can upload and run models.", settings.HOST)
        logger.info("OpenVoiceChanger %s ready on http://%s:%d", app.version, settings.HOST, settings.PORT)
        if settings.RESTORE_LAST_VOICE:
            threading.Thread(target=app.state.model_manager.restore_last_voice, name="restore-voice",
                             daemon=True).start()
        if on_ready is not None:
            on_ready()
        watcher = asyncio.create_task(app.state.lifecycle.watch()) if app.state.lifecycle.enabled else None
        try:
            yield
        finally:
            if watcher is not None:
                watcher.cancel()
            # Closed by closing the page: a training run carries on and exits by itself.
            app.state.training.close(stop_workers=not app.state.lifecycle.closed_by_idle)
            app.state.virtual_mic.close()
            app.state.model_manager.close()

    app = FastAPI(title="OpenVoiceChanger", version=app_version(), lifespan=lifespan)
    app.state.settings = settings
    app.state.assets = assets
    app.state.model_manager = manager or ModelManager(settings, assets)
    app.state.preset_store = PresetStore(settings.PRESETS_PATH)
    app.state.training = TrainingJobs(settings)
    if logbook not in logging.getLogger().handlers:   # the Logs page works however the app is started
        logging.getLogger().addHandler(logbook)
    app.state.lifecycle = lifecycle or Lifecycle(enabled=False)
    app.state.virtual_mic = VirtualMic(settings.DATA_DIR / "downloads")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(settings.CORS_ORIGINS, settings.ALLOW_ANY_ORIGIN),
        allow_credentials=not settings.ALLOW_ANY_ORIGIN,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (system, models, presets, convert, stream, training, logs, presence):
        app.include_router(module.router)

    if (settings.FRONTEND_DIST / "index.html").is_file():
        app.mount("/", StaticFiles(directory=settings.FRONTEND_DIST, html=True), name="frontend")
    else:
        logger.warning("Web UI build not found at %s; serving setup instructions instead.",
                       settings.FRONTEND_DIST)

        @app.get("/", include_in_schema=False)
        async def missing_ui() -> HTMLResponse:
            return HTMLResponse(MISSING_UI_PAGE, status_code=503)
    return app
