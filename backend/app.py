"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import convert, models, presets, stream, system, training
from backend.assets import AssetStore, legacy_asset_directories
from backend.config import ROOT, Settings
from backend.config import settings as default_settings
from backend.model_manager import ModelManager
from backend.presets import PresetStore
from backend.security import cors_origins
from backend.training.jobs import TrainingJobs

logger = logging.getLogger(__name__)


def app_version() -> str:
    try:
        return version("openvoicechanger")
    except PackageNotFoundError:
        return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def create_app(settings: Settings | None = None, *, manager: ModelManager | None = None,
               on_ready: Callable[[], None] | None = None) -> FastAPI:
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
        if on_ready is not None:
            on_ready()
        try:
            yield
        finally:
            app.state.training.close()
            app.state.model_manager.close()

    app = FastAPI(title="OpenVoiceChanger", version=app_version(), lifespan=lifespan)
    app.state.settings = settings
    app.state.assets = assets
    app.state.model_manager = manager or ModelManager(settings, assets)
    app.state.preset_store = PresetStore(settings.PRESETS_PATH)
    app.state.training = TrainingJobs(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(settings.CORS_ORIGINS, settings.ALLOW_ANY_ORIGIN),
        allow_credentials=not settings.ALLOW_ANY_ORIGIN,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (system, models, presets, convert, stream, training):
        app.include_router(module.router)

    if settings.FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=settings.FRONTEND_DIST, html=True), name="frontend")
    else:
        logger.warning("Frontend build not found at %s; run the start script or `npm run build`.",
                       settings.FRONTEND_DIST)
    return app
