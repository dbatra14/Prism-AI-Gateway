"""Application wiring and lifecycle.

Startup order matters and is worth reading once:

    database schema -> seed data -> price table -> providers -> dispatcher
                                                            -> embedder/cache

The embedding model is loaded at startup, not on the first cache lookup. Paying
that cost once during boot is much better than making one unlucky user's request
absorb a multi-second model load.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, chat
from app.cache.embedder import get_embedder
from app.cache.semantic_cache import SemanticCache
from app.config import get_gateway_config, get_settings
from app.db import dispose_engine, get_session_factory, init_db
from app.dispatch import Dispatcher
from app.errors import PrismError, prism_error_handler, unhandled_error_handler
from app.limits import prune_rate_limit_windows
from app.pricing import PriceTable
from app.providers.registry import ProviderRegistry
from app.seed import seed_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("prism")

CONSOLE_DIR = Path(__file__).resolve().parent.parent / "console"

# How often expired cache entries and stale rate-limit windows are swept.
HOUSEKEEPING_INTERVAL_SECONDS = 60


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = get_gateway_config()

    await init_db()

    session_factory = get_session_factory()
    async with session_factory() as session:
        await seed_all(session, settings)
        prices = await PriceTable.load(session)

    registry = ProviderRegistry.from_config(config, settings)

    logger.info("Loading embedding model (%s)...", settings.embedding_model)
    embedder = await asyncio.to_thread(
        get_embedder, settings.embedding_backend, settings.embedding_model
    )
    logger.info("Embedder ready: %s (dim=%d)", embedder.name, embedder.dimension)

    app.state.settings = settings
    app.state.gateway_config = config
    app.state.prices = prices
    app.state.registry = registry
    app.state.dispatcher = Dispatcher(registry, config, prices)
    app.state.cache = SemanticCache(
        embedder,
        ttl_seconds=settings.cache_ttl_seconds,
        max_candidates=settings.cache_max_candidates,
    )

    housekeeping = asyncio.create_task(_housekeeping(app))
    logger.info(
        "Prism ready — providers: %s | aliases: %s",
        ", ".join(registry.names()),
        ", ".join(config.aliases),
    )

    try:
        yield
    finally:
        housekeeping.cancel()
        await registry.aclose()
        await dispose_engine()


async def _housekeeping(app: FastAPI) -> None:
    """Sweep expired cache entries and old rate-limit windows.

    Deliberately a background task: doing this inline would make one unlucky
    request pay for everyone else's cleanup.
    """
    while True:
        try:
            await asyncio.sleep(HOUSEKEEPING_INTERVAL_SECONDS)
            async with get_session_factory()() as session:
                await prune_rate_limit_windows(session)
                await app.state.cache.purge_expired(session)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - housekeeping must never kill the app
            logger.exception("housekeeping pass failed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Prism",
        description="An LLM gateway with smart routing and a semantic cache.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_exception_handler(PrismError, prism_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.include_router(chat.router)
    app.include_router(admin.router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict:
        return {"status": "ok", "service": "prism"}

    if CONSOLE_DIR.exists():
        app.mount("/console", StaticFiles(directory=CONSOLE_DIR, html=True), name="console")

        @app.get("/", include_in_schema=False)
        async def root() -> FileResponse:
            return FileResponse(CONSOLE_DIR / "index.html")

    return app


app = create_app()
