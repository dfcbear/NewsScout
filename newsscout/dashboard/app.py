"""newsscout.dashboard.app
~~~~~~~~~~~~~~~~~~~~~~~~~~~
FastAPI web dashboard for NewsScout — mobile-first, Tailscale-accessible.

Provides searchable breakthrough archive, feedback stats, digest history,
and web-based feedback recording with a lightweight vanilla HTML frontend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from newsscout.config import Settings, get_settings
from newsscout.dashboard.queries import (
    add_interest_keyword,
    add_tracked_repo,
    get_breakthrough_detail,
    get_feedback_stats,
    get_recent_breakthroughs,
    get_topic_radar,
    get_watchlist,
    list_breakthroughs,
    list_digests,
    record_web_feedback,
    remove_interest_keyword,
    remove_tracked_repo,
    toggle_watchlist,
)
from newsscout.dashboard.security import (
    FixedWindowRateLimiter,
    rate_limit_dependency,
    verify_api_key,
)
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.preferences import PreferencesService

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).resolve().parent
_TEMPLATE_PATH = _DASHBOARD_DIR / "templates" / "index.html"


def _read_template() -> str:
    """Reads the dashboard HTML template, caching in memory."""
    if not hasattr(_read_template, "_cache"):
        _read_template._cache = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _read_template._cache


def create_app(settings: Settings | None = None) -> FastAPI:
    """Factory creating a configured FastAPI application instance.

    Args:
        settings: Optional Settings override. Defaults to global get_settings().
    """
    settings = settings or get_settings()
    db = Database(settings.db_path, timeout=5.0)
    preferences = PreferencesService(db, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await db.initialize()
        conn = db.get_sync_connection()
        try:
            apply_migrations(conn=conn)
        finally:
            conn.close()
        yield None
        await db.close()

    app = FastAPI(
        title="NewsScout Dashboard",
        description="Autonomous AI Breakthrough Scout — Web Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Serve static assets if directory exists
    static_dir = _DASHBOARD_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Store references for dependency injection
    app.state.db = db
    app.state.preferences = preferences
    app.state.settings = settings
    app.state.rate_limiter = FixedWindowRateLimiter(
        max_requests=settings.api_rate_limit_per_minute,
        window_seconds=60,
    )

    # Security dependencies applied to all state-changing endpoints
    _security_deps = [Depends(verify_api_key), Depends(rate_limit_dependency)]

    # ========================================================================
    # Page Routes
    # ========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page() -> str:
        """Serves the single-page dashboard HTML."""
        return _read_template()

    # ========================================================================
    # API Routes — Breakthroughs
    # ========================================================================

    @app.get("/api/breakthroughs")
    async def api_list_breakthroughs(
        category: str | None = Query(None),
        min_score: float | None = Query(None),
        search: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        results = await list_breakthroughs(
            db,
            category=category,
            min_score=min_score,
            search=search,
            limit=limit,
            offset=offset,
        )
        return results

    @app.get("/api/breakthroughs/{breakthrough_id}")
    async def api_get_breakthrough(breakthrough_id: int) -> dict[str, Any]:
        detail = await get_breakthrough_detail(db, breakthrough_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Breakthrough not found")
        return detail

    @app.get("/api/recent")
    async def api_recent(
        limit: int = Query(10, ge=1, le=50),
    ) -> list[dict[str, Any]]:
        return await get_recent_breakthroughs(db, limit=limit)

    @app.get("/api/watchlist")
    async def api_watchlist() -> list[dict[str, Any]]:
        return await get_watchlist(db)

    @app.post("/api/breakthroughs/{breakthrough_id}/watchlist", dependencies=_security_deps)
    async def api_toggle_watchlist(breakthrough_id: int) -> dict[str, Any]:
        result = await toggle_watchlist(db, breakthrough_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
        return result

    # ========================================================================
    # API Routes — Feedback
    # ========================================================================

    @app.get("/api/stats")
    async def api_stats() -> dict[str, Any]:
        return await get_feedback_stats(db)

    @app.post("/api/feedback", dependencies=_security_deps)
    async def api_record_feedback(request: Request) -> dict[str, Any]:
        body = await request.json()
        raw_id = body.get("breakthrough_id")
        rating = body.get("rating")
        notes = body.get("notes")

        if raw_id is None or rating is None:
            raise HTTPException(status_code=400, detail="breakthrough_id and rating are required")

        try:
            breakthrough_id = int(raw_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="breakthrough_id must be an integer")

        result = await record_web_feedback(
            db, preferences, breakthrough_id, str(rating), notes
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed"))
        return result

    # ========================================================================
    # API Routes — Digests
    # ========================================================================

    @app.get("/api/digests")
    async def api_list_digests(
        limit: int = Query(20, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return await list_digests(db, limit=limit)

    # ========================================================================
    # API Routes — Topic Radar & Dynamic Sources
    # ========================================================================

    @app.get("/api/topics")
    async def api_get_topics() -> dict[str, Any]:
        """Returns active sources, tracked GitHub repos, and custom keywords."""
        return await get_topic_radar(db)

    @app.post("/api/topics/repos", dependencies=_security_deps)
    async def api_add_tracked_repo(request: Request) -> dict[str, Any]:
        """Adds a GitHub repository to the tracked list."""
        body = await request.json()
        repo = body.get("repo", "")
        if not repo:
            raise HTTPException(status_code=400, detail="repo field is required")
        result = await add_tracked_repo(db, repo)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add repository"))
        return result

    @app.delete("/api/topics/repos/{owner}/{repo}", dependencies=_security_deps)
    async def api_remove_tracked_repo(owner: str, repo: str) -> dict[str, Any]:
        """Removes a GitHub repository from the tracked list."""
        repo_slug = f"{owner}/{repo}"
        result = await remove_tracked_repo(db, repo_slug)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to remove repository"))
        return result

    @app.post("/api/topics/keywords", dependencies=_security_deps)
    async def api_add_interest_keyword(request: Request) -> dict[str, Any]:
        """Adds an interest keyword to the radar."""
        body = await request.json()
        keyword = body.get("keyword", "")
        if not keyword:
            raise HTTPException(status_code=400, detail="keyword field is required")
        result = await add_interest_keyword(db, keyword)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add keyword"))
        return result

    @app.delete("/api/topics/keywords/{keyword}", dependencies=_security_deps)
    async def api_remove_interest_keyword(keyword: str) -> dict[str, Any]:
        """Removes an interest keyword from the radar."""
        result = await remove_interest_keyword(db, keyword)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to remove keyword"))
        return result

    # ========================================================================
    # API Routes — System Status & Gateway Pairing (R1, R2, R3)
    # ========================================================================

    @app.get("/api/system/status")
    async def api_system_status() -> dict[str, Any]:
        """Returns operational status of LLM provider, search engines, and messenger gateways."""
        llm_status = {
            "provider": settings.llm_provider,
            "configured": settings.has_llm_credentials,
            "model": settings.gemini_model if settings.llm_provider == "gemini" else settings.llm_model,
            "base_url": settings.llm_base_url if settings.llm_provider != "gemini" else "https://generativelanguage.googleapis.com",
            "fallback_provider": settings.llm_fallback_provider,
            "fallback_model": settings.llm_fallback_model,
        }

        search_status = {
            "searxng": {
                "enabled": settings.searxng_enabled,
                "base_url": settings.searxng_base_url,
                "zero_key": True,
            },
            "duckduckgo": {
                "enabled": settings.duckduckgo_enabled,
                "zero_key": True,
            },
            "tavily": {
                "enabled": settings.has_tavily_credentials,
                "zero_key": False,
            },
            "exa": {
                "enabled": settings.has_exa_credentials,
                "zero_key": False,
            },
        }

        gateways_status = {
            "telegram": {
                "enabled": settings.has_telegram_credentials,
                "chat_id": bool(settings.telegram_chat_id),
            },
            "signal": {
                "enabled": settings.has_signal_credentials,
                "bridge_url": settings.signal_bridge_url,
                "sender_number": bool(settings.signal_sender_number),
                "recipients_count": len(settings.effective_signal_recipients),
            },
        }

        return {
            "app_name": settings.app_name,
            "environment": settings.environment,
            "llm": llm_status,
            "search": search_status,
            "gateways": gateways_status,
        }

    @app.get("/api/gateways/pairing/{channel}")
    async def api_gateway_pairing(channel: str) -> dict[str, Any]:
        """Returns pairing and QR code status for Signal or Telegram gateways."""
        channel_lower = channel.lower().strip()
        if channel_lower not in ("signal", "telegram"):
            raise HTTPException(status_code=400, detail=f"Unsupported channel: {channel}")

        from newsscout.delivery.dispatcher import create_default_dispatcher
        dispatcher = create_default_dispatcher(settings=settings, db=db, preferences_service=preferences)
        gw = dispatcher.get_gateway(channel_lower)
        if gw is None:
            raise HTTPException(status_code=404, detail=f"Gateway for '{channel}' not registered.")

        try:
            status = await gw.get_pairing_status()
            return {
                "channel": status.channel,
                "status": status.status,
                "qr_data": status.qr_data,
                "qr_image_url": status.qr_image_url,
                "error": status.error,
            }
        except Exception:
            logger.exception("Error fetching pairing status for channel '%s'", channel_lower)
            return {
                "channel": channel_lower,
                "status": "error",
                "error": "Internal server error while fetching pairing status.",
            }

    return app


# Module-level app instance for uvicorn
app = create_app()
