"""tests.test_topics
~~~~~~~~~~~~~~~~~~
Unit and integration tests for dynamic topic & source management (TASK-07):
- Topic radar queries (tracked repos, custom keywords)
- Ingestion source dynamic pattern compilation
- REST API topic endpoints
- Telegram bot slash commands (/track, /interest, /radar, /help)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.dashboard.queries import (
    add_interest_keyword,
    add_tracked_repo,
    get_topic_radar,
    remove_interest_keyword,
    remove_tracked_repo,
)
from newsscout.delivery.telegram_bot import TelegramBot, mask_telegram_url
from newsscout.ingestion.hackernews import compile_topic_pattern, HackerNewsIngestionSource
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations


@pytest.fixture
def topic_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "test_topics.db"
    conn = Database(db_file)
    with conn.get_sync_connection() as c:
        apply_migrations(conn=c)
    return conn


@pytest.mark.asyncio
async def test_get_topic_radar_defaults(topic_db: Database):
    """Initial seeded sources load correct default repos and empty custom keywords."""
    radar = await get_topic_radar(topic_db)
    assert "sources" in radar
    assert len(radar["sources"]) >= 3
    assert "vllm-project/vllm" in radar["tracked_repos"]
    assert "ggerganov/llama.cpp" in radar["tracked_repos"]
    assert isinstance(radar["custom_keywords"], list)


@pytest.mark.asyncio
async def test_add_and_remove_tracked_repo(topic_db: Database):
    """Adds a new repo, verifies presence, and removes it cleanly."""
    res = await add_tracked_repo(topic_db, "facebookresearch/segment-anything-2")
    assert res["success"] is True
    assert "facebookresearch/segment-anything-2" in res["repos"]

    # Verify via radar query
    radar = await get_topic_radar(topic_db)
    assert "facebookresearch/segment-anything-2" in radar["tracked_repos"]

    # Test deduplication
    res2 = await add_tracked_repo(topic_db, "facebookresearch/segment-anything-2")
    assert res2["success"] is True
    assert res2["repos"].count("facebookresearch/segment-anything-2") == 1

    # Remove repo
    rem_res = await remove_tracked_repo(topic_db, "facebookresearch/segment-anything-2")
    assert rem_res["success"] is True
    assert "facebookresearch/segment-anything-2" not in rem_res["repos"]


@pytest.mark.asyncio
async def test_add_tracked_repo_validation(topic_db: Database):
    """Rejects malformed repository slugs."""
    res = await add_tracked_repo(topic_db, "invalid-slug-without-slash")
    assert res["success"] is False
    assert "Invalid repo slug" in res["error"]


@pytest.mark.asyncio
async def test_add_and_remove_interest_keyword(topic_db: Database):
    """Adds a custom keyword, verifies radar, and removes it."""
    res = await add_interest_keyword(topic_db, "neuromorphic")
    assert res["success"] is True
    assert "neuromorphic" in res["keywords"]

    radar = await get_topic_radar(topic_db)
    assert "neuromorphic" in radar["custom_keywords"]

    # Deduplication
    res2 = await add_interest_keyword(topic_db, "NEUROMORPHIC")
    assert res2["success"] is True
    assert res2["keywords"].count("neuromorphic") == 1

    # Remove
    rem_res = await remove_interest_keyword(topic_db, "neuromorphic")
    assert rem_res["success"] is True
    assert "neuromorphic" not in rem_res["keywords"]


def test_compile_topic_pattern_dynamic():
    """Custom keywords are compiled into the regex pattern."""
    pattern = compile_topic_pattern(["liquid neural networks", "bio-ml"])
    assert pattern.search("New breakthrough in liquid neural networks announced") is not None
    assert pattern.search("Advances in bio-ml on edge devices") is not None
    # Standard keyword still works
    assert pattern.search("vllm v0.6 release") is not None
    # Irrelevant text does not match
    assert pattern.search("Recipe for chocolate cake") is None


def test_api_topic_routes(tmp_path: Path):
    """FastAPI endpoints for topics return expected data and handle CRUD."""
    db_file = tmp_path / "api_test.db"
    settings = Settings(db_path=db_file)
    settings.ensure_directories()
    
    app = create_app(settings)
    with TestClient(app) as client:
        # 1. GET /api/topics
        res = client.get("/api/topics")
        assert res.status_code == 200
        data = res.json()
        assert "tracked_repos" in data
        assert "vllm-project/vllm" in data["tracked_repos"]

        # 2. POST /api/topics/repos
        res_post = client.post("/api/topics/repos", json={"repo": "state-spaces/mamba"})
        assert res_post.status_code == 200
        assert "state-spaces/mamba" in res_post.json()["repos"]

        # 3. DELETE /api/topics/repos/{owner}/{repo}
        res_del = client.delete("/api/topics/repos/state-spaces/mamba")
        assert res_del.status_code == 200
        assert "state-spaces/mamba" not in res_del.json()["repos"]

        # 4. POST /api/topics/keywords
        res_kw = client.post("/api/topics/keywords", json={"keyword": "spiking-nn"})
        assert res_kw.status_code == 200
        assert "spiking-nn" in res_kw.json()["keywords"]

        # 5. DELETE /api/topics/keywords/{keyword}
        res_kw_del = client.delete("/api/topics/keywords/spiking-nn")
        assert res_kw_del.status_code == 200
        assert "spiking-nn" not in res_kw_del.json()["keywords"]


@pytest.mark.asyncio
async def test_telegram_slash_commands(topic_db: Database, test_settings: Settings):
    """TelegramBot processes /track, /interest, /radar, /help slash commands."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    req = httpx.Request("POST", "https://api.telegram.org/bot123/sendMessage")
    mock_client.post = AsyncMock(return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 999}}, request=req))

    bot = TelegramBot(test_settings, client=mock_client, db=topic_db)
    bot._external_client = True

    # 1. /help
    res_help = await bot.handle_message_text("/help", chat_id="123")
    assert res_help["action"] == "help"

    # 2. /track
    res_track = await bot.handle_message_text("/track openai/whisper", chat_id="123")
    assert res_track["action"] == "track"
    assert res_track["repo"] == "openai/whisper"

    radar = await get_topic_radar(topic_db)
    assert "openai/whisper" in radar["tracked_repos"]

    # 3. /interest
    res_interest = await bot.handle_message_text("/interest tinygrad", chat_id="123")
    assert res_interest["action"] == "interest"
    assert res_interest["keyword"] == "tinygrad"

    radar = await get_topic_radar(topic_db)
    assert "tinygrad" in radar["custom_keywords"]

    # 4. /radar
    res_radar = await bot.handle_message_text("/radar", chat_id="123")
    assert res_radar["action"] == "radar"


def test_mask_telegram_url():
    """mask_telegram_url redacts bot token."""
    raw = "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11/sendMessage"
    masked = mask_telegram_url(raw)
    assert "123456:ABC-DEF" not in masked
    assert "bot<REDACTED>" in masked


@pytest.mark.asyncio
async def test_telegram_unauthorized_user_blocked(topic_db: Database):
    """Foreign/unauthorized Telegram users cannot issue commands or click buttons in production."""
    prod_settings = Settings(
        environment="production",
        telegram_chat_id="12345678",
        telegram_bot_token="mock_token",
    )
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    req = httpx.Request("POST", "https://api.telegram.org/bot123/sendMessage")
    mock_client.post = AsyncMock(return_value=httpx.Response(200, json={"ok": True, "result": {"message_id": 1}}, request=req))

    bot = TelegramBot(prod_settings, client=mock_client, db=topic_db)
    bot._external_client = True

    # Unauthorized user tries /track
    res = await bot.handle_message_text("/track bad/repo", chat_id="99999999")
    assert res["action"] == "unauthorized"

    # Unauthorized callback
    res_cb = await bot.handle_callback_query("feedback:1:hit", "cb1", chat_id="99999999")
    assert res_cb["action"] == "unauthorized"