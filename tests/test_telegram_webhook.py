"""tests.test_telegram_webhook.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the Telegram webhook endpoint (POST /api/telegram/webhook):
  * Webhook with valid secret token
  * Webhook with invalid secret token (403)
  * Webhook with no secret configured (open mode)
  * Webhook processing a message update (/help command)
  * Webhook processing a callback query (feedback button)
  * Webhook with malformed body (graceful error)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations


@pytest.fixture
def webhook_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "webhook_test.db"
    db = Database(db_file)
    with db.get_sync_connection() as c:
        apply_migrations(conn=c)
    return db


@pytest.fixture
def webhook_settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        db_path=tmp_path / "webhook_test.db",
        telegram_bot_token=SecretStr("123456789:MOCK_BOT_TOKEN"),
        telegram_chat_id="987654321",
        telegram_webhook_url="https://newscout.tail-scale/api/telegram/webhook",
        telegram_webhook_secret=SecretStr("my-secret-token"),
    )


@pytest.fixture
def webhook_settings_no_secret(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        db_path=tmp_path / "webhook_test.db",
        telegram_bot_token=SecretStr("123456789:MOCK_BOT_TOKEN"),
        telegram_chat_id="987654321",
        telegram_webhook_url="https://newscout.tail-scale/api/telegram/webhook",
        telegram_webhook_secret=SecretStr(""),
    )


@pytest.fixture
async def webhook_client(webhook_db: Database, webhook_settings: Settings):
    app = create_app(webhook_settings)
    app.state.db = webhook_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def webhook_client_no_secret(webhook_db: Database, webhook_settings_no_secret: Settings):
    app = create_app(webhook_settings_no_secret)
    app.state.db = webhook_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _mock_post_response(*args, **kwargs):
    """Mock for TelegramBot._post_with_retry returning a successful Telegram API response."""
    return {"ok": True, "result": {"message_id": 42}}


# ============================================================================
# Secret Token Verification Tests
# ============================================================================


@pytest.mark.asyncio
async def test_webhook_valid_secret(webhook_client: AsyncClient):
    """Webhook with correct X-Telegram-Bot-Api-Secret-Token returns 200."""
    update = {"update_id": 1, "message": {"text": "/help", "chat": {"id": 987654321}}}
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_webhook_invalid_secret(webhook_client: AsyncClient):
    """Webhook with wrong secret token returns 403 Forbidden."""
    update = {"update_id": 1, "message": {"text": "/help", "chat": {"id": 987654321}}}
    resp = await webhook_client.post(
        "/api/telegram/webhook",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_webhook_missing_secret(webhook_client: AsyncClient):
    """Webhook without secret token header returns 403 when secret is configured."""
    update = {"update_id": 1, "message": {"text": "/help", "chat": {"id": 987654321}}}
    resp = await webhook_client.post("/api/telegram/webhook", json=update)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_webhook_no_secret_configured(webhook_client_no_secret: AsyncClient):
    """Webhook works without secret token when no secret is configured."""
    update = {"update_id": 1, "message": {"text": "/help", "chat": {"id": 987654321}}}
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client_no_secret.post("/api/telegram/webhook", json=update)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True


# ============================================================================
# Message Update Processing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_webhook_help_command(webhook_client: AsyncClient):
    """Webhook processes /help message and returns help action."""
    update = {
        "update_id": 1,
        "message": {
            "text": "/help",
            "chat": {"id": 987654321},
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "help"


@pytest.mark.asyncio
async def test_webhook_start_command(webhook_client: AsyncClient):
    """Webhook processes /start message and returns help action."""
    update = {
        "update_id": 2,
        "message": {
            "text": "/start",
            "chat": {"id": 987654321},
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "help"


@pytest.mark.asyncio
async def test_webhook_unauthorized_chat(webhook_db: Database, tmp_path: Path):
    """Webhook rejects messages from unauthorized chat IDs."""
    settings = Settings(
        environment="production",
        db_path=tmp_path / "webhook_test.db",
        telegram_bot_token=SecretStr("123456789:MOCK_BOT_TOKEN"),
        telegram_chat_id="987654321",
        telegram_webhook_url="https://newscout.tail-scale/api/telegram/webhook",
        telegram_webhook_secret=SecretStr("my-secret-token"),
    )
    app = create_app(settings)
    app.state.db = webhook_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        update = {
            "update_id": 3,
            "message": {
                "text": "/help",
                "chat": {"id": 111111111},  # Wrong chat ID
            },
        }
        with patch(
            "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
            new_callable=AsyncMock,
            side_effect=_mock_post_response,
        ):
            resp = await client.post(
                "/api/telegram/webhook",
                json=update,
                headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "unauthorized"


@pytest.mark.asyncio
async def test_webhook_unknown_command(webhook_client: AsyncClient):
    """Webhook processes unknown text and returns unknown action."""
    update = {
        "update_id": 4,
        "message": {
            "text": "hello world",
            "chat": {"id": 987654321},
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "unknown"


# ============================================================================
# Callback Query Processing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_webhook_feedback_callback(webhook_client: AsyncClient):
    """Webhook processes feedback callback query."""
    update = {
        "update_id": 5,
        "callback_query": {
            "id": "callback-123",
            "data": "feedback:42:hit",
            "from": {"id": 987654321},
            "message": {
                "chat": {"id": 987654321},
            },
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ), patch(
        "newsscout.storage.preferences.PreferencesService.record_feedback",
        new_callable=AsyncMock,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "feedback"
    assert data["result"]["breakthrough_id"] == 42


@pytest.mark.asyncio
async def test_webhook_play_callback(webhook_client: AsyncClient):
    """Webhook processes play track callback query."""
    update = {
        "update_id": 6,
        "callback_query": {
            "id": "callback-456",
            "data": "play:1:2",
            "from": {"id": 987654321},
            "message": {
                "chat": {"id": 987654321},
            },
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "play_track"
    assert data["result"]["digest_id"] == 1
    assert data["result"]["track_number"] == 2


@pytest.mark.asyncio
async def test_webhook_playall_callback(webhook_client: AsyncClient):
    """Webhook processes play-all callback query."""
    update = {
        "update_id": 7,
        "callback_query": {
            "id": "callback-789",
            "data": "playall:1",
            "from": {"id": 987654321},
            "message": {
                "chat": {"id": 987654321},
            },
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot._post_with_retry",
        new_callable=AsyncMock,
        side_effect=_mock_post_response,
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "play_all"
    assert data["result"]["digest_id"] == 1


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.asyncio
async def test_webhook_empty_update(webhook_client: AsyncClient):
    """Webhook handles empty update gracefully."""
    update = {"update_id": 8}
    resp = await webhook_client.post(
        "/api/telegram/webhook",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["action"] == "ignored"


@pytest.mark.asyncio
async def test_webhook_processing_error_returns_200(webhook_client: AsyncClient):
    """Webhook returns 200 even when processing raises (so Telegram doesn't retry)."""
    update = {
        "update_id": 9,
        "message": {
            "text": "/help",
            "chat": {"id": 987654321},
        },
    }
    with patch(
        "newsscout.delivery.telegram_bot.TelegramBot.process_update",
        side_effect=RuntimeError("boom"),
    ):
        resp = await webhook_client.post(
            "/api/telegram/webhook",
            json=update,
            headers={"X-Telegram-Bot-Api-Secret-Token": "my-secret-token"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["error"] == "processing_error"
