"""tests/test_telegram.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the Telegram bot delivery module:
  * Decision card sending with inline feedback buttons
  * Audio track delivery
  * Digest menu with track selection
  * Callback query handling (feedback, play, playall)
  * Retry logic with exponential backoff
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from newsscout.config import Settings
from newsscout.delivery.telegram_bot import (
    FEEDBACK_BUTTONS,
    TelegramBot,
)
from newsscout.storage.models import (
    Breakthrough,
    Digest,
    DigestSlot,
    DigestStatus,
    Feedback,
    FeedbackRating,
    Stage2Category,
    Track,
    TrackType,
)
from tests.conftest import DataFactory


# ============================================================================
# Helper: Mock httpx responses
# ============================================================================

def make_mock_response(
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock httpx.Response object."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body or {"ok": True, "result": {"message_id": 42}}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=resp
        )
    return resp


# ============================================================================
# TelegramBot Initialization Tests
# ============================================================================

class TestTelegramBotInit:
    """Tests for TelegramBot initialization and lifecycle."""

    def test_init_with_settings(self, test_settings: Settings):
        """Bot initializes with settings and builds base URL."""
        bot = TelegramBot(test_settings)
        assert bot.settings == test_settings
        assert "123456789:MOCK_BOT_TOKEN_FOR_TESTS" in bot._base_url

    def test_init_with_custom_client(self, test_settings: Settings):
        """Bot accepts an external httpx.AsyncClient."""
        client = httpx.AsyncClient()
        bot = TelegramBot(test_settings, client=client)
        assert bot._external_client is True
        assert bot._client is client

    @pytest.mark.asyncio
    async def test_context_manager_creates_client(self, test_settings: Settings):
        """Async context manager creates an httpx.AsyncClient if none provided."""
        bot = TelegramBot(test_settings)
        assert bot._client is None
        async with bot:
            assert bot._client is not None
            assert not bot._client.is_closed
        # After context exit, client should be closed
        assert bot._client.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_with_external_client_not_closed(self, test_settings: Settings):
        """External client is not closed on context exit."""
        client = httpx.AsyncClient()
        bot = TelegramBot(test_settings, client=client)
        async with bot:
            pass
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.asyncio
    async def test_close_without_external_client(self, test_settings: Settings):
        """close() closes internally-created client."""
        bot = TelegramBot(test_settings)
        async with bot:
            pass
        await bot.close()

    def test_client_property_without_context_raises(self, test_settings: Settings):
        """client property raises if not used as context manager."""
        bot = TelegramBot(test_settings)
        with pytest.raises(RuntimeError, match="context manager"):
            _ = bot.client


# ============================================================================
# Feedback Keyboard Tests
# ============================================================================

class TestFeedbackKeyboard:
    """Tests for inline feedback button keyboard construction."""

    def test_build_feedback_keyboard_has_all_ratings(self, test_settings: Settings):
        """Keyboard includes all 4 FeedbackRating buttons."""
        bot = TelegramBot(test_settings)
        keyboard = bot._build_feedback_keyboard(breakthrough_id=1)

        assert "inline_keyboard" in keyboard
        buttons = [btn for row in keyboard["inline_keyboard"] for btn in row]
        assert len(buttons) == 4

        button_texts = [b["text"] for b in buttons]
        for rating in FeedbackRating:
            assert FEEDBACK_BUTTONS[rating] in button_texts

    def test_build_feedback_keyboard_callback_data_format(self, test_settings: Settings):
        """Callback data follows feedback:id:rating format."""
        bot = TelegramBot(test_settings)
        keyboard = bot._build_feedback_keyboard(breakthrough_id=42)

        buttons = [btn for row in keyboard["inline_keyboard"] for btn in row]
        for btn in buttons:
            assert btn["callback_data"].startswith("feedback:42:")
            rating_part = btn["callback_data"].split(":")[2]
            assert rating_part in [r.value for r in FeedbackRating]

    def test_build_feedback_keyboard_2x2_grid(self, test_settings: Settings):
        """Buttons are arranged in a 2x2 grid."""
        bot = TelegramBot(test_settings)
        keyboard = bot._build_feedback_keyboard(breakthrough_id=1)

        rows = keyboard["inline_keyboard"]
        assert len(rows) == 2
        assert len(rows[0]) == 2
        assert len(rows[1]) == 2

    def test_build_feedback_keyboard_none_id(self, test_settings: Settings):
        """Keyboard handles None breakthrough_id gracefully."""
        bot = TelegramBot(test_settings)
        keyboard = bot._build_feedback_keyboard(breakthrough_id=None)
        buttons = [btn for row in keyboard["inline_keyboard"] for btn in row]
        assert len(buttons) == 4
        assert "feedback:None:" in buttons[0]["callback_data"]


# ============================================================================
# Digest Menu Keyboard Tests
# ============================================================================

class TestDigestMenuKeyboard:
    """Tests for digest menu inline keyboard."""

    def test_build_digest_menu_keyboard_playall_button(self, test_settings: Settings):
        """Keyboard includes a Play All button."""
        bot = TelegramBot(test_settings)
        digest = DataFactory.create_digest()
        digest.id = 1
        digest.tracks = [DataFactory.create_track(digest_id=1, track_number=1)]

        keyboard = bot._build_digest_menu_keyboard(digest)
        rows = keyboard["inline_keyboard"]

        assert rows[0][0]["callback_data"] == "playall:1"
        assert "Play All" in rows[0][0]["text"]

    def test_build_digest_menu_keyboard_track_buttons(self, test_settings: Settings):
        """Keyboard includes a button for each track."""
        bot = TelegramBot(test_settings)
        digest = DataFactory.create_digest()
        digest.id = 5
        digest.tracks = [
            DataFactory.create_track(digest_id=5, track_number=i) for i in range(1, 4)
        ]

        keyboard = bot._build_digest_menu_keyboard(digest)
        rows = keyboard["inline_keyboard"]

        # First row is Play All, then one row per track
        assert len(rows) == 4
        for i, row in enumerate(rows[1:], start=1):
            assert row[0]["callback_data"] == f"play:5:{i}"

    def test_build_digest_menu_keyboard_empty_tracks(self, test_settings: Settings):
        """Keyboard with no tracks still has Play All button."""
        bot = TelegramBot(test_settings)
        digest = DataFactory.create_digest()
        digest.id = 1
        digest.tracks = []

        keyboard = bot._build_digest_menu_keyboard(digest)
        rows = keyboard["inline_keyboard"]
        assert len(rows) == 1
        assert rows[0][0]["callback_data"] == "playall:1"


# ============================================================================
# Send Decision Card Tests
# ============================================================================

class TestSendDecisionCard:
    """Tests for sending decision cards via Telegram."""

    @pytest.mark.asyncio
    async def test_send_decision_card_success(self, test_settings: Settings):
        """Decision card is sent with HTML and inline keyboard."""
        bot = TelegramBot(test_settings)
        bt = DataFactory.create_breakthrough()
        bt.id = 1

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response(
            json_body={"ok": True, "result": {"message_id": 123}}
        ))
        bot._client = mock_client
        bot._external_client = True

        msg_id = await bot.send_decision_card(bt)

        assert msg_id == 123
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["parse_mode"] == "HTML"
        assert "reply_markup" in payload
        markup = json.loads(payload["reply_markup"])
        assert "inline_keyboard" in markup

    @pytest.mark.asyncio
    async def test_send_decision_card_custom_chat_id(self, test_settings: Settings):
        """Custom chat_id overrides settings."""
        bot = TelegramBot(test_settings)
        bt = DataFactory.create_breakthrough()
        bt.id = 1

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        await bot.send_decision_card(bt, chat_id="custom_chat_123")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["chat_id"] == "custom_chat_123"


# ============================================================================
# Send Audio Track Tests
# ============================================================================

class TestSendAudioTrack:
    """Tests for sending audio tracks via Telegram."""

    @pytest.mark.asyncio
    async def test_send_audio_track_success(self, test_settings: Settings, tmp_path: Path):
        """Audio track file is sent as multipart."""
        from tests.conftest import create_silent_mp3

        audio_path = tmp_path / "track_1.mp3"
        audio_path.write_bytes(create_silent_mp3(2.0))

        track = DataFactory.create_track(digest_id=1, track_number=1)
        track.audio_path = str(audio_path)

        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response(
            json_body={"ok": True, "result": {"message_id": 200}}
        ))
        bot._client = mock_client
        bot._external_client = True

        msg_id = await bot.send_audio_track(track)

        assert msg_id == 200
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_audio_track_missing_file(self, test_settings: Settings):
        """Missing audio file returns 0 and logs error."""
        track = DataFactory.create_track(digest_id=1, track_number=1)
        track.audio_path = "/nonexistent/path.mp3"

        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        bot._client = mock_client
        bot._external_client = True

        msg_id = await bot.send_audio_track(track)
        assert msg_id == 0
        mock_client.post.assert_not_called()


# ============================================================================
# Send Digest Menu Tests
# ============================================================================

class TestSendDigestMenu:
    """Tests for sending the interactive digest menu."""

    @pytest.mark.asyncio
    async def test_send_digest_menu_success(self, test_settings: Settings):
        """Digest menu is sent with track list and inline buttons."""
        bot = TelegramBot(test_settings)
        digest = DataFactory.create_digest()
        digest.id = 1
        digest.total_duration_seconds = 1200
        digest.tracks = [
            DataFactory.create_track(digest_id=1, track_number=1, title="Track 1"),
            DataFactory.create_track(digest_id=1, track_number=2, title="Track 2"),
        ]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response(
            json_body={"ok": True, "result": {"message_id": 999}}
        ))
        bot._client = mock_client
        bot._external_client = True

        msg_id = await bot.send_digest_menu(digest)

        assert msg_id == 999
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["parse_mode"] == "HTML"
        assert "NewsScout Digest" in payload["text"]
        markup = json.loads(payload["reply_markup"])
        assert "inline_keyboard" in markup
        assert "Play All" in markup["inline_keyboard"][0][0]["text"]

    @pytest.mark.asyncio
    async def test_send_digest_menu_includes_duration(self, test_settings: Settings):
        """Digest menu text includes total duration."""
        bot = TelegramBot(test_settings)
        digest = DataFactory.create_digest()
        digest.id = 1
        digest.total_duration_seconds = 1500  # 25:00
        digest.tracks = [DataFactory.create_track(digest_id=1, track_number=1)]

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        await bot.send_digest_menu(digest)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "25:00" in payload["text"]


# ============================================================================
# Callback Query Handling Tests
# ============================================================================

class TestCallbackQueryHandling:
    """Tests for inline button callback routing."""

    @pytest.mark.asyncio
    async def test_handle_feedback_callback_hit(self, test_settings: Settings, migrated_db):
        """Feedback callback with 'hit' rating is processed correctly."""
        from newsscout.storage.preferences import PreferencesService

        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        pref_service = PreferencesService(migrated_db, test_settings)
        bot._preferences = pref_service

        # Insert a breakthrough to attach feedback to
        bt = DataFactory.create_breakthrough(raw_item_id=1)
        bt.id = 1
        params = bt.to_db_params()
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('github', 'test:1', 'Test', 'http://test.com', 'content', '{}', datetime('now'));
            """,
        )
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, repo_url, breakthrough_score, roi_score,
                category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                params["raw_item_id"], params["title"], params["repo_url"],
                params["breakthrough_score"], params["roi_score"], params["category"],
                params["tldr"], params["use_case"], params["comparison"],
                params["quickstart"], params["hardware_requirements"], params["license"],
                params["card_markdown"], params["evaluation_raw_json"],
                params["is_watchlisted"], params["evaluated_at"],
            ),
        )

        result = await bot.handle_callback_query(
            callback_data="feedback:1:hit",
            callback_query_id="cb_123",
            chat_id="test_chat",
            telegram_user_id=99887766,
        )

        assert result["action"] == "feedback"
        assert result["breakthrough_id"] == 1
        assert result["rating"] == FeedbackRating.HIT

    @pytest.mark.asyncio
    async def test_handle_feedback_callback_all_ratings(self, test_settings: Settings, migrated_db):
        """All 4 feedback ratings are handled correctly."""
        from newsscout.storage.preferences import PreferencesService

        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        pref_service = PreferencesService(migrated_db, test_settings)
        bot._preferences = pref_service

        # Insert a breakthrough to attach feedback to
        bt = DataFactory.create_breakthrough(raw_item_id=1)
        bt.id = 1
        params = bt.to_db_params()
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('github', 'test:1', 'Test', 'http://test.com', 'content', '{}', datetime('now'));
            """,
        )
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, repo_url, breakthrough_score, roi_score,
                category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                params["raw_item_id"], params["title"], params["repo_url"],
                params["breakthrough_score"], params["roi_score"], params["category"],
                params["tldr"], params["use_case"], params["comparison"],
                params["quickstart"], params["hardware_requirements"], params["license"],
                params["card_markdown"], params["evaluation_raw_json"],
                params["is_watchlisted"], params["evaluated_at"],
            ),
        )

        for rating in FeedbackRating:
            result = await bot.handle_callback_query(
                callback_data=f"feedback:1:{rating.value}",
                callback_query_id="cb_123",
                chat_id="test_chat",
                telegram_user_id=99887766,
            )
            assert result["action"] == "feedback"
            assert result["rating"] == rating

    @pytest.mark.asyncio
    async def test_handle_play_callback(self, test_settings: Settings):
        """Play track callback returns play_track action."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot.handle_callback_query(
            callback_data="play:5:3",
            callback_query_id="cb_456",
            chat_id="test_chat",
        )

        assert result["action"] == "play_track"
        assert result["digest_id"] == 5
        assert result["track_number"] == 3
        assert result["chat_id"] == "test_chat"

    @pytest.mark.asyncio
    async def test_handle_playall_callback(self, test_settings: Settings):
        """Play all callback returns play_all action."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot.handle_callback_query(
            callback_data="playall:10",
            callback_query_id="cb_789",
            chat_id="test_chat",
        )

        assert result["action"] == "play_all"
        assert result["digest_id"] == 10
        assert result["chat_id"] == "test_chat"

    @pytest.mark.asyncio
    async def test_handle_unknown_callback(self, test_settings: Settings):
        """Unknown callback format returns unknown action."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot.handle_callback_query(
            callback_data="unknown:action:data",
            callback_query_id="cb_000",
            chat_id="test_chat",
        )

        assert result["action"] == "unknown"

    @pytest.mark.asyncio
    async def test_handle_feedback_invalid_rating(self, test_settings: Settings):
        """Invalid rating string is handled gracefully."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot.handle_callback_query(
            callback_data="feedback:1:invalid_rating",
            callback_query_id="cb_000",
            chat_id="test_chat",
        )

        assert result["action"] == "error"

    @pytest.mark.asyncio
    async def test_handle_play_invalid_track_number(self, test_settings: Settings):
        """Invalid track number is handled gracefully."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot.handle_callback_query(
            callback_data="play:1:abc",
            callback_query_id="cb_000",
            chat_id="test_chat",
        )

        assert result["action"] == "error"


# ============================================================================
# Retry Logic Tests
# ============================================================================

class TestRetryLogic:
    """Tests for HTTP retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_post_with_retry_succeeds_first_try(self, test_settings: Settings):
        """Successful request on first attempt."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        result = await bot._post_with_retry("/sendMessage", {"chat_id": "1", "text": "hi"})
        assert result["ok"] is True
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_post_with_retry_retries_on_429(self, test_settings: Settings):
        """Retries on HTTP 429 and succeeds on retry."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(side_effect=[
            make_mock_response(status_code=429),
            make_mock_response(status_code=200),
        ])
        bot._client = mock_client
        bot._external_client = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await bot._post_with_retry("/sendMessage", {"chat_id": "1", "text": "hi"})

        assert result["ok"] is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_post_with_retry_max_retries_exceeded(self, test_settings: Settings):
        """Raises after max retries on persistent 429."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        error_response = make_mock_response(status_code=429)
        mock_client.post = AsyncMock(return_value=error_response)
        bot._client = mock_client
        bot._external_client = True

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await bot._post_with_retry("/sendMessage", {"chat_id": "1", "text": "hi"})

        assert mock_client.post.call_count == 4  # initial + 3 retries


# ============================================================================
# Send Message Tests
# ============================================================================

class TestSendMessage:
    """Tests for the generic send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, test_settings: Settings):
        """Generic message is sent successfully."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response(
            json_body={"ok": True, "result": {"message_id": 555}}
        ))
        bot._client = mock_client
        bot._external_client = True

        msg_id = await bot.send_message("Hello World")

        assert msg_id == 555
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["text"] == "Hello World"
        assert payload["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_send_message_custom_chat_id(self, test_settings: Settings):
        """Custom chat_id is used when provided."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        await bot.send_message("Test", chat_id="custom_chat")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["chat_id"] == "custom_chat"

    @pytest.mark.asyncio
    async def test_send_message_custom_parse_mode(self, test_settings: Settings):
        """Custom parse_mode is used when provided."""
        bot = TelegramBot(test_settings)
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.post = AsyncMock(return_value=make_mock_response())
        bot._client = mock_client
        bot._external_client = True

        await bot.send_message("Test", parse_mode="MarkdownV2")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["parse_mode"] == "MarkdownV2"


# ============================================================================
# Feedback Button Labels Tests
# ============================================================================

class TestFeedbackButtonLabels:
    """Tests for feedback button label constants."""

    def test_feedback_buttons_contain_emoji(self):
        """All feedback button labels contain emoji."""
        for rating, label in FEEDBACK_BUTTONS.items():
            assert len(label) > 0
            # Check for at least one non-ASCII character (emoji)
            assert any(ord(c) > 127 for c in label)

    def test_feedback_buttons_all_ratings(self):
        """All 4 FeedbackRating values have button labels."""
        assert len(FEEDBACK_BUTTONS) == 4
        for rating in FeedbackRating:
            assert rating in FEEDBACK_BUTTONS

    def test_feedback_button_hit_label(self):
        """HIT button has correct label."""
        assert "Volltreffer" in FEEDBACK_BUTTONS[FeedbackRating.HIT]

    def test_feedback_button_hype_label(self):
        """HYPE button has correct label."""
        assert "banal" in FEEDBACK_BUTTONS[FeedbackRating.HYPE] or "Hype" in FEEDBACK_BUTTONS[FeedbackRating.HYPE]

    def test_feedback_button_known_label(self):
        """KNOWN button has correct label."""
        assert "Kenne" in FEEDBACK_BUTTONS[FeedbackRating.KNOWN]

    def test_feedback_button_inspire_label(self):
        """INSPIRE button has correct label."""
        assert "Inspiration" in FEEDBACK_BUTTONS[FeedbackRating.INSPIRE]
