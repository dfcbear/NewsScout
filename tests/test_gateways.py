"""tests/test_gateways.py
~~~~~~~~~~~~~~~~~~~~~~~~
100% offline mocked unit and integration tests for BaseMessengerGateway,
DeliveryReceipt, PairingQRResult, DeliveryError exception hierarchy,
TelegramGateway, and SignalGateway.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import SecretStr
import pytest

from newsscout.config import Settings
from newsscout.delivery.base import (
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_PAIRING_REQUIRED,
    BaseMessengerGateway,
    DeliveryAuthError,
    DeliveryChannelDisabledError,
    DeliveryConnectionError,
    DeliveryError,
    DeliveryPayloadError,
    DeliveryRateLimitError,
    DeliveryReceipt,
    DeliveryTimeoutError,
    PairingQRResult,
)
from newsscout.delivery.signal_gateway import SignalGateway
from newsscout.delivery.telegram import TelegramGateway
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    DialogueScript,
    Stage2Category,
    Track,
    TrackType,
)


@pytest.fixture
def sample_card() -> DecisionCard:
    return DecisionCard(
        breakthrough_id=42,
        title="vLLM: High-Throughput Serving",
        repo_url="https://github.com/vllm-project/vllm",
        breakthrough_score=9.4,
        roi_score=8.8,
        category=Stage2Category.CORE,
        data=DecisionCardData(
            tldr="Accelerated LLM inference via PagedAttention on RTX 4090.",
            use_case="Local high-concurrency model serving.",
            comparison="3x faster than HuggingFace TGI baseline.",
            quickstart="docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest",
            hardware_requirements="24GB VRAM (RTX 4090)",
            license="Apache-2.0",
        ),
    )


@pytest.fixture
def sample_breakthrough(sample_card: DecisionCard) -> Breakthrough:
    return Breakthrough(
        id=42,
        raw_item_id=101,
        title=sample_card.title,
        repo_url=sample_card.repo_url or "https://example.com",
        breakthrough_score=sample_card.breakthrough_score,
        roi_score=sample_card.roi_score,
        category=sample_card.category,
        tldr=sample_card.data.tldr,
        use_case=sample_card.data.use_case,
        comparison=sample_card.data.comparison,
        quickstart=sample_card.data.quickstart,
        hardware_requirements=sample_card.data.hardware_requirements,
        license=sample_card.data.license,
        card_markdown="### Markdown card",
    )


# ============================================================================
# 1. Base Contracts & Exceptions Tests
# ============================================================================

class TestDeliveryDataContracts:
    def test_delivery_receipt_serialization(self):
        receipt = DeliveryReceipt(
            channel="telegram",
            success=True,
            message_id="101",
            recipient="999",
            details={"foo": "bar"},
        )
        d = receipt.to_dict()
        assert d["channel"] == "telegram"
        assert d["success"] is True
        assert d["message_id"] == "101"
        assert d["recipient"] == "999"
        assert d["error"] is None
        assert d["details"] == {"foo": "bar"}
        assert "delivered_at" in d

    def test_pairing_qr_result_serialization(self):
        res = PairingQRResult(
            channel="signal",
            status=STATUS_PAIRING_REQUIRED,
            qr_data="sgnl://linkdevice?uuid=123",
            qr_image_url="data:image/png;base64,abc",
            message="Scan QR to pair",
        )
        d = res.to_dict()
        assert d["channel"] == "signal"
        assert d["status"] == STATUS_PAIRING_REQUIRED
        assert d["qr_data"] == "sgnl://linkdevice?uuid=123"
        assert d["qr_image_url"] == "data:image/png;base64,abc"
        assert d["message"] == "Scan QR to pair"

    def test_exception_hierarchy(self):
        err = DeliveryRateLimitError("Rate limit exceeded", retry_after=5.5)
        assert isinstance(err, DeliveryError)
        assert err.retry_after == 5.5

        assert issubclass(DeliveryConnectionError, DeliveryError)
        assert issubclass(DeliveryTimeoutError, DeliveryError)
        assert issubclass(DeliveryAuthError, DeliveryError)
        assert issubclass(DeliveryChannelDisabledError, DeliveryError)
        assert issubclass(DeliveryPayloadError, DeliveryError)


# ============================================================================
# 2. Telegram Gateway Tests
# ============================================================================

class TestTelegramGateway:
    @pytest.mark.asyncio
    async def test_enabled_status_and_pairing(self):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        gw = TelegramGateway(settings=settings)
        assert gw.channel_name == "telegram"
        assert gw.is_enabled() is True
        assert await gw.is_connected() is True

        pair = await gw.get_pairing_status()
        assert pair.channel == "telegram"
        assert pair.status == STATUS_CONNECTED
        assert pair.qr_data is None

    @pytest.mark.asyncio
    async def test_disabled_status_when_no_token(self):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("")
        settings.telegram_chat_id = ""

        gw = TelegramGateway(settings=settings)
        assert gw.is_enabled() is False
        assert await gw.is_connected() is False

        pair = await gw.get_pairing_status()
        assert pair.channel == "telegram"
        assert pair.status == STATUS_DISABLED

    @pytest.mark.asyncio
    async def test_send_card_success(self, sample_card):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot._post_with_retry = AsyncMock(return_value={"ok": True, "result": {"message_id": 777}})
        bot._build_feedback_keyboard = MagicMock(return_value={"inline_keyboard": []})

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_card(sample_card)

        assert receipt.success is True
        assert receipt.message_id == "777"
        assert receipt.channel == "telegram"
        assert receipt.recipient == "999"
        bot._post_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_card_with_breakthrough_object(self, sample_breakthrough):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot._post_with_retry = AsyncMock(return_value={"ok": True, "result": {"message_id": 888}})
        bot._build_feedback_keyboard = MagicMock(return_value={"inline_keyboard": []})

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_card(sample_breakthrough)

        assert receipt.success is True
        assert receipt.message_id == "888"

    @pytest.mark.asyncio
    async def test_send_card_disabled_shields_error(self, sample_card):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("")
        settings.telegram_chat_id = "999"

        gw = TelegramGateway(settings=settings)
        receipt = await gw.send_card(sample_card)
        assert receipt.success is False
        assert "not configured" in receipt.error

    @pytest.mark.asyncio
    async def test_send_card_missing_chat_id_shields_error(self, sample_card):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = ""

        gw = TelegramGateway(settings=settings)
        receipt = await gw.send_card(sample_card)
        assert receipt.success is False
        assert "No recipient" in receipt.error

    @pytest.mark.asyncio
    async def test_send_card_exception_shielded(self, sample_card):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot._post_with_retry = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        bot._build_feedback_keyboard = MagicMock(return_value={})

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_card(sample_card)
        assert receipt.success is False
        assert "Connection refused" in receipt.error

    @pytest.mark.asyncio
    async def test_send_audio_success(self, tmp_path):
        audio_file = tmp_path / "briefing.mp3"
        audio_file.write_bytes(b"ID3mockaudio")

        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot._post_with_retry_multipart = AsyncMock(return_value={"ok": True, "result": {"message_id": 555}})

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_audio(audio_file, caption="Daily Digest")

        assert receipt.success is True
        assert receipt.message_id == "555"

    @pytest.mark.asyncio
    async def test_send_audio_with_track_object(self, tmp_path):
        audio_file = tmp_path / "track.mp3"
        audio_file.write_bytes(b"ID3mockaudio")

        track = Track(
            digest_id=1,
            track_number=1,
            track_type=TrackType.EXECUTIVE_SUMMARY,
            title="Executive Summary",
            duration_seconds=185,
            audio_path=str(audio_file),
            script=DialogueScript(turns=[]),
        )

        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot._post_with_retry_multipart = AsyncMock(return_value={"ok": True, "result": {"message_id": 556}})

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_audio(track)

        assert receipt.success is True
        assert receipt.message_id == "556"

    @pytest.mark.asyncio
    async def test_send_audio_missing_file_shields_error(self, tmp_path):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        gw = TelegramGateway(settings=settings)
        receipt = await gw.send_audio(tmp_path / "nonexistent.mp3")
        assert receipt.success is False
        assert "Audio file not found" in receipt.error

    @pytest.mark.asyncio
    async def test_send_text_success(self):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=321)

        gw = TelegramGateway(bot=bot, settings=settings)
        receipt = await gw.send_text("Hello Telegram")
        assert receipt.success is True
        assert receipt.message_id == "321"

    @pytest.mark.asyncio
    async def test_backward_compatibility_delegates(self):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        bot = MagicMock()
        bot.is_authorized.return_value = True
        bot.send_message = AsyncMock(return_value=123)
        bot.handle_callback_query = AsyncMock(return_value={"ok": True})
        bot.handle_message_text = AsyncMock(return_value={"action": "help"})
        bot.poll_updates = AsyncMock(return_value=(1, []))
        bot.process_update = AsyncMock(return_value={"processed": True})

        gw = TelegramGateway(bot=bot, settings=settings)
        assert gw.is_authorized("999") is True
        assert await gw.send_message("Test") == 123
        assert (await gw.handle_callback_query("feedback:1:hit", "cb1", "999"))["ok"] is True
        assert (await gw.handle_message_text("/help", "999"))["action"] == "help"
        assert (await gw.poll_updates()) == (1, [])
        assert (await gw.process_update({}))["processed"] is True

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self):
        settings = MagicMock(spec=Settings)
        settings.telegram_bot_token = SecretStr("123:ABC")
        settings.telegram_chat_id = "999"

        async with TelegramGateway(settings=settings) as gw:
            assert gw.channel_name == "telegram"
            assert gw.is_enabled() is True


# ============================================================================
# 4. Signal Gateway Tests
# ============================================================================

class TestSignalGateway:
    def test_format_signal_card(self, sample_card):
        text = SignalGateway.format_signal_card(sample_card)

        # Standard CommonMark bold
        assert "**vLLM: High-Throughput Serving**" in text
        # Embeds [#id]
        assert "[#42]" in text
        # Markdown link
        assert "([Repository](https://github.com/vllm-project/vllm))" in text
        # Quick-reply legend
        assert "Antworte mit:" in text
        assert "🎯 Hit | 💤 Hype | ✅ Bekannt | 🚀 Inspiration" in text

    def test_is_enabled_logic(self):
        s_disabled = Settings(signal_enabled=False)
        gw = SignalGateway(settings=s_disabled)
        assert gw.is_enabled() is False

        s_enabled = Settings(
            signal_enabled=True,
            signal_bridge_url="http://localhost:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw_enabled = SignalGateway(settings=s_enabled)
        assert gw_enabled.is_enabled() is True
        assert gw_enabled.channel_name == "signal"

    @pytest.mark.asyncio
    async def test_send_card_success(self, sample_card):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v2/send"
            body = json.loads(request.read())
            assert body["number"] == "+491510000000"
            assert "+491701111111" in body["recipients"]
            assert "**vLLM: High-Throughput Serving**" in body["message"]
            return httpx.Response(201, json={"timestamp": 1726700000123})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings, client=client)

        receipt = await gw.send_card(sample_card)
        assert receipt.success is True
        assert receipt.channel == "signal"
        assert receipt.message_id == "1726700000123"
        assert receipt.recipient == "+491701111111"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_send_card_missing_number_error(self, sample_card):
        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings)
        receipt = await gw.send_card(sample_card)
        assert receipt.success is False
        assert "Signal sender number not configured" in receipt.error

    @pytest.mark.asyncio
    async def test_send_audio_success(self, tmp_path):
        audio_file = tmp_path / "podcast.mp3"
        audio_file.write_bytes(b"MOCK_SIGNAL_AUDIO_PAYLOAD")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v2/send"
            body = json.loads(request.read())
            assert body["number"] == "+491510000000"
            assert len(body["base64_attachments"]) == 1
            att = body["base64_attachments"][0]
            assert att.startswith("data:audio/mpeg;base64,")
            return httpx.Response(201, json={"timestamp": 1726700000555})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings, client=client)

        receipt = await gw.send_audio(str(audio_file), caption="Signal Audio Digest")
        assert receipt.success is True
        assert receipt.message_id == "1726700000555"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_send_audio_missing_file(self, tmp_path):
        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings)
        receipt = await gw.send_audio(str(tmp_path / "notfound.mp3"))
        assert receipt.success is False
        assert "Audio file not found" in receipt.error

    @pytest.mark.asyncio
    async def test_send_text_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v2/send"
            return httpx.Response(201, json={"timestamp": 1726700000999})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings, client=client)

        receipt = await gw.send_text("Hello Signal")
        assert receipt.success is True
        assert receipt.message_id == "1726700000999"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_pairing_status_connected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/about":
                return httpx.Response(200, json={"version": "0.13.0"})
            elif request.url.path == "/v1/accounts":
                return httpx.Response(200, json=["+491510000000"])
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings, client=client)

        status = await gw.get_pairing_status()
        assert status.status == STATUS_CONNECTED
        assert status.channel == "signal"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_pairing_status_qr_link(self):
        raw_png = b"\x89PNG\r\n\x1a\n\x00\x00mocksignalqr"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/about":
                return httpx.Response(200, json={"version": "0.13.0"})
            elif request.url.path == "/v1/accounts":
                return httpx.Response(200, json=[])
            elif request.url.path == "/v1/qrcodelink":
                return httpx.Response(200, content=raw_png, headers={"Content-Type": "image/png"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://mock-signal:8080",
            signal_sender_number="+491510000000",
            signal_recipients=["+491701111111"],
        )
        gw = SignalGateway(settings=settings, client=client)

        status = await gw.get_pairing_status()
        assert status.status == STATUS_PAIRING_REQUIRED
        assert status.qr_image_url.startswith("data:image/png;base64,")
        await client.aclose()
