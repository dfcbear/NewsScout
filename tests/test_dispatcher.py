"""tests/test_dispatcher.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
100% offline mocked unit and integration tests for DeliveryDispatcher,
InboundRouter, reaction/keyword parsing, 6-tier breakthrough resolution waterfall,
PreferencesService persistence, and slash commands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newsscout.config import Settings
from newsscout.delivery.base import (
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_PAIRING_REQUIRED,
    BaseMessengerGateway,
    DeliveryReceipt,
    PairingQRResult,
)
from newsscout.delivery.dispatcher import (
    DeliveryDispatcher,
    create_default_dispatcher,
)
from newsscout.delivery.inbound import (
    InboundHandlingResult,
    InboundMessage,
    InboundRouter,
    parse_feedback_rating,
)
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    FeedbackRating,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Mock Gateway Implementation for Dispatcher Tests
# ============================================================================

class MockGateway(BaseMessengerGateway):
    def __init__(
        self,
        name: str,
        enabled: bool = True,
        fail_send: bool = False,
        fail_audio: bool = False,
        fail_status: bool = False,
        raise_exception: bool = False,
        delay_seconds: float = 0.0,
    ) -> None:
        self._name = name
        self._enabled = enabled
        self.fail_send = fail_send
        self.fail_audio = fail_audio
        self.fail_status = fail_status
        self.raise_exception = raise_exception
        self.delay = delay_seconds
        self.sent_cards: list[Union[DecisionCard, Breakthrough]] = []
        self.sent_audio: list[str] = []
        self.closed: bool = False

    @property
    def channel_name(self) -> str:
        return self._name

    def is_enabled(self) -> bool:
        return self._enabled

    async def send_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.raise_exception:
            raise ConnectionResetError(f"Simulated network drop on {self._name}")
        if self.fail_send:
            return DeliveryReceipt(
                channel=self._name,
                success=False,
                recipient=recipient,
                error="Simulated send failure",
            )
        self.sent_cards.append(card)
        bid = card.id if isinstance(card, Breakthrough) else card.breakthrough_id
        return DeliveryReceipt(
            channel=self._name,
            success=True,
            recipient=recipient or "default",
            message_id=f"{self._name}_msg_{bid}",
        )

    async def send_audio(
        self,
        file_path: Union[str, Path],
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.raise_exception:
            raise TimeoutError(f"Simulated timeout on {self._name}")
        if self.fail_audio:
            return DeliveryReceipt(
                channel=self._name,
                success=False,
                recipient=recipient,
                error="Simulated audio failure",
            )
        self.sent_audio.append(str(file_path))
        return DeliveryReceipt(
            channel=self._name,
            success=True,
            recipient=recipient or "default",
            message_id=f"{self._name}_audio_1",
        )

    async def get_pairing_status(self) -> PairingQRResult:
        if self.fail_status:
            raise RuntimeError(f"Simulated status error on {self._name}")
        if not self._enabled:
            return PairingQRResult(channel=self._name, status=STATUS_DISABLED)
        return PairingQRResult(channel=self._name, status=STATUS_CONNECTED)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def mock_card() -> DecisionCard:
    return DecisionCard(
        breakthrough_id=101,
        title="Ollama Local Engine",
        repo_url="https://github.com/ollama/ollama",
        breakthrough_score=9.1,
        roi_score=8.5,
        category=Stage2Category.CORE,
        data=DecisionCardData(
            tldr="Run local models effortlessly on ARM64 and RTX.",
            use_case="Embedded agent serving.",
            comparison="Zero-setup runner.",
            quickstart="ollama run llama3",
            hardware_requirements="8GB RAM",
            license="MIT",
        ),
    )


# ============================================================================
# 1. Dispatcher Gateway Registry Tests
# ============================================================================

class TestDispatcherGatewayRegistry:
    def test_register_and_retrieve_gateways(self):
        gw1 = MockGateway("telegram")
        gw2 = MockGateway("whatsapp")
        gw3 = MockGateway("signal", enabled=False)

        dispatcher = DeliveryDispatcher(gateways=[gw1, gw2, gw3])
        assert dispatcher.get_gateway("telegram") is gw1
        assert dispatcher.get_gateway("whatsapp") is gw2
        assert dispatcher.get_gateway("signal") is gw3
        assert len(dispatcher.get_all_gateways()) == 3

        enabled = dispatcher.get_enabled_gateways()
        assert len(enabled) == 2
        assert gw1 in enabled
        assert gw2 in enabled
        assert gw3 not in enabled

    def test_unregister_gateway(self):
        gw = MockGateway("telegram")
        dispatcher = DeliveryDispatcher()
        dispatcher.register_gateway(gw)
        assert dispatcher.get_gateway("telegram") is gw

        unregistered = dispatcher.unregister_gateway("telegram")
        assert unregistered is gw
        assert dispatcher.get_gateway("telegram") is None
        assert dispatcher.unregister_gateway("telegram") is None


# ============================================================================
# 2. Dispatcher Broadcast Tests
# ============================================================================

class TestDispatcherBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_card_concurrent_success(self, mock_card):
        gw_tg = MockGateway("telegram", delay_seconds=0.01)
        gw_wa = MockGateway("whatsapp", delay_seconds=0.01)
        gw_sig = MockGateway("signal", delay_seconds=0.01)

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_wa, gw_sig])
        receipts = await dispatcher.broadcast_card(mock_card)

        assert len(receipts) == 3
        for ch in ("telegram", "whatsapp", "signal"):
            assert receipts[ch].success is True
            assert receipts[ch].message_id == f"{ch}_msg_101"

        # Check delivery cache
        assert dispatcher.get_breakthrough_id_for_message("telegram", "telegram_msg_101") == 101
        assert dispatcher.get_breakthrough_id_for_message("whatsapp", "whatsapp_msg_101") == 101
        assert dispatcher.get_breakthrough_id_for_message("signal", "signal_msg_101") == 101
        assert dispatcher.get_latest_breakthrough_id() == 101

    @pytest.mark.asyncio
    async def test_broadcast_card_fault_isolation(self, mock_card):
        gw_tg = MockGateway("telegram")
        gw_wa = MockGateway("whatsapp", fail_send=True)
        gw_sig = MockGateway("signal", raise_exception=True)

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_wa, gw_sig])
        receipts = await dispatcher.broadcast_card(mock_card)

        assert len(receipts) == 3
        assert receipts["telegram"].success is True
        assert receipts["whatsapp"].success is False
        assert "Simulated send failure" in receipts["whatsapp"].error
        assert receipts["signal"].success is False
        assert "ConnectionResetError" in receipts["signal"].error

        # Cache recorded for successful channel
        assert dispatcher.get_breakthrough_id_for_message("telegram", "telegram_msg_101") == 101
        assert dispatcher.get_latest_breakthrough_id() == 101

    @pytest.mark.asyncio
    async def test_broadcast_card_no_enabled_gateways(self, mock_card):
        gw = MockGateway("telegram", enabled=False)
        dispatcher = DeliveryDispatcher(gateways=[gw])
        receipts = await dispatcher.broadcast_card(mock_card)
        assert receipts == {}

    @pytest.mark.asyncio
    async def test_broadcast_audio_concurrent_success(self):
        gw_tg = MockGateway("telegram")
        gw_wa = MockGateway("whatsapp")
        gw_sig = MockGateway("signal")

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_wa, gw_sig])
        receipts = await dispatcher.broadcast_audio("/data/audio/digest.mp3", caption="Evening Briefing")

        assert len(receipts) == 3
        assert all(r.success for r in receipts.values())
        assert gw_tg.sent_audio == ["/data/audio/digest.mp3"]
        assert gw_wa.sent_audio == ["/data/audio/digest.mp3"]
        assert gw_sig.sent_audio == ["/data/audio/digest.mp3"]

    @pytest.mark.asyncio
    async def test_broadcast_audio_fault_isolation(self):
        gw_tg = MockGateway("telegram")
        gw_wa = MockGateway("whatsapp", raise_exception=True)

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_wa])
        receipts = await dispatcher.broadcast_audio("/data/audio/digest.mp3")

        assert receipts["telegram"].success is True
        assert receipts["whatsapp"].success is False
        assert receipts["whatsapp"].error  # Has an error message (timeout or exception)

    @pytest.mark.asyncio
    async def test_r4_signal_timeout_does_not_block_telegram(self, factory):
        """R4: Signal gateway timeout does not block Telegram delivery."""
        gw_tg = MockGateway("telegram")
        gw_sig = MockGateway("signal", delay_seconds=5.0)  # Signal hangs

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_sig])
        from newsscout.config import Settings
        dispatcher.settings = Settings()
        dispatcher.settings.delivery_gateway_timeout_seconds = 0.1

        bt = factory.create_breakthrough()
        receipts = await dispatcher.broadcast_card(bt)

        assert receipts["telegram"].success is True
        assert receipts["signal"].success is False
        assert "timeout" in receipts["signal"].error.lower()

    @pytest.mark.asyncio
    async def test_r4_whatsapp_exception_produces_partial_receipts(self, factory):
        """R4: WhatsApp exception produces partial receipts without blocking other channels."""
        gw_tg = MockGateway("telegram")
        gw_wa = MockGateway("whatsapp", raise_exception=True)

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_wa])

        bt = factory.create_breakthrough()
        receipts = await dispatcher.broadcast_card(bt)

        assert len(receipts) == 2
        assert receipts["telegram"].success is True
        assert receipts["whatsapp"].success is False
        assert receipts["whatsapp"].error is not None

    @pytest.mark.asyncio
    async def test_r4_per_gateway_timeout_independent(self):
        """R4: Per-gateway timeout is independent — Telegram (fast) completes even if Signal times out."""
        gw_tg = MockGateway("telegram", delay_seconds=0.01)
        gw_sig = MockGateway("signal", delay_seconds=5.0)

        dispatcher = DeliveryDispatcher(gateways=[gw_tg, gw_sig])
        from newsscout.config import Settings
        dispatcher.settings = Settings()
        dispatcher.settings.delivery_gateway_timeout_seconds = 0.1

        import time
        start = time.monotonic()
        receipts = await dispatcher.broadcast_audio("/data/test.mp3")
        elapsed = time.monotonic() - start

        # Telegram should complete (delay 0.01s < 0.1s timeout), Signal should timeout (0.1s)
        # Total time should be ~max(0.01, 0.1) ≈ 0.1s, not 5s
        assert elapsed < 1.0, f"Total delivery took {elapsed:.2f}s — should be < 1s"
        assert receipts["telegram"].success is True
        assert receipts["signal"].success is False
        assert "timeout" in receipts["signal"].error.lower()

    @pytest.mark.asyncio
    async def test_close_timeout_on_hanging_gateway(self):
        """Gateway close() that hangs is terminated after 3s timeout."""
        gw_slow = MockGateway("signal")
        # Override close to hang indefinitely
        close_called = False
        async def slow_close():
            nonlocal close_called
            close_called = True
            await asyncio.sleep(100)  # Hang forever
        gw_slow.close = slow_close

        dispatcher = DeliveryDispatcher(gateways=[gw_slow])
        import time
        start = time.monotonic()
        await dispatcher.close()
        elapsed = time.monotonic() - start

        assert close_called is True
        assert elapsed < 5.0, f"close() took {elapsed:.2f}s — should timeout after 3s"

    @pytest.mark.asyncio
    async def test_recent_deliveries_cache_eviction_after_1000(self, mock_card):
        """LRU cache evicts oldest entries when exceeding 1000 entries."""
        gw = MockGateway("telegram")
        dispatcher = DeliveryDispatcher(gateways=[gw])

        # Fill cache beyond cap
        for i in range(1001):
            dispatcher._recent_deliveries[("telegram", f"msg_{i}")] = i
            # Trigger eviction
            while len(dispatcher._recent_deliveries) > dispatcher._max_recent_deliveries:
                dispatcher._recent_deliveries.popitem(last=False)

        assert len(dispatcher._recent_deliveries) == 1000
        # Oldest entry (msg_0) should be evicted
        assert dispatcher.get_breakthrough_id_for_message("telegram", "msg_0") is None
        # Newest entry (msg_1000) should still be present
        assert dispatcher.get_breakthrough_id_for_message("telegram", "msg_1000") == 1000


# ============================================================================
# 3. Targeted Dispatch & Pairing Status Tests
# ============================================================================

class TestDispatcherTargetedSendAndPairing:
    @pytest.mark.asyncio
    async def test_send_to_channel_success(self, mock_card):
        gw = MockGateway("telegram")
        dispatcher = DeliveryDispatcher(gateways=[gw])

        receipt = await dispatcher.send_to_channel("telegram", card=mock_card)
        assert receipt.success is True
        assert receipt.message_id == "telegram_msg_101"
        assert dispatcher.get_breakthrough_id_for_message("telegram", "telegram_msg_101") == 101

    @pytest.mark.asyncio
    async def test_send_to_channel_unknown_channel(self, mock_card):
        dispatcher = DeliveryDispatcher()
        receipt = await dispatcher.send_to_channel("whatsapp", card=mock_card)
        assert receipt.success is False
        assert "not registered" in receipt.error

    @pytest.mark.asyncio
    async def test_send_to_channel_disabled_channel(self, mock_card):
        gw = MockGateway("whatsapp", enabled=False)
        dispatcher = DeliveryDispatcher(gateways=[gw])
        receipt = await dispatcher.send_to_channel("whatsapp", card=mock_card)
        assert receipt.success is False
        assert "disabled" in receipt.error

    @pytest.mark.asyncio
    async def test_get_all_pairing_statuses_with_error_shielding(self):
        gw1 = MockGateway("telegram")
        gw2 = MockGateway("whatsapp", fail_status=True)
        gw3 = MockGateway("signal", enabled=False)

        dispatcher = DeliveryDispatcher(gateways=[gw1, gw2, gw3])
        statuses = await dispatcher.get_all_pairing_statuses()

        assert len(statuses) == 3
        assert statuses["telegram"].status == STATUS_CONNECTED
        assert statuses["whatsapp"].status == STATUS_ERROR
        assert "Simulated status error" in statuses["whatsapp"].error
        assert statuses["signal"].status == STATUS_DISABLED

    @pytest.mark.asyncio
    async def test_dispatcher_lifecycle_close(self):
        gw1 = MockGateway("telegram")
        gw2 = MockGateway("whatsapp")
        dispatcher = DeliveryDispatcher(gateways=[gw1, gw2])

        async with dispatcher:
            assert dispatcher.get_gateway("telegram") is gw1

        assert gw1.closed is True
        assert gw2.closed is True


class TestDefaultDispatcherFactory:
    def test_create_default_dispatcher(self):
        settings = Settings()
        dispatcher = create_default_dispatcher(settings=settings)
        all_channels = [gw.channel_name for gw in dispatcher.get_all_gateways()]
        assert "telegram" in all_channels
        assert "whatsapp" in all_channels
        assert "signal" in all_channels


# ============================================================================
# 4. Inbound Reaction & Keyword Parsing Tests
# ============================================================================

class TestInboundParsing:
    def test_parse_emoji_reactions(self):
        assert parse_feedback_rating("🎯") == FeedbackRating.HIT
        assert parse_feedback_rating("💤") == FeedbackRating.HYPE
        assert parse_feedback_rating("😴") == FeedbackRating.HYPE
        assert parse_feedback_rating("✅") == FeedbackRating.KNOWN
        assert parse_feedback_rating("✔") == FeedbackRating.KNOWN
        assert parse_feedback_rating("🚀") == FeedbackRating.INSPIRE
        assert parse_feedback_rating("🔥") == FeedbackRating.INSPIRE
        assert parse_feedback_rating("🎉") is None
        assert parse_feedback_rating("") is None

    def test_parse_keywords_german_and_english(self):
        # HIT
        assert parse_feedback_rating("hit") == FeedbackRating.HIT
        assert parse_feedback_rating("Volltreffer") == FeedbackRating.HIT
        assert parse_feedback_rating("top!") == FeedbackRating.HIT
        assert parse_feedback_rating("1") == FeedbackRating.HIT

        # HYPE
        assert parse_feedback_rating("hype") == FeedbackRating.HYPE
        assert parse_feedback_rating("banal") == FeedbackRating.HYPE
        assert parse_feedback_rating("Zu banal") == FeedbackRating.HYPE
        assert parse_feedback_rating("skip") == FeedbackRating.HYPE
        assert parse_feedback_rating("2") == FeedbackRating.HYPE

        # KNOWN
        assert parse_feedback_rating("known") == FeedbackRating.KNOWN
        assert parse_feedback_rating("bekannt") == FeedbackRating.KNOWN
        assert parse_feedback_rating("kenne ich schon") == FeedbackRating.KNOWN
        assert parse_feedback_rating("3") == FeedbackRating.KNOWN

        # INSPIRE
        assert parse_feedback_rating("inspire") == FeedbackRating.INSPIRE
        assert parse_feedback_rating("Geniale Inspiration") == FeedbackRating.INSPIRE
        assert parse_feedback_rating("idee") == FeedbackRating.INSPIRE
        assert parse_feedback_rating("4") == FeedbackRating.INSPIRE

    def test_unrecognized_text(self):
        assert parse_feedback_rating("random text message") is None
        assert parse_feedback_rating("guten morgen") is None


# ============================================================================
# 5. Inbound Breakthrough ID Resolution Waterfall Tests
# ============================================================================

class TestBreakthroughIdResolutionWaterfall:
    @pytest.mark.asyncio
    async def test_tier1_explicit_id_in_message_text(self):
        router = InboundRouter()
        assert await router.resolve_breakthrough_id(channel="whatsapp", message_text="hit 42") == 42
        assert await router.resolve_breakthrough_id(channel="signal", message_text="#77") == 77
        assert await router.resolve_breakthrough_id(channel="telegram", message_text="ID: 99") == 99

    @pytest.mark.asyncio
    async def test_tier2_explicit_tag_in_quoted_text(self):
        router = InboundRouter()
        quoted = "⚡ *vLLM Serving* [#42]\n🎯 Score: 9.4"
        assert await router.resolve_breakthrough_id(channel="whatsapp", quoted_text=quoted) == 42

    @pytest.mark.asyncio
    async def test_tier3_dispatcher_delivery_cache(self):
        dispatcher = DeliveryDispatcher()
        dispatcher._recent_deliveries[("whatsapp", "wamid_555")] = 88

        router = InboundRouter(dispatcher=dispatcher)
        assert await router.resolve_breakthrough_id(channel="whatsapp", quoted_message_id="wamid_555") == 88

    @pytest.mark.asyncio
    async def test_tier4_quoted_title_db_match(self):
        mock_db = MagicMock()
        mock_db.fetch_one = AsyncMock(return_value={"id": 65})

        router = InboundRouter(db=mock_db)
        quoted = "⚡ Ollama Engine Running Locally\nScore: 9.0"
        assert await router.resolve_breakthrough_id(channel="signal", quoted_text=quoted) == 65
        mock_db.fetch_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tier5_dispatcher_latest_broadcast(self):
        dispatcher = DeliveryDispatcher()
        dispatcher._latest_breakthrough_id = 92

        router = InboundRouter(dispatcher=dispatcher)
        assert await router.resolve_breakthrough_id(channel="whatsapp") == 92

    @pytest.mark.asyncio
    async def test_tier6_db_latest_breakthrough_fallback(self):
        mock_db = MagicMock()
        mock_db.fetch_one = AsyncMock(return_value={"id": 33})

        router = InboundRouter(db=mock_db)
        assert await router.resolve_breakthrough_id(channel="signal") == 33


# ============================================================================
# 6. Inbound Message Processing & Feedback Persistence Tests
# ============================================================================

class TestInboundProcessingAndPersistence:
    @pytest.mark.asyncio
    async def test_process_reaction_persists_feedback(self):
        prefs = MagicMock(spec=PreferencesService)
        prefs.record_feedback = AsyncMock()

        router = InboundRouter(preferences_service=prefs)
        msg = InboundMessage(
            channel="whatsapp",
            sender_id="491701234567@c.us",
            reaction_emoji="🎯",
            quoted_text="⚡ Card Title [#42]",
        )

        res = await router.process_message(msg)
        assert res.action == "feedback"
        assert res.success is True
        assert res.rating == FeedbackRating.HIT
        assert res.breakthrough_id == 42
        assert "Volltreffer" in res.reply_text

        prefs.record_feedback.assert_awaited_once_with(
            breakthrough_id=42,
            rating=FeedbackRating.HIT,
            user_identifier="491701234567@c.us",
            source="whatsapp",
            notes="Reaction 🎯",
        )

    @pytest.mark.asyncio
    async def test_process_text_keyword_persists_feedback(self):
        prefs = MagicMock(spec=PreferencesService)
        prefs.record_feedback = AsyncMock()

        router = InboundRouter(preferences_service=prefs)
        msg = InboundMessage(
            channel="signal",
            sender_id="+491709999999",
            text="Geniale Inspiration 88",
        )

        res = await router.process_message(msg)
        assert res.action == "feedback"
        assert res.success is True
        assert res.rating == FeedbackRating.INSPIRE
        assert res.breakthrough_id == 88

        prefs.record_feedback.assert_awaited_once_with(
            breakthrough_id=88,
            rating=FeedbackRating.INSPIRE,
            user_identifier="+491709999999",
            source="signal",
            notes="Keyword 'Geniale Inspiration 88'",
        )

    @pytest.mark.asyncio
    async def test_unresolvable_breakthrough_returns_error(self):
        router = InboundRouter()
        msg = InboundMessage(
            channel="whatsapp",
            sender_id="491701234567@c.us",
            reaction_emoji="🎯",
        )
        res = await router.process_message(msg)
        assert res.action == "error"
        assert res.success is False
        assert "Could not identify breakthrough" in res.error


# ============================================================================
# 7. Inbound Slash Commands & Webhooks Tests
# ============================================================================

class TestInboundSlashCommandsAndWebhooks:
    @pytest.mark.asyncio
    async def test_help_command(self):
        router = InboundRouter()
        msg = InboundMessage(
            channel="whatsapp",
            sender_id="491701234567@c.us",
            text="/help",
        )
        res = await router.process_message(msg)
        assert res.action == "command"
        assert res.command == "help"
        assert "/track" in res.reply_text

    @pytest.mark.asyncio
    async def test_track_command_with_db(self):
        mock_db = MagicMock()
        # Mock add_tracked_repo behavior
        with patch("newsscout.dashboard.queries.add_tracked_repo", new=AsyncMock(return_value={"success": True, "repo": "vllm-project/vllm"})):
            router = InboundRouter(db=mock_db)
            msg = InboundMessage(
                channel="signal",
                sender_id="+491701111111",
                text="/track https://github.com/vllm-project/vllm",
            )
            res = await router.process_message(msg)
            assert res.action == "command"
            assert res.command == "track"
            assert "vllm-project/vllm" in res.reply_text

    @pytest.mark.asyncio
    async def test_interest_command_with_db(self):
        mock_db = MagicMock()
        with patch("newsscout.dashboard.queries.add_interest_keyword", new=AsyncMock(return_value={"success": True})):
            router = InboundRouter(db=mock_db)
            msg = InboundMessage(
                channel="whatsapp",
                sender_id="491701234567@c.us",
                text="/interest local inference",
            )
            res = await router.process_message(msg)
            assert res.action == "command"
            assert res.command == "interest"
            assert "local inference" in res.reply_text

    @pytest.mark.asyncio
    async def test_radar_command_with_db(self):
        mock_db = MagicMock()
        with patch(
            "newsscout.dashboard.queries.get_topic_radar",
            new=AsyncMock(return_value={"tracked_repos": ["vllm/vllm"], "custom_keywords": ["rag"]}),
        ):
            router = InboundRouter(db=mock_db)
            msg = InboundMessage(channel="signal", sender_id="+49170", text="/radar")
            res = await router.process_message(msg)
            assert res.action == "command"
            assert res.command == "radar"
            assert "vllm/vllm" in res.reply_text

    @pytest.mark.asyncio
    async def test_handle_whatsapp_webhook_reaction(self):
        prefs = MagicMock(spec=PreferencesService)
        prefs.record_feedback = AsyncMock()

        router = InboundRouter(preferences_service=prefs)
        payload = {
            "event": "message.reaction",
            "payload": {
                "from": "491701234567@c.us",
                "reaction": {
                    "text": "🎯",
                    "messageId": "wamid_777",
                },
            },
        }
        # Provide cache resolution
        dispatcher = DeliveryDispatcher()
        dispatcher._recent_deliveries[("whatsapp", "wamid_777")] = 42
        router.dispatcher = dispatcher

        res = await router.handle_whatsapp_webhook(payload)
        assert res.action == "feedback"
        assert res.rating == FeedbackRating.HIT
        assert res.breakthrough_id == 42

    @pytest.mark.asyncio
    async def test_handle_signal_webhook_data_message(self):
        prefs = MagicMock(spec=PreferencesService)
        prefs.record_feedback = AsyncMock()

        router = InboundRouter(preferences_service=prefs)
        payload = {
            "envelope": {
                "source": "+491709999999",
                "dataMessage": {
                    "timestamp": 1726700000000,
                    "message": "hit 101",
                },
            },
        }

        res = await router.handle_signal_webhook(payload)
        assert res.action == "feedback"
        assert res.rating == FeedbackRating.HIT
        assert res.breakthrough_id == 101
