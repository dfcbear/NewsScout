"""tests/test_adversarial_m3.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial challenge test suite for Milestone 3 (InboundRouter & Messenger Gateways).
Stress-tests malformed payloads, emoji variations, ambiguous keywords,
breakthrough resolution waterfall edge cases, and slash command parsing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newsscout.config import Settings
from newsscout.delivery.base import (
    BaseMessengerGateway,
    DeliveryReceipt,
    PairingQRResult,
)
from newsscout.delivery.dispatcher import DeliveryDispatcher
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
# Helpers & Mocks
# ============================================================================

def make_sample_card(breakthrough_id: int = 42, title: str = "vLLM High-Speed Engine") -> DecisionCard:
    return DecisionCard(
        breakthrough_id=breakthrough_id,
        title=title,
        repo_url="https://github.com/vllm-project/vllm",
        category=Stage2Category.CORE,
        breakthrough_score=9.2,
        roi_score=8.8,
        data=DecisionCardData(
            tldr="PagedAttention allows 4x higher throughput on RTX 4090.",
            use_case="Local model inference server with OpenAI API compatibility.",
            comparison="4x throughput compared to vanilla HuggingFace pipeline.",
            quickstart="docker run -p 8000:8000 vllm/vllm-openai:latest",
            hardware_requirements="RTX 4090 (24GB VRAM)",
            license="Apache 2.0",
        ),
    )


class MockDatabase:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = [
            {"id": 42, "title": "vLLM High-Speed Engine", "category": "core", "evaluated_at": "2026-09-19T08:00:00Z"},
            {"id": 99, "title": "Docling Document Converter", "category": "core", "evaluated_at": "2026-09-19T09:00:00Z"},
            {"id": 105, "title": "Discarded SaaS Wrapper", "category": "discard", "evaluated_at": "2026-09-19T09:30:00Z"},
        ]
        self.sources: list[dict[str, Any]] = [
            {
                "id": 1,
                "name": "github_releases",
                "source_type": "api",
                "config_json": json.dumps({"repos": ["vllm-project/vllm"]}),
                "is_active": 1,
                "last_poll_at": None,
            },
            {
                "id": 2,
                "name": "hacker_news",
                "source_type": "firebase",
                "config_json": json.dumps({"custom_keywords": ["rag", "agent"]}),
                "is_active": 1,
                "last_poll_at": None,
            },
        ]
        self.executed_queries: list[tuple[str, tuple]] = []

    async def fetch_one(self, query: str, params: tuple = ()) -> Optional[dict[str, Any]]:
        self.executed_queries.append((query, params))
        if "FROM sources WHERE name = 'github_releases'" in query:
            return self.sources[0]
        if "FROM sources WHERE name = 'hacker_news'" in query:
            return self.sources[1]
        if "FROM breakthroughs WHERE title LIKE" in query:
            pattern = params[0].strip("%").lower()
            for r in self.rows:
                if pattern in r["title"].lower():
                    return r
            return None
        if "category != 'discard'" in query:
            valid = [r for r in self.rows if r["category"] != "discard"]
            return valid[-1] if valid else None
        return None

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict[str, Any]]:
        self.executed_queries.append((query, params))
        if "FROM sources" in query:
            return self.sources
        return self.rows

    async def execute(self, query: str, params: tuple = ()) -> int:
        self.executed_queries.append((query, params))
        return 1


# ============================================================================
# Test Suite 1: Malformed Webhook Payloads
# ============================================================================

class TestMalformedWebhookPayloads:
    """Stress-tests InboundRouter with malformed, missing, or type-violated payloads."""

    @pytest.fixture
    def router(self) -> InboundRouter:
        return InboundRouter()

    @pytest.mark.asyncio
    async def test_waha_empty_payload(self, router: InboundRouter):
        """Empty payload dictionary must return ignored or safe error without crash."""
        res = await router.handle_whatsapp_webhook({})
        assert res.action == "ignored"
        assert res.success is True

    @pytest.mark.asyncio
    async def test_waha_payload_none_value(self, router: InboundRouter):
        """payload key is explicitly None."""
        payload = {"event": "message", "payload": None}
        res = await router.handle_whatsapp_webhook(payload)
        assert res.action == "empty"
        assert res.success is True

    @pytest.mark.asyncio
    async def test_waha_reaction_payload_none(self, router: InboundRouter):
        """event is reaction but payload is None."""
        payload = {"event": "message.reaction", "payload": None}
        res = await router.handle_whatsapp_webhook(payload)
        assert res.action in ("empty", "unknown", "ignored")

    @pytest.mark.asyncio
    async def test_waha_reaction_nested_none_or_missing(self, router: InboundRouter):
        """reaction object has missing text or messageId."""
        payload = {
            "event": "message.reaction",
            "payload": {
                "from": "491701234567@c.us",
                "reaction": {"text": None, "messageId": None},
            },
        }
        res = await router.handle_whatsapp_webhook(payload)
        assert res.action in ("empty", "unknown", "ignored")

    @pytest.mark.asyncio
    async def test_waha_unexpected_payload_types(self, router: InboundRouter):
        """payload field is a string or list instead of dict."""
        # Non-dict payload
        payload = {"event": "message", "payload": "this is a raw string, not a dict"}
        try:
            res = await router.handle_whatsapp_webhook(payload)
            # If handled gracefully
            assert isinstance(res, InboundHandlingResult)
        except AttributeError as err:
            # Empirical verification: does it raise AttributeError?
            pytest.fail(f"VULNERABILITY: InboundRouter crashed on non-dict payload: {err}")

    @pytest.mark.asyncio
    async def test_waha_unexpected_data_types(self, router: InboundRouter):
        """payload._data is a list or string instead of dict."""
        payload = {
            "event": "message",
            "payload": {
                "body": "hit",
                "from": "491701234567@c.us",
                "_data": "not a dictionary",
            },
        }
        try:
            res = await router.handle_whatsapp_webhook(payload)
            assert isinstance(res, InboundHandlingResult)
        except AttributeError as err:
            pytest.fail(f"VULNERABILITY: InboundRouter crashed on non-dict _data: {err}")

    @pytest.mark.asyncio
    async def test_signal_empty_payload(self, router: InboundRouter):
        """Empty Signal webhook payload returns ignored cleanly."""
        res = await router.handle_signal_webhook({})
        assert res.action == "ignored"
        assert res.success is True

    @pytest.mark.asyncio
    async def test_signal_envelope_none(self, router: InboundRouter):
        """Signal envelope is explicitly None."""
        res = await router.handle_signal_webhook({"envelope": None})
        assert res.action == "ignored"

    @pytest.mark.asyncio
    async def test_signal_data_message_none(self, router: InboundRouter):
        """Signal dataMessage is explicitly None."""
        res = await router.handle_signal_webhook({"envelope": {"dataMessage": None}})
        assert res.action == "ignored"

    @pytest.mark.asyncio
    async def test_signal_unexpected_envelope_type(self, router: InboundRouter):
        """Signal envelope is a string or list."""
        payload = {"envelope": "invalid string"}
        try:
            res = await router.handle_signal_webhook(payload)
            assert isinstance(res, InboundHandlingResult)
        except AttributeError as err:
            pytest.fail(f"VULNERABILITY: InboundRouter crashed on non-dict Signal envelope: {err}")

    @pytest.mark.asyncio
    async def test_signal_unexpected_quote_type(self, router: InboundRouter):
        """quote field in dataMessage is a string instead of dict."""
        payload = {
            "envelope": {
                "source": "+491701234567",
                "dataMessage": {
                    "message": "hit",
                    "quote": "not a dict",
                },
            }
        }
        try:
            res = await router.handle_signal_webhook(payload)
            assert isinstance(res, InboundHandlingResult)
        except AttributeError as err:
            pytest.fail(f"VULNERABILITY: InboundRouter crashed on non-dict Signal quote: {err}")


# ============================================================================
# Test Suite 2: Emoji Variations & Normalization
# ============================================================================

class TestEmojiVariations:
    """Stress-tests emoji parsing with variation selectors, skin tones, and unknown emojis."""

    @pytest.mark.parametrize(
        "emoji_str,expected_rating",
        [
            ("🎯", FeedbackRating.HIT),
            ("💤", FeedbackRating.HYPE),
            ("😴", FeedbackRating.HYPE),
            ("✅", FeedbackRating.KNOWN),
            ("✔", FeedbackRating.KNOWN),
            ("🚀", FeedbackRating.INSPIRE),
            ("🔥", FeedbackRating.INSPIRE),
        ],
    )
    def test_standard_emojis(self, emoji_str: str, expected_rating: FeedbackRating):
        assert parse_feedback_rating(emoji_str) == expected_rating

    def test_emoji_variation_selector_16(self):
        """Test emojis sent with Unicode Variation Selector-16 (U+FE0F)."""
        # Standard heavy checkmark vs heavy checkmark with VS-16
        check_vs16 = "✔️"  # '\u2714\ufe0f'
        rating_check = parse_feedback_rating(check_vs16)
        # Target emoji with VS-16
        target_vs16 = "🎯\ufe0f"
        rating_target = parse_feedback_rating(target_vs16)
        # Rocket with VS-16
        rocket_vs16 = "🚀\ufe0f"
        rating_rocket = parse_feedback_rating(rocket_vs16)

        # Record empirical result
        print(f"\n[EMOJI-VS16] ✔️ -> {rating_check}, 🎯\\ufe0f -> {rating_target}, 🚀\\ufe0f -> {rating_rocket}")
        # Does the system accept ✔️ ?
        assert rating_check == FeedbackRating.KNOWN, f"Failed to parse ✔️ with variation selector: got {rating_check}"

    def test_unknown_emojis(self):
        """Unknown emojis like 👍, 👎, ❤️, 🎉 should return None."""
        assert parse_feedback_rating("👍") is None
        assert parse_feedback_rating("👎") is None
        assert parse_feedback_rating("❤️") is None
        assert parse_feedback_rating("🎉") is None
        assert parse_feedback_rating("💩") is None

    def test_emoji_with_whitespace(self):
        """Emojis with leading/trailing spaces, newlines, tabs."""
        assert parse_feedback_rating("  🎯  ") == FeedbackRating.HIT
        assert parse_feedback_rating("\n💤\n") == FeedbackRating.HYPE
        assert parse_feedback_rating("\t✅\t") == FeedbackRating.KNOWN

    def test_emoji_combined_with_card_id(self):
        """User texts emoji followed by card ID, e.g. '🎯 42' or '🚀 #99'."""
        r1 = parse_feedback_rating("🎯 42")
        r2 = parse_feedback_rating("🚀 #99")
        r3 = parse_feedback_rating("✅ 12")
        print(f"\n[EMOJI-WITH-ID] '🎯 42' -> {r1}, '🚀 #99' -> {r2}, '✅ 12' -> {r3}")
        # Note empirical behavior: does parse_feedback_rating preserve emoji intent when followed by ID?


# ============================================================================
# Test Suite 3: Ambiguous and Mixed Keyword Texts
# ============================================================================

class TestAmbiguousAndMixedKeywords:
    """Stress-tests keyword matching against contradictions, negations, and numbers in prose."""

    def test_opposing_sentiments_hit_vs_hype(self):
        """Contradictory sentence: 'Das ist ein Hit und kein Hype' vs 'Das ist ein Hype und kein Hit'."""
        r1 = parse_feedback_rating("Das ist ein Hit und kein Hype")
        r2 = parse_feedback_rating("Das ist ein Hype und kein Hit")
        print(f"\n[OPPOSING] 'Hit und kein Hype' -> {r1}, 'Hype und kein Hit' -> {r2}")
        # Both may evaluate to HIT due to word order and priority ordering in parse_feedback_rating

    def test_negations(self):
        """Negation phrases: 'nicht bekannt', 'kein hit', 'nicht genial'."""
        r_nicht_bekannt = parse_feedback_rating("nicht bekannt")
        r_kein_hit = parse_feedback_rating("kein hit")
        r_nicht_genial = parse_feedback_rating("nicht genial")
        r_kenne_ich_nicht = parse_feedback_rating("kenne ich nicht")

        print(f"\n[NEGATIONS] 'nicht bekannt' -> {r_nicht_bekannt}, 'kein hit' -> {r_kein_hit}")
        print(f"            'nicht genial' -> {r_nicht_genial}, 'kenne ich nicht' -> {r_kenne_ich_nicht}")
        # Observe whether 'nicht bekannt' is falsely classified as KNOWN!

    def test_numbers_in_sentences(self):
        """Prose containing digits 1, 2, 3, 4 should not be confused with feedback options."""
        r_first = parse_feedback_rating("Hier ist mein 1. Feedback")
        r_days = parse_feedback_rating("Ich habe das 2 Tage getestet")
        r_chapter = parse_feedback_rating("Kapitel 3 ist super")
        r_version = parse_feedback_rating("Version 4 gefällt mir")

        print(f"\n[DIGITS-IN-PROSE] 'mein 1. Feedback' -> {r_first}")
        print(f"                 '2 Tage getestet' -> {r_days}")
        print(f"                 'Kapitel 3' -> {r_chapter}")
        print(f"                 'Version 4' -> {r_version}")

    def test_multi_sentence_replies(self):
        """Realistic multi-sentence replies from messenger users."""
        text = "Ich habe vLLM gestern ausprobiert. Setup war ein echter Hit, aber leider schon bekannt."
        rating = parse_feedback_rating(text)
        print(f"\n[MULTI-SENTENCE] '{text}' -> {rating}")


# ============================================================================
# Test Suite 4: Breakthrough Resolution Waterfall
# ============================================================================

class TestBreakthroughWaterfallAdversarial:
    """Stress-tests the 6-tier breakthrough resolution waterfall."""

    @pytest.mark.asyncio
    async def test_numeric_reply_overrides_quoted_card(self):
        """CRITICAL: User replies '1' (meaning option 1: Hit) while quoting card [#99].
        Does Tier 1 hijack card ID to 1, or does Tier 2 resolve to quoted card 99?"""
        router = InboundRouter()
        resolved_id = await router.resolve_breakthrough_id(
            channel="whatsapp",
            quoted_text="⚡ Docling Document Converter [#99]\nScore: 9.2/10",
            message_text="1",  # User replies '1' as rating choice
        )
        print(f"\n[WATERFALL-NUMERIC] User replied '1' on card [#99] -> resolved_id: {resolved_id}")
        # If resolved_id == 1, Tier 1 hijacked the card ID because \b(\d+)\b matched '1'!

    @pytest.mark.asyncio
    async def test_prose_with_number_overrides_quoted_card(self):
        """User replies 'Läuft super auf RTX 4090' while quoting card [#99].
        Does it resolve to 4090 or 99?"""
        router = InboundRouter()
        resolved_id = await router.resolve_breakthrough_id(
            channel="whatsapp",
            quoted_text="⚡ Docling Document Converter [#99]\nScore: 9.2/10",
            message_text="Läuft super auf RTX 4090",
        )
        print(f"\n[WATERFALL-PROSE-NUM] User replied 'Läuft super auf RTX 4090' on [#99] -> resolved_id: {resolved_id}")

    @pytest.mark.asyncio
    async def test_quoted_text_with_title_number(self):
        """Quoted card has '#1 Tool in AI' in title and [#42] at end.
        Does m_tag match #1 or [#42]?"""
        router = InboundRouter()
        card_text = "⚡ #1 AI Agent Framework for Developers [#42]\nScore: 9.5/10"
        resolved_id = await router.resolve_breakthrough_id(
            channel="whatsapp",
            quoted_text=card_text,
            message_text="hit",
        )
        print(f"\n[WATERFALL-QUOTED-HASH1] Quoted '#1 ... [#42]' -> resolved_id: {resolved_id}")

    @pytest.mark.asyncio
    async def test_cache_hit_vs_cache_miss_waterfall(self):
        """Verify Tier 3 cache hit, then cache miss falling back to Tier 5 latest broadcast."""
        dispatcher = DeliveryDispatcher()
        dispatcher._recent_deliveries[("signal", "msg-123")] = 42
        dispatcher._latest_breakthrough_id = 99

        router = InboundRouter(dispatcher=dispatcher)

        # 1. Cache hit
        id_hit = await router.resolve_breakthrough_id(
            channel="signal",
            quoted_message_id="msg-123",
            message_text="hit",
        )
        assert id_hit == 42

        # 2. Cache miss falls back to latest broadcast (Tier 5)
        id_miss = await router.resolve_breakthrough_id(
            channel="signal",
            quoted_message_id="unknown-msg-999",
            message_text="hit",
        )
        assert id_miss == 99

    @pytest.mark.asyncio
    async def test_cache_miss_with_db_fallback(self):
        """When dispatcher has no cache and no broadcast, falls back to latest non-discard DB card (Tier 6)."""
        mock_db = MockDatabase()
        router = InboundRouter(db=mock_db)

        resolved = await router.resolve_breakthrough_id(
            channel="whatsapp",
            message_text="hit",
        )
        # In MockDatabase, latest non-discard is id 99 (id 105 is discard)
        assert resolved == 99


# ============================================================================
# Test Suite 5: Slash Command Parsing
# ============================================================================

class TestSlashCommandParsingAdversarial:
    """Stress-tests slash commands with empty args, whitespace, URLs, and special characters."""

    @pytest.fixture
    def router(self) -> InboundRouter:
        mock_db = MockDatabase()
        return InboundRouter(db=mock_db)

    @pytest.mark.asyncio
    async def test_track_empty_arguments(self, router: InboundRouter):
        """User sends just '/track' or '/track    ' without repo."""
        res1 = await router._handle_slash_command("whatsapp", "/track", "user1")
        res2 = await router._handle_slash_command("signal", "/track    ", "user1")

        print(f"\n[SLASH-TRACK-EMPTY] '/track' -> action: {res1.action}, success: {res1.success}")
        # Must not crash, should return helpful guidance or command error
        assert res1.success is False or res1.action == "unknown"

    @pytest.mark.asyncio
    async def test_interest_empty_arguments(self, router: InboundRouter):
        """User sends just '/interest' or '/interest   ' without keyword."""
        res = await router._handle_slash_command("whatsapp", "/interest", "user1")
        print(f"\n[SLASH-INTEREST-EMPTY] '/interest' -> action: {res.action}, success: {res.success}")
        assert res.success is False or res.action == "unknown"

    @pytest.mark.asyncio
    async def test_track_with_full_github_urls(self, router: InboundRouter):
        """User sends '/track https://github.com/foo/bar.git' or other variants."""
        res_https = await router._handle_slash_command("whatsapp", "/track https://github.com/foo/bar.git", "user1")
        assert res_https.success is True
        assert "foo/bar" in res_https.reply_text

        # What about http://?
        res_http = await router._handle_slash_command("whatsapp", "/track http://github.com/foo/bar.git", "user1")
        print(f"\n[SLASH-TRACK-HTTP] 'http://github.com/foo/bar.git' -> {res_http.reply_text}")

    @pytest.mark.asyncio
    async def test_interest_with_special_characters(self, router: InboundRouter):
        """User sends '/interest LLM & RAG' or '/interest C++ / C#'."""
        res1 = await router._handle_slash_command("whatsapp", "/interest LLM & RAG", "user1")
        assert res1.success is True
        assert "llm & rag" in res1.reply_text.lower()

        res2 = await router._handle_slash_command("signal", "/interest C++ & C#", "user1")
        assert res2.success is True
        assert "c++ & c#" in res2.reply_text.lower()

    @pytest.mark.asyncio
    async def test_command_case_insensitivity(self, router: InboundRouter):
        """Slash commands in UPPERCASE or MixedCase: /TRACK, /INTEREST, /RADAR, /HELP."""
        res_track = await router._handle_slash_command("whatsapp", "/TRACK foo/bar", "user1")
        assert res_track.success is True
        assert res_track.command == "track"

        res_radar = await router._handle_slash_command("whatsapp", "/RADAR", "user1")
        assert res_radar.success is True
        assert res_radar.command == "radar"

        res_help = await router._handle_slash_command("whatsapp", "/HELP", "user1")
        assert res_help.success is True
        assert res_help.command == "help"

    @pytest.mark.asyncio
    async def test_extra_whitespace_in_commands(self, router: InboundRouter):
        """Leading/trailing and internal extra whitespace in slash commands."""
        res = await router._handle_slash_command("whatsapp", "   /track    owner/repo    ", "user1")
        assert res.success is True
        assert "owner/repo" in res.reply_text
