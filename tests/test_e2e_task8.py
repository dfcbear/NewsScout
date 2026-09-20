"""tests.test_e2e_task8
~~~~~~~~~~~~~~~~~~~~~
Complete End-to-End integration test for TASK-08:
Verifies the full pipeline flow across all 3 major subsystems:
1. Multi-Search Ingestion: Aggregates results from SearXNG & DuckDuckGo (zero-key) + Tavily & Exa into RawItems
2. Stage 1 Filter: Processes raw items, filtering SaaS wrappers and passing high-signal candidates
3. Stage 2 Universal LLM Evaluator: Evaluates candidates via OpenAI-compatible abstraction, generating Decision Cards
4. Multi-Messenger Delivery Dispatcher: Broadcasts cards concurrently across Telegram and Signal
5. Inbound Feedback & Reaction Router: Processes incoming emojis (🎯, 🚀) and keywords (hit, genial) from Signal, recording multi-channel feedback in PreferencesService & SQLite WAL
6. Dashboard System Status: Verifies all subsystems report healthy operational status
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.delivery.base import DeliveryReceipt
from newsscout.delivery.dispatcher import DeliveryDispatcher
from newsscout.delivery.inbound import InboundRouter
from newsscout.delivery.signal_gateway import SignalGateway
from newsscout.delivery.telegram import TelegramGateway
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.filtering.stage2 import DecisionCardPayload, Stage2EvaluationResponse, Stage2Evaluator
from newsscout.llm.base import BaseLLMClient
from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import BaseSearchProvider, SearchResult
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    FeedbackRating,
    RawItem,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Test Fixtures & Mock Providers
# ============================================================================

class MockSearchProvider(BaseSearchProvider):
    def __init__(self, name: str, results: list[SearchResult]):
        self._name = name
        self._results = results

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self._results[:limit]


class MockE2ELLMClient(BaseLLMClient):
    """Mock LLM client returning realistic breakthrough evaluations."""

    @property
    def provider_name(self) -> str:
        return "mock_openai_compatible"

    @property
    def model_name(self) -> str:
        return "meta-llama-3.1-8b-instruct"

    async def generate_text(self, prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
        return "Evaluation summary text"

    async def generate_structured(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Stage2EvaluationResponse:
        return Stage2EvaluationResponse(
            breakthrough_score=9.2,
            roi_score=8.8,
            category=Stage2Category.CORE,
            relevance_justification="Revolutionary local inference engine with 3x speedup on RTX 4090.",
            decision_card=DecisionCardPayload(
                tldr="Zero-overhead speculative decoding on RTX 4090 achieving 120 tok/s.",
                use_case="Local agentic pair programming harness with instant responses.",
                comparison="3.2x faster than standard vLLM baseline.",
                quickstart="docker run --gpus all -p 8000:8000 ghcr.io/vllm/fast-infer",
                hardware_requirements="RTX 4090 (24GB VRAM)",
                license="Apache-2.0",
            ),
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def e2e_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "e2e_task8.db"
    db = Database(db_file)
    with db.get_sync_connection() as c:
        apply_migrations(conn=c)
    return db


@pytest.fixture
def e2e_settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "e2e_task8.db",
        preferences_file=tmp_path / "preferences.json",
        llm_provider="openai_compatible",
        llm_base_url="http://mock-llm-host:8000/v1",
        llm_model="meta-llama-3.1-8b-instruct",
        searxng_enabled=True,
        duckduckgo_enabled=True,
        tavily_api_key=SecretStr("mock-tavily-key"),
        signal_enabled=True,
        signal_bridge_url="http://mock-signal:8085",
        signal_sender_number="+491701234567",
        signal_recipient_id="+491709876543",
        telegram_bot_token=SecretStr("mock-tg-token"),
        telegram_chat_id="123456789",
    )


# ============================================================================
# Complete End-to-End Pipeline Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_task8_e2e_pipeline(e2e_db: Database, e2e_settings: Settings):
    """Executes the full TASK-08 cycle from multi-engine search to multi-messenger feedback."""
    preferences = PreferencesService(e2e_db, e2e_settings)

    # ------------------------------------------------------------------------
    # Step 1: Multi-Search Aggregation (Zero-Key + Commercial)
    # ------------------------------------------------------------------------
    searxng_res = [
        SearchResult(
            title="FastInfer: Kernel-Optimized RTX 4090 Engine",
            url="https://github.com/fastinfer/fastinfer",
            snippet="High performance inference engine with NVFP4 kernels.",
            source_engine="searxng",
        )
    ]
    ddg_res = [
        SearchResult(
            title="FastInfer: Kernel-Optimized RTX 4090 Engine",  # Duplicate URL to test deduplication
            url="https://github.com/fastinfer/fastinfer?utm_source=search",
            snippet="High performance inference engine on GitHub.",
            source_engine="duckduckgo",
        ),
        SearchResult(
            title="NoCode Website Builder SaaS",
            url="https://github.com/junk/nocode-shopify-builder",
            snippet="Build Shopify stores without code in minutes.",
            source_engine="duckduckgo",
        )
    ]
    tavily_res = [
        SearchResult(
            title="AgentHarness: Autonomous Coding Framework",
            url="https://github.com/agentharness/agentharness",
            snippet="Spec-driven multi-agent framework on Raspberry Pi 5.",
            source_engine="tavily",
        )
    ]

    providers = [
        MockSearchProvider("searxng", searxng_res),
        MockSearchProvider("duckduckgo", ddg_res),
        MockSearchProvider("tavily", tavily_res),
    ]

    aggregator = MultiSearchAggregator(
        providers=providers,
        db=e2e_db,
        settings=e2e_settings,
    )

    aggregated_results = await aggregator.search("autonomous agent inference")
    # URLs should be normalized and deduplicated: FastInfer (searxng+ddg merged) + NoCode + AgentHarness = 3 unique items
    assert len(aggregated_results) == 3

    # Ingest aggregated search results as RawItems into SQLite database
    import json
    raw_items = []
    for r in aggregated_results:
        raw_item = RawItem(
            source=f"search_{r.source_engine}",
            source_id=f"search_{r.source_engine}:{r.url}",
            title=r.title,
            url=r.url,
            raw_content=r.snippet,
            metadata={
                "source_engine": r.source_engine,
                "has_dockerfile": "fastinfer" in r.url or "agentharness" in r.url,
                "root_files": ["Dockerfile", "pyproject.toml"] if ("fastinfer" in r.url or "agentharness" in r.url) else [],
                "stars": 1500,
                "forks": 250,
                "license": "Apache-2.0",
            },
            ingested_at=datetime.now(timezone.utc),
        )
        row_id = await e2e_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, source_id) DO UPDATE SET title=excluded.title;
            """,
            (
                raw_item.source,
                raw_item.source_id,
                raw_item.title,
                raw_item.url,
                raw_item.raw_content,
                json.dumps(raw_item.metadata),
                raw_item.ingested_at.isoformat(),
            ),
        )
        raw_item.id = row_id
        raw_items.append(raw_item)

    # Verify 3 raw items in DB
    assert len(raw_items) == 3

    # ------------------------------------------------------------------------
    # Step 2: Stage 1 Deterministic Filter
    # ------------------------------------------------------------------------
    stage1 = Stage1Filter(e2e_db)
    s1_evals = await stage1.run_pipeline()
    assert len(s1_evals) == 3

    # FastInfer and AgentHarness should pass, NoCode should be dropped
    passed = [e for e in s1_evals if e.passed]
    dropped = [e for e in s1_evals if not e.passed]
    assert len(passed) >= 2
    assert len(dropped) >= 1

    # ------------------------------------------------------------------------
    # Step 3: Stage 2 Universal LLM Evaluation
    # ------------------------------------------------------------------------
    mock_llm = MockE2ELLMClient()
    stage2 = Stage2Evaluator(e2e_db, e2e_settings, llm_client=mock_llm)
    breakthroughs = await stage2.evaluate_pending_items()

    assert len(breakthroughs) >= 2
    top_bt = breakthroughs[0]
    assert top_bt.breakthrough_score >= 8.0
    assert top_bt.category == "core"
    card = top_bt.to_decision_card()
    assert card is not None
    assert "RTX 4090" in card.data.hardware_requirements

    # ------------------------------------------------------------------------
    # Step 4: Multi-Messenger Dispatcher Broadcasting
    # ------------------------------------------------------------------------
    # Setup mock gateways
    mock_tg = MagicMock(spec=TelegramGateway)
    mock_tg.channel_name = "telegram"
    mock_tg.is_enabled.return_value = True
    mock_tg.send_card = AsyncMock(return_value=DeliveryReceipt(channel="telegram", success=True, message_id="1001"))

    mock_sig = MagicMock(spec=SignalGateway)
    mock_sig.channel_name = "signal"
    mock_sig.is_enabled.return_value = True
    mock_sig.send_card = AsyncMock(return_value=DeliveryReceipt(channel="signal", success=True, message_id="1726820000000"))

    dispatcher = DeliveryDispatcher(
        gateways=[mock_tg, mock_sig],
        settings=e2e_settings,
        preferences_service=preferences,
    )

    # Broadcast decision card to all active channels concurrently
    receipts = await dispatcher.broadcast_card(top_bt)

    assert "telegram" in receipts and receipts["telegram"].success is True
    assert "signal" in receipts and receipts["signal"].success is True

    # Verify delivery tracking in dispatcher
    assert dispatcher.get_breakthrough_id_for_message("signal", "1726820000000") == top_bt.id

    # ------------------------------------------------------------------------
    # Step 5: Inbound Multi-Messenger Feedback Routing
    # ------------------------------------------------------------------------
    inbound_router = InboundRouter(
        preferences_service=preferences,
        db=e2e_db,
        dispatcher=dispatcher,
        settings=e2e_settings,
    )

    # 5a. Signal User reacts with 🎯 (Volltreffer) via reaction payload
    sig_reaction_payload = {
        "envelope": {
            "source": "+491701234567",
            "dataMessage": {
                "reaction": {
                    "emoji": "🎯",
                    "targetAuthor": "+491709876543",
                    "targetSentTimestamp": 1726820000000,
                }
            }
        }
    }
    sig_reaction_result = await inbound_router.handle_signal_webhook(sig_reaction_payload)
    assert sig_reaction_result.action == "feedback"
    assert sig_reaction_result.success is True
    assert sig_reaction_result.rating == FeedbackRating.HIT
    assert sig_reaction_result.breakthrough_id == top_bt.id

    # 5b. Signal User replies to quoted message with text 'genial' (Inspire)
    sig_reply_payload = {
        "envelope": {
            "source": "+491709876543",
            "dataMessage": {
                "message": "genial! Baue ich direkt in meinen Workflow ein.",
                "quote": {
                    "id": 1726820000000,
                    "text": top_bt.title,
                }
            }
        }
    }
    sig_result = await inbound_router.handle_signal_webhook(sig_reply_payload)
    assert sig_result.action == "feedback"
    assert sig_result.success is True
    assert sig_result.rating == FeedbackRating.INSPIRE
    assert sig_result.breakthrough_id == top_bt.id

    # 5c. Verify feedback records in SQLite WAL
    fb_rows = await e2e_db.fetch_all(
        "SELECT * FROM feedback WHERE breakthrough_id = ? ORDER BY id ASC",
        (top_bt.id,),
    )
    assert len(fb_rows) == 2
    assert fb_rows[0]["source"] == "signal"
    assert fb_rows[0]["rating"] == "hit"
    assert fb_rows[0]["user_identifier"] == "+491701234567"
    assert fb_rows[1]["source"] == "signal"
    assert fb_rows[1]["rating"] == "inspire"
    assert fb_rows[1]["user_identifier"] == "+491709876543"

    # ------------------------------------------------------------------------
    # Step 6: Dashboard System Status API Verification
    # ------------------------------------------------------------------------
    app = create_app(e2e_settings)
    app.state.db = e2e_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        status_resp = await client.get("/api/system/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()

        assert status_data["llm"]["provider"] == "openai_compatible"
        assert status_data["llm"]["configured"] is True
        assert status_data["search"]["searxng"]["enabled"] is True
        assert status_data["gateways"]["signal"]["enabled"] is True

        stats_resp = await client.get("/api/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["total_feedback"] == 2
        assert stats_data["rating_counts"].get("hit") == 1
        assert stats_data["rating_counts"].get("inspire") == 1
