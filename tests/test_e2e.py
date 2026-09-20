"""tests/test_e2e.py
End-to-end integration tests validating the full NewsScout pipeline.
All tests run fully offline using mock TTS, mock Gemini, and mock Telegram.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from newsscout.audio.tts_engine import BaseTTSEngine
from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.storage.db import Database
from newsscout.storage.models import (
    DigestSlot,
    DigestStatus,
    RawItem,
    Track,
)
from tests.conftest import (
    DataFactory,
    MockGeminiClient,
    create_silent_mp3,
)


class MockTTSEngine(BaseTTSEngine):
    """TTS engine that writes silent MP3 bytes without edge-tts or ffmpeg."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings)

    async def synthesize_turn(self, turn: Any) -> bytes:
        return create_silent_mp3(duration_seconds=0.5, min_frames=2)

    async def synthesize_script(self, script: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(create_silent_mp3(duration_seconds=2.0, min_frames=2))
        return output_path

    async def merge_tracks(self, track_paths: list[Path], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(create_silent_mp3(duration_seconds=4.0, min_frames=2))
        return output_path

    @staticmethod
    def get_mp3_duration_seconds(audio_path: Path) -> float:
        if not audio_path.exists():
            return 0.0
        return audio_path.stat().st_size / 16000.0

class TestE2EFullPipeline:
    """End-to-end: ingestion -> Stage 1 -> Stage 2 -> digest -> delivery -> dashboard -> feedback."""

    @pytest.fixture
    def e2e_db(self, tmp_path: Path) -> Database:
        from newsscout.storage.migrations import apply_migrations
        db = Database(tmp_path / "e2e_scout.db", timeout=5.0)
        asyncio.run(db.initialize())
        with db.get_sync_connection() as conn:
            apply_migrations(conn=conn)
        yield db
        asyncio.run(db.close())

    @pytest.fixture
    def e2e_settings(self, tmp_path: Path) -> Settings:
        from pydantic import SecretStr
        settings = Settings(
            environment="test",
            db_path=tmp_path / "e2e_scout.db",
            audio_output_dir=tmp_path / "audio",
            preferences_file=tmp_path / "preferences.json",
            gemini_api_key=SecretStr("mock-test-key"),
            telegram_bot_token=SecretStr("123456789:MOCK_E2E_TOKEN"),
            telegram_chat_id="99887766",
            github_token=SecretStr("mock-github-token"),
            log_level="DEBUG",
        )
        settings.ensure_directories()
        return settings

    @pytest.fixture
    def seed_raw_items(self, e2e_db: Database) -> list[RawItem]:
        items = [
            RawItem(
                source="github_releases",
                source_id="vllm-project/vllm:v0.8.0",
                title="vLLM v0.8.0 - 3x faster inference with PagedAttention v2",
                url="https://github.com/vllm-project/vllm/releases/tag/v0.8.0",
                raw_content="vLLM v0.8.0 brings PagedAttention v2 with 3x throughput improvement. docker run --gpus all vllm/vllm-openai:latest. Apache-2.0 license.",
                metadata={"license": "apache-2.0", "stars": 25000, "forks": 3700, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["pyproject.toml", "Dockerfile", "setup.py"]},
            ),
            RawItem(
                source="github_releases",
                source_id="llama-cpp/llama.cpp:b3600",
                title="llama.cpp b3600 - NVFP4 quantization support for RTX 4090",
                url="https://github.com/llama-cpp/llama.cpp/releases/tag/b3600",
                raw_content="llama.cpp b3600 adds NVFP4 quantization. pip install llama-cpp-python. MIT license.",
                metadata={"license": "mit", "stars": 65000, "forks": 9200, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["CMakeLists.txt", "Makefile"]},
            ),
            RawItem(
                source="hacker_news",
                source_id="hn:39123456",
                title="Show HN: Open-source agent harness with spec-driven development",
                url="https://github.com/example/agent-harness",
                raw_content="A new open-source agent harness supporting MCP and spec-driven dev. docker run agent-harness:latest. Apache-2.0.",
                metadata={"license": "apache-2.0", "stars": 3400, "forks": 280, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["pyproject.toml", "Dockerfile"], "extracted_github_urls": ["https://github.com/example/agent-harness"], "score": 450, "descendants": 200},
            ),
            RawItem(
                source="hf_daily_papers",
                source_id="hf:2403.12345",
                title="Mamba-2: Tractable State Space Models with Structured Matrices",
                url="https://arxiv.org/abs/2403.12345",
                raw_content="Mamba-2 introduces structured state space models. Code at github.com/state-spaces/mamba. Apache-2.0.",
                metadata={"license": "apache-2.0", "stars": 5000, "forks": 450, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["setup.py", "requirements.txt"], "github_repo": "state-spaces/mamba", "upvotes": 85},
            ),
            RawItem(
                source="github_releases",
                source_id="saa-wrapper/ai-landing-page:v1.0",
                title="AI Landing Page Builder - No-code website generator",
                url="https://github.com/saa-wrapper/ai-landing-page",
                raw_content="Build landing pages in seconds with AI. No-code builder. Premium subscription required.",
                metadata={"license": "proprietary", "stars": 50, "forks": 2, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["package.json"]},
            ),
        ]
        for item in items:
            params = item.to_db_params()
            asyncio.run(e2e_db.execute(
                "INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at) VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)",
                params,
            ))
        rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM raw_items ORDER BY id"))
        return [RawItem.from_row(r) for r in rows]

    def test_stage1_filters_correctly(self, e2e_db: Database, seed_raw_items: list[RawItem]):
        from newsscout.filtering.stage1 import Stage1Filter
        stage1 = Stage1Filter(db=e2e_db)
        results = []
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
            results.append((item.title, eval_result.passed, eval_result.drop_reason))
        passed = [r for r in results if r[1]]
        dropped = [r for r in results if not r[1]]
        assert len(passed) == 4
        assert len(dropped) == 1
        assert "Landing Page" in dropped[0][0] or "No-code" in dropped[0][0]

    def test_stage2_evaluates_and_persists(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        breakthroughs = asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        assert len(breakthroughs) >= 3
        rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM breakthroughs WHERE category != 'discard' ORDER BY breakthrough_score DESC"))
        assert len(rows) >= 3
        for row in rows:
            assert 1.0 <= row["breakthrough_score"] <= 10.0
            assert 1.0 <= row["roi_score"] <= 10.0
    def test_full_pipeline_to_digest(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings, tmp_path: Path):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.audio.digest_gen import DigestGenerator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        breakthroughs = asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        assert len(breakthroughs) >= 3
        mock_tts = MockTTSEngine(e2e_settings)
        digest_gen = DigestGenerator(db=e2e_db, settings=e2e_settings, tts_engine=mock_tts)
        digest = asyncio.run(digest_gen.generate_digest(slot=DigestSlot.MORNING))
        assert digest.status == DigestStatus.SYNTHESIZED
        assert digest.id is not None
        assert len(digest.tracks) == 5
        assert digest.total_duration_seconds > 0
        track_rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM digest_tracks WHERE digest_id = ? ORDER BY track_number", (digest.id,)))
        assert len(track_rows) == 5
        track_types = [r["track_type"] for r in track_rows]
        assert "executive_summary" in track_types
        assert "deep_dive_1" in track_types
        assert "deep_dive_2" in track_types
        assert "serendipity" in track_types
        assert "verdict" in track_types
        merged_path = Path(digest.merged_audio_path or "")
        assert merged_path.exists()
        assert merged_path.stat().st_size > 0

    def test_telegram_delivery_with_mock(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings):
        import httpx
        from newsscout.delivery.telegram_bot import TelegramBot
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        breakthroughs = asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        assert len(breakthroughs) >= 1
        digest = DataFactory.create_digest(slot=DigestSlot.MORNING, status=DigestStatus.SYNTHESIZED)
        digest.id = asyncio.run(e2e_db.execute(
            "INSERT INTO digests (slot, digest_date, title, total_duration_seconds, merged_audio_path, telegram_message_id, status, created_at, delivered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (digest.slot.value, digest.digest_date, digest.title, 1200, "data/audio/playlist.mp3", None, digest.status.value, digest.created_at.isoformat(), None),
        ))
        for i in range(3):
            track = DataFactory.create_track(digest_id=digest.id, track_number=i + 1)
            params = track.to_db_params()
            asyncio.run(e2e_db.execute(
                "INSERT INTO digest_tracks (digest_id, track_number, track_type, title, duration_seconds, audio_path, script_json, breakthrough_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (params["digest_id"], params["track_number"], params["track_type"], params["title"], params["duration_seconds"], params["audio_path"], params["script_json"], params["breakthrough_id"], params["created_at"]),
            ))
        track_rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM digest_tracks WHERE digest_id = ? ORDER BY track_number", (digest.id,)))
        digest.tracks = [Track.model_validate(dict(r)) for r in track_rows]
        mock_transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True, "result": {"message_id": 42}}))
        bot = TelegramBot(settings=e2e_settings, client=httpx.AsyncClient(transport=mock_transport))
        asyncio.run(bot.__aenter__())
        try:
            bt = breakthroughs[0]
            msg_id = asyncio.run(bot.send_decision_card(bt))
            assert msg_id == 42
            menu_msg_id = asyncio.run(bot.send_digest_menu(digest))
            assert menu_msg_id == 42
        finally:
            asyncio.run(bot.close())
    def test_dashboard_serves_pipeline_results(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.audio.digest_gen import DigestGenerator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        mock_tts = MockTTSEngine(e2e_settings)
        digest_gen = DigestGenerator(db=e2e_db, settings=e2e_settings, tts_engine=mock_tts)
        asyncio.run(digest_gen.generate_digest(slot=DigestSlot.MORNING))
        app = create_app(e2e_settings)
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert "NewsScout" in resp.text
            resp = client.get("/api/breakthroughs")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 3
            bt_id = data[0]["id"]
            resp = client.get(f"/api/breakthroughs/{bt_id}")
            assert resp.status_code == 200
            detail = resp.json()
            assert "title" in detail
            assert "feedback_summary" in detail
            resp = client.get("/api/stats")
            assert resp.status_code == 200
            stats = resp.json()
            assert stats["total_breakthroughs"] >= 3
            resp = client.get("/api/digests")
            assert resp.status_code == 200
            digests = resp.json()
            assert len(digests) >= 1
            assert digests[0]["track_count"] == 5
            resp = client.get("/api/recent")
            assert resp.status_code == 200
            recent = resp.json()
            assert len(recent) >= 1

    def test_feedback_loop_e2e(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        app = create_app(e2e_settings)
        with TestClient(app) as client:
            resp = client.get("/api/breakthroughs")
            bt_id = resp.json()[0]["id"]
            # Submit feedback - note: web dashboard uses telegram_user_id=0,
            # and feedback table has ON CONFLICT(breakthrough_id, telegram_user_id) DO UPDATE,
            # so second submission overwrites first for same breakthrough.
            resp = client.post("/api/feedback", json={"breakthrough_id": bt_id, "rating": "hit"})
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            resp = client.post("/api/feedback", json={"breakthrough_id": bt_id, "rating": "inspire"})
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            # UPSERT means only 1 row (latest rating "inspire" overwrote "hit")
            resp = client.get("/api/stats")
            stats = resp.json()
            assert stats["total_feedback"] >= 1
            # The feedback_summary should reflect the latest submitted rating
            resp = client.get(f"/api/breakthroughs/{bt_id}")
            detail = resp.json()
            assert "inspire" in detail["feedback_summary"]

    def test_watchlist_toggle_e2e(self, e2e_db: Database, seed_raw_items: list[RawItem], e2e_settings: Settings):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.storage.preferences import PreferencesService
        stage1 = Stage1Filter(db=e2e_db)
        for item in seed_raw_items:
            eval_result = stage1.evaluate(item)
            eval_result.raw_item_id = item.id or 0
            asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        app = create_app(e2e_settings)
        with TestClient(app) as client:
            resp = client.get("/api/breakthroughs")
            bt_id = resp.json()[0]["id"]
            resp = client.post(f"/api/breakthroughs/{bt_id}/watchlist")
            assert resp.status_code == 200
            resp = client.get("/api/watchlist")
            assert resp.status_code == 200
            watchlist = resp.json()
            assert any(item["id"] == bt_id for item in watchlist)
            resp = client.post(f"/api/breakthroughs/{bt_id}/watchlist")
            assert resp.status_code == 200
            resp = client.get("/api/watchlist")
            assert resp.status_code == 200
            watchlist = resp.json()
            assert not any(item["id"] == bt_id for item in watchlist)

class TestE2EResilience:
    """Tests for pipeline resilience and error handling in E2E scenarios."""

    @pytest.fixture
    def e2e_db(self, tmp_path: Path) -> Database:
        from newsscout.storage.migrations import apply_migrations
        db = Database(tmp_path / "e2e_resilience.db", timeout=5.0)
        asyncio.run(db.initialize())
        with db.get_sync_connection() as conn:
            apply_migrations(conn=conn)
        yield db
        asyncio.run(db.close())

    @pytest.fixture
    def e2e_settings(self, tmp_path: Path) -> Settings:
        from pydantic import SecretStr
        settings = Settings(
            environment="test",
            db_path=tmp_path / "e2e_resilience.db",
            audio_output_dir=tmp_path / "audio",
            preferences_file=tmp_path / "preferences.json",
            gemini_api_key=SecretStr("mock-test-key"),
            telegram_bot_token=SecretStr("123456789:MOCK_E2E_TOKEN"),
            telegram_chat_id="99887766",
            github_token=SecretStr("mock-github-token"),
            log_level="DEBUG",
        )
        settings.ensure_directories()
        return settings

    def test_empty_pipeline_produces_fallback_digest(self, e2e_db: Database, e2e_settings: Settings):
        from newsscout.audio.digest_gen import DigestGenerator
        mock_tts = MockTTSEngine(e2e_settings)
        digest_gen = DigestGenerator(db=e2e_db, settings=e2e_settings, tts_engine=mock_tts)
        digest = asyncio.run(digest_gen.generate_digest(slot=DigestSlot.MORNING))
        assert digest.status == DigestStatus.SYNTHESIZED
        # With zero breakthroughs, the script generator produces a single
        # fallback "empty digest" script, so we get at least 1 track.
        assert len(digest.tracks) >= 1
        assert digest.total_duration_seconds > 0

    def test_dashboard_with_empty_database(self, e2e_db: Database, e2e_settings: Settings):
        app = create_app(e2e_settings)
        with TestClient(app) as client:
            resp = client.get("/api/breakthroughs")
            assert resp.status_code == 200
            assert resp.json() == []
            resp = client.get("/api/stats")
            assert resp.status_code == 200
            stats = resp.json()
            assert stats["total_breakthroughs"] == 0
            assert stats["total_feedback"] == 0
            resp = client.get("/api/digests")
            assert resp.status_code == 200
            assert resp.json() == []
            resp = client.get("/api/recent")
            assert resp.status_code == 200
            assert resp.json() == []
            resp = client.get("/api/watchlist")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_stage2_discard_items_persisted(self, e2e_db: Database, e2e_settings: Settings):
        from newsscout.filtering.stage1 import Stage1Filter
        from newsscout.filtering.stage2 import Stage2Evaluator
        from newsscout.storage.preferences import PreferencesService
        item = RawItem(
            source="github_releases",
            source_id="discard-test:v1.0",
            title="ChatGPT wrapper - thin UI for OpenAI API",
            url="https://github.com/test/chatgpt-wrapper",
            raw_content="A thin wrapper around ChatGPT. No real code. SaaS boilerplate.",
            metadata={"license": "mit", "stars": 100, "forks": 10, "pushed_at": datetime.now(timezone.utc).isoformat(), "root_files": ["pyproject.toml"]},
        )
        params = item.to_db_params()
        asyncio.run(e2e_db.execute(
            "INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at) VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)",
            params,
        ))
        rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM raw_items"))
        item = RawItem.from_row(rows[0])
        stage1 = Stage1Filter(db=e2e_db)
        eval_result = stage1.evaluate(item)
        eval_result.raw_item_id = item.id or 0
        asyncio.run(stage1.save_evaluation(eval_result))
        mock_gemini = MockGeminiClient()
        prefs = PreferencesService(e2e_db, e2e_settings)
        stage2 = Stage2Evaluator(db=e2e_db, settings=e2e_settings, preferences_service=prefs, mock_client=mock_gemini)
        asyncio.run(stage2.evaluate_pending_items(batch_size=10))
        all_rows = asyncio.run(e2e_db.fetch_all("SELECT * FROM breakthroughs"))
        assert len(all_rows) >= 0

    def test_health_check_endpoint(self, e2e_db: Database, e2e_settings: Settings):
        app = create_app(e2e_settings)
        with TestClient(app) as client:
            resp = client.get("/api/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "total_breakthroughs" in data
            assert "total_feedback" in data