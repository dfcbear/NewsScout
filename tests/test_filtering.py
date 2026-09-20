"""tests/test_filtering.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for NewsScout Two-Stage Filtering Pipeline:
- Stage 1 Deterministic Local Heuristics (Anti-hype blacklists, license checks, code/Docker heuristics, activity checks)
- Stage 2 Gemini 3.8 Flash Evaluator (80/20 profile distribution, prompt engineering, few-shot injection, structured JSON parsing, threshold checks, offline mock fallback, and SQLite persistence)
100% offline with zero external network dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from typing import Any

import httpx
import pytest

from newsscout.config import Settings
from newsscout.filtering.prompts import (
    STAGE2_JSON_SCHEMA,
    SYSTEM_INSTRUCTION,
    build_evaluation_user_prompt,
    format_few_shot_calibration,
)
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.filtering.stage2 import (
    DecisionCardPayload,
    Stage2EvaluationResponse,
    Stage2Evaluator,
    clean_and_parse_json,
)
from newsscout.storage.db import Database
from newsscout.storage.models import (
    Exemplar,
    FeedbackRating,
    RawItem,
    Stage1Evaluation,
    Stage2Category,
)
from newsscout.storage.preferences import ExemplarPair, PreferencesService


# ============================================================================
# Stage 1 Deterministic Filter Tests
# ============================================================================

class TestStage1Filter:
    """Tests deterministic heuristic filtering rules."""

    def test_saas_wrapper_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="wrap/lawyer",
            title="ChatGPT Wrapper for Lawyers",
            url="https://github.com/wrap/lawyer",
            raw_content="A simple wrapper around OpenAI API.",
            metadata={"root_files": ["Dockerfile"]},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "saas_wrapper" in (res.drop_reason or "")

    def test_nocode_landing_page_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="nocode/builder",
            title="No-Code Website Builder in Minutes",
            url="https://github.com/nocode/builder",
            raw_content="Build landing pages in seconds.",
            metadata={"root_files": ["Dockerfile"]},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "no_code_builder" in (res.drop_reason or "")

    def test_seo_spam_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="seo/farm",
            title="SEO Article Generator for Affiliates",
            url="https://github.com/seo/farm",
            raw_content="Rank #1 on Google with automated articles.",
            metadata={"root_files": ["Dockerfile"]},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "seo_spam" in (res.drop_reason or "")

    def test_crypto_tokenomics_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="crypto/token",
            title="New AI Memecoin and Tokenomics Launch",
            url="https://github.com/crypto/token",
            raw_content="Join the token launch presale.",
            metadata={"root_files": ["Dockerfile"]},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "crypto_web3" in (res.drop_reason or "")

    def test_cryptography_not_falsely_rejected(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="rust/crypto",
            title="Fast Rust Cryptography Primitives",
            url="https://github.com/rust/crypto",
            raw_content="High performance cryptographic algorithms with zero external dependencies.",
            metadata={
                "root_files": ["Cargo.toml"],
                "license": "MIT",
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.has_runnable_code is True

    def test_non_commercial_license_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="research/model",
            title="SOTA Vision Model",
            url="https://github.com/research/model",
            raw_content="Weights released under CC-BY-NC-4.0.",
            metadata={
                "root_files": ["Dockerfile", "pyproject.toml"],
                "license": "CC-BY-NC-4.0",
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "Non-commercial" in (res.drop_reason or "")

    def test_osi_approved_permissive_license(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="open/tool",
            title="Open Tool",
            url="https://github.com/open/tool",
            raw_content="Released under Apache 2.0.",
            metadata={
                "root_files": ["Dockerfile", "requirements.txt"],
                "license": "Apache-2.0",
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.detected_license == "Apache-2.0"

    def test_stale_repo_rejection(self) -> None:
        filter_engine = Stage1Filter()
        stale_date = (datetime.now(timezone.utc) - timedelta(days=240)).isoformat()
        raw_item = RawItem(
            source="github_releases",
            source_id="stale/project",
            title="Abandoned Model Project",
            url="https://github.com/stale/project",
            raw_content="Initial release.",
            metadata={
                "root_files": ["Dockerfile"],
                "license": "MIT",
                "pushed_at": stale_date,
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "Stale repository" in (res.drop_reason or "")

    def test_archived_repo_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="archived/project",
            title="Archived Project",
            url="https://github.com/archived/project",
            raw_content="Read only.",
            metadata={
                "root_files": ["Dockerfile"],
                "license": "MIT",
                "archived": True,
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "archived" in (res.drop_reason or "").lower()

    def test_artificial_star_spike_rejection(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="bot/spike",
            title="Fake Star Spiked Project",
            url="https://github.com/bot/spike",
            raw_content="Some tool.",
            metadata={
                "root_files": ["Dockerfile"],
                "license": "MIT",
                "stars": 1500,
                "forks": 0,  # Zero forks on 1500 stars
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "star-to-fork ratio" in (res.drop_reason or "")

    def test_hn_missing_github_dropped(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="hacker_news",
            source_id="hn-999",
            title="AI Revolution Blogpost",
            url="https://medium.com/some-ai-story",
            raw_content="A generic essay about AI.",
            metadata={"extracted_github_urls": []},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "No code repository linked" in (res.drop_reason or "")

    def test_hf_paper_missing_code_dropped(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="hf_daily_papers",
            source_id="arxiv:2409.00000",
            title="Theoretical AI Paper",
            url="https://huggingface.co/papers/2409.00000",
            raw_content="We prove theorems.",
            metadata={"github_repo": None},
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is False
        assert "No linked code repository" in (res.drop_reason or "")

    def test_runnable_code_docker_detected(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="org/valid-runner",
            title="vLLM Optimized Runner",
            url="https://github.com/org/valid-runner",
            raw_content="Fast inference.",
            metadata={
                "root_files": ["Dockerfile", "pyproject.toml"],
                "license": "Apache-2.0",
                "stars": 120,
                "forks": 25,
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.has_docker is True
        assert res.has_runnable_code is True

    def test_runnable_code_release_wheel_detected(self) -> None:
        filter_engine = Stage1Filter()
        raw_item = RawItem(
            source="github_releases",
            source_id="org/wheel-release",
            title="Prebuilt Wheel Binary",
            url="https://github.com/org/wheel-release",
            raw_content="Download wheel below.",
            metadata={
                "assets": ["runner-0.1.0-cp312-manylinux.whl"],
                "license": "MIT",
            },
        )
        res = filter_engine.evaluate(raw_item)
        assert res.passed is True
        assert res.has_runnable_code is True

    @pytest.mark.asyncio
    async def test_sqlite_upsert_idempotency_and_pipeline(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        filter_engine = Stage1Filter(db=migrated_db)

        # Seed raw item
        raw_item = RawItem(
            source="github_releases",
            source_id="test/repo:v1",
            title="Test Tool v1",
            url="https://github.com/test/repo",
            raw_content="docker run my-tool",
            metadata={"root_files": ["Dockerfile"], "license": "MIT"},
        )
        await migrated_db.execute(
            """INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);""",
            raw_item.to_db_params(),
        )
        row = await migrated_db.fetch_one("SELECT id FROM raw_items WHERE source_id = 'test/repo:v1';")
        raw_item_id = row["id"]

        # 1. Run pipeline
        results = await filter_engine.run_pipeline(limit=10)
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].has_docker is True

        # Verify in DB
        eval_row = await migrated_db.fetch_one(
            "SELECT * FROM stage1_evaluations WHERE raw_item_id = ?;", (raw_item_id,)
        )
        assert eval_row is not None
        assert eval_row["passed"] == 1

        # 2. Re-evaluating updates the row rather than failing unique constraint
        eval_obj = Stage1Evaluation.from_row(eval_row)
        eval_obj.drop_reason = "Manual override"
        await filter_engine.save_evaluation(eval_obj)

        eval_row2 = await migrated_db.fetch_one(
            "SELECT * FROM stage1_evaluations WHERE raw_item_id = ?;", (raw_item_id,)
        )
        assert eval_row2["drop_reason"] == "Manual override"


# ============================================================================
# Stage 2 Gemini Flash & Prompt Tests
# ============================================================================

class TestStage2PromptsAndParsing:
    """Tests prompt construction, few-shot formatting, and JSON sanitization."""

    def test_few_shot_calibration_formatting(self) -> None:
        exemplars = ExemplarPair(
            (
                [
                    Exemplar(
                        breakthrough_id=1,
                        title="vLLM NVFP4 Support",
                        rating=FeedbackRating.HIT,
                        category=Stage2Category.CORE,
                        tldr="Direct NVFP4 kernel execution for RTX 4090.",
                        use_case="Local 70B agent inference at 32 t/s.",
                        breakthrough_score=9.4,
                        roi_score=9.2,
                    )
                ],
                [
                    Exemplar(
                        breakthrough_id=2,
                        title="NextGen AI Landing Page Builder",
                        rating=FeedbackRating.HYPE,
                        category=Stage2Category.DISCARD,
                        tldr="Thin wrapper calling OpenAI chat completions.",
                        use_case="No local code or architectural merit.",
                        breakthrough_score=2.1,
                        roi_score=1.5,
                    )
                ],
            )
        )

        formatted = format_few_shot_calibration(exemplars)
        assert "RECENT POSITIVE RATINGS" in formatted
        assert "vLLM NVFP4 Support" in formatted
        assert "Score: 9.4/10" in formatted
        assert "RECENT NEGATIVE RATINGS" in formatted
        assert "NextGen AI Landing Page Builder" in formatted

    def test_build_evaluation_user_prompt_truncation(self) -> None:
        raw_item = RawItem(
            source="github_releases",
            source_id="huge/repo:v1",
            title="Huge Monorepo Release",
            url="https://github.com/huge/repo",
            raw_content="START_MARKER\n" + ("x" * 20000) + "\nEND_MARKER",
            metadata={"stars": 5000},
        )
        prompt = build_evaluation_user_prompt(raw_item, max_content_chars=5000)
        assert "START_MARKER" in prompt
        assert "END_MARKER" in prompt
        assert "... [TRUNCATED FOR LENGTH] ..." in prompt
        assert len(prompt) < 15000

    def test_clean_and_parse_json_variants(self) -> None:
        # 1. Pure valid JSON
        p1 = '{"breakthrough_score": 8.5, "roi_score": 8.0, "category": "core"}'
        assert clean_and_parse_json(p1)["category"] == "core"

        # 2. Markdown fenced with leading and trailing commentary
        p2 = 'Here is the evaluation:\n```json\n{"breakthrough_score": 9.0, "category": "serendipity"}\n```\nHope this helps!'
        assert clean_and_parse_json(p2)["breakthrough_score"] == 9.0

        # 3. Trailing commas repair
        p3 = '{"category": "core", "breakthrough_score": 8.0,}'
        assert clean_and_parse_json(p3)["breakthrough_score"] == 8.0

        # 4. Unrecoverable junk raises descriptive ValueError
        with pytest.raises(ValueError, match="Failed to parse model output as JSON"):
            clean_and_parse_json("I am an AI and I refuse to respond in JSON.")


class TestStage2EvaluationResponse:
    """Tests Pydantic validation and qualifying threshold logic."""

    def test_qualifying_core(self) -> None:
        res = Stage2EvaluationResponse(
            breakthrough_score=8.0,
            roi_score=7.5,
            category=Stage2Category.CORE,
            relevance_justification="Solid local tool.",
            decision_card=DecisionCardPayload(
                tldr="One sentence TLDR.",
                use_case="Agent reasoning.",
                comparison="2x faster.",
                quickstart="docker run tool",
                hardware_requirements="RTX 4090",
                license="Apache 2.0",
            ),
        )
        assert res.is_qualifying is True

    def test_non_qualifying_core_borderline_roi(self) -> None:
        res = Stage2EvaluationResponse(
            breakthrough_score=8.0,
            roi_score=6.9,  # < 7.0
            category=Stage2Category.CORE,
            relevance_justification="Novel but low ROI.",
            decision_card=DecisionCardPayload(
                tldr="TLDR",
                use_case="Use case",
                comparison="Comparison",
                quickstart="Quickstart",
                hardware_requirements="Hardware",
                license="License",
            ),
        )
        assert res.is_qualifying is False

    def test_qualifying_serendipity_novelty_exception_e11(self) -> None:
        # E11: High novelty out-of-bubble (score >= 8.5) allows modest ROI (roi >= 6.0)
        res = Stage2EvaluationResponse(
            breakthrough_score=9.0,
            roi_score=6.5,
            category=Stage2Category.SERENDIPITY,
            relevance_justification="Breakthrough robotics control loop.",
            decision_card=DecisionCardPayload(
                tldr="TLDR",
                use_case="Robotics control loop",
                comparison="Comparison",
                quickstart="Quickstart",
                hardware_requirements="Hardware",
                license="License",
            ),
        )
        assert res.is_qualifying is True

    def test_discard_category_never_qualifies(self) -> None:
        res = Stage2EvaluationResponse(
            breakthrough_score=3.0,
            roi_score=2.0,
            category=Stage2Category.DISCARD,
            relevance_justification="Thin wrapper.",
            decision_card=DecisionCardPayload(
                tldr="Discarded wrapper.",
                use_case="None",
                comparison="None",
                quickstart="None",
                hardware_requirements="None",
                license="None",
            ),
        )
        assert res.is_qualifying is False


class TestStage2EvaluatorPipeline:
    """Tests Stage2Evaluator mock fallback, rate limit handling, and SQLite persistence."""

    @pytest.mark.asyncio
    async def test_mock_gemini_fallback_and_wrapper_discard(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        evaluator = Stage2Evaluator(db=migrated_db, settings=test_settings)

        # 1. Wrapper item -> discard
        wrapper_item = RawItem(
            source="github_releases",
            source_id="wrap/chatgpt:v1",
            title="ChatGPT Wrapper Pro",
            url="https://github.com/wrap/chatgpt",
            raw_content="Thin wrapper app.",
        )
        res_wrap = await evaluator.evaluate_candidate(wrapper_item)
        assert res_wrap.category == Stage2Category.DISCARD
        assert res_wrap.is_qualifying is False

        # 2. Core item -> qualifies
        core_item = RawItem(
            source="github_releases",
            source_id="vllm/vllm:v0.6.1",
            title="vLLM High-Speed Inference Engine",
            url="https://github.com/vllm/vllm",
            raw_content="Native NVFP4 kernels for RTX 4090.",
        )
        res_core = await evaluator.evaluate_candidate(core_item)
        assert res_core.category == Stage2Category.CORE
        assert res_core.breakthrough_score >= 9.0
        assert res_core.is_qualifying is True

        # 3. Serendipity item -> qualifies under serendipity
        serendipity_item = RawItem(
            source="github_releases",
            source_id="unitree/ros2-robotics:v1",
            title="Unitree ROS2 Humanoid Control Loop",
            url="https://github.com/unitree/ros2-robotics",
            raw_content="Low latency VLA control loops on edge ARM64.",
        )
        res_ser = await evaluator.evaluate_candidate(serendipity_item)
        assert res_ser.category == Stage2Category.SERENDIPITY
        assert res_ser.is_qualifying is True

    @pytest.mark.asyncio
    async def test_evaluate_and_persist_breakthrough(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        evaluator = Stage2Evaluator(db=migrated_db, settings=test_settings)

        # Insert raw item and stage 1 evaluation
        raw_item = RawItem(
            source="github_releases",
            source_id="ggerganov/llama.cpp:b3600",
            title="llama.cpp b3600: Direct NVFP4 Matrix Multiplication",
            url="https://github.com/ggerganov/llama.cpp",
            raw_content="Optimized CUDA kernels.",
            metadata={"stars": 60000},
        )
        await migrated_db.execute(
            """INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);""",
            raw_item.to_db_params(),
        )
        row = await migrated_db.fetch_one("SELECT id FROM raw_items WHERE source_id = 'ggerganov/llama.cpp:b3600';")
        raw_item.id = row["id"]

        # Persist via Stage 2
        bt = await evaluator.evaluate_and_persist(raw_item)
        assert bt is not None
        assert bt.raw_item_id == raw_item.id
        assert bt.breakthrough_score >= 7.0
        assert "### llama.cpp" in bt.card_markdown
        assert bt.quickstart != ""

        # Verify record in breakthroughs table
        bt_row = await migrated_db.fetch_one("SELECT * FROM breakthroughs WHERE raw_item_id = ?;", (raw_item.id,))
        assert bt_row is not None
        assert bt_row["category"] in ("core", "serendipity")

        # Re-running evaluate_and_persist updates the row without UNIQUE violation
        bt2 = await evaluator.evaluate_and_persist(raw_item)
        assert bt2 is not None
        assert bt2.id == bt.id

    @pytest.mark.asyncio
    async def test_evaluate_pending_items_batch(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        evaluator = Stage2Evaluator(db=migrated_db, settings=test_settings)

        # Seed 2 raw items, mark both as passed in Stage 1
        for idx, title in enumerate(["vLLM Fast Ingest", "Qdrant Vector Server"], 1):
            await migrated_db.execute(
                f"""INSERT INTO raw_items (id, source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES ({idx}, 'github_releases', 'tool:{idx}', '{title}', 'https://github.com/tool/{idx}', 'Kernel logic', '{{}}', '2026-09-17T12:00:00Z');"""
            )
            await migrated_db.execute(
                f"""INSERT INTO stage1_evaluations (raw_item_id, passed, has_docker, has_runnable_code, evaluated_at)
                VALUES ({idx}, 1, 1, 1, '2026-09-17T12:05:00Z');"""
            )

        breakthroughs = await evaluator.evaluate_pending_items(batch_size=10, concurrency=2)
        assert len(breakthroughs) == 2

        # Verify subsequent call returns empty list because all are processed
        empty_call = await evaluator.evaluate_pending_items(batch_size=10)
        assert len(empty_call) == 0

    @pytest.mark.asyncio
    async def test_call_gemini_with_backoff_on_429(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            mock_body = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "breakthrough_score": 9.0,
                                            "roi_score": 8.5,
                                            "category": "core",
                                            "relevance_justification": "Passed test backoff.",
                                            "decision_card": {
                                                "tldr": "TLDR",
                                                "use_case": "Use case",
                                                "comparison": "Comparison",
                                                "quickstart": "Quickstart",
                                                "hardware_requirements": "Requirements",
                                                "license": "License",
                                            },
                                        }
                                    )
                                }
                            ]
                        }
                    }
                ]
            }
            return httpx.Response(200, json=mock_body)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        # Configure settings with fake gemini key to test transport
        custom_settings = Settings(gemini_api_key="actual-api-key-value-12345")
        evaluator = Stage2Evaluator(db=migrated_db, settings=custom_settings, http_client=client)

        raw_item = RawItem(
            source="github_releases",
            source_id="test/backoff",
            title="Backoff Test Candidate",
            url="https://github.com/test/backoff",
        )
        res = await evaluator.evaluate_candidate(raw_item)
        assert attempts == 2
        assert res.breakthrough_score == 9.0
        assert res.category == Stage2Category.CORE
