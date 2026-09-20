"""tests/test_llm_adversarial.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Stress & Edge-Case Verification Harness for NewsScout Universal LLM Abstraction:
1. JSON repair stress (trailing comma variations, unclosed fences, nested code blocks, unicode characters, unescaped newlines)
2. Simultaneous HTTP 429 quota exceptions and concurrent fallback failovers
3. Cascade failure handling (primary quota + fallback quota, primary quota + fallback 500, fallback validation failure)
4. Batch resilience under cascade failures (evaluate_pending_items isolation)
5. Client resource cleanup, context manager lifecycles, connection pool safety, and external client retention
6. Backend response_format fallback and Pydantic bounds enforcement
100% offline using httpx.MockTransport with zero live network dependencies.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr

from newsscout.config import Settings
from newsscout.filtering.prompts import STAGE2_JSON_SCHEMA, SYSTEM_INSTRUCTION
from newsscout.filtering.stage2 import (
    DecisionCardPayload,
    Stage2EvaluationResponse,
    Stage2Evaluator,
    clean_and_parse_json,
)
from newsscout.llm.base import (
    BaseLLMClient,
    LLMError,
    LLMQuotaExceededException,
    LLMResponseValidationError,
    LLMServerException,
)
from newsscout.llm.gemini_client import GeminiLLMClient
from newsscout.llm.mock import MockLLMClient
from newsscout.llm.openai_client import OpenAICompatibleClient
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem, Stage2Category


# ============================================================================
# Helpers & Fixtures
# ============================================================================

def make_valid_stage2_dict(title: str = "Test Tool", score: float = 9.0) -> dict[str, Any]:
    return {
        "breakthrough_score": score,
        "roi_score": score - 0.5,
        "category": "core",
        "relevance_justification": f"Valid evaluation for {title}.",
        "decision_card": {
            "tldr": f"{title} enables 3x faster local execution.",
            "use_case": "RTX 4090 agent workflows.",
            "comparison": "Outperforms existing baselines.",
            "quickstart": f"docker run -p 8080:8080 {title.lower().replace(' ', '/')}:latest",
            "hardware_requirements": "NVIDIA RTX 4090 (24GB VRAM)",
            "license": "Apache-2.0",
        },
    }


# ============================================================================
# 1. JSON Repair Stress Tests
# ============================================================================

class TestJSONRepairStress:
    """Stress tests clean_and_parse_json against pathological LLM outputs."""

    def test_unclosed_outer_fence_at_start(self) -> None:
        """LLM starts with ```json but forgets the closing ``` at the end."""
        raw = """```json
        {
            "name": "unclosed_fence",
            "score": 8.7
        }"""
        res = clean_and_parse_json(raw)
        assert res["name"] == "unclosed_fence"
        assert res["score"] == 8.7

    def test_unclosed_outer_fence_with_trailing_commentary(self) -> None:
        """LLM starts with ```json, gives JSON, then trailing commentary with no closing fence."""
        raw = """```json
        {
            "tool": "llama.cpp",
            "active": true
        }
        This tool is highly recommended for local inference.
        """
        res = clean_and_parse_json(raw)
        assert res["tool"] == "llama.cpp"
        assert res["active"] is True

    def test_unclosed_fence_with_preceding_text(self) -> None:
        """Conversational text, unclosed fence, and valid JSON."""
        raw = """Here is your JSON response:
        ```json
        {"status": "ok", "count": 42}
        """
        res = clean_and_parse_json(raw)
        assert res["status"] == "ok"
        assert res["count"] == 42

    def test_trailing_comma_variations_deeply_nested(self) -> None:
        """Trailing commas in nested dictionaries, lists, and multi-line structures."""
        raw = """
        {
            "core": {
                "models": [
                    {"id": "m1", "tags": ["fast", "fp16", ], },
                    {"id": "m2", "tags": ["nvfp4", ], },
                ],
                "active": true,
            },
            "status": "ready",
        }
        """
        res = clean_and_parse_json(raw)
        assert len(res["core"]["models"]) == 2
        assert res["core"]["models"][0]["tags"] == ["fast", "fp16"]
        assert res["core"]["active"] is True
        assert res["status"] == "ready"

    def test_multiple_consecutive_trailing_commas(self) -> None:
        """Model produces double or triple trailing commas before closing braces."""
        raw = """
        {
            "numbers": [1, 2, 3,,, ],
            "meta": {"nested": "value",,, },
        }
        """
        res = clean_and_parse_json(raw)
        assert res["numbers"] == [1, 2, 3]
        assert res["meta"]["nested"] == "value"

    def test_trailing_commas_with_whitespace_newlines_tabs(self) -> None:
        """Trailing commas separated by irregular whitespace, tabs, and newlines."""
        raw = "{\n\t\"key\": 100 ,\t \n\r }"
        res = clean_and_parse_json(raw)
        assert res["key"] == 100

    def test_unicode_umlauts_and_sharp_s(self) -> None:
        """Handles German characters (ä, ö, ü, ß) both literal UTF-8 and escaped."""
        raw = """
        {
            "tldr": "Größte Neuerung: Überragende Zuverlässigkeit für KI-Agenten & Präzision.",
            "escaped": "Gro\\u00dfartige L\\u00f6sung f\\u00fcr \\u00c4nderungen.",
            "features": ["Maßstäblich", "Zuverlässig", "Kühles Design"]
        }
        """
        res = clean_and_parse_json(raw)
        assert "Größte Neuerung" in res["tldr"]
        assert "Großartige Lösung" in res["escaped"]
        assert res["features"][0] == "Maßstäblich"

    def test_unicode_emojis_and_cjk(self) -> None:
        """Handles emojis and CJK characters properly in all fields."""
        raw = """
        {
            "emojis": "🎯 🚀 🤖 💻 🧠",
            "model_zh": "GLM 5.2 智谱清言",
            "rationale": "卓越的推理能力和高性价比，适用于Raspberry Pi 5本地监控。"
        }
        """
        res = clean_and_parse_json(raw)
        assert "🎯" in res["emojis"]
        assert "智谱清言" in res["model_zh"]
        assert "Raspberry Pi 5" in res["rationale"]

    def test_unicode_math_and_technical_symbols(self) -> None:
        """Handles symbols: µs, ±, ², ≤, ≥, →, ∑, etc."""
        raw = """
        {
            "latency": "120 µs ± 5 µs",
            "comparison": "Speed ≥ 3.5x baseline → 4090 VRAM usage ≤ 18GB²"
        }
        """
        res = clean_and_parse_json(raw)
        assert "120 µs" in res["latency"]
        assert "≥ 3.5x" in res["comparison"]

    def test_unescaped_newlines_inside_json_string(self) -> None:
        """Model outputs literal unescaped newlines inside string values."""
        raw = """{
            "multiline": "Line 1 of the explanation.
Line 2 of the explanation.
Line 3 of the explanation.",
            "status": "ok"
        }"""
        res = clean_and_parse_json(raw)
        assert "Line 1" in res["multiline"]
        assert "Line 3" in res["multiline"]
        assert res["status"] == "ok"

    def test_nested_markdown_code_blocks_inside_string(self) -> None:
        """String value containing markdown code fence with language specifier."""
        raw = """```json
        {
            "title": "Agent Harness",
            "quickstart": "```bash\\n$ uvx run --from newsscout scout\\n```",
            "comparison": "Normal comparison"
        }
        ```"""
        res = clean_and_parse_json(raw)
        assert res["title"] == "Agent Harness"
        assert "uvx run" in res["quickstart"]

    def test_multiple_code_blocks_with_preamble_and_outro(self) -> None:
        """Model generates introductory bash command, then JSON block, then commentary."""
        raw = """
        First, install the package:
        ```bash
        pip install newsscout
        ```

        Here is the evaluation JSON:
        ```json
        {
            "breakthrough_score": 9.1,
            "roi_score": 8.8
        }
        ```

        Let me know if you need anything else!
        """
        res = clean_and_parse_json(raw)
        assert res["breakthrough_score"] == 9.1
        assert res["roi_score"] == 8.8

    def test_pathological_empty_and_whitespace_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Empty model output"):
            clean_and_parse_json("")
        with pytest.raises(ValueError, match="Empty model output"):
            clean_and_parse_json("   \n\t  ")

    def test_pathological_corrupted_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse model output as JSON"):
            clean_and_parse_json("{This is completely invalid and unclosed JSON")


# ============================================================================
# 2. OpenAICompatibleClient Edge Cases & Error Recovery
# ============================================================================

class TestOpenAIClientAdversarial:
    """Adversarial and boundary test cases for OpenAICompatibleClient."""

    @pytest.mark.asyncio
    async def test_backend_rejects_response_format_auto_retries_plain(self) -> None:
        """Simulates older backends (e.g. Ollama <0.1.28 or llama.cpp) rejecting response_format."""
        call_count = 0
        valid_data = make_valid_stage2_dict()

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            if call_count == 1:
                assert "response_format" in body
                return httpx.Response(
                    400,
                    json={"error": "Unsupported parameter: response_format is not supported by this engine"},
                )
            assert "response_format" not in body
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(valid_data)}}]},
            )

        client = OpenAICompatibleClient(
            base_url="http://localhost:11434/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        res = await client.generate_structured(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt="Evaluate tool",
            response_model=Stage2EvaluationResponse,
            json_schema=STAGE2_JSON_SCHEMA,
        )
        assert call_count == 2
        assert isinstance(res, Stage2EvaluationResponse)
        assert res.breakthrough_score == 9.0

    @pytest.mark.asyncio
    async def test_quota_keywords_trigger_immediate_quota_exception(self) -> None:
        """Verifies that HTTP 429 with quota keywords raises LLMQuotaExceededException immediately."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                429,
                json={"error": {"message": "You have exceeded your monthly credit balance", "type": "billing"}},
            )

        client = OpenAICompatibleClient(
            base_url="https://api.groq.com/openai/v1",
            max_retries=4,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMQuotaExceededException, match="Quota exceeded"):
            await client.generate_text("Prompt")

        assert calls == 1

    @pytest.mark.asyncio
    async def test_network_timeout_retry_and_recovery(self) -> None:
        """Verifies recovery when initial attempt times out but subsequent attempt succeeds."""
        attempt = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise httpx.ReadTimeout("Socket read timed out")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Recovered from timeout"}}]},
            )

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            max_retries=3,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        res = await client.generate_text("Prompt")
        assert attempt == 2
        assert res == "Recovered from timeout"

    @pytest.mark.asyncio
    async def test_pydantic_score_bounds_validation_failure(self) -> None:
        """Model emits breakthrough_score > 10.0 (violating Field(le=10.0))."""
        invalid_data = make_valid_stage2_dict()
        invalid_data["breakthrough_score"] = 15.0  # Invalid! Out of 1.0 - 10.0 range

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(invalid_data)}}]},
            )

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMResponseValidationError, match="Invalid schema"):
            await client.generate_structured(
                prompt="Evaluate",
                response_model=Stage2EvaluationResponse,
            )


# ============================================================================
# 3. Simultaneous Quota Exceptions & Concurrency Stress
# ============================================================================

class TestSimultaneousQuotaFailover:
    """Stress tests high-concurrency requests hitting quota limits simultaneously."""

    @pytest.mark.asyncio
    async def test_concurrent_simultaneous_429_failover(
        self,
        migrated_db: Database,
        test_settings: Settings,
    ) -> None:
        """10 concurrent tasks all hit HTTP 429 Quota Exceeded on primary LLM simultaneously.
        All 10 must seamlessly fail over to fallback LLM concurrently without deadlocks
        or lost items.
        """
        num_tasks = 10
        primary_calls = 0
        fallback_calls = 0
        lock = asyncio.Lock()

        class ConcurrentQuotaPrimary(BaseLLMClient):
            provider_name = "gemini_overloaded"
            model_name = "gemini-3.8-flash"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                nonlocal primary_calls
                async with lock:
                    primary_calls += 1
                await asyncio.sleep(0.01)
                raise LLMQuotaExceededException("Quota exceeded: Resource has been exhausted (HTTP 429)")

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return False

            async def aclose(self) -> None:
                pass

        class ConcurrentWorkingFallback(BaseLLMClient):
            provider_name = "glm_fallback"
            model_name = "glm-4-flash"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                nonlocal fallback_calls
                async with lock:
                    fallback_calls += 1
                await asyncio.sleep(0.01)
                user_prompt = kwargs.get("user_prompt", "") or (args[0] if args else "")
                return Stage2EvaluationResponse.model_validate(
                    make_valid_stage2_dict(title=f"Tool-{fallback_calls}", score=8.5)
                )

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return True

            async def aclose(self) -> None:
                pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=ConcurrentQuotaPrimary(),
            fallback_llm_client=ConcurrentWorkingFallback(),
        )

        # Seed 10 raw items in DB
        items = []
        for i in range(1, num_tasks + 1):
            raw = RawItem(
                id=200 + i,
                source="github_releases",
                source_id=f"repo/tool-{i}:v1.0",
                title=f"Parallel Breakthrough Tool {i}",
                url=f"https://github.com/repo/tool-{i}",
                raw_content=f"High performance tool {i}",
            )
            await migrated_db.execute(
                """INSERT INTO raw_items (id, source, source_id, title, url, raw_content)
                VALUES (?, ?, ?, ?, ?, ?);""",
                (raw.id, raw.source, raw.source_id, raw.title, raw.url, raw.raw_content),
            )
            items.append(raw)

        # Launch all 10 simultaneously
        tasks = [evaluator.evaluate_and_persist(item) for item in items]
        breakthroughs = await asyncio.gather(*tasks)

        # Verifications
        assert primary_calls == num_tasks
        assert fallback_calls == num_tasks
        assert len(breakthroughs) == num_tasks
        assert all(b is not None for b in breakthroughs)

        # Verify all 10 are recorded in SQLite
        rows = await migrated_db.fetch_all(
            "SELECT id, raw_item_id, breakthrough_score FROM breakthroughs WHERE raw_item_id BETWEEN 201 AND 210;"
        )
        assert len(rows) == num_tasks
        for r in rows:
            assert r["breakthrough_score"] == 8.5


# ============================================================================
# 4. Cascade Failure Tests
# ============================================================================

class TestCascadeFailures:
    """Verifies system behavior when both primary and fallback LLMs fail."""

    @pytest.mark.asyncio
    async def test_cascade_failure_primary_quota_fallback_quota(
        self,
        migrated_db: Database,
        test_settings: Settings,
    ) -> None:
        """Primary hits quota, fallback ALSO hits quota -> raises LLMQuotaExceededException."""
        raw_item = RawItem(
            id=301,
            source="hn",
            source_id="item-301",
            title="Double Quota Exhaustion Candidate",
            url="https://news.ycombinator.com/item?id=301",
        )

        class QuotaPrimary(BaseLLMClient):
            provider_name = "gemini"
            model_name = "gemini-3.8-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Primary quota exhausted")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        class QuotaFallback(BaseLLMClient):
            provider_name = "glm"
            model_name = "glm-4-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Fallback quota also exhausted")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=QuotaPrimary(),
            fallback_llm_client=QuotaFallback(),
        )

        with pytest.raises(LLMQuotaExceededException, match="Fallback quota also exhausted"):
            await evaluator.evaluate_candidate(raw_item)

    @pytest.mark.asyncio
    async def test_cascade_failure_primary_quota_fallback_server_error(
        self,
        migrated_db: Database,
        test_settings: Settings,
    ) -> None:
        """Primary hits quota, fallback returns HTTP 503 Server Error -> raises LLMServerException."""
        raw_item = RawItem(
            id=302,
            source="hn",
            source_id="item-302",
            title="Fallback Server Error Candidate",
            url="https://news.ycombinator.com/item?id=302",
        )

        class QuotaPrimary(BaseLLMClient):
            provider_name = "gemini"
            model_name = "gemini-3.8-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Primary quota exhausted")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        class ServerErrorFallback(BaseLLMClient):
            provider_name = "glm"
            model_name = "glm-4-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMServerException("Fallback server returned HTTP 503 Service Unavailable")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=QuotaPrimary(),
            fallback_llm_client=ServerErrorFallback(),
        )

        with pytest.raises(LLMServerException, match="HTTP 503 Service Unavailable"):
            await evaluator.evaluate_candidate(raw_item)

    @pytest.mark.asyncio
    async def test_cascade_failure_primary_quota_fallback_validation_error(
        self,
        migrated_db: Database,
        test_settings: Settings,
    ) -> None:
        """Primary hits quota, fallback returns gibberish that cannot be validated."""
        raw_item = RawItem(
            id=303,
            source="hn",
            source_id="item-303",
            title="Fallback Validation Error Candidate",
            url="https://news.ycombinator.com/item?id=303",
        )

        class QuotaPrimary(BaseLLMClient):
            provider_name = "gemini"
            model_name = "gemini-3.8-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Primary quota exhausted")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        class CorruptedFallback(BaseLLMClient):
            provider_name = "glm"
            model_name = "glm-4-flash"
            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMResponseValidationError("Fallback output missing decision_card")
            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return True
            async def aclose(self) -> None: pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=QuotaPrimary(),
            fallback_llm_client=CorruptedFallback(),
        )

        with pytest.raises(LLMResponseValidationError, match="missing decision_card"):
            await evaluator.evaluate_candidate(raw_item)

    @pytest.mark.asyncio
    async def test_batch_evaluate_pending_items_resilient_to_cascade_failure(
        self,
        migrated_db: Database,
        test_settings: Settings,
    ) -> None:
        """Tests that evaluate_pending_items isolates an individual item cascade failure
        and continues processing other items in the batch without crashing.
        """
        # Seed 2 raw items and stage 1 evaluations
        await migrated_db.execute(
            """INSERT INTO raw_items (id, source, source_id, title, url, raw_content)
            VALUES (401, 'gh', 'repo/fail', 'Failing Item', 'https://gh.com/fail', 'code'),
                   (402, 'gh', 'repo/pass', 'Passing Item', 'https://gh.com/pass', 'code');"""
        )
        await migrated_db.execute(
            """INSERT INTO stage1_evaluations (id, raw_item_id, passed, detected_license, has_docker, has_runnable_code)
            VALUES (401, 401, 1, 'MIT', 1, 1),
                   (402, 402, 1, 'Apache-2.0', 1, 1);"""
        )

        class SelectiveFailoverClient(BaseLLMClient):
            provider_name = "selective_mock"
            model_name = "mock-1"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                user_prompt = kwargs.get("user_prompt", "")
                if "Failing Item" in user_prompt:
                    raise LLMQuotaExceededException("Quota exceeded for Failing Item")
                return Stage2EvaluationResponse.model_validate(
                    make_valid_stage2_dict("Passing Item", score=9.0)
                )

            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return True
            async def aclose(self) -> None: pass

        class FailingFallback(BaseLLMClient):
            provider_name = "failing_fallback"
            model_name = "mock-fb"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMServerException("Fallback server offline")

            async def generate_text(self, *args: Any, **kwargs: Any) -> str: return ""
            async def health_check(self) -> bool: return False
            async def aclose(self) -> None: pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=SelectiveFailoverClient(),
            fallback_llm_client=FailingFallback(),
        )

        # evaluate_pending_items should not raise! It handles individual errors gracefully.
        results = await evaluator.evaluate_pending_items(batch_size=10, concurrency=2)

        # 1 passed, 1 failed gracefully
        assert len(results) == 1
        assert results[0].title == "Passing Item"


# ============================================================================
# 5. Client Resource Cleanup & Context Manager Tests
# ============================================================================

class TestClientResourceCleanup:
    """Verifies proper release of underlying HTTP connection pools and resources."""

    @pytest.mark.asyncio
    async def test_openai_client_context_manager_lifecycle(self) -> None:
        """Using `async with OpenAICompatibleClient()` closes internal client on exit."""
        client = OpenAICompatibleClient(base_url="http://localhost:11434/v1")
        async with client:
            http_c = await client._get_client()
            assert not http_c.is_closed

        assert http_c.is_closed

    @pytest.mark.asyncio
    async def test_gemini_client_context_manager_lifecycle(self) -> None:
        """Using `async with GeminiLLMClient()` closes internal client on exit."""
        client = GeminiLLMClient(api_key="test-key")
        async with client:
            http_c = await client._get_client()
            assert not http_c.is_closed

        assert http_c.is_closed

    @pytest.mark.asyncio
    async def test_client_aclose_is_idempotent(self) -> None:
        """Calling `aclose()` multiple times does not raise or fail."""
        client = OpenAICompatibleClient(base_url="http://localhost:11434/v1")
        http_c = await client._get_client()
        assert not http_c.is_closed

        await client.aclose()
        assert http_c.is_closed

        # Second call must be a safe no-op
        await client.aclose()
        assert http_c.is_closed

    @pytest.mark.asyncio
    async def test_external_http_client_is_not_closed_by_aclose(self) -> None:
        """When an external `http_client` is passed, client.aclose() preserves caller's client."""
        external_client = httpx.AsyncClient()
        try:
            client = OpenAICompatibleClient(
                base_url="http://localhost:11434/v1",
                http_client=external_client,
            )
            await client.aclose()
            # External client must remain open!
            assert not external_client.is_closed
        finally:
            await external_client.aclose()

    @pytest.mark.asyncio
    async def test_client_reopens_after_aclose(self) -> None:
        """If a request is triggered after aclose(), client re-establishes a fresh active client."""
        client = OpenAICompatibleClient(base_url="http://localhost:11434/v1")
        first_c = await client._get_client()
        await client.aclose()
        assert first_c.is_closed

        # Second call to _get_client() re-creates an active client
        second_c = await client._get_client()
        assert not second_c.is_closed
        assert second_c is not first_c
        await client.aclose()
        assert second_c.is_closed
