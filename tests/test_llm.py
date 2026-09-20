"""tests/test_llm.py
~~~~~~~~~~~~~~~~~~~
Comprehensive offline unit and integration test suite for NewsScout Universal LLM Abstraction:
- JSON repair & Markdown code fence stripping
- OpenAICompatibleClient structured JSON, retry backoff, and quota limits
- GeminiLLMClient structured output and error translation
- MockLLMClient deterministic behavior and fault injection
- Model factory instantiation (Gemini, OpenAI-compatible, GLM fallback, Mock)
- Stage2Evaluator dynamic quota failover and persistence
100% offline with zero live network dependencies (using httpx.MockTransport).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

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
    LLMException,
    LLMQuotaExceededException,
    LLMResponseValidationError,
    LLMServerException,
)
from newsscout.llm.factory import create_fallback_llm_client, create_llm_client
from newsscout.llm.gemini_client import GeminiLLMClient
from newsscout.llm.mock import MockLLMClient
from newsscout.llm.openai_client import OpenAICompatibleClient
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem, Stage2Category


# ============================================================================
# Fixtures & Test Models
# ============================================================================

class SampleModel(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=10.0)
    tags: list[str] = Field(default_factory=list)


@pytest.fixture
def sample_raw_item() -> RawItem:
    return RawItem(
        id=101,
        source="github_releases",
        source_id="vllm-project/vllm:v0.7.0",
        title="vLLM v0.7.0: Direct NVFP4 Support",
        url="https://github.com/vllm-project/vllm",
        raw_content="Native NVFP4 kernels for RTX 4090 workstation inference.",
    )


@pytest.fixture
def valid_stage2_json_dict() -> dict[str, Any]:
    return {
        "breakthrough_score": 9.3,
        "roi_score": 9.1,
        "category": "core",
        "relevance_justification": "Direct NVFP4 kernel execution for RTX 4090.",
        "decision_card": {
            "tldr": "Enables native NVFP4 kernel execution inside vLLM.",
            "use_case": "Local agent reasoning with 70B models at 35 t/s.",
            "comparison": "3.5x lower latency than standard FP16 baselines.",
            "quickstart": "docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest",
            "hardware_requirements": "NVIDIA RTX 4090 (24GB VRAM)",
            "license": "Apache-2.0",
        },
    }


# ============================================================================
# 1. TestJSONRepair
# ============================================================================

class TestJSONRepair:
    """Verifies robust JSON extraction and repair against malformed LLM outputs."""

    def test_clean_and_parse_json_plain(self) -> None:
        raw = '{"name": "test", "score": 8.5, "tags": ["a", "b"]}'
        res = clean_and_parse_json(raw)
        assert res["name"] == "test"
        assert res["score"] == 8.5
        assert res["tags"] == ["a", "b"]

    def test_clean_and_parse_json_markdown_fences(self) -> None:
        raw = '```json\n{"name": "fenced", "score": 9.0}\n```'
        assert clean_and_parse_json(raw)["name"] == "fenced"

        raw_upper = '```JSON\n{"name": "fenced_upper", "score": 9.0}\n```'
        assert clean_and_parse_json(raw_upper)["name"] == "fenced_upper"

        raw_bare = '```\n{"name": "fenced_bare", "score": 9.0}\n```'
        assert clean_and_parse_json(raw_bare)["name"] == "fenced_bare"

    def test_clean_and_parse_json_preamble_and_postamble(self) -> None:
        raw = """
        Here is the evaluation result according to your schema:
        ```json
        {
            "name": "wrapped",
            "score": 8.0
        }
        ```
        Let me know if you need further adjustments!
        """
        assert clean_and_parse_json(raw)["name"] == "wrapped"

    def test_clean_and_parse_json_trailing_commas(self) -> None:
        raw = """
        {
            "name": "commas",
            "score": 7.5,
            "tags": ["alpha", "beta", ],
        }
        """
        res = clean_and_parse_json(raw)
        assert res["name"] == "commas"
        assert res["tags"] == ["alpha", "beta"]

    def test_clean_and_parse_json_nested_code_fences_preserved(self) -> None:
        """Crucial: ensures ```bash inside quickstart string is NOT stripped or truncated."""
        raw = """```json
        {
            "name": "nested",
            "score": 9.5,
            "quickstart": "```bash\\n$ docker run -p 8080:8080 app:v1\\n```"
        }
        ```"""
        res = clean_and_parse_json(raw)
        assert res["name"] == "nested"
        assert "docker run" in res["quickstart"]

    def test_clean_and_parse_json_unrecoverable_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse model output as JSON"):
            clean_and_parse_json("I cannot fulfill this request as an AI assistant.")


# ============================================================================
# 2. TestOpenAICompatibleClient
# ============================================================================

class TestOpenAICompatibleClient:
    """Tests OpenAI-compatible client with mocked HTTP transport."""

    @pytest.mark.asyncio
    async def test_generate_text_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer test-api-key"
            body = json.loads(request.content)
            assert body["model"] == "llama3.1:8b"
            assert body["messages"][0]["role"] == "system"
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Text reply from model"}}]
            })

        client = OpenAICompatibleClient(
            base_url="http://localhost:11434/v1",
            api_key="test-api-key",
            model="llama3.1:8b",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        res = await client.generate_text("System", "User")
        assert res == "Text reply from model"

    @pytest.mark.asyncio
    async def test_generate_structured_success(self, valid_stage2_json_dict: dict[str, Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps(valid_stage2_json_dict)}}]
            })

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            model="vllm-model",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        res = await client.generate_structured(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt="Evaluate vLLM",
            response_model=Stage2EvaluationResponse,
            json_schema=STAGE2_JSON_SCHEMA,
        )
        assert isinstance(res, Stage2EvaluationResponse)
        assert res.breakthrough_score == 9.3
        assert res.category == Stage2Category.CORE
        assert res.is_qualifying is True

    @pytest.mark.asyncio
    async def test_quota_limit_exceeded_raises_quota_exception(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={
                "error": {
                    "message": "You have exceeded your current quota. Please check your plan.",
                    "type": "insufficient_quota",
                }
            })

        client = OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMQuotaExceededException, match="Quota exceeded"):
            await client.generate_text("System", "User")

    @pytest.mark.asyncio
    async def test_rate_limit_backoff_with_retry_after(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Recovered after 429"}}]
            })

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        res = await client.generate_text("Sys", "User")
        assert attempts == 2
        assert res == "Recovered after 429"

    @pytest.mark.asyncio
    async def test_custom_headers_and_empty_api_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "Authorization" not in request.headers
            assert request.headers["X-Custom-Tenant"] == "NewsScoutPi5"
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Custom header verified"}}]
            })

        client = OpenAICompatibleClient(
            base_url="http://localhost:11434",
            api_key="",  # Empty for local Ollama
            custom_headers={"X-Custom-Tenant": "NewsScoutPi5"},
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        res = await client.generate_text("Sys", "User")
        assert res == "Custom header verified"

    @pytest.mark.asyncio
    async def test_server_error_retry_and_raise(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(500, text="Internal Server Error")

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            max_retries=2,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMServerException, match="Server error HTTP 500"):
            await client.generate_text("Sys", "User")
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_health_check_endpoint(self) -> None:
        def handler_ok(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json={"data": [{"id": "model-1"}]})

        client_ok = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_ok)),
        )
        assert await client_ok.health_check() is True

        def handler_fail(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Down")

        client_fail = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler_fail)),
        )
        assert await client_fail.health_check() is False

    @pytest.mark.asyncio
    async def test_validation_error_raises_llm_response_validation_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Missing required fields according to SampleModel
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": '{"invalid": "data"}'}}]
            })

        client = OpenAICompatibleClient(
            base_url="http://localhost:8000/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMResponseValidationError):
            await client.generate_structured("prompt", SampleModel)


# ============================================================================
# 3. TestGeminiLLMClient
# ============================================================================

class TestGeminiLLMClient:
    """Tests GeminiLLMClient driver with mocked Google API transport."""

    @pytest.mark.asyncio
    async def test_gemini_generate_structured(self, valid_stage2_json_dict: dict[str, Any]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "generativelanguage.googleapis.com" in request.url.host
            assert request.headers["x-goog-api-key"] == "valid-gemini-key"
            body = json.loads(request.content)
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            assert "responseSchema" in body["generationConfig"]
            return httpx.Response(200, json={
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(valid_stage2_json_dict)}]}}
                ]
            })

        client = GeminiLLMClient(
            api_key="valid-gemini-key",
            model="gemini-3.8-flash",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        res = await client.generate_structured(
            system_instruction=SYSTEM_INSTRUCTION,
            user_prompt="Evaluate",
            response_model=Stage2EvaluationResponse,
            json_schema=STAGE2_JSON_SCHEMA,
        )
        assert res.breakthrough_score == 9.3
        assert res.category == Stage2Category.CORE

    @pytest.mark.asyncio
    async def test_gemini_quota_exceeded_raises_quota_exception(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted (e.g. check quota).",
                    "status": "RESOURCE_EXHAUSTED",
                }
            })

        client = GeminiLLMClient(
            api_key="gemini-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMQuotaExceededException, match="Quota exceeded on gemini"):
            await client.generate_text("Sys", "User")

    @pytest.mark.asyncio
    async def test_gemini_health_check(self) -> None:
        client_with_key = GeminiLLMClient(api_key="valid-key")
        assert await client_with_key.health_check() is True

        client_without_key = GeminiLLMClient(api_key="")
        assert await client_without_key.health_check() is False

    @pytest.mark.asyncio
    async def test_gemini_empty_parts_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": []}}]
            })

        client = GeminiLLMClient(
            api_key="gemini-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMError, match="empty parts"):
            await client.generate_text("Sys", "User")


# ============================================================================
# 4. TestMockLLMClient
# ============================================================================

class TestMockLLMClient:
    """Tests deterministic MockLLMClient and fault injection."""

    @pytest.mark.asyncio
    async def test_mock_llm_client_default_generation(self) -> None:
        mock = MockLLMClient(default_score=9.5, default_roi=9.0)
        assert mock.provider_name == "mock"
        assert mock.model_name == "mock-evaluator-v1"

        res = await mock.generate_structured("Evaluate vLLM kernel", Stage2EvaluationResponse)
        assert res.breakthrough_score == 9.5
        assert res.roi_score == 9.0
        assert res.category == Stage2Category.CORE
        assert res.is_qualifying is True

        txt = await mock.generate_text("Hello scout")
        assert "Hello scout" in txt

    @pytest.mark.asyncio
    async def test_mock_llm_client_categories_heuristic(self) -> None:
        mock = MockLLMClient()

        # Wrapper heuristic -> discard
        res_wrap = await mock.generate_structured("A thin wrapper around OpenAI", Stage2EvaluationResponse)
        assert res_wrap.category == Stage2Category.DISCARD
        assert res_wrap.breakthrough_score < 5.0

        # Robotics heuristic -> serendipity
        res_rob = await mock.generate_structured("ROS2 autonomous robotics controller", Stage2EvaluationResponse)
        assert res_rob.category == Stage2Category.SERENDIPITY

    @pytest.mark.asyncio
    async def test_mock_llm_client_fault_injection(self) -> None:
        quota_mock = MockLLMClient(simulate_quota_exceeded=True)
        with pytest.raises(LLMQuotaExceededException):
            await quota_mock.generate_structured("prompt", Stage2EvaluationResponse)

        server_mock = MockLLMClient(simulate_server_error=True)
        with pytest.raises(LLMServerException):
            await server_mock.generate_text("prompt")

        json_mock = MockLLMClient(simulate_malformed_json=True)
        res_repaired = await json_mock.generate_structured("prompt", Stage2EvaluationResponse)
        assert res_repaired.breakthrough_score == 9.2


# ============================================================================
# 5. TestLLMFactory
# ============================================================================

class TestLLMFactory:
    """Tests dynamic client instantiation via settings."""

    def test_create_gemini_client(self) -> None:
        settings = Settings(llm_provider="gemini", gemini_api_key=SecretStr("gemini-key-123"))
        client = create_llm_client(settings)
        assert isinstance(client, GeminiLLMClient)
        assert client.provider_name == "gemini"

    def test_create_openai_compatible_client(self) -> None:
        settings = Settings(
            llm_provider="openai_compatible",
            llm_base_url="http://4090-box:8000/v1",
            llm_model="llama3.1:70b",
        )
        client = create_llm_client(settings)
        assert isinstance(client, OpenAICompatibleClient)
        assert client.provider_name == "openai_compatible"
        assert client.model_name == "llama3.1:70b"

    def test_create_fallback_glm_client(self) -> None:
        settings = Settings(
            llm_fallback_provider="openai_compatible",
            llm_fallback_base_url="https://open.bigmodel.cn/api/paas/v4",
            llm_fallback_model="glm-4-flash",
            llm_fallback_api_key=SecretStr("glm-key"),
        )
        fb_client = create_fallback_llm_client(settings)
        assert fb_client is not None
        assert isinstance(fb_client, OpenAICompatibleClient)
        assert fb_client.model_name == "glm-4-flash"

    def test_create_fallback_none(self) -> None:
        settings = Settings(llm_fallback_provider="none")
        fb_client = create_fallback_llm_client(settings)
        assert fb_client is None

    def test_create_mock_client(self) -> None:
        settings = Settings(llm_provider="mock")
        client = create_llm_client(settings)
        assert isinstance(client, MockLLMClient)

    def test_create_fallback_mock_client(self) -> None:
        settings = Settings(llm_fallback_provider="mock")
        fb_client = create_fallback_llm_client(settings)
        assert fb_client is not None
        assert isinstance(fb_client, MockLLMClient)


# ============================================================================
# 6. TestStage2EvaluatorLLMIntegration
# ============================================================================

class TestStage2EvaluatorLLMIntegration:
    """Tests Stage2Evaluator with injected BaseLLMClient and automatic fallback."""

    @pytest.mark.asyncio
    async def test_stage2_primary_success(
        self,
        migrated_db: Database,
        test_settings: Settings,
        sample_raw_item: RawItem,
        valid_stage2_json_dict: dict[str, Any],
    ) -> None:
        class MockPrimaryClient(BaseLLMClient):
            provider_name = "mock_primary"
            model_name = "mock-primary-model"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                return Stage2EvaluationResponse.model_validate(valid_stage2_json_dict)

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return True

            async def aclose(self) -> None:
                pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=MockPrimaryClient(),
        )
        res = await evaluator.evaluate_candidate(sample_raw_item)
        assert res.breakthrough_score == 9.3
        assert res.category == Stage2Category.CORE

    @pytest.mark.asyncio
    async def test_stage2_quota_exceeded_seamless_fallback(
        self,
        migrated_db: Database,
        test_settings: Settings,
        sample_raw_item: RawItem,
        valid_stage2_json_dict: dict[str, Any],
    ) -> None:
        class QuotaExceededPrimary(BaseLLMClient):
            provider_name = "gemini_exhausted"
            model_name = "gemini-3.8-flash"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Primary daily quota exhausted")

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return False

            async def aclose(self) -> None:
                pass

        fallback_called = False

        class WorkingFallback(BaseLLMClient):
            provider_name = "glm_backup"
            model_name = "glm-4-flash"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                nonlocal fallback_called
                fallback_called = True
                return Stage2EvaluationResponse.model_validate(valid_stage2_json_dict)

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return True

            async def aclose(self) -> None:
                pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=QuotaExceededPrimary(),
            fallback_llm_client=WorkingFallback(),
        )
        res = await evaluator.evaluate_candidate(sample_raw_item)
        assert fallback_called is True
        assert res.breakthrough_score == 9.3

    @pytest.mark.asyncio
    async def test_stage2_both_primary_and_fallback_fail(
        self,
        migrated_db: Database,
        test_settings: Settings,
        sample_raw_item: RawItem,
    ) -> None:
        class QuotaExceededPrimary(BaseLLMClient):
            provider_name = "primary"
            model_name = "m1"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMQuotaExceededException("Quota exceeded")

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return False

            async def aclose(self) -> None:
                pass

        class FailingFallback(BaseLLMClient):
            provider_name = "fallback"
            model_name = "m2"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                raise LLMServerException("Fallback server crashed")

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return False

            async def aclose(self) -> None:
                pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=QuotaExceededPrimary(),
            fallback_llm_client=FailingFallback(),
        )
        with pytest.raises(LLMServerException, match="Fallback server crashed"):
            await evaluator.evaluate_candidate(sample_raw_item)

    @pytest.mark.asyncio
    async def test_stage2_evaluate_and_persist_flow(
        self,
        migrated_db: Database,
        test_settings: Settings,
        sample_raw_item: RawItem,
        valid_stage2_json_dict: dict[str, Any],
    ) -> None:
        # Seed raw item in DB
        await migrated_db.execute(
            """INSERT INTO raw_items (id, source, source_id, title, url, raw_content)
            VALUES (?, ?, ?, ?, ?, ?);""",
            (
                sample_raw_item.id,
                sample_raw_item.source,
                sample_raw_item.source_id,
                sample_raw_item.title,
                sample_raw_item.url,
                sample_raw_item.raw_content,
            ),
        )

        class MockPrimary(BaseLLMClient):
            provider_name = "mock_primary"
            model_name = "model-1"

            async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
                return Stage2EvaluationResponse.model_validate(valid_stage2_json_dict)

            async def generate_text(self, *args: Any, **kwargs: Any) -> str:
                return ""

            async def health_check(self) -> bool:
                return True

            async def aclose(self) -> None:
                pass

        evaluator = Stage2Evaluator(
            db=migrated_db,
            settings=test_settings,
            llm_client=MockPrimary(),
        )

        breakthrough = await evaluator.evaluate_and_persist(sample_raw_item)
        assert breakthrough is not None
        assert breakthrough.title == sample_raw_item.title
        assert breakthrough.breakthrough_score == 9.3

        # Verify saved in SQLite DB
        row = await migrated_db.fetch_one(
            "SELECT title, breakthrough_score, category FROM breakthroughs WHERE raw_item_id = ?;",
            (sample_raw_item.id,),
        )
        assert row is not None
        assert row["title"] == sample_raw_item.title
        assert row["breakthrough_score"] == 9.3
        assert row["category"] == "core"
