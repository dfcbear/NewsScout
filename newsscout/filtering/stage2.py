"""newsscout.filtering.stage2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stage 2 Deep LLM Evaluator powered by Gemini 3.8 Flash.
Evaluates novelty, ROI, generates 1-minute decision cards, and persists breakthroughs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import random
import re
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from newsscout.config import Settings, get_settings
from newsscout.filtering.card_gen import DecisionCardGenerator
from newsscout.filtering.prompts import (
    STAGE2_JSON_SCHEMA,
    SYSTEM_INSTRUCTION,
    build_evaluation_user_prompt,
)
from newsscout.storage.db import Database
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    RawItem,
    Stage1Evaluation,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService
from newsscout.llm.base import BaseLLMClient, LLMQuotaExceededException
from newsscout.llm.factory import create_fallback_llm_client, create_llm_client
from newsscout.llm.gemini_client import GeminiLLMClient
from newsscout.llm.json_repair import clean_and_parse_json
from newsscout.llm.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Response Validation Models
# ============================================================================

class DecisionCardPayload(BaseModel):
    """Payload representing decision card fields within Gemini response."""

    model_config = ConfigDict(from_attributes=True)

    tldr: str = Field(description="Exactly 1 sentence.")
    use_case: str = Field(description="Workflow application.")
    comparison: str = Field(description="Baseline comparison.")
    quickstart: str = Field(description="Copy-pasteable 1-liner.")
    hardware_requirements: str = Field(description="Hardware specs.")
    license: str = Field(description="License status.")


class Stage2EvaluationResponse(BaseModel):
    """Pydantic validation schema for Gemini 3.8 Flash structured JSON output."""

    model_config = ConfigDict(from_attributes=True)

    breakthrough_score: float = Field(ge=1.0, le=10.0)
    roi_score: float = Field(ge=1.0, le=10.0)
    category: Stage2Category
    relevance_justification: str
    decision_card: DecisionCardPayload

    @property
    def is_qualifying(self) -> bool:
        """Determines if the candidate meets the threshold to be stored as a breakthrough."""
        if self.category == Stage2Category.DISCARD:
            return False
        if self.category == Stage2Category.CORE:
            return self.breakthrough_score >= 7.0 and self.roi_score >= 7.0
        if self.category == Stage2Category.SERENDIPITY:
            # E11: High novelty out-of-bubble discovery allowed with modest ROI
            return (self.breakthrough_score >= 7.0 and self.roi_score >= 7.0) or (
                self.breakthrough_score >= 8.5 and self.roi_score >= 6.0
            )
        return False


# ============================================================================
# Universal LLM Evaluator with Automatic Quota Fallback
# ============================================================================

class Stage2Evaluator:
    """Evaluates candidates using Universal LLM Providers with structured JSON output,
    automatic model fallback upon quota exhaustion, few-shot prompt calibration,
    and SQLite persistence.
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        db: Database,
        settings: Optional[Settings] = None,
        preferences_service: Optional[PreferencesService] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        mock_client: Optional[Any] = None,
        card_generator: Optional[DecisionCardGenerator] = None,
        llm_client: Optional[BaseLLMClient] = None,
        fallback_llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.preferences_service = preferences_service or PreferencesService(db, self.settings)
        self._http_client = http_client
        self._mock_client = mock_client
        self.card_generator = card_generator or DecisionCardGenerator()

        # Check mock test indicators
        if self._mock_client is not None:
            self._is_test_env = True
        elif getattr(self.settings, "llm_provider", "gemini") == "gemini":
            api_key = self.settings.gemini_api_key.get_secret_value().strip().lower()
            self._is_test_env = (
                not self.settings.has_gemini_credentials
                or "mock" in api_key
                or "test" in api_key
                or "placeholder" in api_key
                or api_key in ("", "none", "null")
            )
        else:
            self._is_test_env = False

        # Primary LLM Client Resolution
        if llm_client is not None:
            self.llm_client: Optional[BaseLLMClient] = llm_client
        elif not self._is_test_env:
            try:
                self.llm_client = create_llm_client(self.settings, http_client=self._http_client)
            except Exception as err:
                logger.debug("LLM factory initialization deferred: %s", err)
                self.llm_client = None
        else:
            self.llm_client = None

        # Fallback LLM Client Resolution
        if fallback_llm_client is not None:
            self.fallback_llm_client: Optional[BaseLLMClient] = fallback_llm_client
        elif not self._is_test_env:
            try:
                self.fallback_llm_client = create_fallback_llm_client(self.settings, http_client=self._http_client)
            except Exception as err:
                logger.debug("Fallback LLM initialization deferred: %s", err)
                self.fallback_llm_client = None
        else:
            self.fallback_llm_client = None

    async def evaluate_candidate(
        self,
        raw_item: RawItem,
        stage1_eval: Optional[Stage1Evaluation] = None,
    ) -> Stage2EvaluationResponse:
        """Runs Stage 2 evaluation for a single RawItem."""
        # 1. Backward-Compatible Mock Routing
        use_legacy_mock = (
            self.llm_client is None
            and (
                self._mock_client is not None
                or not getattr(self.settings, "has_llm_credentials", self.settings.has_gemini_credentials)
                or self._is_test_env
            )
        )
        if use_legacy_mock:
            if self._mock_client is not None and hasattr(self._mock_client, "evaluate_candidate"):
                mock_res = await self._mock_client.evaluate_candidate(
                    raw_item.title, raw_item.raw_content or ""
                )
                if isinstance(mock_res, Stage2EvaluationResponse):
                    return mock_res
                card_dict = mock_res.get("card", {})
                hw_lic = card_dict.get("hardware_and_license", "24GB VRAM | Apache-2.0")
                hw = hw_lic.split("|")[0].strip() if "|" in hw_lic else hw_lic
                lic = hw_lic.split("|")[-1].strip() if "|" in hw_lic else "Apache-2.0"
                return Stage2EvaluationResponse(
                    breakthrough_score=float(mock_res.get("breakthrough_score", 8.0)),
                    roi_score=float(mock_res.get("roi_score", 7.5)),
                    category=Stage2Category(mock_res.get("category", "core")),
                    relevance_justification=mock_res.get("rationale", "Mock evaluation"),
                    decision_card=DecisionCardPayload(
                        tldr=card_dict.get("tldr", "TLDR"),
                        use_case=card_dict.get("use_case", "Use case"),
                        comparison=card_dict.get("comparison", "Comparison"),
                        quickstart=card_dict.get("quickstart", "docker run ..."),
                        hardware_requirements=card_dict.get("hardware_requirements", hw),
                        license=card_dict.get("license", lic),
                    ),
                )
            return await self._evaluate_with_mock(raw_item)

        # 2. Fetch Few-Shot Exemplars
        exemplars = await self.preferences_service.get_few_shot_exemplars(
            count=self.settings.few_shot_exemplars_count
        )

        # 3. Render Prompt
        user_prompt = build_evaluation_user_prompt(
            raw_item=raw_item,
            stage1_eval=stage1_eval,
            exemplars=exemplars,
        )

        # 4. Resolve Active LLM Client
        active_client = self.llm_client or create_llm_client(self.settings, http_client=self._http_client)

        # 5. Primary LLM Execution with Automatic Fallback
        try:
            return await active_client.generate_structured(
                system_instruction=SYSTEM_INSTRUCTION,
                user_prompt=user_prompt,
                response_model=Stage2EvaluationResponse,
                json_schema=STAGE2_JSON_SCHEMA,
            )
        except LLMQuotaExceededException as quota_err:
            if self.fallback_llm_client is not None:
                logger.warning(
                    "[Stage2] Primary LLM '%s' quota exceeded: %s. Failing over to fallback LLM '%s'...",
                    active_client.provider_name,
                    quota_err,
                    self.fallback_llm_client.provider_name,
                )
                try:
                    return await self.fallback_llm_client.generate_structured(
                        system_instruction=SYSTEM_INSTRUCTION,
                        user_prompt=user_prompt,
                        response_model=Stage2EvaluationResponse,
                        json_schema=STAGE2_JSON_SCHEMA,
                    )
                except Exception as fb_err:
                    logger.error(
                        "[Stage2] Fallback LLM '%s' also failed: %s",
                        self.fallback_llm_client.provider_name,
                        fb_err,
                    )
                    raise
            logger.error("[Stage2] Primary LLM quota exceeded and no fallback client configured.")
            raise

    async def _call_gemini_with_backoff(
        self,
        payload: dict[str, Any],
        max_attempts: int = 4,
        base_delay: float = 1.0,
    ) -> str:
        """Sends HTTP request to Gemini API with exponential backoff on HTTP 429 / 503."""
        model = self.settings.gemini_model
        api_key = self.settings.gemini_api_key.get_secret_value()
        url = self.GEMINI_API_URL.format(model=model)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        client = self._http_client or httpx.AsyncClient(timeout=45.0)
        close_client = self._http_client is None

        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise ValueError(f"Gemini returned empty candidates: {data}")
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if not content_parts:
                            raise ValueError(f"Gemini returned empty parts: {data}")
                        return content_parts[0].get("text", "")

                    if response.status_code in (429, 503):
                        retry_after = response.headers.get("Retry-After")
                        delay = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else (base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5))
                        )
                        delay = min(delay, 5.0)
                        logger.warning(
                            "Gemini API rate limit / unavailable (HTTP %d, attempt %d/%d). Retrying in %.2fs...",
                            response.status_code,
                            attempt,
                            max_attempts,
                            delay,
                        )
                        if attempt == max_attempts:
                            response.raise_for_status()
                        await asyncio.sleep(delay)
                        continue

                    response.raise_for_status()

                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 5.0)
                    logger.warning(
                        "Gemini network/timeout error (attempt %d/%d): %s. Retrying in %.2fs...",
                        attempt,
                        max_attempts,
                        net_err,
                        delay,
                    )
                    if attempt == max_attempts:
                        raise
                    await asyncio.sleep(delay)

            raise RuntimeError("Exhausted all retry attempts calling Gemini API")
        finally:
            if close_client:
                await client.aclose()

    async def _evaluate_with_mock(self, raw_item: RawItem) -> Stage2EvaluationResponse:
        """Deterministic evaluation fallback when running in mock/offline mode."""
        title_lower = raw_item.title.lower()
        content_lower = (raw_item.raw_content or "").lower()

        # Hard rejection checks
        is_wrapper = any(
            w in title_lower or w in content_lower
            for w in ["chatgpt wrapper", "landing page", "seo generator", "shopify", "crypto", "memecoin"]
        )
        if is_wrapper:
            return Stage2EvaluationResponse(
                breakthrough_score=2.2,
                roi_score=1.5,
                category=Stage2Category.DISCARD,
                relevance_justification=f"Deterministic drop: SaaS wrapper or marketing tool detected in '{raw_item.title}'.",
                decision_card=DecisionCardPayload(
                    tldr="Discarded: Low-effort wrapper or marketing tool lacking local code.",
                    use_case="None",
                    comparison="Inferior to open baselines.",
                    quickstart="N/A",
                    hardware_requirements="N/A",
                    license="N/A",
                ),
            )

        # Check Core vs Serendipity keywords
        is_core = any(
            term in title_lower or term in content_lower
            for term in [
                "vllm", "llama.cpp", "rag", "qdrant", "docling", "agent", "mcp",
                "kernel", "nvfp4", "awq", "exl2", "esp32", "inference", "quantization",
            ]
        )
        is_serendipity = any(
            term in title_lower or term in content_lower
            for term in ["robotics", "ros2", "vla", "neuromorphic", "mamba", "titans", "sensor", "humanoid"]
        )

        if is_core:
            category = Stage2Category.CORE
            score = 9.2
            roi = 9.0
            justification = f"High-signal core breakthrough: accelerates local inference/agent workflows for '{raw_item.title}'."
        elif is_serendipity:
            category = Stage2Category.SERENDIPITY
            score = 8.8
            roi = 7.8
            justification = f"Serendipity hit: novel out-of-bubble architecture/robotics advance in '{raw_item.title}'."
        else:
            category = Stage2Category.CORE
            score = 7.4
            roi = 7.2
            justification = f"Solid local tool meeting user stack criteria: '{raw_item.title}'."

        return Stage2EvaluationResponse(
            breakthrough_score=score,
            roi_score=roi,
            category=category,
            relevance_justification=justification,
            decision_card=DecisionCardPayload(
                tldr=f"Enables 3x faster local execution and seamless workflow integration for {raw_item.title}.",
                use_case="Direct integration with RTX 4090 local agent harnesses and offline vector stores.",
                comparison="3.2x faster inference and 40% lower memory footprint than previous standard baselines.",
                quickstart=f"docker run --gpus all -p 8080:8080 {raw_item.source_id.replace(':', '/')}:latest",
                hardware_requirements="NVIDIA RTX 4090 (24GB VRAM) | CUDA 12.4",
                license="Apache-2.0 (Permissive Open Source)",
            ),
        )

    async def evaluate_and_persist(
        self,
        raw_item: RawItem,
        stage1_eval: Optional[Stage1Evaluation] = None,
    ) -> Optional[Breakthrough]:
        if raw_item.id is None:
            raise ValueError("Cannot evaluate unpersisted raw_item")

        response = await self.evaluate_candidate(raw_item, stage1_eval)

        is_discard = (not response.is_qualifying) or (response.category == Stage2Category.DISCARD)
        category = Stage2Category.DISCARD if is_discard else response.category

        # Build DecisionCardData & format Markdown using DecisionCardGenerator
        card_data = self.card_generator.generate_card_data(
            response.decision_card, category=category
        )

        decision_card = DecisionCard(
            title=raw_item.title,
            repo_url=raw_item.url,
            breakthrough_score=response.breakthrough_score,
            roi_score=response.roi_score,
            category=category,
            data=card_data,
        )
        card_markdown = self.card_generator.render_markdown(decision_card)

        # Construct Breakthrough entity
        breakthrough = Breakthrough(
            raw_item_id=raw_item.id,
            title=raw_item.title,
            repo_url=raw_item.url,
            breakthrough_score=response.breakthrough_score,
            roi_score=response.roi_score,
            category=category,
            tldr=card_data.tldr,
            use_case=card_data.use_case,
            comparison=card_data.comparison,
            quickstart=card_data.quickstart,
            hardware_requirements=card_data.hardware_requirements,
            license=card_data.license,
            card_markdown=card_markdown,
            evaluation_raw=response.model_dump(),
            is_watchlisted=False,
            evaluated_at=datetime.now(timezone.utc),
        )

        # Atomic SQLite UPSERT
        query = """
        INSERT INTO breakthroughs (
            raw_item_id, title, repo_url, breakthrough_score, roi_score, category,
            tldr, use_case, comparison, quickstart, hardware_requirements, license,
            card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at
        ) VALUES (
            :raw_item_id, :title, :repo_url, :breakthrough_score, :roi_score, :category,
            :tldr, :use_case, :comparison, :quickstart, :hardware_requirements, :license,
            :card_markdown, :evaluation_raw_json, :is_watchlisted, :evaluated_at
        )
        ON CONFLICT(raw_item_id) DO UPDATE SET
            title = excluded.title,
            repo_url = excluded.repo_url,
            breakthrough_score = excluded.breakthrough_score,
            roi_score = excluded.roi_score,
            category = excluded.category,
            tldr = excluded.tldr,
            use_case = excluded.use_case,
            comparison = excluded.comparison,
            quickstart = excluded.quickstart,
            hardware_requirements = excluded.hardware_requirements,
            license = excluded.license,
            card_markdown = excluded.card_markdown,
            evaluation_raw_json = excluded.evaluation_raw_json,
            evaluated_at = excluded.evaluated_at;
        """
        params = breakthrough.to_db_params()
        await self.db.execute(query, params)

        if is_discard:
            logger.info(
                "Candidate %s ('%s') dropped by Stage 2 (category=discard, score=%.1f) and recorded as discard",
                str(raw_item.id),
                raw_item.title,
                response.breakthrough_score,
            )
            return None

        # Fetch persisted row with autoincremented ID
        row = await self.db.fetch_one(
            "SELECT * FROM breakthroughs WHERE raw_item_id = ? ORDER BY id DESC LIMIT 1;",
            (raw_item.id,),
        )
        if row:
            return Breakthrough.from_row(row)
        return breakthrough

    async def evaluate_pending_items(
        self,
        batch_size: int = 25,
        concurrency: int = 3,
    ) -> list[Breakthrough]:
        """Queries for items passed by Stage 1 that lack a Stage 2 evaluation,
        and evaluates them concurrently with rate limiting.
        """
        query = """
        SELECT r.*, s1.id as s1_id, s1.passed as s1_passed, s1.detected_license,
               s1.has_docker, s1.has_runnable_code, s1.heuristics_json, s1.evaluated_at as s1_evaluated_at
        FROM raw_items r
        JOIN stage1_evaluations s1 ON r.id = s1.raw_item_id
        LEFT JOIN breakthroughs b ON r.id = b.raw_item_id
        WHERE s1.passed = 1 AND b.id IS NULL
        ORDER BY r.ingested_at DESC
        LIMIT ?;
        """
        rows = await self.db.fetch_all(query, (batch_size,))
        if not rows:
            return []

        sem = asyncio.Semaphore(concurrency)
        results: list[Breakthrough] = []

        async def _process_single(r: Any) -> Optional[Breakthrough]:
            async with sem:
                raw_item = RawItem.from_row(r)
                stage1_eval = Stage1Evaluation(
                    id=r["s1_id"],
                    raw_item_id=raw_item.id or 0,
                    passed=bool(r["s1_passed"]),
                    detected_license=r["detected_license"],
                    has_docker=bool(r["has_docker"]),
                    has_runnable_code=bool(r["has_runnable_code"]),
                    heuristics_json=r["heuristics_json"],
                    evaluated_at=datetime.fromisoformat(r["s1_evaluated_at"]) if r["s1_evaluated_at"] else datetime.now(timezone.utc),
                )
                try:
                    return await self.evaluate_and_persist(raw_item, stage1_eval)
                except Exception as exc:
                    logger.error("Stage 2 evaluation failed for item %s ('%s'): %s", str(raw_item.id), raw_item.title, exc)
                    return None

        tasks = [_process_single(r) for r in rows]
        outcomes = await asyncio.gather(*tasks)
        for bt in outcomes:
            if bt is not None:
                results.append(bt)

        logger.info("Stage 2 processed %d pending items, yielded %d breakthroughs.", len(rows), len(results))
        return results
