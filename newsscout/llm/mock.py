"""newsscout.llm.mock
~~~~~~~~~~~~~~~~~~~~~~
Deterministic Mock LLM client with fault-injection controls for offline testing.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel

from newsscout.llm.base import (
    BaseLLMClient,
    LLMQuotaExceededException,
    LLMServerException,
    T,
)
from newsscout.llm.json_repair import clean_and_parse_json


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM client implementing BaseLLMClient."""

    def __init__(
        self,
        simulate_quota_exceeded: bool = False,
        simulate_server_error: bool = False,
        simulate_malformed_json: bool = False,
        default_score: float = 9.2,
        default_roi: float = 9.0,
        provider_name_override: str = "mock",
    ) -> None:
        self.simulate_quota_exceeded = simulate_quota_exceeded
        self.simulate_server_error = simulate_server_error
        self.simulate_malformed_json = simulate_malformed_json
        self.default_score = default_score
        self.default_roi = default_roi
        self._provider_name = provider_name_override
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return "mock-evaluator-v1"

    async def generate_text(
        self,
        prompt: str = "",
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        *,
        user_prompt: Optional[str] = None,
    ) -> str:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        self.call_count += 1
        if self.simulate_quota_exceeded:
            raise LLMQuotaExceededException("Simulated Mock LLM quota exceeded")
        if self.simulate_server_error:
            raise LLMServerException("Simulated Mock LLM 500 error")
        return f"Mock response for prompt: {actual_prompt[:50]}"

    async def generate_structured(
        self,
        prompt: str = "",
        response_model: type[T] = BaseModel,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        json_schema: Optional[dict[str, Any]] = None,
        *,
        user_prompt: Optional[str] = None,
    ) -> T:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        self.call_count += 1

        if self.simulate_quota_exceeded:
            raise LLMQuotaExceededException("Simulated Mock LLM quota limit reached (HTTP 429)")

        if self.simulate_server_error:
            raise LLMServerException("Simulated Mock LLM 503 unavailable")

        prompt_lower = actual_prompt.lower()
        is_wrapper = any(w in prompt_lower for w in ["wrapper", "landing page", "seo generator"])
        is_serendipity = any(s in prompt_lower for s in ["robotics", "ros2", "vla", "sensor", "neuromorphic"])

        if is_wrapper:
            category = "discard"
            score = 2.5
            roi = 1.8
        elif is_serendipity:
            category = "serendipity"
            score = 8.8
            roi = 7.5
        else:
            category = "core"
            score = self.default_score
            roi = self.default_roi

        raw_dict = {
            "breakthrough_score": score,
            "roi_score": roi,
            "category": category,
            "relevance_justification": f"Mock evaluation for {actual_prompt[:40]}.",
            "decision_card": {
                "tldr": "Accelerates local inference on RTX 4090 by 3.2x.",
                "use_case": "Direct drop-in for local autonomous agent pipelines.",
                "comparison": "3.2x faster than previous baseline architectures.",
                "quickstart": "docker run --gpus all -p 8080:8080 local/tool:latest",
                "hardware_requirements": "NVIDIA RTX 4090 (24GB VRAM) | CUDA 12.4",
                "license": "Apache-2.0 (Permissive Open Source)",
            },
        }

        if self.simulate_malformed_json:
            malformed_text = f"```json\n{json.dumps(raw_dict)},,\n```\nHope that helps!"
            parsed = clean_and_parse_json(malformed_text)
            return response_model.model_validate(parsed)

        return response_model.model_validate(raw_dict)
