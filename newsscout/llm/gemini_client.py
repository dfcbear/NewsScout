"""newsscout.llm.gemini_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Native Google Gemini REST driver implementing BaseLLMClient.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ValidationError

from newsscout.llm.base import (
    BaseLLMClient,
    LLMError,
    LLMQuotaExceededException,
    LLMResponseValidationError,
    LLMServerException,
    T,
)
from newsscout.llm.json_repair import clean_and_parse_json

logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini driver utilizing direct REST API with native responseSchema."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.1,
        timeout_seconds: float = 45.0,
        max_retries: int = 4,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self._model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.endpoint_url = GEMINI_API_URL.format(model=self._model)
        self._external_client = http_client
        self._client: Optional[httpx.AsyncClient] = http_client

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        return self._client

    async def aclose(self) -> None:
        if self._external_client is None and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        return bool(self.api_key)

    async def generate_text(
        self,
        prompt: str = "",
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        *,
        user_prompt: Optional[str] = None,
    ) -> str:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": actual_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }
        return await self._execute_request(payload)

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
        schema = json_schema or response_model.model_json_schema()
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": actual_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}],
            }

        raw_json_str = await self._execute_request(payload)
        parsed_data = clean_and_parse_json(raw_json_str)

        try:
            return response_model.model_validate(parsed_data)
        except ValidationError as val_err:
            logger.error("[%s] Schema validation error: %s", self.provider_name, val_err)
            raise LLMResponseValidationError(f"Invalid Gemini response structure: {val_err}") from val_err

    async def _execute_request(self, payload: dict[str, Any]) -> str:
        client = await self._get_client()
        headers = self._get_headers()
        base_delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.post(self.endpoint_url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise LLMError(f"Gemini returned empty candidates: {data}")
                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if not content_parts:
                        raise LLMError(f"Gemini returned empty parts: {data}")
                    return content_parts[0].get("text", "")

                # Rate Limiting & Quota Limits (HTTP 429 / 503)
                if response.status_code in (429, 503):
                    body_lower = response.text.lower()
                    if "quota" in body_lower or "resource_exhausted" in body_lower or attempt == self.max_retries:
                        logger.error(
                            "[%s] Quota ceiling reached / retries exhausted (HTTP %d): %s",
                            self.provider_name,
                            response.status_code,
                            response.text[:200],
                        )
                        raise LLMQuotaExceededException(
                            f"Quota exceeded on {self.provider_name}: {response.text[:200]}"
                        )

                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 5.0)
                    )
                    logger.warning(
                        "[%s] Rate limit hit (HTTP %d, attempt %d/%d). Sleeping %.2fs",
                        self.provider_name,
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 500:
                    delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 5.0)
                    logger.warning(
                        "[%s] Server error HTTP %d (attempt %d/%d). Sleeping %.2fs",
                        self.provider_name,
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    if attempt == self.max_retries:
                        raise LLMServerException(f"Gemini HTTP {response.status_code}: {response.text[:200]}")
                    await asyncio.sleep(delay)
                    continue

                raise LLMServerException(
                    f"Gemini HTTP {response.status_code} error: {response.text[:300]}"
                )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 5.0)
                logger.warning(
                    "[%s] Network error: %s (attempt %d/%d). Sleeping %.2fs",
                    self.provider_name,
                    net_err,
                    attempt,
                    self.max_retries,
                    delay,
                )
                if attempt == self.max_retries:
                    raise LLMServerException(f"Gemini network error exhausted retries: {net_err}") from net_err
                await asyncio.sleep(delay)

        raise LLMServerException(f"Exhausted all {self.max_retries} retry attempts calling {self.endpoint_url}")
