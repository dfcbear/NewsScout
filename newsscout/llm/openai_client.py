"""newsscout.llm.openai_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Universal OpenAI-compatible client supporting vLLM, Ollama, llama.cpp,
Groq, OpenRouter, GLM 5.2, and OpenAI REST endpoints.
"""
from __future__ import annotations

import asyncio
import json
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

QUOTA_KEYWORDS = (
    "quota",
    "insufficient",
    "exceeded",
    "balance",
    "credit",
    "billing",
    "resource_exhausted",
)


class OpenAICompatibleClient(BaseLLMClient):
    """Universal client for any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        model: str = "llama3.1:8b",
        custom_headers: Optional[dict[str, str]] = None,
        temperature: float = 0.1,
        timeout_seconds: float = 45.0,
        max_retries: int = 4,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/chat/completions"):
            if not clean_url.endswith("/v1"):
                clean_url = f"{clean_url}/v1"
            self.endpoint_url = f"{clean_url}/chat/completions"
        else:
            self.endpoint_url = clean_url

        self.base_url = clean_url
        self.api_key = (api_key or "").strip()
        self._model = model
        self.custom_headers = custom_headers or {}
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._external_client = http_client
        self._client: Optional[httpx.AsyncClient] = http_client

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NewsScout/1.0 (Autonomous AI Scout; Pi 5)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.custom_headers)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds))
        return self._client

    async def aclose(self) -> None:
        if self._external_client is None and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Checks reachability of the models endpoint."""
        try:
            client = await self._get_client()
            models_url = self.endpoint_url.replace("/chat/completions", "/models")
            resp = await client.get(models_url, headers=self._get_headers(), timeout=5.0)
            return resp.status_code in (200, 401, 403)
        except Exception as exc:
            logger.warning("OpenAI-compatible health_check failed: %s: %s", type(exc).__name__, exc)
            return False

    async def generate_text(
        self,
        prompt: str = "",
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        *,
        user_prompt: Optional[str] = None,
    ) -> str:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": actual_prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
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
        schema_guidance = (
            f"\nYou must respond ONLY with valid JSON conforming strictly to this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
        full_system = f"{system_instruction or ''}\n{schema_guidance}".strip()

        messages = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.append({"role": "user", "content": actual_prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            raw_text = await self._execute_request(payload)
        except LLMServerException as exc:
            # If backend rejected response_format, fallback to plain text with prompt schema
            if "response_format" in str(exc).lower() or "400" in str(exc):
                logger.warning(
                    "[%s] Backend rejected response_format; retrying with plain prompt schema guidance",
                    self.provider_name,
                )
                payload.pop("response_format", None)
                raw_text = await self._execute_request(payload)
            else:
                raise

        parsed_dict = clean_and_parse_json(raw_text)
        try:
            return response_model.model_validate(parsed_dict)
        except ValidationError as val_err:
            logger.error(
                "[%s] Schema validation failed for model %s: %s",
                self.provider_name,
                self._model,
                val_err,
            )
            raise LLMResponseValidationError(f"Invalid schema from {self.provider_name}: {val_err}") from val_err

    async def _execute_request(self, payload: dict[str, Any]) -> str:
        client = await self._get_client()
        headers = self._get_headers()
        base_delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.post(self.endpoint_url, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMError(f"Empty choices returned from {self.provider_name}: {data}")
                    content = choices[0].get("message", {}).get("content", "")
                    if content is None or content == "":
                        raise LLMError(f"Empty message content from {self.provider_name}: {data}")
                    return content

                # Rate Limiting & Quota Limits (HTTP 429)
                if response.status_code == 429:
                    body_lower = response.text.lower()
                    if any(q in body_lower for q in QUOTA_KEYWORDS):
                        logger.error(
                            "[%s] Quota ceiling reached (HTTP 429): %s",
                            self.provider_name,
                            response.text[:200],
                        )
                        raise LLMQuotaExceededException(
                            f"Quota exceeded on {self.provider_name}: {response.text[:200]}"
                        )

                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 8.0)
                    )
                    logger.warning(
                        "[%s] Rate limit (HTTP 429, attempt %d/%d). Sleeping %.2fs",
                        self.provider_name,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    if attempt == self.max_retries:
                        raise LLMQuotaExceededException(
                            f"Rate limit exhausted after {self.max_retries} attempts: {response.text[:200]}"
                        )
                    await asyncio.sleep(delay)
                    continue

                # Server Errors (HTTP 500, 502, 503, 504)
                if response.status_code >= 500:
                    delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 8.0)
                    logger.warning(
                        "[%s] Server error HTTP %d (attempt %d/%d). Sleeping %.2fs",
                        self.provider_name,
                        response.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    if attempt == self.max_retries:
                        raise LLMServerException(
                            f"Server error HTTP {response.status_code} from {self.provider_name}: {response.text[:200]}"
                        )
                    await asyncio.sleep(delay)
                    continue

                # Unhandled 4xx errors
                raise LLMServerException(
                    f"HTTP {response.status_code} error from {self.provider_name}: {response.text[:300]}"
                )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                delay = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5), 8.0)
                logger.warning(
                    "[%s] Network/Timeout error: %s (attempt %d/%d). Sleeping %.2fs",
                    self.provider_name,
                    net_err,
                    attempt,
                    self.max_retries,
                    delay,
                )
                if attempt == self.max_retries:
                    raise LLMServerException(
                        f"Network error exhausted {self.max_retries} retries: {net_err}"
                    ) from net_err
                await asyncio.sleep(delay)

        raise LLMServerException(f"Exhausted all {self.max_retries} retry attempts calling {self.endpoint_url}")
