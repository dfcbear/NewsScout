"""newsscout.llm.factory
~~~~~~~~~~~~~~~~~~~~~~~~~~
Factory functions for instantiating primary and fallback LLM clients.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from newsscout.config import Settings, get_settings
from newsscout.llm.base import BaseLLMClient
from newsscout.llm.gemini_client import GeminiLLMClient
from newsscout.llm.mock import MockLLMClient
from newsscout.llm.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)


def create_llm_client(
    settings: Optional[Settings] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> BaseLLMClient:
    """Creates the primary LLM client configured in application settings.

    Args:
        settings: Application Settings (defaults to cached singleton).
        http_client: Optional pre-configured httpx.AsyncClient.

    Returns:
        BaseLLMClient instance.
    """
    cfg = settings or get_settings()
    provider = getattr(cfg, "llm_provider", "gemini").lower()

    if provider == "mock":
        return MockLLMClient()

    if provider == "openai_compatible":
        raw_key = getattr(cfg, "llm_api_key", "")
        api_key = raw_key.get_secret_value() if hasattr(raw_key, "get_secret_value") else str(raw_key or "")
        return OpenAICompatibleClient(
            base_url=getattr(cfg, "llm_base_url", "http://localhost:11434/v1"),
            api_key=api_key,
            model=getattr(cfg, "llm_model", "llama3.1:8b"),
            custom_headers=getattr(cfg, "llm_custom_headers", {}),
            temperature=getattr(cfg, "llm_temperature", 0.1),
            timeout_seconds=getattr(cfg, "llm_timeout_seconds", 45.0),
            http_client=http_client,
        )

    # Default to Gemini client
    raw_gemini_key = getattr(cfg, "gemini_api_key", "")
    gemini_key = (
        raw_gemini_key.get_secret_value()
        if hasattr(raw_gemini_key, "get_secret_value")
        else str(raw_gemini_key or "")
    )
    return GeminiLLMClient(
        api_key=gemini_key,
        model=getattr(cfg, "gemini_model", "gemini-3.8-flash"),
        temperature=getattr(cfg, "llm_temperature", 0.1),
        timeout_seconds=getattr(cfg, "llm_timeout_seconds", 45.0),
        http_client=http_client,
    )


def create_fallback_llm_client(
    settings: Optional[Settings] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Optional[BaseLLMClient]:
    """Creates the fallback LLM client configured in settings for quota failovers.

    Returns None if fallback is disabled ('none') or not configured.
    """
    cfg = settings or get_settings()
    fallback_provider = getattr(cfg, "llm_fallback_provider", "openai_compatible").lower()

    if fallback_provider in ("none", "", "null", "disabled"):
        return None

    if fallback_provider == "mock":
        return MockLLMClient(provider_name_override="mock_fallback")

    if fallback_provider == "openai_compatible":
        raw_fb_key = getattr(cfg, "llm_fallback_api_key", "")
        fb_api_key = (
            raw_fb_key.get_secret_value()
            if hasattr(raw_fb_key, "get_secret_value")
            else str(raw_fb_key or "")
        )
        return OpenAICompatibleClient(
            base_url=getattr(cfg, "llm_fallback_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            api_key=fb_api_key,
            model=getattr(cfg, "llm_fallback_model", "glm-4-flash"),
            custom_headers=getattr(cfg, "llm_custom_headers", {}),
            temperature=getattr(cfg, "llm_temperature", 0.1),
            timeout_seconds=getattr(cfg, "llm_timeout_seconds", 45.0),
            http_client=http_client,
        )

    if fallback_provider == "gemini":
        primary = getattr(cfg, "llm_provider", "").lower()
        if primary != "gemini":
            raw_gemini_key = getattr(cfg, "gemini_api_key", "")
            gemini_key = (
                raw_gemini_key.get_secret_value()
                if hasattr(raw_gemini_key, "get_secret_value")
                else str(raw_gemini_key or "")
            )
            return GeminiLLMClient(
                api_key=gemini_key,
                model=getattr(cfg, "gemini_model", "gemini-3.8-flash"),
                temperature=getattr(cfg, "llm_temperature", 0.1),
                timeout_seconds=getattr(cfg, "llm_timeout_seconds", 45.0),
                http_client=http_client,
            )

    return None
