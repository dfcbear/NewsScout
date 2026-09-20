"""tests.test_config_v2
~~~~~~~~~~~~~~~~~~~~
Unit tests for TASK-08 configuration extensions in Pydantic Settings v2:
- Universal LLM provider configuration & credentials check
- Multi-Search engine settings (SearXNG, DuckDuckGo, Tavily, Exa)
- Multi-Messenger gateway settings (Signal, DeliveryDispatcher)
- Environment variable aliases (NEWSSCOUT_*)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from newsscout.config import Settings


class TestLLMConfigV2:
    """Tests for universal LLM provider configuration."""

    def test_default_llm_settings(self):
        settings = Settings(
            gemini_api_key=SecretStr(""),
            llm_api_key=SecretStr(""),
        )
        assert settings.llm_provider == "gemini"
        assert settings.llm_base_url == "http://localhost:11434/v1"
        assert settings.llm_model == "llama3.1:8b"
        assert settings.llm_temperature == 0.1
        assert settings.llm_timeout_seconds == 45.0
        assert settings.llm_fallback_provider == "openai_compatible"
        assert settings.llm_fallback_model == "glm-4-flash"

    def test_has_llm_credentials_gemini_empty(self):
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key=SecretStr(""),
        )
        assert settings.has_llm_credentials is False

    def test_has_llm_credentials_gemini_present(self):
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key=SecretStr("AIzaSyTestKey123"),
        )
        assert settings.has_llm_credentials is True

    def test_has_llm_credentials_openai_compatible(self):
        settings = Settings(
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.1:8b",
        )
        assert settings.has_llm_credentials is True

    def test_has_llm_credentials_openai_compatible_custom_host(self):
        settings = Settings(
            llm_provider="openai_compatible",
            llm_base_url="http://192.168.1.100:8000/v1",
            llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            llm_api_key=SecretStr("custom-secret"),
        )
        assert settings.has_llm_credentials is True


class TestSearchConfigV2:
    """Tests for multi-search pipeline settings."""

    def test_default_search_settings(self):
        settings = Settings()
        assert settings.searxng_enabled is True
        assert settings.duckduckgo_enabled is True
        assert settings.searxng_base_url == "http://localhost:8080"
        assert settings.searxng_categories == "general,it"
        assert settings.search_timeout_seconds == 8.0
        assert settings.search_max_results_per_engine == 10
        assert settings.search_max_queries_per_cycle == 5

    def test_tavily_credentials_detection(self):
        s_without = Settings(tavily_api_key=SecretStr(""))
        assert s_without.has_tavily_credentials is False

        s_with = Settings(tavily_api_key=SecretStr("tvly-test-12345"))
        assert s_with.has_tavily_credentials is True

    def test_exa_credentials_detection(self):
        s_without = Settings(exa_api_key=SecretStr(""))
        assert s_without.has_exa_credentials is False

        s_with = Settings(exa_api_key=SecretStr("exa-test-67890"))
        assert s_with.has_exa_credentials is True


class TestMessengerConfigV2:
    """Tests for Signal gateway configuration."""

    def test_default_messenger_settings(self):
        settings = Settings()
        assert settings.delivery_channels == ["telegram"]
        assert settings.delivery_timeout_seconds == 15.0
        assert settings.signal_enabled is False
        assert settings.has_signal_credentials is False

    def test_signal_effective_recipients_and_credentials(self):
        settings = Settings(
            signal_enabled=True,
            signal_bridge_url="http://signal-bridge:8080",
            signal_sender_number="+491701234567",
            signal_recipient_id="+491709876543",
        )
        assert len(settings.effective_signal_recipients) == 1
        assert settings.effective_signal_recipients[0] == "+491709876543"
        assert settings.has_signal_credentials is True

    def test_signal_disabled_ignores_credentials(self):
        settings = Settings(
            signal_enabled=False,
            signal_bridge_url="http://signal-bridge:8080",
            signal_sender_number="+491701234567",
            signal_recipient_id="+491709876543",
        )
        assert settings.has_signal_credentials is False


class TestEnvAliasesV2:
    """Tests for environment variable aliases with NEWSSCOUT_ prefixes."""

    def test_newsscout_env_aliases(self):
        env_vars = {
            "NEWSSCOUT_LLM_PROVIDER": "openai_compatible",
            "NEWSSCOUT_LLM_BASE_URL": "http://gpu-box:8000/v1",
            "NEWSSCOUT_LLM_MODEL": "mistral-7b-instruct",
            "NEWSSCOUT_SIGNAL_ENABLED": "true",
            "NEWSSCOUT_SIGNAL_SENDER_NUMBER": "+49111222333",
            "NEWSSCOUT_SEARXNG_BASE_URL": "http://mysearx:8080",
        }
        with patch.dict(os.environ, env_vars):
            s = Settings()
            assert s.llm_provider == "openai_compatible"
            assert s.llm_base_url == "http://gpu-box:8000/v1"
            assert s.llm_model == "mistral-7b-instruct"
            assert s.signal_enabled is True
            assert s.signal_sender_number == "+49111222333"
            assert s.searxng_base_url == "http://mysearx:8080"
