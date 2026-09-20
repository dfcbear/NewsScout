"""newsscout.llm
~~~~~~~~~~~~~~~~~
Universal LLM provider abstraction, structured JSON extraction, and fallback engine.
"""
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
from newsscout.llm.json_repair import clean_and_parse_json
from newsscout.llm.mock import MockLLMClient
from newsscout.llm.openai_client import OpenAICompatibleClient

__all__ = [
    "BaseLLMClient",
    "LLMError",
    "LLMException",
    "LLMQuotaExceededException",
    "LLMServerException",
    "LLMResponseValidationError",
    "OpenAICompatibleClient",
    "GeminiLLMClient",
    "MockLLMClient",
    "clean_and_parse_json",
    "create_llm_client",
    "create_fallback_llm_client",
]
