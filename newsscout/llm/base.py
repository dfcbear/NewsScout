"""newsscout.llm.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base interfaces, protocols, and standard exception taxonomy for LLM clients.
"""
from __future__ import annotations

import abc
from typing import Any, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Exception Hierarchy
# ============================================================================

class LLMError(Exception):
    """Base exception for all NewsScout LLM provider errors."""


# Alias for backward compatibility
LLMException = LLMError


class LLMQuotaExceededException(LLMError):
    """Raised when an LLM provider hits rate limits (429) or billing/quota limits.
    
    This specific exception triggers automatic failover to the configured fallback
    model (e.g. GLM 5.2 or local vLLM) in compliance with the Model Fallback Policy.
    """


class LLMServerException(LLMError):
    """Raised when an LLM provider returns 5xx errors or network drops after retries."""


class LLMResponseValidationError(LLMError):
    """Raised when LLM output cannot be parsed into JSON or fails Pydantic schema validation."""


# ============================================================================
# Base Client Interface
# ============================================================================

class BaseLLMClient(abc.ABC):
    """Abstract base class for all NewsScout LLM clients."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Identifier of the LLM provider (e.g. 'gemini', 'openai_compatible', 'mock')."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Configured model name (e.g. 'gemini-3.8-flash', 'llama3.1:8b', 'glm-4-flash')."""

    @abc.abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> T:
        """Generates structured output conforming strictly to the given Pydantic response_model.

        Args:
            prompt: Main user prompt describing the item to evaluate.
            response_model: Pydantic model class defining the desired schema.
            system_instruction: Optional system instruction / persona setting.
            temperature: Sampling temperature (default: 0.2 for deterministic output).
            json_schema: Optional precomputed JSON schema dictionary.

        Returns:
            Validated instance of response_model.

        Raises:
            LLMQuotaExceededException: If rate limits or quota ceilings are exhausted.
            LLMServerException: If server/network errors persist after all retries.
            LLMResponseValidationError: If output cannot be validated against response_model.
        """

    @abc.abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generates raw text response from the LLM.

        Args:
            prompt: Main user prompt.
            system_instruction: Optional system instruction.
            temperature: Sampling temperature.

        Returns:
            Raw generated string content.
        """

    async def health_check(self) -> bool:
        """Verifies reachability and authentication of the provider endpoint."""
        return True

    async def aclose(self) -> None:
        """Releases underlying HTTP connection pools and resources."""

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()
