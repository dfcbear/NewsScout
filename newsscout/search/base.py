"""newsscout.search.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base classes, standardized SearchResult data contracts,
and exception hierarchy for NewsScout's multi-engine search pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import inspect
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_TIMEOUT: float = 8.0


@dataclass
class SearchResult:
    """Standardized search result emitted by any BaseSearchProvider."""

    title: str
    url: str
    snippet: str
    source_engine: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_score: Optional[float] = None
    published_date: Optional[str] = None

    def __post_init__(self) -> None:
        # Fallback if engine was set in metadata or passed
        if not self.source_engine and "engine" in self.metadata:
            self.source_engine = str(self.metadata["engine"])

    @property
    def engine(self) -> str:
        """Alias for source_engine for backwards compatibility."""
        return self.source_engine

    @property
    def score(self) -> Optional[float]:
        """Alias for raw_score for backwards compatibility."""
        return self.raw_score

    def to_dict(self) -> dict[str, Any]:
        """Serializes SearchResult to a dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_engine": self.source_engine,
            "engine": self.source_engine,
            "metadata": self.metadata,
            "raw_score": self.raw_score,
            "score": self.raw_score,
            "published_date": self.published_date,
        }


# ============================================================================
# Search Exception Hierarchy
# ============================================================================

class SearchError(Exception):
    """Base exception for all search provider operations."""


class SearchTimeoutError(SearchError):
    """Raised when an engine request exceeds its allotted timeout."""


class SearchRateLimitError(SearchError):
    """Raised when an engine returns HTTP 429 or rate-limiting challenge."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SearchUpstreamError(SearchError):
    """Raised on 5xx errors, connection drops, or upstream engine failure."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SearchUnavailableError(SearchError):
    """Raised when an unconfigured or disabled provider is queried."""


class SearchParseError(SearchError):
    """Raised when response payload is malformed or unparseable."""


class SearchProviderError(SearchError):
    """General error raised during provider operations."""


# ============================================================================
# BaseSearchProvider Abstract Contract
# ============================================================================

class BaseSearchProvider(abc.ABC):
    """Abstract interface for all NewsScout search engine providers."""

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = DEFAULT_SEARCH_TIMEOUT,
    ) -> None:
        self._client = client
        self._owns_client = False
        self.timeout = timeout

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique lowercase provider identifier (e.g. 'searxng', 'duckduckgo', 'tavily', 'exa')."""
        ...

    @property
    def requires_api_key(self) -> bool:
        """Indicates whether this provider requires a commercial API key."""
        return False

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Returns True if the provider is enabled and properly configured."""
        ...

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
        """Executes a search query and returns up to `limit` SearchResult items."""
        ...

    def _get_client(self) -> httpx.AsyncClient:
        """Returns the active client or lazily instantiates an internal one."""
        if self._client is not None and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=3.0),
            follow_redirects=True,
        )
        self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Releases client resources if owned by this provider instance."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self._owns_client = False

    async def aclose(self) -> None:
        """Async close alias."""
        await self.close()

    async def __aenter__(self) -> BaseSearchProvider:
        self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
