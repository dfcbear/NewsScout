"""newsscout.ingestion.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class and transport foundation for NewsScout ingestion engines.
Provides async context management, connection pooling, rate-limit backoff,
and normalized RawItem generator interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
import logging
import random
import time
from typing import Any, Final, Optional

import httpx

from newsscout.config import Settings, get_settings
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT: Final[str] = (
    "NewsScout/1.0 (+https://github.com/newsscout; autonomous-ai-scout)"
)
DEFAULT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(15.0, connect=5.0, write=5.0, pool=5.0)
DEFAULT_LIMITS: Final[httpx.Limits] = httpx.Limits(
    max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0
)


class BaseIngestionSource(ABC):
    """Abstract base class for all NewsScout data ingestion engines."""

    source_name: str = "base"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or {}
        self._external_client = client
        self._client: Optional[httpx.AsyncClient] = client
        self._is_closed = False

    async def __aenter__(self) -> BaseIngestionSource:
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "application/json",
            }
            headers.update(self._get_default_headers())
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
                limits=DEFAULT_LIMITS,
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Closes the underlying HTTP client if internally created."""
        if self._external_client is None and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._is_closed = True

    @property
    def client(self) -> httpx.AsyncClient:
        """Returns the active httpx.AsyncClient, raising if uninitialized."""
        if self._client is None or self._client.is_closed:
            raise RuntimeError(
                f"{self.__class__.__name__} must be used as an async context manager "
                f"('async with {self.__class__.__name__}() as src:') or initialized with an active client."
            )
        return self._client

    def _get_default_headers(self) -> dict[str, str]:
        """Override to inject source-specific headers (e.g. auth tokens)."""
        return {}

    async def request_with_retry(
        self,
        url: str,
        method: str = "GET",
        max_retries: int = 3,
        backoff_base: float = 1.5,
        **kwargs: Any,
    ) -> httpx.Response:
        """Executes an HTTP request with exponential backoff and rate-limit handling."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.request(method, url, **kwargs)

                # Rate-limit handling: 429 or 403 rate limit
                if response.status_code in (429, 403):
                    # If this is 403 without rate limit headers or retry exhausted, don't loop endlessly
                    is_rate_limit = (
                        response.status_code == 429
                        or "x-ratelimit-remaining" in response.headers
                        or "retry-after" in response.headers
                    )
                    if not is_rate_limit and response.status_code == 403:
                        response.raise_for_status()

                    if attempt > max_retries:
                        response.raise_for_status()

                    retry_after = response.headers.get("retry-after")
                    reset_time = response.headers.get("x-ratelimit-reset")

                    sleep_seconds = backoff_base ** attempt + random.uniform(0.1, 0.5)
                    if retry_after and retry_after.isdigit():
                        sleep_seconds = float(retry_after)
                    elif reset_time and reset_time.isdigit():
                        sleep_seconds = max(1.0, float(reset_time) - time.time())

                    # Cap maximum sleep to 10 seconds to prevent blocking pipeline
                    sleep_seconds = min(sleep_seconds, 10.0)
                    logger.warning(
                        "[%s] Rate limit hit (%d) on %s. Backing off for %.2fs (attempt %d/%d)",
                        self.source_name,
                        response.status_code,
                        url,
                        sleep_seconds,
                        attempt,
                        max_retries,
                    )
                    await asyncio.sleep(sleep_seconds)
                    continue

                # Server error retry (5xx)
                if response.status_code >= 500 and attempt <= max_retries:
                    sleep_seconds = backoff_base ** attempt + random.uniform(0.1, 0.3)
                    logger.warning(
                        "[%s] Server error %d on %s. Retrying in %.2fs...",
                        self.source_name,
                        response.status_code,
                        url,
                        sleep_seconds,
                    )
                    await asyncio.sleep(sleep_seconds)
                    continue

                response.raise_for_status()
                return response

            except (httpx.RequestError, httpx.TimeoutException) as err:
                if attempt > max_retries:
                    logger.error(
                        "[%s] Request failed after %d retries: %s (%s)",
                        self.source_name,
                        max_retries,
                        url,
                        err,
                    )
                    raise
                sleep_seconds = backoff_base ** attempt + random.uniform(0.1, 0.5)
                logger.warning(
                    "[%s] Network error %s on %s. Retrying in %.2fs...",
                    self.source_name,
                    type(err).__name__,
                    url,
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)

    @abstractmethod
    async def fetch_items(self, limit: Optional[int] = None) -> AsyncIterator[RawItem]:
        """Asynchronously streams normalized RawItem candidates."""
        yield  # type: ignore[misc]

    async def fetch_all(self, limit: Optional[int] = None) -> list[RawItem]:
        """Convenience method collecting all yielded items into a list."""
        items: list[RawItem] = []
        async for item in self.fetch_items(limit=limit):
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
        return items

    async def test_connection(self) -> bool:
        """Verifies endpoint reachability."""
        try:
            await self.fetch_all(limit=1)
            return True
        except Exception as err:
            logger.error("[%s] test_connection failed: %s", self.source_name, err)
            return False
