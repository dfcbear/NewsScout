"""newsscout.search.searxng
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SearXNG metasearch provider using self-hosted JSON API.
Aggregates technical and general search results with zero API key requirements.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Optional, Union

import httpx

from newsscout.config import Settings, get_settings
from newsscout.search.base import (
    BaseSearchProvider,
    SearchResult,
    SearchTimeoutError,
    SearchRateLimitError,
    SearchUpstreamError,
    SearchUnavailableError,
    SearchParseError,
)

logger = logging.getLogger(__name__)

_TAG_CLEANER = re.compile(r"<[^>]+>")


def _clean_text(raw_text: Optional[str]) -> str:
    """Strips HTML tags, unescapes entities, and collapses consecutive whitespace."""
    if not raw_text:
        return ""
    text_without_tags = _TAG_CLEANER.sub(" ", raw_text)
    unescaped = html.unescape(text_without_tags)
    return " ".join(unescaped.split())


class SearXNGSearchProvider(BaseSearchProvider):
    """Zero-key SearXNG metasearch provider querying a local or containerized sidecar."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        categories: Union[str, list[str]] = "general,it",
        language: str = "en",
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
        client: Optional[httpx.AsyncClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        configured_timeout = (
            timeout
            if timeout is not None
            else getattr(self.settings, "search_timeout_seconds", 8.0)
        )
        super().__init__(client=client, timeout=configured_timeout)

        raw_base_url = (
            base_url
            if base_url is not None
            else getattr(self.settings, "searxng_base_url", "http://localhost:8080")
        )
        self.base_url = raw_base_url.rstrip("/") if raw_base_url else ""

        if categories is not None:
            self.categories = categories if isinstance(categories, str) else ",".join(categories)
        else:
            self.categories = getattr(self.settings, "searxng_categories", "general,it")

        self.language = language

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = getattr(self.settings, "searxng_enabled", True)

    @property
    def name(self) -> str:
        return "searxng"

    def is_available(self) -> bool:
        """Returns True if enabled and base_url is set."""
        return bool(self.enabled and self.base_url)

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Queries SearXNG JSON API and parses results into SearchResult items."""
        if not self.is_available():
            raise SearchUnavailableError(
                f"SearXNGSearchProvider is disabled or base_url is empty ({self.base_url})"
            )

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        search_url = f"{self.base_url}/search"
        params = {
            "q": cleaned_query,
            "format": "json",
            "categories": self.categories,
            "language": self.language,
        }
        headers = {
            "User-Agent": "NewsScout/1.0 (+https://github.com/newsscout)",
            "Accept": "application/json",
        }

        client = self._get_client()
        try:
            response = await client.get(
                search_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 429:
                raise SearchRateLimitError(f"SearXNG rate limit (HTTP 429): {response.text[:200]}")

            if response.status_code >= 500:
                raise SearchUpstreamError(
                    f"SearXNG server error HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )

            if response.status_code != 200:
                raise SearchUpstreamError(
                    f"SearXNG returned HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                )

        except httpx.TimeoutException as exc:
            raise SearchTimeoutError(
                f"SearXNG query '{cleaned_query}' timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise SearchUpstreamError(f"SearXNG connection error on {search_url}: {exc}") from exc

        # Parse JSON
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise SearchParseError(
                f"SearXNG response is not valid JSON: {response.text[:200]}"
            ) from exc

        if not isinstance(data, dict):
            raise SearchParseError(
                f"SearXNG returned unexpected JSON root type: {type(data).__name__}"
            )

        if "error" in data:
            raise SearchUpstreamError(f"SearXNG returned error payload: {data['error']}")

        raw_results = data.get("results")
        if not raw_results or not isinstance(raw_results, list):
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            if len(results) >= limit:
                break
            if not isinstance(item, dict):
                continue

            raw_url = str(item.get("url") or "").strip()
            if not raw_url or not (raw_url.startswith("http://") or raw_url.startswith("https://")):
                continue

            title = _clean_text(str(item.get("title") or ""))
            snippet = _clean_text(str(item.get("content") or ""))

            raw_score = item.get("score")
            score_val: Optional[float] = None
            if raw_score is not None:
                try:
                    score_val = float(raw_score)
                except (ValueError, TypeError):
                    score_val = None

            published_date = item.get("publishedDate")
            if published_date is not None:
                published_date = str(published_date)

            metadata = {
                "engine": item.get("engine"),
                "engines": item.get("engines", []),
                "category": item.get("category"),
                "positions": item.get("positions", []),
            }

            results.append(
                SearchResult(
                    title=title or raw_url,
                    url=raw_url,
                    snippet=snippet,
                    source_engine=self.name,
                    metadata=metadata,
                    raw_score=score_val,
                    published_date=published_date,
                )
            )

        return results
