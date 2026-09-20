"""newsscout.search.tavily
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tavily Search API provider. Queries https://api.tavily.com/search when
tavily_api_key is configured, gracefully skipping when credentials are absent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from newsscout.config import Settings, get_settings
from newsscout.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchProvider(BaseSearchProvider):
    """Commercial search engine adapter for Tavily API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        search_depth: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        configured_timeout = (
            timeout
            if timeout is not None
            else getattr(self.settings, "search_timeout_seconds", 10.0)
        )
        super().__init__(client=client, timeout=configured_timeout)

        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            tavily_key = getattr(self.settings, "tavily_api_key", None)
            self._api_key = (
                tavily_key.get_secret_value().strip()
                if tavily_key and hasattr(tavily_key, "get_secret_value")
                else ""
            )

        self.search_depth = (
            search_depth
            if search_depth is not None
            else getattr(self.settings, "tavily_search_depth", "basic")
        )

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def requires_api_key(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Returns True only if a non-empty Tavily API key is configured."""
        return bool(self._api_key and self._api_key.strip())

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Queries Tavily API and parses results into SearchResult records."""
        if not self.is_available():
            logger.debug("[tavily] Provider not available (missing API key). Skipping.")
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        payload = {
            "api_key": self._api_key,
            "query": cleaned_query,
            "search_depth": self.search_depth,
            "max_results": max(1, min(limit, 20)),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            client = self._get_client()
            resp = await client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if resp.status_code == 401:
                logger.error("[tavily] Authentication failed (HTTP 401). Invalid API key.")
                return []
            if resp.status_code == 429:
                logger.warning("[tavily] Rate limit or quota exhausted (HTTP 429).")
                return []
            if resp.status_code >= 400:
                logger.warning("[tavily] HTTP error %d: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            raw_results = data.get("results", [])
            if not isinstance(raw_results, list):
                return []

            results: list[SearchResult] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                title = str(item.get("title") or "").strip() or url
                snippet = str(item.get("content") or "").strip()

                raw_score = item.get("score")
                score_val: Optional[float] = None
                if raw_score is not None:
                    try:
                        score_val = float(raw_score)
                    except (ValueError, TypeError):
                        score_val = None

                published_date = item.get("published_date")
                if published_date is not None:
                    published_date = str(published_date)

                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine=self.name,
                        metadata={
                            "score": score_val,
                            "published_date": published_date,
                            "query": cleaned_query,
                            "search_depth": self.search_depth,
                        },
                        raw_score=score_val,
                        published_date=published_date,
                    )
                )
            return results

        except httpx.TimeoutException:
            logger.warning(
                "[tavily] Request timed out after %.1fs for query: '%s'",
                self.timeout,
                cleaned_query,
            )
            return []
        except httpx.RequestError as exc:
            logger.warning("[tavily] Network error querying Tavily: %s", exc)
            return []
        except Exception as exc:
            logger.error("[tavily] Unexpected error during search: %s", exc, exc_info=True)
            return []
