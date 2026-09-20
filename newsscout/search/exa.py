"""newsscout.search.exa
~~~~~~~~~~~~~~~~~~~~~~~~
Exa Search API provider. Queries https://api.exa.ai/search when
exa_api_key is configured, gracefully skipping when credentials are absent.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from newsscout.config import Settings, get_settings
from newsscout.search.base import BaseSearchProvider, SearchResult

logger = logging.getLogger(__name__)

EXA_SEARCH_URL = "https://api.exa.ai/search"


class ExaSearchProvider(BaseSearchProvider):
    """Commercial neural search adapter for Exa API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        search_type: Optional[str] = None,
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
            exa_key = getattr(self.settings, "exa_api_key", None)
            self._api_key = (
                exa_key.get_secret_value().strip()
                if exa_key and hasattr(exa_key, "get_secret_value")
                else ""
            )

        self.search_type = (
            search_type
            if search_type is not None
            else getattr(self.settings, "exa_search_type", "neural")
        )

    @property
    def name(self) -> str:
        return "exa"

    @property
    def requires_api_key(self) -> bool:
        return True

    def is_available(self) -> bool:
        """Returns True only if a non-empty Exa API key is configured."""
        return bool(self._api_key and self._api_key.strip())

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Queries Exa API and parses results into SearchResult records."""
        if not self.is_available():
            logger.debug("[exa] Provider not available (missing API key). Skipping.")
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        payload = {
            "query": cleaned_query,
            "type": self.search_type,
            "num_results": max(1, min(limit, 20)),
            "contents": {
                "text": {"max_characters": 1000},
                "highlights": {"num_sentences": 2, "highlights_per_url": 1},
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._api_key,
        }

        try:
            client = self._get_client()
            resp = await client.post(
                EXA_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

            if resp.status_code == 401:
                logger.error("[exa] Authentication failed (HTTP 401). Invalid API key.")
                return []
            if resp.status_code == 429:
                logger.warning("[exa] Rate limit or credits exhausted (HTTP 429).")
                return []
            if resp.status_code >= 400:
                logger.warning("[exa] HTTP error %d: %s", resp.status_code, resp.text[:200])
                return []

            data = resp.json()
            raw_results = data.get("results", [])
            if not isinstance(raw_results, list):
                return []

            results: list[SearchResult] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or item.get("id") or "").strip()
                if not url:
                    continue
                title = str(item.get("title") or "").strip() or url

                # Extract snippet prioritizing highlights then raw text
                snippet = ""
                highlights = item.get("highlights")
                if isinstance(highlights, list) and highlights:
                    snippet = " ... ".join(str(h).strip() for h in highlights if h)
                if not snippet:
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        snippet = text_val[:500].strip()

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

                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source_engine=self.name,
                        metadata={
                            "score": score_val,
                            "published_date": published_date,
                            "author": item.get("author"),
                            "query": cleaned_query,
                            "search_type": data.get("resolvedSearchType", self.search_type),
                        },
                        raw_score=score_val,
                        published_date=published_date,
                    )
                )
            return results

        except httpx.TimeoutException:
            logger.warning(
                "[exa] Request timed out after %.1fs for query: '%s'",
                self.timeout,
                cleaned_query,
            )
            return []
        except httpx.RequestError as exc:
            logger.warning("[exa] Network error querying Exa: %s", exc)
            return []
        except Exception as exc:
            logger.error("[exa] Unexpected error during search: %s", exc, exc_info=True)
            return []
