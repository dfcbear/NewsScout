"""newsscout.search.duckduckgo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Zero-key DuckDuckGo web search provider using HTML and Lite endpoints with
standard library HTMLParser (zero external dependencies).
"""

from __future__ import annotations

from html.parser import HTMLParser
import logging
import random
from typing import Any, Optional
import urllib.parse

import httpx

from newsscout.config import Settings, get_settings
from newsscout.search.base import (
    BaseSearchProvider,
    SearchResult,
    SearchTimeoutError,
    SearchRateLimitError,
    SearchUnavailableError,
)

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
]


class _DuckDuckGoHTMLParser(HTMLParser):
    """Parses DuckDuckGo HTML and Lite search results into SearchResult items."""

    def __init__(self, limit: int = 10) -> None:
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[SearchResult] = []
        self._current: Optional[dict[str, Any]] = None
        self._in_title = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def _flush_current(self) -> None:
        if self._current is not None and self._current.get("url"):
            title = " ".join("".join(self._title_parts).split())
            snippet = " ".join("".join(self._snippet_parts).split())
            if not self._current.get("title"):
                self._current["title"] = title
            if not self._current.get("snippet"):
                self._current["snippet"] = snippet

            if self._current["title"] and self._current["url"]:
                rank = len(self.results) + 1
                self.results.append(
                    SearchResult(
                        title=self._current["title"],
                        url=self._current["url"],
                        snippet=self._current.get("snippet", ""),
                        source_engine="duckduckgo",
                        metadata={"rank": rank},
                        raw_score=round(1.0 / rank, 3),
                    )
                )

        self._current = None
        self._title_parts = []
        self._snippet_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if len(self.results) >= self.limit:
            return

        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        classes = attr_dict.get("class", "").split()

        # DuckDuckGo HTML: <a class="result__a" href="..."> or <a class="result-link" href="...">
        if tag == "a" and any("result__a" in c or "result-link" in c for c in classes):
            self._flush_current()
            raw_href = attr_dict.get("href", "")
            target_url = self._extract_destination_url(raw_href)

            # Ignore relative anchor links or empty URLs
            if target_url and (target_url.startswith("http://") or target_url.startswith("https://")):
                self._current = {
                    "url": target_url,
                    "title": "",
                    "snippet": "",
                }
                self._in_title = True
                self._in_snippet = False
                self._title_parts = []
            return

        # DuckDuckGo HTML: <a class="result__snippet" ...> or <div class="result__snippet" ...>
        # DuckDuckGo Lite: <td class="result-snippet">
        if any("result__snippet" in c or "result-snippet" in c for c in classes):
            self._in_title = False
            self._in_snippet = True
            self._snippet_parts = []
            return

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            if self._current is not None:
                self._current["title"] = " ".join("".join(self._title_parts).split())
            return

        if self._in_snippet and tag in ("a", "div", "td", "p", "span"):
            self._in_snippet = False
            if self._current is not None:
                self._current["snippet"] = " ".join("".join(self._snippet_parts).split())
                self._flush_current()
            return

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush_current()

    @staticmethod
    def _extract_destination_url(raw_href: str) -> str:
        if not raw_href:
            return ""
        if "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return urllib.parse.unquote(qs["uddg"][0])
        if raw_href.startswith("//"):
            return "https:" + raw_href
        return raw_href


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """Zero-key DuckDuckGo search provider using HTML and Lite endpoints."""

    HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

    def __init__(
        self,
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

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = getattr(self.settings, "duckduckgo_enabled", True)

    @property
    def name(self) -> str:
        return "duckduckgo"

    def is_available(self) -> bool:
        """Returns True if enabled."""
        return self.enabled

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Queries DuckDuckGo HTML/Lite endpoints and parses results into SearchResult items."""
        if not self.is_available():
            raise SearchUnavailableError("DuckDuckGoSearchProvider is disabled")

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # Try HTML endpoint first, fallback to Lite on anti-bot/rate-limit
        for endpoint in [self.HTML_ENDPOINT, self.LITE_ENDPOINT]:
            results = await self._query_endpoint(endpoint, cleaned_query, limit)
            if results is not None:
                return results

        return []

    async def _query_endpoint(
        self,
        endpoint: str,
        query: str,
        limit: int,
    ) -> Optional[list[SearchResult]]:
        client = self._get_client()
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://duckduckgo.com/",
        }
        data = {"q": query, "b": ""}

        try:
            response = await client.post(
                endpoint,
                data=data,
                headers=headers,
                timeout=self.timeout,
            )

            # Rate limit / anti-bot detection
            if response.status_code in (429, 202):
                logger.warning("[DuckDuckGo] Endpoint %s returned HTTP %d", endpoint, response.status_code)
                return None

            body_lower = response.text.lower()
            if "anomaly-detected" in body_lower or "automated traffic" in body_lower or "bots" in body_lower:
                logger.warning("[DuckDuckGo] Endpoint %s triggered anti-bot challenge", endpoint)
                return None

            if response.status_code != 200:
                logger.warning("[DuckDuckGo] Endpoint %s returned HTTP %d", endpoint, response.status_code)
                return None

            parser = _DuckDuckGoHTMLParser(limit=limit)
            parser.feed(response.text)
            parser.close()
            return parser.results

        except httpx.TimeoutException as exc:
            logger.warning("[DuckDuckGo] Timeout querying %s: %s", endpoint, exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("[DuckDuckGo] Request error querying %s: %s", endpoint, exc)
            return None
