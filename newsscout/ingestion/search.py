"""newsscout.ingestion.search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ingestion engine bridging web search backends (SearXNG, DuckDuckGo, Tavily, Exa)
into NewsScout's IngestionPipeline. Formulates targeted exploratory queries from
user interest keywords and radar topics, collects deduplicated search results,
and emits normalized RawItem instances for downstream Stage 1 evaluation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Any, Final, Optional, Sequence

import httpx

from newsscout.config import Settings, get_settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import BaseSearchProvider, SearchResult
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
)
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_QUERIES: Final[list[str]] = [
    "autonomous agents github",
    "vLLM alternative local inference",
    "local inference benchmark RTX 4090",
    "spec-driven agent harness",
    "fast LLM serving kernel NVFP4",
    "high performance vector store Qdrant",
    "robotics VLA control ROS2",
    "Mamba state space model github",
]

GITHUB_REPO_REGEX: Final[re.Pattern[str]] = re.compile(
    r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)


class SearchIngestionSource(BaseIngestionSource):
    """Ingestion engine aggregating web search results into normalized RawItems."""

    source_name: str = "web_search"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict[str, Any]] = None,
        aggregator: Optional[MultiSearchAggregator] = None,
        providers: Optional[list[BaseSearchProvider]] = None,
        queries: Optional[Sequence[str]] = None,
        db: Optional[Database] = None,
        max_queries_per_cycle: Optional[int] = None,
        results_per_query: Optional[int] = None,
    ) -> None:
        super().__init__(settings=settings, client=client, config=config)
        self.db = db
        self.aggregator = aggregator or MultiSearchAggregator(
            settings=self.settings,
            providers=providers,
            client=self._external_client,
            db=self.db,
        )
        self.custom_queries = list(queries) if queries is not None else self.config.get("queries")
        self.max_queries_per_cycle = (
            max_queries_per_cycle
            if max_queries_per_cycle is not None
            else getattr(self.settings, "search_max_queries_per_cycle", 5)
        )
        self.results_per_query = (
            results_per_query
            if results_per_query is not None
            else getattr(self.settings, "search_max_results_per_engine", 10)
        )

    def get_search_queries(self) -> list[str]:
        """Formulates queries from custom queries, config keywords, and defaults."""
        if self.custom_queries:
            return list(self.custom_queries)[: self.max_queries_per_cycle]

        queries = list(DEFAULT_SEARCH_QUERIES)
        extra_keywords = self.config.get("interest_keywords", [])
        for kw in extra_keywords:
            cleaned = kw.strip() if isinstance(kw, str) else ""
            if cleaned:
                queries.append(f"{cleaned} github release")
                queries.append(f"{cleaned} benchmark open source")

        return queries[: self.max_queries_per_cycle]

    async def fetch_items(self, limit: Optional[int] = None) -> AsyncIterator[RawItem]:
        """Executes search queries concurrently across active providers and streams RawItems."""
        queries = self.get_search_queries()
        if not queries:
            logger.info("[search_ingestion] No search queries configured.")
            return

        seen_urls: set[str] = set()
        yielded_count = 0

        for query in queries:
            if limit is not None and yielded_count >= limit:
                break

            try:
                results: list[SearchResult] = await self.aggregator.search(
                    query=query,
                    limit_per_provider=self.results_per_query,
                )
            except Exception as err:
                logger.error("[search_ingestion] Failed search for query '%s': %s", query, err)
                continue

            # Deduplicate against SQLite database if connected
            if self.db is not None and results:
                try:
                    results = await self.aggregator.filter_existing_raw_items(results, self.db)
                except Exception as err:
                    logger.warning("[search_ingestion] SQLite filter error: %s", err)

            for res in results:
                canonical_url = clean_and_canonicalize_url(res.url)
                if not canonical_url or canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                raw_item = self._create_raw_item(res, canonical_url, query)
                yield raw_item
                yielded_count += 1
                if limit is not None and yielded_count >= limit:
                    break

    def _create_raw_item(self, res: SearchResult, canonical_url: str, query: str) -> RawItem:
        """Constructs a normalized RawItem from a SearchResult with rich metadata."""
        combined_text = f"{canonical_url} {res.snippet} {res.title}"
        extracted_github_urls = extract_all_github_urls(combined_text)
        github_repo = extract_github_repo(canonical_url) or extract_github_repo(res.snippet)

        url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        source_id = f"search:{url_hash}"
        published_at = (
            res.published_date
            or res.metadata.get("published_date")
            or datetime.now(timezone.utc).isoformat()
        )

        metadata: dict[str, Any] = {
            "source_type": "search",
            "search_engine": res.source_engine,
            "query": query,
            "snippet": res.snippet,
            "score": res.raw_score,
            "published_at": published_at,
            "github_repo": github_repo,
            "extracted_github_urls": extracted_github_urls,
            **res.metadata,
        }

        return RawItem(
            source=self.source_name,
            source_id=source_id,
            title=res.title,
            url=canonical_url,
            raw_content=res.snippet or res.title,
            metadata=metadata,
            ingested_at=datetime.now(timezone.utc),
        )
