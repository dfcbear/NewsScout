"""newsscout.search.aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MultiSearchAggregator: parallel multi-engine search execution, timeout isolation,
cross-engine deduplication, Reciprocal Rank Fusion, and SQLite raw_items filtering.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import inspect
import logging
from typing import Any, Optional, Sequence

import httpx

from newsscout.config import Settings, get_settings
from newsscout.search.base import BaseSearchProvider, SearchResult
from newsscout.search.duckduckgo import DuckDuckGoSearchProvider
from newsscout.search.exa import ExaSearchProvider
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
)
from newsscout.search.searxng import SearXNGSearchProvider
from newsscout.search.tavily import TavilySearchProvider
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)


class MultiSearchAggregator:
    """Orchestrates parallel search across multiple providers with RRF scoring and deduplication."""

    def __init__(
        self,
        providers: Optional[Sequence[BaseSearchProvider]] = None,
        settings: Optional[Settings] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        db: Optional[Database] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "search_timeout_seconds", 8.0)
        )
        self.client = client
        self.db = db

        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = [
                SearXNGSearchProvider(settings=self.settings, client=self.client),
                DuckDuckGoSearchProvider(settings=self.settings, client=self.client),
                TavilySearchProvider(settings=self.settings, client=self.client),
                ExaSearchProvider(settings=self.settings, client=self.client),
            ]

    async def get_active_providers(self) -> list[BaseSearchProvider]:
        """Returns the list of currently available providers."""
        active: list[BaseSearchProvider] = []
        for p in self.providers:
            try:
                avail = p.is_available()
                if inspect.isawaitable(avail):
                    avail = await avail
                if avail:
                    active.append(p)
            except Exception as err:
                logger.warning("[aggregator] Error checking availability for '%s': %s", p.name, err)
        return active

    async def search(
        self,
        query: str,
        limit_per_provider: int = 10,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Queries all active providers in parallel and returns deduplicated, ranked results."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        active_providers = await self.get_active_providers()
        if not active_providers:
            logger.warning("[aggregator] No search providers are currently available.")
            return []

        async def _safe_search(provider: BaseSearchProvider) -> list[SearchResult]:
            provider_name = provider.name
            try:
                return await asyncio.wait_for(
                    provider.search(cleaned_query, limit=limit_per_provider, **kwargs),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[aggregator] Engine '%s' timed out after %.1fs",
                    provider_name,
                    self.timeout_seconds,
                )
                return []
            except Exception as err:
                logger.warning(
                    "[aggregator] Engine '%s' error: %s",
                    provider_name,
                    err,
                )
                return []

        results_by_provider = await asyncio.gather(
            *[_safe_search(p) for p in active_providers],
            return_exceptions=False,
        )

        flattened: list[SearchResult] = []
        for res_list in results_by_provider:
            flattened.extend(res_list)

        return self.deduplicate_and_rank(flattened)

    def deduplicate_and_rank(
        self,
        results: Sequence[SearchResult],
    ) -> list[SearchResult]:
        """Normalizes URLs, deduplicates identical items across engines, and computes RRF scores."""
        if not results:
            return []

        grouped: dict[str, list[SearchResult]] = {}

        for item in results:
            canonical = clean_and_canonicalize_url(item.url)
            if not canonical:
                continue
            if canonical not in grouped:
                grouped[canonical] = []
            grouped[canonical].append(item)

        merged_results: list[SearchResult] = []

        for canonical_url, cluster in grouped.items():
            discovered_by: list[str] = []
            engine_ranks: dict[str, int] = {}
            snippets: list[str] = []
            best_title = cluster[0].title
            best_published_date: Optional[str] = None

            # Calculate Reciprocal Rank Fusion (RRF) score
            rrf_score = 0.0

            for idx, item in enumerate(cluster):
                engine = item.source_engine or item.metadata.get("engine") or "unknown"
                if engine not in discovered_by:
                    discovered_by.append(engine)

                # Determine 1-based rank from metadata or position
                rank = item.metadata.get("rank") or item.metadata.get("engine_rank") or (idx + 1)
                try:
                    rank_int = int(rank)
                except (ValueError, TypeError):
                    rank_int = idx + 1
                engine_ranks[engine] = rank_int
                rrf_score += 1.0 / (60.0 + rank_int)

                if item.snippet and item.snippet not in snippets:
                    snippets.append(item.snippet)

                # Prefer longest informative title
                if len(item.title) > len(best_title):
                    best_title = item.title

                if item.published_date and not best_published_date:
                    best_published_date = item.published_date
                elif item.metadata.get("published_date") and not best_published_date:
                    best_published_date = str(item.metadata["published_date"])

            # Multi-source confirmation bonus
            multi_source_bonus = 0.2 * (len(discovered_by) - 1)
            final_score = round(rrf_score + multi_source_bonus, 4)

            # Combined snippet
            combined_snippet = " | ".join(snippets) if snippets else ""

            # Extract repository metadata
            repo = extract_github_repo(canonical_url) or extract_github_repo(combined_snippet)
            extracted_urls = extract_all_github_urls(canonical_url + " " + combined_snippet)

            merged_metadata = {
                "discovered_by": discovered_by,
                "engine_ranks": engine_ranks,
                "github_repo": repo,
                "extracted_github_urls": extracted_urls,
                "multi_source_count": len(discovered_by),
                "score": final_score,
            }

            merged_results.append(
                SearchResult(
                    title=best_title,
                    url=canonical_url,
                    snippet=combined_snippet,
                    source_engine="multi_search",
                    raw_score=final_score,
                    published_date=best_published_date,
                    metadata=merged_metadata,
                )
            )

        # Sort descending by fused score
        merged_results.sort(key=lambda x: x.raw_score or 0.0, reverse=True)
        return merged_results

    async def filter_existing_raw_items(
        self,
        results: Sequence[SearchResult],
        db: Optional[Database] = None,
    ) -> list[SearchResult]:
        """Filters out SearchResults whose canonical URL already exists in SQLite raw_items."""
        target_db = db or self.db
        if target_db is None or not results:
            return list(results)

        urls = [item.url for item in results]
        existing_urls: set[str] = set()

        # Batch query in chunks of 500 parameters (safe for SQLite bounds)
        chunk_size = 500
        for i in range(0, len(urls), chunk_size):
            chunk = urls[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT url FROM raw_items WHERE url IN ({placeholders});"
            rows = await target_db.fetch_all(query, tuple(chunk))
            for row in rows:
                existing_urls.add(str(row[0]))

        return [item for item in results if item.url not in existing_urls]

    def to_raw_items(
        self,
        results: Sequence[SearchResult],
        query: str = "",
    ) -> list[RawItem]:
        """Converts SearchResult instances to RawItem domain models for Stage 1 ingestion."""
        raw_items: list[RawItem] = []

        for item in results:
            canonical_url = clean_and_canonicalize_url(item.url)
            # Deterministic source_id using SHA256 of canonical URL
            url_hash = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
            source_id = f"search:{url_hash}"

            metadata = dict(item.metadata)
            metadata["source_type"] = "search"
            metadata["search_query"] = query
            metadata["score"] = item.raw_score
            metadata["ingestion_method"] = "multi_search"

            raw_item = RawItem(
                source="web_search",
                source_id=source_id,
                title=item.title,
                url=canonical_url,
                raw_content=item.snippet or item.title,
                metadata=metadata,
                ingested_at=datetime.now(timezone.utc),
            )
            raw_items.append(raw_item)

        return raw_items

    async def close(self) -> None:
        """Closes all underlying providers."""
        for p in self.providers:
            await p.close()

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> MultiSearchAggregator:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
