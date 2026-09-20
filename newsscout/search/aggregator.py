"""newsscout.search.aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MultiSearchAggregator: parallel multi-engine search execution, timeout isolation,
cross-engine deduplication, Reciprocal Rank Fusion, and SQLite raw_items filtering.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def _select_newest_date(
    current: Optional[str], candidate: Optional[str]
) -> Optional[str]:
    """Returns the newer of two date strings (or the non-None one).

    Dates are compared by ISO-8601 string ordering, which is correct for
    properly formatted dates. Falls back to the non-None value if either
    is missing or unparseable.
    """
    if not candidate:
        return current
    if not current:
        return candidate
    # ISO-8601 string comparison works for same-format dates
    try:
        return candidate if candidate > current else current
    except Exception:
        return current


# ============================================================================
# Circuit Breaker for Search Providers (R3)
# ============================================================================

@dataclass
class CircuitBreakerState:
    """Tracks the circuit breaker state for a single search provider."""

    consecutive_failures: int = 0
    open_until: Optional[datetime] = None  # UTC timestamp when cooldown expires

    @property
    def is_open(self) -> bool:
        """Returns True if the circuit is currently open (in cooldown)."""
        if self.open_until is None:
            return False
        return datetime.now(timezone.utc) < self.open_until

    def record_failure(self, threshold: int, cooldown_seconds: float) -> None:
        """Records a consecutive failure and opens the circuit if threshold is reached."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= threshold:
            self.open_until = datetime.now(timezone.utc) + timedelta(seconds=cooldown_seconds)
            logger.warning(
                "[circuit-breaker] Provider circuit OPENED after %d consecutive failures. "
                "Cooldown: %.0fs",
                self.consecutive_failures,
                cooldown_seconds,
            )

    def record_success(self) -> None:
        """Records a successful call and resets the circuit."""
        if self.consecutive_failures > 0 or self.open_until is not None:
            logger.info(
                "[circuit-breaker] Provider circuit CLOSED after recovery (was %d failures).",
                self.consecutive_failures,
            )
        self.consecutive_failures = 0
        self.open_until = None

    def reset(self) -> None:
        """Manually resets the circuit breaker state (e.g. after cooldown expiry test)."""
        self.consecutive_failures = 0
        self.open_until = None


class MultiSearchAggregator:
    """Orchestrates parallel search across multiple providers with RRF scoring and deduplication.

    Includes a per-provider circuit breaker (R3) that temporarily disables providers
    after consecutive failures.
    """

    def __init__(
        self,
        providers: Optional[Sequence[BaseSearchProvider]] = None,
        settings: Optional[Settings] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        db: Optional[Database] = None,
        circuit_failure_threshold: Optional[int] = None,
        circuit_cooldown_seconds: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self.settings, "search_timeout_seconds", 8.0)
        )
        self.client = client
        self.db = db

        # Circuit breaker configuration (R3)
        self._circuit_failure_threshold: int = (
            circuit_failure_threshold
            if circuit_failure_threshold is not None
            else getattr(self.settings, "circuit_breaker_failure_threshold", 3)
        )
        self._circuit_cooldown_seconds: float = (
            circuit_cooldown_seconds
            if circuit_cooldown_seconds is not None
            else getattr(self.settings, "circuit_breaker_cooldown_seconds", 1800.0)
        )
        self._circuit_states: dict[str, CircuitBreakerState] = {}

        if providers is not None:
            self.providers = list(providers)
        else:
            self.providers = [
                SearXNGSearchProvider(settings=self.settings, client=self.client),
                DuckDuckGoSearchProvider(settings=self.settings, client=self.client),
                TavilySearchProvider(settings=self.settings, client=self.client),
                ExaSearchProvider(settings=self.settings, client=self.client),
            ]

    def _get_circuit_state(self, provider_name: str) -> CircuitBreakerState:
        """Returns (or creates) the circuit breaker state for a provider."""
        if provider_name not in self._circuit_states:
            self._circuit_states[provider_name] = CircuitBreakerState()
        return self._circuit_states[provider_name]

    def is_provider_in_cooldown(self, provider_name: str) -> bool:
        """Returns True if the provider's circuit breaker is currently open."""
        return self._get_circuit_state(provider_name).is_open

    def get_circuit_state(self, provider_name: str) -> CircuitBreakerState:
        """Public accessor for inspecting circuit breaker state (for tests/diagnostics)."""
        return self._get_circuit_state(provider_name)

    async def get_active_providers(self) -> list[BaseSearchProvider]:
        """Returns the list of currently available providers, excluding those in cooldown."""
        active: list[BaseSearchProvider] = []
        for p in self.providers:
            try:
                avail = p.is_available()
                if inspect.isawaitable(avail):
                    avail = await avail
                if not avail:
                    continue
                # Check circuit breaker (R3)
                state = self._get_circuit_state(p.name)
                if state.is_open:
                    logger.info(
                        "[aggregator] Provider '%s' is in circuit-breaker cooldown, skipping.",
                        p.name,
                    )
                    continue
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
        """Queries all active providers in parallel and returns deduplicated, ranked results.

        Providers in circuit-breaker cooldown are automatically excluded.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        active_providers = await self.get_active_providers()
        if not active_providers:
            logger.warning("[aggregator] No search providers are currently available.")
            return []

        async def _safe_search(provider: BaseSearchProvider) -> list[SearchResult]:
            provider_name = provider.name
            circuit = self._get_circuit_state(provider_name)
            try:
                results = await asyncio.wait_for(
                    provider.search(cleaned_query, limit=limit_per_provider, **kwargs),
                    timeout=self.timeout_seconds,
                )
                # Success — reset circuit breaker
                circuit.record_success()
                return results
            except asyncio.TimeoutError:
                logger.warning(
                    "[aggregator] Engine '%s' timed out after %.1fs",
                    provider_name,
                    self.timeout_seconds,
                )
                circuit.record_failure(self._circuit_failure_threshold, self._circuit_cooldown_seconds)
                return []
            except Exception as err:
                logger.warning(
                    "[aggregator] Engine '%s' error: %s",
                    provider_name,
                    err,
                )
                circuit.record_failure(self._circuit_failure_threshold, self._circuit_cooldown_seconds)
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

                # Select the newest published date across all sources
                if item.published_date:
                    best_published_date = _select_newest_date(best_published_date, item.published_date)
                if item.metadata.get("published_date"):
                    best_published_date = _select_newest_date(
                        best_published_date, str(item.metadata["published_date"])
                    )

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
            # Skip entries with empty/None canonical URL (e.g. rejected schemes)
            if not canonical_url:
                logger.warning("Skipping search result with invalid/empty canonical URL: %s", item.url)
                continue
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
        """Closes all underlying providers, isolating failures."""
        for p in self.providers:
            try:
                await p.close()
            except Exception:
                logger.warning("Error closing search provider %s", type(p).__name__, exc_info=True)

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> MultiSearchAggregator:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
