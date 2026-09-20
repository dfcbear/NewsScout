"""tests/test_adversarial_m2_2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial challenge and empirical stress-test suite for NewsScout TASK-08 Milestone 2:
Parallel Multi-Search Aggregation Pipeline (MultiSearchAggregator & SearchIngestionSource).

Categories tested:
1. Concurrency & Failure Isolation:
   - 3 out of 4 engines timing out / crashing with unhandled exceptions.
   - All 4 engines failing.
   - Slow engine timeout isolation vs fast engine.
   - Provider availability failure handling.
   - High-concurrency simultaneous multi-search calls.

2. Canonical Deduplication & Multi-Source Confirmation Boost:
   - All 4 engines returning identical entity under 4 radically different URL formats.
   - Deduplication down to exactly 1 result.
   - Correct mathematical multi-source boost calculation (+0.2 per additional engine).
   - Single engine returning internal duplicates (no bogus boost).
   - Non-GitHub URL normalization and parameter stripping (https + www + query + fragment).
   - Scheme divergence boundary on non-GitHub URLs (http vs https).

3. Reciprocal Rank Fusion (RRF) Ranking:
   - Mathematical precision and formula verification.
   - Multi-engine consensus rank dominance over single-engine top rank.
   - Monotonicity across rank degradation.
   - Input permutation invariance.
   - Adversarial rank inputs in metadata (strings, non-numeric, 0, negative values).
   - ZeroDivisionError vulnerability analysis for rank_int == -60.

4. SQLite Raw Items Deduplication:
   - Filtering of pre-existing URLs from SQLite `raw_items`.
   - Batch chunking stress test (> 500 parameters) avoiding SQLite variable limits.
   - Safe behavior when DB is None or disconnected.

5. End-to-End Pipeline & Stage 1 Interoperability:
   - SearchIngestionSource running inside IngestionPipeline.run_cycle().
   - Multi-cycle idempotency.
   - Stage 1 heuristic filtering: open-source tools pass; marketing/no-code wrappers drop.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Optional, Sequence
from unittest.mock import AsyncMock

import httpx
import pytest

from newsscout.config import Settings
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.ingestion.pipeline import IngestionPipeline
from newsscout.ingestion.search import SearchIngestionSource
from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import (
    BaseSearchProvider,
    SearchError,
    SearchResult,
    SearchTimeoutError,
    SearchUpstreamError,
)
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
)
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem


# ============================================================================
# Adversarial Mock Search Providers
# ============================================================================

class ControlledMockProvider(BaseSearchProvider):
    """Configurable mock provider supporting delays, exceptions, and custom results."""

    def __init__(
        self,
        name: str,
        results: Optional[list[SearchResult]] = None,
        delay: float = 0.0,
        exception_to_raise: Optional[Exception] = None,
        is_available_flag: bool = True,
        is_available_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._results = results if results is not None else []
        self._delay = delay
        self._exception = exception_to_raise
        self._is_available_flag = is_available_flag
        self._is_available_exception = is_available_exception
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        if self._is_available_exception:
            raise self._is_available_exception
        return self._is_available_flag

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._exception:
            raise self._exception
        return self._results[:limit]


# ============================================================================
# 1. Concurrency & Engine Fault Isolation
# ============================================================================

class TestAdversarialConcurrencyAndFaultIsolation:
    """Stress tests MultiSearchAggregator under severe network and engine failures."""

    @pytest.mark.asyncio
    async def test_three_out_of_four_engines_timeout_or_crash_returns_healthy_results(self) -> None:
        """Adversarial scenario: 3 engines fail catastrophically; 1 engine survives.

        - Engine 1: Times out (hangs longer than timeout_seconds)
        - Engine 2: Unhandled catastrophic RuntimeError
        - Engine 3: Upstream network connection drop (SearchUpstreamError)
        - Engine 4: Healthy engine returning 3 valid breakthrough results.

        VERIFY:
        1. MultiSearchAggregator does NOT raise any exception.
        2. Exactly the 3 results from the healthy engine are returned.
        3. Fused results are properly formatted and ranked.
        """
        healthy_results = [
            SearchResult(
                title=f"Healthy Breakthrough {i}",
                url=f"https://github.com/healthy-org/tool-{i}",
                snippet=f"Novel high-performance local AI tool {i}",
                source_engine="healthy_engine",
                metadata={"rank": i + 1},
            )
            for i in range(3)
        ]

        p1_timeout = ControlledMockProvider(name="timeout_engine", delay=1.0)
        p2_crash = ControlledMockProvider(
            name="crash_engine",
            exception_to_raise=RuntimeError("Catastrophic memory error in engine C-extension"),
        )
        p3_network = ControlledMockProvider(
            name="network_fail_engine",
            exception_to_raise=httpx.ConnectError("Connection refused by upstream proxy"),
        )
        p4_healthy = ControlledMockProvider(name="healthy_engine", results=healthy_results)

        aggregator = MultiSearchAggregator(
            providers=[p1_timeout, p2_crash, p3_network, p4_healthy],
            timeout_seconds=0.1,  # Fast timeout so p1 triggers asyncio.TimeoutError
        )

        results = await aggregator.search(query="novel local inference")

        assert len(results) == 3
        returned_urls = [r.url for r in results]
        assert "https://github.com/healthy-org/tool-0" in returned_urls
        assert "https://github.com/healthy-org/tool-1" in returned_urls
        assert "https://github.com/healthy-org/tool-2" in returned_urls
        for r in results:
            assert r.metadata["discovered_by"] == ["healthy_engine"]
            assert r.metadata["multi_source_count"] == 1

    @pytest.mark.asyncio
    async def test_all_four_engines_fail_returns_empty_list_cleanly(self) -> None:
        """When all 4 engines fail simultaneously, aggregator must return [] without raising."""
        p1 = ControlledMockProvider(name="e1", delay=0.5)
        p2 = ControlledMockProvider(name="e2", exception_to_raise=ValueError("Corrupt JSON"))
        p3 = ControlledMockProvider(name="e3", exception_to_raise=SearchUpstreamError("503 Service Unavailable"))
        p4 = ControlledMockProvider(name="e4", exception_to_raise=KeyError("Missing expected key"))

        aggregator = MultiSearchAggregator(
            providers=[p1, p2, p3, p4],
            timeout_seconds=0.05,
        )

        results = await aggregator.search(query="local agents")
        assert results == []

    @pytest.mark.asyncio
    async def test_slow_engine_timeout_isolation_does_not_delay_pipeline(self) -> None:
        """Verifies that a 10-second hanging engine is aborted at timeout_seconds (0.15s)

        and does not hold back fast engines.
        """
        fast_results = [
            SearchResult(title="Fast Result", url="https://github.com/fast/tool", snippet="Quick", source_engine="fast")
        ]
        p_fast = ControlledMockProvider(name="fast", results=fast_results, delay=0.01)
        p_slow = ControlledMockProvider(name="slow", delay=5.0)

        aggregator = MultiSearchAggregator(
            providers=[p_fast, p_slow],
            timeout_seconds=0.15,
        )

        start_time = asyncio.get_running_loop().time()
        results = await aggregator.search("fast query")
        elapsed = asyncio.get_running_loop().time() - start_time

        assert len(results) == 1
        assert results[0].url == "https://github.com/fast/tool"
        assert elapsed < 0.5  # Must not wait for the 5-second slow engine

    @pytest.mark.asyncio
    async def test_provider_availability_exception_isolated(self) -> None:
        """If a provider's is_available() raises an unhandled exception, it is skipped."""
        p_bad_avail = ControlledMockProvider(
            name="broken_avail",
            is_available_exception=RuntimeError("Disk full checking config cache"),
        )
        p_good = ControlledMockProvider(
            name="good",
            results=[SearchResult(title="Good", url="https://github.com/good/tool", snippet="OK", source_engine="good")],
        )

        aggregator = MultiSearchAggregator(providers=[p_bad_avail, p_good])
        active = await aggregator.get_active_providers()
        assert len(active) == 1
        assert active[0].name == "good"

        results = await aggregator.search("test")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_high_concurrency_parallel_searches(self) -> None:
        """Executes 20 concurrent search requests on the same MultiSearchAggregator instance."""
        items = [
            SearchResult(
                title=f"Tool {i}",
                url=f"https://github.com/test/tool-{i}",
                snippet="High concurrency test",
                source_engine="test_engine",
            )
            for i in range(10)
        ]
        provider = ControlledMockProvider(name="test_engine", results=items, delay=0.02)
        aggregator = MultiSearchAggregator(providers=[provider], timeout_seconds=1.0)

        tasks = [aggregator.search(f"query {i}") for i in range(20)]
        all_results = await asyncio.gather(*tasks)

        assert len(all_results) == 20
        for res_list in all_results:
            assert len(res_list) == 10
            assert res_list[0].url == "https://github.com/test/tool-0"


# ============================================================================
# 2. Canonical Deduplication & Multi-Source Boost
# ============================================================================

class TestAdversarialDeduplicationAndBoost:
    """Stress tests URL canonicalization, clustering, and multi-source score boosting."""

    def test_four_engines_return_same_repo_under_different_formatting_deduplicates_to_one(self) -> None:
        """Adversarial scenario: All 4 engines return the exact same GitHub repo

        under wildly disparate representations:
        1. SearXNG: Deep subpath with trailing slash & UTM tracking params
        2. DuckDuckGo: Wrapped in /l/?uddg= redirect URL with URL-encoding & ref anchor
        3. Tavily: Protocol-relative with default port :443 and .git suffix
        4. Exa: Bare slug with extra branch info /tree/master/docs

        VERIFY:
        1. Deduplicates to EXACTLY 1 search result.
        2. URL is normalized to canonical 'https://github.com/vllm-project/vllm'.
        3. All 4 engines are recorded in 'discovered_by'.
        4. Multi-source score boost is applied: 0.2 * (4 - 1) = +0.60.
        5. Snippets from all engines are merged cleanly without duplicates.
        """
        item1 = SearchResult(
            title="vLLM: Easy, Fast, Cheap LLM Serving",
            url="https://github.com/vllm-project/vllm/tree/main/benchmarks/?utm_source=twitter&utm_medium=cpc#speed",
            snippet="High throughput inference engine.",
            source_engine="searxng",
            metadata={"rank": 1},
        )
        item2 = SearchResult(
            title="vllm-project/vllm on GitHub",
            url="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm%2Freleases%2Ftag%2Fv0.6.0%3Ffbclid%3D12345&rut=abc",
            snippet="High throughput inference engine.",  # duplicate snippet
            source_engine="duckduckgo",
            metadata={"rank": 2},
        )
        item3 = SearchResult(
            title="vLLM: Memory Efficient LLM Serving with PagedAttention",
            url="//github.com:443/vllm-project/vllm.git",
            snippet="State-of-the-art serving for RTX 4090 and server GPUs.",
            source_engine="tavily",
            metadata={"rank": 1},
        )
        item4 = SearchResult(
            title="vLLM official repository",
            url="https://www.github.com/vllm-project/vllm/blob/master/README.md#getting-started",
            snippet="Supports vLLM kernels and quantization.",
            source_engine="exa",
            metadata={"rank": 3},
        )

        aggregator = MultiSearchAggregator(providers=[])
        deduped = aggregator.deduplicate_and_rank([item1, item2, item3, item4])

        assert len(deduped) == 1
        result = deduped[0]
        assert result.url == "https://github.com/vllm-project/vllm"

        discovered = result.metadata["discovered_by"]
        assert len(discovered) == 4
        assert set(discovered) == {"searxng", "duckduckgo", "tavily", "exa"}
        assert result.metadata["multi_source_count"] == 4

        # Verify mathematical RRF calculation:
        # rank 1: 1/(60+1) = 1/61 ≈ 0.016393
        # rank 2: 1/(60+2) = 1/62 ≈ 0.016129
        # rank 1: 1/(60+1) = 1/61 ≈ 0.016393
        # rank 3: 1/(60+3) = 1/63 ≈ 0.015873
        # rrf_sum = 0.016393 + 0.016129 + 0.016393 + 0.015873 = 0.064788
        # multi_source_bonus = 0.2 * (4 - 1) = 0.6
        # final_score = round(0.064788 + 0.6, 4) = 0.6648
        expected_score = round(1 / 61 + 1 / 62 + 1 / 61 + 1 / 63 + 0.6, 4)
        assert result.raw_score == expected_score

        # Verify snippet merging (duplicate snippet removed, distinct snippets joined with ' | ')
        assert "High throughput inference engine." in result.snippet
        assert "State-of-the-art serving for RTX 4090 and server GPUs." in result.snippet
        assert "Supports vLLM kernels and quantization." in result.snippet
        assert result.snippet.count("High throughput inference engine.") == 1

        # Longest title selected:
        assert result.title == "vLLM: Memory Efficient LLM Serving with PagedAttention"

    def test_single_engine_returning_duplicates_receives_no_multi_source_boost(self) -> None:
        """If a single engine returns the same repo 3 times across different subpaths,

        it must deduplicate to 1, but multi_source_count must remain 1 and bonus must be 0.0.
        """
        items = [
            SearchResult(
                title="vLLM Repo",
                url="https://github.com/vllm-project/vllm",
                snippet="Inference",
                source_engine="searxng",
                metadata={"rank": 1},
            ),
            SearchResult(
                title="vLLM Tree",
                url="https://github.com/vllm-project/vllm/tree/main",
                snippet="Tree view",
                source_engine="searxng",
                metadata={"rank": 4},
            ),
            SearchResult(
                title="vLLM Releases",
                url="https://github.com/vllm-project/vllm/releases",
                snippet="Releases",
                source_engine="searxng",
                metadata={"rank": 8},
            ),
        ]

        aggregator = MultiSearchAggregator(providers=[])
        deduped = aggregator.deduplicate_and_rank(items)

        assert len(deduped) == 1
        result = deduped[0]
        assert result.metadata["discovered_by"] == ["searxng"]
        assert result.metadata["multi_source_count"] == 1

        # Multi-source bonus for 1 engine: 0.2 * (1 - 1) = 0.0
        # rrf = 1/61 + 1/64 + 1/68 ≈ 0.016393 + 0.015625 + 0.014706 = 0.0467
        expected_score = round(1 / 61 + 1 / 64 + 1 / 68, 4)
        assert result.raw_score == expected_score

    def test_non_github_url_deduplication_and_tracking_strip(self) -> None:
        """Verifies deduplication on non-GitHub HTTPS URLs (e.g. arXiv papers, benchmarks)."""
        item1 = SearchResult(
            title="Docling Paper",
            url="https://arxiv.org/abs/2408.09869?utm_source=chatgpt&utm_campaign=ai",
            snippet="Docling layout parsing paper.",
            source_engine="ddg",
        )
        item2 = SearchResult(
            title="Docling Document Parser",
            url="https://www.arxiv.org:443/abs/2408.09869/#comments",
            snippet="High accuracy formula parsing.",
            source_engine="tavily",
        )

        aggregator = MultiSearchAggregator(providers=[])
        deduped = aggregator.deduplicate_and_rank([item1, item2])

        assert len(deduped) == 1
        assert deduped[0].url == "https://arxiv.org/abs/2408.09869"
        assert deduped[0].metadata["multi_source_count"] == 2

    def test_scheme_divergence_on_non_github_urls_boundary(self) -> None:
        """Adversarial discovery: clean_and_canonicalize_url only upgrades scheme to https

        for github.com domains. For non-GitHub domains, http:// and https:// remain distinct schemes.
        """
        url_https = clean_and_canonicalize_url("https://example.com/tool")
        url_http = clean_and_canonicalize_url("http://example.com/tool")

        # Verify exact behavior: http is NOT coerced to https for non-GitHub domains
        assert url_https == "https://example.com/tool"
        assert url_http == "http://example.com/tool"


# ============================================================================
# 3. Reciprocal Rank Fusion (RRF) Ranking
# ============================================================================

class TestAdversarialRRFRanking:
    """Verifies mathematical stability, monotonicity, and consensus dominance of RRF."""

    def test_consensus_dominance_over_single_engine_high_rank(self) -> None:
        """Core property of Reciprocal Rank Fusion + confirmation boost:

        A result confirmed by multiple engines at lower ranks (#5 on Engine 1, #5 on Engine 2)
        MUST rank higher than a result returned by only 1 engine at rank #1.
        """
        single_engine_top = SearchResult(
            title="Niche Tool",
            url="https://github.com/niche/top-ranked-only-on-one",
            snippet="Seen only once",
            source_engine="searxng",
            metadata={"rank": 1},
        )
        multi_engine_lower = [
            SearchResult(
                title="Consensus Tool",
                url="https://github.com/consensus/tool",
                snippet="Engine 1 saw it",
                source_engine="searxng",
                metadata={"rank": 5},
            ),
            SearchResult(
                title="Consensus Tool",
                url="https://github.com/consensus/tool",
                snippet="Engine 2 confirmed it",
                source_engine="duckduckgo",
                metadata={"rank": 5},
            ),
        ]

        aggregator = MultiSearchAggregator(providers=[])
        ranked = aggregator.deduplicate_and_rank([single_engine_top, *multi_engine_lower])

        assert len(ranked) == 2
        # First item must be the consensus tool
        assert ranked[0].url == "https://github.com/consensus/tool"
        assert ranked[1].url == "https://github.com/niche/top-ranked-only-on-one"
        assert ranked[0].raw_score > ranked[1].raw_score

    def test_monotonicity_across_ranks(self) -> None:
        """As rank number increases (worse rank), RRF score must strictly decrease."""
        items = [
            SearchResult(
                title=f"Rank {r}",
                url=f"https://github.com/test/rank-{r}",
                snippet="snippet",
                source_engine="engine",
                metadata={"rank": r},
            )
            for r in [1, 2, 5, 10, 20, 50]
        ]
        aggregator = MultiSearchAggregator(providers=[])
        ranked = aggregator.deduplicate_and_rank(items)

        scores = [r.raw_score for r in ranked]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], f"Score at index {i} ({scores[i]}) not strictly > {scores[i+1]}"

    def test_input_permutation_invariance(self) -> None:
        """The mathematical score of an item must not depend on the order in which

        providers or items were ingested.
        """
        items_a = [
            SearchResult(title="Tool", url="https://github.com/org/tool", snippet="A", source_engine="ddg", metadata={"rank": 2}),
            SearchResult(title="Tool", url="https://github.com/org/tool", snippet="B", source_engine="tavily", metadata={"rank": 4}),
        ]
        items_b = [
            SearchResult(title="Tool", url="https://github.com/org/tool", snippet="B", source_engine="tavily", metadata={"rank": 4}),
            SearchResult(title="Tool", url="https://github.com/org/tool", snippet="A", source_engine="ddg", metadata={"rank": 2}),
        ]

        aggregator = MultiSearchAggregator(providers=[])
        score_a = aggregator.deduplicate_and_rank(items_a)[0].raw_score
        score_b = aggregator.deduplicate_and_rank(items_b)[0].raw_score

        assert score_a == score_b

    def test_adversarial_metadata_ranks(self) -> None:
        """Tests resilience against malformed rank metadata:

        - String rank ("3")
        - Missing or None rank
        - Non-numeric rank ("invalid")
        - Zero rank (0)
        """
        items = [
            SearchResult(title="T1", url="https://github.com/t/1", snippet="", source_engine="e1", metadata={"rank": "3"}),
            SearchResult(title="T2", url="https://github.com/t/2", snippet="", source_engine="e1", metadata={"rank": None}),
            SearchResult(title="T3", url="https://github.com/t/3", snippet="", source_engine="e1", metadata={"rank": "invalid"}),
            SearchResult(title="T4", url="https://github.com/t/4", snippet="", source_engine="e1", metadata={"rank": 0}),
        ]

        aggregator = MultiSearchAggregator(providers=[])
        ranked = aggregator.deduplicate_and_rank(items)
        assert len(ranked) == 4
        for r in ranked:
            assert isinstance(r.raw_score, float)
            assert r.raw_score > 0.0

    def test_adversarial_negative_sixty_rank_behavior(self) -> None:
        """Adversarial stress: if an engine injects rank="-60", denominator is (60.0 + (-60)) = 0.

        Demonstrates that string rank="-60" triggers ZeroDivisionError if unvalidated.
        """
        item_malicious = SearchResult(
            title="Malicious Item",
            url="https://github.com/exploit/crash",
            snippet="",
            source_engine="bad_engine",
            metadata={"rank": "-60"},
        )
        aggregator = MultiSearchAggregator(providers=[])

        # We verify that string "-60" in rank raises ZeroDivisionError in unpatched code
        with pytest.raises(ZeroDivisionError):
            aggregator.deduplicate_and_rank([item_malicious])


# ============================================================================
# 4. SQLite Deduplication against raw_items
# ============================================================================

class TestAdversarialSQLiteDeduplication:
    """Tests deduplication against SQLite `raw_items` table under high load."""

    @pytest.mark.asyncio
    async def test_sqlite_deduplication_filters_existing_preserves_new(self, migrated_db: Database) -> None:
        """Seeds SQLite raw_items with 5 URLs; passes 8 search results (5 duplicate, 3 new).

        VERIFY: Exactly 3 new items remain; 5 duplicates are filtered out.
        """
        existing_urls = [f"https://github.com/existing-org/repo-{i}" for i in range(5)]
        for i, url in enumerate(existing_urls):
            await migrated_db.execute(
                """
                INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)
                """,
                {
                    "source": "github_releases",
                    "source_id": f"gh:{i}",
                    "title": f"Existing Repo {i}",
                    "url": url,
                    "raw_content": "Old content",
                    "metadata_json": "{}",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        new_urls = [f"https://github.com/new-org/repo-{i}" for i in range(3)]
        all_candidate_urls = existing_urls + new_urls
        candidates = [
            SearchResult(
                title=f"Search {u}",
                url=u,
                snippet="Test snippet",
                source_engine="multi_search",
            )
            for u in all_candidate_urls
        ]

        aggregator = MultiSearchAggregator(providers=[], db=migrated_db)
        filtered = await aggregator.filter_existing_raw_items(candidates)

        assert len(filtered) == 3
        returned_urls = [item.url for item in filtered]
        assert set(returned_urls) == set(new_urls)

    @pytest.mark.asyncio
    async def test_sqlite_parameter_chunking_stress_test(self, migrated_db: Database) -> None:
        """SQLite enforces a parameter limit per SQL statement (usually 999).

        When 1,200 search results are filtered against the database,
        the aggregator must partition the query into chunks (<= 500)
        and execute without SQLite 'too many SQL variables' error.
        """
        total_items = 1200
        existing_count = 600

        # Seed 600 existing items into DB in chunks of 100
        for chunk_idx in range(0, existing_count, 100):
            items_to_insert = [
                {
                    "source": "bulk_seed",
                    "source_id": f"bulk:{i}",
                    "title": f"Bulk {i}",
                    "url": f"https://github.com/bulk-org/repo-{i}",
                    "raw_content": "Content",
                    "metadata_json": "{}",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
                for i in range(chunk_idx, chunk_idx + 100)
            ]
            query = """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)
            """
            await migrated_db.execute_many(query, items_to_insert)

        # Prepare 1,200 candidate search results (0..599 existing, 600..1199 new)
        candidate_results = [
            SearchResult(
                title=f"Result {i}",
                url=f"https://github.com/bulk-org/repo-{i}",
                snippet="Batch chunking test",
                source_engine="searxng",
            )
            for i in range(total_items)
        ]

        aggregator = MultiSearchAggregator(providers=[], db=migrated_db)

        # Execute chunked deduplication
        filtered = await aggregator.filter_existing_raw_items(candidate_results)

        assert len(filtered) == 600
        expected_new_urls = {f"https://github.com/bulk-org/repo-{i}" for i in range(600, 1200)}
        assert set(item.url for item in filtered) == expected_new_urls

    @pytest.mark.asyncio
    async def test_filter_existing_raw_items_with_empty_or_none_db(self) -> None:
        """If results list is empty or db is None, returns input cleanly."""
        aggregator = MultiSearchAggregator(providers=[], db=None)
        items = [SearchResult(title="T", url="https://github.com/a/b", snippet="", source_engine="e")]

        # db is None
        assert await aggregator.filter_existing_raw_items(items) == items
        # items is empty
        assert await aggregator.filter_existing_raw_items([]) == []


# ============================================================================
# 5. End-to-End Pipeline & Stage 1 Interoperability
# ============================================================================

class TestAdversarialEndToEndPipeline:
    """Tests SearchIngestionSource within IngestionPipeline and Stage 1 Heuristics."""

    @pytest.mark.asyncio
    async def test_search_ingestion_source_inside_pipeline_cycle(self, migrated_db: Database) -> None:
        """Executes a full IngestionPipeline.run_cycle() using SearchIngestionSource.

        Simulates discoveries of:
        1. A tier-1 open-source tool (vLLM)
        2. A paper with linked GitHub implementation in snippet
        3. A marketing/no-code SaaS tool (no repo, no code)

        VERIFY:
        1. Pipeline executes cleanly and records metrics.
        2. All 3 items are written to SQLite raw_items with source="web_search".
        3. Stage 1 Heuristic Filter evaluates the raw_items:
           - Item 1 (vLLM) PASSES Stage 1
           - Item 2 (Paper with repo) PASSES Stage 1
           - Item 3 (No-code marketing wrapper) is DROPPED by Stage 1
        """
        search_results = [
            SearchResult(
                title="vLLM: High-Throughput Serving Engine",
                url="https://github.com/vllm-project/vllm",
                snippet="Fast LLM inference on RTX 4090 and H100 with PagedAttention.",
                source_engine="searxng",
            ),
            SearchResult(
                title="DeepSeek-Coder: When the Code Writes Itself",
                url="https://arxiv.org/abs/2401.99999",
                snippet="Open code models. Implementation available at https://github.com/deepseek-ai/DeepSeek-Coder",
                source_engine="duckduckgo",
            ),
            SearchResult(
                title="AI Website Builder 2026",
                url="https://landing-page-generator-ai.io",
                snippet="Generate marketing websites in 1 click without code or programming.",
                source_engine="duckduckgo",
            ),
        ]

        mock_provider = ControlledMockProvider(name="mock_search", results=search_results)
        aggregator = MultiSearchAggregator(providers=[mock_provider], db=migrated_db)

        search_source = SearchIngestionSource(
            aggregator=aggregator,
            queries=["ai breakthrough"],
            db=migrated_db,
        )

        pipeline = IngestionPipeline(
            db=migrated_db,
            sources=[search_source],
        )

        # Run pipeline cycle
        metrics = await pipeline.run_cycle()
        assert metrics["sources"]["web_search"]["status"] == "success"
        assert metrics["sources"]["web_search"]["persisted_count"] == 3

        # Verify DB persistence
        rows = await migrated_db.fetch_all(
            "SELECT id, source, source_id, title, url, raw_content, metadata_json, ingested_at FROM raw_items WHERE source = 'web_search';"
        )
        assert len(rows) == 3

        # Reconstruct RawItems from DB
        raw_items_from_db = [RawItem.from_row(r) for r in rows]
        assert len(raw_items_from_db) == 3

        vllm_item = next(item for item in raw_items_from_db if "vllm" in item.url)
        deepseek_item = next(item for item in raw_items_from_db if "arxiv" in item.url)
        marketing_item = next(item for item in raw_items_from_db if "landing-page" in item.url)

        # Evaluate through Stage 1 Filter
        stage1 = Stage1Filter()

        # 1. vLLM must PASS (has_runnable_code=True)
        vllm_eval = stage1.evaluate(vllm_item)
        assert vllm_eval.passed is True
        assert vllm_eval.has_runnable_code is True

        # 2. DeepSeek paper with linked GitHub in snippet must PASS
        deepseek_eval = stage1.evaluate(deepseek_item)
        assert deepseek_eval.passed is True
        assert deepseek_eval.has_runnable_code is True

        # 3. Pure marketing SaaS must be DROPPED
        marketing_eval = stage1.evaluate(marketing_item)
        assert marketing_eval.passed is False
        assert "No runnable artifacts" in marketing_eval.drop_reason

    @pytest.mark.asyncio
    async def test_search_ingestion_multi_cycle_deduplication(self, migrated_db: Database) -> None:
        """Running two consecutive pipeline cycles with identical search output

        must result in 0 new items persisted in cycle 2.
        """
        results = [
            SearchResult(
                title="Docling",
                url="https://github.com/DS4SD/docling",
                snippet="Document conversion.",
                source_engine="searxng",
            )
        ]
        provider = ControlledMockProvider(name="searxng", results=results)
        agg = MultiSearchAggregator(providers=[provider], db=migrated_db)
        src = SearchIngestionSource(aggregator=agg, queries=["docling"], db=migrated_db)
        pipeline = IngestionPipeline(db=migrated_db, sources=[src])

        # Cycle 1: inserts 1 item
        c1_metrics = await pipeline.run_cycle()
        assert c1_metrics["sources"]["web_search"]["persisted_count"] == 1

        # Cycle 2: identical query and search results -> filtered by SQLite deduplication
        c2_metrics = await pipeline.run_cycle()
        assert c2_metrics["sources"]["web_search"]["persisted_count"] == 0

        # DB has exactly 1 record
        rows = await migrated_db.fetch_all("SELECT COUNT(*) FROM raw_items WHERE source = 'web_search';")
        assert rows[0][0] == 1

    def test_search_query_generation_and_keyword_expansion(self) -> None:
        """Tests that interest keywords in config properly expand into structured search queries."""
        config = {
            "interest_keywords": ["quantization NVFP4", "robotics ROS2"],
        }
        source = SearchIngestionSource(
            config=config,
            max_queries_per_cycle=4,
        )
        queries = source.get_search_queries()

        assert len(queries) == 4
        # Default queries come first
        assert "autonomous agents github" in queries
        assert "vLLM alternative local inference" in queries
