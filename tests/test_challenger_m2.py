"""tests/test_challenger_m2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial empirical challenge suite for NewsScout Milestone 2:
- URL canonicalization with complex schemes, edge cases, tracking params, and fragments
- DuckDuckGo uddg redirect unwrapping (normal, relative, encoded, malformed)
- GitHub URL normalization and repo extraction edge cases
- SearXNG parser stress: entity injection, unclosed tags, malformed JSON, 500/502 error bodies
- DuckDuckGo parser stress: malformed HTML, entity unescaping, anti-bot challenge pages, fallback
- MultiSearchAggregator concurrency, timeout isolation, RRF ranking, and chunked SQLite deduplication
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from typing import Any, Optional
from unittest.mock import AsyncMock, patch
import urllib.parse

import httpx
import pytest

from newsscout.config import Settings
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.ingestion.search import SearchIngestionSource
from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import (
    BaseSearchProvider,
    SearchError,
    SearchParseError,
    SearchRateLimitError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
    SearchUpstreamError,
)
from newsscout.search.duckduckgo import DuckDuckGoSearchProvider, _DuckDuckGoHTMLParser
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
    extract_github_repo_url,
    normalize_url,
    _unwrap_search_redirect,
)
from newsscout.search.searxng import SearXNGSearchProvider, _clean_text
from newsscout.storage.db import Database
from newsscout.storage.migrations import async_apply_migrations
from newsscout.storage.models import RawItem


# ============================================================================
# 1. URL Canonicalization & Normalization Stress
# ============================================================================

class TestURLCanonicalizationStress:
    """Adversarial stress testing for clean_and_canonicalize_url."""

    def test_empty_and_non_string_inputs(self) -> None:
        assert clean_and_canonicalize_url("") == ""
        assert clean_and_canonicalize_url("   ") == ""
        assert clean_and_canonicalize_url(None) == ""  # type: ignore
        assert clean_and_canonicalize_url(12345) == ""  # type: ignore

    def test_case_and_www_normalization(self) -> None:
        url = "HTTP://WWW.GITHUB.COM/vllm-project/vllm"
        assert clean_and_canonicalize_url(url) == "https://github.com/vllm-project/vllm"

        url2 = "HTTPS://WWW.Example.Com:443/Path/To/Resource/"
        assert clean_and_canonicalize_url(url2) == "https://example.com/Path/To/Resource"

        url3 = "http://www.sub.domain.org:80/test/"
        assert clean_and_canonicalize_url(url3) == "http://sub.domain.org/test"

    def test_default_and_non_default_ports(self) -> None:
        assert clean_and_canonicalize_url("http://example.com:80/api") == "http://example.com/api"
        assert clean_and_canonicalize_url("https://example.com:443/api") == "https://example.com/api"
        assert clean_and_canonicalize_url("http://example.com:8080/api") == "http://example.com:8080/api"
        assert clean_and_canonicalize_url("https://example.com:8443/api") == "https://example.com:8443/api"

    def test_excessive_and_trailing_slashes(self) -> None:
        url = "https://example.com///a//b///c////"
        assert clean_and_canonicalize_url(url) == "https://example.com/a/b/c"
        assert clean_and_canonicalize_url("https://example.com/") == "https://example.com"
        assert clean_and_canonicalize_url("https://example.com") == "https://example.com"

    def test_anchor_fragments_stripped(self) -> None:
        assert clean_and_canonicalize_url("https://example.com/page#section-1") == "https://example.com/page"
        assert clean_and_canonicalize_url("https://example.com/page?foo=bar#section-1") == "https://example.com/page?foo=bar"
        assert clean_and_canonicalize_url("https://github.com/vllm-project/vllm#readme") == "https://github.com/vllm-project/vllm"

    def test_comprehensive_tracking_params_removal(self) -> None:
        tracking_query = (
            "utm_source=twitter&utm_medium=cpc&utm_campaign=ai&utm_term=agent"
            "&utm_content=v1&utm_id=42&fbclid=XYZ123&gclid=ABC456&dclid=DCLID"
            "&msclkid=MS123&ref=producthunt&ref_src=embed&spm=100.200&trk=feed"
            "&mc_cid=mail1&mc_eid=mail2&keep_me=important&alpha=1&beta=2"
        )
        url = f"https://example.com/article?{tracking_query}"
        cleaned = clean_and_canonicalize_url(url)
        assert cleaned == "https://example.com/article?alpha=1&beta=2&keep_me=important"

    def test_bare_github_slugs(self) -> None:
        assert clean_and_canonicalize_url("vllm-project/vllm") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("ggerganov/llama.cpp.git") == "https://github.com/ggerganov/llama.cpp"
        # Reserved root names are not treated as bare repo slugs
        assert clean_and_canonicalize_url("trending/repositories") == "https://trending/repositories"
        assert clean_and_canonicalize_url("explore/topics") == "https://explore/topics"

    def test_github_repo_edge_cases(self) -> None:
        # Releases, tags, tree, blob, commits, branches -> canonicalized to repo root
        assert clean_and_canonicalize_url("https://github.com/owner/repo/releases/tag/v1.0") == "https://github.com/owner/repo"
        assert clean_and_canonicalize_url("https://github.com/owner/repo/tree/main/src/lib") == "https://github.com/owner/repo"
        assert clean_and_canonicalize_url("https://github.com/owner/repo/blob/master/README.md") == "https://github.com/owner/repo"
        assert clean_and_canonicalize_url("https://github.com/owner/repo/commits/main") == "https://github.com/owner/repo"
        assert clean_and_canonicalize_url("https://github.com/owner/repo.git") == "https://github.com/owner/repo"

        # Non-code subpaths (issues, pull) retain their distinct page path in clean_and_canonicalize_url
        assert clean_and_canonicalize_url("https://github.com/owner/repo/issues/1") == "https://github.com/owner/repo/issues/1"
        assert clean_and_canonicalize_url("https://github.com/owner/repo/pull/2") == "https://github.com/owner/repo/pull/2"

        # But extract_github_repo extracts the repository owner/repo cleanly for all of them!
        assert extract_github_repo("https://github.com/owner/repo/issues/1") == "owner/repo"
        assert extract_github_repo("https://github.com/owner/repo/pull/2") == "owner/repo"
        assert extract_github_repo("https://github.com/owner/repo/releases/tag/v1.0") == "owner/repo"
        assert extract_github_repo("https://github.com/owner/repo.git") == "owner/repo"

        # extract_github_repo_url returns the root repo URL
        assert extract_github_repo_url("https://github.com/owner/repo/issues/1") == "https://github.com/owner/repo"
        assert extract_github_repo_url("https://github.com/owner/repo/pull/2") == "https://github.com/owner/repo"

    def test_ddg_redirect_unwrapping(self) -> None:
        # Full DDG redirect URL
        target = "https://github.com/vllm-project/vllm?foo=bar&utm_source=ddg"
        encoded = urllib.parse.quote(target)
        ddg_url = f"https://duckduckgo.com/l/?uddg={encoded}&rut=1"
        cleaned = clean_and_canonicalize_url(ddg_url)
        assert cleaned == "https://github.com/vllm-project/vllm?foo=bar"

        # Protocol-relative DDG redirect
        proto_rel = f"//duckduckgo.com/l/?uddg={encoded}"
        assert clean_and_canonicalize_url(proto_rel) == "https://github.com/vllm-project/vllm?foo=bar"

        # Empty uddg
        empty_ddg = "https://duckduckgo.com/l/?uddg="
        assert clean_and_canonicalize_url(empty_ddg) == "https://duckduckgo.com/l"

    def test_extract_all_github_urls_from_text(self) -> None:
        text = (
            "Check out https://github.com/vllm-project/vllm.git and also "
            "https://github.com/ggerganov/llama.cpp/tree/master for C++ inference! "
            "Avoid https://github.com/pricing and duplicate https://github.com/vllm-project/vllm."
        )
        urls = extract_all_github_urls(text)
        assert len(urls) == 2
        assert "https://github.com/vllm-project/vllm" in urls
        assert "https://github.com/ggerganov/llama.cpp" in urls


# ============================================================================
# 2. SearXNG Parser Adversarial Stress
# ============================================================================

class TestSearXNGAdversarialParsing:
    """Stress tests for SearXNGSearchProvider parsing and error handling."""

    def test_clean_text_entity_injection_and_unclosed_tags(self) -> None:
        # Entities unescaped
        raw = "Title with &amp; &quot;quotes&quot; and &#39;apostrophe&#39; &lt;script&gt;alert(1)&lt;/script&gt;"
        cleaned = _clean_text(raw)
        assert cleaned == 'Title with & "quotes" and \'apostrophe\' <script>alert(1)</script>'

        # Unclosed and nested HTML tags stripped
        raw_html = "<div><h2><b class='highlight'>vLLM</b> Serving</h2><p>Fast <i>inference<br>engine"
        cleaned_html = _clean_text(raw_html)
        assert cleaned_html == "vLLM Serving Fast inference engine"

        # None or empty
        assert _clean_text(None) == ""
        assert _clean_text("") == ""
        assert _clean_text("   \n\t  ") == ""

    @pytest.mark.asyncio
    async def test_searxng_malformed_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="Not JSON at all! <html><body>Broken</body></html>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        with pytest.raises(SearchParseError) as exc_info:
            await provider.search("test")
        assert "not valid JSON" in str(exc_info.value)
        await provider.close()

    @pytest.mark.asyncio
    async def test_searxng_unexpected_json_root_type(self) -> None:
        # Root is a list instead of dict
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"url": "https://example.com"}])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        with pytest.raises(SearchParseError) as exc_info:
            await provider.search("test")
        assert "unexpected JSON root type" in str(exc_info.value)
        await provider.close()

    @pytest.mark.asyncio
    async def test_searxng_error_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "Rate limit reached on engines: google"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        with pytest.raises(SearchUpstreamError) as exc_info:
            await provider.search("test")
        assert "Rate limit reached on engines" in str(exc_info.value)
        await provider.close()

    @pytest.mark.asyncio
    async def test_searxng_http_500_and_502_error_bodies(self) -> None:
        for code in (500, 502, 503):
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(code, text=f"<html><body><h1>{code} Bad Gateway</h1></body></html>")

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

            with pytest.raises(SearchUpstreamError) as exc_info:
                await provider.search("test")
            assert exc_info.value.status_code == code
            await provider.close()

    @pytest.mark.asyncio
    async def test_searxng_http_429_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Too many requests")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        with pytest.raises(SearchRateLimitError) as exc_info:
            await provider.search("test")
        assert "HTTP 429" in str(exc_info.value)
        await provider.close()

    @pytest.mark.asyncio
    async def test_searxng_resilient_to_dirty_results(self) -> None:
        payload = {
            "results": [
                # Valid result with unclosed tags and entities
                {
                    "url": "https://github.com/vllm-project/vllm",
                    "title": "vLLM: <b>Fast serving</b> &amp; inference",
                    "content": "A high-throughput inference engine for <i>RTX 4090",
                    "score": "3.5",  # string score coerced to float
                },
                # Invalid URL (non-http)
                {
                    "url": "javascript:alert(1)",
                    "title": "XSS attack",
                },
                # Missing URL
                {
                    "title": "No URL item",
                    "content": "Should be skipped",
                },
                # Non-dict item in results list
                "just a string",
                42,
                # Score unparseable
                {
                    "url": "https://github.com/ggerganov/llama.cpp",
                    "title": "llama.cpp",
                    "content": "Inference in C++",
                    "score": "invalid_score",  # should become None without crash
                },
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        results = await provider.search("test", limit=10)
        assert len(results) == 2

        assert results[0].url == "https://github.com/vllm-project/vllm"
        assert results[0].title == "vLLM: Fast serving & inference"
        assert results[0].snippet == "A high-throughput inference engine for RTX 4090"
        assert results[0].raw_score == 3.5

        assert results[1].url == "https://github.com/ggerganov/llama.cpp"
        assert results[1].raw_score is None

        await provider.close()


# ============================================================================
# 3. DuckDuckGo Parser Adversarial Stress
# ============================================================================

class TestDuckDuckGoAdversarialParsing:
    """Stress tests for DuckDuckGoSearchProvider HTML parsing, entities, anti-bot."""

    def test_ddg_parser_unclosed_tags_and_entities(self) -> None:
        html = """
        <div class="results">
          <div class="result">
            <h2 class="result__title">
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm&rut=1">
                vLLM &amp; &quot;High-Throughput&quot; &#39;Engine&#39;
            </h2>
            <div class="result__snippet">
              PagedAttention &lt;b&gt;NVFP4&lt;/b&gt; &amp; Triton kernels.
            </div>
          </div>
          <div class="result">
            <h2 class="result__title">
              <a class="result__a" href="javascript:alert(1)">Malicious XSS Link</a>
            </h2>
          </div>
          <div class="result">
            <h2 class="result__title">
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fggerganov%2Fllama.cpp">
                llama.cpp: Fast C/C++
              </a>
            </h2>
            <div class="result__snippet">Run local models without Python overhead.</div>
          </div>
        </div>
        """
        parser = _DuckDuckGoHTMLParser(limit=10)
        parser.feed(html)
        parser.close()

        assert len(parser.results) == 2
        # Verify entity unescaping and quotes
        assert parser.results[0].title == 'vLLM & "High-Throughput" \'Engine\''
        assert parser.results[0].url == "https://github.com/vllm-project/vllm"
        assert "PagedAttention <b>NVFP4</b> & Triton" in parser.results[0].snippet

        # Verify second valid item (malicious javascript link was discarded)
        assert parser.results[1].url == "https://github.com/ggerganov/llama.cpp"

    def test_ddg_parser_deeply_nested_html(self) -> None:
        nested_html = (
            '<div class="results"><div class="result"><h2 class="result__title">'
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdeep">'
            + '<span><div><b><i>' * 20
            + 'Deep Title'
            + '</i></b></div></span>' * 20
            + '</a></h2>'
            '<div class="result__snippet">'
            + '<span><div>' * 20
            + 'Deep Snippet'
            + '</div></span>' * 20
            + '</div></div></div>'
        )
        parser = _DuckDuckGoHTMLParser(limit=5)
        parser.feed(nested_html)
        parser.close()

        assert len(parser.results) == 1
        assert parser.results[0].title == "Deep Title"
        assert parser.results[0].snippet == "Deep Snippet"
        assert parser.results[0].url == "https://example.com/deep"

    def test_ddg_parser_empty_and_corrupt_html(self) -> None:
        parser = _DuckDuckGoHTMLParser(limit=5)
        parser.feed("")
        parser.close()
        assert parser.results == []

        parser2 = _DuckDuckGoHTMLParser(limit=5)
        parser2.feed("<<<>>>><<<///a href=bad><>><<<???>>")
        parser2.close()
        assert parser2.results == []

    @pytest.mark.asyncio
    async def test_ddg_anti_bot_detection_and_fallback(self) -> None:
        # Case 1: HTML endpoint triggers "anomaly-detected", Lite endpoint succeeds!
        challenge_html = "<html><body><h1>anomaly-detected</h1><p>Automated bots blocked.</p></body></html>"
        lite_success_html = """
        <table>
          <tr>
            <td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fsgl-project%2Fsglang">SGLang</a></td>
          </tr>
          <tr>
            <td class="result-snippet">Fast serving framework</td>
          </tr>
        </table>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "html.duckduckgo.com" in url_str:
                return httpx.Response(200, text=challenge_html)
            elif "lite.duckduckgo.com" in url_str:
                return httpx.Response(200, text=lite_success_html)
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DuckDuckGoSearchProvider(client=client)

        results = await provider.search("sglang", limit=5)
        assert len(results) == 1
        assert results[0].url == "https://github.com/sgl-project/sglang"
        assert results[0].title == "SGLang"

        await provider.close()

    @pytest.mark.asyncio
    async def test_ddg_both_endpoints_fail_gracefully(self) -> None:
        # Both endpoints return 429 rate limit
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limited")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = DuckDuckGoSearchProvider(client=client)

        # Provider catches and returns [] without crashing
        results = await provider.search("test", limit=5)
        assert results == []

        await provider.close()


# ============================================================================
# 4. MultiSearchAggregator Concurrency, Timeouts & Deduplication Stress
# ============================================================================

class TestAggregatorStressAndEdgeCases:
    """Stress tests for MultiSearchAggregator concurrency, timeouts, and SQLite chunking."""

    @pytest.mark.asyncio
    async def test_aggregator_timeout_isolation(self) -> None:
        """Verifies that a hanging provider times out without killing fast providers."""
        class FastProvider(BaseSearchProvider):
            @property
            def name(self) -> str:
                return "fast"
            def is_available(self) -> bool:
                return True
            async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
                return [SearchResult(title="Fast Result", url="https://example.com/fast", snippet="quick")]

        class HangingProvider(BaseSearchProvider):
            @property
            def name(self) -> str:
                return "hanging"
            def is_available(self) -> bool:
                return True
            async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
                await asyncio.sleep(5.0)
                return [SearchResult(title="Slow Result", url="https://example.com/slow", snippet="too late")]

        aggregator = MultiSearchAggregator(
            providers=[FastProvider(), HangingProvider()],
            timeout_seconds=0.1,  # Fast timeout
        )

        results = await aggregator.search("test")
        assert len(results) == 1
        assert results[0].url == "https://example.com/fast"

    @pytest.mark.asyncio
    async def test_cross_engine_deduplication_and_rrf_boost(self) -> None:
        """Multiple engines return different representations of the same repo URL."""
        res_ddg = SearchResult(
            title="vLLM: Serving Engine",
            url="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm&rut=1",
            snippet="High throughput inference engine.",
            source_engine="duckduckgo",
            metadata={"rank": 1},
        )
        res_searxng = SearchResult(
            title="vLLM Project Official",
            url="http://www.github.com/vllm-project/vllm.git",
            snippet="Memory efficient serving engine for LLMs.",
            source_engine="searxng",
            metadata={"rank": 2},
        )
        res_tavily = SearchResult(
            title="vLLM Repository",
            url="https://github.com/vllm-project/vllm/releases/tag/v0.6.0?utm_source=twitter#readme",
            snippet="Release v0.6.0 with NVFP4 support.",
            source_engine="tavily",
            metadata={"rank": 1},
        )

        aggregator = MultiSearchAggregator(providers=[])
        ranked = aggregator.deduplicate_and_rank([res_ddg, res_searxng, res_tavily])

        # All 3 representations collapse into ONE canonical repo URL
        assert len(ranked) == 1
        top = ranked[0]
        assert top.url == "https://github.com/vllm-project/vllm"
        assert top.metadata["discovered_by"] == ["duckduckgo", "searxng", "tavily"]
        assert top.metadata["multi_source_count"] == 3
        # RRF formula: 1/(60+1) + 1/(60+2) + 1/(60+1) + 0.2*(3-1)
        expected_bonus = 0.2 * 2
        assert top.raw_score is not None and top.raw_score > expected_bonus
        assert top.metadata["github_repo"] == "vllm-project/vllm"

    @pytest.mark.asyncio
    async def test_sqlite_deduplication_chunking_stress(self, tmp_path: Any) -> None:
        """Stress tests filter_existing_raw_items with > 600 items to verify chunking loop."""
        db_path = tmp_path / "test_dedup.db"
        db = Database(str(db_path))
        await db.initialize()
        await async_apply_migrations(db)

        # Seed 50 URLs in raw_items
        seeded_urls = [f"https://example.com/item/{i}" for i in range(50)]
        async with db.transaction() as conn:
            for u in seeded_urls:
                conn.execute(
                    "INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                    ("test", f"id_{u}", "Title", u, "Content", "{}"),
                )

        # Generate 650 search results (50 duplicates + 600 fresh URLs)
        test_results = [
            SearchResult(title=f"Result {i}", url=f"https://example.com/item/{i}", snippet="")
            for i in range(650)
        ]

        aggregator = MultiSearchAggregator(providers=[], db=db)
        filtered = await aggregator.filter_existing_raw_items(test_results)

        # Exactly 600 new items should survive
        assert len(filtered) == 600
        surviving_urls = {item.url for item in filtered}
        for u in seeded_urls:
            assert u not in surviving_urls

        await db.close()

    def test_to_raw_items_conversion_and_stage1_compatibility(self) -> None:
        """Verifies that SearchResult conversion generates valid RawItems compatible with Stage 1."""
        res = SearchResult(
            title="vLLM: Serving Engine",
            url="https://github.com/vllm-project/vllm",
            snippet="High throughput inference engine.",
            source_engine="multi_search",
            metadata={
                "github_repo": "vllm-project/vllm",
                "extracted_github_urls": ["https://github.com/vllm-project/vllm"],
            },
            raw_score=0.45,
        )

        aggregator = MultiSearchAggregator(providers=[])
        raw_items = aggregator.to_raw_items([res], query="vllm")

        assert len(raw_items) == 1
        item = raw_items[0]
        assert isinstance(item, RawItem)
        assert item.source == "web_search"
        assert item.source_id.startswith("search:")
        assert item.metadata["github_repo"] == "vllm-project/vllm"

        # Pass to Stage1Filter and verify it awards has_runnable_code = True
        stage1 = Stage1Filter()
        result = stage1.evaluate(item)
        assert result.passed is True
        assert result.has_runnable_code is True

    @pytest.mark.asyncio
    async def test_aggregator_zero_available_providers(self) -> None:
        """Aggregator with zero available providers returns [] gracefully."""
        aggregator = MultiSearchAggregator(providers=[])
        results = await aggregator.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_aggregator_empty_or_whitespace_query(self) -> None:
        """Aggregator returns [] immediately on empty query with no network calls."""
        class MockProvider(BaseSearchProvider):
            @property
            def name(self) -> str:
                return "mock"
            def is_available(self) -> bool:
                return True
            async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
                raise AssertionError("Should not be called!")

        aggregator = MultiSearchAggregator(providers=[MockProvider()])
        assert await aggregator.search("") == []
        assert await aggregator.search("   \n\t  ") == []


# ============================================================================
# 5. Unicode, Massive Payloads, and Extreme Edge Cases
# ============================================================================

class TestExtremeAdversarialEdgeCases:
    """Stress tests for Unicode, massive payloads, and malformed query strings."""

    def test_ddg_parser_unicode_umlauts_and_emojis(self) -> None:
        html = """
        <div class="results">
          <div class="result">
            <h2 class="result__title">
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fowner%2Fai-gr%C3%BCndung">
                🚀 Künstliche Intelligenz: Gründer-Modell &amp; Durchbruch ✨
              </a>
            </h2>
            <div class="result__snippet">
              Große Sprachmodelle für deutsche Ingenieure — höchste Effizienz mit vLLM.
            </div>
          </div>
        </div>
        """
        parser = _DuckDuckGoHTMLParser(limit=5)
        parser.feed(html)
        parser.close()

        assert len(parser.results) == 1
        res = parser.results[0]
        assert "🚀" in res.title
        assert "Gründer-Modell" in res.title
        assert "Größe Sprachmodelle" not in res.snippet
        assert "Große Sprachmodelle" in res.snippet

    def test_ddg_parser_massive_html_payload(self) -> None:
        """Feed 2MB of valid and malformed HTML to verify memory and speed."""
        items = []
        for i in range(1000):
            items.append(f"""
            <div class="result">
              <h2 class="result__title">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fitem_{i}">
                  Item Title {i}
                </a>
              </h2>
              <div class="result__snippet">Snippet text for item number {i} with some padding content to expand size.</div>
            </div>
            """)
        huge_html = f'<div class="results">{"".join(items)}</div>'
        assert len(huge_html) > 200_000

        parser = _DuckDuckGoHTMLParser(limit=25)
        parser.feed(huge_html)
        parser.close()

        # Should stop exactly at limit
        assert len(parser.results) == 25
        assert parser.results[0].title == "Item Title 0"
        assert parser.results[24].title == "Item Title 24"

    @pytest.mark.asyncio
    async def test_searxng_result_item_missing_title_or_content(self) -> None:
        """When title or content is None, SearXNGSearchProvider falls back gracefully."""
        payload = {
            "results": [
                {
                    "url": "https://github.com/vllm-project/vllm",
                    "title": None,
                    "content": None,
                },
                {
                    "url": "https://github.com/ggerganov/llama.cpp",
                    # Missing title and content entirely
                },
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = SearXNGSearchProvider(base_url="http://localhost:8080", client=client)

        results = await provider.search("test", limit=5)
        assert len(results) == 2
        # Title falls back to URL
        assert results[0].title == "https://github.com/vllm-project/vllm"
        assert results[0].snippet == ""
        assert results[1].title == "https://github.com/ggerganov/llama.cpp"
        assert results[1].snippet == ""

        await provider.close()

