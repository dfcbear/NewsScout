"""tests/test_search.py
~~~~~~~~~~~~~~~~~~~~
Comprehensive 100% offline mock test suite for NewsScout Milestone 2:
- Base search protocol, data structures, and exception hierarchy
- SearXNG zero-key provider
- DuckDuckGo zero-key provider (stdlib HTMLParser, uddg redirect unwrapping, Lite fallback)
- Tavily optional commercial provider (graceful skip when missing key, error isolation)
- Exa optional commercial provider (neural search, highlights/text parsing, graceful skip)
- URL normalizer and GitHub repo extraction
- MultiSearchAggregator (parallel execution, timeout isolation, RRF ranking, SQLite deduplication)
- SearchIngestionSource (RawItem conversion, Stage 1 heuristic compatibility, pipeline persistence)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import SecretStr
import pytest

from newsscout.config import Settings
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.ingestion.pipeline import IngestionPipeline
from newsscout.ingestion.search import SearchIngestionSource
from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import (
    BaseSearchProvider,
    SearchError,
    SearchParseError,
    SearchProviderError,
    SearchRateLimitError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
    SearchUpstreamError,
)
from newsscout.search.duckduckgo import DuckDuckGoSearchProvider, _DuckDuckGoHTMLParser
from newsscout.search.exa import ExaSearchProvider
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
    extract_github_repo_url,
    normalize_url,
)
from newsscout.search.searxng import SearXNGSearchProvider, _clean_text
from newsscout.search.tavily import TavilySearchProvider
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem


# ============================================================================
# Synthetic Mock Payloads
# ============================================================================

MOCK_SEARXNG_JSON = {
    "query": "vllm inference",
    "number_of_results": 2,
    "results": [
        {
            "url": "https://github.com/vllm-project/vllm",
            "title": "vLLM: <b>Easy, fast, and cheap</b> LLM serving",
            "content": "A <i>high-throughput</i> and memory-efficient inference engine &amp; server.",
            "engine": "github",
            "engines": ["github", "google"],
            "positions": [1, 2],
            "score": 3.5,
            "category": "it",
            "publishedDate": "2026-09-18T10:00:00Z",
        },
        {
            "url": "https://github.com/vllm-project/vllm/tree/main/benchmarks?utm_source=hackernews",
            "title": "vLLM Benchmarks",
            "content": "Benchmarking suites for RTX 4090 and H100.",
            "engine": "google",
            "engines": ["google"],
            "positions": [3],
            "score": 1.8,
            "category": "general",
        },
    ],
}

MOCK_DDG_HTML = """
<!DOCTYPE html>
<html>
<body>
<div class="results">
  <div class="result results_links results_links_deep web-result">
    <div class="result__body">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm&rut=123">vLLM: High-Throughput Serving Engine</a>
      </h2>
      <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm&rut=123">
        A high-throughput and memory-efficient inference and serving engine for LLMs with <b>PagedAttention</b> &amp; NVFP4.
      </a>
    </div>
  </div>
  <div class="result results_links results_links_deep web-result">
    <div class="result__body">
      <h2 class="result__title">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fggerganov%2Fllama.cpp&rut=456">llama.cpp: LLM inference in C/C++</a>
      </h2>
      <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fggerganov%2Fllama.cpp&rut=456">
        Port of Facebook&#39;s LLaMA model in C/C++ with minimal dependencies.
      </a>
    </div>
  </div>
</div>
</body>
</html>
"""

MOCK_DDG_LITE_HTML = """
<!DOCTYPE html>
<html>
<body>
<table>
  <tr>
    <td valign="top">1.&nbsp;</td>
    <td>
      <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fsgl-project%2Fsglang&rut=789">SGLang: Fast Serving Framework</a>
    </td>
  </tr>
  <tr>
    <td>&nbsp;</td>
    <td class="result-snippet">
      SGLang is a fast serving framework for large language models and vision-language models.
    </td>
  </tr>
</table>
</body>
</html>
"""

MOCK_TAVILY_JSON = {
    "query": "autonomous agents",
    "results": [
        {
            "title": "AutoGPT: Autonomous Agent Platform",
            "url": "https://github.com/Significant-Gravitas/AutoGPT",
            "content": "AutoGPT is an open-source autonomous agent platform.",
            "score": 0.98,
            "published_date": "2026-09-15",
        }
    ],
}

MOCK_EXA_JSON = {
    "results": [
        {
            "title": "Hermes Agent Framework",
            "url": "https://github.com/NousResearch/Hermes-Function-Calling",
            "highlights": [
                "Hermes function calling and autonomous agent execution harness.",
                "Optimized for structured tool calling and low latency.",
            ],
            "text": "Full text documentation for Hermes agent...",
            "score": 0.94,
            "publishedDate": "2026-09-16",
            "author": "NousResearch",
        }
    ],
    "resolvedSearchType": "neural",
}


# ============================================================================
# 1. TestURLNormalizer
# ============================================================================

class TestURLNormalizer:
    """Verifies URL sanitization, tracking removal, and GitHub repo canonicalization."""

    def test_strip_utm_params(self) -> None:
        raw = "https://example.com/tool?utm_source=twitter&utm_medium=social&utm_campaign=launch&valid_param=1"
        clean = clean_and_canonicalize_url(raw)
        assert "utm_source" not in clean
        assert "utm_medium" not in clean
        assert "utm_campaign" not in clean
        assert clean == "https://example.com/tool?valid_param=1"

    def test_strip_tracking_params(self) -> None:
        raw = "https://example.com/path?fbclid=xyz123&gclid=abc456&ref=producthunt&keep=true"
        clean = clean_and_canonicalize_url(raw)
        assert "fbclid" not in clean
        assert "gclid" not in clean
        assert "ref=" not in clean
        assert clean == "https://example.com/path?keep=true"

    def test_strip_url_fragments(self) -> None:
        raw = "https://github.com/vllm-project/vllm#installation-instructions"
        clean = clean_and_canonicalize_url(raw)
        assert "#" not in clean
        assert clean == "https://github.com/vllm-project/vllm"

    def test_normalize_scheme_and_host(self) -> None:
        # Prepending https, lowercasing host, stripping www.
        assert clean_and_canonicalize_url("HTTP://WWW.GITHUB.COM/vllm-project/vllm") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("www.example.com/page") == "https://example.com/page"
        assert clean_and_canonicalize_url("http://example.com:80/page") == "http://example.com/page"
        assert clean_and_canonicalize_url("https://example.com:443/page") == "https://example.com/page"

    def test_normalize_trailing_slashes(self) -> None:
        assert clean_and_canonicalize_url("https://example.com/path///") == "https://example.com/path"
        assert clean_and_canonicalize_url("https://example.com/") == "https://example.com"

    def test_canonicalize_github_repo_urls(self) -> None:
        # Canonicalization of repo paths and subpaths
        assert clean_and_canonicalize_url("https://github.com/vllm-project/vllm.git") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("https://github.com/vllm-project/vllm/tree/main") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("https://github.com/vllm-project/vllm/blob/master/README.md") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("https://github.com/vllm-project/vllm/releases/tag/v0.6.0") == "https://github.com/vllm-project/vllm"

    def test_canonicalize_bare_github_slugs(self) -> None:
        assert clean_and_canonicalize_url("vllm-project/vllm") == "https://github.com/vllm-project/vllm"
        assert clean_and_canonicalize_url("ggerganov/llama.cpp.git") == "https://github.com/ggerganov/llama.cpp"

    def test_github_reserved_roots(self) -> None:
        # Reserved words must not be treated as repos
        assert clean_and_canonicalize_url("https://github.com/explore") == "https://github.com/explore"
        assert clean_and_canonicalize_url("https://github.com/pricing") == "https://github.com/pricing"
        assert extract_github_repo("https://github.com/explore") is None
        assert extract_github_repo("explore/features") is None

    def test_duckduckgo_redirect_unwrapping(self) -> None:
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm&rut=123"
        clean = clean_and_canonicalize_url(wrapped)
        assert clean == "https://github.com/vllm-project/vllm"

    def test_extract_github_repo(self) -> None:
        assert extract_github_repo("https://github.com/vllm-project/vllm") == "vllm-project/vllm"
        assert extract_github_repo("vllm-project/vllm") == "vllm-project/vllm"
        assert extract_github_repo("https://example.com/other") is None
        assert extract_github_repo("") is None

    def test_extract_github_repo_url(self) -> None:
        assert extract_github_repo_url("https://github.com/vllm-project/vllm/tree/main") == "https://github.com/vllm-project/vllm"
        assert extract_github_repo_url("https://example.com", text="Check out https://github.com/ollama/ollama!") == "https://github.com/ollama/ollama"
        assert extract_github_repo_url("https://example.com", text="No repo here") is None

    def test_extract_all_github_urls(self) -> None:
        text = "See https://github.com/vllm-project/vllm and also https://github.com/ggerganov/llama.cpp.git for details."
        extracted = extract_all_github_urls(text)
        assert len(extracted) == 2
        assert "https://github.com/vllm-project/vllm" in extracted
        assert "https://github.com/ggerganov/llama.cpp" in extracted

    def test_edge_cases_and_malformed_urls(self) -> None:
        assert clean_and_canonicalize_url("") == ""
        assert clean_and_canonicalize_url("   ") == ""
        assert clean_and_canonicalize_url(None) == ""
        assert normalize_url("https://example.com") == "https://example.com"


# ============================================================================
# 2. TestBaseSearchProvider
# ============================================================================

class TestBaseSearchProvider:
    """Verifies BaseSearchProvider lifecycle, SearchResult contracts, and error hierarchy."""

    def test_search_result_dataclass_defaults(self) -> None:
        res = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test Snippet",
            source_engine="searxng",
            raw_score=0.85,
        )
        assert res.title == "Test Title"
        assert res.url == "https://example.com"
        assert res.engine == "searxng"
        assert res.score == 0.85
        d = res.to_dict()
        assert d["title"] == "Test Title"
        assert d["source_engine"] == "searxng"
        assert d["engine"] == "searxng"
        assert d["raw_score"] == 0.85

    def test_search_error_hierarchy(self) -> None:
        err = SearchTimeoutError("timeout")
        assert isinstance(err, SearchError)
        err2 = SearchRateLimitError("rate limited", retry_after=30.0)
        assert isinstance(err2, SearchError)
        assert err2.retry_after == 30.0
        err3 = SearchUpstreamError("server down", status_code=502)
        assert isinstance(err3, SearchError)
        assert err3.status_code == 502
        assert isinstance(SearchParseError("parse error"), SearchError)
        assert isinstance(SearchUnavailableError("unavailable"), SearchError)
        assert isinstance(SearchProviderError("provider error"), SearchError)

    @pytest.mark.asyncio
    async def test_base_search_provider_abstract_interface(self) -> None:
        class DummyProvider(BaseSearchProvider):
            @property
            def name(self) -> str:
                return "dummy"

            def is_available(self) -> bool:
                return True

            async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
                return [
                    SearchResult(
                        title=f"Result for {query}",
                        url=f"https://example.com/{query}",
                        snippet="Dummy snippet",
                        source_engine=self.name,
                    )
                ]

        provider = DummyProvider()
        assert provider.name == "dummy"
        assert provider.is_available() is True
        assert provider.requires_api_key is False

        async with provider as p:
            results = await p.search("test", limit=5)
            assert len(results) == 1
            assert results[0].title == "Result for test"


# ============================================================================
# 3. TestSearXNGProvider
# ============================================================================

class TestSearXNGProvider:
    """Verifies SearXNGSearchProvider querying, JSON parsing, tag cleaning, and errors."""

    @pytest.mark.asyncio
    async def test_searxng_valid_json_response(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/search" in str(request.url)
            assert "format=json" in str(request.url)
            return httpx.Response(200, json=MOCK_SEARXNG_JSON)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        results = await provider.search("vllm inference", limit=10)

        assert len(results) == 2
        first = results[0]
        assert first.title == "vLLM: Easy, fast, and cheap LLM serving"  # <b> tags cleaned
        assert first.url == "https://github.com/vllm-project/vllm"
        assert "high-throughput and memory-efficient" in first.snippet
        assert first.source_engine == "searxng"
        assert first.raw_score == 3.5
        assert first.published_date == "2026-09-18T10:00:00Z"
        assert first.metadata["engine"] == "github"

    @pytest.mark.asyncio
    async def test_searxng_empty_results(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"query": "nothing", "results": []})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        results = await provider.search("nothing", limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_searxng_server_error_500(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        with pytest.raises(SearchUpstreamError) as exc_info:
            await provider.search("error query")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_searxng_rate_limit_429(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate Limit Exceeded")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        with pytest.raises(SearchRateLimitError):
            await provider.search("rate limited query")

    @pytest.mark.asyncio
    async def test_searxng_timeout_handling(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timeout querying SearXNG")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        with pytest.raises(SearchTimeoutError):
            await provider.search("timeout query")

    @pytest.mark.asyncio
    async def test_searxng_malformed_json(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>502 Bad Gateway</html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = SearXNGSearchProvider(settings=test_settings, client=client)
        with pytest.raises(SearchParseError):
            await provider.search("bad json")

    @pytest.mark.asyncio
    async def test_searxng_disabled_availability(self, test_settings: Settings) -> None:
        provider = SearXNGSearchProvider(settings=test_settings, enabled=False)
        assert provider.is_available() is False
        with pytest.raises(SearchUnavailableError):
            await provider.search("anything")


# ============================================================================
# 4. TestDuckDuckGoProvider
# ============================================================================

class TestDuckDuckGoProvider:
    """Verifies DuckDuckGoSearchProvider HTML/Lite parsing, redirect decoding, and fallbacks."""

    @pytest.mark.asyncio
    async def test_duckduckgo_html_parsing_success(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "duckduckgo.com" in str(request.url)
            return httpx.Response(200, text=MOCK_DDG_HTML)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = DuckDuckGoSearchProvider(settings=test_settings, client=client)
        results = await provider.search("vllm", limit=5)

        assert len(results) == 2
        assert results[0].title == "vLLM: High-Throughput Serving Engine"
        assert results[0].url == "https://github.com/vllm-project/vllm"
        assert "PagedAttention" in results[0].snippet
        assert results[0].source_engine == "duckduckgo"
        assert results[0].raw_score == 1.0

        assert results[1].title == "llama.cpp: LLM inference in C/C++"
        assert results[1].url == "https://github.com/ggerganov/llama.cpp"

    @pytest.mark.asyncio
    async def test_duckduckgo_uddg_redirect_decoding(self) -> None:
        parser = _DuckDuckGoHTMLParser(limit=5)
        raw_href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fvllm-project%2Fvllm%3Futm_source%3Dddg&rut=abc"
        dest = parser._extract_destination_url(raw_href)
        assert dest == "https://github.com/vllm-project/vllm?utm_source=ddg"

    @pytest.mark.asyncio
    async def test_duckduckgo_empty_html_results(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>No results found</body></html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = DuckDuckGoSearchProvider(settings=test_settings, client=client)
        results = await provider.search("nonexistent", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_duckduckgo_lite_fallback(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "/html/" in url_str:
                # Primary HTML endpoint triggers anti-bot challenge (202)
                return httpx.Response(202, text="Challenge")
            if "/lite/" in url_str:
                # Lite fallback returns valid table
                return httpx.Response(200, text=MOCK_DDG_LITE_HTML)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = DuckDuckGoSearchProvider(settings=test_settings, client=client)
        results = await provider.search("sglang", limit=5)

        assert len(results) == 1
        assert results[0].title == "SGLang: Fast Serving Framework"
        assert results[0].url == "https://github.com/sgl-project/sglang"
        assert "vision-language models" in results[0].snippet

    @pytest.mark.asyncio
    async def test_duckduckgo_anti_bot_backoff(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>anomaly-detected for bots</body></html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = DuckDuckGoSearchProvider(settings=test_settings, client=client)
        results = await provider.search("test", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_duckduckgo_disabled_availability(self, test_settings: Settings) -> None:
        provider = DuckDuckGoSearchProvider(settings=test_settings, enabled=False)
        assert provider.is_available() is False
        with pytest.raises(SearchUnavailableError):
            await provider.search("test")


# ============================================================================
# 5. TestTavilyProvider
# ============================================================================

class TestTavilyProvider:
    """Verifies TavilySearchProvider availability check, API querying, and error shielding."""

    def test_tavily_is_available_without_key(self, test_settings: Settings) -> None:
        provider = TavilySearchProvider(api_key="", settings=test_settings)
        assert provider.is_available() is False
        assert provider.requires_api_key is True

    def test_tavily_is_available_with_key(self, test_settings: Settings) -> None:
        provider = TavilySearchProvider(api_key="tvly-test-key-123", settings=test_settings)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_tavily_valid_search(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "api.tavily.com" in str(request.url)
            body = json.loads(request.content)
            assert body["api_key"] == "tvly-valid-key"
            assert body["query"] == "autonomous agents"
            return httpx.Response(200, json=MOCK_TAVILY_JSON)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = TavilySearchProvider(
            api_key="tvly-valid-key",
            settings=test_settings,
            client=client,
        )
        results = await provider.search("autonomous agents", limit=5)

        assert len(results) == 1
        res = results[0]
        assert res.title == "AutoGPT: Autonomous Agent Platform"
        assert res.url == "https://github.com/Significant-Gravitas/AutoGPT"
        assert res.source_engine == "tavily"
        assert res.raw_score == 0.98

    @pytest.mark.asyncio
    async def test_tavily_skips_when_unavailable(self, test_settings: Settings) -> None:
        provider = TavilySearchProvider(api_key="", settings=test_settings)
        # Must return empty list with zero network requests
        results = await provider.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_tavily_http_401_and_429(self, test_settings: Settings) -> None:
        for status in [401, 429, 500]:
            def handler(request: httpx.Request, s=status) -> httpx.Response:
                return httpx.Response(s, text="API Error")

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)

            provider = TavilySearchProvider(api_key="key", settings=test_settings, client=client)
            results = await provider.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_tavily_timeout_and_error(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timeout")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = TavilySearchProvider(api_key="key", settings=test_settings, client=client)
        results = await provider.search("test")
        assert results == []


# ============================================================================
# 6. TestExaProvider
# ============================================================================

class TestExaProvider:
    """Verifies ExaSearchProvider availability check, highlights/text parsing, and error shielding."""

    def test_exa_is_available_without_key(self, test_settings: Settings) -> None:
        provider = ExaSearchProvider(api_key="", settings=test_settings)
        assert provider.is_available() is False
        assert provider.requires_api_key is True

    def test_exa_is_available_with_key(self, test_settings: Settings) -> None:
        provider = ExaSearchProvider(api_key="exa-test-key-456", settings=test_settings)
        assert provider.is_available() is True

    @pytest.mark.asyncio
    async def test_exa_valid_search_with_highlights(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "api.exa.ai" in str(request.url)
            assert request.headers.get("x-api-key") == "exa-key"
            return httpx.Response(200, json=MOCK_EXA_JSON)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = ExaSearchProvider(api_key="exa-key", settings=test_settings, client=client)
        results = await provider.search("hermes", limit=5)

        assert len(results) == 1
        res = results[0]
        assert res.title == "Hermes Agent Framework"
        assert res.url == "https://github.com/NousResearch/Hermes-Function-Calling"
        assert "Hermes function calling" in res.snippet
        assert res.source_engine == "exa"
        assert res.raw_score == 0.94
        assert res.metadata["author"] == "NousResearch"

    @pytest.mark.asyncio
    async def test_exa_valid_search_fallback_to_text(self, test_settings: Settings) -> None:
        payload = {
            "results": [
                {
                    "title": "Fallback Text Tool",
                    "url": "https://example.com/tool",
                    "highlights": [],
                    "text": "This is raw fallback text for snippet extraction.",
                }
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        provider = ExaSearchProvider(api_key="exa-key", settings=test_settings, client=client)
        results = await provider.search("test", limit=5)

        assert len(results) == 1
        assert "This is raw fallback text" in results[0].snippet

    @pytest.mark.asyncio
    async def test_exa_skips_when_unavailable(self, test_settings: Settings) -> None:
        provider = ExaSearchProvider(api_key="", settings=test_settings)
        results = await provider.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_exa_error_handling_401_429_timeout(self, test_settings: Settings) -> None:
        for status in [401, 429, 503]:
            def handler(request: httpx.Request, s=status) -> httpx.Response:
                return httpx.Response(s, text="Error")

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)
            provider = ExaSearchProvider(api_key="key", settings=test_settings, client=client)
            assert await provider.search("test") == []


# ============================================================================
# 7. TestMultiSearchAggregator
# ============================================================================

class MockProvider(BaseSearchProvider):
    """Test double for search providers."""

    def __init__(
        self,
        name: str,
        results: list[SearchResult],
        available: bool = True,
        delay_seconds: float = 0.0,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self._name = name
        self.results = results
        self.available = available
        self.delay_seconds = delay_seconds
        self.fail = fail

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self.available

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[SearchResult]:
        if self.fail:
            raise SearchUpstreamError("Simulated upstream explosion", status_code=500)
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        return self.results[:limit]


class TestMultiSearchAggregator:
    """Verifies parallel search, timeout shielding, RRF ranking, and SQLite deduplication."""

    @pytest.mark.asyncio
    async def test_aggregator_parallel_execution(self, test_settings: Settings) -> None:
        res1 = SearchResult("vLLM", "https://github.com/vllm-project/vllm", "Snippet 1", "searxng")
        res2 = SearchResult("llama.cpp", "https://github.com/ggerganov/llama.cpp", "Snippet 2", "duckduckgo")

        p1 = MockProvider("searxng", [res1])
        p2 = MockProvider("duckduckgo", [res2])

        aggregator = MultiSearchAggregator(providers=[p1, p2], settings=test_settings)
        results = await aggregator.search("local inference", limit_per_provider=5)

        assert len(results) == 2
        urls = [r.url for r in results]
        assert "https://github.com/vllm-project/vllm" in urls
        assert "https://github.com/ggerganov/llama.cpp" in urls

    @pytest.mark.asyncio
    async def test_aggregator_per_provider_timeout_isolation(self, test_settings: Settings) -> None:
        fast_res = SearchResult("Fast Tool", "https://github.com/fast/tool", "Fast snippet", "searxng")
        p_fast = MockProvider("fast", [fast_res], delay_seconds=0.01)
        p_slow = MockProvider("slow", [], delay_seconds=1.5)

        aggregator = MultiSearchAggregator(
            providers=[p_fast, p_slow],
            settings=test_settings,
            timeout_seconds=0.1,  # Short timeout for slow provider
        )
        results = await aggregator.search("query")

        assert len(results) == 1
        assert results[0].url == "https://github.com/fast/tool"

    @pytest.mark.asyncio
    async def test_aggregator_error_isolation(self, test_settings: Settings) -> None:
        healthy_res = SearchResult("Healthy", "https://github.com/healthy/repo", "Healthy snippet", "searxng")
        p_healthy = MockProvider("searxng", [healthy_res])
        p_crashing = MockProvider("broken", [], fail=True)

        aggregator = MultiSearchAggregator(
            providers=[p_healthy, p_crashing],
            settings=test_settings,
        )
        results = await aggregator.search("query")

        assert len(results) == 1
        assert results[0].url == "https://github.com/healthy/repo"

    @pytest.mark.asyncio
    async def test_aggregator_cross_engine_deduplication(self, test_settings: Settings) -> None:
        # Same repository found with different query params/subpaths by 2 engines
        res1 = SearchResult(
            title="vLLM: LLM Serving",
            url="https://github.com/vllm-project/vllm?utm_source=searx",
            snippet="High throughput engine.",
            source_engine="searxng",
            metadata={"rank": 1},
        )
        res2 = SearchResult(
            title="vLLM Project Official Repo",
            url="https://github.com/vllm-project/vllm/tree/main#readme",
            snippet="PagedAttention architecture.",
            source_engine="duckduckgo",
            metadata={"rank": 2},
        )

        p1 = MockProvider("searxng", [res1])
        p2 = MockProvider("duckduckgo", [res2])

        aggregator = MultiSearchAggregator(providers=[p1, p2], settings=test_settings)
        results = await aggregator.search("vllm")

        assert len(results) == 1
        merged = results[0]
        assert merged.url == "https://github.com/vllm-project/vllm"
        assert set(merged.metadata["discovered_by"]) == {"searxng", "duckduckgo"}
        assert merged.metadata["multi_source_count"] == 2
        assert merged.metadata["github_repo"] == "vllm-project/vllm"
        assert "https://github.com/vllm-project/vllm" in merged.metadata["extracted_github_urls"]

    @pytest.mark.asyncio
    async def test_aggregator_multi_source_confirmation_boost(self, test_settings: Settings) -> None:
        # Single-source item at rank 1 vs multi-source item
        single_item = SearchResult("Solo Item", "https://github.com/solo/tool", "Solo", "searxng", metadata={"rank": 1})
        multi_item1 = SearchResult("Multi Item", "https://github.com/multi/tool", "Multi 1", "searxng", metadata={"rank": 3})
        multi_item2 = SearchResult("Multi Item", "https://github.com/multi/tool", "Multi 2", "duckduckgo", metadata={"rank": 3})

        p1 = MockProvider("searxng", [single_item, multi_item1])
        p2 = MockProvider("duckduckgo", [multi_item2])

        aggregator = MultiSearchAggregator(providers=[p1, p2], settings=test_settings)
        results = await aggregator.search("tools")

        assert len(results) == 2
        # Multi item receives +0.2 boost for 2 engines and should rank higher
        assert results[0].url == "https://github.com/multi/tool"
        assert results[1].url == "https://github.com/solo/tool"

    @pytest.mark.asyncio
    async def test_aggregator_snippet_merging(self, test_settings: Settings) -> None:
        res1 = SearchResult("Tool", "https://github.com/owner/tool", "Feature A", "searxng")
        res2 = SearchResult("Tool", "https://github.com/owner/tool", "Feature B", "duckduckgo")

        aggregator = MultiSearchAggregator(providers=[MockProvider("s", [res1]), MockProvider("d", [res2])], settings=test_settings)
        results = await aggregator.search("tool")

        assert len(results) == 1
        assert "Feature A" in results[0].snippet
        assert "Feature B" in results[0].snippet

    @pytest.mark.asyncio
    async def test_aggregator_sqlite_deduplication_against_raw_items(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        # Seed an existing item in SQLite raw_items table
        existing_url = "https://github.com/vllm-project/vllm"
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('github_releases', 'vllm:v0.1', 'vLLM', :url, 'content', '{}', '2026-09-18T00:00:00Z');
            """,
            {"url": existing_url},
        )

        res1 = SearchResult("vLLM", existing_url, "Snippet", "searxng")
        res2 = SearchResult("New Tool", "https://github.com/new/tool", "New Snippet", "searxng")

        aggregator = MultiSearchAggregator(providers=[], settings=test_settings, db=migrated_db)
        filtered = await aggregator.filter_existing_raw_items([res1, res2])

        assert len(filtered) == 1
        assert filtered[0].url == "https://github.com/new/tool"

    def test_aggregator_to_raw_items_conversion(self, test_settings: Settings) -> None:
        aggregator = MultiSearchAggregator(providers=[], settings=test_settings)
        search_res = SearchResult(
            title="SGLang",
            url="https://github.com/sgl-project/sglang",
            snippet="Fast serving framework.",
            source_engine="multi_search",
            raw_score=0.25,
            metadata={"github_repo": "sgl-project/sglang", "discovered_by": ["searxng"]},
        )

        raw_items = aggregator.to_raw_items([search_res], query="sglang fast serving")
        assert len(raw_items) == 1
        item = raw_items[0]
        assert item.source == "web_search"
        assert item.source_id.startswith("search:")
        assert item.title == "SGLang"
        assert item.url == "https://github.com/sgl-project/sglang"
        assert item.metadata["source_type"] == "search"
        assert item.metadata["search_query"] == "sglang fast serving"
        assert item.metadata["github_repo"] == "sgl-project/sglang"


# ============================================================================
# 8. TestSearchIngestionSource
# ============================================================================

class TestSearchIngestionSource:
    """Verifies SearchIngestionSource integration with Stage 1 heuristics and pipeline."""

    @pytest.mark.asyncio
    async def test_search_ingestion_yields_raw_items(self, test_settings: Settings) -> None:
        res = SearchResult(
            title="Qdrant Fast Vector Search",
            url="https://github.com/qdrant/qdrant",
            snippet="High-performance vector search engine.",
            source_engine="searxng",
            raw_score=0.9,
        )
        mock_agg = MultiSearchAggregator(providers=[MockProvider("m", [res])], settings=test_settings)

        source = SearchIngestionSource(
            settings=test_settings,
            aggregator=mock_agg,
            queries=["qdrant vector search"],
        )

        items: list[RawItem] = []
        async for item in source.fetch_items():
            items.append(item)

        assert len(items) == 1
        assert items[0].source == "web_search"
        assert items[0].url == "https://github.com/qdrant/qdrant"
        assert items[0].metadata["github_repo"] == "qdrant/qdrant"
        assert "https://github.com/qdrant/qdrant" in items[0].metadata["extracted_github_urls"]

    @pytest.mark.asyncio
    async def test_search_ingestion_stage1_heuristic_pass(self, test_settings: Settings) -> None:
        # Verify that a SearchIngestionSource RawItem passes Stage 1 deterministic heuristics
        res = SearchResult(
            title="Docling Multimodal Parser",
            url="https://github.com/DS4SD/docling",
            snippet="Fast document and formula parsing for RAG.",
            source_engine="multi_search",
        )
        mock_agg = MultiSearchAggregator(providers=[MockProvider("m", [res])], settings=test_settings)
        source = SearchIngestionSource(settings=test_settings, aggregator=mock_agg, queries=["docling"])

        raw_item: Optional[RawItem] = None
        async for item in source.fetch_items():
            raw_item = item
            break

        assert raw_item is not None
        filter_engine = Stage1Filter()
        evaluation = filter_engine.evaluate(raw_item)

        # Passes because github_repo / extracted_github_urls gives has_runnable_code = True
        assert evaluation.passed is True
        assert evaluation.has_runnable_code is True

    @pytest.mark.asyncio
    async def test_search_ingestion_pipeline_cycle_persistence(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        res = SearchResult(
            title="vLLM: Fast Serving",
            url="https://github.com/vllm-project/vllm",
            snippet="PagedAttention NVFP4 kernel execution.",
            source_engine="multi_search",
        )
        mock_agg = MultiSearchAggregator(
            providers=[MockProvider("m", [res])],
            settings=test_settings,
            db=migrated_db,
        )
        search_src = SearchIngestionSource(
            settings=test_settings,
            aggregator=mock_agg,
            db=migrated_db,
            queries=["vllm kernel"],
        )

        pipeline = IngestionPipeline(
            db=migrated_db,
            settings=test_settings,
            sources=[search_src],
        )

        metrics = await pipeline.run_cycle()
        assert metrics["total_inserted_or_updated"] == 1
        assert metrics["sources"]["web_search"]["status"] == "success"

        # Verify persisted in SQLite raw_items table
        rows = await migrated_db.fetch_all("SELECT url, title, source FROM raw_items WHERE source = 'web_search';")
        assert len(rows) == 1
        assert rows[0]["url"] == "https://github.com/vllm-project/vllm"
        assert rows[0]["title"] == "vLLM: Fast Serving"
