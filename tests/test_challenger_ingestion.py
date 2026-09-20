"""tests/test_challenger_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenge & Adversarial Stress Suite for NewsScout Ingestion Subsystem:
1. GitHub Releases & Atom Feed Parsing:
   - Truncated Atom XML payloads and mid-tag corruption
   - Empty feeds, non-XML HTML error payloads (Cloudflare / GitHub 502 HTML)
   - Missing entry tags (<title>, <link>, <updated>, <content>, href-less link)
   - Atom ID collisions when entries lack release tags and colons
   - HTML stripping regex resilience on massive and malformed payloads
2. Network Faults, Rate Limits & Disconnects:
   - HTTP 429 rate limit with Retry-After backoff and eventual recovery
   - HTTP 429 rate limit exhaustion raising HTTPStatusError
   - HTTP 403 with x-ratelimit-remaining: 0 vs non-rate-limit 403
   - Server errors (500, 502, 503) retry backoff and recovery vs exhaustion
   - Caught network errors (ConnectError, ReadTimeout, WriteTimeout)
   - Empirical defect demonstration: Unhandled network errors (ConnectTimeout, RemoteProtocolError)
     bypassing retries in BaseIngestionSource.request_with_retry
3. Hacker News Firebase Ingestion Edge Cases:
   - Zero score and zero descendants dropped
   - Division by zero prevention when score is 0
   - Non-numeric score and descendants crashing unprotected filter (Empirical Defect)
   - Missing 'id' field in payload raising KeyError (Empirical Defect)
   - Batch abortion: single malformed story crashing entire fetch_items generator
   - Extreme comment ratios (e.g. 80.0x comments-to-score)
   - Negative score/descendants and dead/deleted stories
   - Ask/Show HN permalinks and regex extraction of GitHub URLs
   - Malformed topstories.json payloads (null, dict, non-numeric)
4. Hugging Face Daily Papers Ingestion Edge Cases:
   - Theoretical papers without code strictly dropped
   - Non-GitHub repository URLs (GitLab, Bitbucket, personal URLs) strictly dropped
   - Empirical defect demonstration: bare repo slug ('vllm-project/vllm') dropped by extract_github_repo_url
   - Non-numeric upvotes crashing unprotected processor (Empirical Defect)
   - Batch abortion: single malformed paper crashing entire fetch_items generator
   - Markdown edge cases (unclosed links, code fences, XSS tags, trailing parens)
5. IngestionPipeline Concurrency & SQLite WAL Deduplication:
   - In-batch duplicate deduplication in single execute_many call
   - Timestamp immutability: ingested_at preserved across ON CONFLICT updates
   - Multi-task concurrent persistence on single Database instance
   - Multi-connection concurrent persistence on multiple Database instances
   - High-volume burst concurrency with overlapping duplicate items
   - Pipeline run_cycle error isolation across healthy and failing sources
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

import httpx
import pytest

from newsscout.config import Settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.ingestion.github import (
    GitHubIngestionSource,
    strip_html_tags,
)
from newsscout.ingestion.hackernews import (
    AI_TOPIC_PATTERN,
    HN_BASE_URL,
    HackerNewsIngestionSource,
)
from newsscout.ingestion.huggingface import (
    HF_DAILY_PAPERS_URL,
    HuggingFaceIngestionSource,
    extract_github_repo_url,
)
from newsscout.ingestion.pipeline import IngestionPipeline
from newsscout.storage.db import Database, create_connection
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import RawItem


# ============================================================================
# 1. GitHub Releases & Atom Feed Parsing Stress & Corruption
# ============================================================================

class TestGitHubAtomFeedStressAndCorruption:
    """Adversarial stress-testing of Atom XML ingestion and HTML stripping."""

    @pytest.mark.asyncio
    async def test_truncated_atom_xml_mid_tag_raises_parse_error(self, test_settings: Settings) -> None:
        """Empirically verifies that truncated XML in _fetch_via_atom raises ET.ParseError."""
        truncated_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<feed xmlns="http://www.w3.org/2005/Atom">\n'
            '  <entry>\n'
            '    <title>v1.0.0</title>\n'
            '    <content type="html"><p>Incomplete tag without clo'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=truncated_xml)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["vllm-project/vllm"], force_atom_feed=True
        ) as source:
            with pytest.raises(ET.ParseError):
                async for _ in source._fetch_via_atom("vllm-project/vllm"):
                    pass

    @pytest.mark.asyncio
    async def test_truncated_atom_xml_handled_gracefully_in_fetch_items(self, test_settings: Settings) -> None:
        """Verifies fetch_items isolates per-repo XML ParseErrors so subsequent repos succeed."""
        valid_atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>tag:github.com,2008:Repository/12345/v1.0</id>
            <link rel="alternate" type="text/html" href="https://github.com/good/repo/releases/tag/v1.0"/>
            <title>v1.0: Stable</title>
            <content type="html">Clean release notes.</content>
          </entry>
        </feed>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "bad/repo" in url_str:
                return httpx.Response(200, text="<feed><entry><title>Trun")
            if "good/repo" in url_str:
                return httpx.Response(200, text=valid_atom_xml)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings,
            client=client,
            target_repos=["bad/repo", "good/repo"],
            force_atom_feed=True,
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 1
        assert items[0].source_id == "good/repo:v1.0"

    @pytest.mark.asyncio
    async def test_empty_atom_feed_raises_parse_error(self, test_settings: Settings) -> None:
        """Verifies empty or whitespace-only Atom responses raise ET.ParseError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="   \n  ")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["vllm-project/vllm"], force_atom_feed=True
        ) as source:
            with pytest.raises(ET.ParseError):
                async for _ in source._fetch_via_atom("vllm-project/vllm"):
                    pass

    @pytest.mark.asyncio
    async def test_atom_feed_returning_html_error_payload(self, test_settings: Settings) -> None:
        """Tests behavior when GitHub/Cloudflare returns HTML error page instead of Atom XML."""
        html_page = (
            "<!DOCTYPE html>\n"
            "<html><head><title>502 Bad Gateway</title></head>\n"
            "<body><center><h1>502 Bad Gateway</h1></center><hr>cloudflare</body></html>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=html_page)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["vllm-project/vllm"], force_atom_feed=True
        ) as source:
            # HTML without atom entries: either ParseError or yields 0 items
            try:
                items = [it async for it in source._fetch_via_atom("vllm-project/vllm")]
                assert len(items) == 0
            except ET.ParseError:
                pass  # ParseError is also an acceptable rejection for non-XML HTML

    @pytest.mark.asyncio
    async def test_atom_feed_missing_entry_tags(self, test_settings: Settings) -> None:
        """Tests Atom feeds where entries miss optional elements: title, link href, updated, content."""
        sparse_atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <!-- Missing title, link has no href, missing updated, missing content -->
            <id>tag:github.com,2008:Repository/1/sparse-entry</id>
            <link rel="alternate" type="text/html" />
          </entry>
          <entry>
            <!-- Title with colon for tag fallback, missing link entirely -->
            <title>v2.1.0: Performance Release</title>
            <updated>2026-09-18T00:00:00Z</updated>
            <content type="html">&lt;p&gt;Speedups&lt;/p&gt;</content>
          </entry>
        </feed>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sparse_atom_xml)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["test/sparse-repo"], force_atom_feed=True
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 2
        # First entry fallback defaults
        assert items[0].title == "test/sparse-repo Release"
        assert items[0].url == "https://github.com/test/sparse-repo"
        assert items[0].metadata["tag"] == "release"
        assert items[0].raw_content == ""

        # Second entry title tag extraction
        assert items[1].metadata["tag"] == "v2.1.0"
        assert items[1].raw_content == "Speedups"
        assert items[1].metadata["published_at"] == "2026-09-18T00:00:00Z"

    @pytest.mark.asyncio
    async def test_atom_feed_duplicate_entries_with_no_tag_or_colon(self, test_settings: Settings) -> None:
        """Empirically demonstrates that entries without /releases/tag/ or colons collide on same source_id."""
        colliding_atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <title>Weekly Release Update 1</title>
            <link rel="alternate" type="text/html" href="https://github.com/test/colliding/releases"/>
            <content type="html">Update 1 content</content>
          </entry>
          <entry>
            <title>Weekly Release Update 2</title>
            <link rel="alternate" type="text/html" href="https://github.com/test/colliding/releases"/>
            <content type="html">Update 2 content</content>
          </entry>
        </feed>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=colliding_atom_xml)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["test/colliding"], force_atom_feed=True
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 2
        # Both share identical source_id because tag fallback defaulted to "release"
        assert items[0].source_id == "test/colliding:release"
        assert items[1].source_id == "test/colliding:release"

    def test_strip_html_tags_stress_and_entities(self) -> None:
        """Stress-tests strip_html_tags with massive payloads, entities, and malformed tags."""
        # 1. None and empty
        assert strip_html_tags("") == ""
        assert strip_html_tags(None) == ""  # type: ignore[arg-type]

        # 2. HTML entities
        entity_html = "&amp; &quot;quoted&quot; &#39;single&#39; &#x2F;slash"
        clean_entities = strip_html_tags(entity_html)
        assert clean_entities == '& "quoted" \'single\' /slash'

        # Escaped HTML tags are unescaped and then stripped
        escaped_tags = "&lt;b&gt;bold text&lt;/b&gt;"
        assert strip_html_tags(escaped_tags) == "bold text"

        # 3. Deeply nested tags
        nested_html = "<div><p><span><b><i><strong>Deep Nesting</strong></i></b></span></p></div>"
        assert strip_html_tags(nested_html) == "Deep Nesting"

        # 4. Unclosed tags
        unclosed_html = "<p>First line<br>Second line with unclosed <b>bold text<p>Third line"
        clean_unclosed = strip_html_tags(unclosed_html)
        assert "First line" in clean_unclosed
        assert "Second line with unclosed bold text" in clean_unclosed
        assert "Third line" in clean_unclosed

        # 5. Large payload (100KB with 1000 tags)
        large_html = "".join(f"<p>Chunk <b>{i}</b>: Some release details.<br/></p>" for i in range(1000))
        clean_large = strip_html_tags(large_html)
        assert len(clean_large) > 10000
        assert "<p>" not in clean_large
        assert "<b>" not in clean_large


# ============================================================================
# 2. Network Faults, Rate Limits & Disconnects
# ============================================================================

class TestNetworkFaultsAndRateLimiting:
    """Stress-tests BaseIngestionSource.request_with_retry under adversarial network conditions."""

    class DummySource(BaseIngestionSource):
        source_name = "dummy_network_test"

        async def fetch_items(self, limit: Optional[int] = None):
            yield  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_rate_limit_429_with_retry_after(self, test_settings: Settings) -> None:
        """Tests HTTP 429 backoff with Retry-After header and eventual success."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 429 with 0 second Retry-After for rapid test execution
                return httpx.Response(429, headers={"Retry-After": "0"}, text="Too Many Requests")
            return httpx.Response(200, json={"status": "ok"})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            resp = await src.request_with_retry("https://api.example.com/test", max_retries=3)

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_429_exhaustion_raises_http_status_error(self, test_settings: Settings) -> None:
        """Tests HTTP 429 exhaustion raises HTTPStatusError after exceeding max_retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(429, headers={"Retry-After": "0"}, text="Rate Limit Exceeded")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await src.request_with_retry("https://api.example.com/test", max_retries=2)

        assert exc_info.value.response.status_code == 429
        # Initial attempt + 2 retries = 3 attempts before raising on attempt 3 (attempt > 2)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_403_with_ratelimit_header(self, test_settings: Settings) -> None:
        """Tests HTTP 403 treated as rate limit when x-ratelimit-remaining: 0 is present."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(403, headers={"x-ratelimit-remaining": "0", "Retry-After": "0"}, text="Forbidden")
            return httpx.Response(200, json={"data": "recovered"})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            resp = await src.request_with_retry("https://api.example.com/test", max_retries=2)

        assert resp.status_code == 200
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_forbidden_403_without_ratelimit_headers_fails_immediately(self, test_settings: Settings) -> None:
        """Tests that non-rate-limit 403 Forbidden raises immediately without wasting retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(403, text="Permission Denied")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await src.request_with_retry("https://api.example.com/test", max_retries=3)

        assert exc_info.value.response.status_code == 403
        assert call_count == 1  # No retries for true 403 Forbidden!

    @pytest.mark.asyncio
    async def test_server_errors_500_502_503_retry_and_recover(self, test_settings: Settings) -> None:
        """Tests 500, 502, 503 transient gateway errors retrying and recovering on 200."""
        statuses = [500, 502, 503, 200]
        idx = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal idx
            st = statuses[min(idx, len(statuses) - 1)]
            idx += 1
            return httpx.Response(st, text=f"Status {st}")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            resp = await src.request_with_retry("https://api.example.com/test", max_retries=4, backoff_base=1.0)

        assert resp.status_code == 200
        assert idx == 4

    @pytest.mark.asyncio
    async def test_server_errors_500_exhaustion_raises_http_status_error(self, test_settings: Settings) -> None:
        """Tests persistent 500 server error raises HTTPStatusError once max_retries exhausted."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await src.request_with_retry("https://api.example.com/test", max_retries=2, backoff_base=1.0)

        assert exc_info.value.response.status_code == 500
        assert call_count == 3  # Attempt 1, Retry 1, Retry 2, then raise

    @pytest.mark.asyncio
    async def test_network_connect_error_and_read_timeout_retries(self, test_settings: Settings) -> None:
        """Tests caught network exceptions (ConnectError, ReadTimeout) are retried."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Failed to establish TCP connection", request=request)
            if call_count == 2:
                raise httpx.ReadTimeout("Socket read timed out", request=request)
            return httpx.Response(200, json={"success": True})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with self.DummySource(settings=test_settings, client=client) as src:
            resp = await src.request_with_retry("https://api.example.com/test", max_retries=3, backoff_base=1.0)

        assert resp.status_code == 200
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_unhandled_network_errors_bypass_retry_defect(self, test_settings: Settings) -> None:
        """EMPIRICAL DEFECT DEMONSTRATION:
        BaseIngestionSource.request_with_retry catches only (ConnectError, ReadTimeout, WriteTimeout).
        It FAILS to catch httpx.ConnectTimeout and httpx.RemoteProtocolError, causing them
        to crash immediately on attempt 1 without retry.
        """
        # Case A: ConnectTimeout
        call_count_a = 0

        def handler_timeout(request: httpx.Request) -> httpx.Response:
            nonlocal call_count_a
            call_count_a += 1
            raise httpx.ConnectTimeout("SYN packet dropped / connection timeout", request=request)

        transport_a = httpx.MockTransport(handler_timeout)
        client_a = httpx.AsyncClient(transport=transport_a)

        async with self.DummySource(settings=test_settings, client=client_a) as src:
            with pytest.raises(httpx.ConnectTimeout):
                await src.request_with_retry("https://api.example.com/test", max_retries=3)

        # Empirically proves ConnectTimeout is retried 3 times (1 initial + 3 retries = 4 calls)
        assert call_count_a == 4, (
            f"Expected ConnectTimeout to retry 3 times (4 calls total), got {call_count_a} calls"
        )

        # Case B: RemoteProtocolError (Server disconnected without sending response)
        call_count_b = 0

        def handler_disconnect(request: httpx.Request) -> httpx.Response:
            nonlocal call_count_b
            call_count_b += 1
            raise httpx.RemoteProtocolError("Server disconnected without sending response", request=request)

        transport_b = httpx.MockTransport(handler_disconnect)
        client_b = httpx.AsyncClient(transport=transport_b)

        async with self.DummySource(settings=test_settings, client=client_b) as src:
            with pytest.raises(httpx.RemoteProtocolError):
                await src.request_with_retry("https://api.example.com/test", max_retries=3)

        # Empirically proves RemoteProtocolError is retried 3 times (1 initial + 3 retries = 4 calls)
        assert call_count_b == 4, (
            f"Expected RemoteProtocolError to retry 3 times (4 calls total), got {call_count_b} calls"
        )


# ============================================================================
# 3. Hacker News Firebase Ingestion Edge Cases
# ============================================================================

class TestHackerNewsEdgeCases:
    """Adversarial testing for HackerNewsIngestionSource ratio filters and schema edge cases."""

    def test_hn_zero_score_and_zero_descendants_dropped(self, test_settings: Settings) -> None:
        """Tests that a story with 0 score and 0 descendants is dropped."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120)
        data = {
            "id": 201,
            "type": "story",
            "title": "vLLM High Performance Inference Engine",
            "score": 0,
            "descendants": 0,
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(data)
        assert item is None

    def test_hn_division_by_zero_safety_when_score_zero(self, test_settings: Settings) -> None:
        """Verifies division-by-zero guard prevents crash when score is 0 and min_score is -1."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=-1, min_comment_ratio=-1.0)
        data = {
            "id": 202,
            "type": "story",
            "title": "vLLM Inference Engine Update",
            "score": 0,
            "descendants": 10,
            "url": "https://github.com/vllm-project/vllm",
        }
        # score = 0 should compute comment_ratio = 0.0 safely without ZeroDivisionError
        item = source._filter_and_normalize(data)
        assert item is not None
        assert item.metadata["score"] == 0
        assert item.metadata["comment_ratio"] == 0.0

    def test_hn_non_numeric_score_crashes_unprotected_filter_defect(self, test_settings: Settings) -> None:
        """Verifies HackerNewsIngestionSource._filter_and_normalize handles non-numeric score safely."""
        source = HackerNewsIngestionSource(settings=test_settings)
        data = {
            "id": 203,
            "type": "story",
            "title": "vLLM Autonomous Agent Framework",
            "score": "NaN",
            "descendants": 50,
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(data)
        assert item is None

    def test_hn_non_numeric_descendants_crashes_unprotected_filter_defect(self, test_settings: Settings) -> None:
        """Verifies non-numeric descendants string defaults to 0 safely without crashing."""
        source = HackerNewsIngestionSource(settings=test_settings)
        data = {
            "id": 204,
            "type": "story",
            "title": "vLLM Autonomous Agent Framework",
            "score": 250,
            "descendants": "N/A",
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(data)
        assert item is None

    def test_hn_missing_id_field_crashes_with_key_error_defect(self, test_settings: Settings) -> None:
        """Verifies payload missing 'id' is safely dropped instead of raising KeyError."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=100)
        data = {
            "type": "story",
            "title": "vLLM Autonomous Agent Framework",
            "score": 250,
            "descendants": 150,
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(data)
        assert item is None

    @pytest.mark.asyncio
    async def test_hn_malformed_story_in_batch_aborts_entire_fetch_items_generator(
        self, test_settings: Settings
    ) -> None:
        """Verifies malformed story #2 does not abort fetch_items; valid stories #1 and #3 are preserved."""
        stories = {
            1: {"id": 1, "type": "story", "title": "vLLM Engine 1", "score": 200, "descendants": 100, "url": "https://github.com/vllm-project/vllm"},
            2: {"id": 2, "type": "story", "title": "vLLM Engine 2 (Corrupted Score)", "score": "corrupted", "descendants": 100, "url": "https://github.com/vllm-project/vllm"},
            3: {"id": 3, "type": "story", "title": "vLLM Engine 3 (Valid)", "score": 200, "descendants": 100, "url": "https://github.com/vllm-project/vllm"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "topstories.json" in url_str:
                return httpx.Response(200, json=[1, 2, 3])
            for sid, payload in stories.items():
                if f"item/{sid}.json" in url_str:
                    return httpx.Response(200, json=payload)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HackerNewsIngestionSource(
            settings=test_settings, client=client, min_score=120, min_comment_ratio=0.4
        ) as source:
            items = await source.fetch_all()
            assert len(items) == 2
            assert {it.source_id for it in items} == {"1", "3"}

    def test_hn_extreme_comment_ratios_admitted(self, test_settings: Settings) -> None:
        """Tests stories with extreme comment ratios (e.g. 80.0x comments-to-score) are admitted."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120, min_comment_ratio=0.4)
        data = {
            "id": 205,
            "type": "story",
            "title": "vLLM Breakthrough Discussion",
            "score": 125,
            "descendants": 10000,  # ratio = 80.0
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(data)
        assert item is not None
        assert item.metadata["comment_ratio"] == 80.0

    def test_hn_borderline_comment_ratio_boundary(self, test_settings: Settings) -> None:
        """Tests exact boundary condition for comment_ratio > 0.4."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120, min_comment_ratio=0.4)

        # Exactly 0.4: descendants=48, score=120 -> 48/120 = 0.4000 <= 0.4 -> DROPPED
        d_exact = {
            "id": 206,
            "type": "story",
            "title": "vLLM Exactly 0.4 Ratio",
            "score": 125,
            "descendants": 50,  # 50 / 125 = 0.4000 <= 0.4
            "url": "https://github.com/vllm-project/vllm",
        }
        assert source._filter_and_normalize(d_exact) is None

        # Slightly above 0.4: descendants=51, score=125 -> 51/125 = 0.408 > 0.4 -> ADMITTED
        d_above = {
            "id": 207,
            "type": "story",
            "title": "vLLM Above 0.4 Ratio",
            "score": 125,
            "descendants": 51,  # 51 / 125 = 0.408 > 0.4
            "url": "https://github.com/vllm-project/vllm",
        }
        item = source._filter_and_normalize(d_above)
        assert item is not None
        assert item.metadata["comment_ratio"] == 0.408

    def test_hn_dead_and_deleted_stories_dropped(self, test_settings: Settings) -> None:
        """Verifies dead or deleted stories are dropped even with high scores and comments."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=100)
        d_dead = {
            "id": 208, "type": "story", "title": "vLLM Dead Story", "score": 500, "descendants": 300,
            "dead": True, "url": "https://github.com/vllm-project/vllm",
        }
        d_deleted = {
            "id": 209, "type": "story", "title": "vLLM Deleted Story", "score": 500, "descendants": 300,
            "deleted": True, "url": "https://github.com/vllm-project/vllm",
        }
        assert source._filter_and_normalize(d_dead) is None
        assert source._filter_and_normalize(d_deleted) is None

    def test_hn_ask_show_hn_permalink_generation(self, test_settings: Settings) -> None:
        """Tests Ask/Show HN posts without URL construct item permalink and populate text."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120, min_comment_ratio=0.4)
        data = {
            "id": 210,
            "type": "story",
            "title": "Ask HN: How are you optimizing vLLM kernels on RTX 4090?",
            "score": 150,
            "descendants": 90,
            "url": "",  # Empty URL
            "text": "We are benchmarking custom Triton kernels for RTX 4090 local inference.",
        }
        item = source._filter_and_normalize(data)
        assert item is not None
        assert item.url == "https://news.ycombinator.com/item?id=210"
        assert item.metadata["is_ask_or_show_hn"] is True
        assert "Triton kernels" in item.raw_content

    def test_hn_extracted_github_urls_trailing_period_defect(self, test_settings: Settings) -> None:
        """Verifies trailing punctuation like periods is stripped from extracted GitHub URLs."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120, min_comment_ratio=0.4)
        data = {
            "id": 211,
            "type": "story",
            "title": "Show HN: Autonomous Agent Frameworks",
            "score": 200,
            "descendants": 100,
            "url": "https://github.com/org1/repo-primary",
            "text": "Check out https://github.com/org2/repo-secondary and https://github.com/org3/repo-tertiary.",
        }
        item = source._filter_and_normalize(data)
        assert item is not None
        extracted = item.metadata["extracted_github_urls"]
        assert "https://github.com/org3/repo-tertiary" in extracted
        assert "https://github.com/org3/repo-tertiary." not in extracted

    def test_hn_extracted_github_urls_clean_whitespace(self, test_settings: Settings) -> None:
        """Verifies clean whitespace-separated GitHub URLs are extracted correctly."""
        source = HackerNewsIngestionSource(settings=test_settings, min_score=120, min_comment_ratio=0.4)
        data = {
            "id": 212,
            "type": "story",
            "title": "Show HN: Autonomous Agent Frameworks",
            "score": 200,
            "descendants": 100,
            "url": "https://github.com/org1/repo-primary",
            "text": "Check out https://github.com/org2/repo-secondary and https://github.com/org3/repo-tertiary for code",
        }
        item = source._filter_and_normalize(data)
        assert item is not None
        extracted = item.metadata["extracted_github_urls"]
        assert len(extracted) == 3
        assert "https://github.com/org1/repo-primary" in extracted
        assert "https://github.com/org2/repo-secondary" in extracted
        assert "https://github.com/org3/repo-tertiary" in extracted

    @pytest.mark.asyncio
    async def test_hn_topstories_malformed_responses(self, test_settings: Settings) -> None:
        """Tests topstories.json returning null, dictionary error, string, or mixed non-digits."""
        for text_payload in ['null', '{"error": "Permission Denied"}', '"bad string"', '[101, "abc", null, 102]']:
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, text=text_payload)

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)

            async with HackerNewsIngestionSource(settings=test_settings, client=client) as src:
                ids = await src._fetch_top_story_ids()

            if "101" in text_payload:
                # Filtered to valid numeric digits only
                assert ids == [101, 102]
            else:
                assert ids == []

    @pytest.mark.asyncio
    async def test_hn_topstories_non_json_crashes_with_json_decode_error_defect(self, test_settings: Settings) -> None:
        """Verifies non-JSON response in topstories.json returns empty list safely without crashing."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HackerNewsIngestionSource(settings=test_settings, client=client) as src:
            ids = await src._fetch_top_story_ids()
            assert ids == []


# ============================================================================
# 4. Hugging Face Daily Papers Ingestion Edge Cases
# ============================================================================

class TestHuggingFaceEdgeCases:
    """Adversarial testing for HuggingFaceIngestionSource code extraction and schemas."""

    def test_hf_paper_without_code_strictly_dropped(self) -> None:
        """Confirms theoretical papers with no code links are strictly dropped."""
        paper = {
            "id": "2409.50001",
            "title": "Theoretical Bounds of Attention",
            "summary": "We prove mathematical convergence without implementation.",
            "upvotes": 50,
        }
        item = {"id": "2409.50001", "paper": paper, "upvotes": 50}
        source = HuggingFaceIngestionSource()
        assert source._process_paper(item) is None

    def test_hf_paper_with_non_github_urls_strictly_dropped(self) -> None:
        """Empirically verifies papers with GitLab, Bitbucket, or personal URLs are dropped.
        
        The SPEC strictly requires linked GitHub implementations.
        """
        non_github_cases = [
            ("https://gitlab.com/group/ai-repo", "GitLab"),
            ("https://bitbucket.org/team/ai-repo", "Bitbucket"),
            ("https://huggingface.co/models/custom", "Hugging Face Model Hub"),
            ("https://ai.stanford.edu/~prof/project", "Personal Academic Page"),
        ]

        source = HuggingFaceIngestionSource()
        for repo_url, platform in non_github_cases:
            paper = {
                "id": "2409.50002",
                "title": f"Paper with {platform} Code",
                "githubRepo": repo_url,
                "projectPage": repo_url,
                "summary": f"Code available at {repo_url}.",
                "upvotes": 25,
            }
            item = {"id": "2409.50002", "paper": paper, "upvotes": 25}
            processed = source._process_paper(item)
            assert processed is None, f"Paper with {platform} URL should have been dropped, but got: {processed}"

    def test_hf_bare_slug_in_github_repo_field_dropped_defect(self) -> None:
        """Verifies bare slug in githubRepo field is normalized to full https://github.com URL."""
        paper_data = {
            "githubRepo": "vllm-project/vllm",  # Bare slug without 'github.com'
            "summary": "High throughput inference engine.",
        }
        detected_url = extract_github_repo_url(paper_data)
        assert detected_url == "https://github.com/vllm-project/vllm"

    def test_hf_non_numeric_upvotes_crashes_processor_defect(self) -> None:
        """Verifies non-numeric upvotes defaults to 0 safely without crashing."""
        source = HuggingFaceIngestionSource()
        item = {
            "id": "2409.50003",
            "paper": {
                "id": "2409.50003",
                "title": "vLLM Paper",
                "upvotes": "invalid",
                "githubRepo": "https://github.com/vllm-project/vllm",
            },
        }
        processed = source._process_paper(item)
        assert processed is None

    @pytest.mark.asyncio
    async def test_hf_malformed_paper_in_batch_aborts_fetch_items(self, test_settings: Settings) -> None:
        """Verifies malformed paper in batch does not abort fetch_items; valid paper is processed."""
        papers_payload = [
            {
                "id": "2409.00001",
                "paper": {
                    "id": "2409.00001",
                    "title": "Corrupt Paper",
                    "upvotes": "corrupted_val",
                    "githubRepo": "https://github.com/org/repo1",
                },
            },
            {
                "id": "2409.00002",
                "paper": {
                    "id": "2409.00002",
                    "title": "Valid Paper",
                    "upvotes": 45,
                    "githubRepo": "https://github.com/org/repo2",
                },
            },
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=papers_payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HuggingFaceIngestionSource(settings=test_settings, client=client) as source:
            items = await source.fetch_all()
            assert len(items) == 1
            assert items[0].source_id == "arxiv:2409.00002"

    def test_hf_markdown_and_repo_extraction_variants(self) -> None:
        """Tests extract_github_repo_url across various markdown, suffix, and URL formatting styles."""
        # 1. Trailing .git stripped
        p1 = {"githubRepo": "https://github.com/vllm-project/vllm.git"}
        assert extract_github_repo_url(p1) == "https://github.com/vllm-project/vllm"

        # 2. Trailing slash stripped
        p2 = {"githubRepo": "https://github.com/vllm-project/vllm/"}
        assert extract_github_repo_url(p2) == "https://github.com/vllm-project/vllm"

        # 3. Inside Markdown link in summary
        p3 = {"summary": "Our full benchmark suite: [Code Repository](https://github.com/DS4SD/docling). Check it out!"}
        assert extract_github_repo_url(p3) == "https://github.com/DS4SD/docling"

        # 4. Enclosed in parentheses
        p4 = {"summary": "Official implementation (see https://github.com/ggerganov/llama.cpp) for details."}
        assert extract_github_repo_url(p4) == "https://github.com/ggerganov/llama.cpp"

        # 5. Malformed markdown with raw HTML / XSS script tags
        p5 = {
            "summary": (
                "<script>alert('xss')</script> "
                "Project code at <a href='https://github.com/sgl-project/sglang'>GitHub</a>"
            )
        }
        assert extract_github_repo_url(p5) == "https://github.com/sgl-project/sglang"

    @pytest.mark.asyncio
    async def test_hf_malformed_api_payload(self, test_settings: Settings) -> None:
        """Tests daily_papers endpoint returning non-list or list with non-dict elements."""
        for text_payload in ['{"error": "rate limit"}', '"not json list"', '[null, "string", 123]']:
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, text=text_payload)

            transport = httpx.MockTransport(handler)
            client = httpx.AsyncClient(transport=transport)

            async with HuggingFaceIngestionSource(settings=test_settings, client=client) as source:
                items = await source.fetch_all()

            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_hf_malformed_api_non_json_crashes_with_json_decode_error_defect(
        self, test_settings: Settings
    ) -> None:
        """Verifies non-JSON response in fetch_items returns empty list safely without crashing."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HuggingFaceIngestionSource(settings=test_settings, client=client) as source:
            items = await source.fetch_all()
            assert items == []


# ============================================================================
# 5. IngestionPipeline Concurrency & SQLite WAL Deduplication
# ============================================================================

class TestIngestionPipelineConcurrencyAndDeduplication:
    """Stress-tests SQLite WAL deduplication and high-concurrency ingestion persistence."""

    @pytest.mark.asyncio
    async def test_in_batch_duplicate_items_deduplication(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """Tests that a single execute_many batch containing identical items succeeds and deduplicates."""
        pipeline = IngestionPipeline(db=migrated_db, settings=test_settings)

        # Batch with 5 identical copies of the same item
        item = RawItem(
            source="github_releases",
            source_id="vllm-project/vllm:v0.6.2",
            title="vLLM v0.6.2",
            url="https://github.com/vllm-project/vllm/releases/tag/v0.6.2",
            raw_content="Content v0.6.2",
            metadata={"batch_iter": 1},
        )
        batch = [item] * 5

        count = await pipeline.persist_raw_items(batch)
        assert count == 5

        # Only 1 unique row must exist in SQLite raw_items
        rows = await migrated_db.fetch_all(
            "SELECT * FROM raw_items WHERE source = ? AND source_id = ?;",
            ("github_releases", "vllm-project/vllm:v0.6.2"),
        )
        assert len(rows) == 1
        assert rows[0]["title"] == "vLLM v0.6.2"

    @pytest.mark.asyncio
    async def test_sequential_updates_preserve_ingested_at(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """Verifies ON CONFLICT DO UPDATE updates metadata but preserves initial ingested_at timestamp."""
        pipeline = IngestionPipeline(db=migrated_db, settings=test_settings)

        t0 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
        item_initial = RawItem(
            source="hacker_news",
            source_id="30001",
            title="Initial Title",
            url="https://news.ycombinator.com/item?id=30001",
            raw_content="Initial text",
            metadata={"score": 150},
            ingested_at=t0,
        )
        await pipeline.persist_raw_items([item_initial])

        row_t0 = await migrated_db.fetch_one(
            "SELECT * FROM raw_items WHERE source_id = '30001';"
        )
        assert row_t0 is not None
        assert row_t0["ingested_at"] == t0.isoformat()
        assert row_t0["title"] == "Initial Title"

        # Update at later time T1
        t1 = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
        item_updated = RawItem(
            source="hacker_news",
            source_id="30001",
            title="Updated Title With More Points",
            url="https://news.ycombinator.com/item?id=30001",
            raw_content=None,  # Should coalesce to initial text!
            metadata={"score": 600, "descendants": 350},
            ingested_at=t1,
        )
        await pipeline.persist_raw_items([item_updated])

        row_t1 = await migrated_db.fetch_one(
            "SELECT * FROM raw_items WHERE source_id = '30001';"
        )
        assert row_t1 is not None
        assert row_t1["title"] == "Updated Title With More Points"
        # Ingested at preserved!
        assert row_t1["ingested_at"] == t0.isoformat()
        # Raw content coalesced to non-null original text!
        assert row_t1["raw_content"] == "Initial text"
        meta = json.loads(row_t1["metadata_json"])
        assert meta["score"] == 600

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_writers_single_db_instance(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """20 concurrent asyncio tasks persisting items simultaneously on the same Database instance."""
        pipeline = IngestionPipeline(db=migrated_db, settings=test_settings)
        num_tasks = 20
        items_per_task = 10

        async def _persist_task(task_id: int) -> int:
            items = [
                RawItem(
                    source="hf_daily_papers",
                    source_id=f"arxiv:2409.{task_id:02d}{i:02d}",
                    title=f"Paper from Task {task_id} item {i}",
                    url=f"https://huggingface.co/papers/2409.{task_id:02d}{i:02d}",
                    raw_content="Abstract content",
                    metadata={"task_id": task_id, "index": i},
                )
                for i in range(items_per_task)
            ]
            return await pipeline.persist_raw_items(items)

        results = await asyncio.gather(*[_persist_task(t) for t in range(num_tasks)])
        assert sum(results) == num_tasks * items_per_task

        total_rows = await migrated_db.fetch_val("SELECT COUNT(*) FROM raw_items WHERE source = 'hf_daily_papers';")
        assert total_rows == num_tasks * items_per_task

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_writers_multiple_db_instances(
        self, temp_db_path: Path, test_settings: Settings
    ) -> None:
        """10 concurrent asyncio tasks, each using an independent Database instance to the same WAL file."""
        # Initialize schema first
        init_db = Database(temp_db_path, timeout=5.0)
        await init_db.initialize()
        with init_db.get_sync_connection() as conn:
            apply_migrations(conn=conn)
        await init_db.close()

        num_instances = 10
        items_per_instance = 10

        async def _multi_db_worker(worker_id: int) -> int:
            db_inst = Database(temp_db_path, timeout=5.0)
            await db_inst.initialize()
            try:
                pipeline = IngestionPipeline(db=db_inst, settings=test_settings)
                items = [
                    RawItem(
                        source="multi_db_test",
                        source_id=f"item_{worker_id}_{i}",
                        title=f"Item from Worker {worker_id} - {i}",
                        url=f"https://example.com/{worker_id}/{i}",
                        raw_content="Content",
                    )
                    for i in range(items_per_instance)
                ]
                return await pipeline.persist_raw_items(items)
            finally:
                await db_inst.close()

        results = await asyncio.gather(*[_multi_db_worker(w) for w in range(num_instances)])
        assert sum(results) == num_instances * items_per_instance

        # Verify final count in DB
        verify_db = Database(temp_db_path, timeout=5.0)
        await verify_db.initialize()
        count = await verify_db.fetch_val("SELECT COUNT(*) FROM raw_items WHERE source = 'multi_db_test';")
        await verify_db.close()
        assert count == num_instances * items_per_instance

    @pytest.mark.asyncio
    async def test_high_volume_concurrency_stress_with_duplicate_bursts(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """25 concurrent tasks persisting overlapping items (burst of concurrent duplicates).
        
        Validates:
        - Zero deadlocks under concurrent SQLite WAL upserts
        - ON CONFLICT DO UPDATE handles concurrent duplicate bursts cleanly
        - Exact number of distinct items match
        """
        pipeline = IngestionPipeline(db=migrated_db, settings=test_settings)
        num_tasks = 25
        distinct_keys = 20  # Overlapping: 25 tasks compete on 20 distinct keys

        async def _burst_writer(task_id: int) -> int:
            items = [
                RawItem(
                    source="burst_test",
                    source_id=f"key_{k}",
                    title=f"Key {k} written by Task {task_id}",
                    url=f"https://example.com/burst/{k}",
                    raw_content=f"Payload {k} from task {task_id}",
                    metadata={"writer": task_id, "timestamp": datetime.now(timezone.utc).isoformat()},
                )
                for k in range(distinct_keys)
            ]
            return await pipeline.persist_raw_items(items)

        results = await asyncio.gather(*[_burst_writer(t) for t in range(num_tasks)])
        assert len(results) == num_tasks

        # Verify that exactly 20 distinct rows exist
        rows = await migrated_db.fetch_all("SELECT * FROM raw_items WHERE source = 'burst_test';")
        assert len(rows) == distinct_keys

    @pytest.mark.asyncio
    async def test_pipeline_run_cycle_error_isolation(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        """Tests that run_cycle isolates a failing source without failing healthy sources."""
        class HealthySource(BaseIngestionSource):
            source_name = "healthy_source"

            async def fetch_items(self, limit: Optional[int] = None):
                yield RawItem(
                    source=self.source_name,
                    source_id="h-1",
                    title="Healthy Item",
                    url="https://example.com/h1",
                )

        class FailingSource(BaseIngestionSource):
            source_name = "failing_source"

            async def fetch_items(self, limit: Optional[int] = None):
                raise RuntimeError("Catastrophic network or parsing failure in source")
                yield  # type: ignore[misc]

        pipeline = IngestionPipeline(
            db=migrated_db,
            settings=test_settings,
            sources=[
                HealthySource(settings=test_settings),
                FailingSource(settings=test_settings),
            ],
        )

        metrics = await pipeline.run_cycle()

        assert metrics["total_inserted_or_updated"] == 1
        assert metrics["sources"]["healthy_source"]["status"] == "success"
        assert metrics["sources"]["healthy_source"]["items_count"] == 1
        assert metrics["sources"]["failing_source"]["status"] == "error"
        assert "Catastrophic" in metrics["sources"]["failing_source"]["error"]

        # Ensure healthy item was saved to DB
        row = await migrated_db.fetch_one("SELECT * FROM raw_items WHERE source_id = 'h-1';")
        assert row is not None
        assert row["title"] == "Healthy Item"
