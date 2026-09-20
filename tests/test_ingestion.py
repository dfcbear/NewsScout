"""tests/test_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit test suite for NewsScout ingestion layer:
- GitHub Releases poller with REST, tags, and Atom feed fallback
- Hacker News Firebase poller with sceptic comment ratio calculation
- Hugging Face Daily Papers poller strictly requiring linked code
- IngestionPipeline and SQLite WAL raw_items upsert idempotency
100% offline using httpx.MockTransport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from newsscout.config import Settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.ingestion.github import (
    DEFAULT_TIER1_REPOS,
    GitHubIngestionSource,
    strip_html_tags,
)
from newsscout.ingestion.hackernews import (
    AI_TOPIC_PATTERN,
    HackerNewsIngestionSource,
)
from newsscout.ingestion.huggingface import (
    HuggingFaceIngestionSource,
    extract_github_repo_url,
)
from newsscout.ingestion.pipeline import IngestionPipeline
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem


# ============================================================================
# Synthetic Mock Payloads
# ============================================================================

MOCK_GITHUB_REST_RELEASES = [
    {
        "tag_name": "v0.6.1",
        "name": "vLLM v0.6.1: Native NVFP4 Kernel Support",
        "html_url": "https://github.com/vllm-project/vllm/releases/tag/v0.6.1",
        "body": "Added direct NVFP4 kernel execution for RTX 4090 with 3x lower latency.",
        "published_at": "2026-09-17T12:00:00Z",
        "prerelease": False,
        "assets": [{"name": "vllm-0.6.1-cp312-cp312-linux_aarch64.whl"}],
        "author": {"login": "vllm-release-bot"},
    }
]

MOCK_GITHUB_REST_TAGS = [
    {
        "name": "v1.2.0",
        "commit": {"sha": "abcdef1234567890abcdef1234567890abcdef12"},
    }
]

MOCK_GITHUB_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">
  <id>tag:github.com,2008:https://github.com/vllm-project/vllm/releases</id>
  <link type="text/html" rel="alternate" href="https://github.com/vllm-project/vllm/releases"/>
  <title>Release notes from vllm</title>
  <updated>2026-09-17T12:00:00Z</updated>
  <entry>
    <id>tag:github.com,2008:Repository/12345/v0.6.1</id>
    <updated>2026-09-17T12:00:00Z</updated>
    <link rel="alternate" type="text/html" href="https://github.com/vllm-project/vllm/releases/tag/v0.6.1"/>
    <title>v0.6.1: Fast Kernel Update</title>
    <content type="html">&lt;p&gt;Added &lt;b&gt;NVFP4&lt;/b&gt; support.&lt;br/&gt;&lt;ul&gt;&lt;li&gt;3x faster&lt;/li&gt;&lt;/ul&gt;&lt;/p&gt;</content>
  </entry>
</feed>
"""


# ============================================================================
# GitHub Ingestion Tests
# ============================================================================

class TestGitHubIngestionSource:
    """Tests GitHub ingestion via REST and Atom feeds."""

    def test_strip_html_tags(self) -> None:
        raw_html = "<p>First line.<br/>Second line with <b>bold</b>.</p><ul><li>Item 1</li><li>Item 2</li></ul>"
        clean = strip_html_tags(raw_html)
        assert "First line." in clean
        assert "Second line with bold." in clean
        assert "• Item 1" in clean
        assert "<p>" not in clean
        assert "<b>" not in clean

    @pytest.mark.asyncio
    async def test_fetch_via_rest_success(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "api.github.com" in request.url.host
            return httpx.Response(200, json=MOCK_GITHUB_REST_RELEASES)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["vllm-project/vllm"]
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 1
        item = items[0]
        assert item.source == "github_releases"
        assert item.source_id == "vllm-project/vllm:v0.6.1"
        assert "vLLM v0.6.1" in item.title
        assert item.metadata["owner"] == "vllm-project"
        assert item.metadata["assets"] == ["vllm-0.6.1-cp312-cp312-linux_aarch64.whl"]
        assert item.metadata["ingestion_method"] == "rest"

    @pytest.mark.asyncio
    async def test_fetch_via_tags_fallback_when_no_releases(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "/releases" in url_str:
                return httpx.Response(200, json=[])
            if "/tags" in url_str:
                return httpx.Response(200, json=MOCK_GITHUB_REST_TAGS)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["custom/tag-only-repo"]
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 1
        assert items[0].source_id == "custom/tag-only-repo:v1.2.0"
        assert items[0].metadata["ingestion_method"] == "rest_tag"
        assert items[0].metadata["commit_sha"].startswith("abcdef12")

    @pytest.mark.asyncio
    async def test_fallback_to_atom_feed_on_rate_limit(self, test_settings: Settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "api.github.com" in url_str:
                # 403 Rate limit hit
                return httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={"message": "rate limit"})
            if "releases.atom" in url_str:
                return httpx.Response(200, text=MOCK_GITHUB_ATOM_XML)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings, client=client, target_repos=["vllm-project/vllm"]
        ) as source:
            items = await source.fetch_all()

        assert len(items) == 1
        item = items[0]
        assert item.source_id == "vllm-project/vllm:v0.6.1"
        assert item.metadata["ingestion_method"] == "atom"
        assert "NVFP4" in item.raw_content
        assert "• 3x faster" in item.raw_content

    @pytest.mark.asyncio
    async def test_force_atom_feed_directly(self, test_settings: Settings) -> None:
        called_rest = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called_rest
            url_str = str(request.url)
            if "api.github.com" in url_str:
                called_rest = True
                return httpx.Response(200, json=MOCK_GITHUB_REST_RELEASES)
            if "releases.atom" in url_str:
                return httpx.Response(200, text=MOCK_GITHUB_ATOM_XML)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with GitHubIngestionSource(
            settings=test_settings,
            client=client,
            target_repos=["vllm-project/vllm"],
            force_atom_feed=True,
        ) as source:
            items = await source.fetch_all()

        assert not called_rest
        assert len(items) == 1
        assert items[0].metadata["ingestion_method"] == "atom"


# ============================================================================
# Hacker News Ingestion Tests
# ============================================================================

class TestHackerNewsIngestionSource:
    """Tests Hacker News Firebase ingestion and sceptic ratio filtering."""

    @pytest.mark.asyncio
    async def test_hackernews_sceptic_ratio_and_ai_filter(self, test_settings: Settings) -> None:
        stories = {
            101: {
                "id": 101,
                "type": "story",
                "title": "vLLM 0.6: High Throughput Local Inference Engine",
                "score": 250,
                "descendants": 125,  # ratio = 125 / 250 = 0.5 > 0.4
                "url": "https://github.com/vllm-project/vllm",
                "by": "ai_dev",
                "time": 1726580000,
            },
            102: {
                "id": 102,
                "type": "story",
                "title": "Autonomous Coding Agent Framework in Rust",
                "score": 100,  # score <= 120 (DROPPED)
                "descendants": 60,
                "url": "https://github.com/agent/framework",
                "by": "rustacean",
            },
            103: {
                "id": 103,
                "type": "story",
                "title": "New LLM Architecture Benchmark",
                "score": 300,
                "descendants": 30,  # ratio = 30 / 300 = 0.1 <= 0.4 (DROPPED)
                "url": "https://github.com/bench/llm",
                "by": "benchmarker",
            },
            104: {
                "id": 104,
                "type": "story",
                "title": "Why Sourdough Bread is the Best Chemistry Project",
                "score": 350,
                "descendants": 200,  # ratio = 200 / 350 = 0.57 > 0.4, but NOT AI (DROPPED)
                "url": "https://baker.org/sourdough",
                "by": "baker",
            },
            105: {
                "id": 105,
                "type": "story",
                "title": "Show HN: Qdrant Local Vector Search on Microcontrollers",
                "score": 180,
                "descendants": 90,  # ratio = 0.5 > 0.4, AI topic, Ask/Show HN without URL (E04)
                "text": "We ported Qdrant vector index routines to embedded C++ microcontrollers.",
                "url": "",
                "by": "qdrant_team",
            },
            106: {
                "id": 106,
                "type": "story",
                "title": "Astroturfed AI Wrapper with Zero Comments",
                "score": 200,
                "descendants": 0,  # E03: ratio = 0.0 <= 0.4 (DROPPED)
                "url": "https://hype.ai",
                "by": "spammer",
            },
        }

        def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "topstories.json" in url_str:
                return httpx.Response(200, json=[101, 102, 103, 104, 105, 106])
            for sid, payload in stories.items():
                if f"item/{sid}.json" in url_str:
                    return httpx.Response(200, json=payload)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HackerNewsIngestionSource(
            settings=test_settings,
            client=client,
            min_score=120,
            min_comment_ratio=0.4,
        ) as source:
            items = await source.fetch_all()

        # Only item 101 and item 105 should qualify
        assert len(items) == 2
        item_ids = [item.source_id for item in items]
        assert "101" in item_ids
        assert "105" in item_ids

        # Verify E04 handling on item 105
        item_105 = next(it for it in items if it.source_id == "105")
        assert item_105.url == "https://news.ycombinator.com/item?id=105"
        assert item_105.metadata["is_ask_or_show_hn"] is True
        assert "Qdrant" in item_105.title
        assert item_105.metadata["comment_ratio"] == 0.5


# ============================================================================
# Hugging Face Daily Papers Ingestion Tests
# ============================================================================

class TestHuggingFaceIngestionSource:
    """Tests Hugging Face Daily Papers ingestion and strict code requirement."""

    def test_extract_github_repo_url(self) -> None:
        # 1. Direct githubRepo
        p1 = {"githubRepo": "https://github.com/vllm-project/vllm"}
        assert extract_github_repo_url(p1) == "https://github.com/vllm-project/vllm"

        # 2. Project page containing GitHub link
        p2 = {"projectPage": "https://github.com/DS4SD/docling.git"}
        assert extract_github_repo_url(p2) == "https://github.com/DS4SD/docling"

        # 3. Summary containing GitHub link
        p3 = {"summary": "Our implementation is available at https://github.com/org/repo."}
        assert extract_github_repo_url(p3) == "https://github.com/org/repo"

        # 4. No link
        p4 = {"summary": "We propose a purely theoretical mathematical proof."}
        assert extract_github_repo_url(p4) is None

    @pytest.mark.asyncio
    async def test_fetch_papers_strictly_requires_code(self, test_settings: Settings) -> None:
        mock_papers_payload = [
            {
                "id": "2409.11111",
                "paper": {
                    "id": "2409.11111",
                    "title": "Titans: Learning to Memorize at Test Time",
                    "summary": "Novel architecture with code at https://github.com/google-research/titans.",
                    "upvotes": 42,
                    "githubRepo": "https://github.com/google-research/titans",
                },
                "publishedAt": "2026-09-17T10:00:00Z",
                "upvotes": 42,
            },
            {
                "id": "2409.22222",
                "paper": {
                    "id": "2409.22222",
                    "title": "Purely Theoretical Transformer Convergence Bounds",
                    "summary": "We prove that under infinite width transformers converge.",
                    "upvotes": 25,
                    "githubRepo": None,
                },
                "publishedAt": "2026-09-17T11:00:00Z",
                "upvotes": 25,
            },
            {
                "id": "2409.33333",
                "paper": {
                    "id": "2409.33333",
                    "title": "Low Upvote Paper With Code",
                    "summary": "Code at https://github.com/some/repo",
                    "upvotes": 2,  # upvotes < 5 (DROPPED)
                    "githubRepo": "https://github.com/some/repo",
                },
                "upvotes": 2,
            },
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            assert "daily_papers" in str(request.url)
            return httpx.Response(200, json=mock_papers_payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)

        async with HuggingFaceIngestionSource(
            settings=test_settings, client=client, min_upvotes=5
        ) as source:
            items = await source.fetch_all()

        # Only 2409.11111 should qualify (2409.22222 has no code, 2409.33333 has <5 upvotes)
        assert len(items) == 1
        item = items[0]
        assert item.source_id == "arxiv:2409.11111"
        assert item.metadata["github_repo"] == "https://github.com/google-research/titans"
        assert item.metadata["upvotes"] == 42


# ============================================================================
# Pipeline & SQLite Persistence Tests
# ============================================================================

class TestIngestionPipeline:
    """Tests IngestionPipeline orchestration, database upsert, and state management."""

    @pytest.mark.asyncio
    async def test_persist_raw_items_and_idempotent_upsert(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        pipeline = IngestionPipeline(db=migrated_db, settings=test_settings)

        item1 = RawItem(
            source="github_releases",
            source_id="vllm-project/vllm:v0.6.0",
            title="vLLM v0.6.0",
            url="https://github.com/vllm-project/vllm/releases/tag/v0.6.0",
            raw_content="Initial release notes",
            metadata={"assets_count": 1, "score": 10},
        )

        # 1. First insert
        count = await pipeline.persist_raw_items([item1])
        assert count == 1

        row = await migrated_db.fetch_one(
            "SELECT * FROM raw_items WHERE source = ? AND source_id = ?;",
            ("github_releases", "vllm-project/vllm:v0.6.0"),
        )
        assert row is not None
        assert row["title"] == "vLLM v0.6.0"
        initial_ingested_at = row["ingested_at"]

        # 2. Update same item with new metadata (ON CONFLICT DO UPDATE)
        item1_updated = RawItem(
            source="github_releases",
            source_id="vllm-project/vllm:v0.6.0",
            title="vLLM v0.6.0 (Updated)",
            url="https://github.com/vllm-project/vllm/releases/tag/v0.6.0",
            raw_content="Updated release notes",
            metadata={"assets_count": 5, "score": 50},
        )
        count_up = await pipeline.persist_raw_items([item1_updated])
        assert count_up == 1

        row2 = await migrated_db.fetch_one(
            "SELECT * FROM raw_items WHERE source = ? AND source_id = ?;",
            ("github_releases", "vllm-project/vllm:v0.6.0"),
        )
        assert row2 is not None
        assert row2["title"] == "vLLM v0.6.0 (Updated)"
        # ingested_at preserved!
        assert row2["ingested_at"] == initial_ingested_at
        meta = json.loads(row2["metadata_json"])
        assert meta["assets_count"] == 5

    @pytest.mark.asyncio
    async def test_run_cycle_with_custom_mock_source(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        class MockSource(BaseIngestionSource):
            source_name = "mock_test_source"

            async def fetch_items(self, limit: Any = None):
                yield RawItem(
                    source=self.source_name,
                    source_id="test-1",
                    title="Mock Item 1",
                    url="https://example.com/test-1",
                    raw_content="Mock description",
                )

        pipeline = IngestionPipeline(
            db=migrated_db, settings=test_settings, sources=[MockSource(settings=test_settings)]
        )

        metrics = await pipeline.run_cycle()
        assert metrics["total_inserted_or_updated"] == 1
        assert "mock_test_source" in metrics["sources"]
        assert metrics["sources"]["mock_test_source"]["status"] == "success"

        # Verify DB entry
        row = await migrated_db.fetch_one("SELECT * FROM raw_items WHERE source_id = 'test-1';")
        assert row is not None
        assert row["title"] == "Mock Item 1"
