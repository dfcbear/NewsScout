"""tests/test_dedup.py
~~~~~~~~~~~~~~~~~~~
Comprehensive unit and benchmark test suite for TASK-10 Milestone 1:
- URL normalizer tracking parameter extension (50+ params, 12 prefix families)
- Semantic query parameter preservation
- Lightweight heuristic text similarity (Token Jaccard + 2-shingling)
- Suffix stemming and semver / model size token preservation
- Headline rewording detection (>= 0.80) and distinct model preservation (< 0.80)
- Intra-batch and inter-batch deduplication with attribution merging
- Pipeline integration and token cost protection
- Performance benchmark (<100ms for 50 items)
100% offline, zero live network dependencies.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any

import pytest

from newsscout.config import Settings
from newsscout.filtering.dedup import (
    _light_stem,
    calculate_text_similarity,
    deduplicate_against_db,
    deduplicate_items,
    extract_shingles,
    extract_tokens,
    is_near_duplicate,
    merge_attributions,
    normalize_text,
    pick_primary_item,
    token_jaccard,
)
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.ingestion.pipeline import IngestionPipeline
from newsscout.search.normalizer import (
    TRACKING_PARAMS,
    TRACKING_PREFIXES,
    clean_and_canonicalize_url,
    is_tracking_param,
)
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem


# ============================================================================
# 1. URL Normalizer Tracking Parameter Extension Tests
# ============================================================================

class TestURLNormalizerTrackingExtension:
    """Verifies that 50+ tracking parameters and 12 prefix families are stripped."""

    def test_tracking_params_catalog_size(self) -> None:
        """Confirms catalog contains 50+ tracking parameters and 12+ prefix families."""
        assert len(TRACKING_PARAMS) >= 50
        assert len(TRACKING_PREFIXES) >= 12

    @pytest.mark.parametrize(
        "param_name",
        [
            "wickedid",
            "yclid",
            "guccounter",
            "guce_referrer",
            "guce_referrer_usqp",
            "mc_cid",
            "mc_eid",
            "_hsenc",
            "_hsmi",
            "vero_id",
            "subid",
            "partner",
            "affiliate_id",
            "origin",
            "feedName",
            "curator",
            "ttclid",
            "li_fat_id",
            "s_cid",
            "sc_cid",
        ],
    )
    def test_is_tracking_param_exact_matches(self, param_name: str) -> None:
        assert is_tracking_param(param_name) is True
        assert is_tracking_param(param_name.upper()) is True

    @pytest.mark.parametrize(
        "param_name",
        [
            "utm_campaign_id",
            "pk_campaign",
            "pk_kwd",
            "piwik_campaign",
            "matomo_campaign",
            "hsa_acc",
            "hsa_cam",
            "_hsenc",
            "mc_cid",
            "sc_cid",
            "nd_feed",
            "aff_id",
            "aff_sub",
            "wt_mc",
            "ga_source",
            "_ga_session",
            "xtor_123",
        ],
    )
    def test_is_tracking_param_prefix_matches(self, param_name: str) -> None:
        assert is_tracking_param(param_name) is True

    @pytest.mark.parametrize(
        "param_name",
        ["q", "query", "id", "item_id", "v", "version", "page", "tag", "sort", "filter", "repo"],
    )
    def test_is_tracking_param_preserves_legitimate_params(self, param_name: str) -> None:
        assert is_tracking_param(param_name) is False

    def test_clean_and_canonicalize_url_strips_syndication_and_tracking(self) -> None:
        url = (
            "https://techcrunch.com/2026/09/20/openai-gpt5/?"
            "utm_source=feed&utm_medium=rss&wickedid=w123&yclid=y456&"
            "guccounter=1&guce_referrer=ref&mc_cid=m123&_hsenc=h123&"
            "partner=syndicate&subid=news&pk_campaign=daily&q=deeplearning"
        )
        cleaned = clean_and_canonicalize_url(url)
        assert "utm_source" not in cleaned
        assert "wickedid" not in cleaned
        assert "yclid" not in cleaned
        assert "guccounter" not in cleaned
        assert "mc_cid" not in cleaned
        assert "_hsenc" not in cleaned
        assert "partner" not in cleaned
        assert "subid" not in cleaned
        assert "pk_campaign" not in cleaned
        assert "q=deeplearning" in cleaned
        assert cleaned == "https://techcrunch.com/2026/09/20/openai-gpt5?q=deeplearning"

    def test_identical_canonical_urls_with_different_tracking(self) -> None:
        url1 = "https://example.com/article?utm_source=twitter&ref=social"
        url2 = "https://example.com/article?utm_source=newsletter&mc_cid=12345&guccounter=1"
        assert clean_and_canonicalize_url(url1) == clean_and_canonicalize_url(url2)
        assert clean_and_canonicalize_url(url1) == "https://example.com/article"


# ============================================================================
# 2. Text Similarity & Normalization Tests
# ============================================================================

class TestLightweightTextSimilarity:
    """Verifies heuristic text similarity, stemming, versioning, and shingling."""

    def test_stemming_rules(self) -> None:
        assert _light_stem("releases") == "releas"
        assert _light_stem("released") == "releas"
        assert _light_stem("releasing") == "releas"
        assert _light_stem("models") == "model"
        assert _light_stem("processes") == "process"
        assert _light_stem("capabilities") == "capability"

    def test_stemming_preserves_versions_and_model_sizes(self) -> None:
        # Model sizes and semver tokens must NOT be altered by stemmer
        assert _light_stem("405b") == "405b"
        assert _light_stem("70b") == "70b"
        assert _light_stem("8b") == "8b"
        assert _light_stem("3.1") == "3.1"
        assert _light_stem("0.6.1") == "0.6.1"
        assert _light_stem("gpt-5") == "gpt-5"

    def test_normalize_text_strips_aggregator_prefixes(self) -> None:
        assert normalize_text("Show HN: New local inference engine") == "new local inference engine"
        assert normalize_text("Ask HN: Best vector store in 2026") == "best vector store in 2026"
        assert normalize_text("[D] Theoretical Bounds of Attention") == "theoretical bounds of attention"
        assert normalize_text("Release: vLLM 0.6.1") == "vllm 0.6.1"
        assert normalize_text("Announcing: Llama 3.1") == "llama 3.1"

    def test_extract_tokens_strips_leading_v_from_semver(self) -> None:
        t1 = extract_tokens("vLLM v0.6.1: Native NVFP4 Kernel Support")
        t2 = extract_tokens("vLLM 0.6.1: Native NVFP4 Kernel Support")
        assert "0.6.1" in t1
        assert "0.6.1" in t2
        assert t1 == t2

    def test_extract_tokens_preserves_distinct_model_sizes(self) -> None:
        t1 = extract_tokens("Llama 3.1 405B Released")
        t2 = extract_tokens("Llama 3.1 70B Released")
        assert "405b" in t1
        assert "70b" in t2
        assert t1 != t2

    def test_headline_rewording_detection_score_above_80(self) -> None:
        """Active vs passive / reordered words must score >= 0.80."""
        title1 = "OpenAI releases GPT-5"
        title2 = "GPT-5 Released by OpenAI"
        score = calculate_text_similarity(title1, title2)
        assert score >= 0.80
        # In fact, token sets are identical after stemming and stop word stripping
        assert score == 1.0

    def test_distinct_model_sizes_score_below_80(self) -> None:
        """Different parameter sizes of the same model family must score < 0.80."""
        title1 = "Llama 3.1 405B Released"
        title2 = "Llama 3.1 70B Released"
        score = calculate_text_similarity(title1, title2)
        assert score < 0.80
        # 3 matching tokens (llama, 3.1, releas) out of 5 distinct tokens -> score is safely below 0.80
        assert 0.40 <= score <= 0.65

    def test_distinct_version_numbers_score_below_80(self) -> None:
        """Different minor or major versions must score < 0.80 when titles are short."""
        title1 = "vLLM 0.6.0 Released"
        title2 = "vLLM 0.7.0 Released"
        score = calculate_text_similarity(title1, title2)
        assert score < 0.80

    def test_completely_different_articles_score_very_low(self) -> None:
        title1 = "OpenAI releases GPT-5"
        title2 = "Google releases Gemini 2.0 Pro"
        score = calculate_text_similarity(title1, title2)
        assert score < 0.25

    def test_empty_and_none_text_handling(self) -> None:
        assert calculate_text_similarity("", "") == 1.0
        assert calculate_text_similarity("test", "") == 0.0
        assert calculate_text_similarity(None, "test") == 0.0


# ============================================================================
# 3. Near-Duplicate Detection & Attribution Merging Tests
# ============================================================================

class TestNearDuplicateAndAttribution:
    """Verifies is_near_duplicate and metadata attribution merging."""

    def test_is_near_duplicate_title_only(self) -> None:
        is_dup, score = is_near_duplicate(
            "OpenAI releases GPT-5", "", "GPT-5 Released by OpenAI", "", threshold=0.80
        )
        assert is_dup is True
        assert score >= 0.80

    def test_is_near_duplicate_with_teaser_text(self) -> None:
        # Borderline title similarity elevated by high teaser overlap
        title1 = "vLLM 0.6.1 Fast Kernel Update"
        title2 = "vLLM 0.6.1 GPU Acceleration Notes"
        text1 = "Added native NVFP4 kernel execution for RTX 4090 with 3x lower latency."
        text2 = "Added native NVFP4 kernel execution for RTX 4090 with 3x lower latency."

        is_dup, score = is_near_duplicate(title1, text1, title2, text2, threshold=0.80)
        assert is_dup is True
        assert score >= 0.80

    def test_is_not_near_duplicate_distinct_models(self) -> None:
        is_dup, score = is_near_duplicate(
            "Llama 3.1 405B Released", "", "Llama 3.1 70B Released", "", threshold=0.80
        )
        assert is_dup is False
        assert score < 0.80

    def test_pick_primary_item_prefers_github_repo(self) -> None:
        item_no_repo = RawItem(
            source="hacker_news",
            source_id="123",
            title="vLLM Inference Engine",
            url="https://news.ycombinator.com/item?id=123",
            raw_content="Check out this engine",
            metadata={},
        )
        item_with_repo = RawItem(
            source="web_search",
            source_id="search:456",
            title="vLLM Inference Engine",
            url="https://github.com/vllm-project/vllm",
            raw_content="Short",
            metadata={"github_repo": "https://github.com/vllm-project/vllm"},
        )
        primary, duplicate = pick_primary_item(item_no_repo, item_with_repo)
        assert primary == item_with_repo
        assert duplicate == item_no_repo

    def test_pick_primary_item_prefers_longer_content_when_tied(self) -> None:
        item_short = RawItem(
            source="hacker_news",
            source_id="1",
            title="AI Tool",
            url="https://example.com/1",
            raw_content="Short teaser",
        )
        item_long = RawItem(
            source="web_search",
            source_id="2",
            title="AI Tool",
            url="https://example.com/2",
            raw_content="A much longer technical description of how this system works...",
        )
        primary, duplicate = pick_primary_item(item_short, item_long)
        assert primary == item_long
        assert duplicate == item_short

    def test_merge_attributions_schema(self) -> None:
        primary = RawItem(
            source="hacker_news",
            source_id="101",
            title="OpenAI releases GPT-5",
            url="https://news.ycombinator.com/item?id=101",
            metadata={"score": 250},
        )
        duplicate = RawItem(
            source="web_search",
            source_id="search:tc-gpt5",
            title="GPT-5 Released by OpenAI",
            url="https://techcrunch.com/gpt5?utm_source=twitter",
            metadata={"rank": 1},
        )

        merge_attributions(primary, duplicate)

        assert primary.metadata["duplicate_count"] == 2
        sources = primary.metadata["sources"]
        assert len(sources) == 2
        assert sources[0]["source"] == "hacker_news"
        assert sources[0]["source_id"] == "101"
        assert sources[1]["source"] == "web_search"
        assert sources[1]["source_id"] == "search:tc-gpt5"

        alt_urls = primary.metadata["alternate_urls"]
        assert "https://techcrunch.com/gpt5?utm_source=twitter" in alt_urls
        assert "https://techcrunch.com/gpt5" in alt_urls


# ============================================================================
# 4. Intra-Batch Deduplication Tests
# ============================================================================

class TestIntraBatchDeduplication:
    """Verifies batch deduplication and secondary record filtering."""

    def test_deduplicate_items_empty_and_single(self) -> None:
        assert deduplicate_items([]) == []
        item = RawItem(source="hn", source_id="1", title="Title", url="https://example.com/1")
        assert deduplicate_items([item]) == [item]

    def test_deduplicate_items_merges_url_and_reworded_duplicates(self) -> None:
        items = [
            # Story A1: HN post
            RawItem(
                source="hacker_news",
                source_id="101",
                title="OpenAI releases GPT-5",
                url="https://news.ycombinator.com/item?id=101",
                raw_content="OpenAI announced their new frontier model GPT-5.",
            ),
            # Story A2: Web search near-duplicate of Story A
            RawItem(
                source="web_search",
                source_id="search:101",
                title="GPT-5 Released by OpenAI",
                url="https://techcrunch.com/gpt5?utm_source=rss",
                raw_content="OpenAI announced their new frontier model GPT-5.",
            ),
            # Story B: Distinct model size (MUST BE PRESERVED)
            RawItem(
                source="github_releases",
                source_id="meta:llama-405b",
                title="Llama 3.1 405B Released",
                url="https://github.com/meta-llama/llama-models/releases/tag/405b",
                raw_content="Meta announces Llama 3.1 405B flagship model weights.",
                metadata={"github_repo": "https://github.com/meta-llama/llama-models"},
            ),
            # Story C: Another distinct model size (MUST BE PRESERVED)
            RawItem(
                source="github_releases",
                source_id="meta:llama-70b",
                title="Llama 3.1 70B Released",
                url="https://github.com/meta-llama/llama-models/releases/tag/70b",
                raw_content="Meta announces Llama 3.1 70B mid-range model weights.",
                metadata={"github_repo": "https://github.com/meta-llama/llama-models"},
            ),
            # Story A3: Exact URL duplicate of Story A2 with different query parameters
            RawItem(
                source="web_search",
                source_id="search:102",
                title="Breaking: GPT-5 Released by OpenAI",
                url="https://techcrunch.com/gpt5?wickedid=w123&mc_cid=m456",
                raw_content="OpenAI announced their new frontier model GPT-5.",
            ),
        ]

        deduped = deduplicate_items(items, threshold=0.80)

        # Total 5 items -> Story A (merged from 3 copies) + Story B (405B) + Story C (70B) = 3 items
        assert len(deduped) == 3

        titles = [item.title for item in deduped]
        # Story B and Story C preserved
        assert "Llama 3.1 405B Released" in titles
        assert "Llama 3.1 70B Released" in titles

        # Story A merged
        story_a = next(it for it in deduped if "GPT-5" in it.title)
        assert story_a.metadata.get("duplicate_count") == 3
        assert len(story_a.metadata.get("sources", [])) == 3


# ============================================================================
# 5. Inter-Batch SQLite Deduplication Tests
# ============================================================================

class TestInterBatchDeduplication:
    """Verifies SQLite raw_items deduplication against recent historical items."""

    @pytest.mark.asyncio
    async def test_deduplicate_against_db_merges_and_drops(self, migrated_db: Database) -> None:
        # Pre-seed DB with an existing item from 1 day ago
        initial_item = RawItem(
            source="hacker_news",
            source_id="hn:555",
            title="vLLM v0.6.1: Native NVFP4 Kernel Support",
            url="https://github.com/vllm-project/vllm/releases/tag/v0.6.1",
            raw_content="Direct NVFP4 kernel execution for RTX 4090.",
            metadata={"github_repo": "https://github.com/vllm-project/vllm"},
            ingested_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
            """,
            initial_item.to_db_params(),
        )

        incoming_duplicate = RawItem(
            source="web_search",
            source_id="search:999",
            title="vLLM 0.6.1: Native NVFP4 Kernel Support",
            url="https://techblog.com/vllm-nvfp4?utm_source=twitter",
            raw_content="Direct NVFP4 kernel execution for RTX 4090.",
            metadata={},
        )
        incoming_unique = RawItem(
            source="github_releases",
            source_id="sglang:v0.4.0",
            title="SGLang v0.4.0 High-Performance Serving",
            url="https://github.com/sgl-project/sglang/releases/tag/v0.4.0",
            raw_content="RadixAttention multi-turn acceleration.",
        )

        surviving = await deduplicate_against_db(
            [incoming_duplicate, incoming_unique],
            db=migrated_db,
            lookback_days=7,
            threshold=0.80,
        )

        # Duplicate should be dropped; unique should survive
        assert len(surviving) == 1
        assert surviving[0].source_id == "sglang:v0.4.0"

        # Verify existing DB record was updated with merged attribution
        row = await migrated_db.fetch_one(
            "SELECT metadata_json FROM raw_items WHERE source_id = 'hn:555';"
        )
        assert row is not None
        meta = json.loads(row["metadata_json"])
        assert meta["duplicate_count"] == 2
        assert any(s["source"] == "web_search" for s in meta["sources"])

    @pytest.mark.asyncio
    async def test_deduplicate_against_db_ignores_records_outside_lookback(
        self, migrated_db: Database
    ) -> None:
        # Pre-seed DB with an item from 10 days ago (outside 7-day lookback)
        old_item = RawItem(
            source="hacker_news",
            source_id="hn:old",
            title="Old Story That Happened 10 Days Ago",
            url="https://example.com/old",
            raw_content="Old content",
            ingested_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
            """,
            old_item.to_db_params(),
        )

        incoming = RawItem(
            source="web_search",
            source_id="search:new_repeat",
            title="Old Story That Happened 10 Days Ago",
            url="https://example.com/new_repeat",
            raw_content="Old content repeated",
        )

        surviving = await deduplicate_against_db(
            [incoming],
            db=migrated_db,
            lookback_days=7,
            threshold=0.80,
        )

        # Because the existing record is outside the 7-day window, incoming is preserved
        assert len(surviving) == 1
        assert surviving[0].source_id == "search:new_repeat"


# ============================================================================
# 6. Pipeline Ingestion Integration Tests
# ============================================================================

class TestPipelineIngestionDeduplication:
    """Verifies that IngestionPipeline prevents duplicate entries from reaching DB."""

    @pytest.mark.asyncio
    async def test_run_cycle_drops_near_duplicates_before_persistence(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        class SyndicatedMockSource(BaseIngestionSource):
            source_name = "syndicated_source"

            async def fetch_items(self, limit: Any = None):
                yield RawItem(
                    source=self.source_name,
                    source_id="syn:1",
                    title="OpenAI releases GPT-5",
                    url="https://news.example.com/gpt5",
                    raw_content="OpenAI announces GPT-5 with test-time compute.",
                )
                yield RawItem(
                    source=self.source_name,
                    source_id="syn:2",
                    title="GPT-5 Released by OpenAI",
                    url="https://blog.example.com/gpt5?utm_source=feed",
                    raw_content="OpenAI announces GPT-5 with test-time compute.",
                )

        pipeline = IngestionPipeline(
            db=migrated_db,
            settings=test_settings,
            sources=[SyndicatedMockSource(settings=test_settings)],
        )

        metrics = await pipeline.run_cycle()

        # Both items fetched, but only 1 persisted!
        assert metrics["sources"]["syndicated_source"]["items_count"] == 2
        assert metrics["sources"]["syndicated_source"]["persisted_count"] == 1
        assert metrics["sources"]["syndicated_source"]["deduped_count"] == 1
        assert metrics["total_inserted_or_updated"] == 1

        # Verify only 1 row in SQLite DB
        rows = await migrated_db.fetch_all("SELECT * FROM raw_items WHERE title LIKE '%GPT-5%';")
        assert len(rows) == 1
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["duplicate_count"] == 2


# ============================================================================
# 7. Performance Benchmark Guarantee (<100ms for 50 items)
# ============================================================================

class TestDeduplicationPerformanceBenchmark:
    """Ensures deduplication executes in <100ms for 50 items (1,225 pairs) on Pi 5."""

    def test_benchmark_50_items_pairwise_execution_time(self) -> None:
        # Create 50 synthetic items with diverse titles and contents
        items: list[RawItem] = []
        for i in range(50):
            model_size = f"{((i % 5) + 1) * 7}b"
            version = f"v0.{i % 10}.0"
            items.append(
                RawItem(
                    source=f"source_{i % 4}",
                    source_id=f"item_{i}",
                    title=f"Tool {i // 2} Model {model_size} {version} Release and Inference Benchmark",
                    url=f"https://example.com/tool/{i // 2}?tracking_id={i}&utm_source=test",
                    raw_content=f"Detailed evaluation notes for tool {i // 2} on RTX 4090 with {model_size} weights.",
                    metadata={"index": i},
                )
            )

        start_time = time.perf_counter()
        deduped = deduplicate_items(items, threshold=0.80)
        elapsed_seconds = time.perf_counter() - start_time

        # Mandatory requirement: must run in <100ms (0.100s) on Raspberry Pi 5
        assert elapsed_seconds < 0.100, f"Benchmark exceeded 100ms: {elapsed_seconds * 1000:.2f}ms"
        assert len(deduped) > 0


# ============================================================================
# 8. Keyset Pagination Tests (Paket 2)
# ============================================================================

class TestKeysetPaginationDeduplication:
    """Verifies that deduplicate_against_db uses keyset pagination for >1000 items."""

    @pytest.mark.asyncio
    async def test_paging_with_more_than_1000_items(self, migrated_db: Database) -> None:
        """Ensure deduplicate_against_db correctly paginates through >1000 rows."""
        # Insert 1200 raw_items into the database
        items_to_insert: list[RawItem] = []
        for i in range(1200):
            items_to_insert.append(
                RawItem(
                    source="bulk_source",
                    source_id=f"bulk:{i}",
                    title=f"Bulk Item {i}",
                    url=f"https://example.com/bulk/{i}",
                    raw_content=f"Content for item {i}",
                    metadata={"index": i},
                )
            )

        # Persist all items
        for item in items_to_insert:
            await migrated_db.execute(
                """
                INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)
                """,
                {
                    "source": item.source,
                    "source_id": item.source_id,
                    "title": item.title,
                    "url": item.url,
                    "raw_content": item.raw_content,
                    "metadata_json": json.dumps(item.metadata),
                    "ingested_at": item.ingested_at.isoformat(),
                },
            )

        # Now deduplicate with an incoming item that matches one of the existing URLs
        incoming = RawItem(
            source="new_source",
            source_id="new:1",
            title="Bulk Item 500",
            url="https://example.com/bulk/500",
            raw_content="Content for item 500",
            metadata={"index": 500},
        )

        surviving = await deduplicate_against_db(
            [incoming],
            db=migrated_db,
            lookback_days=7,
            threshold=0.80,
        )

        # The incoming item should match the existing item 500 and be dropped
        assert len(surviving) == 0


# ============================================================================
# 9. URL Match With Different Titles Tests (Paket 2)
# ============================================================================

class TestUrlMatchWithDifferentTitles:
    """Verifies that URL match guard runs title similarity check even when URLs are identical."""

    def test_same_url_different_titles_not_merged(self) -> None:
        """Items with identical URLs but very different titles should NOT be merged."""
        items = [
            RawItem(
                source="source_a",
                source_id="a:1",
                title="vLLM v0.6.0 NVFP4 Kernel Support",
                url="https://github.com/vllm-project/vllm",
                raw_content="NVFP4 quantization kernel for RTX 4090.",
            ),
            RawItem(
                source="source_b",
                source_id="b:1",
                title="Completely Unrelated Project About Cooking",
                url="https://github.com/vllm-project/vllm",
                raw_content="A recipe collection for home chefs.",
            ),
        ]

        # With threshold 0.80, these should NOT be merged because the title
        # similarity is below threshold even though URLs match
        deduped = deduplicate_items(items, threshold=0.80)
        assert len(deduped) == 2

    def test_same_url_similar_titles_merged(self) -> None:
        """Items with identical URLs and similar titles should be merged."""
        items = [
            RawItem(
                source="source_a",
                source_id="a:1",
                title="vLLM v0.6.0 NVFP4 Kernel Support",
                url="https://github.com/vllm-project/vllm",
                raw_content="NVFP4 quantization kernel for RTX 4090.",
            ),
            RawItem(
                source="source_b",
                source_id="b:1",
                title="vLLM v0.6.0 NVFP4 Kernel Release",
                url="https://github.com/vllm-project/vllm",
                raw_content="NVFP4 quantization kernel for RTX 4090.",
            ),
        ]

        deduped = deduplicate_items(items, threshold=0.80)
        assert len(deduped) == 1
