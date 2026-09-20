"""tests/test_challenger_t10_m1.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial challenge test suite for TASK-10 Milestone 1:
- DB Resilience: empty DB, locked DB, missing columns, corrupted JSON metadata.
- Redundant Call Elimination: verify zero redundant Stage 1 and Stage 2 LLM calls.
- CPU Latency Benchmarking: exact millisecond measurements for 50 and 100 items.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any, Optional

import pytest

from newsscout.config import Settings
from newsscout.filtering.dedup import (
    calculate_text_similarity,
    deduplicate_against_db,
    deduplicate_items,
    is_near_duplicate,
)
from newsscout.filtering.stage1 import Stage1Filter
from newsscout.filtering.stage2 import Stage2Evaluator
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.ingestion.pipeline import IngestionPipeline
from pydantic import BaseModel
from newsscout.llm.base import BaseLLMClient, T
from newsscout.storage.db import Database, create_connection
from newsscout.storage.models import RawItem


# ============================================================================
# 1. DB Resilience Stress Tests
# ============================================================================

class TestDeduplicateAgainstDBResilience:
    """Stress tests deduplicate_against_db against edge cases:

    - Empty database
    - Corrupted JSON metadata (invalid syntax, empty string, non-dict JSON)
    - Missing columns
    - Database lock (concurrent exclusive transaction)
    """

    @pytest.mark.asyncio
    async def test_empty_database_returns_all_items(self, migrated_db: Database) -> None:
        """Verifies deduplicate_against_db cleanly returns all items when raw_items is empty."""
        items = [
            RawItem(
                source="test_source",
                source_id="1",
                title="vLLM v0.7.0 Released",
                url="https://github.com/vllm-project/vllm/releases/tag/v0.7.0",
                raw_content="Kernel updates",
            ),
            RawItem(
                source="test_source",
                source_id="2",
                title="SGLang v0.5.0 Released",
                url="https://github.com/sgl-project/sglang/releases/tag/v0.5.0",
                raw_content="Serving engine updates",
            ),
        ]
        surviving = await deduplicate_against_db(items, db=migrated_db, lookback_days=7)
        assert len(surviving) == 2
        assert surviving[0].source_id == "1"
        assert surviving[1].source_id == "2"

    @pytest.mark.asyncio
    async def test_corrupted_json_syntax_in_db_row(self, migrated_db: Database) -> None:
        """Adversarial test: If any row in raw_items has invalid/corrupted JSON in metadata_json,

        deduplicate_against_db should gracefully skip the corrupted row and still process
        incoming items (production resilience — one bad row must not crash the entire cycle).
        """
        # Directly insert a row with corrupted JSON syntax into raw_items
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('broken_src', 'broken:1', 'Broken Meta Story', 'https://example.com/broken', 'content', '{corrupted: json,', :now);
            """,
            {"now": datetime.now(timezone.utc).isoformat()},
        )

        incoming = [
            RawItem(
                source="new_src",
                source_id="new:1",
                title="Valid Clean Story",
                url="https://example.com/clean",
                raw_content="clean content",
            )
        ]

        # Should NOT raise — corrupted row is skipped, incoming item passes through
        surviving = await deduplicate_against_db(incoming, db=migrated_db, lookback_days=7)
        assert len(surviving) == 1
        assert surviving[0].source_id == "new:1"

    @pytest.mark.asyncio
    async def test_empty_string_json_in_db_row(self, migrated_db: Database) -> None:
        """Adversarial test: If metadata_json is an empty string '',

        deduplicate_against_db should gracefully skip the corrupted row.
        """
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('empty_json_src', 'empty:1', 'Empty JSON Story', 'https://example.com/empty_json', 'content', '', :now);
            """,
            {"now": datetime.now(timezone.utc).isoformat()},
        )

        incoming = [
            RawItem(
                source="new_src",
                source_id="new:2",
                title="Another Clean Story",
                url="https://example.com/clean2",
                raw_content="clean content 2",
            )
        ]

        # Should NOT raise — empty JSON row is skipped, incoming item passes through
        surviving = await deduplicate_against_db(incoming, db=migrated_db, lookback_days=7)
        assert len(surviving) == 1
        assert surviving[0].source_id == "new:2"

    @pytest.mark.asyncio
    async def test_non_dict_json_in_db_row(self, migrated_db: Database) -> None:
        """Adversarial test: If metadata_json is a valid JSON array or integer (e.g. '[]' or '123'),

        deduplicate_against_db should gracefully skip the invalid row.
        """
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('array_json_src', 'array:1', 'Array JSON Story', 'https://example.com/array_json', 'content', '[1, 2, 3]', :now);
            """,
            {"now": datetime.now(timezone.utc).isoformat()},
        )

        incoming = [
            RawItem(
                source="new_src",
                source_id="new:3",
                title="Clean Story 3",
                url="https://example.com/clean3",
                raw_content="clean content 3",
            )
        ]

        # Should NOT raise — invalid metadata type row is skipped, incoming item passes through
        surviving = await deduplicate_against_db(incoming, db=migrated_db, lookback_days=7)
        assert len(surviving) == 1
        assert surviving[0].source_id == "new:3"

    @pytest.mark.asyncio
    async def test_missing_table_or_columns(self, temp_db: Database) -> None:
        """Adversarial test: If the raw_items table does not exist or has missing columns,

        deduplicate_against_db raises sqlite3.OperationalError.
        """
        # temp_db is unmigrated, so raw_items does not exist
        incoming = [
            RawItem(
                source="src",
                source_id="1",
                title="Title",
                url="https://example.com",
            )
        ]
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            await deduplicate_against_db(incoming, db=temp_db, lookback_days=7)

        assert "no such table: raw_items" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_database_locked_during_update(self, temp_db_path: Path, migrated_db: Database) -> None:
        """Adversarial test: If an external connection holds an exclusive lock on SQLite

        when deduplicate_against_db attempts to update metadata_json of a matched record.
        """
        # 1. Insert a record into raw_items that matches incoming item
        initial_item = RawItem(
            source="hacker_news",
            source_id="hn:100",
            title="OpenAI releases GPT-5 with Reasoning",
            url="https://news.example.com/gpt5",
            raw_content="Announcement content",
            ingested_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
            """,
            initial_item.to_db_params(),
        )

        incoming = [
            RawItem(
                source="web_search",
                source_id="search:100",
                title="GPT-5 with Reasoning released by OpenAI",
                url="https://blog.example.com/gpt5?utm_source=twitter",
                raw_content="Announcement content",
            )
        ]

        # 2. Open a separate connection and acquire an EXCLUSIVE lock with busy_timeout=100ms
        lock_conn = sqlite3.connect(str(temp_db_path), timeout=0.1, isolation_level=None)
        try:
            lock_conn.execute("PRAGMA busy_timeout = 100;")
            lock_conn.execute("BEGIN EXCLUSIVE;")

            # Configure migrated_db with short timeout so it doesn't block for 5 seconds
            migrated_db.timeout = 0.2

            with pytest.raises(sqlite3.OperationalError) as exc_info:
                await deduplicate_against_db(incoming, db=migrated_db, lookback_days=7)

            assert "database is locked" in str(exc_info.value)
        finally:
            lock_conn.execute("ROLLBACK;")
            lock_conn.close()


# ============================================================================
# 2. Zero Redundant Stage 1 and Stage 2 LLM Calls Verification
# ============================================================================

class MockTrackingLLMClient(BaseLLMClient):
    """Test double tracking total LLM evaluate calls and prompts."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.evaluated_titles: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-tracking"

    @property
    def model_name(self) -> str:
        return "mock-tracking-v1"

    async def generate_text(
        self,
        prompt: str = "",
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        *,
        user_prompt: Optional[str] = None,
    ) -> str:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        self.call_count += 1
        return json.dumps({
            "breakthrough_score": 8.5,
            "roi_score": 9.0,
            "category": "core",
            "relevance_justification": "High novelty in autonomous agent execution on RTX 4090.",
            "decision_card": {
                "tldr": "Executes local agent reasoning 3x faster on RTX 4090.",
                "use_case": "Direct drop-in for local autonomous agent execution.",
                "comparison": "3x faster than baseline vLLM on RTX 4090.",
                "quickstart": "docker run -d -p 8000:8000 vllm/vllm:latest",
                "hardware_requirements": "RTX 4090 (24GB VRAM)",
                "license": "Apache-2.0",
            },
        })

    async def generate_structured(
        self,
        prompt: str = "",
        response_model: type[T] = BaseModel,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        json_schema: Optional[dict[str, Any]] = None,
        *,
        user_prompt: Optional[str] = None,
    ) -> T:
        actual_prompt = user_prompt if user_prompt is not None else prompt
        self.call_count += 1
        for line in actual_prompt.splitlines():
            if line.startswith("Title:") or "Title:" in line:
                self.evaluated_titles.append(line)
        raw_dict = {
            "breakthrough_score": 8.5,
            "roi_score": 9.0,
            "category": "core",
            "relevance_justification": "High novelty in autonomous agent execution on RTX 4090.",
            "decision_card": {
                "tldr": "Executes local agent reasoning 3x faster on RTX 4090.",
                "use_case": "Direct drop-in for local autonomous agent execution.",
                "comparison": "3x faster than baseline vLLM on RTX 4090.",
                "quickstart": "docker run -d -p 8000:8000 vllm/vllm:latest",
                "hardware_requirements": "RTX 4090 (24GB VRAM)",
                "license": "Apache-2.0",
            },
        }
        return response_model.model_validate(raw_dict)

    async def close(self) -> None:
        pass


class TestZeroRedundantLLMCalls:
    """Verifies end-to-end that duplicate detection strictly eliminates redundant

    Stage 1 heuristic evaluations and Stage 2 LLM calls.
    """

    @pytest.mark.asyncio
    async def test_intra_batch_and_inter_batch_zero_redundant_llm_calls(
        self, migrated_db: Database, test_settings: Settings
    ) -> None:
        tracking_llm = MockTrackingLLMClient()
        stage1 = Stage1Filter(db=migrated_db)
        stage2 = Stage2Evaluator(db=migrated_db, settings=test_settings, llm_client=tracking_llm)

        # --------------------------------------------------------------------
        # CYCLE 1: Ingest 3 items:
        # - Item A1: "vLLM v0.7.0 Native Quantization Kernel" (qualifying GitHub repo)
        # - Item B1: "Llama.cpp Introduces GGUF v3" (qualifying GitHub repo)
        # - Item A2: "Native Quantization Kernel in vLLM 0.7.0" (near duplicate of A1 in same batch)
        # --------------------------------------------------------------------
        class Cycle1Source(BaseIngestionSource):
            source_name = "cycle1_source"

            async def fetch_items(self, limit: Any = None):
                yield RawItem(
                    source=self.source_name,
                    source_id="c1:1",
                    title="vLLM v0.7.0 Native Quantization Kernel",
                    url="https://github.com/vllm-project/vllm/releases/tag/v0.7.0",
                    raw_content="Dockerfile pyproject.toml Native quantization kernels for vLLM on RTX 4090.",
                    metadata={
                        "github_repo": "https://github.com/vllm-project/vllm",
                        "root_files": ["Dockerfile", "pyproject.toml"],
                        "license": "Apache-2.0",
                    },
                )
                yield RawItem(
                    source=self.source_name,
                    source_id="c1:2",
                    title="Llama.cpp Introduces GGUF v3",
                    url="https://github.com/ggerganov/llama.cpp/releases/tag/b3000",
                    raw_content="Dockerfile pyproject.toml GGUF v3 quantization for fast local inference.",
                    metadata={
                        "github_repo": "https://github.com/ggerganov/llama.cpp",
                        "root_files": ["Dockerfile", "pyproject.toml"],
                        "license": "MIT",
                    },
                )
                yield RawItem(
                    source=self.source_name,
                    source_id="c1:3",
                    title="Native Quantization Kernel in vLLM 0.7.0",
                    url="https://techblog.example.com/vllm-070-kernels?utm_source=twitter",
                    raw_content="Dockerfile pyproject.toml Native quantization kernels for vLLM on RTX 4090.",
                    metadata={
                        "github_repo": "https://github.com/vllm-project/vllm",
                        "root_files": ["Dockerfile", "pyproject.toml"],
                        "license": "Apache-2.0",
                    },
                )

        pipeline1 = IngestionPipeline(
            db=migrated_db,
            settings=test_settings,
            sources=[Cycle1Source(settings=test_settings)],
        )

        metrics1 = await pipeline1.run_cycle()

        # In Cycle 1: 3 items fetched, 1 duplicate dropped intra-batch -> 2 persisted
        assert metrics1["sources"]["cycle1_source"]["items_count"] == 3
        assert metrics1["sources"]["cycle1_source"]["persisted_count"] == 2
        assert metrics1["sources"]["cycle1_source"]["deduped_count"] == 1

        # Run Stage 1
        s1_evals1 = await stage1.run_pipeline()
        assert len(s1_evals1) == 2, f"Expected 2 Stage 1 evaluations, got {len(s1_evals1)}"
        assert all(e.passed for e in s1_evals1)

        # Run Stage 2
        breakthroughs1 = await stage2.evaluate_pending_items()
        assert len(breakthroughs1) == 2, f"Expected 2 breakthroughs, got {len(breakthroughs1)}"
        assert tracking_llm.call_count == 2, f"Expected exactly 2 LLM calls, got {tracking_llm.call_count}"

        # --------------------------------------------------------------------
        # CYCLE 2: Ingest 2 items:
        # - Item A3: "vLLM 0.7.0 Quantization Kernels Released" (syndicated wire of A1 from Cycle 1)
        # - Item C1: "SGLang v0.5.0 High-Performance Serving" (brand new unique item)
        # --------------------------------------------------------------------
        class Cycle2Source(BaseIngestionSource):
            source_name = "cycle2_source"

            async def fetch_items(self, limit: Any = None):
                yield RawItem(
                    source=self.source_name,
                    source_id="c2:1",
                    title="vLLM 0.7.0 Quantization Kernels Released",
                    url="https://syndicate.example.com/vllm-kernels?partner=rss&utm_medium=feed",
                    raw_content="Dockerfile pyproject.toml Native quantization kernels for vLLM on RTX 4090.",
                    metadata={
                        "github_repo": "https://github.com/vllm-project/vllm",
                        "root_files": ["Dockerfile", "pyproject.toml"],
                        "license": "Apache-2.0",
                    },
                )
                yield RawItem(
                    source=self.source_name,
                    source_id="c2:2",
                    title="SGLang v0.5.0 High-Performance Serving",
                    url="https://github.com/sgl-project/sglang/releases/tag/v0.5.0",
                    raw_content="Dockerfile pyproject.toml RadixAttention and fast structured decoding.",
                    metadata={
                        "github_repo": "https://github.com/sgl-project/sglang",
                        "root_files": ["Dockerfile", "pyproject.toml"],
                        "license": "Apache-2.0",
                    },
                )

        pipeline2 = IngestionPipeline(
            db=migrated_db,
            settings=test_settings,
            sources=[Cycle2Source(settings=test_settings)],
        )

        metrics2 = await pipeline2.run_cycle()

        # In Cycle 2: 2 items fetched, Item A3 dropped against DB -> only 1 persisted!
        assert metrics2["sources"]["cycle2_source"]["items_count"] == 2
        assert metrics2["sources"]["cycle2_source"]["persisted_count"] == 1
        assert metrics2["sources"]["cycle2_source"]["deduped_count"] == 1

        # Run Stage 1 in Cycle 2
        s1_evals2 = await stage1.run_pipeline()
        assert len(s1_evals2) == 1, f"Expected 1 Stage 1 evaluation, got {len(s1_evals2)}"
        assert s1_evals2[0].raw_item_id is not None

        # Run Stage 2 in Cycle 2
        breakthroughs2 = await stage2.evaluate_pending_items()
        assert len(breakthroughs2) == 1, f"Expected 1 breakthrough, got {len(breakthroughs2)}"

        # Total LLM calls should be exactly 2 (from Cycle 1) + 1 (from Cycle 2) = 3 calls
        # If A2 and A3 were evaluated, call_count would be 5.
        assert tracking_llm.call_count == 3, (
            f"Expected exactly 3 LLM calls across both cycles, but got {tracking_llm.call_count}. "
            f"Redundant LLM calls were NOT prevented!"
        )

        # Check DB raw_items table: only 3 unique rows exist (A1, B1, C1)
        rows = await migrated_db.fetch_all("SELECT id, title, metadata_json FROM raw_items ORDER BY id ASC;")
        assert len(rows) == 3

        # Check A1's metadata: it should have duplicate_count = 3 (A1 + A2 + A3)
        a1_row = next(r for r in rows if "vLLM" in r["title"])
        a1_meta = json.loads(a1_row["metadata_json"])
        assert a1_meta.get("duplicate_count") == 3
        assert len(a1_meta.get("sources", [])) == 3


# ============================================================================
# 3. CPU Latency Benchmark Tests for 50 and 100 Items
# ============================================================================

class TestCPULatencyBenchmarks:
    """Measures precise milliseconds taken to deduplicate 50 and 100 items on CPU."""

    def _generate_synthetic_items(self, count: int) -> list[RawItem]:
        items: list[RawItem] = []
        for i in range(count):
            # Introduce ~20% duplicates by repeating stories
            cluster_id = i // 2 if i < (count * 0.4) else i
            model_size = f"{((cluster_id % 4) + 1) * 7}b"
            version = f"v0.{cluster_id % 12}.0"
            is_reworded = (i % 2 == 1) and (i < count * 0.4)

            if is_reworded:
                title = f"Release of Model {model_size} {version} for Breakthrough Tool {cluster_id}"
            else:
                title = f"Tool {cluster_id} Model {model_size} {version} Released for Inference"

            items.append(
                RawItem(
                    source=f"source_{i % 5}",
                    source_id=f"item_{i}",
                    title=title,
                    url=f"https://example.com/tools/{cluster_id}?tracker={i}&utm_medium=rss",
                    raw_content=f"Detailed evaluation notes for tool {cluster_id} with {model_size} weights.",
                    metadata={"index": i, "cluster": cluster_id},
                    ingested_at=datetime.now(timezone.utc) - timedelta(hours=i),
                )
            )
        return items

    def test_benchmark_intra_batch_50_items(self) -> None:
        """Measures CPU latency for deduplicating 50 items (1,225 pairs)."""
        durations_ms: list[float] = []
        # Warmup
        items = self._generate_synthetic_items(50)
        deduplicate_items(items, threshold=0.80)

        for _ in range(10):
            items = self._generate_synthetic_items(50)
            t0 = time.perf_counter()
            deduped = deduplicate_items(items, threshold=0.80)
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

        mean_ms = statistics.mean(durations_ms)
        median_ms = statistics.median(durations_ms)
        min_ms = min(durations_ms)
        max_ms = max(durations_ms)

        print(f"\n[BENCHMARK] 50 Items Intra-Batch Latency: mean={mean_ms:.2f}ms, median={median_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms")

        # Must strictly satisfy < 100ms requirement on Raspberry Pi 5 / host CPU
        assert mean_ms < 100.0, f"50 items mean latency {mean_ms:.2f}ms exceeded 100ms"
        assert max_ms < 100.0, f"50 items max latency {max_ms:.2f}ms exceeded 100ms"
        assert len(deduped) < 50, "Expected duplicates to be detected and merged"

    def test_benchmark_intra_batch_100_items(self) -> None:
        """Measures CPU latency for deduplicating 100 items (4,950 pairs)."""
        durations_ms: list[float] = []
        # Warmup
        items = self._generate_synthetic_items(100)
        deduplicate_items(items, threshold=0.80)

        for _ in range(10):
            items = self._generate_synthetic_items(100)
            t0 = time.perf_counter()
            deduped = deduplicate_items(items, threshold=0.80)
            t1 = time.perf_counter()
            durations_ms.append((t1 - t0) * 1000.0)

        mean_ms = statistics.mean(durations_ms)
        median_ms = statistics.median(durations_ms)
        min_ms = min(durations_ms)
        max_ms = max(durations_ms)

        print(f"\n[BENCHMARK] 100 Items Intra-Batch Latency: mean={mean_ms:.2f}ms, median={median_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms")

        # Even for 100 items (4,950 pairs), pure Python algorithm should run in <100ms
        assert mean_ms < 100.0, f"100 items mean latency {mean_ms:.2f}ms exceeded 100ms"
        assert max_ms < 100.0, f"100 items max latency {max_ms:.2f}ms exceeded 100ms"
        assert len(deduped) < 100

    @pytest.mark.asyncio
    async def test_benchmark_inter_batch_against_db_50_and_100_items(self, migrated_db: Database) -> None:
        """Measures CPU + SQLite latency for deduplicating against 50 and 100 historical items in DB."""
        # Pre-populate DB with 100 items
        db_items = self._generate_synthetic_items(100)
        insert_query = """
        INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
        VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
        """
        for item in db_items:
            await migrated_db.execute(insert_query, item.to_db_params())

        # Test A: Deduplicate 50 incoming items against 100 DB items
        incoming_50 = self._generate_synthetic_items(50)
        # Shift titles slightly
        durations_50_ms: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            surviving = await deduplicate_against_db(incoming_50, db=migrated_db, lookback_days=7)
            t1 = time.perf_counter()
            durations_50_ms.append((t1 - t0) * 1000.0)

        mean_50 = statistics.mean(durations_50_ms)
        print(f"\n[BENCHMARK] 50 Items Inter-Batch (vs 100 DB items): mean={mean_50:.2f}ms, min={min(durations_50_ms):.2f}ms, max={max(durations_50_ms):.2f}ms")

        # Test B: Deduplicate 100 incoming items against 100 DB items
        incoming_100 = self._generate_synthetic_items(100)
        durations_100_ms: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            surviving = await deduplicate_against_db(incoming_100, db=migrated_db, lookback_days=7)
            t1 = time.perf_counter()
            durations_100_ms.append((t1 - t0) * 1000.0)

        mean_100 = statistics.mean(durations_100_ms)
        print(f"\n[BENCHMARK] 100 Items Inter-Batch (vs 100 DB items): mean={mean_100:.2f}ms, min={min(durations_100_ms):.2f}ms, max={max(durations_100_ms):.2f}ms")

        assert mean_50 < 100.0, f"50 items inter-batch mean {mean_50:.2f}ms exceeded 100ms"
        assert mean_100 < 100.0, f"100 items inter-batch mean {mean_100:.2f}ms exceeded 100ms"
