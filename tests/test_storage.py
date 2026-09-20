"""tests/test_storage.py
~~~~~~~~~~~~~~~~~~~~~
Exhaustive unit test suite for NewsScout storage layer:
- Database pragmas & WAL enforcement
- 8-table schema migrations, indexes, idempotency, and atomic rollback
- Asynchronous concurrency & serialized writes
- Busy timeout waiting and contention handling
- Dynamic preference store, atomic JSON sync, and few-shot calibration
- Pydantic v2 domain model validation and row roundtrip
- Pure-Python MP3 generator verification
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from pydantic import ValidationError
import pytest

from newsscout.storage.db import Database, create_connection
from newsscout.storage.migrations import (
    Migration,
    MigrationError,
    apply_migrations,
    async_apply_migrations,
    get_applied_versions,
    verify_schema_integrity,
)
import newsscout.storage.migrations as mig_mod
from newsscout.storage.models import (
    Breakthrough,
    DecisionCardData,
    FeedbackRating,
    RawItem,
    Source,
    SourceType,
    Stage1Evaluation,
    Stage2Category,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Group 1: Database Pragmas & Configuration Verification
# ============================================================================

class TestDatabasePragmas:
    """Verifies SQLite WAL pragmas, performance settings, and relational integrity."""

    def test_pragmas_applied_on_connection(self, temp_db_path: Path):
        conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        try:
            journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            assert journal_mode.lower() == "wal"

            synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
            assert synchronous == 1  # 1 == NORMAL

            busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
            assert busy_timeout == 5000

            foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            assert foreign_keys == 1

            temp_store = conn.execute("PRAGMA temp_store;").fetchone()[0]
            assert temp_store == 2  # 2 == MEMORY

            cache_size = conn.execute("PRAGMA cache_size;").fetchone()[0]
            assert cache_size == -64000  # 64MB cache
        finally:
            conn.close()

    def test_foreign_key_enforcement_raises_integrity_error(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            sync_db_conn.execute(
                """
                INSERT INTO stage1_evaluations (raw_item_id, passed, has_docker, has_runnable_code)
                VALUES (999999, 1, 1, 1);
                """
            )

    def test_wal_checkpoint_truncate(self, temp_db_path: Path):
        conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        try:
            conn.execute("CREATE TABLE test_cp (id INT PRIMARY KEY, val TEXT);")
            for i in range(20):
                conn.execute("INSERT INTO test_cp VALUES (?, ?);", (i, f"test_val_{i}"))
            res = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
            assert res[0] == 0  # Busy flag is 0 (checkpoint succeeded)
        finally:
            conn.close()


# ============================================================================
# Group 2: Schema Migrations & Integrity
# ============================================================================

class TestSchemaMigrations:
    """Verifies schema creation, migration idempotency, indexes, and integrity checks."""

    def test_migration_creates_all_8_tables(self, sync_db_conn: sqlite3.Connection):
        applied = apply_migrations(conn=sync_db_conn)
        assert applied == [1, 2]

        cursor = sync_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {
            "schema_migrations",
            "sources",
            "raw_items",
            "stage1_evaluations",
            "breakthroughs",
            "digests",
            "digest_tracks",
            "feedback",
            "user_preferences",
        }
        assert expected_tables.issubset(tables)

    def test_indexes_created(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)

        cursor = sync_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        expected_indexes = {
            "idx_raw_items_source",
            "idx_raw_items_ingested",
            "idx_raw_items_url",
            "idx_stage1_passed",
            "idx_stage1_evaluated",
            "idx_breakthroughs_cat",
            "idx_breakthroughs_scores",
            "idx_breakthroughs_watchlisted",
            "idx_breakthroughs_evaluated",
            "idx_digests_date_slot",
            "idx_digests_status",
            "idx_digest_tracks_digest",
            "idx_digest_tracks_breakthrough",
            "idx_feedback_breakthrough",
            "idx_feedback_user",
            "idx_feedback_rating",
            "idx_feedback_created",
            "idx_user_preferences_key",
        }
        assert expected_indexes.issubset(indexes)

    def test_migration_is_idempotent(self, sync_db_conn: sqlite3.Connection):
        assert apply_migrations(conn=sync_db_conn) == [1, 2]
        assert apply_migrations(conn=sync_db_conn) == []
        assert apply_migrations(conn=sync_db_conn) == []

        versions = get_applied_versions(sync_db_conn)
        assert versions == {1, 2}

    def test_migration_seed_sources(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)

        cursor = sync_db_conn.execute("SELECT name, source_type, is_active FROM sources ORDER BY name;")
        sources = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

        assert "github_releases" in sources
        assert sources["github_releases"] == ("api", 1)
        assert "hacker_news" in sources
        assert sources["hacker_news"] == ("firebase", 1)
        assert "hf_daily_papers" in sources
        assert sources["hf_daily_papers"] == ("api", 1)

    def test_migration_atomic_rollback(self, sync_db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch):
        apply_migrations(conn=sync_db_conn)

        faulty_migration = Migration(
            version=3,
            name="003_broken_migration",
            up_sql="CREATE TABLE valid_test (id INT); INVALID SQL STATEMENT HERE;",
        )
        monkeypatch.setattr(mig_mod, "MIGRATIONS", mig_mod.MIGRATIONS + [faulty_migration])

        with pytest.raises(MigrationError):
            apply_migrations(conn=sync_db_conn)

        # Verify version 3 is NOT recorded in ledger
        assert get_applied_versions(sync_db_conn) == {1, 2}

        # Verify table valid_test was completely rolled back
        cursor = sync_db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='valid_test';"
        )
        assert cursor.fetchone() is None

    def test_integrity_check_clean(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)
        is_healthy, errors = verify_schema_integrity(sync_db_conn)
        assert is_healthy is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_async_apply_migrations(self, temp_db: Database):
        applied = await async_apply_migrations(temp_db)
        assert applied == [1, 2]
        applied_again = await async_apply_migrations(temp_db)
        assert applied_again == []


# ============================================================================
# Group 3: Concurrent Read/Write Transactions
# ============================================================================

class TestConcurrentTransactions:
    """Verifies SQLite WAL concurrency under parallel async task workloads."""

    @pytest.mark.asyncio
    async def test_concurrent_writers_and_readers(self, migrated_db: Database, factory: Any):
        # 1. Seed base raw items
        for i in range(5):
            raw = factory.create_raw_item(source_id=f"test/repo:{i}")
            await migrated_db.execute(
                """
                INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (raw.source, raw.source_id, raw.title, raw.url, raw.raw_content, json.dumps(raw.metadata)),
            )

        # Also seed a breakthrough so feedback foreign key succeeds
        bt = factory.create_breakthrough(raw_item_id=1)
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements,
                                      license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
             bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements,
             bt.license, bt.card_markdown),
        )

        # 2. Concurrently execute 10 writes and 10 reads
        write_errors: list[tuple[int, Exception]] = []
        read_errors: list[tuple[int, Exception]] = []

        async def _reader(reader_id: int):
            try:
                for _ in range(5):
                    rows = await migrated_db.fetch_all("SELECT * FROM raw_items;")
                    assert len(rows) >= 5
                    await asyncio.sleep(0.01)
            except Exception as e:
                read_errors.append((reader_id, e))

        async def _writer(writer_id: int):
            try:
                await migrated_db.execute(
                    """
                    INSERT INTO feedback (breakthrough_id, rating, user_identifier, source)
                    VALUES (?, ?, ?, ?);
                    """,
                    (1, "hit", str(1000 + writer_id), "telegram"),
                )
            except Exception as e:
                write_errors.append((writer_id, e))

        readers = [asyncio.create_task(_reader(i)) for i in range(10)]
        writers = [asyncio.create_task(_writer(i)) for i in range(10)]

        await asyncio.gather(*readers, *writers)

        assert len(read_errors) == 0, f"Reader errors: {read_errors}"
        assert len(write_errors) == 0, f"Writer errors: {write_errors}"

        # Verify all 10 feedbacks were saved
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        assert count == 10

    @pytest.mark.asyncio
    async def test_read_your_own_writes(self, migrated_db: Database, factory: Any):
        raw = factory.create_raw_item(source_id="unique/read-own-write")
        row_id = await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (raw.source, raw.source_id, raw.title, raw.url, raw.raw_content, json.dumps(raw.metadata)),
        )
        assert row_id > 0

        row = await migrated_db.fetch_one("SELECT * FROM raw_items WHERE id = ?;", (row_id,))
        assert row is not None
        assert row["source_id"] == "unique/read-own-write"

    @pytest.mark.asyncio
    async def test_transaction_context_manager_commit(self, migrated_db: Database):
        async with migrated_db.transaction() as conn:
            conn.execute(
                "INSERT INTO sources (name, source_type, is_active) VALUES (?, ?, ?);",
                ("custom_source", "api", 1),
            )

        row = await migrated_db.fetch_one("SELECT * FROM sources WHERE name = 'custom_source';")
        assert row is not None
        assert row["name"] == "custom_source"

    @pytest.mark.asyncio
    async def test_transaction_context_manager_rollback_on_error(self, migrated_db: Database):
        with pytest.raises(ValueError, match="Intentional failure"):
            async with migrated_db.transaction() as conn:
                conn.execute(
                    "INSERT INTO sources (name, source_type, is_active) VALUES (?, ?, ?);",
                    ("rollback_source", "api", 1),
                )
                raise ValueError("Intentional failure")

        row = await migrated_db.fetch_one("SELECT * FROM sources WHERE name = 'rollback_source';")
        assert row is None


# ============================================================================
# Group 4: Busy Timeout Verification
# ============================================================================

class TestBusyTimeout:
    """Verifies SQLite busy_timeout waiting semantics and lock contention."""

    def test_busy_timeout_waits_and_succeeds(self, temp_db_path: Path):
        conn_a = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        conn_b = create_connection(temp_db_path, timeout=5.0, init_wal=False)
        conn_a.execute("CREATE TABLE lock_test (id INT PRIMARY KEY);")

        lock_released = threading.Event()

        def _hold_lock():
            conn_a.execute("BEGIN EXCLUSIVE;")
            conn_a.execute("INSERT INTO lock_test VALUES (1);")
            time.sleep(0.3)
            conn_a.execute("COMMIT;")
            conn_a.close()
            lock_released.set()

        t = threading.Thread(target=_hold_lock)
        t.start()
        time.sleep(0.05)  # Ensure conn_a acquired the lock

        # conn_b attempts write with 5000ms busy_timeout
        start_time = time.monotonic()
        conn_b.execute("BEGIN IMMEDIATE;")
        conn_b.execute("INSERT INTO lock_test VALUES (2);")
        conn_b.execute("COMMIT;")
        elapsed = time.monotonic() - start_time
        conn_b.close()
        t.join()

        assert elapsed >= 0.25, "conn_b did not wait for conn_a to release lock"
        assert lock_released.is_set()

    def test_busy_timeout_exceeded_raises(self, temp_db_path: Path):
        conn_a = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        conn_b = create_connection(temp_db_path, timeout=0.1, init_wal=False)
        conn_b.execute("PRAGMA busy_timeout = 100;")  # Explicit short 100ms timeout
        conn_a.execute("CREATE TABLE lock_timeout_test (id INT PRIMARY KEY);")

        conn_a.execute("BEGIN EXCLUSIVE;")
        conn_a.execute("INSERT INTO lock_timeout_test VALUES (1);")

        # conn_b with 100ms timeout must raise OperationalError
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            conn_b.execute("BEGIN IMMEDIATE;")

        conn_a.execute("ROLLBACK;")
        conn_a.close()
        conn_b.close()


# ============================================================================
# Group 5: Preference Store & JSON Synchronization
# ============================================================================

class TestPreferenceStoreAndSync:
    """Verifies feedback UPSERT, atomic preferences.json synchronization, and few-shot generation."""

    @pytest.mark.asyncio
    async def test_feedback_upsert_and_json_sync(self, migrated_db: Database, test_settings: Any, factory: Any):
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # 1. Seed raw item and breakthrough
        raw = factory.create_raw_item()
        raw_id = await migrated_db.execute(
            "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
            (raw.source, raw.source_id, raw.title, raw.url),
        )
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements,
                                      license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
             bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements,
             bt.license, bt.card_markdown),
        )

        # 2. Record rating
        fb = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.HIT,
            telegram_user_id=12345,
            notes="Outstanding on RTX 4090",
        )
        assert fb.rating == FeedbackRating.HIT

        # Verify SQLite row
        row = await migrated_db.fetch_one("SELECT * FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert row is not None
        assert row["rating"] == "hit"

        # Verify preferences.json created and valid
        pref_file = Path(test_settings.preferences_file)
        assert pref_file.exists()
        pref_data = json.loads(pref_file.read_text(encoding="utf-8"))
        assert "feedback_history" in pref_data
        assert len(pref_data["feedback_history"]) == 1
        assert "exemplars" in pref_data
        assert "stats" in pref_data

        # 3. Update vote (UPSERT)
        fb_updated = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            telegram_user_id=12345,
            notes="Changed to Inspire",
        )
        assert fb_updated.rating == FeedbackRating.INSPIRE

        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        assert count == 1  # No duplicate rows created!

    @pytest.mark.asyncio
    async def test_corrupted_json_recovery(self, migrated_db: Database, test_settings: Any):
        service = PreferencesService(db=migrated_db, settings=test_settings)
        pref_file = Path(test_settings.preferences_file)
        pref_file.write_text("{CORRUPTED INVALID JSON CONTENT")

        # Must not crash; should recover gracefully
        data = await service.load_preferences_from_json()
        assert isinstance(data, dict)
        assert "version" in data

    @pytest.mark.asyncio
    async def test_cold_start_exemplars_fallback(self, migrated_db: Database, test_settings: Any):
        service = PreferencesService(db=migrated_db, settings=test_settings)
        pos, neg = await service.get_few_shot_exemplars(count=3)

        assert len(pos) == 3
        assert len(neg) == 2  # Default negative count is 2 in cold start
        assert pos[0].title == "vLLM Native NVFP4 Kernel Execution"
        assert neg[0].rating == FeedbackRating.HYPE

    @pytest.mark.asyncio
    async def test_few_shot_exemplars_with_rated_data(self, migrated_db: Database, test_settings: Any, factory: Any):
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Ingest and rate a breakthrough
        raw = factory.create_raw_item(source_id="test/custom_repo")
        raw_id = await migrated_db.execute(
            "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
            (raw.source, raw.source_id, raw.title, raw.url),
        )
        bt = factory.create_breakthrough(raw_item_id=raw_id, title="Custom Fast Agent Harness")
        bt_id = await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements,
                                      license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
             bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements,
             bt.license, bt.card_markdown),
        )
        await service.record_feedback(breakthrough_id=bt_id, rating=FeedbackRating.HIT, telegram_user_id=777)

        pos, _ = await service.get_few_shot_exemplars(count=3)
        assert pos[0].title == "Custom Fast Agent Harness"

    @pytest.mark.asyncio
    async def test_preference_stats_calculation(self, migrated_db: Database, test_settings: Any, factory: Any):
        service = PreferencesService(db=migrated_db, settings=test_settings)
        stats_empty = await service.get_preference_stats()
        assert stats_empty["total_ratings"] == 0
        assert stats_empty["hit_rate_pct"] == 0.0

        # Seed 2 raw items, 2 breakthroughs, 2 feedbacks (1 hit, 1 hype)
        for i, rating in enumerate([FeedbackRating.HIT, FeedbackRating.HYPE], start=1):
            raw = factory.create_raw_item(source_id=f"stat_repo_{i}")
            raw_id = await migrated_db.execute(
                "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
                (raw.source, raw.source_id, raw.title, raw.url),
            )
            bt = factory.create_breakthrough(raw_item_id=raw_id, title=f"Stat BT {i}")
            bt_id = await migrated_db.execute(
                """
                INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                          tldr, use_case, comparison, quickstart, hardware_requirements,
                                          license, card_markdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
                 bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements,
                 bt.license, bt.card_markdown),
            )
            await service.record_feedback(breakthrough_id=bt_id, rating=rating, telegram_user_id=888)

        stats = await service.get_preference_stats()
        assert stats["total_ratings"] == 2
        assert stats["hit_count"] == 1
        assert stats["hype_count"] == 1
        assert stats["hit_rate_pct"] == 50.0


# ============================================================================
# Group 6: Domain Models Validation
# ============================================================================

class TestPydanticModelsValidation:
    """Verifies domain model constraints, enums, score bounds, and SQLite row mapping."""

    def test_score_bounds_validation(self, factory: Any):
        valid = factory.create_breakthrough(breakthrough_score=9.5, roi_score=8.5)
        assert valid.breakthrough_score == 9.5

        with pytest.raises(ValidationError):
            factory.create_breakthrough(breakthrough_score=10.5)

        with pytest.raises(ValidationError):
            factory.create_breakthrough(breakthrough_score=0.2)

    def test_category_enum_validation(self, factory: Any):
        valid = factory.create_breakthrough(category=Stage2Category.SERENDIPITY)
        assert valid.category == Stage2Category.SERENDIPITY

        with pytest.raises(ValidationError):
            Breakthrough(
                raw_item_id=1,
                title="Test",
                breakthrough_score=8.0,
                roi_score=8.0,
                category="invalid_category",  # type: ignore
                tldr="Tldr",
                use_case="Use case",
                comparison="Comp",
                quickstart="docker run",
                hardware_requirements="4090",
                license="MIT",
                card_markdown="Card",
            )

    def test_decision_card_render_contains_all_components(self):
        card = DecisionCardData(
            tldr="1-sentence breakthrough summary.",
            use_case="Direct RTX 4090 acceleration.",
            comparison="4x lower latency than vLLM baseline.",
            quickstart="docker run --gpus all -p 8000:8000 image:latest",
            hardware_requirements="24GB VRAM (RTX 4090)",
            license="Apache-2.0",
        )
        md = card.render_markdown(title="FastInference Engine")
        assert "FastInference Engine" in md
        assert "**TL;DR**:" in md
        assert "**Workflow Use Case**:" in md
        assert "**Baseline Comparison**:" in md
        assert "`docker run --gpus all -p 8000:8000 image:latest`" in md
        assert "24GB VRAM (RTX 4090)" in md

    def test_decision_card_render_telegram_html(self, factory: Any):
        bt = factory.create_breakthrough()
        dc = bt.to_decision_card()
        html = dc.render_telegram_html()
        assert "<b>" in html
        assert "<code>" in html
        assert bt.title in html

    @pytest.mark.asyncio
    async def test_roundtrip_row_conversion(self, migrated_db: Database, factory: Any):
        # 1. Test Source roundtrip
        src = factory.create_source(name="test_feed", source_type=SourceType.FEED)
        await migrated_db.execute(
            """
            INSERT INTO sources (name, source_type, config_json, is_active, last_poll_at, created_at)
            VALUES (:name, :source_type, :config_json, :is_active, :last_poll_at, :created_at);
            """,
            src.to_db_params(),
        )
        row = await migrated_db.fetch_one("SELECT * FROM sources WHERE name = 'test_feed';")
        assert row is not None
        src_recovered = Source.from_row(row)
        assert src_recovered.name == src.name
        assert src_recovered.source_type == SourceType.FEED
        assert src_recovered.is_active is True

        # 2. Test RawItem roundtrip
        raw = factory.create_raw_item(source="test_feed", source_id="item_42")
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
            """,
            raw.to_db_params(),
        )
        row_raw = await migrated_db.fetch_one("SELECT * FROM raw_items WHERE source_id = 'item_42';")
        assert row_raw is not None
        raw_recovered = RawItem.from_row(row_raw)
        assert raw_recovered.title == raw.title
        assert raw_recovered.metadata == raw.metadata

    def test_mock_mp3_factory_generates_valid_bitstream(self, mock_mp3_factory: Any):
        mp3_file = mock_mp3_factory("sample.mp3", duration_seconds=1.0)
        assert mp3_file.exists()
        raw_bytes = mp3_file.read_bytes()
        # Header starts with 0xFFFB (MPEG-1 Layer 3 Sync)
        assert raw_bytes[0] == 0xFF
        assert raw_bytes[1] == 0xFB
        assert len(raw_bytes) > 400
