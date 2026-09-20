"""tests/test_adversarial_stress.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adversarial Concurrency & Stress Harness for Milestone 1:
1. SQLite WAL high-concurrency competing read/write threads and tasks under rapid fire
2. Migration idempotency and rollback under intentional SQL errors
3. Atomic preferences.json file replacement and reader/writer race conditions
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from typing import Any

import pytest

from newsscout.config import Settings
from newsscout.storage.db import Database, create_connection, sync_transaction
from newsscout.storage.migrations import (
    Migration,
    MigrationError,
    apply_migrations,
    get_applied_versions,
    verify_schema_integrity,
)
import newsscout.storage.migrations as mig_mod
from newsscout.storage.models import FeedbackRating
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Section 1: SQLite WAL Concurrency & busy_timeout=5000 Stress Testing
# ============================================================================

class TestWALConcurrencyStress:
    """Stress-tests SQLite WAL concurrency under heavy multithreaded and multi-task loads."""

    def test_multithreaded_competing_writers_no_locked_errors(self, temp_db_path: Path):
        """25 concurrent threads rapidly writing with individual connections.
        
        Validates:
        - busy_timeout=5000 queues competing threads without 'database is locked'.
        - Zero lost updates across 250 cumulative writes.
        """
        # Initialize schema first
        init_conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        init_conn.execute("CREATE TABLE stress_counter (thread_id INT, iter_id INT, val TEXT);")
        init_conn.close()

        num_threads = 25
        writes_per_thread = 10
        errors: list[Exception] = []
        barrier = threading.Barrier(num_threads)

        def _worker(thread_id: int):
            try:
                conn = create_connection(temp_db_path, timeout=5.0, init_wal=False)
                barrier.wait()  # Synchronize threads for maximal contention
                for i in range(writes_per_thread):
                    with sync_transaction(conn):
                        conn.execute(
                            "INSERT INTO stress_counter (thread_id, iter_id, val) VALUES (?, ?, ?);",
                            (thread_id, i, f"data_{thread_id}_{i}"),
                        )
                conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Encountered {len(errors)} errors during concurrent writing: {errors}"

        # Verify integrity and total count
        verify_conn = create_connection(temp_db_path, timeout=5.0, init_wal=False)
        total_rows = verify_conn.execute("SELECT COUNT(*) FROM stress_counter;").fetchone()[0]
        verify_conn.close()
        assert total_rows == num_threads * writes_per_thread

    def test_concurrent_readers_never_blocked_by_long_write_transaction(self, temp_db_path: Path):
        """Validates that under WAL mode, concurrent readers are NEVER blocked by a long write transaction."""
        init_conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        init_conn.execute("CREATE TABLE read_test (id INT PRIMARY KEY, val TEXT);")
        for i in range(50):
            init_conn.execute("INSERT INTO read_test VALUES (?, ?);", (i, f"initial_{i}"))
        init_conn.close()

        write_acquired = threading.Event()
        write_done = threading.Event()
        read_errors: list[Exception] = []
        read_times: list[float] = []

        def _long_writer():
            conn = create_connection(temp_db_path, timeout=5.0, init_wal=False)
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("INSERT INTO read_test VALUES (999, 'held_value');")
            write_acquired.set()
            time.sleep(0.4)  # Hold write lock for 400ms
            conn.execute("COMMIT;")
            conn.close()
            write_done.set()

        writer_thread = threading.Thread(target=_long_writer)
        writer_thread.start()
        assert write_acquired.wait(timeout=2.0)

        # Concurrently read while write transaction is actively held
        def _reader():
            try:
                t0 = time.monotonic()
                r_conn = create_connection(temp_db_path, timeout=5.0, init_wal=False)
                rows = r_conn.execute("SELECT COUNT(*) FROM read_test;").fetchone()[0]
                # In WAL mode, snapshot isolation should see 50 rows (uncommitted 999 not visible)
                assert rows == 50
                r_conn.close()
                read_times.append(time.monotonic() - t0)
            except Exception as e:
                read_errors.append(e)

        reader_threads = [threading.Thread(target=_reader) for _ in range(20)]
        for rt in reader_threads:
            rt.start()
        for rt in reader_threads:
            rt.join()

        writer_thread.join()

        assert len(read_errors) == 0, f"Readers failed during active write: {read_errors}"
        # Readers should not have waited for the 400ms write transaction to commit
        max_read_time = max(read_times)
        assert max_read_time < 0.2, f"Reader took {max_read_time:.3f}s - blocked by writer!"

    @pytest.mark.asyncio
    async def test_high_concurrency_async_tasks_competing(self, migrated_db: Database, factory: Any):
        """50 concurrent async tasks (25 writers and 25 readers) hammering the async Database instance."""
        # Pre-seed items
        for i in range(10):
            raw = factory.create_raw_item(source_id=f"bench:{i}")
            await migrated_db.execute(
                "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
                (raw.source, raw.source_id, raw.title, raw.url),
            )

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

        read_errors: list[Exception] = []
        write_errors: list[Exception] = []

        async def _async_reader(task_id: int):
            try:
                for _ in range(5):
                    rows = await migrated_db.fetch_all("SELECT id, title FROM breakthroughs;")
                    assert len(rows) >= 1
                    val = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
                    assert val is not None
                    await asyncio.sleep(0.005)
            except Exception as exc:
                read_errors.append(exc)

        async def _async_writer(task_id: int):
            try:
                for step in range(3):
                    user_id = task_id * 100 + step
                    await migrated_db.execute(
                        "INSERT INTO feedback (breakthrough_id, rating, user_identifier) VALUES (?, ?, ?);",
                        (1, "hit", str(user_id)),
                    )
                    await asyncio.sleep(0.005)
            except Exception as exc:
                write_errors.append(exc)

        readers = [asyncio.create_task(_async_reader(i)) for i in range(25)]
        writers = [asyncio.create_task(_async_writer(i)) for i in range(25)]

        await asyncio.gather(*readers, *writers)

        assert len(read_errors) == 0, f"Async read errors: {read_errors}"
        assert len(write_errors) == 0, f"Async write errors: {write_errors}"

        # 25 writers * 3 inserts = 75 feedback rows
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        assert count == 75

    def test_busy_timeout_oracle_behavior(self, temp_db_path: Path):
        """Empirical verification of busy_timeout:
        - Connection B with short timeout (100ms) times out.
        - Connection C with sufficient timeout (1000ms) waits and succeeds.
        """
        init_conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        init_conn.execute("CREATE TABLE oracle_test (id INT PRIMARY KEY);")
        init_conn.close()

        # Conn A holds exclusive lock for 300ms
        conn_a = create_connection(temp_db_path, timeout=5.0, init_wal=False)
        conn_a.execute("BEGIN EXCLUSIVE;")
        conn_a.execute("INSERT INTO oracle_test VALUES (1);")

        def _release_after_delay():
            time.sleep(0.3)
            conn_a.execute("COMMIT;")
            conn_a.close()

        threading.Thread(target=_release_after_delay).start()

        # Connection with 100ms timeout must fail
        conn_fast = create_connection(temp_db_path, timeout=0.1, init_wal=False)
        conn_fast.execute("PRAGMA busy_timeout = 100;")
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            conn_fast.execute("BEGIN IMMEDIATE;")
        conn_fast.close()

        # Connection with 2000ms timeout must wait and succeed
        conn_patient = create_connection(temp_db_path, timeout=2.0, init_wal=False)
        conn_patient.execute("PRAGMA busy_timeout = 2000;")
        t0 = time.monotonic()
        conn_patient.execute("BEGIN IMMEDIATE;")
        conn_patient.execute("INSERT INTO oracle_test VALUES (2);")
        conn_patient.execute("COMMIT;")
        conn_patient.close()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.1, f"Did not wait for release, elapsed: {elapsed}"


# ============================================================================
# Section 2: Schema Migration Rollback & Concurrency Stress Testing
# ============================================================================

class TestMigrationAdversarialStress:
    """Stress-tests migration idempotency, rollback on intentional failure, and concurrency."""

    def test_multi_statement_migration_atomic_rollback(self, sync_db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch):
        """Validates that a multi-statement migration that fails halfway rolls back ALL preceding statements."""
        apply_migrations(conn=sync_db_conn)

        # Migration creates table A, inserts row into A, then attempts invalid table creation
        faulty_sql = """
        CREATE TABLE canary_table_a (id INTEGER PRIMARY KEY, marker TEXT);
        INSERT INTO canary_table_a (id, marker) VALUES (1, 'survived_part1');
        INVALID SQL SYNTAX THAT CANNOT BE PARSED;
        """
        bad_migration = Migration(
            version=99,
            name="099_faulty_multi_statement",
            up_sql=faulty_sql,
        )
        monkeypatch.setattr(mig_mod, "MIGRATIONS", mig_mod.MIGRATIONS + [bad_migration])

        with pytest.raises(MigrationError):
            apply_migrations(conn=sync_db_conn)

        # Verify canary_table_a does NOT exist (all statements rolled back)
        cursor = sync_db_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='canary_table_a';")
        assert cursor.fetchone() is None

        # Verify version 99 was NOT recorded in schema_migrations
        versions = get_applied_versions(sync_db_conn)
        assert 99 not in versions

    def test_foreign_key_violation_in_migration_triggers_rollback(self, sync_db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch):
        """Validates that inserting an orphan foreign key inside a migration causes atomic rollback."""
        apply_migrations(conn=sync_db_conn)

        fk_violation_sql = """
        INSERT INTO stage1_evaluations (raw_item_id, passed, has_docker, has_runnable_code)
        VALUES (99999999, 1, 1, 1);
        """
        bad_migration = Migration(
            version=88,
            name="088_fk_violation",
            up_sql=fk_violation_sql,
        )
        monkeypatch.setattr(mig_mod, "MIGRATIONS", mig_mod.MIGRATIONS + [bad_migration])

        with pytest.raises(MigrationError):
            apply_migrations(conn=sync_db_conn)

        versions = get_applied_versions(sync_db_conn)
        assert 88 not in versions

    def test_rapid_loop_idempotency(self, sync_db_conn: sqlite3.Connection):
        """Calling apply_migrations 50 times repeatedly in a tight loop."""
        first_run = apply_migrations(conn=sync_db_conn)
        assert first_run == [1, 2]

        for _ in range(50):
            res = apply_migrations(conn=sync_db_conn)
            assert res == []

        is_healthy, errors = verify_schema_integrity(sync_db_conn)
        assert is_healthy is True
        assert errors == []

    def test_concurrent_apply_migrations_race_condition_fresh_db(self, temp_db_path: Path):
        """Challenger Stress Test: What happens when 10 threads call apply_migrations simultaneously on a brand new DB?
        
        Exposes whether create_connection / apply_migrations handles concurrent connection
        initialization without 'database is locked' during PRAGMA journal_mode = WAL.
        """
        num_threads = 10
        errors: list[Exception] = []
        applied_results: list[list[int]] = []
        barrier = threading.Barrier(num_threads)

        def _migrator():
            try:
                barrier.wait()
                applied = apply_migrations(db_path=temp_db_path)
                applied_results.append(applied)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_migrator) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent apply_migrations on fresh DB produced errors: {errors}"

    def test_concurrent_apply_migrations_existing_db(self, temp_db_path: Path):
        """10 threads calling apply_migrations concurrently on an already initialized WAL DB."""
        # Initialize WAL mode once first
        init_conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
        init_conn.close()

        num_threads = 10
        errors: list[Exception] = []
        applied_results: list[list[int]] = []
        barrier = threading.Barrier(num_threads)

        def _migrator():
            try:
                barrier.wait()
                applied = apply_migrations(db_path=temp_db_path)
                applied_results.append(applied)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_migrator) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent apply_migrations on initialized DB produced errors: {errors}"


# ============================================================================
# Section 3: Preferences JSON Atomic Replacement & Concurrency Stress Testing
# ============================================================================

class TestPreferencesConcurrencyStress:
    """Stress-tests preferences.json atomic file replacement and reader/writer race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_feedback_recording_preferences_json_integrity(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """20 concurrent coroutines calling PreferencesService.record_feedback.
        
        Validates:
        - No file collision or PermissionError on Windows during .tmp + os.replace.
        - preferences.json is not corrupted or truncated.
        - Exactly 20 ratings recorded without race conditions.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Pre-seed 20 breakthroughs
        bt_ids: list[int] = []
        for i in range(20):
            raw = factory.create_raw_item(source_id=f"stress_pref:{i}")
            raw_id = await migrated_db.execute(
                "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
                (raw.source, raw.source_id, raw.title, raw.url),
            )
            bt = factory.create_breakthrough(raw_item_id=raw_id, title=f"Breakthrough {i}")
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
            bt_ids.append(bt_id)

        # Concurrently record feedback for all 20 breakthroughs
        async def _record(idx: int, bt_id: int):
            rating = FeedbackRating.HIT if idx % 2 == 0 else FeedbackRating.INSPIRE
            return await service.record_feedback(
                breakthrough_id=bt_id,
                rating=rating,
                telegram_user_id=5000 + idx,
                notes=f"Stress note {idx}",
            )

        tasks = [_record(i, bt_ids[i]) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify that SQLite DB has all 20 feedbacks
        db_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        assert db_count == 20, f"Expected 20 rows in SQLite DB, found {db_count}"

        # Verify preferences.json is valid and contains all 20 feedbacks
        pref_file = Path(test_settings.preferences_file)
        assert pref_file.exists()
        content = pref_file.read_text(encoding="utf-8")
        parsed = json.loads(content)

        # STRESS-TEST FAILURE DETECTION:
        # In concurrent execution, an earlier sync_preferences_file call can finish AFTER a later one,
        # overwriting preferences.json with a stale snapshot (e.g. 16 instead of 20 ratings).
        assert len(parsed["feedback_history"]) == 20, (
            f"Stale snapshot overwrite in preferences.json! DB has 20 rows, but preferences.json has "
            f"{len(parsed['feedback_history'])} history entries."
        )
        assert parsed["stats"]["total_ratings"] == 20

    @pytest.mark.asyncio
    async def test_concurrent_readers_and_writers_preferences_json(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Simultaneous rapid readers loading preferences.json while writers update it.
        
        Validates:
        - Readers never read a half-written file (JSONDecodeError).
        - Atomic rename prevents dirty reads.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Seed 1 initial feedback so file exists
        raw = factory.create_raw_item(source_id="seed:0")
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
        await service.record_feedback(breakthrough_id=bt_id, rating=FeedbackRating.HIT, telegram_user_id=1)

        read_errors: list[Exception] = []
        write_errors: list[Exception] = []
        stop_event = asyncio.Event()

        async def _reader():
            while not stop_event.is_set():
                try:
                    data = await service.load_preferences_from_json()
                    assert isinstance(data, dict)
                    assert "version" in data
                except Exception as exc:
                    read_errors.append(exc)
                await asyncio.sleep(0.002)

        async def _writer(writer_id: int):
            for step in range(5):
                try:
                    # Update feedback
                    await service.record_feedback(
                        breakthrough_id=bt_id,
                        rating=FeedbackRating.INSPIRE if step % 2 == 0 else FeedbackRating.HIT,
                        telegram_user_id=writer_id,
                        notes=f"Step {step}",
                    )
                except Exception as exc:
                    write_errors.append(exc)
                await asyncio.sleep(0.01)

        readers = [asyncio.create_task(_reader()) for _ in range(5)]
        writers = [asyncio.create_task(_writer(100 + i)) for i in range(5)]

        await asyncio.gather(*writers)
        stop_event.set()
        await asyncio.gather(*readers)

        assert len(write_errors) == 0, f"Writer errors in preferences: {write_errors}"
        assert len(read_errors) == 0, f"Reader errors in preferences: {read_errors}"

    @pytest.mark.asyncio
    async def test_rapid_fire_sync_preferences_collision(
        self, migrated_db: Database, test_settings: Settings
    ):
        """Rapid-fire concurrent calls to sync_preferences_file with zero delays.
        
        Tests whether static `.tmp` filename (preferences.json.tmp) collides when
        multiple coroutines attempt simultaneous file open/replace.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        errors: list[Exception] = []

        async def _hammer():
            try:
                await service.sync_preferences_file()
            except Exception as e:
                errors.append(e)

        # Launch 30 simultaneous syncs
        await asyncio.gather(*[_hammer() for _ in range(30)])

        assert len(errors) == 0, f"Rapid-fire sync_preferences_file produced errors: {errors}"
        # Validate that the resulting file is valid JSON
        pref_file = Path(test_settings.preferences_file)
        data = json.loads(pref_file.read_text(encoding="utf-8"))
        assert "version" in data

    @pytest.mark.asyncio
    async def test_corrupted_json_concurrent_recovery(
        self, migrated_db: Database, test_settings: Settings
    ):
        """Simultaneous calls to load_preferences_from_json when preferences.json is corrupted."""
        service = PreferencesService(db=migrated_db, settings=test_settings)
        pref_file = Path(test_settings.preferences_file)
        pref_file.write_text("{BAD_JSON_GARBAGE_WITHOUT_CLOSING_BRACE")

        errors: list[Exception] = []

        async def _reader():
            try:
                data = await service.load_preferences_from_json()
                assert "version" in data
            except Exception as e:
                errors.append(e)

        await asyncio.gather(*[_reader() for _ in range(10)])
        assert len(errors) == 0, f"Concurrent recovery produced errors: {errors}"
