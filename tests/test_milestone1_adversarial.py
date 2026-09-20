"""tests/test_milestone1_adversarial.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Adversarial Challenge Suite for TASK-08 Milestone 1:
1. Concurrency: Multi-threaded apply_migrations (25+ threads, fresh DB, partially migrated DB, competing locks)
2. Multi-Channel Feedback: Burst recording across 25+ workers with Signal phone numbers and UUIDs, Telegram IDs, Web users
3. Constraint Collisions: Simultaneous UPSERT races on same (breakthrough_id, user_identifier), whitespace & legacy collision
4. SQL Injection Attacks: Injections in user_identifier, notes, source, rating
5. Foreign Key Integrity & Boundary Payloads: Non-existent breakthrough_id, 500KB notes, multi-channel few-shot calibration
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

import pytest

from newsscout.config import Settings
from newsscout.storage.db import Database, create_connection, sync_transaction
from newsscout.storage.migrations import (
    MIGRATIONS,
    Migration,
    MigrationError,
    apply_migrations,
    get_applied_versions,
    verify_schema_integrity,
)
from newsscout.storage.models import (
    Feedback,
    FeedbackCreate,
    FeedbackRating,
    FeedbackSource,
)
from newsscout.storage.preferences import PreferencesService


# ============================================================================
# Section 1: Adversarial Concurrency in Schema Migrations
# ============================================================================

class TestAdversarialMigration002Concurrency:
    """Stress-tests Migration 002 and apply_migrations under high multi-threaded contention."""

    def test_fresh_db_25_threads_concurrent_apply_migrations(self, tmp_path: Path):
        """25 threads simultaneously calling apply_migrations on a completely new database file.

        Verifies:
        - No 'database is locked' errors or race conditions on table creation.
        - Exactly migrations [1, 2, 3] applied across all threads.
        - Evolved schema structure (feedback table has user_identifier, not telegram_user_id).
        - PRAGMA integrity_check and foreign_key_check pass with zero errors.
        """
        db_path = tmp_path / "fresh_concurrent.db"
        num_threads = 25
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        applied_histories: list[list[int]] = []

        def _worker():
            try:
                barrier.wait()
                applied = apply_migrations(db_path=db_path)
                applied_histories.append(applied)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent apply_migrations on fresh DB: {errors}"
        # Exactly one thread should have performed the migration [1, 2, 3], others should return []
        total_applied_migrations = sum(applied_histories, [])
        assert sorted(total_applied_migrations) == [1, 2, 3]

        # Verify DB schema on disk
        conn = create_connection(db_path, timeout=5.0)
        try:
            assert get_applied_versions(conn) == {1, 2, 3}
            cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback);").fetchall()}
            assert "user_identifier" in cols
            assert "telegram_user_id" not in cols

            is_healthy, diag_errors = verify_schema_integrity(conn)
            assert is_healthy is True
            assert diag_errors == []
        finally:
            conn.close()

    def test_partially_migrated_db_25_threads_concurrent_migration_002(self, tmp_path: Path):
        """Database with Migration 1 applied and legacy data.
        25 threads simultaneously call apply_migrations to execute Migration 002.

        Verifies:
        - Safe upgrade under concurrency without data loss or corruption.
        - Existing legacy records are preserved and mapped to user_identifier.
        """
        db_path = tmp_path / "partially_migrated.db"

        # Apply only Migration 1
        conn = create_connection(db_path, timeout=5.0, init_wal=True)
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT);")
        conn.executescript(MIGRATIONS[0].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (1, '001_initial_schema');")

        # Insert raw item and breakthrough
        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (10, 'gh', '10', 'Test', 'https://gh.com');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (10, 10, 'BT10', 8.5, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        # Insert 3 legacy feedbacks (one telegram with int id, one web with NULL id)
        conn.execute("INSERT INTO feedback (id, breakthrough_id, rating, telegram_user_id, source, notes) VALUES (1, 10, 'hit', 777888, 'telegram', 'Nice');")
        conn.execute("INSERT INTO feedback (id, breakthrough_id, rating, telegram_user_id, source, notes) VALUES (2, 10, 'inspire', NULL, 'web', 'Web vote');")
        conn.close()

        # Concurrently launch 25 threads attempting to apply migration 2
        num_threads = 25
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        applied_histories: list[list[int]] = []

        def _worker():
            try:
                barrier.wait()
                applied = apply_migrations(db_path=db_path)
                applied_histories.append(applied)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent upgrade to Migration 2: {errors}"
        total_applied = sum(applied_histories, [])
        assert total_applied == [2, 3]

        # Verify data on disk
        verify_conn = create_connection(db_path, timeout=5.0)
        try:
            assert get_applied_versions(verify_conn) == {1, 2, 3}
            rows = verify_conn.execute("SELECT id, breakthrough_id, rating, user_identifier, source, notes FROM feedback ORDER BY id ASC;").fetchall()
            assert len(rows) == 2
            assert tuple(rows[0]) == (1, 10, "hit", "777888", "telegram", "Nice")
            assert tuple(rows[1]) == (2, 10, "inspire", "default", "web", "Web vote")

            is_healthy, diag = verify_schema_integrity(verify_conn)
            assert is_healthy is True
            assert diag == []
        finally:
            verify_conn.close()

    def test_migration_002_handles_duplicate_legacy_null_user_ids(self, sync_db_conn: sqlite3.Connection):
        """In Migration 001, multiple rows could have telegram_user_id = NULL because SQLite UNIQUE allows multiple NULLs.
        Migration 002 converts NULL to 'default' and enforces UNIQUE(breakthrough_id, user_identifier).

        Verifies that INSERT OR REPLACE in MIGRATION_002_UP deduplicates without raising UNIQUE constraint failure.
        """
        conn = sync_db_conn
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT);")
        conn.executescript(MIGRATIONS[0].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (1, '001_initial_schema');")

        # Seed 1 breakthrough
        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 't', '1', 'T', 'U');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        # Multiple feedbacks with telegram_user_id = NULL on the same breakthrough
        conn.execute("INSERT INTO feedback (id, breakthrough_id, rating, telegram_user_id, source, notes) VALUES (1, 1, 'hit', NULL, 'web', 'First web note');")
        conn.execute("INSERT INTO feedback (id, breakthrough_id, rating, telegram_user_id, source, notes) VALUES (2, 1, 'inspire', NULL, 'web', 'Second web note');")

        # Apply migration 2
        conn.executescript(MIGRATIONS[1].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (2, '002_multi_messenger_feedback');")

        # Verify that deduplication succeeded: exactly 1 row with user_identifier='default'
        rows = conn.execute("SELECT id, breakthrough_id, rating, user_identifier, notes FROM feedback;").fetchall()
        assert len(rows) == 1
        # The later row (id 2) replaced id 1
        assert rows[0][1] == 1
        assert rows[0][2] == "inspire"
        assert rows[0][3] == "default"
        assert rows[0][4] == "Second web note"

        is_healthy, diag = verify_schema_integrity(conn)
        assert is_healthy is True
        assert diag == []

    def test_migration_002_full_cycle_rollback_and_reapply_clean_numeric(self, sync_db_conn: sqlite3.Connection):
        """Executes UP -> SEED (numeric + web) -> DOWN -> CHECK -> UP -> CHECK full lifecycle with clean inputs."""
        conn = sync_db_conn
        apply_migrations(conn=conn)

        # Seed data in v2: numeric Telegram ID, +phone Signal, and web
        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 't', '1', 'T', 'U');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'hit', '12345', 'telegram');")
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'inspire', '+491701234567', 'signal');")
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'known', 'default', 'web');")

        # 1. Rollback
        conn.executescript(MIGRATIONS[1].down_sql)
        conn.execute("DELETE FROM schema_migrations WHERE version = 2;")
        assert get_applied_versions(conn) == {1, 3}

        v1_rows = conn.execute("SELECT breakthrough_id, rating, telegram_user_id, source FROM feedback ORDER BY id ASC;").fetchall()
        assert len(v1_rows) == 3
        assert v1_rows[0][2] == 12345
        assert v1_rows[1][2] is None  # Signal phone with + -> NULL in v1
        assert v1_rows[2][2] is None  # 'default' -> NULL in v1

        is_healthy, diag = verify_schema_integrity(conn)
        assert is_healthy is True
        assert diag == []

        # 2. Re-apply Migration 2
        conn.executescript(MIGRATIONS[1].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (2, '002_multi_messenger_feedback');")
        assert get_applied_versions(conn) == {1, 2, 3}

        is_healthy2, diag2 = verify_schema_integrity(conn)
        assert is_healthy2 is True
        assert diag2 == []

    def test_empirical_bug_signal_uuid_corrupted_during_migration_002_rollback(self, sync_db_conn: sqlite3.Connection):
        """EMPIRICAL CHALLENGE BUG DEMONSTRATION:
        Exposes that MIGRATION_002_DOWN uses `user_identifier GLOB '[0-9]*'`, which matches
        any string starting with a digit. SQLite's `CAST('3b9a...' AS INTEGER)` then truncates
        the hex UUID to the single leading integer `3`, corrupting user identity and causing
        unique constraint collisions if multiple UUIDs start with the same digit.
        """
        conn = sync_db_conn
        apply_migrations(conn=conn)

        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 't', '1', 'T', 'U');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        # Seed Signal feedback with UUID starting with '3'
        signal_uuid = "3b9a0c24-4f81-428a-86a1-305f884a1e90"
        conn.execute(
            "INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'hit', ?, 'signal');",
            (signal_uuid,),
        )

        # Execute MIGRATION_002_DOWN rollback
        conn.executescript(MIGRATIONS[1].down_sql)

        # Query the rolled-back record (feedback_v1 was renamed back to feedback)
        row = conn.execute("SELECT breakthrough_id, rating, telegram_user_id, source FROM feedback;").fetchone()
        telegram_user_id = row[2]

        # Under the remediated CASE pattern:
        # source = 'telegram' AND user_identifier NOT GLOB '*[^0-9]*' AND user_identifier != '' AND user_identifier != 'default'
        # Signal UUIDs are safely preserved with telegram_user_id = NULL, and source is defaulted to 'web'.
        assert telegram_user_id is None, (
            f"Remediation verified: Signal UUID '{signal_uuid}' safely maps to telegram_user_id = NULL "
            f"instead of being corrupted to integer {telegram_user_id}."
        )
        assert row[3] == "web", f"Expected source to fallback to 'web', got '{row[3]}'"

    def test_empirical_bug_signal_uuid_collision_data_loss_under_rollback(self, sync_db_conn: sqlite3.Connection):
        """EMPIRICAL CHALLENGE BUG DEMONSTRATION 2 (REMEDIATED):
        Verifies that when two distinct Signal users with UUIDs starting with the same digit
        (e.g., '3a...' and '3b...') vote on the same breakthrough, MIGRATION_002_DOWN safely
        maps both to telegram_user_id = NULL. In SQLite, NULL != NULL in UNIQUE constraints,
        so BOTH ratings are preserved without collision or data loss.
        """
        conn = sync_db_conn
        apply_migrations(conn=conn)

        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 't', '1', 'T', 'U');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        # User 1 with Signal UUID starting with 3
        uuid_user1 = "3a111111-0000-0000-0000-000000000000"
        conn.execute(
            "INSERT INTO feedback (breakthrough_id, rating, user_identifier, source, notes) VALUES (1, 'hit', ?, 'signal', 'User 1 rating');",
            (uuid_user1,),
        )
        # User 2 with Signal UUID ALSO starting with 3
        uuid_user2 = "3b222222-0000-0000-0000-000000000000"
        conn.execute(
            "INSERT INTO feedback (breakthrough_id, rating, user_identifier, source, notes) VALUES (1, 'inspire', ?, 'signal', 'User 2 rating');",
            (uuid_user2,),
        )

        # In v2, there are 2 distinct feedback rows
        v2_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
        assert v2_count == 2

        # Execute MIGRATION_002_DOWN
        conn.executescript(MIGRATIONS[1].down_sql)

        # In rolled-back v1 table, both have telegram_user_id = NULL, so both records are preserved!
        v1_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
        assert v1_count == 2, f"Remediation verified: Expected 2 preserved records, got {v1_count}"
        ratings = {r[0] for r in conn.execute("SELECT rating FROM feedback;").fetchall()}
        assert ratings == {"hit", "inspire"}

    def test_concurrent_apply_migrations_independent_connections(self, tmp_path: Path):
        """20 threads each opening their own independent sqlite3 connection calling apply_migrations(conn=conn)."""
        db_path = tmp_path / "independent_conns.db"
        # Seed fresh file
        c = create_connection(db_path, timeout=10.0, init_wal=True)
        c.close()

        num_threads = 20
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        applied_results: list[list[int]] = []

        def _worker():
            try:
                conn = create_connection(db_path, timeout=10.0, init_wal=False)
                barrier.wait()
                res = apply_migrations(conn=conn)
                applied_results.append(res)
                conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent independent connection migrations: {errors}"
        total_applied = sum(applied_results, [])
        assert sorted(total_applied) == [1, 2, 3]


# ============================================================================
# Section 2: Adversarial Multi-Channel Concurrency Burst
# ============================================================================

class TestAdversarialMultiChannelBurst:
    """Stress-tests concurrent bursts across 25+ workers with Signal phone numbers and UUIDs, and Telegram IDs."""

    @pytest.mark.asyncio
    async def test_concurrent_burst_25_workers_diverse_channels_and_identifiers(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """25 workers simultaneously recording feedback with varied channel types and identifiers:
        - Signal: international phone numbers with spaces, plus signs, dashes
        - Signal: UUIDs with hyphens, uppercase, braces
        - Telegram: numeric IDs, negative chat IDs
        - Web: alphanumeric usernames and session keys
        - Edge: non-ascii names, emojis in username, padded whitespace

        Verifies:
        - All operations succeed without exceptions.
        - SQLite feedback table contains exactly the recorded unique records.
        - preferences.json is valid and not corrupted.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Seed 5 breakthroughs
        bt_ids: list[int] = []
        for i in range(5):
            raw = factory.create_raw_item(source_id=f"burst_item_{i}")
            raw_id = await migrated_db.execute(
                "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
                (raw.source, raw.source_id, raw.title, raw.url),
            )
            bt = factory.create_breakthrough(raw_item_id=raw_id, title=f"Burst BT {i}")
            bt_id = await migrated_db.execute("""
                INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                          tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
                  bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))
            bt_ids.append(bt_id)

        # Define 25 diverse user profiles
        test_profiles = [
            # Signal phone (5)
            {"user": "+49 170 1234567", "source": "signal", "rating": FeedbackRating.HIT},
            {"user": "+1-800-555-0199", "source": "signal", "rating": FeedbackRating.INSPIRE},
            {"user": "+81 90 1234 5678", "source": "signal", "rating": FeedbackRating.HIT},
            {"user": "0044 20 7946 0912", "source": "signal", "rating": FeedbackRating.HYPE},
            {"user": "+33.6.12.34.56.78", "source": "signal", "rating": FeedbackRating.KNOWN},
            # Signal (5)
            {"user": "3b9a0c24-4f81-428a-86a1-305f884a1e90", "source": "signal", "rating": FeedbackRating.HIT},
            {"user": "A1B2C3D4-E5F6-7890-ABCD-EF1234567890", "source": "signal", "rating": FeedbackRating.INSPIRE},
            {"user": "{12345678-1234-5678-1234-567812345678}", "source": "signal", "rating": FeedbackRating.KNOWN},
            {"user": "signal-uuid-0987654321", "source": "signal", "rating": FeedbackRating.HYPE},
            {"user": str(uuid.uuid4()), "source": "signal", "rating": FeedbackRating.INSPIRE},
            # Telegram (5)
            {"user": "123456789", "source": "telegram", "rating": FeedbackRating.HIT, "tg_id": 123456789},
            {"user": "987654321", "source": "telegram", "rating": FeedbackRating.HYPE, "tg_id": 987654321},
            {"user": "555666777", "source": "telegram", "rating": FeedbackRating.KNOWN, "tg_id": 555666777},
            {"user": "-1001234567890", "source": "telegram", "rating": FeedbackRating.INSPIRE},
            {"user": "9223372036854775806", "source": "telegram", "rating": FeedbackRating.HIT},
            # Web (5)
            {"user": "web", "source": "web", "rating": FeedbackRating.HIT},
            {"user": "web:sess:abc123xyz", "source": "web", "rating": FeedbackRating.INSPIRE},
            {"user": "dashboard_admin", "source": "web", "rating": FeedbackRating.KNOWN},
            {"user": "anonymous_web_user", "source": "web", "rating": FeedbackRating.HYPE},
            {"user": "web_guest_42", "source": "web", "rating": FeedbackRating.HIT},
            # Edge / Unicode / Padded (5)
            {"user": "  padded_user_name  ", "source": "telegram", "rating": FeedbackRating.HIT},
            {"user": "münchen_ü_user", "source": "web", "rating": FeedbackRating.INSPIRE},
            {"user": "user_with_emoji_🎯", "source": "signal", "rating": FeedbackRating.HIT},
            {"user": "user.name+tag@sub.domain", "source": "signal", "rating": FeedbackRating.KNOWN},
            {"user": "default", "source": "web", "rating": FeedbackRating.HYPE},
        ]

        errors: list[Exception] = []
        feedback_results: list[Feedback] = []

        async def _worker(idx: int, profile: dict[str, Any]):
            try:
                # Target breakthrough alternates across the 5 breakthroughs
                target_bt = bt_ids[idx % len(bt_ids)]
                tg_id = profile.get("tg_id")
                fb = await service.record_feedback(
                    breakthrough_id=target_bt,
                    rating=profile["rating"],
                    user_identifier=profile["user"],
                    source=profile["source"],
                    notes=f"Burst worker note {idx} from {profile['source']}",
                    telegram_user_id=tg_id,
                )
                feedback_results.append(fb)
            except Exception as exc:
                errors.append(exc)

        tasks = [_worker(i, test_profiles[i]) for i in range(len(test_profiles))]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Encountered errors during 25-worker burst: {errors}"
        assert len(feedback_results) == 25

        # Verify SQLite DB
        db_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        # Check unique (breakthrough_id, user_identifier)
        expected_unique = len({(idx % len(bt_ids), profile["user"].strip()) for idx, profile in enumerate(test_profiles)})
        assert db_count == expected_unique

        # Verify preferences.json integrity
        pref_file = Path(test_settings.preferences_file)
        assert pref_file.exists()
        parsed = json.loads(pref_file.read_text(encoding="utf-8"))
        assert parsed["stats"]["total_ratings"] == db_count
        assert len(parsed["feedback_history"]) == db_count


# ============================================================================
# Section 3: Adversarial Unique Constraint Collisions & Races
# ============================================================================

class TestAdversarialConstraintCollisions:
    """Stress-tests atomic UPSERT under simultaneous racing updates to the SAME record."""

    @pytest.mark.asyncio
    async def test_same_user_same_breakthrough_simultaneous_racing_upserts(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """20 workers simultaneously updating feedback for the EXACT SAME breakthrough and user.
        
        Validates:
        - Atomic UPSERT handles simultaneous concurrency without 'UNIQUE constraint failed'.
        - Exactly 1 row in the feedback table for that (breakthrough_id, user_identifier).
        - Final state matches one of the valid ratings.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="collision_item")
        raw_id = await migrated_db.execute(
            "INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);",
            (raw.source, raw.source_id, raw.title, raw.url),
        )
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value,
              bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        target_user = "+4917099988877"
        num_racers = 20
        ratings = [FeedbackRating.HIT, FeedbackRating.HYPE, FeedbackRating.INSPIRE, FeedbackRating.KNOWN]
        errors: list[Exception] = []

        async def _racer(idx: int):
            try:
                selected_rating = ratings[idx % len(ratings)]
                await service.record_feedback(
                    breakthrough_id=bt_id,
                    rating=selected_rating,
                    user_identifier=target_user,
                    source="signal",
                    notes=f"Race attempt #{idx}",
                )
            except Exception as exc:
                errors.append(exc)

        tasks = [_racer(i) for i in range(num_racers)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Encountered errors during simultaneous racing UPSERT: {errors}"

        # Must have exactly ONE row in SQLite
        rows = await migrated_db.fetch_all(
            "SELECT * FROM feedback WHERE breakthrough_id = ? AND user_identifier = ?;",
            (bt_id, target_user),
        )
        assert len(rows) == 1
        assert rows[0]["user_identifier"] == target_user
        assert rows[0]["rating"] in [r.value for r in ratings]

    @pytest.mark.asyncio
    async def test_legacy_and_new_identifier_upsert_collision(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Worker 1 calls with legacy `telegram_user_id=12345`.
        Worker 2 calls with `user_identifier="12345"`.

        Verifies both resolve to the same user_identifier and update the same record.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="legacy_new_item")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        # 1. Legacy call
        fb1 = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.HIT,
            telegram_user_id=12345,
            source="telegram",
            notes="Legacy call",
        )
        assert fb1.user_identifier == "12345"
        assert fb1.telegram_user_id == 12345

        # 2. Modern call with string
        fb2 = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            user_identifier="12345",
            source="telegram",
            notes="Modern call",
        )
        assert fb2.user_identifier == "12345"
        assert fb2.rating == FeedbackRating.INSPIRE

        # Exactly 1 row must exist in DB
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert count == 1

    @pytest.mark.asyncio
    async def test_whitespace_and_none_identifier_resolution(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """User identifiers like None, '', and '   ' must all normalize to 'default'."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="whitespace_item")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        # 1. None identifier
        fb1 = await service.record_feedback(breakthrough_id=bt_id, rating=FeedbackRating.HIT, user_identifier=None)
        assert fb1.user_identifier == "default"

        # 2. Empty string
        fb2 = await service.record_feedback(breakthrough_id=bt_id, rating=FeedbackRating.HYPE, user_identifier="")
        assert fb2.user_identifier == "default"

        # 3. Whitespace only
        fb3 = await service.record_feedback(breakthrough_id=bt_id, rating=FeedbackRating.INSPIRE, user_identifier="   \t\n  ")
        assert fb3.user_identifier == "default"

        # Count must remain 1
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert count == 1


# ============================================================================
# Section 4: Adversarial SQL Injection Attacks
# ============================================================================

class TestAdversarialSQLInjectionAttempts:
    """Attempts SQL injections across all input vectors: user_identifier, notes, source, rating."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malicious_user_id",
        [
            "' OR '1'='1",
            "'; DROP TABLE feedback; --",
            "'; DROP TABLE breakthroughs; --",
            "1; ATTACH DATABASE 'pwn.db' AS pwn; --",
            "admin' UNION SELECT 1, 1, 'hit', 'admin', 'telegram', 'pwn', datetime('now') --",
            "Robert'); DROP TABLE Students;--",
            "\\' OR 1=1 --",
            "' UNION ALL SELECT null, null, null, null, null, null, null--",
        ],
    )
    async def test_sql_injection_in_user_identifier(
        self, migrated_db: Database, test_settings: Settings, factory: Any, malicious_user_id: str
    ):
        """Attempts SQL injection payloads in user_identifier."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id=f"sqli_raw_{uuid.uuid4().hex[:8]}")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        # Attempt recording with injection payload
        fb = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.HIT,
            user_identifier=malicious_user_id,
            source="signal",
            notes="SQLi attempt in user_identifier",
        )

        # Injection payload must be safely stored as literal text
        assert fb.user_identifier == malicious_user_id.strip()

        # Database tables must remain intact
        fb_table_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback;")
        assert fb_table_count >= 1
        bt_table_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM breakthroughs;")
        assert bt_table_count >= 1

        # Verify schema integrity
        with migrated_db.get_sync_connection() as conn:
            is_healthy, errors = verify_schema_integrity(conn)
            assert is_healthy is True
            assert errors == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malicious_notes",
        [
            "'); DROP TABLE feedback; --",
            "' OR 1=1 --",
            "/* comment */'; DELETE FROM breakthroughs; /*",
            "'; UPDATE feedback SET rating = 'hit'; --",
            "\x00'; DROP TABLE feedback; --",
        ],
    )
    async def test_sql_injection_in_notes(
        self, migrated_db: Database, test_settings: Settings, factory: Any, malicious_notes: str
    ):
        """Attempts SQL injection payloads in the notes commentary field."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id=f"sqli_notes_{uuid.uuid4().hex[:8]}")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        fb = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            user_identifier="attacker",
            source="telegram",
            notes=malicious_notes,
        )

        assert fb.notes == malicious_notes

        with migrated_db.get_sync_connection() as conn:
            is_healthy, errors = verify_schema_integrity(conn)
            assert is_healthy is True
            assert errors == []

    @pytest.mark.asyncio
    async def test_injected_source_fails_db_check_constraint(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Attempting to inject SQL into source violates CHECK constraint and fails safely."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="sqli_src")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            await service.record_feedback(
                breakthrough_id=bt_id,
                rating=FeedbackRating.HIT,
                user_identifier="attacker",
                source="telegram' OR 1=1; --",
            )

    @pytest.mark.asyncio
    async def test_foreign_key_violation_for_nonexistent_breakthrough(
        self, migrated_db: Database, test_settings: Settings
    ):
        """Inserting feedback for an uncreated breakthrough_id fails with IntegrityError (foreign key)."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            await service.record_feedback(
                breakthrough_id=99999999,
                rating=FeedbackRating.HIT,
                user_identifier="some_user",
                source="signal",
            )

    @pytest.mark.asyncio
    async def test_concurrent_racing_sql_injection_burst(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """20 workers simultaneously hammering record_feedback with adversarial SQL injection strings.
        
        Validates:
        - SQLite parameter substitution prevents SQL injection under concurrent load.
        - Database schema remains intact with zero tables dropped or injected.
        - All injection strings are safely stored as literal values.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="sqli_burst_raw")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        malicious_payloads = [
            "'; DROP TABLE feedback; --",
            "'; DROP TABLE breakthroughs; --",
            "' OR '1'='1",
            "1; ATTACH DATABASE 'evil.db' AS evil; --",
            "Robert'); DROP TABLE Students;--",
            "admin' UNION SELECT 1, 1, 'hit', 'admin', 'telegram', 'pwn', datetime('now') --",
            "\\' OR 1=1 --",
            "/* comment */'; DELETE FROM breakthroughs; /*",
            "'; UPDATE feedback SET rating = 'hit'; --",
            "\x00'; DROP TABLE feedback; --",
        ] * 2  # 20 workers

        errors: list[Exception] = []

        async def _worker(idx: int, payload: str):
            try:
                await service.record_feedback(
                    breakthrough_id=bt_id,
                    rating=FeedbackRating.HIT if idx % 2 == 0 else FeedbackRating.INSPIRE,
                    user_identifier=f"attacker_{idx}_{payload[:10]}",
                    source="telegram",
                    notes=f"Note {payload}",
                )
            except Exception as e:
                errors.append(e)

        tasks = [_worker(i, malicious_payloads[i]) for i in range(len(malicious_payloads))]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Errors in concurrent SQL injection burst: {errors}"

        # Feedback and Breakthrough tables must remain completely intact
        fb_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert fb_count == len(malicious_payloads)
        bt_count = await migrated_db.fetch_val("SELECT COUNT(*) FROM breakthroughs;")
        assert bt_count >= 1

        with migrated_db.get_sync_connection() as conn:
            is_healthy, diag = verify_schema_integrity(conn)
            assert is_healthy is True
            assert diag == []


# ============================================================================
# Section 5: Adversarial Boundary & Stress Cases
# ============================================================================

class TestAdversarialBoundaryAndStressCases:
    """Stress-tests large payloads, rapid single-user state oscillation, and multi-channel exemplar calibration."""

    @pytest.mark.asyncio
    async def test_large_notes_payload_and_special_characters(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Records feedback with a 100KB notes payload containing emojis, markdown, and unicode."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="large_payload_raw")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        large_text = ("🚀 NewsScout breakthrough note with unicode: äöüß—€ " * 2000)[:100_000]

        fb = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            user_identifier="+4917011122233",
            source="signal",
            notes=large_text,
        )

        assert len(fb.notes) == 100_000

        # Verify preferences.json snapshot parses large payload cleanly
        loaded = await service.load_preferences_from_json()
        assert len(loaded["feedback_history"]) >= 1
        stored_notes = loaded["feedback_history"][0]["notes"]
        assert len(stored_notes) == 100_000

    @pytest.mark.asyncio
    async def test_rapid_alternating_ratings_single_user(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Single user flips rating 40 times in rapid sequence."""
        service = PreferencesService(db=migrated_db, settings=test_settings)

        raw = factory.create_raw_item(source_id="flip_flop_raw")
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        sequence = [FeedbackRating.HIT, FeedbackRating.HYPE, FeedbackRating.KNOWN, FeedbackRating.INSPIRE] * 10
        for idx, rating in enumerate(sequence):
            fb = await service.record_feedback(
                breakthrough_id=bt_id,
                rating=rating,
                user_identifier="flip_user",
                source="signal",
                notes=f"Flip #{idx}",
            )
            assert fb.rating == rating

        # Database must still have only 1 row
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert count == 1

        # Final rating must be the last one (INSPIRE)
        row = await migrated_db.fetch_one("SELECT rating FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert row["rating"] == FeedbackRating.INSPIRE.value

    @pytest.mark.asyncio
    async def test_exemplar_calibration_with_multichannel_feedback(
        self, migrated_db: Database, test_settings: Settings, factory: Any
    ):
        """Verifies that get_few_shot_exemplars and get_preference_stats correctly incorporate
        feedback from Signal, Web, and Telegram.
        """
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Seed 4 breakthroughs
        bt_ids = []
        for i in range(4):
            raw = factory.create_raw_item(source_id=f"calib_raw_{i}")
            raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
            bt = factory.create_breakthrough(raw_item_id=raw_id, title=f"Calib Tool {i}")
            bt_id = await migrated_db.execute("""
                INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))
            bt_ids.append(bt_id)

        # 1. Signal HIT
        await service.record_feedback(breakthrough_id=bt_ids[0], rating=FeedbackRating.HIT, user_identifier="+49170123456", source="signal")
        # 2. Signal INSPIRE
        await service.record_feedback(breakthrough_id=bt_ids[1], rating=FeedbackRating.INSPIRE, user_identifier=str(uuid.uuid4()), source="signal")
        # 3. Web HYPE
        await service.record_feedback(breakthrough_id=bt_ids[2], rating=FeedbackRating.HYPE, user_identifier="web", source="web")
        # 4. Telegram KNOWN
        await service.record_feedback(breakthrough_id=bt_ids[3], rating=FeedbackRating.KNOWN, telegram_user_id=123456, source="telegram")

        # Check stats
        stats = await service.get_preference_stats()
        assert stats["total_ratings"] == 4
        assert stats["hit_count"] == 1
        assert stats["inspire_count"] == 1
        assert stats["hype_count"] == 1
        assert stats["known_count"] == 1
        assert stats["positive_count"] == 2
        assert stats["hit_rate_pct"] == 50.0

        # Check exemplars
        exemplars = await service.get_few_shot_exemplars(limit_positive=5, limit_negative=5)
        pos_titles = [e.title for e in exemplars.positive]
        neg_titles = [e.title for e in exemplars.negative]

        # The 2 positive ones must be from Signal (bt 0 and bt 1)
        assert "Calib Tool 0" in pos_titles
        assert "Calib Tool 1" in pos_titles
        # The negative one must be from Web (bt 2)
        assert "Calib Tool 2" in neg_titles
