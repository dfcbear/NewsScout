"""tests/test_storage_v2.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated test suite for Migration 002 and Multi-Messenger Storage Evolution:
- Schema migration from v1 to v2 preserving existing records
- Rollback atomicity and idempotency
- Multi-channel source constraints ('telegram', 'web', 'whatsapp', 'signal')
- Multi-identifier UPSERT semantics (phone numbers, UUIDs, web users)
- Pydantic models (FeedbackCreate, Feedback, FeedbackRating) validation
"""

from __future__ import annotations

import sqlite3
from typing import Any
import pytest

from newsscout.storage.migrations import (
    MIGRATIONS,
    Migration,
    apply_migrations,
    get_applied_versions,
    verify_schema_integrity,
)
from newsscout.storage.models import Feedback, FeedbackCreate, FeedbackRating, FeedbackSource
from newsscout.storage.preferences import PreferencesService


class TestMigration002Evolution:
    """Verifies schema transformation from Migration 001 to 002."""

    def test_migration_002_applied_in_sequence(self, sync_db_conn: sqlite3.Connection):
        applied = apply_migrations(conn=sync_db_conn)
        assert applied == [1, 2]
        assert get_applied_versions(sync_db_conn) == {1, 2}

        # Check table columns
        cols = {r[1] for r in sync_db_conn.execute("PRAGMA table_info(feedback);").fetchall()}
        assert "user_identifier" in cols
        assert "telegram_user_id" not in cols

        # Verify integrity
        is_healthy, errors = verify_schema_integrity(sync_db_conn)
        assert is_healthy is True
        assert errors == []

    def test_data_migration_preserves_legacy_records(self, sync_db_conn: sqlite3.Connection):
        # 1. Apply only migration 1
        conn = sync_db_conn
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT);")
        conn.executescript(MIGRATIONS[0].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (1, '001_initial_schema');")

        # 2. Seed breakthrough and v1 feedback
        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 'test', '1', 'Title', 'http://url');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, telegram_user_id, source, notes) VALUES (1, 'hit', 12345678, 'telegram', 'Great!');")
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, telegram_user_id, source, notes) VALUES (1, 'inspire', NULL, 'web', 'From web');")

        # 3. Apply migration 2
        conn.executescript(MIGRATIONS[1].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (2, '002_multi_messenger_feedback');")

        # 4. Verify migrated rows
        rows = conn.execute("SELECT breakthrough_id, rating, user_identifier, source, notes FROM feedback ORDER BY id ASC;").fetchall()
        assert len(rows) == 2
        assert tuple(rows[0]) == (1, "hit", "12345678", "telegram", "Great!")
        assert tuple(rows[1]) == (1, "inspire", "default", "web", "From web")

    def test_migration_002_rollback_and_reapply(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)
        conn = sync_db_conn

        # Seed v2 record
        conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 'test', '1', 'Title', 'http://url');")
        conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)
        conn.execute("INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'hit', '987654', 'telegram');")

        # Execute rollback script
        conn.executescript(MIGRATIONS[1].down_sql)
        conn.execute("DELETE FROM schema_migrations WHERE version = 2;")

        cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback);").fetchall()}
        assert "telegram_user_id" in cols
        assert "user_identifier" not in cols

        # Re-apply migration 2
        conn.executescript(MIGRATIONS[1].up_sql)
        conn.execute("INSERT INTO schema_migrations (version, name) VALUES (2, '002_multi_messenger_feedback');")

        cols_after = {r[1] for r in conn.execute("PRAGMA table_info(feedback);").fetchall()}
        assert "user_identifier" in cols_after
        assert "telegram_user_id" not in cols_after

    def test_source_check_constraint_allows_whatsapp_and_signal(self, sync_db_conn: sqlite3.Connection):
        apply_migrations(conn=sync_db_conn)
        sync_db_conn.execute("INSERT INTO raw_items (id, source, source_id, title, url) VALUES (1, 'test', '1', 'Title', 'http://url');")
        sync_db_conn.execute("""
            INSERT INTO breakthroughs (id, raw_item_id, title, breakthrough_score, roi_score, category,
                                      tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (1, 1, 'BT1', 9.0, 9.0, 'core', 'tldr', 'uc', 'comp', 'qs', 'hw', 'lic', 'md');
        """)

        # Valid sources
        for src in ["telegram", "web", "whatsapp", "signal"]:
            sync_db_conn.execute(f"INSERT OR REPLACE INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'hit', '{src}_user', '{src}');")

        # Invalid source raises IntegrityError
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            sync_db_conn.execute("INSERT INTO feedback (breakthrough_id, rating, user_identifier, source) VALUES (1, 'hit', 'slack_user', 'slack');")


class TestMultiMessengerPreferencesService:
    """Verifies PreferencesService operations with diverse messenger identifiers."""

    @pytest.mark.asyncio
    async def test_record_feedback_multi_channel_upsert(self, migrated_db, test_settings, factory):
        service = PreferencesService(db=migrated_db, settings=test_settings)

        # Seed breakthrough
        raw = factory.create_raw_item()
        raw_id = await migrated_db.execute("INSERT INTO raw_items (source, source_id, title, url) VALUES (?, ?, ?, ?);", (raw.source, raw.source_id, raw.title, raw.url))
        bt = factory.create_breakthrough(raw_item_id=raw_id)
        bt_id = await migrated_db.execute("""
            INSERT INTO breakthroughs (raw_item_id, title, breakthrough_score, roi_score, category, tldr, use_case, comparison, quickstart, hardware_requirements, license, card_markdown)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (bt.raw_item_id, bt.title, bt.breakthrough_score, bt.roi_score, bt.category.value, bt.tldr, bt.use_case, bt.comparison, bt.quickstart, bt.hardware_requirements, bt.license, bt.card_markdown))

        # 1. WhatsApp feedback with phone number
        fb_wa = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.HIT,
            user_identifier="+491701234567",
            source="whatsapp",
            notes="Sent via WhatsApp reaction",
        )
        assert fb_wa.user_identifier == "+491701234567"
        assert fb_wa.source == "whatsapp"

        # 2. Signal feedback with UUID
        fb_sig = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            user_identifier="3b9a0c24-4f81-428a-86a1-305f884a1e90",
            source="signal",
            notes="Sent via Signal reaction",
        )
        assert fb_sig.user_identifier == "3b9a0c24-4f81-428a-86a1-305f884a1e90"
        assert fb_sig.source == "signal"

        # 3. Web feedback with default identifier
        fb_web = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.HYPE,
            user_identifier="web",
            source="web",
        )
        assert fb_web.user_identifier == "web"

        # 4. Telegram feedback via legacy kwarg
        fb_tg = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.KNOWN,
            telegram_user_id=987654,
            source="telegram",
        )
        assert fb_tg.user_identifier == "987654"
        assert fb_tg.telegram_user_id == 987654

        # Verify 4 distinct rows exist
        count = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert count == 4

        # 5. Test UPSERT: WhatsApp user updates vote to INSPIRE
        fb_wa_up = await service.record_feedback(
            breakthrough_id=bt_id,
            rating=FeedbackRating.INSPIRE,
            user_identifier="+491701234567",
            source="whatsapp",
        )
        assert fb_wa_up.rating == FeedbackRating.INSPIRE

        # Count must remain 4 (no duplicate inserted)
        count_after = await migrated_db.fetch_val("SELECT COUNT(*) FROM feedback WHERE breakthrough_id = ?;", (bt_id,))
        assert count_after == 4


class TestPydanticFeedbackModels:
    """Verifies FeedbackCreate, Feedback, and FeedbackRating helpers."""

    def test_feedback_create_normalization(self):
        # 1. Normalizes uppercase rating string
        fc = FeedbackCreate(breakthrough_id=1, rating="HIT", user_identifier="+49123")
        assert fc.rating == FeedbackRating.HIT
        assert fc.user_identifier == "+49123"
        assert fc.source == "telegram"

        # 2. Backward compatibility with telegram_user_id in dict
        data = {"breakthrough_id": 2, "rating": "hype", "telegram_user_id": 99999}
        fc2 = FeedbackCreate.model_validate(data)
        assert fc2.user_identifier == "99999"

        # 3. Invalid source raises error
        with pytest.raises(ValueError, match="Invalid source"):
            FeedbackCreate(breakthrough_id=3, rating=FeedbackRating.HIT, source="discord")

    def test_feedback_rating_methods(self):
        assert FeedbackRating.HIT.is_positive is True
        assert FeedbackRating.INSPIRE.is_positive is True
        assert FeedbackRating.HYPE.is_positive is False
        assert FeedbackRating.KNOWN.is_positive is False

        assert FeedbackRating.HIT.emoji == "🎯"
        assert FeedbackRating.INSPIRE.emoji == "🚀"
        assert FeedbackRating.HIT.label_de == "Volltreffer"

        # Parse from emoji
        assert FeedbackRating.from_emoji("🎯") == FeedbackRating.HIT
        assert FeedbackRating.from_emoji("🚀") == FeedbackRating.INSPIRE
        assert FeedbackRating.from_emoji("💤") == FeedbackRating.HYPE
        assert FeedbackRating.from_emoji("✅") == FeedbackRating.KNOWN
        assert FeedbackRating.from_emoji("❓") is None

        # Parse from keyword
        assert FeedbackRating.from_keyword("volltreffer") == FeedbackRating.HIT
        assert FeedbackRating.from_keyword("1") == FeedbackRating.HIT
        assert FeedbackRating.from_keyword("hype") == FeedbackRating.HYPE
        assert FeedbackRating.from_keyword("bekannt") == FeedbackRating.KNOWN
        assert FeedbackRating.from_keyword("genial") == FeedbackRating.INSPIRE
        assert FeedbackRating.from_keyword("unknown") is None
