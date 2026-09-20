"""Database Schema Migrations Runner for NewsScout.

Provides transactional, versioned database schema upgrades, 8-table DDL,
performance indexes, foreign key cascading, and integrity verification.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import sqlite3
import threading
from typing import TYPE_CHECKING, Final

from newsscout.storage.db import create_connection, sync_transaction

if TYPE_CHECKING:
    from newsscout.storage.db import Database

logger = logging.getLogger(__name__)

_MIGRATION_LOCK = threading.Lock()


class MigrationError(Exception):
    """Raised when a schema migration fails to execute."""


@dataclass(frozen=True)
class Migration:
    """Represents an atomic schema migration unit."""

    version: int
    name: str
    up_sql: str
    down_sql: str = ""


# SQL DDL for Migration 001: Core 8-Table Schema & Indexes
MIGRATION_001_UP: Final[str] = """
-- 1. Ingestion Sources Registry
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    last_poll_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 2. Raw Ingested Items
CREATE TABLE IF NOT EXISTS raw_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    raw_content TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, source_id)
);
CREATE INDEX IF NOT EXISTS idx_raw_items_source ON raw_items(source);
CREATE INDEX IF NOT EXISTS idx_raw_items_ingested ON raw_items(ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_items_url ON raw_items(url);

-- 3. Stage 1 Heuristic Evaluations
CREATE TABLE IF NOT EXISTS stage1_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    passed INTEGER NOT NULL,
    drop_reason TEXT,
    detected_license TEXT,
    has_docker INTEGER NOT NULL DEFAULT 0,
    has_runnable_code INTEGER NOT NULL DEFAULT 0,
    heuristics_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(raw_item_id)
);
CREATE INDEX IF NOT EXISTS idx_stage1_passed ON stage1_evaluations(passed);
CREATE INDEX IF NOT EXISTS idx_stage1_evaluated ON stage1_evaluations(evaluated_at DESC);

-- 4. Stage 2 Evaluated Breakthroughs & Decision Cards
CREATE TABLE IF NOT EXISTS breakthroughs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_item_id INTEGER NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    repo_url TEXT,
    breakthrough_score REAL NOT NULL CHECK(breakthrough_score >= 1.0 AND breakthrough_score <= 10.0),
    roi_score REAL NOT NULL CHECK(roi_score >= 1.0 AND roi_score <= 10.0),
    category TEXT NOT NULL CHECK(category IN ('core', 'serendipity', 'discard')),
    tldr TEXT NOT NULL,
    use_case TEXT NOT NULL,
    comparison TEXT NOT NULL,
    quickstart TEXT NOT NULL,
    hardware_requirements TEXT NOT NULL,
    license TEXT NOT NULL,
    card_markdown TEXT NOT NULL,
    evaluation_raw_json TEXT NOT NULL DEFAULT '{}',
    is_watchlisted INTEGER NOT NULL DEFAULT 0,
    evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(raw_item_id)
);
CREATE INDEX IF NOT EXISTS idx_breakthroughs_cat ON breakthroughs(category);
CREATE INDEX IF NOT EXISTS idx_breakthroughs_scores ON breakthroughs(breakthrough_score DESC, roi_score DESC);
CREATE INDEX IF NOT EXISTS idx_breakthroughs_watchlisted ON breakthroughs(is_watchlisted);
CREATE INDEX IF NOT EXISTS idx_breakthroughs_evaluated ON breakthroughs(evaluated_at DESC);

-- 5. Twice-Daily Audio Digests
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot TEXT NOT NULL CHECK(slot IN ('morning', 'afternoon')),
    digest_date TEXT NOT NULL,
    title TEXT NOT NULL,
    total_duration_seconds INTEGER NOT NULL DEFAULT 0,
    merged_audio_path TEXT,
    telegram_message_id INTEGER,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'generating', 'synthesized', 'delivered', 'failed')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    delivered_at TEXT,
    UNIQUE(digest_date, slot)
);
CREATE INDEX IF NOT EXISTS idx_digests_date_slot ON digests(digest_date, slot);
CREATE INDEX IF NOT EXISTS idx_digests_status ON digests(status);

-- 6. Individual Chapter Tracks per Digest
CREATE TABLE IF NOT EXISTS digest_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_id INTEGER NOT NULL REFERENCES digests(id) ON DELETE CASCADE,
    track_number INTEGER NOT NULL CHECK(track_number BETWEEN 1 AND 5),
    track_type TEXT NOT NULL CHECK(track_type IN ('executive_summary', 'deep_dive_1', 'deep_dive_2', 'serendipity', 'verdict')),
    title TEXT NOT NULL,
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    audio_path TEXT NOT NULL,
    script_json TEXT NOT NULL,
    breakthrough_id INTEGER REFERENCES breakthroughs(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(digest_id, track_number)
);
CREATE INDEX IF NOT EXISTS idx_digest_tracks_digest ON digest_tracks(digest_id);
CREATE INDEX IF NOT EXISTS idx_digest_tracks_breakthrough ON digest_tracks(breakthrough_id);

-- 7. User Feedback & Preference Events
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breakthrough_id INTEGER NOT NULL REFERENCES breakthroughs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK(rating IN ('hit', 'hype', 'known', 'inspire')),
    telegram_user_id INTEGER,
    source TEXT NOT NULL DEFAULT 'telegram' CHECK(source IN ('telegram', 'web')),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(breakthrough_id, telegram_user_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_breakthrough ON feedback(breakthrough_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at DESC);

-- 8. Persistent User Preferences & Dynamic Few-Shot Store
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pref_key TEXT NOT NULL UNIQUE,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(pref_key);

-- Seed Initial Monitored Sources
INSERT OR IGNORE INTO sources (name, source_type, config_json, is_active) VALUES
('github_releases', 'api', '{"repos": ["vllm-project/vllm", "ggerganov/llama.cpp", "ollama/ollama", "sgl-project/sglang", "qdrant/qdrant", "DS4SD/docling"]}', 1),
('hacker_news', 'firebase', '{"min_score": 120, "min_ratio": 0.4}', 1),
('hf_daily_papers', 'api', '{"min_upvotes": 5}', 1);
"""

# SQL DDL for Migration 002: Multi-Messenger Feedback Table Evolution
MIGRATION_002_UP: Final[str] = """
-- 1. Create evolved feedback table with user_identifier and expanded source check
CREATE TABLE feedback_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breakthrough_id INTEGER NOT NULL REFERENCES breakthroughs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK(rating IN ('hit', 'hype', 'known', 'inspire')),
    user_identifier TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL DEFAULT 'telegram' CHECK(source IN ('telegram', 'web', 'whatsapp', 'signal')),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(breakthrough_id, user_identifier)
);

-- 2. Migrate existing feedback records with deduplication on (breakthrough_id, user_identifier)
INSERT OR REPLACE INTO feedback_v2 (id, breakthrough_id, rating, user_identifier, source, notes, created_at)
SELECT 
    id, 
    breakthrough_id, 
    rating, 
    COALESCE(CAST(telegram_user_id AS TEXT), 'default') AS user_identifier, 
    source, 
    notes, 
    created_at
FROM feedback
ORDER BY id ASC;

-- 3. Replace old table
DROP TABLE feedback;
ALTER TABLE feedback_v2 RENAME TO feedback;

-- 4. Recreate performance indexes
CREATE INDEX idx_feedback_breakthrough ON feedback(breakthrough_id);
CREATE INDEX idx_feedback_user ON feedback(user_identifier);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_created ON feedback(created_at DESC);
"""

MIGRATION_002_DOWN: Final[str] = """
-- 1. Rollback table to feedback_v1 with telegram_user_id
CREATE TABLE feedback_v1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breakthrough_id INTEGER NOT NULL REFERENCES breakthroughs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK(rating IN ('hit', 'hype', 'known', 'inspire')),
    telegram_user_id INTEGER,
    source TEXT NOT NULL DEFAULT 'telegram' CHECK(source IN ('telegram', 'web')),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(breakthrough_id, telegram_user_id)
);

-- 2. Copy records back, coercing numeric identifiers to integers and defaulting non-numeric to NULL
INSERT OR REPLACE INTO feedback_v1 (id, breakthrough_id, rating, telegram_user_id, source, notes, created_at)
SELECT 
    id, 
    breakthrough_id, 
    rating, 
    CASE 
        WHEN source = 'telegram' 
             AND user_identifier NOT GLOB '*[^0-9]*' 
             AND user_identifier != '' 
             AND user_identifier != 'default' 
        THEN CAST(user_identifier AS INTEGER)
        ELSE NULL 
    END AS telegram_user_id, 
    CASE 
        WHEN source IN ('telegram', 'web') THEN source 
        ELSE 'web' 
    END AS source, 
    notes, 
    created_at
FROM feedback
ORDER BY id ASC;

-- 3. Replace evolved table
DROP TABLE feedback;
ALTER TABLE feedback_v1 RENAME TO feedback;

-- 4. Recreate original indexes
CREATE INDEX idx_feedback_breakthrough ON feedback(breakthrough_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
CREATE INDEX idx_feedback_created ON feedback(created_at DESC);
"""

MIGRATION_003_UP = """
-- Migration 003: Add composite index for keyset pagination in deduplicate_against_db
-- Supports ORDER BY ingested_at DESC, id DESC with WHERE ingested_at >= :cutoff
CREATE INDEX IF NOT EXISTS idx_raw_items_ingested_url ON raw_items(ingested_at DESC, url);
"""

# Registry of all migrations in sequential order
MIGRATIONS: Final[list[Migration]] = [
    Migration(
        version=1,
        name="001_initial_schema",
        up_sql=MIGRATION_001_UP,
    ),
    Migration(
        version=2,
        name="002_multi_messenger_feedback",
        up_sql=MIGRATION_002_UP,
        down_sql=MIGRATION_002_DOWN,
    ),
    Migration(
        version=3,
        name="003_dedup_keyset_pagination_index",
        up_sql=MIGRATION_003_UP,
    ),
]



def ensure_migration_table(conn: sqlite3.Connection) -> None:
    """Creates the migration history ledger if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )


def get_applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Returns the set of migration versions that have already been applied."""
    ensure_migration_table(conn)
    cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    return {row[0] for row in cursor.fetchall()}


def apply_migrations(
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[int]:
    """Applies all pending database migrations in an atomic transaction.

    Args:
        db_path: Path to the SQLite database file.
        conn: Optional existing open connection (used in tests or caller management).

    Returns:
        A list of integer version numbers applied in this run.

    Raises:
        MigrationError: If any migration statement fails.
    """
    with _MIGRATION_LOCK:
        should_close = False
        if conn is None:
            if db_path is None:
                raise ValueError("Either db_path or conn must be supplied to apply_migrations")
            conn = create_connection(db_path, timeout=10.0)
            should_close = True

        applied_now: list[int] = []
        try:
            ensure_migration_table(conn)
            applied_versions = get_applied_versions(conn)

            for migration in MIGRATIONS:
                if migration.version in get_applied_versions(conn):
                    continue

                logger.info("Applying migration %03d_%s...", migration.version, migration.name)
                try:
                    prev_applied = get_applied_versions(conn)
                    full_script = (
                        f"BEGIN IMMEDIATE;\n"
                        f"{migration.up_sql}\n"
                        f"INSERT OR IGNORE INTO schema_migrations (version, name) VALUES ({migration.version}, '{migration.name}');\n"
                        f"COMMIT;"
                    )
                    conn.executescript(full_script)
                    new_applied = get_applied_versions(conn)
                    if migration.version in new_applied and migration.version not in prev_applied:
                        applied_now.append(migration.version)
                        logger.info("Successfully applied migration %03d_%s.", migration.version, migration.name)
                except Exception as exc:
                    try:
                        conn.execute("ROLLBACK;")
                    except Exception:
                        pass
                    logger.error(
                        "Migration %03d_%s failed: %s. Rolling back.",
                        migration.version,
                        migration.name,
                        exc,
                    )
                    raise MigrationError(
                        f"Failed executing migration {migration.version} ({migration.name}): {exc}"
                    ) from exc

            return applied_now
        finally:
            if should_close:
                conn.close()


async def async_apply_migrations(db: Database) -> list[int]:
    """Executes schema migrations asynchronously on the provided Database instance."""
    def _run() -> list[int]:
        with db.get_sync_connection() as conn:
            return apply_migrations(conn=conn)

    return await asyncio.to_thread(_run)


def verify_schema_integrity(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
    """Runs PRAGMA integrity_check/quick_check and foreign_key_check.

    Returns:
        A tuple of (is_healthy, list_of_error_messages).
    """
    errors: list[str] = []

    # 1. Quick integrity check
    integrity_rows = conn.execute("PRAGMA quick_check;").fetchall()
    for row in integrity_rows:
        val = row[0]
        if str(val).lower() != "ok":
            errors.append(f"Integrity check error: {val}")

    # 2. Foreign key check
    fk_rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
    for row in fk_rows:
        errors.append(
            f"Foreign key violation: table={row[0]}, rowid={row[1]}, parent={row[2]}, fkid={row[3]}"
        )

    return len(errors) == 0, errors
