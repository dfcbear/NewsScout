"""NewsScout Storage Subsystem.

Provides SQLite WAL connection management, relational schema migrations,
Pydantic v2 domain models, and dynamic preference synchronization.
"""

from newsscout.storage.db import (
    Database,
    apply_pragmas,
    create_connection,
    get_connection,
    sync_transaction,
)
from newsscout.storage.migrations import (
    Migration,
    MigrationError,
    apply_migrations,
    async_apply_migrations,
    get_applied_versions,
    verify_schema_integrity,
)
from newsscout.storage.models import (
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    Digest,
    DigestSlot,
    DigestStatus,
    Feedback,
    FeedbackRating,
    RawItem,
    Source,
    SourceType,
    Stage1Evaluation,
    Stage2Category,
    Track,
    TrackType,
    UserPreference,
)
from newsscout.storage.preferences import PreferencesService

__all__ = [
    "Database",
    "apply_pragmas",
    "create_connection",
    "get_connection",
    "sync_transaction",
    "Migration",
    "MigrationError",
    "apply_migrations",
    "async_apply_migrations",
    "get_applied_versions",
    "verify_schema_integrity",
    "SourceType",
    "Stage2Category",
    "DigestSlot",
    "DigestStatus",
    "TrackType",
    "FeedbackRating",
    "DecisionCardData",
    "Source",
    "RawItem",
    "Stage1Evaluation",
    "Breakthrough",
    "DecisionCard",
    "Track",
    "Digest",
    "Feedback",
    "UserPreference",
    "PreferencesService",
]
