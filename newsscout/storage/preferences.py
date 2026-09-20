"""newsscout.storage.preferences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Persistent user preference store, feedback synchronization service,
and dynamic few-shot exemplar generator for Stage 2 LLM calibration.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import uuid

from newsscout.config import Settings, get_settings
from newsscout.storage.models import (
    Exemplar,
    Feedback,
    FeedbackRating,
    Stage2Category,
)

logger = logging.getLogger("newsscout.storage.preferences")


# ============================================================================
# Cold-Start Exemplar Seed Profiles (Used when 0 or few ratings exist)
# ============================================================================

DEFAULT_COLD_START_POSITIVE_EXEMPLARS: list[Exemplar] = [
    Exemplar(
        breakthrough_id=0,
        title="vLLM Native NVFP4 Kernel Execution",
        rating=FeedbackRating.HIT,
        category=Stage2Category.CORE,
        tldr="Enables direct NVFP4 inference on RTX 4090 without host VRAM spill.",
        use_case="Local agent execution of 70B models at 32 tokens/sec on 24GB VRAM.",
        breakthrough_score=9.4,
        roi_score=9.2,
        repo_url="https://github.com/vllm-project/vllm",
    ),
    Exemplar(
        breakthrough_id=-1,
        title="Docling Multimodal Table & Formula Parser",
        rating=FeedbackRating.HIT,
        category=Stage2Category.CORE,
        tldr="High-speed local OCR and PDF table extraction to structured markdown.",
        use_case="Offline RAG ingestion pipeline parsing technical papers directly to Qdrant.",
        breakthrough_score=8.8,
        roi_score=9.0,
        repo_url="https://github.com/DS4SD/docling",
    ),
    Exemplar(
        breakthrough_id=-2,
        title="Unitree ROS2 Humanoid Control Loop",
        rating=FeedbackRating.INSPIRE,
        category=Stage2Category.SERENDIPITY,
        tldr="Open-source low-latency balance control loop running on edge ARM64.",
        use_case="Inspiration for high-speed deterministic control systems in robotics.",
        breakthrough_score=9.1,
        roi_score=7.5,
        repo_url="https://github.com/unitreerobotics/unitree_ros2",
    ),
]

DEFAULT_COLD_START_NEGATIVE_EXEMPLARS: list[Exemplar] = [
    Exemplar(
        breakthrough_id=-10,
        title="NextGen AI Landing Page Builder",
        rating=FeedbackRating.HYPE,
        category=Stage2Category.DISCARD,
        tldr="Thin Tailwind web UI calling OpenAI chat completions API.",
        use_case="None — pure marketing SaaS wrapper with no local code or novel architecture.",
        breakthrough_score=2.1,
        roi_score=1.5,
        repo_url="https://github.com/marketing/ai-landing",
    ),
    Exemplar(
        breakthrough_id=-11,
        title="Automated SEO Blog Post Generator",
        rating=FeedbackRating.HYPE,
        category=Stage2Category.DISCARD,
        tldr="Script spamming LLM prompts to write keyword-stuffed markdown articles.",
        use_case="Zero architectural value, unreproducible benchmarks, marketing hype.",
        breakthrough_score=1.8,
        roi_score=1.2,
        repo_url="https://github.com/spammer/seo-ai",
    ),
]


class ExemplarPair(tuple):
    """2-tuple of (positive_exemplars, negative_exemplars) with convenience properties."""

    @property
    def positive(self) -> list[Exemplar]:
        return self[0]

    @property
    def negative(self) -> list[Exemplar]:
        return self[1]

    def all(self) -> list[Exemplar]:
        return self[0] + self[1]


class PreferencesService:
    """Service managing user ratings, preference persistence in SQLite,
    atomic JSON exports, and dynamic few-shot calibration for Gemini Flash.
    """

    def __init__(self, db: Any, settings: Optional[Settings] = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self._sync_lock = asyncio.Lock()

    async def record_feedback(
        self,
        breakthrough_id: int,
        rating: FeedbackRating | str,
        user_identifier: str | int | None = None,
        source: str = "telegram",
        notes: Optional[str] = None,
        *,
        telegram_user_id: Optional[int] = None,
    ) -> Feedback:
        """Persists a rating to SQLite with atomic UPSERT semantics and triggers
        an atomic sync of data/preferences.json.

        Supports string phone numbers (WhatsApp), UUIDs (Signal), integer Telegram IDs,
        and web dashboard identifiers.
        """
        if isinstance(rating, str):
            rating = FeedbackRating(rating.lower())

        # Resolve user identifier with 100% backward-compatibility for telegram_user_id
        if user_identifier is None:
            if telegram_user_id is not None:
                resolved_user = str(telegram_user_id)
            else:
                resolved_user = "default"
        else:
            resolved_user = str(user_identifier).strip() or "default"

        query = """
        INSERT INTO feedback (breakthrough_id, rating, user_identifier, source, notes, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(breakthrough_id, user_identifier) DO UPDATE SET
            rating = excluded.rating,
            source = excluded.source,
            notes = excluded.notes,
            created_at = datetime('now');
        """
        await self.db.execute(
            query,
            (breakthrough_id, rating.value, resolved_user, source, notes),
        )

        logger.info(
            "Recorded feedback: breakthrough_id=%d rating=%s user_identifier=%s source=%s",
            breakthrough_id,
            rating.value,
            resolved_user,
            source,
        )

        # Regenerate data/preferences.json snapshot atomically
        await self.sync_preferences_file()

        # Fetch and return the updated row
        row = await self.db.fetch_one(
            """
            SELECT * FROM feedback 
            WHERE breakthrough_id = ? AND user_identifier = ?
            ORDER BY id DESC LIMIT 1;
            """,
            (breakthrough_id, resolved_user),
        )
        if row is None:
            row = await self.db.fetch_one(
                "SELECT * FROM feedback WHERE breakthrough_id = ? ORDER BY id DESC LIMIT 1;",
                (breakthrough_id,),
            )
        return Feedback.from_row(row)


    async def sync_preferences_file(self) -> Path:
        """Extracts current feedback aggregations, top exemplars, and writes
        an atomic snapshot to data/preferences.json via unique .tmp + os.replace.
        """
        async with self._sync_lock:
            target_path = Path(self.settings.preferences_file)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Fetch Aggregated Statistics
            stats = await self.get_preference_stats()

            # 2. Fetch Recent Exemplars from DB
            pos_exemplars, neg_exemplars = await self._fetch_db_exemplars(limit=10)

            # 3. Fetch Full Feedback History
            history_rows = await self.db.fetch_all(
                "SELECT * FROM feedback ORDER BY created_at DESC;"
            )
            feedback_history = [dict(r) for r in history_rows]

            # 4. Fetch Stored Topic Weights
            topic_weights_row = await self.db.fetch_one(
                "SELECT value_json FROM user_preferences WHERE pref_key = 'topic_weights';"
            )
            topic_weights: dict[str, float] = {}
            if topic_weights_row and topic_weights_row["value_json"]:
                try:
                    topic_weights = json.loads(topic_weights_row["value_json"])
                except Exception:
                    topic_weights = {}

            if not topic_weights:
                topic_weights = {
                    "agent_harnesses": 1.3,
                    "local_inference_4090": 1.4,
                    "rag_qdrant_docling": 1.2,
                    "c_cpp_embedded": 1.1,
                    "robotics_ros2": 1.1,
                    "saas_wrappers": 0.0,
                    "seo_generators": 0.0,
                }

            # 5. Construct Payload
            payload = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "stats": stats,
                "feedback_history": feedback_history,
                "exemplars": {
                    "positive": [ex.model_dump() for ex in pos_exemplars],
                    "negative": [ex.model_dump() for ex in neg_exemplars],
                },
                "topic_weights": topic_weights,
            }

            # 6. Atomic File Replacement (write unique .tmp -> os.replace)
            tmp_path = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, target_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

            # 7. Update user_preferences table in DB
            await self.db.execute(
                """
                INSERT INTO user_preferences (pref_key, value_json, updated_at)
                VALUES ('few_shot_exemplars', ?, datetime('now'))
                ON CONFLICT(pref_key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at;
                """,
                (json.dumps(payload),),
            )

            logger.debug("Successfully synced preferences to %s", target_path)
            return target_path

    async def load_preferences_from_json(self) -> dict[str, Any]:
        """Loads and parses the preferences JSON file.

        Recovers gracefully if missing or corrupted by backing up corrupted file
        and regenerating cleanly from SQLite.
        """
        target_path = Path(self.settings.preferences_file)
        if not target_path.exists():
            await self.sync_preferences_file()

        try:
            content = target_path.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception as exc:
            logger.warning("Corrupted preferences file %s (%s). Recovering...", target_path, exc)
            corrupted_path = target_path.with_suffix(".corrupted")
            try:
                os.replace(target_path, corrupted_path)
            except Exception:
                pass
            await self.sync_preferences_file()
            return json.loads(target_path.read_text(encoding="utf-8"))

    async def get_few_shot_exemplars(
        self,
        count: Optional[int] = None,
        limit_positive: int = 5,
        limit_negative: int = 5,
    ) -> ExemplarPair:
        """Returns calibrated positive and negative exemplars for Stage 2 evaluation prompts.

        Falls back to canonical cold-start exemplars if insufficient ratings exist.
        """
        if count is not None:
            limit_positive = count
            limit_negative = count

        db_pos, db_neg = await self._fetch_db_exemplars(limit=max(limit_positive, limit_negative))

        # Fill positive exemplars
        pos_list = db_pos[:limit_positive]
        if len(pos_list) < limit_positive:
            needed = limit_positive - len(pos_list)
            pos_list.extend(DEFAULT_COLD_START_POSITIVE_EXEMPLARS[:needed])

        # Fill negative exemplars
        neg_list = db_neg[:limit_negative]
        if len(neg_list) < limit_negative:
            needed = limit_negative - len(neg_list)
            neg_list.extend(DEFAULT_COLD_START_NEGATIVE_EXEMPLARS[:needed])

        return ExemplarPair((pos_list, neg_list))

    async def get_preference_stats(self) -> dict[str, Any]:
        """Calculates feedback summary statistics for the web dashboard."""
        rows = await self.db.fetch_all(
            "SELECT rating, COUNT(*) as cnt FROM feedback GROUP BY rating;"
        )
        counts = {r["rating"]: r["cnt"] for r in rows}
        total = sum(counts.values())

        hits = counts.get(FeedbackRating.HIT.value, 0)
        inspires = counts.get(FeedbackRating.INSPIRE.value, 0)
        hypes = counts.get(FeedbackRating.HYPE.value, 0)
        knowns = counts.get(FeedbackRating.KNOWN.value, 0)

        positive_count = hits + inspires
        hit_rate = round((positive_count / total * 100), 1) if total > 0 else 0.0

        return {
            "total_ratings": total,
            "hit_count": hits,
            "inspire_count": inspires,
            "hype_count": hypes,
            "known_count": knowns,
            "positive_count": positive_count,
            "hit_rate_pct": hit_rate,
        }

    async def _fetch_db_exemplars(
        self, limit: int = 10
    ) -> tuple[list[Exemplar], list[Exemplar]]:
        """Queries database for rated breakthroughs joined with feedback."""
        query = """
        SELECT 
            b.id as breakthrough_id,
            b.title,
            f.rating,
            b.category,
            b.tldr,
            b.use_case,
            b.breakthrough_score,
            b.roi_score,
            b.repo_url
        FROM feedback f
        JOIN breakthroughs b ON f.breakthrough_id = b.id
        ORDER BY f.created_at DESC;
        """
        rows = await self.db.fetch_all(query)
        pos: list[Exemplar] = []
        neg: list[Exemplar] = []
        seen_pos: set[int] = set()
        seen_neg: set[int] = set()

        for r in rows:
            ex = Exemplar(
                breakthrough_id=r["breakthrough_id"],
                title=r["title"],
                rating=FeedbackRating(r["rating"]),
                category=Stage2Category(r["category"]),
                tldr=r["tldr"],
                use_case=r["use_case"],
                breakthrough_score=float(r["breakthrough_score"]),
                roi_score=float(r["roi_score"]),
                repo_url=r["repo_url"],
            )
            if ex.rating in (FeedbackRating.HIT, FeedbackRating.INSPIRE):
                if ex.breakthrough_id not in seen_pos and len(pos) < limit:
                    seen_pos.add(ex.breakthrough_id)
                    pos.append(ex)
            elif ex.rating == FeedbackRating.HYPE:
                if ex.breakthrough_id not in seen_neg and len(neg) < limit:
                    seen_neg.add(ex.breakthrough_id)
                    neg.append(ex)

        return pos, neg
