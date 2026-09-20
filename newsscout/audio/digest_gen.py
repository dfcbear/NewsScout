"""newsscout.audio.digest_gen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Digest orchestrator that coordinates the full audio digest lifecycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from newsscout.audio.script_gen import DialogueScriptGenerator
from newsscout.audio.tts_engine import BaseTTSEngine, EdgeTTSEngine
from newsscout.config import Settings, get_settings
from newsscout.llm.base import BaseLLMClient
from newsscout.storage.models import (
    Breakthrough,
    Digest,
    DigestSlot,
    DigestStatus,
    Stage2Category,
    Track,
    TrackType,
)

logger = logging.getLogger("newsscout.audio.digest_gen")


class DigestGenerator:
    """Orchestrates the full audio digest generation pipeline."""

    def __init__(
        self,
        db: Any,
        settings: Optional[Settings] = None,
        tts_engine: Optional[BaseTTSEngine] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.tts_engine = tts_engine or EdgeTTSEngine(self.settings)
        self.script_gen = DialogueScriptGenerator(self.settings, llm_client=llm_client)

    async def generate_digest(
        self,
        slot: DigestSlot = DigestSlot.MORNING,
        breakthroughs: Optional[list[Breakthrough]] = None,
    ) -> Digest:
        """Generate a complete audio digest for the given time slot."""
        if breakthroughs is None:
            breakthroughs = await self._fetch_breakthroughs()

        scripts = await self.script_gen.generate_scripts_async(
            breakthroughs,
            min_duration_minutes=self.settings.audio_min_duration_minutes,
            max_duration_minutes=self.settings.audio_max_duration_minutes,
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slot_name = "Morgen" if slot == DigestSlot.MORNING else "Nachmittag"
        digest = Digest(
            slot=slot,
            digest_date=today,
            title=f"NewsScout {slot_name} Digest - {today}",
            total_duration_seconds=0,
            status=DigestStatus.GENERATING,
        )

        digest_id = await self._create_digest_record(digest)
        digest.id = digest_id

        try:
            audio_dir = Path(self.settings.audio_output_dir) / f"digest_{digest_id}"
            audio_dir.mkdir(parents=True, exist_ok=True)

            track_paths: list[Path] = []
            tracks: list[Track] = []

            for i, script in enumerate(scripts, start=1):
                track_path = audio_dir / f"track_{i}.mp3"
                await self.tts_engine.synthesize_script(script, track_path)
                track_paths.append(track_path)

                duration = self.tts_engine.get_mp3_duration_seconds(track_path)

                bt_id: Optional[int] = None
                if script.track_type in (TrackType.DEEP_DIVE_1, TrackType.DEEP_DIVE_2):
                    idx = 0 if script.track_type == TrackType.DEEP_DIVE_1 else 1
                    core_items = [
                        b for b in breakthroughs
                        if b.category == Stage2Category.CORE
                    ]
                    core_items.sort(key=lambda b: b.breakthrough_score, reverse=True)
                    if idx < len(core_items) and core_items[idx].id:
                        bt_id = core_items[idx].id

                track = Track(
                    digest_id=digest_id,
                    track_number=i,
                    track_type=script.track_type,
                    title=script.title,
                    duration_seconds=int(duration),
                    audio_path=str(track_path),
                    script=script,
                    breakthrough_id=bt_id,
                )
                await self._create_track_record(track)
                tracks.append(track)

            merged_path = audio_dir / "playlist.mp3"
            if len(track_paths) > 1:
                await self.tts_engine.merge_tracks(track_paths, merged_path)
            elif track_paths:
                merged_path.write_bytes(track_paths[0].read_bytes())

            total_duration = sum(t.duration_seconds for t in tracks)
            digest.tracks = tracks
            digest.total_duration_seconds = total_duration
            digest.merged_audio_path = str(merged_path)
            digest.status = DigestStatus.SYNTHESIZED

            await self._update_digest_record(digest)

            logger.info(
                "Digest %d generated: %d tracks, %d seconds total",
                digest_id,
                len(tracks),
                total_duration,
            )
            return digest

        except Exception as e:
            logger.error("Digest generation failed: %s", e)
            digest.status = DigestStatus.FAILED
            await self._update_digest_record(digest)
            raise

    async def _fetch_breakthroughs(self) -> list[Breakthrough]:
        # Prioritize breakthroughs that have not yet been featured in any digest track
        rows = await self.db.fetch_all(
            """
            SELECT * FROM breakthroughs
            WHERE category IN ('core', 'serendipity')
              AND id NOT IN (
                  SELECT DISTINCT breakthrough_id FROM digest_tracks WHERE breakthrough_id IS NOT NULL
              )
            ORDER BY breakthrough_score DESC, evaluated_at DESC
            LIMIT 20;
            """
        )
        if not rows:
            # Fallback to recent top breakthroughs if all have already been featured
            rows = await self.db.fetch_all(
                """
                SELECT * FROM breakthroughs
                WHERE category IN ('core', 'serendipity')
                ORDER BY evaluated_at DESC, breakthrough_score DESC
                LIMIT 20;
                """
            )
        return [Breakthrough.from_row(row) for row in rows]

    async def _create_digest_record(self, digest: Digest) -> int:
        params = digest.to_db_params()
        return await self.db.execute(
            """
            INSERT INTO digests (slot, digest_date, title, total_duration_seconds,
                merged_audio_path, telegram_message_id, status, created_at, delivered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                params["slot"],
                params["digest_date"],
                params["title"],
                params["total_duration_seconds"],
                params["merged_audio_path"],
                params["telegram_message_id"],
                params["status"],
                params["created_at"],
                params["delivered_at"],
            ),
        )

    async def _create_track_record(self, track: Track) -> int:
        params = track.to_db_params()
        return await self.db.execute(
            """
            INSERT INTO digest_tracks (digest_id, track_number, track_type, title,
                duration_seconds, audio_path, script_json, breakthrough_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                params["digest_id"],
                params["track_number"],
                params["track_type"],
                params["title"],
                params["duration_seconds"],
                params["audio_path"],
                params["script_json"],
                params["breakthrough_id"],
                params["created_at"],
            ),
        )

    async def _update_digest_record(self, digest: Digest) -> None:
        params = digest.to_db_params()
        await self.db.execute(
            """
            UPDATE digests SET
                total_duration_seconds = ?,
                merged_audio_path = ?,
                status = ?
            WHERE id = ?;
            """,
            (
                params["total_duration_seconds"],
                params["merged_audio_path"],
                params["status"],
                digest.id,
            ),
        )
