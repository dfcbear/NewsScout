"""newsscout.scheduler
~~~~~~~~~~~~~~~~~~~~~~
APScheduler-based scheduler for twice-daily digest generation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from newsscout.config import Settings, get_settings
from newsscout.storage.models import Digest, DigestSlot

logger = logging.getLogger("newsscout.scheduler")


class DigestScheduler:
    """Schedules and executes twice-daily audio digest generation jobs."""

    def __init__(
        self,
        db: Any,
        settings: Optional[Settings] = None,
        digest_generator: Any = None,
        telegram_bot: Any = None,
        dispatcher: Any = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self._digest_generator = digest_generator
        self._telegram_bot = telegram_bot
        self._dispatcher = dispatcher
        self._scheduler: Any = None

    async def start(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler = AsyncIOScheduler()

        # Recurring pipeline ingestion & evaluation cycle
        self._scheduler.add_job(
            self._run_pipeline_job,
            IntervalTrigger(
                hours=self.settings.pipeline_interval_hours,
                timezone=self.settings.timezone,
            ),
            id="recurring_pipeline",
            replace_existing=True,
        )

        morning_parts = self.settings.schedule_morning.split(":")
        afternoon_parts = self.settings.schedule_afternoon.split(":")

        self._scheduler.add_job(
            self._run_digest_job,
            CronTrigger(
                hour=int(morning_parts[0]),
                minute=int(morning_parts[1]),
                timezone=self.settings.timezone,
            ),
            args=[DigestSlot.MORNING],
            id="morning_digest",
            replace_existing=True,
        )

        self._scheduler.add_job(
            self._run_digest_job,
            CronTrigger(
                hour=int(afternoon_parts[0]),
                minute=int(afternoon_parts[1]),
                timezone=self.settings.timezone,
            ),
            args=[DigestSlot.AFTERNOON],
            id="afternoon_digest",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info(
            "Scheduler started: pipeline_interval=%dh, morning=%s, afternoon=%s, tz=%s",
            self.settings.pipeline_interval_hours,
            self.settings.schedule_morning,
            self.settings.schedule_afternoon,
            self.settings.timezone,
        )

    async def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            await asyncio.sleep(0.05)
            logger.info("Scheduler stopped")

    async def trigger_digest(self, slot: DigestSlot = DigestSlot.MORNING) -> Digest:
        return await self._run_digest_job(slot)

    async def trigger_pipeline(self) -> dict[str, Any]:
        """Manually triggers an immediate pipeline cycle."""
        return await self._run_pipeline_job()

    async def _run_pipeline_job(self) -> dict[str, Any]:
        """Runs ingestion -> Stage 1 heuristics -> Stage 2 evaluation cycle."""
        logger.info("Starting scheduled ingestion & evaluation pipeline job")
        results: dict[str, Any] = {"ingestion": {}, "stage1_count": 0, "breakthroughs_count": 0}

        try:
            from newsscout.ingestion.pipeline import IngestionPipeline
            from newsscout.filtering.stage1 import Stage1Filter
            from newsscout.filtering.stage2 import Stage2Evaluator

            # 1. Run multi-source ingestion
            pipeline = IngestionPipeline(self.db, self.settings)
            ingest_metrics = await pipeline.run_cycle()
            results["ingestion"] = ingest_metrics

            # 2. Run Stage 1 heuristic filter
            stage1 = Stage1Filter(self.db)
            s1_results = await stage1.run_pipeline()
            results["stage1_count"] = len(s1_results)

            # 3. Run Stage 2 deep evaluator (if LLM credentials exist)
            if self.settings.has_llm_credentials:
                stage2 = Stage2Evaluator(self.db, self.settings)
                breakthroughs = await stage2.evaluate_pending_items()
                results["breakthroughs_count"] = len(breakthroughs)

                # Broadcast decision cards for high-scoring items
                eligible = [
                    bt for bt in breakthroughs
                    if bt.category in ("core", "serendipity") and bt.breakthrough_score >= 8.0
                ]
                if eligible:
                    if self._dispatcher is not None:
                        for bt in eligible:
                            try:
                                await self._dispatcher.broadcast_card(bt)
                            except Exception as err:
                                logger.warning("Failed to broadcast decision card for %s: %s", bt.title, err)
                    elif self.settings.has_telegram_credentials or self._telegram_bot is not None:
                        if self._telegram_bot is None:
                            from newsscout.delivery.telegram_bot import TelegramBot
                            self._telegram_bot = TelegramBot(self.settings, db=self.db)

                        async with self._telegram_bot as bot:
                            for bt in eligible:
                                try:
                                    await bot.send_decision_card(bt)
                                except Exception as err:
                                    logger.warning("Failed to send decision card for %s: %s", bt.title, err)

            logger.info(
                "Pipeline job completed: ingested %d items, stage1 processed %d, breakthroughs %d",
                ingest_metrics.get("total_inserted_or_updated", 0),
                results["stage1_count"],
                results["breakthroughs_count"],
            )
        except Exception as e:
            logger.error("Pipeline job failed: %s", e, exc_info=True)
            results["error"] = str(e)

        return results

    async def _run_digest_job(self, slot: DigestSlot) -> Digest:
        logger.info("Starting digest generation for slot: %s", slot.value)

        try:
            if self._digest_generator is None:
                from newsscout.audio.digest_gen import DigestGenerator
                llm_client = None
                if self.settings.has_llm_credentials:
                    try:
                        from newsscout.llm.factory import create_llm_client
                        llm_client = create_llm_client(self.settings)
                    except Exception as err:
                        logger.warning("Could not initialize LLM client for digest generator: %s", err)
                self._digest_generator = DigestGenerator(
                    self.db, self.settings, llm_client=llm_client
                )

            digest = await self._digest_generator.generate_digest(slot=slot)

            # Broadcast audio via DeliveryDispatcher across active channels if present
            if self._dispatcher is not None and digest.merged_audio_path:
                caption = f"🎙️ {digest.title} ({slot.value.capitalize()})"
                try:
                    await self._dispatcher.broadcast_audio(
                        str(digest.merged_audio_path), caption=caption
                    )
                except Exception as exc:
                    logger.warning("Dispatcher broadcast_audio failed: %s", exc)

            if self.settings.has_telegram_credentials or self._telegram_bot is not None:
                if self._telegram_bot is None:
                    from newsscout.delivery.telegram_bot import TelegramBot
                    self._telegram_bot = TelegramBot(self.settings, db=self.db)

                async with self._telegram_bot as bot:
                    msg_id = await bot.send_digest_menu(digest)
                    logger.info("Digest menu sent, message_id=%d", msg_id)

                    from newsscout.storage.models import DigestStatus
                    digest.status = DigestStatus.DELIVERED
                    digest.telegram_message_id = msg_id
                    await self.db.execute(
                        "UPDATE digests SET status = ?, telegram_message_id = ? WHERE id = ?;",
                        (digest.status.value, msg_id, digest.id),
                    )

            logger.info("Digest job completed for slot: %s", slot.value)
            return digest

        except Exception as e:
            logger.error("Digest job failed for slot %s: %s", slot.value, e)
            raise


async def main() -> None:
    import uvicorn
    from newsscout.storage.db import Database
    from newsscout.storage.migrations import apply_migrations
    from newsscout.dashboard.app import create_app

    settings = get_settings()
    settings.ensure_directories()

    db = Database(settings.db_path)
    await db.initialize()
    conn = db.get_sync_connection()
    try:
        apply_migrations(conn=conn)
    finally:
        conn.close()

    from newsscout.delivery.dispatcher import create_default_dispatcher
    from newsscout.storage.preferences import PreferencesService

    preferences = PreferencesService(db, settings)
    dispatcher = create_default_dispatcher(settings=settings, db=db, preferences_service=preferences)
    scheduler = DigestScheduler(db, settings, dispatcher=dispatcher)
    await scheduler.start()

    app = create_app(settings)
    server_config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(server_config)

    try:
        await server.serve()
    finally:
        await scheduler.stop()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
