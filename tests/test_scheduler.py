"""tests.test_scheduler
~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for DigestScheduler (resolving M4 audit finding):
- Initialization and cron trigger configuration
- Trigger digest execution
- Scheduled pipeline run execution (Ingestion -> Stage 1 -> Stage 2)
- Error handling and graceful shutdown
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newsscout.config import Settings
from newsscout.scheduler import DigestScheduler
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import Digest, DigestSlot, DigestStatus


@pytest.fixture
def sched_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "sched_test.db"
    conn = Database(db_file)
    with conn.get_sync_connection() as c:
        apply_migrations(conn=c)
    return conn


@pytest.mark.asyncio
async def test_scheduler_init(sched_db: Database, test_settings: Settings):
    """Scheduler initializes with settings and database."""
    scheduler = DigestScheduler(sched_db, test_settings)
    assert scheduler.db == sched_db
    assert scheduler.settings == test_settings
    assert scheduler._scheduler is None


@pytest.mark.asyncio
async def test_scheduler_start_and_stop(sched_db: Database, test_settings: Settings):
    """Scheduler starts APScheduler jobs and stops cleanly."""
    scheduler = DigestScheduler(sched_db, test_settings)
    await scheduler.start()

    assert scheduler._scheduler is not None
    assert scheduler._scheduler.running is True

    # Check registered jobs
    job_ids = [j.id for j in scheduler._scheduler.get_jobs()]
    assert "morning_digest" in job_ids
    assert "afternoon_digest" in job_ids
    assert "recurring_pipeline" in job_ids

    await scheduler.stop()
    assert scheduler._scheduler.running is False


@pytest.mark.asyncio
async def test_trigger_digest_calls_generator(sched_db: Database, test_settings: Settings):
    """trigger_digest invokes DigestGenerator and updates status."""
    mock_digest = Digest(
        id=1,
        slot=DigestSlot.MORNING,
        digest_date="2026-09-18",
        title="Morning Test Digest",
        status=DigestStatus.SYNTHESIZED,
    )
    mock_generator = MagicMock()
    mock_generator.generate_digest = AsyncMock(return_value=mock_digest)

    mock_bot = MagicMock()
    mock_bot.__aenter__ = AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = AsyncMock(return_value=None)
    mock_bot.send_digest_menu = AsyncMock(return_value=42)

    scheduler = DigestScheduler(
        sched_db,
        test_settings,
        digest_generator=mock_generator,
        telegram_bot=mock_bot,
    )
    result = await scheduler.trigger_digest(DigestSlot.MORNING)

    assert result.id == 1
    mock_generator.generate_digest.assert_called_once_with(slot=DigestSlot.MORNING)
    mock_bot.send_digest_menu.assert_called_once_with(mock_digest)


@pytest.mark.asyncio
async def test_trigger_pipeline_executes_stages(sched_db: Database, test_settings: Settings):
    """trigger_pipeline coordinates ingestion, Stage 1, and Stage 2."""
    scheduler = DigestScheduler(sched_db, test_settings)

    mock_ingest_metrics = {"total_inserted_or_updated": 3, "sources": {}}
    mock_s1_results = [MagicMock(id=1), MagicMock(id=2)]

    with patch("newsscout.ingestion.pipeline.IngestionPipeline.run_cycle", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_ingest_metrics
        with patch("newsscout.filtering.stage1.Stage1Filter.run_pipeline", new_callable=AsyncMock) as mock_s1:
            mock_s1.return_value = mock_s1_results

            res = await scheduler.trigger_pipeline()

            assert res["ingestion"] == mock_ingest_metrics
            assert res["stage1_count"] == 2
            mock_ingest.assert_called_once()
            mock_s1.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_error_handled_gracefully(sched_db: Database, test_settings: Settings):
    """Pipeline failures are caught and recorded without crashing."""
    scheduler = DigestScheduler(sched_db, test_settings)

    with patch("newsscout.ingestion.pipeline.IngestionPipeline.run_cycle", side_effect=RuntimeError("Network down")):
        res = await scheduler.trigger_pipeline()
        assert "error" in res
        assert "Network down" in res["error"]