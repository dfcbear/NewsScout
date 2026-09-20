"""newsscout.ingestion.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified ingestion orchestrator that coordinates active ingestion sources,
normalizes incoming candidate payloads, and performs atomic upserts into
the SQLite WAL raw_items table.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional, Sequence

from newsscout.config import Settings, get_settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.ingestion.github import GitHubIngestionSource
from newsscout.ingestion.hackernews import HackerNewsIngestionSource
from newsscout.ingestion.huggingface import HuggingFaceIngestionSource
from newsscout.ingestion.search import SearchIngestionSource
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem, Source

logger = logging.getLogger(__name__)

SOURCE_REGISTRY: dict[str, type[BaseIngestionSource]] = {
    "github_releases": GitHubIngestionSource,
    "hacker_news": HackerNewsIngestionSource,
    "hf_daily_papers": HuggingFaceIngestionSource,
    "web_search": SearchIngestionSource,
    "multi_search": SearchIngestionSource,
}


class IngestionPipeline:
    """Orchestrator for executing ingestion cycles and persisting RawItems."""

    def __init__(
        self,
        db: Database,
        settings: Optional[Settings] = None,
        sources: Optional[list[BaseIngestionSource]] = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.custom_sources = sources

    async def run_cycle(self, limit_per_source: Optional[int] = None) -> dict[str, Any]:
        """Runs an ingestion cycle across all active sources.

        Returns a dictionary of execution metrics per source.
        """
        active_sources = self.custom_sources or await self._load_configured_sources()
        metrics: dict[str, Any] = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "sources": {},
            "total_inserted_or_updated": 0,
        }

        async def _run_source(src: BaseIngestionSource) -> tuple[str, list[RawItem], Optional[str]]:
            src_name = src.source_name
            items: list[RawItem] = []
            error_msg: Optional[str] = None
            try:
                async with src:
                    async for item in src.fetch_items(limit=limit_per_source):
                        items.append(item)
            except Exception as err:
                logger.error("[pipeline] Error running source '%s': %s", src_name, err, exc_info=True)
                error_msg = str(err)
            return src_name, items, error_msg

        results = await asyncio.gather(*[_run_source(src) for src in active_sources])

        for src_name, items, err in results:
            if err:
                metrics["sources"][src_name] = {"status": "error", "error": err, "items_count": 0}
                continue

            inserted_count = await self.persist_raw_items(items)
            await self._update_source_last_poll(src_name)

            metrics["sources"][src_name] = {
                "status": "success",
                "items_count": len(items),
                "persisted_count": inserted_count,
            }
            metrics["total_inserted_or_updated"] += inserted_count

        metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
        return metrics

    async def persist_raw_items(self, items: Sequence[RawItem]) -> int:
        """Atomically upserts a batch of RawItems into SQLite raw_items table."""
        if not items:
            return 0

        query = """
        INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
        VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)
        ON CONFLICT(source, source_id) DO UPDATE SET
            title = excluded.title,
            url = excluded.url,
            raw_content = coalesce(excluded.raw_content, raw_items.raw_content),
            metadata_json = excluded.metadata_json;
        """
        params_seq = [item.to_db_params() for item in items]
        await self.db.execute_many(query, params_seq)
        return len(items)

    async def _load_configured_sources(self) -> list[BaseIngestionSource]:
        """Reads enabled sources from SQLite sources table, instantiating their engines."""
        rows = await self.db.fetch_all("SELECT * FROM sources WHERE is_active = 1;")
        sources: list[BaseIngestionSource] = []

        for row in rows:
            src_entity = Source.from_row(row)
            cls = SOURCE_REGISTRY.get(src_entity.name)
            if cls:
                sources.append(cls(settings=self.settings, config=src_entity.config))
            else:
                logger.warning("[pipeline] Unrecognized source name in DB: %s", src_entity.name)

        if not sources:
            # Cold-start fallback if DB table is unseeded
            sources = [
                GitHubIngestionSource(settings=self.settings),
                HackerNewsIngestionSource(settings=self.settings),
                HuggingFaceIngestionSource(settings=self.settings),
            ]
        return sources

    async def _update_source_last_poll(self, source_name: str) -> None:
        """Updates last_poll_at timestamp in the sources table."""
        now_iso = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE sources SET last_poll_at = :last_poll WHERE name = :name;",
            {"last_poll": now_iso, "name": source_name},
        )
