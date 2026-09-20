"""newsscout.ingestion.hackernews
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hacker News Firebase API poller with concurrent batch fetching, AI domain
keyword filtering, and community sceptic-ratio calculation (score > 120 and
descendants/score > 0.4).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
import logging
import re
from typing import Any, Final, Optional

import httpx

from newsscout.config import Settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)

HN_BASE_URL: Final[str] = "https://hacker-news.firebaseio.com/v0"

DEFAULT_AI_KEYWORDS: Final[list[str]] = [
    "ai", "llm", "llms", "gpt", "gpt-4", "gpt-5", "gemini", "claude", "llama", "llama-3", "vllm", "sglang", "ollama",
    "transformer", "transformers", "diffusion", "agent", "agents", "agentic", "mcp", "rag", "embedding", "embeddings",
    "vector", "qdrant", "chroma", "quantization", "nvfp4", "exl2", "awq", "gguf", "mamba", "titans", "robotics", "vla", "ros2",
    "cuda", "triton", "tensorrt", "inference", "prompt", "neural", "deep learning", "machine learning", "open-source model",
    "whisper", "tts", "stt", "vision model", "multimodal", "docling", "deepseek",
]

def compile_topic_pattern(custom_keywords: Optional[Sequence[str]] = None) -> re.Pattern[str]:
    """Compiles a regex pattern matching default AI keywords plus any dynamic custom keywords."""
    words = list(DEFAULT_AI_KEYWORDS)
    if custom_keywords:
        for kw in custom_keywords:
            cleaned = kw.strip()
            if cleaned and cleaned.lower() not in [w.lower() for w in words]:
                words.append(re.escape(cleaned))
    pattern_str = r"\b(" + "|".join(words) + r")\b"
    return re.compile(pattern_str, re.IGNORECASE)

# Backward compatible default regex pattern
AI_TOPIC_PATTERN: Final[re.Pattern[str]] = compile_topic_pattern()


class HackerNewsIngestionSource(BaseIngestionSource):
    """Polls Hacker News Firebase API for AI breakthroughs meeting critical sceptic ratios."""

    source_name: str = "hacker_news"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict[str, Any]] = None,
        min_score: Optional[int] = None,
        min_comment_ratio: Optional[float] = None,
        max_poll_items: int = 60,
        concurrency_limit: int = 20,
        custom_keywords: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(settings=settings, client=client, config=config)
        self.min_score = (
            min_score
            if min_score is not None
            else self.config.get("min_score", self.settings.hn_min_score)
        )
        self.min_comment_ratio = (
            min_comment_ratio
            if min_comment_ratio is not None
            else self.config.get("min_ratio", self.settings.hn_min_comment_ratio)
        )
        self.max_poll_items = max_poll_items
        self.concurrency_limit = concurrency_limit
        
        cfg_kw = self.config.get("custom_keywords") or []
        kws = list(custom_keywords or cfg_kw)
        self.topic_pattern = compile_topic_pattern(kws) if kws else AI_TOPIC_PATTERN
        self.max_poll_items = max_poll_items
        self.concurrency_limit = concurrency_limit

    async def fetch_items(self, limit: Optional[int] = None) -> AsyncIterator[RawItem]:
        """Fetches top stories, concurrently evaluates item payloads, and yields qualified items."""
        top_ids = await self._fetch_top_story_ids()
        if not top_ids:
            return

        target_ids = top_ids[: self.max_poll_items]
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        async def _fetch_one(item_id: int) -> Optional[dict[str, Any]]:
            async with semaphore:
                try:
                    url = f"{HN_BASE_URL}/item/{item_id}.json"
                    resp = await self.request_with_retry(url)
                    data = resp.json()
                    return data if isinstance(data, dict) else None
                except Exception as err:
                    logger.warning("[hackernews] Failed to fetch item %d: %s", item_id, err)
                    return None

        tasks = [_fetch_one(item_id) for item_id in target_ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=False)

        count = 0
        for item_data in raw_results:
            if item_data is None:
                continue

            try:
                raw_item = self._filter_and_normalize(item_data)
            except Exception as err:
                logger.warning("[hackernews] Error normalizing item: %s", err)
                continue

            if raw_item is not None:
                yield raw_item
                count += 1
                if limit is not None and count >= limit:
                    break

    async def _fetch_top_story_ids(self) -> list[int]:
        """Pulls the array of current top story IDs from Firebase."""
        url = f"{HN_BASE_URL}/topstories.json"
        resp = await self.request_with_retry(url)
        try:
            data = resp.json()
        except Exception:
            return []
        if isinstance(data, list):
            return [int(x) for x in data if isinstance(x, (int, str)) and str(x).isdigit()]
        return []

    def _filter_and_normalize(self, data: dict[str, Any]) -> Optional[RawItem]:
        """Applies sceptic ratio and AI topic filters, returning RawItem if qualified."""
        if data.get("type") != "story":
            return None
        if data.get("dead") or data.get("deleted"):
            return None

        try:
            score = int(data.get("score") or 0)
        except (ValueError, TypeError):
            return None

        if score <= self.min_score:
            return None

        # Schema distinction: descendants is total comments; kids is top-level only
        descendants = data.get("descendants")
        try:
            descendants_count = int(descendants) if descendants is not None else 0
        except (ValueError, TypeError):
            descendants_count = 0

        # Compute sceptic ratio
        comment_ratio = descendants_count / score if score > 0 else 0.0
        if comment_ratio <= self.min_comment_ratio:
            logger.debug(
                "[hackernews] Dropping item %s ('%s'): score %d, ratio %.2f <= %.2f",
                str(data.get("id")),
                data.get("title"),
                score,
                comment_ratio,
                self.min_comment_ratio,
            )
            return None

        title = str(data.get("title") or "")
        text = str(data.get("text") or "")
        target_url = str(data.get("url") or "")
        combined_text = f"{title} {target_url} {text}"

        # AI topic filter
        if not self.topic_pattern.search(combined_text):
            logger.debug("[hackernews] Dropping non-AI item %s: '%s'", str(data.get("id")), title)
            return None

        item_id = str(data.get("id") or "")
        if not item_id:
            return None

        final_url = target_url if target_url else f"https://news.ycombinator.com/item?id={item_id}"

        # Extract possible GitHub URLs from text or url for downstream Stage 1 heuristics
        extracted_github_urls = []
        for url_match in re.finditer(r"https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", combined_text):
            clean_url = url_match.group(0).rstrip("/.")
            extracted_github_urls.append(clean_url)

        return RawItem(
            source=self.source_name,
            source_id=item_id,
            title=title,
            url=final_url,
            raw_content=text or title,
            metadata={
                "hn_id": item_id,
                "score": score,
                "descendants": descendants_count,
                "comment_ratio": round(comment_ratio, 4),
                "author": data.get("by", ""),
                "timestamp": data.get("time"),
                "hn_permalink": f"https://news.ycombinator.com/item?id={item_id}",
                "is_ask_or_show_hn": not bool(target_url),
                "extracted_github_urls": extracted_github_urls,
            },
            ingested_at=datetime.now(timezone.utc),
        )
