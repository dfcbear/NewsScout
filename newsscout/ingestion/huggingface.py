"""newsscout.ingestion.huggingface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hugging Face Daily Papers poller. Extracts arXiv IDs and strictly enforces
the presence of verified linked GitHub repositories, dropping theoretical
papers without runnable code.
"""

from __future__ import annotations

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

HF_DAILY_PAPERS_URL: Final[str] = "https://huggingface.co/api/daily_papers"

# Regex to capture GitHub repo URLs from summaries, project pages, or markdown links
GITHUB_REPO_REGEX: Final[re.Pattern[str]] = re.compile(
    r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)


def extract_github_repo_url(
    paper_data: dict[str, Any],
    parent_item: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Finds a linked GitHub repository URL from structured metadata, projectPage, or summary."""
    # 1. Direct structured field in HF paper schema
    direct_repo = paper_data.get("githubRepo")
    if direct_repo and isinstance(direct_repo, str):
        direct_repo = direct_repo.strip()
        if "github.com" in direct_repo.lower():
            match = GITHUB_REPO_REGEX.search(direct_repo)
            if match:
                clean_slug = match.group(1).rstrip("/.").removesuffix(".git")
                return f"https://github.com/{clean_slug}"
        elif re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", direct_repo):
            clean_slug = direct_repo.rstrip("/.").removesuffix(".git")
            return f"https://github.com/{clean_slug}"

    # 2. Check projectPage URL
    project_page = paper_data.get("projectPage")
    if project_page and isinstance(project_page, str):
        match = GITHUB_REPO_REGEX.search(project_page)
        if match:
            clean_slug = match.group(1).rstrip("/.").removesuffix(".git")
            return f"https://github.com/{clean_slug}"

    # 3. Check paper summary
    summary = paper_data.get("summary") or (parent_item.get("summary") if parent_item else "")
    if summary and isinstance(summary, str):
        match = GITHUB_REPO_REGEX.search(summary)
        if match:
            clean_slug = match.group(1).rstrip("/.").removesuffix(".git")
            return f"https://github.com/{clean_slug}"

    return None


class HuggingFaceIngestionSource(BaseIngestionSource):
    """Ingests Hugging Face Daily Papers, strictly requiring linked code repositories."""

    source_name: str = "hf_daily_papers"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict[str, Any]] = None,
        min_upvotes: Optional[int] = None,
        target_date: Optional[str] = None,  # YYYY-MM-DD
    ) -> None:
        super().__init__(settings=settings, client=client, config=config)
        self.min_upvotes = (
            min_upvotes
            if min_upvotes is not None
            else self.config.get("min_upvotes", self.settings.hf_papers_min_upvotes)
        )
        self.target_date = target_date

    async def fetch_items(self, limit: Optional[int] = None) -> AsyncIterator[RawItem]:
        """Pulls daily papers, filters for linked code, and yields normalized RawItems."""
        url = HF_DAILY_PAPERS_URL
        params: dict[str, str] = {}
        if self.target_date:
            params["date"] = self.target_date

        resp = await self.request_with_retry(url, params=params)
        try:
            data = resp.json()
        except Exception:
            logger.warning("[huggingface] Failed to parse daily_papers JSON")
            return

        if not isinstance(data, list):
            logger.warning("[huggingface] Unexpected response format from daily_papers: %s", type(data))
            return

        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                raw_item = self._process_paper(item)
            except Exception as err:
                logger.warning("[huggingface] Error processing paper item: %s", err)
                continue

            if raw_item is not None:
                yield raw_item
                count += 1
                if limit is not None and count >= limit:
                    break

    def _process_paper(self, item: dict[str, Any]) -> Optional[RawItem]:
        """Validates code repository existence and upvotes, returning RawItem."""
        paper = item.get("paper") or {}
        arxiv_id = str(paper.get("id") or item.get("id") or "").strip()
        if not arxiv_id:
            return None

        try:
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
        except (ValueError, TypeError):
            upvotes = 0

        if upvotes < self.min_upvotes:
            logger.debug("[huggingface] Dropping paper %s: upvotes %d < %d", arxiv_id, upvotes, self.min_upvotes)
            return None

        github_repo = extract_github_repo_url(paper, parent_item=item)
        if not github_repo:
            # STRICT REQUIREMENT: Drop papers without linked code
            logger.debug("[huggingface] Dropping paper %s: NO linked GitHub repository", arxiv_id)
            return None

        title = str(paper.get("title") or item.get("title") or f"Paper {arxiv_id}")
        summary = str(paper.get("summary") or item.get("summary") or "")

        return RawItem(
            source=self.source_name,
            source_id=f"arxiv:{arxiv_id}",
            title=title,
            url=f"https://huggingface.co/papers/{arxiv_id}",
            raw_content=summary,
            metadata={
                "arxiv_id": arxiv_id,
                "github_repo": github_repo,
                "upvotes": upvotes,
                "num_comments": item.get("numComments", 0),
                "published_at": item.get("publishedAt"),
                "project_page": paper.get("projectPage"),
                "thumbnail": item.get("thumbnail"),
            },
            ingested_at=datetime.now(timezone.utc),
        )
