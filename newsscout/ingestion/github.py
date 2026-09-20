"""newsscout.ingestion.github
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ingestion engine for GitHub Releases and tags. Combines GitHub REST API
with an automated zero-token fallback to repository Atom syndication feeds
(releases.atom) to survive rate-limit exhaustion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import html
import logging
import re
from typing import Any, Final, Optional
import xml.etree.ElementTree as ET

import httpx

from newsscout.config import Settings
from newsscout.ingestion.base import BaseIngestionSource
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)

DEFAULT_TIER1_REPOS: Final[list[str]] = [
    "vllm-project/vllm",
    "ggerganov/llama.cpp",
    "ollama/ollama",
    "sgl-project/sglang",
    "qdrant/qdrant",
    "DS4SD/docling",
]

ATOM_NS: Final[dict[str, str]] = {"atom": "http://www.w3.org/2005/Atom"}


def strip_html_tags(html_text: str) -> str:
    """Converts HTML markup into clean readable text/markdown without external deps."""
    if not html_text:
        return ""
    text = html.unescape(html_text)
    # Replace line breaks and paragraph closings
    text = re.sub(r"<(?:br\s*/?|/p|/li)>", "\n", text, flags=re.IGNORECASE)
    # Replace list items with bullet
    text = re.sub(r"<li[^>]*>", "• ", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple consecutive empty lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


class GitHubIngestionSource(BaseIngestionSource):
    """Monitors GitHub releases and tags via REST API with automated Atom feed fallback."""

    source_name: str = "github_releases"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        config: Optional[dict[str, Any]] = None,
        target_repos: Optional[list[str]] = None,
        force_atom_feed: bool = False,
    ) -> None:
        super().__init__(settings=settings, client=client, config=config)
        self.target_repos = (
            target_repos
            or self.config.get("repos")
            or DEFAULT_TIER1_REPOS
        )
        self.force_atom_feed = force_atom_feed

    def _get_default_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github.v3+json",
        }
        token = self.settings.github_token.get_secret_value().strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def fetch_items(self, limit: Optional[int] = None) -> AsyncIterator[RawItem]:
        """Streams normalized RawItem releases across all configured repositories."""
        count = 0
        for repo_slug in self.target_repos:
            try:
                async for item in self._fetch_repo_releases(repo_slug):
                    yield item
                    count += 1
                    if limit is not None and count >= limit:
                        return
            except Exception as err:
                logger.error("[github] Failed to fetch releases for %s: %s", repo_slug, err)

    async def _fetch_repo_releases(self, repo_slug: str) -> AsyncIterator[RawItem]:
        """Fetches releases for a single repository, trying REST then falling back to Atom."""
        if not self.force_atom_feed:
            try:
                async for item in self._fetch_via_rest(repo_slug):
                    yield item
                return
            except (httpx.HTTPStatusError, httpx.RequestError) as rest_err:
                status = getattr(getattr(rest_err, "response", None), "status_code", None)
                logger.warning(
                    "[github] REST API error (status=%s) for %s. Falling back to Atom feed: %s",
                    status,
                    repo_slug,
                    rest_err,
                )

        # Fallback to Atom feed
        async for item in self._fetch_via_atom(repo_slug):
            yield item

    async def _fetch_via_rest(self, repo_slug: str) -> AsyncIterator[RawItem]:
        """Ingests releases using api.github.com REST endpoints."""
        url = f"https://api.github.com/repos/{repo_slug}/releases?per_page=10"
        resp = await self.request_with_retry(url)
        releases = resp.json()

        if not isinstance(releases, list) or len(releases) == 0:
            # Edge case E01: No formal releases, fallback to git tags
            logger.info("[github] No formal releases found via REST for %s. Checking tags...", repo_slug)
            async for tag_item in self._fetch_tags_via_rest(repo_slug):
                yield tag_item
            return

        for rel in releases:
            tag = rel.get("tag_name") or "latest"
            release_name = rel.get("name") or tag
            html_url = rel.get("html_url") or f"https://github.com/{repo_slug}/releases/tag/{tag}"
            body = rel.get("body") or ""
            published_at = rel.get("published_at") or datetime.now(timezone.utc).isoformat()
            assets = rel.get("assets", [])

            raw_item = RawItem(
                source=self.source_name,
                source_id=f"{repo_slug}:{tag}",
                title=f"{repo_slug} {tag}: {release_name}",
                url=html_url,
                raw_content=body,
                metadata={
                    "owner": repo_slug.split("/")[0],
                    "repo": repo_slug.split("/")[1] if "/" in repo_slug else repo_slug,
                    "tag": tag,
                    "published_at": published_at,
                    "is_prerelease": bool(rel.get("prerelease")),
                    "assets": [a.get("name", "") for a in assets if isinstance(a, dict)] if assets else [],
                    "assets_count": len(assets),
                    "ingestion_method": "rest",
                    "author": rel.get("author", {}).get("login", "") if isinstance(rel.get("author"), dict) else "",
                },
                ingested_at=datetime.now(timezone.utc),
            )
            yield raw_item

    async def _fetch_tags_via_rest(self, repo_slug: str) -> AsyncIterator[RawItem]:
        """Probes git tags when formal releases do not exist (Edge Case E01)."""
        url = f"https://api.github.com/repos/{repo_slug}/tags?per_page=5"
        resp = await self.request_with_retry(url)
        tags = resp.json()
        if not isinstance(tags, list):
            return

        for tag_obj in tags:
            tag_name = tag_obj.get("name", "")
            commit_sha = tag_obj.get("commit", {}).get("sha", "") if isinstance(tag_obj.get("commit"), dict) else ""
            html_url = f"https://github.com/{repo_slug}/releases/tag/{tag_name}"

            yield RawItem(
                source=self.source_name,
                source_id=f"{repo_slug}:{tag_name}",
                title=f"{repo_slug} tag {tag_name}",
                url=html_url,
                raw_content=f"Release tag {tag_name} on commit {commit_sha[:8]}",
                metadata={
                    "owner": repo_slug.split("/")[0],
                    "repo": repo_slug.split("/")[1] if "/" in repo_slug else repo_slug,
                    "tag": tag_name,
                    "commit_sha": commit_sha,
                    "is_prerelease": False,
                    "ingestion_method": "rest_tag",
                },
                ingested_at=datetime.now(timezone.utc),
            )

    async def _fetch_via_atom(self, repo_slug: str) -> AsyncIterator[RawItem]:
        """Ingests releases via public Atom feed (https://github.com/{owner}/{repo}/releases.atom)."""
        url = f"https://github.com/{repo_slug}/releases.atom"
        resp = await self.request_with_retry(url, headers={"Accept": "application/atom+xml, text/xml"})
        xml_text = resp.text

        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            title_el = entry.find("atom:title", ATOM_NS)
            if title_el is None:
                title_el = entry.find("{http://www.w3.org/2005/Atom}title")
            title = title_el.text if title_el is not None and title_el.text else "Release"

            link_el = entry.find("atom:link", ATOM_NS)
            if link_el is None:
                link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            html_url = link_el.attrib.get("href", "") if link_el is not None else ""

            updated_el = entry.find("atom:updated", ATOM_NS)
            if updated_el is None:
                updated_el = entry.find("{http://www.w3.org/2005/Atom}updated")
            updated = updated_el.text if updated_el is not None and updated_el.text else ""

            content_el = entry.find("atom:content", ATOM_NS)
            if content_el is None:
                content_el = entry.find("{http://www.w3.org/2005/Atom}content")
            raw_html = content_el.text if content_el is not None and content_el.text else ""
            clean_body = strip_html_tags(raw_html)

            tag = "release"
            if "/releases/tag/" in html_url:
                tag = html_url.split("/releases/tag/")[-1]
            elif ":" in title:
                tag = title.split(":")[0].strip()

            yield RawItem(
                source=self.source_name,
                source_id=f"{repo_slug}:{tag}",
                title=f"{repo_slug} {title}",
                url=html_url or f"https://github.com/{repo_slug}",
                raw_content=clean_body,
                metadata={
                    "owner": repo_slug.split("/")[0],
                    "repo": repo_slug.split("/")[1] if "/" in repo_slug else repo_slug,
                    "tag": tag,
                    "published_at": updated,
                    "is_prerelease": "rc" in tag.lower() or "beta" in tag.lower(),
                    "ingestion_method": "atom",
                },
                ingested_at=datetime.now(timezone.utc),
            )
