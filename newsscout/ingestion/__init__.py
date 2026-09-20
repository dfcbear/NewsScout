"""newsscout.ingestion
~~~~~~~~~~~~~~~~~~~~
Multi-source ingestion engines and pipeline for NewsScout.
"""

from __future__ import annotations

from newsscout.ingestion.base import (
    DEFAULT_LIMITS,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    BaseIngestionSource,
)
from newsscout.ingestion.github import (
    DEFAULT_TIER1_REPOS,
    GitHubIngestionSource,
    strip_html_tags,
)
from newsscout.ingestion.hackernews import (
    AI_TOPIC_PATTERN,
    HN_BASE_URL,
    HackerNewsIngestionSource,
)
from newsscout.ingestion.huggingface import (
    HF_DAILY_PAPERS_URL,
    HuggingFaceIngestionSource,
    extract_github_repo_url,
)
from newsscout.ingestion.pipeline import (
    SOURCE_REGISTRY,
    IngestionPipeline,
)
from newsscout.ingestion.search import SearchIngestionSource

__all__ = [
    "BaseIngestionSource",
    "GitHubIngestionSource",
    "HackerNewsIngestionSource",
    "HuggingFaceIngestionSource",
    "SearchIngestionSource",
    "IngestionPipeline",
    "DEFAULT_LIMITS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_USER_AGENT",
    "DEFAULT_TIER1_REPOS",
    "strip_html_tags",
    "AI_TOPIC_PATTERN",
    "HN_BASE_URL",
    "HF_DAILY_PAPERS_URL",
    "extract_github_repo_url",
    "SOURCE_REGISTRY",
]
