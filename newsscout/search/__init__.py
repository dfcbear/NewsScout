"""newsscout.search
~~~~~~~~~~~~~~~~~~~
Modular multi-engine search aggregation pipeline for active breakthrough discovery.
Provides zero-key providers (SearXNG, DuckDuckGo) and optional commercial providers
(Tavily, Exa) with URL normalization and parallel deduplication.
"""

from newsscout.search.aggregator import MultiSearchAggregator
from newsscout.search.base import (
    BaseSearchProvider,
    SearchError,
    SearchParseError,
    SearchProviderError,
    SearchRateLimitError,
    SearchResult,
    SearchTimeoutError,
    SearchUnavailableError,
    SearchUpstreamError,
)
from newsscout.search.duckduckgo import DuckDuckGoSearchProvider
from newsscout.search.exa import ExaSearchProvider
from newsscout.search.normalizer import (
    clean_and_canonicalize_url,
    extract_all_github_urls,
    extract_github_repo,
    extract_github_repo_url,
    normalize_url,
)
from newsscout.search.searxng import SearXNGSearchProvider
from newsscout.search.tavily import TavilySearchProvider

__all__ = [
    "SearchResult",
    "BaseSearchProvider",
    "SearchError",
    "SearchTimeoutError",
    "SearchRateLimitError",
    "SearchUpstreamError",
    "SearchUnavailableError",
    "SearchParseError",
    "SearchProviderError",
    "SearXNGSearchProvider",
    "DuckDuckGoSearchProvider",
    "TavilySearchProvider",
    "ExaSearchProvider",
    "clean_and_canonicalize_url",
    "normalize_url",
    "extract_github_repo",
    "extract_github_repo_url",
    "extract_all_github_urls",
    "MultiSearchAggregator",
]
