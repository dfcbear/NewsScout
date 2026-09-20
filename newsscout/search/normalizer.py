"""newsscout.search.normalizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
URL sanitization, tracking parameter removal, anchor/fragment stripping,
and canonical GitHub URL standardizer for multi-engine search results.
"""

from __future__ import annotations

import re
from typing import Final, Optional, Set
from urllib.parse import parse_qsl, unquote, unquote_plus, urlencode, urlparse, urlunparse

TRACKING_PARAMS: Final[Set[str]] = {
    # --- UTM & Core Campaign Tags ---
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_reader",
    "utm_name",
    "utm_cid",
    "utm_term_id",
    # --- Ad Network Click Identifiers ---
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "yclid",
    "twclid",
    "wbraid",
    "gbraid",
    "wickedid",
    "rb_clickid",
    "zanpid",
    "irclickid",
    "ttclid",
    "li_fat_id",
    "s_cid",
    "sc_cid",
    "gclsrc",
    "ef_id",
    # --- Referral & Social Platform Tags ---
    "ref",
    "ref_src",
    "ref_url",
    "source",
    "spm",
    "feature",
    "trk",
    "ocid",
    "igshid",
    "origin",
    "partner",
    "affiliate_id",
    "aff_platform",
    "aff_trace_key",
    "subid",
    "subid1",
    "subid2",
    "subid3",
    "subid4",
    "subid5",
    "curator",
    # --- Marketing & Email Automation ---
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "vero_id",
    "vero_conv",
    "mkt_tok",
    "ml_subscriber",
    "ml_subscriber_hash",
    "_openstat",
    "nr_email_referer",
    "cmpid",
    "mbid",
    "ncid",
    "ftag",
    "taid",
    # --- Syndication & RSS Feeds ---
    "rss",
    "feed",
    "feedname",
    "feedtype",
    "sr_share",
    "xtor",
    "xtref",
    # --- Privacy Consent / Redirect Bounces ---
    "guccounter",
    "guce_referrer",
    "guce_referrer_usqp",
    # --- Analytics & Session Tags ---
    "_ga",
    "_gl",
    "s_kwcid",
}

# Prefix-based tracking parameter detection (12 prefix families + HubSpot & GA4 variants)
TRACKING_PREFIXES: Final[tuple[str, ...]] = (
    "utm_",     # Google Analytics & general campaign tags
    "pk_",      # Piwik / Matomo analytics
    "piwik_",   # Piwik legacy
    "matomo_",  # Matomo campaign & keyword tags
    "hsa_",     # HubSpot ad tracking
    "_hs",      # HubSpot email & security tracking
    "mc_",      # Mailchimp campaign tracking
    "sc_",      # Snapchat / Adobe campaign tracking
    "nd_",      # News syndication delivery tags
    "aff_",     # Affiliate network tracking
    "wt_",      # Webtrekk analytics
    "ga_",      # Google Analytics 4 tags
    "_ga_",     # Google Analytics 4 cross-domain session tags
    "xtor",     # AT Internet / Piano analytics
)


def is_tracking_param(param_name: str) -> bool:
    """Checks if a query parameter is a tracking, syndication, or analytics tag."""
    if not param_name or not isinstance(param_name, str):
        return False
    p = param_name.lower()
    if p in TRACKING_PARAMS:
        return True
    return any(p.startswith(prefix) for prefix in TRACKING_PREFIXES)


GITHUB_RESERVED_ROOTS: Final[Set[str]] = {
    "about",
    "collections",
    "contact",
    "events",
    "explore",
    "features",
    "issues",
    "join",
    "login",
    "marketplace",
    "notifications",
    "organizations",
    "orgs",
    "pricing",
    "pulls",
    "readme",
    "security",
    "settings",
    "site",
    "sponsors",
    "topics",
    "trending",
}

GITHUB_REPO_SUBPATHS: Final[Set[str]] = {
    "archive",
    "blob",
    "branches",
    "commits",
    "network",
    "releases",
    "stargazers",
    "tags",
    "tree",
    "watchers",
}

GITHUB_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9][a-zA-Z0-9-]*)/([a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)

BARE_GITHUB_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^([a-zA-Z0-9][a-zA-Z0-9-]*)/([a-zA-Z0-9_.-]+)$"
)


def _unwrap_search_redirect(url: str) -> str:
    """Unwraps search engine redirection URLs (e.g. DuckDuckGo uddg).

    Handles:
    - ``duckduckgo.com/l/?uddg=...`` (standard redirect)
    - ``r.duckduckgo.com/...?uddg=...`` (y.js redirect format)
    - URL-encoded ``+`` in the ``uddg`` parameter
    """
    if "duckduckgo.com/l/?" in url or "r.duckduckgo.com/" in url:
        try:
            parsed = urlparse(url)
            params = dict(parse_qsl(parsed.query))
            if "uddg" in params and params["uddg"]:
                return unquote_plus(params["uddg"])
        except Exception:
            pass
    return url


def clean_and_canonicalize_url(url: str) -> str:
    """Sanitizes and normalizes a URL to a clean canonical format.

    - Strips search engine redirect wrappers (e.g. DuckDuckGo uddg).
    - Trims whitespace and normalizes empty inputs.
    - Adds default 'https://' if scheme is omitted.
    - Lowercases scheme and netloc; strips 'www.'.
    - Strips default ports (:80 on http, :443 on https).
    - Normalizes path slashes (collapses '//', strips trailing slashes).
    - Canonicalizes GitHub repository paths (strips .git, branch/tree/blob/release subpaths).
    - Strips UTM and tracking parameters, preserving and sorting legitimate query params.
    - Strips URL fragments/anchors unconditionally.
    """
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()
    if not url:
        return ""

    # Unwrap search engine redirects
    url = _unwrap_search_redirect(url)

    # Check for bare GitHub repository slug (e.g. "vllm-project/vllm")
    bare_match = BARE_GITHUB_SLUG_PATTERN.match(url)
    if bare_match:
        owner, repo = bare_match.groups()
        if owner.lower() not in GITHUB_RESERVED_ROOTS:
            repo_clean = repo.removesuffix(".git").rstrip("/")
            return f"https://github.com/{owner}/{repo_clean}"

    # Detect existing scheme before prepending https://
    # Reject dangerous non-http(s) schemes early (javascript:, data:, file:, mailto:, etc.)
    if "://" in url:
        try:
            early_parsed = urlparse(url)
            if early_parsed.scheme and early_parsed.scheme.lower() not in ("http", "https"):
                return ""
        except Exception:
            return ""
    elif ":" in url.split("/")[0]:
        # URL has a scheme but no :// (e.g. javascript:alert(1), mailto:user@example.com)
        potential_scheme = url.split(":")[0].lower()
        if potential_scheme not in ("http", "https"):
            return ""

    # Handle protocol-relative URL
    if url.startswith("//"):
        url = "https:" + url
    elif "://" not in url:
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower() if parsed.netloc else ""

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Strip 'www.' prefix
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Path normalization
    path = parsed.path or ""
    path = re.sub(r"/+", "/", path)

    # GitHub repository normalization
    if netloc in ("github.com", "api.github.com"):
        scheme = "https"
        netloc = "github.com"
        segments = [s for s in path.strip("/").split("/") if s]
        if len(segments) >= 2:
            owner = segments[0]
            repo = segments[1].removesuffix(".git")
            if owner.lower() not in GITHUB_RESERVED_ROOTS:
                if len(segments) == 2 or (len(segments) > 2 and segments[2].lower() in GITHUB_REPO_SUBPATHS):
                    path = f"/{owner}/{repo}"
                else:
                    path = f"/{owner}/{repo}" + "/" + "/".join(segments[2:])

    # Strip trailing slash
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    elif path == "/":
        path = ""

    # Query parameters cleanup
    clean_query = ""
    if parsed.query:
        raw_params = parse_qsl(parsed.query, keep_blank_values=False)
        kept_params = [
            (k, v)
            for k, v in raw_params
            if not is_tracking_param(k)
        ]
        if kept_params:
            kept_params.sort(key=lambda item: item[0])
            clean_query = urlencode(kept_params)

    # Strip fragment completely
    return urlunparse((scheme, netloc, path, "", clean_query, ""))


# Alias for backward/convenience compatibility
normalize_url = clean_and_canonicalize_url


def extract_github_repo(url_or_slug: str) -> Optional[str]:
    """Extracts 'owner/repo' from a URL or slug if it refers to a valid GitHub repository."""
    if not url_or_slug or not isinstance(url_or_slug, str):
        return None

    clean = clean_and_canonicalize_url(url_or_slug)
    parsed = urlparse(clean)
    if parsed.netloc != "github.com":
        return None

    segments = [s for s in parsed.path.strip("/").split("/") if s]
    if len(segments) >= 2:
        owner, repo = segments[0], segments[1]
        if owner.lower() not in GITHUB_RESERVED_ROOTS:
            return f"{owner}/{repo}"
    return None


def extract_github_repo_url(url: str, text: str = "") -> Optional[str]:
    """Extracts canonical 'https://github.com/owner/repo' URL from a URL or fallback text."""
    repo = extract_github_repo(url)
    if repo:
        return f"https://github.com/{repo}"

    if text:
        all_urls = extract_all_github_urls(text)
        if all_urls:
            return all_urls[0]

    return None


def extract_all_github_urls(text: str) -> list[str]:
    """Extracts and canonicalizes all distinct GitHub repository URLs from text."""
    if not text or not isinstance(text, str):
        return []

    seen: set[str] = set()
    results: list[str] = []

    for match in GITHUB_URL_PATTERN.finditer(text):
        owner, repo = match.group(1), match.group(2)
        if owner.lower() in GITHUB_RESERVED_ROOTS:
            continue
        clean_repo = repo.rstrip(r".,!?:;)'\"\]}").removesuffix(".git").rstrip("/.")
        if not clean_repo:
            continue
        canonical = clean_and_canonicalize_url(f"https://github.com/{owner}/{clean_repo}")
        if canonical and canonical not in seen:
            seen.add(canonical)
            results.append(canonical)

    return results
