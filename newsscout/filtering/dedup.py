"""newsscout.filtering.dedup
~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-Stage-1 Content Deduplication Engine for NewsScout.
Provides lightweight heuristic text similarity (Token Jaccard + 2-shingling)
operating in <100ms for 50 items on Raspberry Pi 5 without external vector DBs.
Protects LLM token budget by merging syndicated press releases, agency wire copies,
and reworded headlines before Stage 1 / Stage 2 evaluation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import re
from typing import Any, Final, Optional, Sequence, Set, Tuple

from newsscout.search.normalizer import clean_and_canonicalize_url
from newsscout.storage.db import Database
from newsscout.storage.models import RawItem

logger = logging.getLogger(__name__)

# Stop words to exclude from text similarity calculation
STOPWORDS: Final[Set[str]] = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "via",
    "new",
}

# Prefix stripping for common tech aggregator prefixes
PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:show\s+hn|ask\s+hn|tell\s+hn|release|announcing|breaking)\s*[:\-]\s*|\[[^\]]+\]\s*[:\-]?\s*)",
    re.IGNORECASE,
)

# Token extraction pattern supporting alphanumeric tokens with hyphens and periods (e.g. gpt-5, v0.6.1, 405b)
TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*\b")


def _light_stem(word: str) -> str:
    """Deterministic light suffix normalization for English tech headlines.

    Preserves numbers, semver tokens, and model sizes (e.g. 405b, 70b, 3.1).
    Only stems purely alphabetic words with length > 3.
    """
    if not word.isalpha() or len(word) <= 3:
        return word

    # -ies -> -y (capabilities -> capability)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # -sses -> -ss (processes -> process)
    if word.endswith("sses"):
        return word[:-2]

    # -ing (releasing -> releas, serving -> serv)
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        if stem.endswith(("e", "at", "iz", "is")):
            return stem
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in ("s", "l", "z"):
            stem = stem[:-1]
        return stem

    # -ed (released -> releas, created -> creat)
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith("e"):
            stem = stem[:-1]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in ("s", "l", "z"):
            stem = stem[:-1]
        return stem

    # -es (releases -> releas, fixes -> fix)
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("shes", "ches", "xes", "zes")):
            return word[:-2]
        if word.endswith("ses"):
            return word[:-2]
        return word[:-1]

    # trailing -s (models -> model, kernels -> kernel, but not process, status, analysis)
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]

    return word


def normalize_text(text: str) -> str:
    """Cleans and standardizes headline or lead teaser text."""
    if not text or not isinstance(text, str):
        return ""
    # Strip common prefixes (e.g. "Show HN:", "[D]", "Release:")
    text = PREFIX_RE.sub("", text.strip())
    return text.lower().strip()


def extract_tokens(text: str) -> set[str]:
    """Extracts normalized and stemmed tokens, preserving semver and model sizes."""
    cleaned = normalize_text(text)
    if not cleaned:
        return set()

    tokens: set[str] = set()
    for match in TOKEN_RE.finditer(cleaned):
        raw_token = match.group(0).strip(".-")
        if not raw_token or raw_token in STOPWORDS:
            continue

        # Strip leading 'v' before digit in version strings (e.g. v0.6.1 -> 0.6.1)
        if raw_token.startswith("v") and len(raw_token) > 1 and raw_token[1].isdigit():
            token = raw_token[1:]
        else:
            token = raw_token

        if not token or token in STOPWORDS:
            continue

        stemmed = _light_stem(token)
        tokens.add(stemmed)

    return tokens


def extract_shingles(text: str, n: int = 2) -> set[tuple[str, ...]]:
    """Extracts word n-shingles from text."""
    cleaned = normalize_text(text)
    if not cleaned:
        return set()

    words: list[str] = []
    for match in TOKEN_RE.finditer(cleaned):
        raw_token = match.group(0).strip(".-")
        if not raw_token or raw_token in STOPWORDS:
            continue
        if raw_token.startswith("v") and len(raw_token) > 1 and raw_token[1].isdigit():
            token = raw_token[1:]
        else:
            token = raw_token
        if not token or token in STOPWORDS:
            continue
        words.append(_light_stem(token))

    if len(words) < n:
        return {tuple(words)} if words else set()

    shingles: set[tuple[str, ...]] = set()
    for i in range(len(words) - n + 1):
        shingles.add(tuple(words[i : i + n]))
    return shingles


def token_jaccard(s1: set[Any], s2: set[Any]) -> float:
    """Calculates Jaccard similarity coefficient |s1 & s2| / |s1 | s2|."""
    if not s1 or not s2:
        return 1.0 if not s1 and not s2 else 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0.0


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculates text similarity combining Token Jaccard and 2-shingling over normalized text.

    Preserves semantic versions and model parameters while detecting reworded headlines.
    """
    if not text1 or not text2:
        return 1.0 if not text1 and not text2 else 0.0

    t1 = extract_tokens(text1)
    t2 = extract_tokens(text2)

    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0

    token_sim = token_jaccard(t1, t2)

    # If token similarity is 1.0 (exact match or word reordering), return immediately
    if token_sim >= 0.999:
        return 1.0

    sh1 = extract_shingles(text1, n=2)
    sh2 = extract_shingles(text2, n=2)

    if not sh1 or not sh2:
        return token_sim

    shingle_sim = token_jaccard(sh1, sh2)

    # Weight token_sim higher (0.70) than shingle_sim (0.30) to tolerate word order rephrasing
    combined = 0.70 * token_sim + 0.30 * shingle_sim
    if token_sim >= 0.80:
        return max(token_sim, combined)
    return combined


def is_near_duplicate(
    title1: str,
    text1: str,
    title2: str,
    text2: str,
    threshold: float = 0.80,
) -> tuple[bool, float]:
    """Determines whether two items are near-duplicates using title and teaser text.

    Returns:
        tuple of (is_duplicate: bool, similarity_score: float)
    """
    # 1. Compare titles first (primary signal)
    title_sim = calculate_text_similarity(title1, title2)
    if title_sim >= threshold:
        return True, title_sim

    t1_clean = (text1 or "").strip()
    t2_clean = (text2 or "").strip()

    if t1_clean and t2_clean:
        body_sim = calculate_text_similarity(t1_clean, t2_clean)

        # 2. Borderline title similarity (>= 0.50) with supporting body
        if title_sim >= 0.50:
            combined_sim = 0.65 * title_sim + 0.35 * body_sim
            if combined_sim >= threshold:
                return True, combined_sim

        # 3. High body similarity (e.g. syndicated wire copy / verbatim press release)
        # where editors reworded the title, but title shares subject context (>= 0.20)
        if body_sim >= 0.85 and title_sim >= 0.20:
            combined_sim = 0.25 * title_sim + 0.75 * body_sim
            if combined_sim >= threshold:
                return True, combined_sim

        return False, max(title_sim, 0.5 * title_sim + 0.5 * body_sim)

    return False, title_sim


def pick_primary_item(a: RawItem, b: RawItem) -> tuple[RawItem, RawItem]:
    """Picks which RawItem should be primary based on signal quality heuristics.

    Returns:
        (primary, duplicate)
    Preference hierarchy:
    1. Item with verified GitHub repository in metadata
    2. Longer content / description
    3. Earlier ingested_at timestamp
    """
    a_meta = a.metadata if isinstance(a.metadata, dict) else {}
    b_meta = b.metadata if isinstance(b.metadata, dict) else {}

    a_has_repo = bool(a_meta.get("github_repo"))
    b_has_repo = bool(b_meta.get("github_repo"))
    if a_has_repo != b_has_repo:
        return (a, b) if a_has_repo else (b, a)

    a_len = len(a.raw_content or "")
    b_len = len(b.raw_content or "")
    if a_len != b_len:
        return (a, b) if a_len > b_len else (b, a)

    if a.ingested_at and b.ingested_at:
        return (a, b) if a.ingested_at <= b.ingested_at else (b, a)

    return (a, b)


def merge_attributions(primary: RawItem, duplicate: RawItem) -> None:
    """Merges source attributions and alternate URLs from duplicate into primary RawItem."""
    if not isinstance(primary.metadata, dict):
        primary.metadata = {}
    if not isinstance(duplicate.metadata, dict):
        duplicate.metadata = {}

    # Initialize primary sources list if missing
    sources = primary.metadata.setdefault(
        "sources",
        [
            {
                "source": primary.source,
                "source_id": primary.source_id,
                "url": primary.url,
                "title": primary.title,
            }
        ],
    )

    # Gather duplicate sources
    dup_sources = duplicate.metadata.get("sources")
    if isinstance(dup_sources, list) and dup_sources:
        for s in dup_sources:
            if isinstance(s, dict) and not any(
                existing.get("source") == s.get("source")
                and existing.get("source_id") == s.get("source_id")
                for existing in sources
            ):
                sources.append(s)
    else:
        dup_source_entry = {
            "source": duplicate.source,
            "source_id": duplicate.source_id,
            "url": duplicate.url,
            "title": duplicate.title,
        }
        if not any(
            existing.get("source") == duplicate.source
            and existing.get("source_id") == duplicate.source_id
            for existing in sources
        ):
            sources.append(dup_source_entry)

    # Merge alternate URLs
    alt_urls = primary.metadata.setdefault("alternate_urls", [])
    primary_clean_url = clean_and_canonicalize_url(primary.url)

    urls_to_consider = [duplicate.url]
    dup_alt_urls = duplicate.metadata.get("alternate_urls")
    if isinstance(dup_alt_urls, list):
        urls_to_consider.extend(dup_alt_urls)

    for u in urls_to_consider:
        if not u or not isinstance(u, str):
            continue
        clean_u = clean_and_canonicalize_url(u)
        if u != primary.url and u not in alt_urls:
            alt_urls.append(u)
        if clean_u and clean_u != primary_clean_url and clean_u not in alt_urls:
            alt_urls.append(clean_u)

    # Preserve GitHub repo reference if duplicate has one and primary lacks one
    if duplicate.metadata.get("github_repo") and not primary.metadata.get("github_repo"):
        primary.metadata["github_repo"] = duplicate.metadata["github_repo"]

    # Increment duplicate count tally
    dup_tally = duplicate.metadata.get("duplicate_count", 1)
    primary.metadata["duplicate_count"] = (
        primary.metadata.get("duplicate_count", 1) + dup_tally
    )


def deduplicate_items(items: list[RawItem], threshold: float = 0.80) -> list[RawItem]:
    """Performs intra-batch deduplication on a list of RawItems.

    Detects exact canonical URL duplicates and content-level near-duplicates.
    Merges source attributions into the surviving primary item's metadata.
    Guarantees execution in <100ms for 50 items on Raspberry Pi 5.
    """
    if not items:
        return []
    if len(items) == 1:
        return [items[0]]

    # Pre-extract cleaned URLs, tokens, and raw text once per item for O(N) extraction overhead
    extracted_data = []
    for item in items:
        clean_url = clean_and_canonicalize_url(item.url)
        raw_text = (
            item.metadata.get("snippet")
            or item.metadata.get("teaser")
            or item.raw_content
            or ""
        )
        t_tokens = extract_tokens(item.title)
        b_tokens = extract_tokens(raw_text) if raw_text else set()
        extracted_data.append((item, clean_url, t_tokens, b_tokens, raw_text))

    surviving_items: list[RawItem] = []
    surviving_data: list[tuple[RawItem, str, set[str], set[str], str]] = []

    for incoming, inc_url, inc_t_tokens, inc_b_tokens, inc_raw_text in extracted_data:
        matched_idx = -1

        for idx, (existing, ex_url, ex_t_tokens, ex_b_tokens, ex_raw_text) in enumerate(surviving_data):
            ex_alt_urls = existing.metadata.get("alternate_urls", [])

            # Step 1: URL matching (exact canonical URL match or match against merged alternate URLs)
            url_matched = False
            if inc_url and (
                inc_url == ex_url
                or inc_url in ex_alt_urls
                or incoming.url in ex_alt_urls
            ):
                url_matched = True

            title_token_sim = token_jaccard(ex_t_tokens, inc_t_tokens)

            # Guard: If canonical URLs matched because both point to the same root repository (e.g. GitHub releases),
            # verify that the titles are not distinct models / releases (< threshold title similarity)
            if url_matched:
                if title_token_sim < threshold:
                    url_matched = False

            if url_matched:
                matched_idx = idx
                break

            # Step 2: Fast title token similarity match
            if title_token_sim >= threshold:
                matched_idx = idx
                break

            # Fast body token similarity check
            if inc_b_tokens and ex_b_tokens:
                body_token_sim = token_jaccard(ex_b_tokens, inc_b_tokens)
                if title_token_sim >= 0.50:
                    combined = 0.65 * title_token_sim + 0.35 * body_token_sim
                    if combined >= threshold:
                        matched_idx = idx
                        break
                if body_token_sim >= 0.85 and title_token_sim >= 0.20:
                    combined = 0.25 * title_token_sim + 0.75 * body_token_sim
                    if combined >= threshold:
                        matched_idx = idx
                        break

            # Fallback to full is_near_duplicate if borderline
            if 0.50 <= title_token_sim < threshold:
                is_dup, _ = is_near_duplicate(
                    existing.title,
                    ex_raw_text,
                    incoming.title,
                    inc_raw_text,
                    threshold=threshold,
                )
                if is_dup:
                    matched_idx = idx
                    break

        if matched_idx >= 0:
            existing, ex_url, ex_t_tokens, ex_b_tokens, ex_raw_text = surviving_data[matched_idx]
            primary, duplicate = pick_primary_item(existing, incoming)
            merge_attributions(primary, duplicate)
            if primary is incoming:
                surviving_items[matched_idx] = incoming
                surviving_data[matched_idx] = (incoming, inc_url, inc_t_tokens, inc_b_tokens, inc_raw_text)
            else:
                surviving_items[matched_idx] = existing
        else:
            surviving_items.append(incoming)
            surviving_data.append((incoming, inc_url, inc_t_tokens, inc_b_tokens, inc_raw_text))

    return surviving_items


async def deduplicate_against_db(
    items: Sequence[RawItem],
    db: Database,
    lookback_days: int = 7,
    threshold: float = 0.80,
) -> list[RawItem]:
    """Deduplicates incoming items against recently persisted raw_items in SQLite.

    For each incoming item:
    - If it matches an existing item in SQLite (by canonical URL or content similarity >= threshold),
      updates the existing item's metadata in the database with merged attribution,
      and drops the incoming item from the batch so no Stage 1/2 calls occur.
    - If genuinely unique, keeps the item in the surviving list.

    Returns:
        List of genuinely unique RawItems to be persisted.
    """
    if not items:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

    # Keyset pagination: fetch raw_items in chunks of 500 ordered by ingested_at DESC, id DESC
    # to avoid loading the entire table into memory on the Raspberry Pi.
    PAGE_SIZE = 500
    existing_items: list[RawItem] = []
    last_ingested: str | None = None
    last_id: int | None = None

    while True:
        if last_ingested is None:
            query = """
                SELECT id, source, source_id, title, url, raw_content, metadata_json, ingested_at
                FROM raw_items
                WHERE ingested_at >= :cutoff
                ORDER BY ingested_at DESC, id DESC
                LIMIT :limit;
            """
            params: dict[str, Any] = {"cutoff": cutoff, "limit": PAGE_SIZE}
        else:
            query = """
                SELECT id, source, source_id, title, url, raw_content, metadata_json, ingested_at
                FROM raw_items
                WHERE ingested_at >= :cutoff
                  AND (ingested_at < :last_ingested OR (ingested_at = :last_ingested AND id < :last_id))
                ORDER BY ingested_at DESC, id DESC
                LIMIT :limit;
            """
            params = {
                "cutoff": cutoff,
                "last_ingested": last_ingested,
                "last_id": last_id,
                "limit": PAGE_SIZE,
            }

        rows = await db.fetch_all(query, params)
        if not rows:
            break

        for r in rows:
            try:
                existing_items.append(RawItem.from_row(r))
            except Exception:
                continue

        if len(rows) < PAGE_SIZE:
            break

        last_row = rows[-1]
        last_ingested = last_row["ingested_at"]
        last_id = last_row["id"]

    if not existing_items:
        return list(items)

    # Pre-index existing items by canonical URL for O(1) exact lookups
    url_to_existing: dict[str, RawItem] = {}
    for it in existing_items:
        clean_url = clean_and_canonicalize_url(it.url)
        if clean_url:
            url_to_existing[clean_url] = it

    # Pre-extract tokens for all existing items once (avoid re-extraction per comparison)
    existing_data: list[tuple[RawItem, set[str], set[str], str]] = []
    for it in existing_items:
        ex_text = (
            it.metadata.get("snippet")
            or it.metadata.get("teaser")
            or it.raw_content
            or ""
        )
        existing_data.append((it, extract_tokens(it.title), extract_tokens(ex_text) if ex_text else set(), ex_text))

    surviving: list[RawItem] = []
    pending_updates: list[dict[str, Any]] = []

    for incoming in items:
        incoming_clean_url = clean_and_canonicalize_url(incoming.url)
        incoming_text = (
            incoming.metadata.get("snippet")
            or incoming.metadata.get("teaser")
            or incoming.raw_content
            or ""
        )
        incoming_t_tokens = extract_tokens(incoming.title)
        incoming_b_tokens = extract_tokens(incoming_text) if incoming_text else set()

        matched_existing: Optional[RawItem] = None

        # Check 1: Exact canonical URL match
        if incoming_clean_url and incoming_clean_url in url_to_existing:
            candidate = url_to_existing[incoming_clean_url]
            if incoming.url == candidate.url or token_jaccard(extract_tokens(candidate.title), incoming_t_tokens) >= threshold:
                matched_existing = candidate
        else:
            # Check 2: Fast 3-stage token filter (no expensive shingle extraction)
            for existing, ex_t_tokens, ex_b_tokens, ex_text in existing_data:
                title_token_sim = token_jaccard(ex_t_tokens, incoming_t_tokens)
                # Stage A: Title tokens alone exceed threshold → match
                if title_token_sim >= threshold:
                    matched_existing = existing
                    break
                # Stage B: Borderline title + body tokens combined → match
                if title_token_sim >= 0.50 and ex_b_tokens and incoming_b_tokens:
                    body_token_sim = token_jaccard(ex_b_tokens, incoming_b_tokens)
                    combined = 0.65 * title_token_sim + 0.35 * body_token_sim
                    if combined >= threshold:
                        matched_existing = existing
                        break
                    # Stage C: High body similarity with low title context → match
                    if body_token_sim >= 0.85 and title_token_sim >= 0.20:
                        combined = 0.25 * title_token_sim + 0.75 * body_token_sim
                        if combined >= threshold:
                            matched_existing = existing
                            break

        if matched_existing is not None:
            # Merge incoming attribution into matched DB record
            merge_attributions(matched_existing, incoming)
            pending_updates.append({
                "metadata_json": json.dumps(matched_existing.metadata),
                "id": matched_existing.id,
            })
            logger.info(
                "[dedup] Discarded near-duplicate '%s' (matches DB id=%s, source=%s)",
                incoming.title,
                matched_existing.id,
                matched_existing.source,
            )
        else:
            surviving.append(incoming)

    # Batch all metadata updates in a single DB call
    if pending_updates:
        await db.execute_many(
            "UPDATE raw_items SET metadata_json = :metadata_json WHERE id = :id;",
            pending_updates,
        )

    return surviving
