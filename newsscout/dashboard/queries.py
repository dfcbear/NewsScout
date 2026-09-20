"""newsscout.dashboard.queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async database query helpers for the NewsScout web dashboard.

Provides filtered breakthrough listings, feedback statistics, digest history,
and web-based feedback recording — all built on the existing SQLite WAL Database.
"""

from __future__ import annotations

import json
from typing import Any

from newsscout.storage.db import Database
from newsscout.storage.models import FeedbackRating
from newsscout.storage.preferences import PreferencesService


async def list_breakthroughs(
    db: Database,
    *,
    category: str | None = None,
    min_score: float | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Returns filtered breakthrough rows for the dashboard table."""
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)

    where_parts: list[str] = []
    params: list[Any] = []

    if category:
        where_parts.append("b.category = ?")
        params.append(category)

    if min_score is not None:
        where_parts.append("b.breakthrough_score >= ?")
        params.append(float(min_score))

    if search:
        where_parts.append("(LOWER(b.title) LIKE ? OR LOWER(b.tldr) LIKE ?)")
        term = f"%{search.lower()}%"
        params.extend([term, term])

    where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    query = f"""
        SELECT b.id, b.title, b.repo_url, b.breakthrough_score, b.roi_score,
               b.category, b.tldr, b.license, b.is_watchlisted, b.evaluated_at
        FROM breakthroughs b
        {where_clause}
        ORDER BY b.breakthrough_score DESC, b.roi_score DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    rows = await db.fetch_all(query, tuple(params))
    return [dict(r) for r in rows] if rows else []


async def get_breakthrough_detail(db: Database, breakthrough_id: int) -> dict[str, Any] | None:
    """Returns full detail for a single breakthrough, including feedback counts."""
    row = await db.fetch_one(
        "SELECT * FROM breakthroughs WHERE id = ?",
        (breakthrough_id,),
    )
    if row is None:
        return None

    result = dict(row)

    feedback_rows = await db.fetch_all(
        "SELECT rating, COUNT(*) as count FROM feedback WHERE breakthrough_id = ? GROUP BY rating",
        (breakthrough_id,),
    )
    feedback_summary: dict[str, int] = {}
    if feedback_rows:
        for fr in feedback_rows:
            feedback_summary[fr["rating"]] = fr["count"]
    result["feedback_summary"] = feedback_summary

    return result


async def get_feedback_stats(db: Database) -> dict[str, Any]:
    """Returns aggregate feedback statistics for the dashboard overview."""
    total_breakthroughs = await db.fetch_val(
        "SELECT COUNT(*) FROM breakthroughs WHERE category != 'discard'"
    )
    total_feedback = await db.fetch_val("SELECT COUNT(*) FROM feedback")

    rating_rows = await db.fetch_all(
        "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating ORDER BY count DESC"
    )
    rating_counts: dict[str, int] = {}
    if rating_rows:
        for r in rating_rows:
            rating_counts[r["rating"]] = r["count"]

    avg_scores = await db.fetch_one(
        """SELECT AVG(breakthrough_score) as avg_breakthrough, AVG(roi_score) as avg_roi
           FROM breakthroughs WHERE category != 'discard'"""
    )
    avg_breakthrough = avg_scores["avg_breakthrough"] if avg_scores and avg_scores["avg_breakthrough"] else 0.0
    avg_roi = avg_scores["avg_roi"] if avg_scores and avg_scores["avg_roi"] else 0.0

    category_rows = await db.fetch_all(
        "SELECT category, COUNT(*) as count FROM breakthroughs GROUP BY category"
    )
    category_counts: dict[str, int] = {}
    if category_rows:
        for cr in category_rows:
            category_counts[cr["category"]] = cr["count"]

    return {
        "total_breakthroughs": total_breakthroughs or 0,
        "total_feedback": total_feedback or 0,
        "rating_counts": rating_counts,
        "avg_breakthrough_score": round(avg_breakthrough, 2),
        "avg_roi_score": round(avg_roi, 2),
        "category_counts": category_counts,
    }


async def list_digests(db: Database, limit: int = 20) -> list[dict[str, Any]]:
    """Returns recent digest records with track counts."""
    limit = min(max(limit, 1), 100)
    rows = await db.fetch_all(
        """SELECT d.id, d.slot, d.digest_date, d.title, d.total_duration_seconds,
                  d.status, d.delivered_at, d.merged_audio_path,
                  (SELECT COUNT(*) FROM digest_tracks WHERE digest_id = d.id) as track_count
           FROM digests d
           ORDER BY d.digest_date DESC, d.id DESC
           LIMIT ?""",
        (limit,),
    )
    return [dict(r) for r in rows] if rows else []


async def record_web_feedback(
    db: Database,
    preferences: PreferencesService,
    breakthrough_id: int,
    rating: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Records feedback submitted from the web dashboard."""
    valid_ratings = {r.value for r in FeedbackRating}
    if rating not in valid_ratings:
        return {"success": False, "error": f"Invalid rating '{rating}'. Must be one of {valid_ratings}"}

    bt = await db.fetch_one("SELECT id FROM breakthroughs WHERE id = ?", (breakthrough_id,))
    if bt is None:
        return {"success": False, "error": f"Breakthrough {breakthrough_id} not found"}

    feedback = await preferences.record_feedback(
        breakthrough_id=breakthrough_id,
        rating=rating,
        user_identifier="web",
        source="web",
        notes=notes,
    )
    return {"success": True, "feedback_id": feedback.id, "rating": rating}


async def get_recent_breakthroughs(db: Database, limit: int = 10) -> list[dict[str, Any]]:
    """Returns the most recently evaluated breakthroughs for the dashboard feed."""
    limit = min(max(limit, 1), 50)
    rows = await db.fetch_all(
        """SELECT id, title, breakthrough_score, roi_score, category, tldr, evaluated_at
           FROM breakthroughs WHERE category != 'discard'
           ORDER BY evaluated_at DESC LIMIT ?""",
        (limit,),
    )
    return [dict(r) for r in rows] if rows else []


async def get_watchlist(db: Database) -> list[dict[str, Any]]:
    """Returns all watchlisted breakthroughs."""
    rows = await db.fetch_all(
        """SELECT id, title, repo_url, breakthrough_score, roi_score, category, tldr
           FROM breakthroughs WHERE is_watchlisted = 1
           ORDER BY breakthrough_score DESC"""
    )
    return [dict(r) for r in rows] if rows else []


async def toggle_watchlist(db: Database, breakthrough_id: int) -> dict[str, Any]:
    """Toggles the watchlist status of a breakthrough."""
    row = await db.fetch_one(
        "SELECT is_watchlisted FROM breakthroughs WHERE id = ?", (breakthrough_id,)
    )
    if row is None:
        return {"success": False, "error": "Breakthrough not found"}

    new_val = 0 if row["is_watchlisted"] else 1
    await db.execute(
        "UPDATE breakthroughs SET is_watchlisted = ? WHERE id = ?",
        (new_val, breakthrough_id),
    )
    return {"success": True, "is_watchlisted": bool(new_val)}


async def get_topic_radar(db: Database) -> dict[str, Any]:
    """Returns active sources, tracked GitHub repos, and interest keywords."""
    rows = await db.fetch_all("SELECT * FROM sources ORDER BY id ASC")
    sources = []
    github_repos: list[str] = []
    custom_keywords: list[str] = []

    for r in rows:
        cfg = json.loads(r["config_json"]) if r["config_json"] else {}
        sources.append({
            "id": r["id"],
            "name": r["name"],
            "source_type": r["source_type"],
            "is_active": bool(r["is_active"]),
            "last_poll_at": r["last_poll_at"],
            "config": cfg,
        })
        if r["name"] == "github_releases":
            github_repos = list(cfg.get("repos", []))
        elif r["name"] == "hacker_news":
            custom_keywords = list(cfg.get("custom_keywords", []))

    return {
        "sources": sources,
        "tracked_repos": github_repos,
        "custom_keywords": custom_keywords,
    }


async def add_tracked_repo(db: Database, repo_slug: str) -> dict[str, Any]:
    """Adds a repository to github_releases config in sources table."""
    repo_slug = repo_slug.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    if not repo_slug or "/" not in repo_slug:
        return {"success": False, "error": "Invalid repo slug. Expected format: 'owner/repo'"}

    row = await db.fetch_one("SELECT id, config_json FROM sources WHERE name = 'github_releases'")
    if row is None:
        cfg = {"repos": [repo_slug]}
        await db.execute(
            "INSERT INTO sources (name, source_type, config_json, is_active) VALUES ('github_releases', 'api', ?, 1)",
            (json.dumps(cfg),),
        )
        return {"success": True, "repo": repo_slug, "repos": cfg["repos"]}

    cfg = json.loads(row["config_json"]) if row["config_json"] else {}
    repos = list(cfg.get("repos", []))
    if repo_slug not in repos:
        repos.append(repo_slug)
        cfg["repos"] = repos
        await db.execute(
            "UPDATE sources SET config_json = ? WHERE id = ?",
            (json.dumps(cfg), row["id"]),
        )
    return {"success": True, "repo": repo_slug, "repos": repos}


async def remove_tracked_repo(db: Database, repo_slug: str) -> dict[str, Any]:
    """Removes a repository from github_releases config in sources table."""
    repo_slug = repo_slug.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
    row = await db.fetch_one("SELECT id, config_json FROM sources WHERE name = 'github_releases'")
    if row is None:
        return {"success": False, "error": "Source github_releases not found"}

    cfg = json.loads(row["config_json"]) if row["config_json"] else {}
    repos = list(cfg.get("repos", []))
    if repo_slug in repos:
        repos.remove(repo_slug)
        cfg["repos"] = repos
        await db.execute(
            "UPDATE sources SET config_json = ? WHERE id = ?",
            (json.dumps(cfg), row["id"]),
        )
    return {"success": True, "repo": repo_slug, "repos": repos}


async def add_interest_keyword(db: Database, keyword: str) -> dict[str, Any]:
    """Adds a keyword to hacker_news custom_keywords config in sources table."""
    keyword = keyword.strip().lower()
    if not keyword:
        return {"success": False, "error": "Keyword cannot be empty"}

    row = await db.fetch_one("SELECT id, config_json FROM sources WHERE name = 'hacker_news'")
    if row is None:
        cfg = {"min_score": 120, "min_ratio": 0.4, "custom_keywords": [keyword]}
        await db.execute(
            "INSERT INTO sources (name, source_type, config_json, is_active) VALUES ('hacker_news', 'firebase', ?, 1)",
            (json.dumps(cfg),),
        )
        return {"success": True, "keyword": keyword, "keywords": cfg["custom_keywords"]}

    cfg = json.loads(row["config_json"]) if row["config_json"] else {}
    keywords = list(cfg.get("custom_keywords", []))
    if keyword not in keywords:
        keywords.append(keyword)
        cfg["custom_keywords"] = keywords
        await db.execute(
            "UPDATE sources SET config_json = ? WHERE id = ?",
            (json.dumps(cfg), row["id"]),
        )
    return {"success": True, "keyword": keyword, "keywords": keywords}


async def remove_interest_keyword(db: Database, keyword: str) -> dict[str, Any]:
    """Removes a keyword from hacker_news custom_keywords config in sources table."""
    keyword = keyword.strip().lower()
    row = await db.fetch_one("SELECT id, config_json FROM sources WHERE name = 'hacker_news'")
    if row is None:
        return {"success": False, "error": "Source hacker_news not found"}

    cfg = json.loads(row["config_json"]) if row["config_json"] else {}
    keywords = list(cfg.get("custom_keywords", []))
    if keyword in keywords:
        keywords.remove(keyword)
        cfg["custom_keywords"] = keywords
        await db.execute(
            "UPDATE sources SET config_json = ? WHERE id = ?",
            (json.dumps(cfg), row["id"]),
        )
    return {"success": True, "keyword": keyword, "keywords": keywords}
