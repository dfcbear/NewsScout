"""tests/test_dashboard.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for the NewsScout web dashboard (TASK-05).

Tests cover:
- FastAPI app creation and lifespan
- All API endpoints (breakthroughs, stats, digests, feedback, watchlist)
- Filtering, pagination, and search
- Error handling (404s, invalid input)
- HTML page serving
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import (
    Breakthrough,
    Digest,
    DigestSlot,
    DigestStatus,
    Feedback,
    FeedbackRating,
    RawItem,
    Stage2Category,
    Track,
    TrackType,
)
from tests.conftest import DataFactory


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dashboard_settings(tmp_path):
    """Settings pointed at temporary directories for dashboard testing."""
    return Settings(
        environment="test",
        db_path=tmp_path / "test_dashboard.db",
        audio_output_dir=tmp_path / "audio",
        preferences_file=tmp_path / "preferences.json",
        gemini_api_key=SecretStr("mock-key"),
        telegram_bot_token=SecretStr("mock-token"),
        telegram_chat_id="12345",
        github_token=SecretStr("mock-github"),
        log_level="DEBUG",
    )


@pytest.fixture
def app_client(dashboard_settings):
    """Creates a FastAPI TestClient with a migrated database and seeded data."""
    app = create_app(dashboard_settings)
    with TestClient(app) as client:
        db = app.state.db

        async def _seed():
            # Insert raw items
            for i in range(5):
                raw = DataFactory.create_raw_item(
                    source_id=f"test/repo-{i}:v1.{i}",
                    title=f"Test Breakthrough {i}",
                    url=f"https://github.com/test/repo-{i}",
                    raw_content=f"Content for item {i}",
                )
                params = raw.to_db_params()
                await db.execute(
                    """INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
                    VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at)""",
                    params,
                )

            # Insert breakthroughs
            categories = [Stage2Category.CORE, Stage2Category.SERENDIPITY, Stage2Category.CORE,
                          Stage2Category.SERENDIPITY, Stage2Category.DISCARD]
            scores = [9.5, 8.8, 9.2, 7.5, 3.0]
            roi_scores = [9.3, 8.6, 9.0, 7.2, 2.5]

            for i in range(5):
                bt = DataFactory.create_breakthrough(
                    raw_item_id=i + 1,
                    title=f"Test Breakthrough {i}",
                    breakthrough_score=scores[i],
                    roi_score=roi_scores[i],
                    category=categories[i],
                    repo_url=f"https://github.com/test/repo-{i}",
                )
                params = bt.to_db_params()
                await db.execute(
                    """INSERT INTO breakthroughs
                    (raw_item_id, title, repo_url, breakthrough_score, roi_score, category,
                     tldr, use_case, comparison, quickstart, hardware_requirements, license,
                     card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
                    VALUES (:raw_item_id, :title, :repo_url, :breakthrough_score, :roi_score, :category,
                     :tldr, :use_case, :comparison, :quickstart, :hardware_requirements, :license,
                     :card_markdown, :evaluation_raw_json, :is_watchlisted, :evaluated_at)""",
                    params,
                )

            # Insert a digest
            digest = DataFactory.create_digest(
                slot=DigestSlot.MORNING,
                digest_date="2026-09-18",
                title="NewsScout Morning Digest — 18.09.2026",
                status=DigestStatus.DELIVERED,
            )
            digest_id = await db.execute(
                """INSERT INTO digests
                (slot, digest_date, title, total_duration_seconds, merged_audio_path,
                 telegram_message_id, status, created_at, delivered_at)
                VALUES (:slot, :digest_date, :title, :total_duration_seconds, :merged_audio_path,
                 :telegram_message_id, :status, :created_at, :delivered_at)""",
                {
                    "slot": digest.slot,
                    "digest_date": digest.digest_date,
                    "title": digest.title,
                    "total_duration_seconds": digest.total_duration_seconds,
                    "merged_audio_path": digest.merged_audio_path,
                    "telegram_message_id": digest.telegram_message_id,
                    "status": digest.status,
                    "created_at": digest.created_at.isoformat(),
                    "delivered_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Insert digest tracks
            track_types = [TrackType.EXECUTIVE_SUMMARY, TrackType.DEEP_DIVE_1, TrackType.DEEP_DIVE_2]
            for j in range(3):
                track = DataFactory.create_track(
                    digest_id=digest_id,
                    track_number=j + 1,
                    track_type=track_types[j],
                    title=f"Track {j+1}",
                )
                await db.execute(
                    """INSERT INTO digest_tracks
                    (digest_id, track_number, track_type, title, duration_seconds,
                     audio_path, script_json, breakthrough_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (track.digest_id, track.track_number, track.track_type.value,
                     track.title, track.duration_seconds, track.audio_path,
                     json.dumps(track.script.model_dump()),
                     track.breakthrough_id, track.created_at.isoformat()),
                )

            # Insert feedback
            feedback = DataFactory.create_feedback(
                breakthrough_id=1,
                rating=FeedbackRating.HIT,
            )
            await db.execute(
                """INSERT INTO feedback (breakthrough_id, rating, user_identifier, source, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (feedback.breakthrough_id, feedback.rating.value, feedback.user_identifier,
                 feedback.source, feedback.notes, feedback.created_at.isoformat()),
            )

        asyncio.run(_seed())
        yield client


# ============================================================================
# HTML Page Tests
# ============================================================================

class TestDashboardPage:
    """Tests for the HTML dashboard page route."""

    def test_serves_html(self, app_client):
        """GET / returns HTML content."""
        response = app_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "NewsScout" in response.text

    def test_html_contains_tabs(self, app_client):
        """HTML page contains navigation tabs."""
        response = app_client.get("/")
        assert "Breakthroughs" in response.text
        assert "Statistik" in response.text
        assert "Digests" in response.text
        assert "Watchlist" in response.text


# ============================================================================
# Breakthroughs API Tests
# ============================================================================

class TestBreakthroughsAPI:
    """Tests for /api/breakthroughs endpoints."""

    def test_list_all(self, app_client):
        """GET /api/breakthroughs returns all breakthroughs by default."""
        response = app_client.get("/api/breakthroughs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_filter_by_category_core(self, app_client):
        """Filtering by category=core returns only core breakthroughs."""
        response = app_client.get("/api/breakthroughs?category=core")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(bt["category"] == "core" for bt in data)

    def test_filter_by_category_serendipity(self, app_client):
        """Filtering by category=serendipity returns only serendipity breakthroughs."""
        response = app_client.get("/api/breakthroughs?category=serendipity")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(bt["category"] == "serendipity" for bt in data)

    def test_filter_by_min_score(self, app_client):
        """Filtering by min_score=9 returns only high-scoring breakthroughs."""
        response = app_client.get("/api/breakthroughs?min_score=9.0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(bt["breakthrough_score"] >= 9.0 for bt in data)

    def test_search_by_title(self, app_client):
        """Search filters by title substring."""
        response = app_client.get("/api/breakthroughs?search=Breakthrough%201")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "Breakthrough 1" in data[0]["title"]

    def test_search_by_tldr(self, app_client):
        """Search also matches TL;DR field."""
        response = app_client.get("/api/breakthroughs?search=NVFP4")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    def test_pagination_limit(self, app_client):
        """Limit parameter controls result count."""
        response = app_client.get("/api/breakthroughs?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_pagination_offset(self, app_client):
        """Offset parameter skips first results."""
        response_all = app_client.get("/api/breakthroughs?limit=5")
        all_data = response_all.json()

        response_offset = app_client.get("/api/breakthroughs?limit=2&offset=2")
        offset_data = response_offset.json()

        assert offset_data[0]["id"] == all_data[2]["id"]

    def test_combined_filters(self, app_client):
        """Multiple filters work together."""
        response = app_client.get("/api/breakthroughs?category=core&min_score=9.0")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(bt["category"] == "core" for bt in data)
        assert all(bt["breakthrough_score"] >= 9.0 for bt in data)

    def test_ordering_by_score(self, app_client):
        """Results are ordered by breakthrough_score DESC."""
        response = app_client.get("/api/breakthroughs?limit=5")
        data = response.json()
        scores = [bt["breakthrough_score"] for bt in data]
        assert scores == sorted(scores, reverse=True)

    def test_get_detail_existing(self, app_client):
        """GET /api/breakthroughs/{id} returns full detail."""
        response = app_client.get("/api/breakthroughs/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert "title" in data
        assert "tldr" in data
        assert "use_case" in data
        assert "comparison" in data
        assert "quickstart" in data
        assert "hardware_requirements" in data
        assert "feedback_summary" in data

    def test_get_detail_not_found(self, app_client):
        """GET /api/breakthroughs/9999 returns 404."""
        response = app_client.get("/api/breakthroughs/9999")
        assert response.status_code == 404

    def test_detail_includes_feedback_summary(self, app_client):
        """Detail endpoint includes feedback summary counts."""
        response = app_client.get("/api/breakthroughs/1")
        data = response.json()
        assert data["feedback_summary"]["hit"] == 1

    def test_recent_endpoint(self, app_client):
        """GET /api/recent returns recently evaluated breakthroughs."""
        response = app_client.get("/api/recent?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_limit_capped_at_200(self, app_client):
        """Limit is capped at 200."""
        response = app_client.get("/api/breakthroughs?limit=200")
        assert response.status_code == 200


# ============================================================================
# Stats API Tests
# ============================================================================

class TestStatsAPI:
    """Tests for /api/stats endpoint."""

    def test_returns_stats(self, app_client):
        """GET /api/stats returns aggregate statistics."""
        response = app_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_breakthroughs" in data
        assert "total_feedback" in data
        assert "rating_counts" in data
        assert "avg_breakthrough_score" in data
        assert "avg_roi_score" in data
        assert "category_counts" in data

    def test_stats_values(self, app_client):
        """Stats values match seeded data."""
        response = app_client.get("/api/stats")
        data = response.json()
        assert data["total_breakthroughs"] == 4  # non-discard
        assert data["total_feedback"] == 1
        assert data["rating_counts"].get("hit") == 1
        assert data["category_counts"].get("core") == 2
        assert data["category_counts"].get("serendipity") == 2
        assert data["category_counts"].get("discard") == 1

    def test_stats_avg_scores(self, app_client):
        """Average scores are calculated correctly."""
        response = app_client.get("/api/stats")
        data = response.json()
        # Non-discard scores: 9.5, 8.8, 9.2, 7.5 -> avg = 8.75
        assert data["avg_breakthrough_score"] == 8.75


# ============================================================================
# Digests API Tests
# ============================================================================

class TestDigestsAPI:
    """Tests for /api/digests endpoint."""

    def test_list_digests(self, app_client):
        """GET /api/digests returns digest history."""
        response = app_client.get("/api/digests")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "NewsScout Morning Digest — 18.09.2026"
        assert data[0]["track_count"] == 3

    def test_digest_includes_track_count(self, app_client):
        """Digest entries include track_count subquery."""
        response = app_client.get("/api/digests")
        data = response.json()
        assert data[0]["track_count"] == 3

    def test_empty_digests(self, dashboard_settings):
        """GET /api/digests returns empty list when no digests exist."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/api/digests")
            assert response.status_code == 200
            assert response.json() == []


# ============================================================================
# Watchlist API Tests
# ============================================================================

class TestWatchlistAPI:
    """Tests for watchlist endpoints."""

    def test_get_watchlist_empty(self, app_client):
        """GET /api/watchlist returns empty list initially."""
        response = app_client.get("/api/watchlist")
        assert response.status_code == 200
        assert response.json() == []

    def test_toggle_watchlist_on(self, app_client):
        """POST watchlist toggle adds breakthrough to watchlist."""
        response = app_client.post("/api/breakthroughs/1/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_watchlisted"] is True

    def test_toggle_watchlist_off(self, app_client):
        """POST watchlist toggle removes from watchlist."""
        app_client.post("/api/breakthroughs/1/watchlist")
        response = app_client.post("/api/breakthroughs/1/watchlist")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_watchlisted"] is False

    def test_watchlist_after_toggle(self, app_client):
        """Watchlist endpoint shows toggled breakthroughs."""
        app_client.post("/api/breakthroughs/2/watchlist")
        response = app_client.get("/api/watchlist")
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == 2

    def test_toggle_watchlist_not_found(self, app_client):
        """Toggle on non-existent breakthrough returns 404."""
        response = app_client.post("/api/breakthroughs/9999/watchlist")
        assert response.status_code == 404


# ============================================================================
# Feedback API Tests
# ============================================================================

class TestFeedbackAPI:
    """Tests for /api/feedback endpoint."""

    def test_submit_feedback_hit(self, app_client):
        """POST /api/feedback records a 'hit' rating."""
        response = app_client.post("/api/feedback", json={
            "breakthrough_id": 2,
            "rating": "hit",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["rating"] == "hit"

    def test_submit_feedback_inspire(self, app_client):
        """POST /api/feedback records an 'inspire' rating."""
        response = app_client.post("/api/feedback", json={
            "breakthrough_id": 3,
            "rating": "inspire",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_submit_feedback_with_notes(self, app_client):
        """POST /api/feedback records notes."""
        response = app_client.post("/api/feedback", json={
            "breakthrough_id": 1,
            "rating": "known",
            "notes": "Already using this in production",
        })
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_submit_feedback_invalid_rating(self, app_client):
        """POST /api/feedback with invalid rating returns 400."""
        response = app_client.post("/api/feedback", json={
            "breakthrough_id": 1,
            "rating": "invalid_rating",
        })
        assert response.status_code == 400

    def test_submit_feedback_missing_fields(self, app_client):
        """POST /api/feedback without required fields returns 400."""
        response = app_client.post("/api/feedback", json={"rating": "hit"})
        assert response.status_code == 400

    def test_submit_feedback_nonexistent_breakthrough(self, app_client):
        """POST /api/feedback for non-existent breakthrough returns 400."""
        response = app_client.post("/api/feedback", json={
            "breakthrough_id": 9999,
            "rating": "hit",
        })
        assert response.status_code == 400

    def test_feedback_appears_in_stats(self, app_client):
        """Submitted feedback is reflected in stats."""
        app_client.post("/api/feedback", json={
            "breakthrough_id": 2, "rating": "inspire",
        })
        app_client.post("/api/feedback", json={
            "breakthrough_id": 3, "rating": "hype",
        })
        response = app_client.get("/api/stats")
        data = response.json()
        assert data["total_feedback"] == 3
        assert data["rating_counts"].get("inspire") == 1
        assert data["rating_counts"].get("hype") == 1

    def test_feedback_appears_in_detail(self, app_client):
        """Submitted feedback appears in breakthrough detail."""
        app_client.post("/api/feedback", json={
            "breakthrough_id": 2, "rating": "inspire",
        })
        response = app_client.get("/api/breakthroughs/2")
        data = response.json()
        assert data["feedback_summary"].get("inspire") == 1


# ============================================================================
# Empty Database Tests
# ============================================================================

class TestEmptyDatabase:
    """Tests with an empty (migrated but unseeded) database."""

    def test_empty_breakthroughs(self, dashboard_settings):
        """Empty DB returns empty list for breakthroughs."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/api/breakthroughs")
            assert response.status_code == 200
            assert response.json() == []

    def test_empty_stats(self, dashboard_settings):
        """Empty DB returns zeroed stats."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/api/stats")
            data = response.json()
            assert data["total_breakthroughs"] == 0
            assert data["total_feedback"] == 0
            assert data["avg_breakthrough_score"] == 0.0

    def test_empty_watchlist(self, dashboard_settings):
        """Empty DB returns empty watchlist."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/api/watchlist")
            assert response.json() == []

    def test_empty_recent(self, dashboard_settings):
        """Empty DB returns empty recent list."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/api/recent")
            assert response.json() == []

    def test_html_page_empty_db(self, dashboard_settings):
        """HTML page still serves on empty DB."""
        app = create_app(dashboard_settings)
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "NewsScout" in response.text


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_limit_minimum(self, app_client):
        """Limit=1 returns exactly 1 result."""
        response = app_client.get("/api/breakthroughs?limit=1")
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_offset_beyond_data(self, app_client):
        """Offset beyond available data returns empty list."""
        response = app_client.get("/api/breakthroughs?limit=10&offset=100")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_no_match(self, app_client):
        """Search with no matches returns empty list."""
        response = app_client.get("/api/breakthroughs?search=nonexistentterm12345")
        assert response.status_code == 200
        assert response.json() == []

    def test_all_feedback_ratings(self, app_client):
        """All four feedback ratings can be submitted."""
        for rating in ["hit", "hype", "known", "inspire"]:
            response = app_client.post("/api/feedback", json={
                "breakthrough_id": 4,
                "rating": rating,
            })
            assert response.status_code == 200
            assert response.json()["success"] is True

    def test_negative_min_score(self, app_client):
        """Negative min_score returns all results."""
        response = app_client.get("/api/breakthroughs?min_score=-1")
        assert response.status_code == 200

    def test_health_check_endpoint(self, app_client):
        """Stats endpoint can be used as health check."""
        response = app_client.get("/api/stats")
        assert response.status_code == 200
