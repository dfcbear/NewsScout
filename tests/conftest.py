"""tests/conftest.py
~~~~~~~~~~~~~~~~~
Global pytest fixtures, pure-Python mock MP3 generator, temporary SQLite WAL
database fixtures, sample domain data factories, and forward-compatible test doubles.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Optional

from pydantic import SecretStr
import pytest
import pytest_asyncio

from newsscout.config import Settings
from newsscout.storage.db import Database, create_connection
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import (
    Breakthrough,
    DecisionCardData,
    Digest,
    DigestSlot,
    DigestStatus,
    Feedback,
    FeedbackRating,
    RawItem,
    Source,
    SourceType,
    Stage1Evaluation,
    Stage2Category,
    Track,
    TrackType,
    UserPreference,
)


# ============================================================================
# Pure-Python Mock MP3 Generator (Zero External Dependencies)
# ============================================================================

def create_silent_mp3(
    duration_seconds: float = 1.0,
    bitrate_kbps: int = 128,
    sample_rate: int = 44100,
    min_frames: int = 2,
) -> bytes:
    """Generates valid, silent MPEG-1 Layer 3 MP3 bitstream frames in memory.

    Enforces min_frames >= 2 (max(2, ...)) ensuring compatibility with FFmpeg's MP3 demuxer.
    """
    header = bytes([0xFF, 0xFB, 0x90, 0x04])
    frame_len = int(144 * bitrate_kbps * 1000 / sample_rate)  # 417 bytes
    side_info = b"\x00" * 32  # 32 bytes side info for MPEG-1 Layer III stereo
    payload = b"\x00" * (frame_len - 4 - len(side_info))
    single_frame = header + side_info + payload

    samples_per_frame = 1152
    frames_count = max(min_frames, int(duration_seconds * sample_rate / samples_per_frame))
    return single_frame * frames_count


def generate_mock_mp3_bytes(
    duration_seconds: float = 1.0,
    bitrate_kbps: int = 128,
    sample_rate: int = 44100,
    min_frames: int = 1,
) -> bytes:
    """Generates valid, silent MPEG-1 Layer 3 MP3 bitstream frames in memory.

    Verified with ffprobe and ffmpeg concat demuxer (-f concat -safe 0 -c copy).
    """
    return create_silent_mp3(
        duration_seconds=duration_seconds,
        bitrate_kbps=bitrate_kbps,
        sample_rate=sample_rate,
        min_frames=min_frames,
    )


@pytest.fixture
def mock_mp3_factory(tmp_path: Path) -> Callable[[str, float], Path]:
    """Pytest fixture returning a factory function to create valid MP3 files on disk."""
    def _create(filename: str = "mock_track.mp3", duration_seconds: float = 1.0) -> Path:
        out_path = tmp_path / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = create_silent_mp3(duration_seconds=duration_seconds, min_frames=2)
        out_path.write_bytes(data)
        return out_path

    return _create


# ============================================================================
# Temporary SQLite WAL Database Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Returns a unique temporary database file path."""
    return tmp_path / "test_ai_scout.db"


@pytest.fixture
def sync_db_conn(temp_db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yields an open synchronous connection configured with WAL mode and all pragmas."""
    conn = create_connection(temp_db_path, timeout=5.0, init_wal=True)
    try:
        yield conn
    finally:
        conn.close()


@pytest_asyncio.fixture
async def temp_db(temp_db_path: Path) -> AsyncIterator[Database]:
    """Yields an initialized, unmigrated Database instance."""
    db = Database(temp_db_path, timeout=5.0)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest_asyncio.fixture
async def migrated_db(temp_db_path: Path) -> AsyncIterator[Database]:
    """Yields an initialized Database instance with Migration 001 applied."""
    db = Database(temp_db_path, timeout=5.0)
    await db.initialize()
    with db.get_sync_connection() as conn:
        apply_migrations(conn=conn)
    try:
        yield db
    finally:
        await db.close()


# ============================================================================
# Domain Data Factories
# ============================================================================

class DataFactory:
    """Factory creating domain model test fixtures with realistic data."""

    @staticmethod
    def create_source(
        name: str = "github_releases",
        source_type: SourceType = SourceType.API,
        config: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> Source:
        return Source(
            name=name,
            source_type=source_type,
            config=config or {"repos": ["vllm-project/vllm", "ggerganov/llama.cpp"]},
            is_active=is_active,
            last_poll_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_raw_item(
        source: str = "github_releases",
        source_id: str = "vllm-project/vllm:v0.6.0",
        title: str = "vLLM v0.6.0 with NVFP4 Support",
        url: str = "https://github.com/vllm-project/vllm/releases/tag/v0.6.0",
        raw_content: str = "Native NVFP4 kernel execution for RTX 4090.",
        metadata: dict[str, Any] | None = None,
    ) -> RawItem:
        return RawItem(
            source=source,
            source_id=source_id,
            title=title,
            url=url,
            raw_content=raw_content,
            metadata=metadata or {"stars": 28000, "tag": "v0.6.0"},
            ingested_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_stage1_eval(
        raw_item_id: int = 1,
        passed: bool = True,
        drop_reason: str | None = None,
        detected_license: str = "Apache-2.0",
        has_docker: bool = True,
        has_runnable_code: bool = True,
    ) -> Stage1Evaluation:
        return Stage1Evaluation(
            raw_item_id=raw_item_id,
            passed=passed,
            drop_reason=drop_reason,
            detected_license=detected_license,
            has_docker=has_docker,
            has_runnable_code=has_runnable_code,
            heuristics={"dockerfile_present": True, "pyproject_present": True},
            evaluated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_breakthrough(
        raw_item_id: int = 1,
        title: str = "vLLM NVFP4 Kernel Execution",
        breakthrough_score: float = 9.4,
        roi_score: float = 9.2,
        category: Stage2Category = Stage2Category.CORE,
        repo_url: str = "https://github.com/vllm-project/vllm",
        hardware_requirements: str | None = None,
        license: str | None = None,
    ) -> Breakthrough:
        card = DecisionCardData(
            tldr="Direct NVFP4 kernel execution inside vLLM without quantization perplexity loss.",
            use_case="Local agent execution with 70B models at 32 tokens/sec on 24GB VRAM.",
            comparison="3x lower latency compared to standard FP16 vLLM baseline.",
            quickstart="docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest",
            hardware_requirements=hardware_requirements or "NVIDIA RTX 4090 (24GB VRAM), 32GB System RAM",
            license=license or "Apache-2.0",
        )
        return Breakthrough(
            raw_item_id=raw_item_id,
            title=title,
            repo_url=repo_url,
            breakthrough_score=breakthrough_score,
            roi_score=roi_score,
            category=category,
            tldr=card.tldr,
            use_case=card.use_case,
            comparison=card.comparison,
            quickstart=card.quickstart,
            hardware_requirements=card.hardware_requirements,
            license=card.license,
            card_markdown=f"# {title}\n\n**TL;DR**: {card.tldr}\n\n**Use Case**: {card.use_case}\n\n**Comparison**: {card.comparison}\n\n**Quickstart**: `{card.quickstart}`\n\n**Hardware**: {card.hardware_requirements} | {card.license}",
            evaluation_raw={"model": "gemini-3.8-flash", "tokens": 450},
            is_watchlisted=False,
            evaluated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_digest(
        slot: DigestSlot = DigestSlot.MORNING,
        digest_date: str = "2026-09-18",
        title: str = "NewsScout Morning Digest — 18.09.2026",
        status: DigestStatus = DigestStatus.PENDING,
    ) -> Digest:
        return Digest(
            slot=slot,
            digest_date=digest_date,
            title=title,
            total_duration_seconds=1200,
            merged_audio_path="data/audio/NewsScout_Digest_20260918_morning.mp3",
            telegram_message_id=None,
            status=status,
            created_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_track(
        digest_id: int = 1,
        track_number: int = 1,
        track_type: TrackType = TrackType.EXECUTIVE_SUMMARY,
        title: str = "Track 1: Executive Summary & Agenda",
        duration_seconds: int = 180,
    ) -> Track:
        return Track(
            digest_id=digest_id,
            track_number=track_number,
            track_type=track_type,
            title=title,
            duration_seconds=duration_seconds,
            audio_path=f"data/audio/digest_1_track_{track_number}.mp3",
            script_json=[
                {"speaker": "Conrad", "text": "Willkommen zum NewsScout Audio Digest."},
                {"speaker": "Katja", "text": "Guten Morgen! Heute haben wir ein echtes Highlight."},
            ],
            breakthrough_id=None,
            created_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def create_feedback(
        breakthrough_id: int = 1,
        rating: FeedbackRating = FeedbackRating.HIT,
        telegram_user_id: int = 99887766,
        source: str = "telegram",
    ) -> Feedback:
        return Feedback(
            breakthrough_id=breakthrough_id,
            rating=rating,
            telegram_user_id=telegram_user_id,
            source=source,
            notes="Evaluated directly on RTX 4090",
            created_at=datetime.now(timezone.utc),
        )


@pytest.fixture
def factory() -> type[DataFactory]:
    """Pytest fixture providing the DataFactory class."""
    return DataFactory


# ============================================================================
# Mock Settings Fixture
# ============================================================================

@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Provides an isolated Settings instance pointed to temporary directories."""
    settings = Settings(
        environment="test",
        db_path=tmp_path / "test_ai_scout.db",
        audio_output_dir=tmp_path / "audio",
        preferences_file=tmp_path / "preferences.json",
        gemini_api_key=SecretStr("mock-gemini-test-key"),
        telegram_bot_token=SecretStr("123456789:MOCK_BOT_TOKEN_FOR_TESTS"),
        telegram_chat_id="99887766",
        github_token=SecretStr("mock-github-token"),
        log_level="DEBUG",
    )
    settings.ensure_directories()
    return settings


# ============================================================================
# Forward-Compatible Test Doubles for Future Milestones
# ============================================================================

class MockGeminiClient:
    """Mock Gemini 3.8 Flash Client for offline evaluation testing."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.call_count = 0

    async def evaluate_candidate(self, candidate_title: str, content: str) -> dict[str, Any]:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Simulated Gemini API failure")

        is_core = any(term in candidate_title.lower() for term in ["vllm", "llama", "rag", "qdrant", "kernel"])
        category = "core" if is_core else "serendipity"
        score = 9.2 if is_core else 8.5

        return {
            "breakthrough_score": score,
            "roi_score": score - 0.2,
            "category": category,
            "rationale": f"High novelty evaluation for {candidate_title}.",
            "card": {
                "tldr": f"Breakthrough acceleration achieved in {candidate_title}.",
                "use_case": "Direct workflow integration for local agent reasoning.",
                "comparison": "3x performance uplift over standard baselines.",
                "quickstart": "docker run --gpus all local/test:latest",
                "hardware_and_license": "24GB VRAM | Apache-2.0",
            },
        }


class MockTelegramClient:
    """Mock Telegram Bot API client for offline delivery testing."""

    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_audios: list[dict[str, Any]] = []
        self.callback_answers: list[dict[str, Any]] = []

    async def send_message(self, chat_id: str, text: str, reply_markup: Any = None) -> int:
        msg_id = len(self.sent_messages) + 1
        self.sent_messages.append({
            "message_id": msg_id,
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
        })
        return msg_id

    async def send_audio(self, chat_id: str, audio_path: str, caption: str = "") -> int:
        msg_id = len(self.sent_audios) + 1000
        self.sent_audios.append({
            "message_id": msg_id,
            "chat_id": chat_id,
            "audio_path": audio_path,
            "caption": caption,
        })
        return msg_id


@pytest.fixture
def mock_gemini() -> MockGeminiClient:
    return MockGeminiClient()


@pytest.fixture
def mock_telegram() -> MockTelegramClient:
    return MockTelegramClient()
