"""tests/test_challenger_edge_cases.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Empirical Challenge & Stress Test Suite for Milestone 1:
1. Pure-Python MP3 silent frame generator verification via ffprobe and ffmpeg concat demuxer.
2. FFmpeg / ffprobe minimum frame boundary analysis (single-frame failure vs multi-frame success).
3. Comprehensive Pydantic model validation on malformed inputs, out-of-bounds scores, missing fields, and invalid enums.
4. SQL injection resilience across Pydantic models and SQLite WAL persistence.
5. Telegram HTML & Markdown rendering safety and injection analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import pytest
from pydantic import ValidationError

from newsscout.storage.db import Database, create_connection
from newsscout.storage.migrations import apply_migrations
from newsscout.storage.models import (
    AudioTurn,
    Breakthrough,
    DecisionCard,
    DecisionCardData,
    DialogueScript,
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
from tests.conftest import generate_mock_mp3_bytes


# ============================================================================
# Task 1: Empirical Verification of Pure-Python MP3 Generator (ffprobe / ffmpeg)
# ============================================================================

class TestMP3GeneratorEmpiricalChallenge:
    """Empirically challenges pure-Python MP3 silent frame generator using ffprobe and ffmpeg."""

    @pytest.fixture(autouse=True)
    def check_tools(self) -> None:
        """Verify ffprobe and ffmpeg binaries are available on system."""
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe binary is required for empirical MP3 verification but was not found.")
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg binary is required for empirical MP3 verification but was not found.")

    def _run_ffprobe(self, file_path: Path) -> dict[str, Any]:
        """Runs ffprobe and returns parsed JSON output."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-print_format", "json",
            str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"ffprobe failed on {file_path}:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert res.stderr == "", f"ffprobe emitted unexpected warnings/errors: {res.stderr}"
        return json.loads(res.stdout)

    def test_single_frame_rejected_by_ffprobe_demuxer(self, tmp_path: Path) -> None:
        """Empirically demonstrates that a 1-frame MP3 file is rejected by ffprobe / ffmpeg demuxer.

        MPEG audio demuxers (libavformat/mp3dec.c) require at least two consecutive sync frames
        to establish stream sync. When generate_mock_mp3_bytes() produces 1 frame (duration <= 0.052s),
        ffprobe fails with 'Failed to find two consecutive MPEG audio frames'.
        """
        # 1 frame = 1152 samples / 44100 Hz = 0.02612s
        data = generate_mock_mp3_bytes(duration_seconds=0.02)
        assert len(data) == 417, f"Single frame must be exactly 417 bytes, got {len(data)}"

        mp3_file = tmp_path / "single_frame.mp3"
        mp3_file.write_bytes(data)

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-print_format", "json",
            str(mp3_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode != 0, "Expected ffprobe to fail on a single-frame MP3"
        assert "Failed to find two consecutive MPEG audio frames" in res.stderr

    def test_two_frames_recognized_by_ffprobe(self, tmp_path: Path) -> None:
        """Verifies that 2 frames (duration >= 0.053s) are immediately recognized by ffprobe as valid MPEG-1 Layer 3."""
        # 2 frames = 2 * 417 = 834 bytes (~0.0522s)
        data = generate_mock_mp3_bytes(duration_seconds=0.06)
        assert len(data) == 834, f"2 frames must be exactly 834 bytes, got {len(data)}"

        mp3_file = tmp_path / "two_frames.mp3"
        mp3_file.write_bytes(data)

        probe = self._run_ffprobe(mp3_file)
        assert len(probe["streams"]) == 1
        stream = probe["streams"][0]

        assert stream["codec_name"] == "mp3"
        assert "MPEG audio layer 3" in stream["codec_long_name"]
        assert stream["codec_type"] == "audio"
        assert stream["sample_rate"] == "44100"
        assert stream["channels"] == 2
        assert stream["channel_layout"] == "stereo"
        assert probe["format"]["format_name"] == "mp3"

    def test_multi_frame_stream_recognized_by_ffprobe(self, tmp_path: Path) -> None:
        """Verifies multi-frame MP3 files (1s, 3s, 5s) have accurate duration, bitrate, and no corruption."""
        for duration in [1.0, 3.0, 5.0]:
            mp3_file = tmp_path / f"multi_frame_{int(duration)}s.mp3"
            mp3_file.write_bytes(generate_mock_mp3_bytes(duration_seconds=duration))

            probe = self._run_ffprobe(mp3_file)
            stream = probe["streams"][0]
            fmt = probe["format"]

            assert stream["codec_name"] == "mp3"
            assert stream["sample_rate"] == "44100"
            assert stream["channels"] == 2
            assert stream["bit_rate"] == "128000"
            parsed_duration = float(fmt["duration"])
            # Duration should be within 1 frame (26ms) of requested duration
            assert abs(parsed_duration - duration) < 0.05, (
                f"Expected duration ~{duration}s, got {parsed_duration}s"
            )

    def test_ffmpeg_concat_demuxer_stream_copy(self, tmp_path: Path) -> None:
        """Verifies multi-frame MP3 files can be concatenated via ffmpeg concat demuxer with -c copy."""
        t1 = tmp_path / "track_1.mp3"
        t2 = tmp_path / "track_2.mp3"
        t3 = tmp_path / "track_3.mp3"
        merged = tmp_path / "merged_output.mp3"
        concat_list = tmp_path / "concat.txt"

        t1.write_bytes(generate_mock_mp3_bytes(duration_seconds=2.0))
        t2.write_bytes(generate_mock_mp3_bytes(duration_seconds=3.0))
        t3.write_bytes(generate_mock_mp3_bytes(duration_seconds=1.5))

        # Build ffmpeg concat script using relative filenames
        lines = [f"file '{t1.name}'", f"file '{t2.name}'", f"file '{t3.name}'"]
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(merged),
        ]
        res = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"ffmpeg concat failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert res.stderr == "", f"ffmpeg concat emitted errors: {res.stderr}"
        assert merged.exists()

        # Validate merged output with ffprobe
        probe = self._run_ffprobe(merged)
        stream = probe["streams"][0]
        fmt = probe["format"]

        assert stream["codec_name"] == "mp3"
        assert stream["sample_rate"] == "44100"
        total_duration = float(fmt["duration"])
        expected_duration = 2.0 + 3.0 + 1.5  # 6.5s
        assert abs(total_duration - expected_duration) < 0.1, (
            f"Expected merged duration ~{expected_duration}s, got {total_duration}s"
        )

    def test_ffmpeg_concat_fails_on_single_frame_files(self, tmp_path: Path) -> None:
        """Demonstrates that ffmpeg concat demuxer fails when encountering 1-frame MP3 files."""
        f1 = tmp_path / "short_1.mp3"
        f2 = tmp_path / "short_2.mp3"
        out = tmp_path / "failed_merged.mp3"
        c_list = tmp_path / "concat_short.txt"

        f1.write_bytes(generate_mock_mp3_bytes(duration_seconds=0.01))
        f2.write_bytes(generate_mock_mp3_bytes(duration_seconds=0.01))
        c_list.write_text(f"file '{f1.name}'\nfile '{f2.name}'\n", encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(c_list),
            "-c", "copy",
            str(out),
        ]
        res = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, check=False)
        assert res.returncode != 0, "Expected ffmpeg concat to fail on 1-frame MP3s"
        assert "Failed to find two consecutive MPEG audio frames" in res.stderr

    def test_ffmpeg_decodes_synthesized_mp3_without_errors(self, tmp_path: Path) -> None:
        """Verifies synthesized MP3 bitstream decodes cleanly through full ffmpeg MPEG audio decoder."""
        mp3_file = tmp_path / "decode_test.mp3"
        mp3_file.write_bytes(generate_mock_mp3_bytes(duration_seconds=2.5))

        # Decode to null sink (-f null -)
        cmd = ["ffmpeg", "-v", "error", "-i", str(mp3_file), "-f", "null", "-"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"ffmpeg decoding failed:\nSTDERR: {res.stderr}"
        assert res.stderr == "", f"ffmpeg audio decoder emitted errors: {res.stderr}"

    @pytest.mark.parametrize("duration", [0.0, -1.0, 0.001, 10.5])
    def test_mp3_generator_boundary_durations(self, duration: float) -> None:
        """Verifies generator handles boundary/negative/zero durations gracefully (minimum 1 frame)."""
        data = generate_mock_mp3_bytes(duration_seconds=duration)
        assert len(data) >= 417
        assert data.startswith(b"\xFF\xFB\x90\x04")


# ============================================================================
# Task 2: Pydantic Validation on Malformed, Out-of-Bounds & Missing Fields
# ============================================================================

class TestPydanticModelValidationChallenges:
    """Stress-tests Pydantic v2 domain models against boundary violations and corrupt data."""

    def test_breakthrough_score_lower_boundary_rejection(self) -> None:
        """Verifies breakthrough_score < 1.0 is strictly rejected."""
        for invalid_score in [0.999, 0.0, -1.0, -100.0]:
            with pytest.raises(ValidationError) as exc_info:
                Breakthrough(
                    raw_item_id=1,
                    title="Test",
                    breakthrough_score=invalid_score,
                    roi_score=5.0,
                    category=Stage2Category.CORE,
                    tldr="TLDR",
                    use_case="Use case",
                    comparison="Comparison",
                    quickstart="docker run",
                    hardware_requirements="None",
                    license="MIT",
                    card_markdown="# Test",
                )
            assert "breakthrough_score" in str(exc_info.value)
            assert "greater than or equal to 1" in str(exc_info.value)

    def test_breakthrough_score_upper_boundary_rejection(self) -> None:
        """Verifies breakthrough_score > 10.0 is strictly rejected."""
        for invalid_score in [10.001, 11.0, 100.0]:
            with pytest.raises(ValidationError) as exc_info:
                Breakthrough(
                    raw_item_id=1,
                    title="Test",
                    breakthrough_score=invalid_score,
                    roi_score=5.0,
                    category=Stage2Category.CORE,
                    tldr="TLDR",
                    use_case="Use case",
                    comparison="Comparison",
                    quickstart="docker run",
                    hardware_requirements="None",
                    license="MIT",
                    card_markdown="# Test",
                )
            assert "breakthrough_score" in str(exc_info.value)
            assert "less than or equal to 10" in str(exc_info.value)

    def test_roi_score_boundary_rejection(self) -> None:
        """Verifies roi_score bounds [1.0, 10.0] are strictly enforced."""
        for invalid_score in [0.0, 0.5, 10.1, 20.0]:
            with pytest.raises(ValidationError) as exc_info:
                Breakthrough(
                    raw_item_id=1,
                    title="Test",
                    breakthrough_score=5.0,
                    roi_score=invalid_score,
                    category=Stage2Category.CORE,
                    tldr="TLDR",
                    use_case="Use case",
                    comparison="Comparison",
                    quickstart="docker run",
                    hardware_requirements="None",
                    license="MIT",
                    card_markdown="# Test",
                )
            assert "roi_score" in str(exc_info.value)

    def test_score_exact_boundaries_accepted(self) -> None:
        """Verifies exactly 1.0 and 10.0 are valid scores."""
        bt_min = Breakthrough(
            raw_item_id=1,
            title="Min Score",
            breakthrough_score=1.0,
            roi_score=1.0,
            category=Stage2Category.CORE,
            tldr="TLDR",
            use_case="Use case",
            comparison="Comparison",
            quickstart="docker run",
            hardware_requirements="None",
            license="MIT",
            card_markdown="# Test",
        )
        assert bt_min.breakthrough_score == 1.0
        assert bt_min.roi_score == 1.0

        bt_max = Breakthrough(
            raw_item_id=1,
            title="Max Score",
            breakthrough_score=10.0,
            roi_score=10.0,
            category=Stage2Category.SERENDIPITY,
            tldr="TLDR",
            use_case="Use case",
            comparison="Comparison",
            quickstart="docker run",
            hardware_requirements="None",
            license="MIT",
            card_markdown="# Test",
        )
        assert bt_max.breakthrough_score == 10.0
        assert bt_max.roi_score == 10.0

    def test_score_non_numeric_and_special_floats_rejection(self) -> None:
        """Verifies NaN, Inf, and non-numeric strings are rejected for scores."""
        for bad_score in [float("nan"), float("inf"), float("-inf"), "not-a-number", None]:
            with pytest.raises(ValidationError):
                Breakthrough(
                    raw_item_id=1,
                    title="Test",
                    breakthrough_score=bad_score,  # type: ignore[arg-type]
                    roi_score=5.0,
                    category=Stage2Category.CORE,
                    tldr="TLDR",
                    use_case="Use case",
                    comparison="Comparison",
                    quickstart="docker run",
                    hardware_requirements="None",
                    license="MIT",
                    card_markdown="# Test",
                )

    def test_decision_card_bounds_enforcement(self) -> None:
        """Verifies DecisionCard enforces score bounds identical to Breakthrough."""
        card_data = DecisionCardData(
            tldr="TLDR",
            use_case="Use case",
            comparison="Comp",
            quickstart="docker run",
            hardware_requirements="None",
            license="MIT",
        )
        with pytest.raises(ValidationError):
            DecisionCard(
                title="Invalid Card",
                breakthrough_score=0.5,
                roi_score=5.0,
                category=Stage2Category.CORE,
                data=card_data,
            )
        with pytest.raises(ValidationError):
            DecisionCard(
                title="Invalid Card",
                breakthrough_score=5.0,
                roi_score=10.5,
                category=Stage2Category.CORE,
                data=card_data,
            )

    def test_missing_required_fields_rejection(self) -> None:
        """Verifies omission of mandatory fields raises ValidationError across models."""
        # 1. DecisionCardData missing fields
        with pytest.raises(ValidationError) as exc:
            DecisionCardData(tldr="Only tldr")  # type: ignore[call-arg]
        errs = str(exc.value)
        assert "use_case" in errs
        assert "comparison" in errs
        assert "quickstart" in errs

        # 2. RawItem missing fields
        with pytest.raises(ValidationError) as exc:
            RawItem(title="No source")  # type: ignore[call-arg]
        assert "source" in str(exc.value)
        assert "source_id" in str(exc.value)

        # 3. Track missing fields
        with pytest.raises(ValidationError) as exc:
            Track(title="Incomplete track")  # type: ignore[call-arg]
        assert "digest_id" in str(exc.value)

        # 4. Source missing fields
        with pytest.raises(ValidationError) as exc:
            Source(name="Only name")  # type: ignore[call-arg]
        assert "source_type" in str(exc.value)

        # 5. Stage1Evaluation missing fields
        with pytest.raises(ValidationError) as exc:
            Stage1Evaluation(raw_item_id=1)  # type: ignore[call-arg]
        assert "passed" in str(exc.value)

        # 6. Digest missing fields
        with pytest.raises(ValidationError) as exc:
            Digest(slot=DigestSlot.MORNING)  # type: ignore[call-arg]
        assert "digest_date" in str(exc.value)
        assert "title" in str(exc.value)

        # 7. Feedback missing fields
        with pytest.raises(ValidationError) as exc:
            Feedback(rating=FeedbackRating.HIT)  # type: ignore[call-arg]
        assert "breakthrough_id" in str(exc.value)

        # 8. UserPreference missing fields
        with pytest.raises(ValidationError) as exc:
            UserPreference()  # type: ignore[call-arg]
        assert "pref_key" in str(exc.value)

    def test_invalid_enum_values_rejection(self) -> None:
        """Verifies invalid enum values trigger ValidationError across all domain models."""
        # Invalid Stage2Category
        with pytest.raises(ValidationError):
            Breakthrough(
                raw_item_id=1,
                title="Test",
                breakthrough_score=5.0,
                roi_score=5.0,
                category="invalid_category",  # type: ignore[arg-type]
                tldr="TLDR",
                use_case="Use case",
                comparison="Comparison",
                quickstart="docker run",
                hardware_requirements="None",
                license="MIT",
                card_markdown="# Test",
            )

        # Invalid SourceType
        with pytest.raises(ValidationError):
            Source(name="src", source_type="web_scraper")  # type: ignore[arg-type]

        # Invalid FeedbackRating
        with pytest.raises(ValidationError):
            Feedback(breakthrough_id=1, rating="super_like")  # type: ignore[arg-type]

        # Invalid DigestSlot
        with pytest.raises(ValidationError):
            Digest(slot="midnight", digest_date="2026-09-18", title="Midnight")  # type: ignore[arg-type]

        # Invalid DigestStatus
        with pytest.raises(ValidationError):
            Digest(slot=DigestSlot.MORNING, digest_date="2026-09-18", title="Test", status="unknown_status")  # type: ignore[arg-type]

        # Invalid TrackType
        with pytest.raises(ValidationError):
            Track(
                digest_id=1,
                track_number=1,
                track_type="commercial_break",  # type: ignore[arg-type]
                title="Track",
                audio_path="test.mp3",
                script=DialogueScript(track_number=1, turns=[]),
            )

    def test_track_number_and_duration_constraints(self) -> None:
        """Verifies Track constraints: track_number in [1, 5] and duration_seconds >= 0."""
        script = DialogueScript(track_number=1, turns=[])

        # track_number = 0 (< 1)
        with pytest.raises(ValidationError):
            Track(
                digest_id=1,
                track_number=0,
                track_type=TrackType.EXECUTIVE_SUMMARY,
                title="Track 0",
                duration_seconds=100,
                audio_path="test.mp3",
                script=script,
            )

        # track_number = 6 (> 5)
        with pytest.raises(ValidationError):
            Track(
                digest_id=1,
                track_number=6,
                track_type=TrackType.EXECUTIVE_SUMMARY,
                title="Track 6",
                duration_seconds=100,
                audio_path="test.mp3",
                script=script,
            )

        # duration_seconds = -10 (< 0)
        with pytest.raises(ValidationError):
            Track(
                digest_id=1,
                track_number=1,
                track_type=TrackType.EXECUTIVE_SUMMARY,
                title="Track 1",
                duration_seconds=-10,
                audio_path="test.mp3",
                script=script,
            )

    def test_corrupt_json_in_row_normalizer_raises_validation_error(self) -> None:
        """Verifies corrupt JSON strings passed to models raise ValidationError."""
        with pytest.raises(ValidationError):
            RawItem.model_validate({
                "source": "test",
                "source_id": "item1",
                "title": "Title",
                "url": "http://test",
                "metadata_json": "{corrupted json",
            })

        with pytest.raises(ValidationError):
            Source.model_validate({
                "name": "test_src",
                "source_type": "api",
                "config_json": "['unclosed list",
            })

        with pytest.raises(ValidationError):
            Stage1Evaluation.model_validate({
                "raw_item_id": 1,
                "passed": True,
                "heuristics_json": "corrupt{heuristics",
            })


# ============================================================================
# Task 3: SQL Injection Resilience Across Models and SQLite Persistence
# ============================================================================

class TestSQLInjectionResilience:
    """Stress-tests string fields with aggressive SQL injection payloads."""

    SQLI_PAYLOADS = [
        "Robert'); DROP TABLE breakthroughs; --",
        "' OR '1'='1",
        "'; DELETE FROM sources; DELETE FROM raw_items; --",
        "1 UNION SELECT 1, 'admin', 'hacked', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; --",
        "'; ATTACH DATABASE ':memory:' AS evil; --",
        "'; UPDATE user_preferences SET value_json = '{}'; --",
        "\" OR \"\"=\"",
        "'; PRAGMA writable_schema = 1; --",
        "/* multiline comment */ '; DROP TABLE schema_migrations; --",
        "NUL byte \x00 in middle of string",
        "Unicode quotes ‘ OR ‘1’=‘1’ and em-dash —",
        "A" * 10000,  # 10 KB string
    ]

    @pytest.mark.parametrize("payload", SQLI_PAYLOADS)
    def test_pydantic_accepts_sqli_payloads_as_literal_text(self, payload: str) -> None:
        """Pydantic should preserve raw text payloads without crash or unintended sanitization."""
        card_data = DecisionCardData(
            tldr=payload,
            use_case=payload,
            comparison=payload,
            quickstart=f"docker run --rm -it {payload}",
            hardware_requirements=payload,
            license=payload,
        )
        assert card_data.tldr == payload

        bt = Breakthrough(
            raw_item_id=1,
            title=payload,
            repo_url=f"https://github.com/{payload}",
            breakthrough_score=8.5,
            roi_score=8.0,
            category=Stage2Category.CORE,
            tldr=card_data.tldr,
            use_case=card_data.use_case,
            comparison=card_data.comparison,
            quickstart=card_data.quickstart,
            hardware_requirements=card_data.hardware_requirements,
            license=card_data.license,
            card_markdown=card_data.render_markdown(payload),
        )
        assert bt.title == payload

    @pytest.mark.asyncio
    async def test_sqli_persistence_in_sqlite_wal(self, migrated_db: Database) -> None:
        """Verifies parameterized execution prevents SQL injection from executing or damaging tables."""
        for idx, payload in enumerate(self.SQLI_PAYLOADS, start=1):
            # 1. Insert into raw_items
            raw = RawItem(
                source=f"sqli_source_{idx}",
                source_id=f"sqli_id_{idx}",
                title=payload,
                url=f"https://test.com/{idx}",
                raw_content=payload,
                metadata={"test_sqli": payload},
            )
            raw_id = await migrated_db.execute(
                """
                INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
                VALUES (:source, :source_id, :title, :url, :raw_content, :metadata_json, :ingested_at);
                """,
                raw.to_db_params(),
            )

            # 2. Insert into breakthroughs referencing raw_items
            bt = Breakthrough(
                raw_item_id=raw_id,
                title=payload,
                repo_url=f"https://github.com/{payload}",
                breakthrough_score=9.0,
                roi_score=9.0,
                category=Stage2Category.CORE,
                tldr=payload,
                use_case=payload,
                comparison=payload,
                quickstart=payload,
                hardware_requirements=payload,
                license=payload,
                card_markdown=f"# {payload}",
            )
            bt_id = await migrated_db.execute(
                """
                INSERT INTO breakthroughs (
                    raw_item_id, title, repo_url, breakthrough_score, roi_score,
                    category, tldr, use_case, comparison, quickstart,
                    hardware_requirements, license, card_markdown, evaluation_raw_json,
                    is_watchlisted, evaluated_at
                ) VALUES (
                    :raw_item_id, :title, :repo_url, :breakthrough_score, :roi_score,
                    :category, :tldr, :use_case, :comparison, :quickstart,
                    :hardware_requirements, :license, :card_markdown, :evaluation_raw_json,
                    :is_watchlisted, :evaluated_at
                );
                """,
                bt.to_db_params(),
            )

            # 3. Query back and verify verbatim roundtrip
            row = await migrated_db.fetch_one(
                "SELECT * FROM breakthroughs WHERE id = :id;",
                {"id": bt_id},
            )
            assert row is not None
            bt_recovered = Breakthrough.from_row(row)
            assert bt_recovered.title == payload
            assert bt_recovered.tldr == payload

        # 4. Critical verification: Ensure no tables were dropped or modified
        tables = await migrated_db.fetch_all("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = {r["name"] for r in tables}
        expected_tables = {
            "schema_migrations", "sources", "raw_items", "stage1_evaluations",
            "breakthroughs", "digests", "digest_tracks", "feedback", "user_preferences"
        }
        assert expected_tables.issubset(table_names), f"Tables missing after SQLi attack: {expected_tables - table_names}"


# ============================================================================
# Task 4: Telegram HTML & Markdown Rendering Security & Injection Analysis
# ============================================================================

class TestRenderSecurityAndEscaping:
    """Analyzes whether HTML special characters in DecisionCard fields break Telegram HTML rendering."""

    def test_decision_card_render_telegram_html_raw_tags_behavior(self) -> None:
        """Examines DecisionCard.render_telegram_html() behavior when inputs contain unescaped HTML characters.

        Telegram Bot API HTML mode requires <, >, and & to be entity-escaped:
        - '&' -> '&amp;'
        - '<' -> '&lt;'
        - '>' -> '&gt;'
        If unescaped, Telegram's API returns HTTP 400 'can't parse entities'.
        """
        raw_tldr = "Enables <Model> & <Pipeline> with speed > 3x."
        card = DecisionCard(
            title="FastLLM <v2.0>",
            repo_url="https://github.com/fast/fastllm",
            breakthrough_score=9.0,
            roi_score=8.5,
            category=Stage2Category.CORE,
            data=DecisionCardData(
                tldr=raw_tldr,
                use_case="Agents & RAG",
                comparison="3x faster than baseline",
                quickstart="docker run --rm -v <dir>:/data image:latest",
                hardware_requirements="24GB VRAM",
                license="Apache-2.0",
            ),
        )

        rendered_html = card.render_telegram_html()

        # Verify remediation: The model escapes HTML special characters to prevent Telegram parse errors
        assert "&lt;dir&gt;" in rendered_html
        assert "&lt;Model&gt;" in rendered_html
        assert "&amp;" in rendered_html
        assert "<dir>" not in rendered_html
        assert "<Model>" not in rendered_html
