"""tests/test_audio.py
~~~~~~~~~~~~~~~~~~~~~
Tests for the audio digest generation pipeline:
  * TTS engine (EdgeTTSEngine) with mock synthesis
  * Dialogue script generator (DialogueScriptGenerator)
  * Digest generator orchestrator (DigestGenerator)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newsscout.audio.tts_engine import BaseTTSEngine, EdgeTTSEngine
from newsscout.audio.script_gen import DialogueScriptGenerator
from newsscout.audio.digest_gen import DigestGenerator
from newsscout.config import Settings
from newsscout.storage.models import (
    AudioTurn,
    Breakthrough,
    DialogueScript,
    Digest,
    DigestSlot,
    DigestStatus,
    Stage2Category,
    Track,
    TrackType,
)
from tests.conftest import DataFactory


# ============================================================================
# BaseTTSEngine Tests
# ============================================================================

class TestBaseTTSEngine:
    """Tests for BaseTTSEngine abstract interface and shared utilities."""

    def test_estimate_duration_seconds_basic(self):
        """Word count estimation at default 130 wpm."""
        text = "Das ist ein Test mit genau zehn Woertern drin."
        duration = BaseTTSEngine.estimate_duration_seconds(text)
        assert duration > 0
        # 9 words at 130 wpm
        assert duration == pytest.approx((9 / 130) * 60, rel=0.01)

    def test_estimate_duration_seconds_empty_text(self):
        """Empty text yields zero duration."""
        assert BaseTTSEngine.estimate_duration_seconds("") == 0.0

    def test_estimate_duration_seconds_custom_wpm(self):
        """Custom words-per-minute rate is respected."""
        text = "eins zwei drei vier fuenf"
        duration = BaseTTSEngine.estimate_duration_seconds(text, wpm=200)
        assert duration == pytest.approx((5 / 200) * 60, rel=0.01)

    def test_get_speaker_voice_conrad(self, test_settings: Settings):
        """Conrad maps to tts_voice_male."""
        voice = BaseTTSEngine.get_speaker_voice("Conrad", test_settings)
        assert voice == test_settings.tts_voice_male

    def test_get_speaker_voice_katja(self, test_settings: Settings):
        """Katja maps to tts_voice_female."""
        voice = BaseTTSEngine.get_speaker_voice("Katja", test_settings)
        assert voice == test_settings.tts_voice_female

    def test_get_speaker_voice_unknown_defaults_to_male(self, test_settings: Settings):
        """Unknown speaker defaults to male voice."""
        voice = BaseTTSEngine.get_speaker_voice("Unknown", test_settings)
        assert voice == test_settings.tts_voice_male

    def test_base_tts_engine_is_abstract(self):
        """BaseTTSEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTTSEngine()


# ============================================================================
# EdgeTTSEngine Tests
# ============================================================================

class TestEdgeTTSEngine:
    """Tests for the default edge-tts engine implementation."""

    def test_init_creates_output_dir(self, test_settings: Settings):
        """Engine initialization ensures audio output directory exists."""
        engine = EdgeTTSEngine(test_settings)
        assert Path(test_settings.audio_output_dir).exists()
        assert engine.settings == test_settings

    @pytest.mark.asyncio
    async def test_synthesize_turn_with_mock(self, test_settings: Settings):
        """synthesize_turn returns bytes from edge_tts.Communicate."""
        mock_audio_data = b"FAKE_MP3_DATA"

        with patch("edge_tts.Communicate") as mock_comm_class:
            mock_comm = mock_comm_class.return_value
            mock_comm.stream = MockAsyncStream([
                {"type": "audio", "data": mock_audio_data}
            ])

            engine = EdgeTTSEngine(test_settings)
            turn = AudioTurn(speaker="Conrad", text="Hallo Welt")
            result = await engine.synthesize_turn(turn)

        assert result == mock_audio_data

    @pytest.mark.asyncio
    async def test_synthesize_turn_empty_audio(self, test_settings: Settings):
        """synthesize_turn handles empty audio gracefully."""
        with patch("edge_tts.Communicate") as mock_comm_class:
            mock_comm = mock_comm_class.return_value
            mock_comm.stream = MockAsyncStream([])

            engine = EdgeTTSEngine(test_settings)
            turn = AudioTurn(speaker="Katja", text="Test text")
            result = await engine.synthesize_turn(turn)

        assert result == b""

    @pytest.mark.asyncio
    async def test_synthesize_script_with_turns(self, test_settings: Settings):
        """synthesize_script concatenates turns into a single MP3."""
        mock_audio_data = b"FAKE_MP3_DATA_12345"

        with patch("edge_tts.Communicate") as mock_comm_class:
            mock_comm = mock_comm_class.return_value
            mock_comm.stream = MockAsyncStream([
                {"type": "audio", "data": mock_audio_data}
            ])

            engine = EdgeTTSEngine(test_settings)
            script = DialogueScript(
                track_number=1,
                track_type=TrackType.EXECUTIVE_SUMMARY,
                title="Test Track",
                turns=[
                    AudioTurn(speaker="Conrad", text="Hallo", pause_after_ms=0),
                    AudioTurn(speaker="Katja", text="Welt", pause_after_ms=0),
                ],
            )
            output_path = Path(test_settings.audio_output_dir) / "test_track.mp3"
            result = await engine.synthesize_script(script, output_path)

        assert result == output_path
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_synthesize_script_empty_turns(self, test_settings: Settings):
        """synthesize_script with empty turns produces a silence file."""
        engine = EdgeTTSEngine(test_settings)
        script = DialogueScript(
            track_number=1,
            track_type=TrackType.EXECUTIVE_SUMMARY,
            title="Empty Track",
            turns=[],
        )
        output_path = Path(test_settings.audio_output_dir) / "empty_track.mp3"
        result = await engine.synthesize_script(script, output_path)
        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_merge_tracks_empty(self, test_settings: Settings):
        """merge_tracks with empty list produces a silence file."""
        engine = EdgeTTSEngine(test_settings)
        output_path = Path(test_settings.audio_output_dir) / "merged_empty.mp3"
        result = await engine.merge_tracks([], output_path)
        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_merge_tracks_with_files(self, test_settings: Settings):
        """merge_tracks concatenates multiple MP3 files."""
        from tests.conftest import create_silent_mp3

        engine = EdgeTTSEngine(test_settings)
        track1 = Path(test_settings.audio_output_dir) / "t1.mp3"
        track2 = Path(test_settings.audio_output_dir) / "t2.mp3"
        track1.write_bytes(create_silent_mp3(1.0))
        track2.write_bytes(create_silent_mp3(1.0))

        output_path = Path(test_settings.audio_output_dir) / "merged.mp3"
        result = await engine.merge_tracks([track1, track2], output_path)
        assert result == output_path
        assert output_path.exists()

    def test_get_mp3_duration_seconds_existing_file(self, test_settings: Settings, tmp_path: Path):
        """get_mp3_duration_seconds returns positive for existing file."""
        from tests.conftest import create_silent_mp3
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(create_silent_mp3(5.0))
        duration = EdgeTTSEngine.get_mp3_duration_seconds(audio_file)
        assert duration > 0

    def test_get_mp3_duration_seconds_nonexistent(self, tmp_path: Path):
        """get_mp3_duration_seconds returns 0 for missing file."""
        duration = EdgeTTSEngine.get_mp3_duration_seconds(tmp_path / "nonexistent.mp3")
        assert duration == 0.0

    def test_generate_silence_returns_bytes(self):
        """_generate_silence returns valid MP3 bytes."""
        silence = EdgeTTSEngine._generate_silence(2.0)
        assert isinstance(silence, bytes)
        assert len(silence) > 0
        assert silence[:2] == b"\xff\xfb"


# ============================================================================
# DialogueScriptGenerator Tests
# ============================================================================

class TestDialogueScriptGenerator:
    """Tests for German 2-speaker dialogue script generation."""

    def test_generate_scripts_with_breakthroughs(self, test_settings: Settings):
        """Full script generation with multiple breakthroughs."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [
            DataFactory.create_breakthrough(raw_item_id=1, title="Tool A", breakthrough_score=9.5),
            DataFactory.create_breakthrough(raw_item_id=2, title="Tool B", breakthrough_score=8.8),
            DataFactory.create_breakthrough(raw_item_id=3, title="Tool C", breakthrough_score=8.0, category=Stage2Category.SERENDIPITY),
        ]
        scripts = gen.generate_scripts(breakthroughs)
        assert len(scripts) >= 3
        assert all(isinstance(s, DialogueScript) for s in scripts)
        assert scripts[0].track_type == TrackType.EXECUTIVE_SUMMARY
        assert all(s.track_number == i + 1 for i, s in enumerate(scripts))

    def test_generate_scripts_empty_list(self, test_settings: Settings):
        """Empty breakthrough list produces a fallback script."""
        gen = DialogueScriptGenerator(test_settings)
        scripts = gen.generate_scripts([])
        assert len(scripts) == 1
        assert scripts[0].track_type == TrackType.EXECUTIVE_SUMMARY
        assert len(scripts[0].turns) > 0
        assert "keine" in scripts[0].turns[0].text.lower() or "ruhig" in scripts[0].turns[0].text.lower()

    def test_generate_scripts_has_conrad_and_katja(self, test_settings: Settings):
        """Scripts include both Conrad and Katja speakers."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [DataFactory.create_breakthrough(raw_item_id=1)]
        scripts = gen.generate_scripts(breakthroughs)
        all_speakers = set()
        for s in scripts:
            for t in s.turns:
                all_speakers.add(t.speaker)
        assert "Conrad" in all_speakers
        assert "Katja" in all_speakers

    def test_generate_scripts_executive_summary_content(self, test_settings: Settings):
        """Executive summary mentions breakthrough count and titles."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [
            DataFactory.create_breakthrough(raw_item_id=1, title="Amazing Tool"),
            DataFactory.create_breakthrough(raw_item_id=2, title="Another Tool"),
        ]
        scripts = gen.generate_scripts(breakthroughs)
        exec_script = scripts[0]
        combined_text = " ".join(t.text for t in exec_script.turns)
        assert "Amazing Tool" in combined_text
        assert "Another Tool" in combined_text

    def test_generate_scripts_deep_dive_content(self, test_settings: Settings):
        """Deep dive tracks contain breakthrough details."""
        gen = DialogueScriptGenerator(test_settings)
        bt = DataFactory.create_breakthrough(
            raw_item_id=1, title="Deep Dive Tool", breakthrough_score=9.5
        )
        scripts = gen.generate_scripts([bt])
        deep_dive = [s for s in scripts if s.track_type in (TrackType.DEEP_DIVE_1, TrackType.DEEP_DIVE_2)]
        assert len(deep_dive) >= 1
        dd_text = " ".join(t.text for t in deep_dive[0].turns)
        assert "Deep Dive Tool" in dd_text

    def test_generate_scripts_serendipity_content(self, test_settings: Settings):
        """Serendipity track is generated when serendipity items exist."""
        gen = DialogueScriptGenerator(test_settings)
        serendipity_bt = DataFactory.create_breakthrough(
            raw_item_id=1, title="Wild Robot Thing",
            category=Stage2Category.SERENDIPITY, breakthrough_score=8.5
        )
        core_bt = DataFactory.create_breakthrough(
            raw_item_id=2, title="Core Tool", breakthrough_score=9.5
        )
        scripts = gen.generate_scripts([core_bt, serendipity_bt])
        serendipity_scripts = [s for s in scripts if s.track_type == TrackType.SERENDIPITY]
        assert len(serendipity_scripts) >= 1
        ser_text = " ".join(t.text for t in serendipity_scripts[0].turns)
        assert "Wild Robot Thing" in ser_text

    def test_generate_scripts_verdict_content(self, test_settings: Settings):
        """Verdict track mentions the top-scored breakthrough."""
        gen = DialogueScriptGenerator(test_settings)
        bt_top = DataFactory.create_breakthrough(
            raw_item_id=1, title="Top Pick", breakthrough_score=9.8
        )
        bt_low = DataFactory.create_breakthrough(
            raw_item_id=2, title="Low Pick", breakthrough_score=7.5
        )
        scripts = gen.generate_scripts([bt_top, bt_low])
        verdict_scripts = [s for s in scripts if s.track_type == TrackType.VERDICT]
        assert len(verdict_scripts) == 1
        verdict_text = " ".join(t.text for t in verdict_scripts[0].turns)
        assert "Top Pick" in verdict_text

    def test_generate_scripts_duration_adjustment(self, test_settings: Settings):
        """Total target duration stays within min/max bounds."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [DataFactory.create_breakthrough(raw_item_id=i) for i in range(1, 6)]
        scripts = gen.generate_scripts(breakthroughs, min_duration_minutes=15, max_duration_minutes=30)
        total = sum(s.target_duration_minutes for s in scripts)
        assert 15 <= total <= 30

    def test_generate_scripts_track_numbering(self, test_settings: Settings):
        """Track numbers are sequential starting from 1."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [DataFactory.create_breakthrough(raw_item_id=1)]
        scripts = gen.generate_scripts(breakthroughs)
        for i, s in enumerate(scripts):
            assert s.track_number == i + 1

    def test_generate_scripts_all_text_is_german(self, test_settings: Settings):
        """Generated dialogue text contains German words."""
        gen = DialogueScriptGenerator(test_settings)
        breakthroughs = [DataFactory.create_breakthrough(raw_item_id=1)]
        scripts = gen.generate_scripts(breakthroughs)
        all_text = " ".join(t.text for s in scripts for t in s.turns)
        german_indicators = ["der", "die", "das", "und", "ist", "wir", "mit", "fuer"]
        found = sum(1 for w in german_indicators if w in all_text.lower())
        assert found >= 2

    def test_generate_scripts_dialectic_controversy(self, test_settings: Settings):
        """Deep dive script implements authentic dialectic debate: thesis, antithesis, synthesis."""
        gen = DialogueScriptGenerator(test_settings)
        bt = DataFactory.create_breakthrough(
            raw_item_id=1,
            title="GPU Inference Engine",
            hardware_requirements="1x RTX 4090 24GB VRAM",
            license="Apache 2.0",
        )
        scripts = gen.generate_scripts([bt])
        deep_dives = [s for s in scripts if s.track_type in (TrackType.DEEP_DIVE_1, TrackType.DEEP_DIVE_2)]
        assert len(deep_dives) >= 1
        dd = deep_dives[0]

        # Verify speakers alternate Conrad and Katja
        speakers = [t.speaker for t in dd.turns]
        assert "Conrad" in speakers and "Katja" in speakers
        assert len(dd.turns) >= 6

        katja_text = " ".join(t.text for t in dd.turns if t.speaker == "Katja")
        conrad_text = " ".join(t.text for t in dd.turns if t.speaker == "Conrad")

        # Katja acts as skeptic / reality check (Antithesis)
        assert any(term in katja_text for term in ["Marketing", "VRAM", "4090", "Lizenz", "Production", "Sandbox"])
        # Conrad acts as visionary / practical quickstart (Thesis & Rebuttal)
        assert any(term in conrad_text for term in ["architektonisch", "Quickstart", "Docker", "Score"])

    @pytest.mark.asyncio
    async def test_generate_scripts_async_with_mock_llm(self, test_settings: Settings):
        """Asynchronous script generation using an LLM client."""
        from newsscout.audio.prompts import DialogueTurnPayload, TrackScriptPayload
        from newsscout.llm.base import BaseLLMClient

        mock_payload = TrackScriptPayload(
            turns=[
                DialogueTurnPayload(speaker="Conrad", text="Das ist ein bahnbrechendes Tool."),
                DialogueTurnPayload(speaker="Katja", text="Schauen wir uns die Relevanz in Produktion an."),
                DialogueTurnPayload(speaker="Conrad", text="Die Benchmarks sind 3x besser."),
                DialogueTurnPayload(speaker="Katja", text="Aber wie sieht die Latenz unter Last aus?"),
                DialogueTurnPayload(speaker="Conrad", text="Mit Docker ist es sofort testbar."),
                DialogueTurnPayload(speaker="Katja", text="Ein 15-Minuten-Spike ist es wert."),
            ]
        )

        mock_llm = MagicMock(spec=BaseLLMClient)
        mock_llm.generate_structured = AsyncMock(return_value=mock_payload)

        gen = DialogueScriptGenerator(test_settings, llm_client=mock_llm)
        bt = DataFactory.create_breakthrough(raw_item_id=1, title="Novel DB Engine")
        scripts = await gen.generate_scripts_async([bt])

        assert len(scripts) >= 3
        assert mock_llm.generate_structured.call_count >= 3
        # Check that turns from LLM payload were populated
        assert any(t.text == "Das ist ein bahnbrechendes Tool." for s in scripts for t in s.turns)
        assert any(t.text == "Aber wie sieht die Latenz unter Last aus?" for s in scripts for t in s.turns)

    @pytest.mark.asyncio
    async def test_generate_scripts_async_empty_breakthroughs(self, test_settings: Settings):
        """Empty breakthrough list returns fallback empty digest script without calling LLM."""
        from newsscout.llm.base import BaseLLMClient

        mock_llm = MagicMock(spec=BaseLLMClient)
        mock_llm.generate_structured = AsyncMock()

        gen = DialogueScriptGenerator(test_settings, llm_client=mock_llm)
        scripts = await gen.generate_scripts_async([])

        assert len(scripts) == 1
        assert scripts[0].track_type == TrackType.EXECUTIVE_SUMMARY
        mock_llm.generate_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_scripts_async_fallback_on_llm_error(self, test_settings: Settings):
        """LLM failure gracefully falls back to deterministic script generation."""
        from newsscout.llm.base import BaseLLMClient, LLMServerException

        mock_llm = MagicMock(spec=BaseLLMClient)
        mock_llm.generate_structured = AsyncMock(side_effect=LLMServerException("Quota exhausted / 503"))

        gen = DialogueScriptGenerator(test_settings, llm_client=mock_llm)
        bt = DataFactory.create_breakthrough(raw_item_id=1, title="Fallback Tool")
        scripts = await gen.generate_scripts_async([bt])

        assert len(scripts) >= 3
        # Fallback generated scripts contain breakthrough title
        all_text = " ".join(t.text for s in scripts for t in s.turns)
        assert "Fallback Tool" in all_text


# ============================================================================
# DigestGenerator Tests
# ============================================================================

class TestDigestGenerator:
    """Tests for the digest orchestrator."""

    @pytest.mark.asyncio
    async def test_generate_digest_with_mock_tts(self, migrated_db, test_settings: Settings):
        """Full digest generation with mock TTS engine."""
        from tests.conftest import create_silent_mp3

        # Insert a breakthrough into the DB
        bt = DataFactory.create_breakthrough(raw_item_id=1)
        params = bt.to_db_params()
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('github', 'test:1', 'Test', 'http://test.com', 'content', '{}', datetime('now'));
            """,
        )
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, repo_url, breakthrough_score, roi_score,
                category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                params["raw_item_id"], params["title"], params["repo_url"],
                params["breakthrough_score"], params["roi_score"], params["category"],
                params["tldr"], params["use_case"], params["comparison"],
                params["quickstart"], params["hardware_requirements"], params["license"],
                params["card_markdown"], params["evaluation_raw_json"],
                params["is_watchlisted"], params["evaluated_at"],
            ),
        )

        # Create mock TTS engine
        mock_tts = MagicMock(spec=BaseTTSEngine)
        mock_tts.synthesize_script = AsyncMock(
            side_effect=lambda script, path: Path(path).write_bytes(create_silent_mp3(3.0)) or path
        )
        mock_tts.merge_tracks = AsyncMock(
            side_effect=lambda paths, path: Path(path).write_bytes(create_silent_mp3(10.0)) or path
        )
        mock_tts.get_mp3_duration_seconds = MagicMock(return_value=180.0)
        mock_tts.settings = test_settings

        gen = DigestGenerator(migrated_db, test_settings, tts_engine=mock_tts)
        breakthroughs = [bt]
        bt.id = 1
        digest = await gen.generate_digest(
            slot=DigestSlot.MORNING,
            breakthroughs=breakthroughs,
        )

        assert digest.status == DigestStatus.SYNTHESIZED
        assert digest.id is not None
        assert len(digest.tracks) >= 3
        assert digest.total_duration_seconds > 0
        assert digest.merged_audio_path is not None

    @pytest.mark.asyncio
    async def test_generate_digest_empty_breakthroughs(self, migrated_db, test_settings: Settings):
        """Digest generation with no breakthroughs produces a fallback digest."""
        from tests.conftest import create_silent_mp3

        mock_tts = MagicMock(spec=BaseTTSEngine)
        mock_tts.synthesize_script = AsyncMock(
            side_effect=lambda script, path: Path(path).write_bytes(create_silent_mp3(2.0)) or path
        )
        mock_tts.merge_tracks = AsyncMock(
            side_effect=lambda paths, path: Path(path).write_bytes(create_silent_mp3(2.0)) or path
        )
        mock_tts.get_mp3_duration_seconds = MagicMock(return_value=120.0)
        mock_tts.settings = test_settings

        gen = DigestGenerator(migrated_db, test_settings, tts_engine=mock_tts)
        digest = await gen.generate_digest(
            slot=DigestSlot.AFTERNOON,
            breakthroughs=[],
        )

        assert digest.status == DigestStatus.SYNTHESIZED
        assert len(digest.tracks) >= 1
        assert digest.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_generate_digest_db_records_created(self, migrated_db, test_settings: Settings):
        """Digest and track records are persisted in the database."""
        from tests.conftest import create_silent_mp3

        mock_tts = MagicMock(spec=BaseTTSEngine)
        mock_tts.synthesize_script = AsyncMock(
            side_effect=lambda script, path: Path(path).write_bytes(create_silent_mp3(3.0)) or path
        )
        mock_tts.merge_tracks = AsyncMock(
            side_effect=lambda paths, path: Path(path).write_bytes(create_silent_mp3(15.0)) or path
        )
        mock_tts.get_mp3_duration_seconds = MagicMock(return_value=180.0)
        mock_tts.settings = test_settings

        gen = DigestGenerator(migrated_db, test_settings, tts_engine=mock_tts)
        bt = DataFactory.create_breakthrough(raw_item_id=1)
        bt.id = 1

        # Insert raw_item and breakthrough so FK constraints are satisfied
        await migrated_db.execute(
            """
            INSERT INTO raw_items (source, source_id, title, url, raw_content, metadata_json, ingested_at)
            VALUES ('github', 'test:1', 'Test', 'http://test.com', 'content', '{}', datetime('now'));
            """,
        )
        bt_params = bt.to_db_params()
        await migrated_db.execute(
            """
            INSERT INTO breakthroughs (raw_item_id, title, repo_url, breakthrough_score, roi_score,
                category, tldr, use_case, comparison, quickstart, hardware_requirements, license,
                card_markdown, evaluation_raw_json, is_watchlisted, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                bt_params["raw_item_id"], bt_params["title"], bt_params["repo_url"],
                bt_params["breakthrough_score"], bt_params["roi_score"], bt_params["category"],
                bt_params["tldr"], bt_params["use_case"], bt_params["comparison"],
                bt_params["quickstart"], bt_params["hardware_requirements"], bt_params["license"],
                bt_params["card_markdown"], bt_params["evaluation_raw_json"],
                bt_params["is_watchlisted"], bt_params["evaluated_at"],
            ),
        )

        digest = await gen.generate_digest(breakthroughs=[bt])

        # Verify digest record in DB
        digest_row = await migrated_db.fetch_one(
            "SELECT * FROM digests WHERE id = ?;", (digest.id,)
        )
        assert digest_row is not None
        assert digest_row["status"] == DigestStatus.SYNTHESIZED.value

        # Verify track records in DB
        track_rows = await migrated_db.fetch_all(
            "SELECT * FROM digest_tracks WHERE digest_id = ? ORDER BY track_number;",
            (digest.id,)
        )
        assert len(track_rows) >= 3

    @pytest.mark.asyncio
    async def test_generate_digest_failure_sets_status(self, migrated_db, test_settings: Settings):
        """Digest generation failure sets status to FAILED."""
        mock_tts = MagicMock(spec=BaseTTSEngine)
        mock_tts.synthesize_script = AsyncMock(side_effect=RuntimeError("TTS failed"))
        mock_tts.settings = test_settings

        gen = DigestGenerator(migrated_db, test_settings, tts_engine=mock_tts)
        bt = DataFactory.create_breakthrough(raw_item_id=1)

        with pytest.raises(RuntimeError):
            await gen.generate_digest(breakthroughs=[bt])

        # Check that the digest was marked as failed
        rows = await migrated_db.fetch_all("SELECT * FROM digests;")
        assert len(rows) >= 1
        assert rows[0]["status"] == DigestStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_generate_digest_afternoon_slot(self, migrated_db, test_settings: Settings):
        """Afternoon slot is correctly stored."""
        from tests.conftest import create_silent_mp3

        mock_tts = MagicMock(spec=BaseTTSEngine)
        mock_tts.synthesize_script = AsyncMock(
            side_effect=lambda script, path: Path(path).write_bytes(create_silent_mp3(3.0)) or path
        )
        mock_tts.merge_tracks = AsyncMock(
            side_effect=lambda paths, path: Path(path).write_bytes(create_silent_mp3(15.0)) or path
        )
        mock_tts.get_mp3_duration_seconds = MagicMock(return_value=180.0)
        mock_tts.settings = test_settings

        gen = DigestGenerator(migrated_db, test_settings, tts_engine=mock_tts)
        digest = await gen.generate_digest(
            slot=DigestSlot.AFTERNOON,
            breakthroughs=[DataFactory.create_breakthrough(raw_item_id=1)],
        )

        assert digest.slot == DigestSlot.AFTERNOON
        row = await migrated_db.fetch_one("SELECT * FROM digests WHERE id = ?;", (digest.id,))
        assert row["slot"] == DigestSlot.AFTERNOON.value


# ============================================================================
# Helper: Mock async stream for edge_tts.Communicate.stream()
# ============================================================================

class MockAsyncStream:
    """Mock async iterator simulating edge_tts Communicate.stream().

    edge_tts Communicate.stream() is called as a method and returns an
    async iterator, so we need __call__ to return self (the async iterator).
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._index = 0

    def __call__(self) -> "MockAsyncStream":
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._index]
        self._index += 1
        return item
