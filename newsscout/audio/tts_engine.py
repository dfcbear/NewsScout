"""newsscout.audio.tts_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Modular TTS engine abstraction with edge-tts as the default backend.

Provides a pluggable architecture for text-to-speech synthesis:
  * BaseTTSEngine   - abstract interface
  * EdgeTTSEngine   - default implementation using Microsoft Edge TTS

All audio concatenation uses ffmpeg stream-copy (no re-encoding) for
fast, lossless track assembly on Raspberry Pi 5.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from newsscout.config import Settings, get_settings
from newsscout.storage.models import AudioTurn, DialogueScript

logger = logging.getLogger("newsscout.audio.tts_engine")

# Speaker name -> config voice field mapping
SPEAKER_VOICE_MAP: dict[str, str] = {
    "Conrad": "tts_voice_male",
    "Katja": "tts_voice_female",
}


class BaseTTSEngine(ABC):
    """Abstract base class for pluggable TTS engines."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    async def synthesize_turn(self, turn: AudioTurn) -> bytes:
        """Synthesize a single AudioTurn to MP3 bytes."""
        ...

    @abstractmethod
    async def synthesize_script(
        self, script: DialogueScript, output_path: Path
    ) -> Path:
        """Synthesize all turns in a DialogueScript into a single MP3 file."""
        ...

    @abstractmethod
    async def merge_tracks(
        self, track_paths: list[Path], output_path: Path
    ) -> Path:
        """Merge multiple track MP3s into a single playlist MP3."""
        ...

    # -- shared utilities --

    @staticmethod
    def estimate_duration_seconds(text: str, wpm: int = 130) -> float:
        """Estimate spoken duration from word count at given words-per-minute."""
        word_count = len(text.split())
        return (word_count / wpm) * 60.0

    @staticmethod
    def get_speaker_voice(speaker: str, settings: Settings) -> str:
        """Resolve a speaker name to a configured TTS voice identifier."""
        field_name = SPEAKER_VOICE_MAP.get(speaker, "tts_voice_male")
        return getattr(settings, field_name, settings.tts_voice_male)


class EdgeTTSEngine(BaseTTSEngine):
    """Default TTS engine using Microsoft Edge Neural TTS (edge-tts library).

    Requires the `edge-tts` package and `ffmpeg` on PATH.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings)
        self._output_dir = Path(self.settings.audio_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize_turn(self, turn: AudioTurn) -> bytes:
        """Synthesize a single AudioTurn to MP3 bytes using edge-tts."""
        import edge_tts

        voice = self.get_speaker_voice(turn.speaker, self.settings)
        communicate = edge_tts.Communicate(turn.text, voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        audio_bytes = b"".join(chunks)

        if not audio_bytes:
            logger.warning(
                "Empty audio for turn (speaker=%s, text=%s...)",
                turn.speaker,
                turn.text[:50],
            )

        return audio_bytes

    async def synthesize_script(
        self, script: DialogueScript, output_path: Path
    ) -> Path:
        """Synthesize all turns in a DialogueScript into a single MP3 file.

        Each turn is synthesized individually, then concatenated via
        ffmpeg stream-copy. Pauses between turns are inserted as silence.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not script.turns:
            output_path.write_bytes(self._generate_silence(1.0))
            return output_path

        temp_files: list[Path] = []
        try:
            for i, turn in enumerate(script.turns):
                audio_bytes = await self.synthesize_turn(turn)
                temp_path = output_path.parent / f"_turn_{i:03d}.mp3"
                temp_path.write_bytes(audio_bytes)
                temp_files.append(temp_path)

                if turn.pause_after_ms > 0:
                    pause_path = output_path.parent / f"_pause_{i:03d}.mp3"
                    pause_path.write_bytes(
                        self._generate_silence(turn.pause_after_ms / 1000.0)
                    )
                    temp_files.append(pause_path)

            await self._ffmpeg_concat(temp_files, output_path)
        finally:
            for f in temp_files:
                try:
                    f.unlink()
                except OSError:
                    pass

        logger.info(
            "Synthesized track %d (%s) -> %s (%d turns)",
            script.track_number,
            script.track_type.value,
            output_path,
            len(script.turns),
        )
        return output_path

    async def merge_tracks(
        self, track_paths: list[Path], output_path: Path
    ) -> Path:
        """Merge multiple track MP3s into a single playlist MP3."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not track_paths:
            output_path.write_bytes(self._generate_silence(1.0))
            return output_path

        await self._ffmpeg_concat(track_paths, output_path)

        logger.info(
            "Merged %d tracks -> %s",
            len(track_paths),
            output_path,
        )
        return output_path

    @staticmethod
    def _generate_silence(duration_seconds: float) -> bytes:
        """Generate a minimal valid silent MP3 for pauses."""
        header = bytes([0xFF, 0xFB, 0x90, 0x04])
        frame_len = 417
        side_info = b"\x00" * 32
        payload = b"\x00" * (frame_len - 4 - 32)
        single_frame = header + side_info + payload
        frames = max(2, int(duration_seconds * 44100 / 1152))
        return single_frame * frames

    @staticmethod
    async def _ffmpeg_concat(
        input_paths: list[Path], output_path: Path
    ) -> None:
        """Concatenate MP3 files using ffmpeg stream-copy (no re-encoding)."""
        # Escape path for FFmpeg concat demuxer: use forward slashes and escape single quotes
        def _escape_path(p: Path) -> str:
            clean = p.resolve().as_posix().replace("'", "'\\''")
            return f"file '{clean}'"

        list_content = "\n".join(_escape_path(p) for p in input_paths)
        list_file = output_path.parent / "_concat_list.txt"
        list_file.write_text(list_content, encoding="utf-8")

        try:
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(output_path),
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    logger.error(
                        "ffmpeg concat failed (rc=%d): %s",
                        proc.returncode,
                        stderr.decode()[:500] if stderr else "",
                    )
                    output_path.write_bytes(b"".join(
                        p.read_bytes() for p in input_paths
                    ))
            except (FileNotFoundError, OSError) as err:
                logger.warning("ffmpeg not found (%s), falling back to raw byte concatenation", err)
                output_path.write_bytes(b"".join(
                    p.read_bytes() for p in input_paths
                ))
        finally:
            try:
                list_file.unlink()
            except OSError:
                pass

    @staticmethod
    def get_mp3_duration_seconds(audio_path: Path) -> float:
        """Estimate MP3 duration from file size (rough heuristic for tests)."""
        if not audio_path.exists():
            return 0.0
        file_size = audio_path.stat().st_size
        return file_size / 16000.0
