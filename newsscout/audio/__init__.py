"""newsscout.audio
~~~~~~~~~~~~~~~~~~~
Audio digest generation package: TTS synthesis, dialogue scripting,
and digest orchestration for twice-daily German 2-speaker briefings.
"""

from newsscout.audio.tts_engine import BaseTTSEngine, EdgeTTSEngine
from newsscout.audio.script_gen import DialogueScriptGenerator
from newsscout.audio.digest_gen import DigestGenerator

__all__ = [
    "BaseTTSEngine",
    "EdgeTTSEngine",
    "DialogueScriptGenerator",
    "DigestGenerator",
]
