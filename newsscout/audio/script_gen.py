"""newsscout.audio.script_gen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dialogue script generator for German 2-speaker audio digest tracks.

Produces DialogueScript objects for each of the 5 track types:
  1. Executive Summary  (2-3 min)
  2. Deep Dive 1        (6-10 min)
  3. Deep Dive 2        (6-10 min)
  4. Serendipity        (5-7 min)
  5. Evening Verdict    (2 min)

Supports dynamic LLM-driven dialectic dialogue generation (Conrad & Katja)
with domain-tailored technical debate, plus a robust deterministic fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

from newsscout.audio.prompts import (
    DIALOGUE_SYSTEM_PROMPT,
    TrackScriptPayload,
    format_deep_dive_prompt,
    format_executive_summary_prompt,
    format_serendipity_prompt,
    format_verdict_prompt,
)
from newsscout.config import Settings, get_settings
from newsscout.llm.base import BaseLLMClient
from newsscout.llm.factory import create_llm_client
from newsscout.storage.models import (
    AudioTurn,
    Breakthrough,
    DialogueScript,
    Stage2Category,
    TrackType,
)

logger = logging.getLogger("newsscout.audio.script_gen")

WORDS_PER_MINUTE = 130

TRACK_DURATIONS: dict[TrackType, float] = {
    TrackType.EXECUTIVE_SUMMARY: 3.0,
    TrackType.DEEP_DIVE_1: 8.0,
    TrackType.DEEP_DIVE_2: 8.0,
    TrackType.SERENDIPITY: 6.0,
    TrackType.VERDICT: 2.0,
}

TRACK_TITLES: dict[TrackType, str] = {
    TrackType.EXECUTIVE_SUMMARY: "Executive Summary & Tagesagenda",
    TrackType.DEEP_DIVE_1: "Deep Dive 1",
    TrackType.DEEP_DIVE_2: "Deep Dive 2",
    TrackType.SERENDIPITY: "Serendipity: Ueber den Tellerrand",
    TrackType.VERDICT: "Evening Verdict & Empfehlungen",
}


class DialogueScriptGenerator:
    """Generates German 2-speaker dialogue scripts from breakthrough data.
    
    Uses the configured Universal LLM client to produce organic, domain-tailored
    controversies between Conrad (Innovator) and Katja (Principal SRE/Skeptic).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if llm_client is not None:
            self.llm_client: Optional[BaseLLMClient] = llm_client
        elif self.settings.has_llm_credentials:
            try:
                self.llm_client = create_llm_client(self.settings)
            except Exception as e:
                logger.warning("Could not instantiate LLM client for script gen: %s", e)
                self.llm_client = None
        else:
            self.llm_client = None

    async def generate_scripts_async(
        self,
        breakthroughs: list[Breakthrough],
        min_duration_minutes: int = 15,
        max_duration_minutes: int = 30,
    ) -> list[DialogueScript]:
        """Asynchronously generate dialogue scripts using the active LLM client.
        
        Falls back automatically to deterministic script generation if the LLM client
        is unconfigured, quota-exhausted, or encounters server errors.
        """
        if not breakthroughs:
            return [self._generate_empty_digest_script()]

        if self.llm_client is not None:
            try:
                return await self._generate_scripts_with_llm(
                    breakthroughs, min_duration_minutes, max_duration_minutes
                )
            except Exception as e:
                logger.warning(
                    "LLM dialogue generation failed (%s). Falling back to deterministic scripts.", e
                )

        return self.generate_scripts(
            breakthroughs, min_duration_minutes, max_duration_minutes
        )

    async def _generate_scripts_with_llm(
        self,
        breakthroughs: list[Breakthrough],
        min_duration_minutes: int,
        max_duration_minutes: int,
    ) -> list[DialogueScript]:
        """Generates all 5 tracks dynamically using the LLM client."""
        core_items = [b for b in breakthroughs if b.category == Stage2Category.CORE]
        serendipity_items = [b for b in breakthroughs if b.category == Stage2Category.SERENDIPITY]

        core_items.sort(key=lambda b: b.breakthrough_score, reverse=True)
        serendipity_items.sort(key=lambda b: b.breakthrough_score, reverse=True)

        scripts: list[DialogueScript] = []

        # 1. Executive Summary
        exec_script = await self._gen_executive_summary_llm(breakthroughs)
        scripts.append(exec_script)

        # 2. Deep Dives
        deep_dive_items = core_items[:2] if core_items else breakthroughs[:2]
        if len(deep_dive_items) >= 1:
            scripts.append(
                await self._gen_deep_dive_llm(deep_dive_items[0], TrackType.DEEP_DIVE_1, 2)
            )
        if len(deep_dive_items) >= 2:
            scripts.append(
                await self._gen_deep_dive_llm(deep_dive_items[1], TrackType.DEEP_DIVE_2, 3)
            )

        # 3. Serendipity
        ser_items = serendipity_items or (breakthroughs[2:] if len(breakthroughs) > 2 else [breakthroughs[-1]])
        scripts.append(await self._gen_serendipity_llm(ser_items, len(scripts) + 1))

        # 4. Evening Verdict
        scripts.append(await self._gen_verdict_llm(breakthroughs, len(scripts) + 1))

        for i, script in enumerate(scripts, start=1):
            script.track_number = i

        self._adjust_durations(scripts, min_duration_minutes, max_duration_minutes)
        return scripts

    async def _gen_executive_summary_llm(
        self, breakthroughs: list[Breakthrough]
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.EXECUTIVE_SUMMARY]
        prompt = format_executive_summary_prompt(breakthroughs, int(target_min))
        payload: TrackScriptPayload = await self.llm_client.generate_structured(  # type: ignore[union-attr]
            prompt=prompt,
            response_model=TrackScriptPayload,
            system_instruction=DIALOGUE_SYSTEM_PROMPT,
            temperature=0.3,
        )
        turns = [AudioTurn(speaker=t.speaker, text=t.text) for t in payload.turns]
        return DialogueScript(
            track_number=1,
            track_type=TrackType.EXECUTIVE_SUMMARY,
            title=TRACK_TITLES[TrackType.EXECUTIVE_SUMMARY],
            target_duration_minutes=target_min,
            turns=turns,
        )

    async def _gen_deep_dive_llm(
        self, breakthrough: Breakthrough, track_type: TrackType, track_number: int
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[track_type]
        title = f"{TRACK_TITLES[track_type]}: {breakthrough.title}"
        prompt = format_deep_dive_prompt(breakthrough, title, int(target_min))
        payload: TrackScriptPayload = await self.llm_client.generate_structured(  # type: ignore[union-attr]
            prompt=prompt,
            response_model=TrackScriptPayload,
            system_instruction=DIALOGUE_SYSTEM_PROMPT,
            temperature=0.3,
        )
        turns = [AudioTurn(speaker=t.speaker, text=t.text) for t in payload.turns]
        return DialogueScript(
            track_number=track_number,
            track_type=track_type,
            title=title,
            target_duration_minutes=target_min,
            turns=turns,
        )

    async def _gen_serendipity_llm(
        self, breakthroughs: list[Breakthrough], track_number: int
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.SERENDIPITY]
        prompt = format_serendipity_prompt(breakthroughs, int(target_min))
        payload: TrackScriptPayload = await self.llm_client.generate_structured(  # type: ignore[union-attr]
            prompt=prompt,
            response_model=TrackScriptPayload,
            system_instruction=DIALOGUE_SYSTEM_PROMPT,
            temperature=0.3,
        )
        turns = [AudioTurn(speaker=t.speaker, text=t.text) for t in payload.turns]
        return DialogueScript(
            track_number=track_number,
            track_type=TrackType.SERENDIPITY,
            title=TRACK_TITLES[TrackType.SERENDIPITY],
            target_duration_minutes=target_min,
            turns=turns,
        )

    async def _gen_verdict_llm(
        self, breakthroughs: list[Breakthrough], track_number: int
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.VERDICT]
        prompt = format_verdict_prompt(breakthroughs, int(target_min))
        payload: TrackScriptPayload = await self.llm_client.generate_structured(  # type: ignore[union-attr]
            prompt=prompt,
            response_model=TrackScriptPayload,
            system_instruction=DIALOGUE_SYSTEM_PROMPT,
            temperature=0.3,
        )
        turns = [AudioTurn(speaker=t.speaker, text=t.text) for t in payload.turns]
        return DialogueScript(
            track_number=track_number,
            track_type=TrackType.VERDICT,
            title=TRACK_TITLES[TrackType.VERDICT],
            target_duration_minutes=target_min,
            turns=turns,
        )

    def generate_scripts(
        self,
        breakthroughs: list[Breakthrough],
        min_duration_minutes: int = 15,
        max_duration_minutes: int = 30,
    ) -> list[DialogueScript]:
        """Generate all dialogue scripts for a digest deterministically."""
        if not breakthroughs:
            return [self._generate_empty_digest_script()]

        core_items = [
            b for b in breakthroughs
            if b.category == Stage2Category.CORE
        ]
        serendipity_items = [
            b for b in breakthroughs
            if b.category == Stage2Category.SERENDIPITY
        ]

        core_items.sort(key=lambda b: b.breakthrough_score, reverse=True)
        serendipity_items.sort(key=lambda b: b.breakthrough_score, reverse=True)

        scripts: list[DialogueScript] = []

        scripts.append(self._gen_executive_summary(breakthroughs))

        deep_dive_items = core_items[:2] if core_items else breakthroughs[:2]
        if len(deep_dive_items) >= 1:
            scripts.append(self._gen_deep_dive(deep_dive_items[0], TrackType.DEEP_DIVE_1))
        if len(deep_dive_items) >= 2:
            scripts.append(self._gen_deep_dive(deep_dive_items[1], TrackType.DEEP_DIVE_2))

        if serendipity_items:
            scripts.append(self._gen_serendipity(serendipity_items))
        elif len(breakthroughs) > 2:
            scripts.append(self._gen_serendipity([breakthroughs[-1]]))

        scripts.append(self._gen_verdict(breakthroughs))

        for i, script in enumerate(scripts, start=1):
            script.track_number = i

        self._adjust_durations(scripts, min_duration_minutes, max_duration_minutes)

        return scripts

    def _gen_executive_summary(
        self, breakthroughs: list[Breakthrough]
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.EXECUTIVE_SUMMARY]
        turns: list[AudioTurn] = []

        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                f"Willkommen zum NewsScout Audio Digest. "
                f"Ich bin Conrad und an meiner Seite wie immer Katja. "
                f"Heute haben wir {len(breakthroughs)} neue Kandidaten durch unsere Filter geschleust."
            ),
        ))

        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                f"Hallo zusammen! Und ich sage gleich vorweg: Ich habe mir die Rohdaten "
                f"und Commits genau angeschaut. Da ist wieder eine gehoerige Portion Hype "
                f"im Umlauf. Aber bei zwei Funden bin ich wirklich gespannt, wie du die "
                f"nachher in der Diskussion verteidigen willst, Conrad."
            ),
        ))

        turns.append(AudioTurn(
            speaker="Conrad",
            text="Herausforderung angenommen! Gehen wir kurz die Tagesagenda durch:",
        ))

        for i, b in enumerate(breakthroughs):
            speaker = "Conrad" if i % 2 == 0 else "Katja"
            other = "Katja" if speaker == "Conrad" else "Conrad"

            turns.append(AudioTurn(
                speaker=speaker,
                text=f"Nummer {i + 1}: {b.title}. {b.tldr}",
            ))

            if i < len(breakthroughs) - 1:
                if other == "Katja":
                    turns.append(AudioTurn(
                        speaker="Katja",
                        text="Klingt ambitioniert. Mal sehen, ob die Zahlen in der Praxis standhalten. Was kommt als Naechstes?",
                    ))
                else:
                    turns.append(AudioTurn(
                        speaker="Conrad",
                        text="Und die Baseline-Messungen dazu schauen wir uns gleich im Detail an. Weiter geht's:",
                    ))

        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                "Das ist die Uebersicht. Gleich im ersten Deep Dive nehmen wir die "
                "Architektur und die Trade-offs schonungslos auseinander. Bleibt dran!"
            ),
        ))

        return DialogueScript(
            track_number=1,
            track_type=TrackType.EXECUTIVE_SUMMARY,
            title=TRACK_TITLES[TrackType.EXECUTIVE_SUMMARY],
            target_duration_minutes=target_min,
            turns=turns,
        )

    def _gen_deep_dive(
        self, breakthrough: Breakthrough, track_type: TrackType
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[track_type]
        turns: list[AudioTurn] = []

        # 1. These: Conrad stellt die technologische Innovation vor
        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                f"Kommen wir zum Deep Dive: {breakthrough.title}. "
                f"Katja, das ist architektonisch genau das, worauf wir gewartet haben! "
                f"{breakthrough.tldr}"
            ),
        ))

        # 2. Antithese I: Katja hinterfragt den praktischen Nutzen und Hype
        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                f"Moment, Conrad, brems deine Begeisterung mal kurz. "
                f"Das Papier und die README-Benchmarks sind geduldig. "
                f"Wo liegt denn hier der reale Mehrwert fuer unseren Tech-Stack, "
                f"abseits vom ueblichen Marketing-Getoese?"
            ),
        ))

        # 3. Replik I: Conrad verteidigt den Use Case und Baseline-Vergleich
        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                f"Der Workflow-Vorteil ist absolut greifbar: {breakthrough.use_case}. "
                f"Und schau dir den Vergleich an: {breakthrough.comparison}. "
                f"Das ist keine theoretische Spielerei, das loest ein echtes Bottleneck!"
            ),
        ))

        # 4. Antithese II: Katja bohrt bei Hardware, Skalierbarkeit und Lizenz nach
        hardware = breakthrough.hardware_requirements or "Unspezifiziert"
        license_str = breakthrough.license or "Unbekannt"
        
        if any(kw in hardware.lower() for kw in ("4090", "vram", "gpu", "cuda", "gb")):
            hw_critique = (
                f"Schau dir die Ressourcen an: Gefordert sind {hardware}. "
                f"Auf einer 24-Gigabyte-Karte wie unserer 4090 kann das bei laengeren Kontexten "
                f"oder parallelen Agent-Prozessen verdammt eng werden. "
                f"Wenn der KV-Cache volllaeuft, riskieren wir Out-of-Memory-Crashes. "
            )
        else:
            hw_critique = (
                f"Die Hardware-Anforderung ist zwar mit '{hardware}' angegeben, "
                f"aber wie stabil skaliert das unter Dauerlast auf dem Raspberry Pi oder Server? "
            )

        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                f"Trotzdem muessen wir ueber die versteckten Kosten reden. "
                f"{hw_critique}"
                f"Dazu kommt die Lizenzfrage: Status ist {license_str}. "
                f"Koennen wir das ohne rechtliche Fallstricke oder Vendor-Lock-in betreiben?"
            ),
        ))

        # 5. Replik II: Conrad raeumt Risiken ein, praesentiert aber den Quickstart als isolierten Testpfad
        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                f"Deine Skepsis ist berechtigt, Katja. Genau deshalb setzen wir das nicht ungetestet produktiv ein. "
                f"Aber der Einstieg ist extrem schmerzfrei. Der Quickstart-Befehl lautet: "
                f"{breakthrough.quickstart}. "
                f"In einem isolierten Docker-Container oder venv koennen wir das heute Abend risikofrei verifizieren."
            ),
        ))

        # 6. Synthese: Katja und Conrad wägen Score und ROI ab und formulieren ein ausgewogenes Urteil
        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                f"In Ordnung, ein isolierter Smoke-Test ist fair. "
                f"Der Breakthrough Score steht bei {breakthrough.breakthrough_score:.1f} von 10, "
                f"bei einem ROI-Wert von {breakthrough.roi_score:.1f}. "
                f"Mein Urteil: Fuer die Experimentier-Sandbox ein klares Go, "
                f"aber fuer kritische Production-Pipelines warten wir noch mindestens zwei Minor-Releases ab."
            ),
        ))

        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                f"Damit kann ich voll leben. Innovation mit Bedacht: "
                f"Score {breakthrough.breakthrough_score:.1f} rechtfertigt das Anschauen, "
                f"aber wir behalten die Finger am Puls."
            ),
        ))

        return DialogueScript(
            track_number=2 if track_type == TrackType.DEEP_DIVE_1 else 3,
            track_type=track_type,
            title=f"{TRACK_TITLES[track_type]}: {breakthrough.title}",
            target_duration_minutes=target_min,
            turns=turns,
        )

    def _gen_serendipity(
        self, breakthroughs: list[Breakthrough]
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.SERENDIPITY]
        turns: list[AudioTurn] = []

        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                "Und jetzt zu unserem Serendipity-Track. "
                "Hier verlassen wir bewusst unsere gewohnte Entwickler-Blase."
            ),
        ))

        for b in breakthroughs:
            turns.append(AudioTurn(
                speaker="Conrad",
                text=f"Schau dir das an: {b.title}. {b.tldr}",
            ))
            turns.append(AudioTurn(
                speaker="Katja",
                text=(
                    f"Das stammt zwar nicht aus unserem Kern-Stack, aber der Ansatz ist faszinierend: "
                    f"{b.use_case}. Trotzdem frage ich mich: Ist das schon praxistauglich "
                    f"oder eher Grundlagenforschung?"
                ),
            ))
            turns.append(AudioTurn(
                speaker="Conrad",
                text=(
                    f"Selbst wenn es experimentell ist: Genau solche Querdenker-Architekturen "
                    f"bringen oft die entscheidenden Impulse fuer unsere eigenen Systeme."
                ),
            ))

        turns.append(AudioTurn(
            speaker="Katja",
            text="Stimmt. Inspiration schadet nie, solange man nicht voreilig die Architektur danach umbaut.",
        ))

        return DialogueScript(
            track_number=4,
            track_type=TrackType.SERENDIPITY,
            title=TRACK_TITLES[TrackType.SERENDIPITY],
            target_duration_minutes=target_min,
            turns=turns,
        )

    def _gen_verdict(
        self, breakthroughs: list[Breakthrough]
    ) -> DialogueScript:
        target_min = TRACK_DURATIONS[TrackType.VERDICT]
        turns: list[AudioTurn] = []

        top_bt = (
            max(breakthroughs, key=lambda b: b.breakthrough_score)
            if breakthroughs
            else None
        )

        turns.append(AudioTurn(
            speaker="Conrad",
            text=(
                "Kommen wir zum Evening Action Verdict. "
                "Was nehmen wir heute ganz konkret mit an die Tastatur?"
            ),
        ))

        if top_bt:
            turns.append(AudioTurn(
                speaker="Katja",
                text=(
                    f"Unser klarer Top-Pick heute ist {top_bt.title} mit einem "
                    f"Breakthrough Score von {top_bt.breakthrough_score:.1f}. "
                    f"Ich gebe zu: Trotz meiner Skepsis rechtfertigt der ROI-Wert "
                    f"einen 15-Minuten-Spike heute Abend."
                ),
            ))
            turns.append(AudioTurn(
                speaker="Conrad",
                text=(
                    f"Absolut! Der Quickstart geht schnell von der Hand: "
                    f"{top_bt.quickstart}. "
                    f"Wer heute Abend noch etwas ausprobieren will, hat damit den besten Hebel."
                ),
            ))
        else:
            turns.append(AudioTurn(
                speaker="Katja",
                text="Heute war die Ausbeute eher verhalten. Lieber die bestehende Codebase konsolidieren.",
            ))
            turns.append(AudioTurn(
                speaker="Conrad",
                text="Manchmal ist kein Tool das beste Tool. Morgen scannen wir weiter.",
            ))

        turns.append(AudioTurn(
            speaker="Katja",
            text=(
                "Und wie immer: Wenn ihr eine andere Meinung dazu habt "
                "oder das Tool laengst kennt, gebt uns Feedback – per Telegram-Button, "
                "oder einfach als Emoji-Reaktion auf WhatsApp und Signal!"
            ),
        ))

        turns.append(AudioTurn(
            speaker="Conrad",
            text="Danke fuer's Zuhoeren und bis zum naechsten NewsScout Audio Digest! Tschuess!",
        ))

        return DialogueScript(
            track_number=5,
            track_type=TrackType.VERDICT,
            title=TRACK_TITLES[TrackType.VERDICT],
            target_duration_minutes=target_min,
            turns=turns,
        )

    def _generate_empty_digest_script(self) -> DialogueScript:
        turns = [
            AudioTurn(
                speaker="Conrad",
                text=(
                    "Willkommen zum NewsScout Audio Digest. "
                    "Leute, wir haben heute einen ruhigen Tag."
                ),
            ),
            AudioTurn(
                speaker="Katja",
                text=(
                    "Ja, heute konnten wir keine signifikanten Durchbrueche finden. "
                    "Aber das ist okay - nicht jeder Tag bringt Revolutionen."
                ),
            ),
            AudioTurn(
                speaker="Conrad",
                text=(
                    "Genau. Wir bleiben dran und melden uns sofort, "
                    "wenn etwas Bahnbrechendes passiert."
                ),
            ),
            AudioTurn(
                speaker="Katja",
                text="Bis zum naechsten Digest! Tschuess!",
            ),
        ]
        return DialogueScript(
            track_number=1,
            track_type=TrackType.EXECUTIVE_SUMMARY,
            title="NewsScout Digest - Keine Funde heute",
            target_duration_minutes=2.0,
            turns=turns,
        )

    @staticmethod
    def _adjust_durations(
        scripts: list[DialogueScript],
        min_minutes: int,
        max_minutes: int,
    ) -> None:
        total = sum(s.target_duration_minutes for s in scripts)
        if total < min_minutes:
            factor = min_minutes / total if total > 0 else 1.0
            for s in scripts:
                s.target_duration_minutes = round(
                    s.target_duration_minutes * factor, 1
                )
        elif total > max_minutes:
            factor = max_minutes / total
            for s in scripts:
                s.target_duration_minutes = round(
                    s.target_duration_minutes * factor, 1
                )
