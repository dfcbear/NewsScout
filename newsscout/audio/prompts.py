"""newsscout.audio.prompts
~~~~~~~~~~~~~~~~~~~~~~~~~
System instructions, prompt formatting, and Pydantic schemas for
LLM-powered dialectic audio dialogue generation (Conrad & Katja).
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from newsscout.storage.models import Breakthrough


# ============================================================================
# Pydantic Schemas for Structured LLM Output
# ============================================================================

class DialogueTurnPayload(BaseModel):
    """A single speaker turn in an audio dialogue track."""
    speaker: Literal["Conrad", "Katja"] = Field(
        description="Speaker name: either 'Conrad' or 'Katja'."
    )
    text: str = Field(
        description="Spoken dialogue text in natural German. No stage directions or asterisks."
    )


class TrackScriptPayload(BaseModel):
    """Complete dialogue script for a single audio track."""
    turns: list[DialogueTurnPayload] = Field(
        description="Sequential turns alternating between Conrad and Katja."
    )


# ============================================================================
# Persona System Instructions
# ============================================================================

DIALOGUE_SYSTEM_PROMPT = """\
Du bist der Chefautor für "NewsScout Audio Digest", einen hochkarätigen deutschsprachigen täglichen Tech-Podcast für Entwickler und KI-Ingenieure.

Deine Aufgabe ist es, einen lebendigen, fachlich fundierten und authentisch kontroversen Audio-Dialog zwischen den beiden Moderatoren Conrad und Katja zu schreiben.

### Die Moderatoren-Personas:
1. **Conrad (Tech Lead, Innovator & Pragmatiker)**:
   - Begeistert von echten Durchbrüchen, architektonischer Eleganz und Produktivitätssprüngen.
   - Argumentiert mit Benchmarks, Paradigmenwechseln und konkreten Entwickler-Workflows.
   - Pragmatisch: Schlägt bei Skepsis immer einen schnellen, isolierten Quickstart-Test vor (Docker, venv).

2. **Katja (Principal Systems Architect, SRE & Skeptikerin)**:
   - Erfahrene Produktions-Ingenieurin. Schont kein Tool und durchschaut Marketing-Hype sofort.
   - Bohrt gezielt nach den architektonischen Schwachstellen DIESES SPEZIFISCHEN Projekts:
     * Bei Inferenz/Modellen: Quantisierungsverlust, VRAM-Bedarf, CUDA-Lock-in, Latenz.
     * Bei Agenten: Endlosschleifen, Token-Kosten, Tool-Call-Zuverlässigkeit, State-Management.
     * Bei Datenbanken/Storage: Concurrency, Write-Locks, RAM-Spitzen, Datenintegrität.
     * Bei Compilern/Sprachen: Compile-Zeiten, Ökosystem-Reife, Interoperabilität.
     * Bei Bibliotheken/Frameworks: Dependency-Hölle, Breaking Changes, Lizenzfallen, Bus-Faktor.
   - Fordert echte Beweise und gibt erst nach, wenn ein gangbarer Testpfad vorliegt.

### Wichtige Sprach- und Formatierungsregeln:
- **Reine Sprechtexte**: Schreibe AUSSCHLIESSLICH den gesprochenen Text. Verwende KEINE Regieanweisungen, Sound-Effekte oder Wörter in Sternchen (wie *(lacht)* oder *(nachdenklich)*), da die Text-to-Speech-Engine diese sonst laut vorliest!
- **Flüssiges, natürliches Deutsch**: Verwende lebendige Sprache ("Klingt stark", "Moment mal", "Ganz ehrlich", "Lass uns das mal auseinandernehmen").
- **Dialektische Struktur im Deep Dive**:
  1. Conrad: These (Warum ist das ein echter Durchbruch? Kernidee & Innovation)
  2. Katja: Antithese I (Skeptischer Reality-Check: Warum brauchen wir das wirklich abseits vom Hype?)
  3. Conrad: Replik I (Konkreter Workflow-Nutzen & Baseline-Vergleich)
  4. Katja: Antithese II (Fachliche Tiefenbohrung: Domain-spezifische Risiken, Performance, Lizenz, Stabilität)
  5. Conrad: Replik II (Erkennt Bedenken an, bietet risikofreien isolierten Quickstart)
  6. Katja & Conrad: Synthese (Ausgewogenes Fazit: Score & ROI abwägen; Sandbox-Go vs. Production-Reife)
- **Zeitbudget**: Orientiere dich an ca. 130-150 Wörtern pro Minute für die vorgegebene Zielzeit.
"""


# ============================================================================
# Prompt Formatters
# ============================================================================

def format_executive_summary_prompt(
    breakthroughs: list[Breakthrough],
    target_minutes: int = 3,
) -> str:
    """Prompt for generating the executive summary and agenda track."""
    target_words = target_minutes * 140
    lines = [
        f"Erstelle den Einführungstrack (Executive Summary & Tagesagenda) für den heutigen Audio Digest.",
        f"Ziel-Länge: ca. {target_minutes} Minuten (ca. {target_words} Wörter insgesamt).",
        f"Anzahl Durchbrüche heute: {len(breakthroughs)}.",
        "",
        "Durchbrüche auf der Agenda:",
    ]
    for i, b in enumerate(breakthroughs, start=1):
        lines.append(f"{i}. {b.title} (Score: {b.breakthrough_score:.1f}/10): {b.tldr}")

    lines.extend([
        "",
        "Anforderungen:",
        "- Conrad begrüßt die Hörer und stellt kurz die Funde vor.",
        "- Katja meldet sich mit gesunder Skepsis und kündigt an, die heutigen Tools genau unter die Lupe zu nehmen.",
        "- Kurzer schneller Teaser aller Durchbrüche, danach Übergang zum ersten Deep Dive.",
        "- Wechselnde Sprecherturns (Conrad / Katja).",
    ])
    return "\n".join(lines)


def format_deep_dive_prompt(
    breakthrough: Breakthrough,
    track_title: str,
    target_minutes: int = 6,
) -> str:
    """Prompt for generating a domain-tailored dialectic deep dive track."""
    target_words = target_minutes * 140
    return f"""\
Erstelle einen detaillierten, dialektisch-kontroversen Deep Dive Track für folgenden Durchbruch:

Titel: {breakthrough.title}
Track: {track_title}
Ziel-Länge: ca. {target_minutes} Minuten (ca. {target_words} Wörter insgesamt).

PROJEKT-DATEN:
- TL;DR: {breakthrough.tldr}
- Use Case: {breakthrough.use_case}
- Baseline-Vergleich: {breakthrough.comparison}
- Quickstart: {breakthrough.quickstart}
- Hardware: {breakthrough.hardware_requirements or 'Keine besonderen Angaben'}
- Lizenz: {breakthrough.license or 'Unbekannt'}
- Breakthrough Score: {breakthrough.breakthrough_score:.1f} / 10
- ROI Score: {breakthrough.roi_score:.1f} / 10
- Repository: {breakthrough.repo_url}

ANFORDERUNGEN:
- Folge der 6-stufigen dialektischen Struktur:
  1. Conrad: These zur technologischen Innovation und Architektur.
  2. Katja: Antithese I – hinterfragt den Hype und praktischen Nutzen für den Entwickleralltag.
  3. Conrad: Replik I – verteidigt den konkreten Use Case und den Vorteil gegenüber bisherigen Standards.
  4. Katja: Antithese II – bohrt fachspezifisch nach! Kritisiere genau die Aspekte, die für diese Art von Tool kritisch sind (z. B. Speicher/VRAM bei Inferenz, Concurrency/Locks bei Datenbanken, Tokenkosten bei Agenten, Lizenz-Einschränkungen, API-Stabilität).
  5. Conrad: Replik II – räumt berechtigte Risiken ein und nennt den Quickstart-Befehl als isolierten Testpfad.
  6. Katja & Conrad: Synthese – gemeinsames Urteil unter Berücksichtigung von Breakthrough Score ({breakthrough.breakthrough_score:.1f}) und ROI ({breakthrough.roi_score:.1f}).
- Halte die Turns lebendig und dialogisch (mindestens 6 Turns, abwechselnd Conrad und Katja).
"""


def format_serendipity_prompt(
    breakthroughs: list[Breakthrough],
    target_minutes: int = 5,
) -> str:
    """Prompt for generating the serendipity (out-of-the-bubble) track."""
    target_words = target_minutes * 140
    lines = [
        f"Erstelle den Serendipity-Track ('Blick über den Tellerrand') für den heutigen Audio Digest.",
        f"Ziel-Länge: ca. {target_minutes} Minuten (ca. {target_words} Wörter insgesamt).",
        "",
        "Serendipity-Kandidaten:",
    ]
    for b in breakthroughs:
        lines.append(
            f"- {b.title} (Score: {b.breakthrough_score:.1f}): {b.tldr}. "
            f"Use Case: {b.use_case}"
        )

    lines.extend([
        "",
        "Anforderungen:",
        "- Katja moderiert an: Warum ist es wichtig, den Blick über die eigene Entwickler-Blase hinaus zu wagen?",
        "- Conrad und Katja diskutieren die Funde mit Neugierde: Wie können Ideen aus Robotik, Wissenschaft oder exotischen Architekturen unsere eigenen Systeme inspirieren?",
        "- Konstruktiver, inspirierender Dialog mit abwechselnden Turns.",
    ])
    return "\n".join(lines)


def format_verdict_prompt(
    breakthroughs: list[Breakthrough],
    target_minutes: int = 2,
) -> str:
    """Prompt for generating the evening action verdict track."""
    target_words = target_minutes * 140
    top_bt = max(breakthroughs, key=lambda b: b.breakthrough_score) if breakthroughs else None
    top_title = top_bt.title if top_bt else "keines"
    top_quickstart = top_bt.quickstart if top_bt else ""

    return f"""\
Erstelle den abschließenden Action-Verdict-Track für den heutigen Audio Digest.
Ziel-Länge: ca. {target_minutes} Minuten (ca. {target_words} Wörter insgesamt).

Top-Empfehlung des Tages: {top_title}
Quickstart: {top_quickstart}

ANFORDERUNGEN:
- Kurzes, prägnantes Tagesfazit zwischen Conrad und Katja.
- Klare Handlungsempfehlung für den heutigen Feierabend: Lohnt sich ein 15-Minuten-Spike?
- Erinnerung an die Hörer: Interaktive Feedback-Buttons im Telegram-Bot bzw. Emoji-Reaktionen in Signal (🎯 Hit, 💤 Hype, ✅ Bekannt, 🚀 Inspiration) nutzen.
- Freundliche Verabschiedung bis zur nächsten Ausgabe.
"""
