"""newsscout.storage.models
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic v2 data models for NewsScout domain entities, database records,
and payload validation contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import html
import json
import re
import sqlite3
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator



# ============================================================================
# Telegram HTML Safe Truncation Utility
# ============================================================================

def _safe_truncate_telegram_html(html_text: str, max_chars: int = 4096) -> str:
    """Truncates Telegram HTML to max_chars, closing any open tags.

    Ensures that truncation never leaves unclosed HTML tags (e.g. <code> without
    </code>), which would cause Telegram Bot API to reject the message with
    HTTP 400 Bad Request.
    """
    if len(html_text) <= max_chars:
        return html_text

    marker = "\n\n<i>[Truncated]</i>"
    tag_re = re.compile(r'<(/?)(\w+)[^>]*>')

    # Reserve space for marker and potential closing tags
    budget = max_chars - len(marker) - 100
    if budget < 100:
        budget = max_chars - len(marker) - 20

    truncated = html_text[:budget]

    # Don't cut inside a tag
    last_gt = truncated.rfind('>')
    last_lt = truncated.rfind('<')
    if last_lt > last_gt:
        truncated = truncated[:last_gt + 1] if last_gt >= 0 else truncated[:last_lt]

    # Find open tags by scanning the truncated content
    stack: list[str] = []
    for m in tag_re.finditer(truncated):
        closing = m.group(1)
        tag = m.group(2).lower()
        full_match = m.group(0)
        is_self_closing = full_match.endswith('/>')
        if closing:
            if stack and stack[-1] == tag:
                stack.pop()
        elif not is_self_closing:
            stack.append(tag)

    close_str = ''.join(f'</{t}>' for t in reversed(stack))
    result = truncated + close_str + marker

    # Final safety: if still too long, hard trim
    if len(result) > max_chars:
        result = result[:max_chars]

    return result


# ============================================================================
# Enums
# ============================================================================

class SourceType(str, Enum):
    API = "api"
    FIREBASE = "firebase"
    FEED = "feed"
    SEARCH = "search"


class Stage2Category(str, Enum):
    CORE = "core"
    SERENDIPITY = "serendipity"
    DISCARD = "discard"


class DigestSlot(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"


class DigestStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    SYNTHESIZED = "synthesized"
    DELIVERED = "delivered"
    FAILED = "failed"


class TrackType(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    DEEP_DIVE_1 = "deep_dive_1"
    DEEP_DIVE_2 = "deep_dive_2"
    SERENDIPITY = "serendipity"
    VERDICT = "verdict"


class FeedbackRating(str, Enum):
    HIT = "hit"          # 🎯 Volltreffer
    HYPE = "hype"        # 💤 Zu banal / Hype
    KNOWN = "known"      # ✅ Kenne ich schon
    INSPIRE = "inspire"  # 🚀 Geniale Inspiration

    @property
    def is_positive(self) -> bool:
        """Returns True if rating represents a positive signal for few-shot calibration."""
        return self in (FeedbackRating.HIT, FeedbackRating.INSPIRE)

    @property
    def emoji(self) -> str:
        """Returns the canonical emoji representing this rating."""
        mapping = {
            FeedbackRating.HIT: "🎯",
            FeedbackRating.HYPE: "💤",
            FeedbackRating.KNOWN: "✅",
            FeedbackRating.INSPIRE: "🚀",
        }
        return mapping[self]

    @property
    def label_de(self) -> str:
        """Returns the German user-facing label."""
        mapping = {
            FeedbackRating.HIT: "Volltreffer",
            FeedbackRating.HYPE: "Zu banal / Hype",
            FeedbackRating.KNOWN: "Kenne ich schon",
            FeedbackRating.INSPIRE: "Geniale Inspiration",
        }
        return mapping[self]

    @classmethod
    def from_emoji(cls, emoji_char: str) -> Optional[FeedbackRating]:
        """Parses an emoji reaction character into a FeedbackRating."""
        mapping = {
            "🎯": cls.HIT,
            "💤": cls.HYPE,
            "😴": cls.HYPE,
            "✅": cls.KNOWN,
            "✔": cls.KNOWN,
            "🚀": cls.INSPIRE,
            "🔥": cls.INSPIRE,
        }
        return mapping.get(emoji_char.strip().replace("\ufe0f", ""))

    @classmethod
    def from_keyword(cls, text: str) -> Optional[FeedbackRating]:
        """Parses a text keyword or number into a FeedbackRating."""
        cleaned = text.strip().lower()
        if cleaned in ("hit", "volltreffer", "top", "1", "🎯"):
            return cls.HIT
        if cleaned in ("hype", "banal", "zu banal", "skip", "2", "💤"):
            return cls.HYPE
        if cleaned in ("known", "bekannt", "kenne ich", "kenne ich schon", "3", "✅"):
            return cls.KNOWN
        if cleaned in ("inspire", "inspiration", "genial", "geniale inspiration", "4", "🚀"):
            return cls.INSPIRE
        return None


class FeedbackSource(str, Enum):
    TELEGRAM = "telegram"
    WEB = "web"
    SIGNAL = "signal"



# ============================================================================
# Value Objects & Nested Models
# ============================================================================

class DecisionCardData(BaseModel):
    """Structured fields of the 1-Minute Decision Card."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    tldr: str = Field(
        description="Exactly 1 sentence: what is now possible that was impossible before."
    )
    use_case: str = Field(
        description="Concrete workflow application in the user's stack."
    )
    comparison: str = Field(
        description="Performance or architectural comparison vs existing baselines."
    )
    quickstart: str = Field(
        description="Copy-pasteable 1-liner command (e.g. docker run... or uvx...)."
    )
    hardware_requirements: str = Field(
        description="Hardware requirements, VRAM, and platform constraints."
    )
    license: str = Field(
        description="Open source license status and commercial usability."
    )

    def render_markdown(self, title: Optional[str] = None) -> str:
        """Renders GitHub-flavored Markdown card snippet."""
        title_str = f"### {title}\n\n" if title else ""
        return (
            f"{title_str}"
            f"> **TL;DR**: {self.tldr}\n\n"
            f"* **Workflow Use Case**: {self.use_case}\n"
            f"* **Baseline Comparison**: {self.comparison}\n"
            f"* **Quickstart**:\n"
            f"  `{self.quickstart}`\n\n"
            f"* **Hardware Requirements & License**: `{self.hardware_requirements}` | `{self.license}`"
        )


class AudioTurn(BaseModel):
    """Single speaker turn in a German technical dialogue."""

    model_config = ConfigDict(from_attributes=True)

    speaker: str = Field(description="Speaker name: 'Conrad' or 'Katja'")
    voice: str = Field(default="de-DE-ConradNeural", description="TTS voice identifier")
    text: str = Field(description="Spoken text in German")
    pause_after_ms: int = Field(default=250, description="Pause duration after turn in ms")


class DialogueScript(BaseModel):
    """Structured dialogue script for an audio track."""

    model_config = ConfigDict(from_attributes=True)

    track_number: int = Field(default=1, ge=1, le=5)
    track_type: TrackType = TrackType.EXECUTIVE_SUMMARY
    title: str = "Audio Track"
    target_duration_minutes: float = Field(default=0.0)
    turns: list[AudioTurn] = Field(default_factory=list)


class Exemplar(BaseModel):
    """Positive or negative preference exemplar for Stage 2 few-shot calibration."""

    model_config = ConfigDict(from_attributes=True)

    breakthrough_id: int
    title: str
    rating: FeedbackRating
    category: Stage2Category
    tldr: str
    use_case: str
    breakthrough_score: float
    roi_score: float
    repo_url: Optional[str] = None


# ============================================================================
# Core Database Entities
# ============================================================================

class Source(BaseModel):
    """Source configuration and polling state."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    name: str
    source_type: SourceType
    config: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    last_poll_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "config_json" in data:
                cfg = data.pop("config_json")
                data["config"] = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
            if "is_active" in data:
                data["is_active"] = bool(data["is_active"])
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Source:
        data = dict(row)
        return cls.model_validate(data)

    def to_db_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type.value,
            "config_json": json.dumps(self.config),
            "is_active": 1 if self.is_active else 0,
            "last_poll_at": self.last_poll_at.isoformat() if self.last_poll_at else None,
            "created_at": self.created_at.isoformat(),
        }


class RawItem(BaseModel):
    """Unprocessed ingested candidate from a data source."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    source: str
    source_id: str
    title: str
    url: str
    raw_content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "metadata_json" in data:
                meta = data.pop("metadata_json")
                data["metadata"] = json.loads(meta) if isinstance(meta, str) else (meta or {})
        return data

    @property
    def metadata_json(self) -> dict[str, Any]:
        """Convenience property for accessing metadata dict."""
        return self.metadata

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RawItem:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "raw_content": self.raw_content,
            "metadata_json": json.dumps(self.metadata),
            "ingested_at": self.ingested_at.isoformat(),
        }


class Stage1Evaluation(BaseModel):
    """Deterministic local filter output."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    raw_item_id: int
    passed: bool
    drop_reason: Optional[str] = None
    detected_license: Optional[str] = None
    has_docker: bool = False
    has_runnable_code: bool = False
    heuristics: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "heuristics_json" in data:
                heur = data.pop("heuristics_json")
                data["heuristics"] = json.loads(heur) if isinstance(heur, str) else (heur or {})
            if "passed" in data:
                data["passed"] = bool(data["passed"])
            if "has_docker" in data:
                data["has_docker"] = bool(data["has_docker"])
            if "has_runnable_code" in data:
                data["has_runnable_code"] = bool(data["has_runnable_code"])
        return data

    @property
    def heuristics_json(self) -> dict[str, Any]:
        return self.heuristics

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Stage1Evaluation:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        return {
            "raw_item_id": self.raw_item_id,
            "passed": 1 if self.passed else 0,
            "drop_reason": self.drop_reason,
            "detected_license": self.detected_license,
            "has_docker": 1 if self.has_docker else 0,
            "has_runnable_code": 1 if self.has_runnable_code else 0,
            "heuristics_json": json.dumps(self.heuristics),
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class DecisionCard(BaseModel):
    """Rich representation of the 1-Minute Decision Card with markdown and HTML rendering."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    breakthrough_id: Optional[int] = None
    title: str
    repo_url: Optional[str] = None
    breakthrough_score: float = Field(ge=1.0, le=10.0)
    roi_score: float = Field(ge=1.0, le=10.0)
    category: Stage2Category
    data: DecisionCardData

    def render_markdown(self) -> str:
        """Renders standard GitHub-flavored Markdown card."""
        badge = "🎯 CORE BREAKTHROUGH" if self.category == Stage2Category.CORE else "🚀 SERENDIPITY HIT"
        repo_link = f" ([Repository]({self.repo_url}))" if self.repo_url else ""
        return (
            f"### {self.title} {repo_link}\n"
            f"**{badge}** — Score: `{self.breakthrough_score:.1f}/10` | ROI: `{self.roi_score:.1f}/10`\n\n"
            f"> **TL;DR**: {self.data.tldr}\n\n"
            f"* **Workflow Use Case**: {self.data.use_case}\n"
            f"* **Baseline Comparison**: {self.data.comparison}\n"
            f"* **Quickstart**:\n"
            f"  ```bash\n"
            f"  {self.data.quickstart}\n"
            f"  ```\n"
            f"* **Hardware Requirements & License**: `{self.data.hardware_requirements}` | `{self.data.license}`"
        )

    def render_telegram_html(self, max_chars: int = 4096) -> str:
        """Renders clean Telegram-compatible HTML representation with entity escaping.

        Guarantees output length <= max_chars (default 4096, Telegram's limit)
        with tag-safe truncation that closes any open HTML tags.
        """
        badge = "🎯 <b>CORE BREAKTHROUGH</b>" if self.category == Stage2Category.CORE else "🚀 <b>SERENDIPITY HIT</b>"
        repo_link = f' (<a href="{html.escape(self.repo_url)}">Repo</a>)' if self.repo_url else ""
        escaped_title = html.escape(self.title)
        escaped_tldr = html.escape(self.data.tldr)
        escaped_use_case = html.escape(self.data.use_case)
        escaped_comp = html.escape(self.data.comparison)
        escaped_quickstart = html.escape(self.data.quickstart)
        escaped_hardware = html.escape(self.data.hardware_requirements)
        escaped_license = html.escape(self.data.license)
        rendered = (
            f"⚡ <b>{escaped_title}</b>{repo_link}\n"
            f"{badge} | Score: <b>{self.breakthrough_score:.1f}</b> | ROI: <b>{self.roi_score:.1f}</b>\n\n"
            f"💡 <b>TL;DR</b>: {escaped_tldr}\n\n"
            f"🛠 <b>Workflow Use Case</b>: {escaped_use_case}\n"
            f"📊 <b>Baseline Comparison</b>: {escaped_comp}\n"
            f"🚀 <b>Quickstart</b>: <code>{escaped_quickstart}</code>\n"
            f"💻 <b>Hardware & License</b>: {escaped_hardware} | <i>{escaped_license}</i>"
        )
        return _safe_truncate_telegram_html(rendered, max_chars)


class Breakthrough(BaseModel):
    """Validated breakthrough item with Stage 2 evaluation and decision card."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    id: Optional[int] = None
    raw_item_id: int
    title: str
    repo_url: Optional[str] = None
    breakthrough_score: float = Field(ge=1.0, le=10.0)
    roi_score: float = Field(ge=1.0, le=10.0)
    category: Stage2Category
    tldr: str
    use_case: str
    comparison: str
    quickstart: str
    hardware_requirements: str
    license: str
    card_markdown: str
    evaluation_raw: dict[str, Any] = Field(default_factory=dict)
    is_watchlisted: bool = False
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evaluation_raw_json" in data:
                raw_eval = data.pop("evaluation_raw_json")
                data["evaluation_raw"] = json.loads(raw_eval) if isinstance(raw_eval, str) else (raw_eval or {})
            if "is_watchlisted" in data:
                data["is_watchlisted"] = bool(data["is_watchlisted"])
            if "category" in data and isinstance(data["category"], str):
                data["category"] = Stage2Category(data["category"])
        return data

    @property
    def evaluation_raw_json(self) -> dict[str, Any]:
        return self.evaluation_raw

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Breakthrough:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        return {
            "raw_item_id": self.raw_item_id,
            "title": self.title,
            "repo_url": self.repo_url,
            "breakthrough_score": self.breakthrough_score,
            "roi_score": self.roi_score,
            "category": self.category.value,
            "tldr": self.tldr,
            "use_case": self.use_case,
            "comparison": self.comparison,
            "quickstart": self.quickstart,
            "hardware_requirements": self.hardware_requirements,
            "license": self.license,
            "card_markdown": self.card_markdown,
            "evaluation_raw_json": json.dumps(self.evaluation_raw),
            "is_watchlisted": 1 if self.is_watchlisted else 0,
            "evaluated_at": self.evaluated_at.isoformat(),
        }

    def to_decision_card(self) -> DecisionCard:
        """Constructs a DecisionCard instance from this Breakthrough."""
        return DecisionCard(
            breakthrough_id=self.id,
            title=self.title,
            repo_url=self.repo_url,
            breakthrough_score=self.breakthrough_score,
            roi_score=self.roi_score,
            category=self.category,
            data=DecisionCardData(
                tldr=self.tldr,
                use_case=self.use_case,
                comparison=self.comparison,
                quickstart=self.quickstart,
                hardware_requirements=self.hardware_requirements,
                license=self.license,
            ),
        )


class Track(BaseModel):
    """Individual audio chapter track within a Digest."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    digest_id: int
    track_number: int = Field(ge=1, le=5)
    track_type: TrackType
    title: str
    duration_seconds: int = Field(default=0, ge=0)
    audio_path: str
    script: DialogueScript
    breakthrough_id: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_script = data.pop("script_json", None)
            if raw_script is not None:
                if isinstance(raw_script, str):
                    parsed = json.loads(raw_script)
                else:
                    parsed = raw_script
                if isinstance(parsed, list):
                    turns = [
                        AudioTurn(
                            speaker=item.get("speaker", "Conrad"),
                            voice=item.get("voice", "de-DE-ConradNeural"),
                            text=item.get("text", ""),
                            pause_after_ms=item.get("pause_after_ms", 250),
                        )
                        for item in parsed
                    ]
                    data["script"] = DialogueScript(
                        track_number=data.get("track_number", 1),
                        track_type=TrackType(data.get("track_type", "executive_summary")),
                        title=data.get("title", "Track"),
                        turns=turns,
                    )
                elif isinstance(parsed, dict):
                    data["script"] = DialogueScript.model_validate(parsed)
            elif "script" in data and isinstance(data["script"], dict):
                data["script"] = DialogueScript.model_validate(data["script"])
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Track:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        return {
            "digest_id": self.digest_id,
            "track_number": self.track_number,
            "track_type": self.track_type.value,
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "audio_path": self.audio_path,
            "script_json": json.dumps(self.script.model_dump()),
            "breakthrough_id": self.breakthrough_id,
            "created_at": self.created_at.isoformat(),
        }


# Alias for backward compatibility
DigestTrack = Track


class Digest(BaseModel):
    """Twice-daily podcast episode and track container."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    slot: DigestSlot
    digest_date: str = Field(description="Date in YYYY-MM-DD format")
    title: str
    total_duration_seconds: int = Field(default=0, ge=0)
    merged_audio_path: Optional[str] = None
    telegram_message_id: Optional[int] = None
    status: DigestStatus = DigestStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    tracks: list[Track] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row, tracks: Optional[list[Track]] = None) -> Digest:
        data = dict(row)
        data["tracks"] = tracks or []
        return cls.model_validate(data)

    def to_db_params(self) -> dict[str, Any]:
        return {
            "slot": self.slot.value,
            "digest_date": self.digest_date,
            "title": self.title,
            "total_duration_seconds": self.total_duration_seconds,
            "merged_audio_path": self.merged_audio_path,
            "telegram_message_id": self.telegram_message_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
        }


class FeedbackCreate(BaseModel):
    """Input payload contract for recording user feedback across any channel."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    breakthrough_id: int = Field(gt=0, description="Target breakthrough ID")
    rating: FeedbackRating = Field(description="Rating enum or valid lowercase rating string")
    user_identifier: str = Field(
        default="default",
        description="Unique user identifier: phone number, Signal UUID, Telegram user ID, or 'default'",
    )
    source: str = Field(
        default="telegram",
        description="Origin channel: 'telegram', 'web', or 'signal'",
    )
    notes: Optional[str] = Field(default=None, description="Optional user commentary or context")

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "rating" in data and isinstance(data["rating"], str):
                data["rating"] = FeedbackRating(data["rating"].lower())

            # Allow backward-compatible telegram_user_id in input dicts
            if ("user_identifier" not in data or data["user_identifier"] is None) and "telegram_user_id" in data:
                tg_id = data.get("telegram_user_id")
                data["user_identifier"] = str(tg_id) if tg_id is not None else "default"
            elif "user_identifier" in data and data["user_identifier"] is not None:
                data["user_identifier"] = str(data["user_identifier"]).strip() or "default"
            else:
                data["user_identifier"] = "default"

            if "source" in data and isinstance(data["source"], str):
                data["source"] = data["source"].strip().lower()
        return data

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        valid_sources = {"telegram", "web", "signal"}
        if v not in valid_sources:
            raise ValueError(f"Invalid source '{v}'. Must be one of {valid_sources}")
        return v


class Feedback(BaseModel):
    """User feedback event from Telegram, Web UI, or Signal."""

    model_config = ConfigDict(from_attributes=True, validate_assignment=True)

    id: Optional[int] = None
    breakthrough_id: int
    rating: FeedbackRating
    user_identifier: str = "default"
    telegram_user_id: Optional[int] = None
    source: str = "telegram"
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "rating" in data and isinstance(data["rating"], str):
                data["rating"] = FeedbackRating(data["rating"].lower())

            user_id = data.get("user_identifier")
            tg_id = data.get("telegram_user_id")

            if user_id is not None:
                uid_str = str(user_id).strip() or "default"
                data["user_identifier"] = uid_str
                if tg_id is None and uid_str.isdigit():
                    data["telegram_user_id"] = int(uid_str)
                elif tg_id is not None:
                    data["telegram_user_id"] = int(tg_id)
            elif tg_id is not None:
                data["user_identifier"] = str(tg_id)
                data["telegram_user_id"] = int(tg_id)
            else:
                data["user_identifier"] = "default"
                data["telegram_user_id"] = None

            if "source" in data and isinstance(data["source"], str):
                data["source"] = data["source"].strip().lower()
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Feedback:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        """Maps model to SQLite parameter dictionary matching evolved feedback schema."""
        return {
            "breakthrough_id": self.breakthrough_id,
            "rating": self.rating.value,
            "user_identifier": self.user_identifier,
            "source": self.source,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }



class UserPreference(BaseModel):
    """Persistent preference record stored in SQLite user_preferences table."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    pref_key: str
    value: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "value_json" in data:
                raw_val = data.pop("value_json")
                data["value"] = json.loads(raw_val) if isinstance(raw_val, str) else (raw_val or {})
        return data

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> UserPreference:
        return cls.model_validate(dict(row))

    def to_db_params(self) -> dict[str, Any]:
        return {
            "pref_key": self.pref_key,
            "value_json": json.dumps(self.value),
            "updated_at": self.updated_at.isoformat(),
        }
