"""newsscout.delivery.inbound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inbound interaction router for Signal, Telegram, and multi-messenger interactions.
Parses emoji reactions, text keywords, card quote-replies, and slash commands.
Persists ratings to PreferencesService and updates Topic Radar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Optional

from newsscout.config import Settings, get_settings
from newsscout.delivery.dispatcher import DeliveryDispatcher
from newsscout.storage.models import FeedbackRating
from newsscout.storage.preferences import PreferencesService

logger = logging.getLogger("newsscout.delivery.inbound")


@dataclass
class InboundMessage:
    """Normalized representation of an inbound message or reaction."""
    channel: str                          # "signal", "telegram"
    sender_id: str                        # phone number, Signal UUID, chat ID
    text: Optional[str] = None            # message body text
    reaction_emoji: Optional[str] = None  # emoji if reaction event
    message_id: Optional[str] = None      # incoming message ID
    quoted_message_id: Optional[str] = None  # ID of message being replied to
    quoted_text: Optional[str] = None     # text content of quoted message
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class InboundHandlingResult:
    """Result of processing an inbound interaction."""
    action: str                           # "feedback", "command", "ignored", "unknown", "error", "empty"
    success: bool = True
    channel: str = ""
    reply_text: Optional[str] = None      # Optional formatted reply to send back
    breakthrough_id: Optional[int] = None
    rating: Optional[FeedbackRating] = None
    command: Optional[str] = None
    error: Optional[str] = None


def parse_feedback_rating(input_str: str) -> Optional[FeedbackRating]:
    """Parses an emoji or keyword string into a FeedbackRating enum.

    Supports:
    - 🎯, 💤, 😴, ✅, ✔, 🚀, 🔥
    - hit, volltreffer, top, 1
    - hype, banal, zu banal, skip, 2, schrott
    - known, bekannt, kenne ich, kenne ich schon, 3, alt
    - inspire, inspiration, genial, geniale inspiration, idee, 4, geil
    """
    if not input_str:
        return None

    cleaned = input_str.strip().replace("\ufe0f", "")

    # 1. Check emoji directly
    from_emo = FeedbackRating.from_emoji(cleaned)
    if from_emo is not None:
        return from_emo

    # 2. Check direct keyword
    from_kw = FeedbackRating.from_keyword(cleaned)
    if from_kw is not None:
        return from_kw

    # 3. Extended keyword heuristics with normalized alphanumeric text
    lowered = re.sub(r"[^\w\s]", "", cleaned.lower()).strip()
    words = lowered.split()

    hit_terms = {"hit", "volltreffer", "top", "1", "treffer"}
    hype_terms = {"hype", "banal", "zu banal", "skip", "2", "schrott"}
    known_terms = {"known", "bekannt", "kenne ich", "kenne ich schon", "3", "alt"}
    inspire_terms = {"inspire", "inspiration", "genial", "geniale inspiration", "idee", "4", "geil"}

    if lowered in hit_terms or any(w in hit_terms for w in words):
        return FeedbackRating.HIT
    if lowered in hype_terms or any(w in hype_terms for w in words):
        return FeedbackRating.HYPE
    if lowered in known_terms or any(w in known_terms for w in words):
        return FeedbackRating.KNOWN
    if lowered in inspire_terms or any(w in inspire_terms for w in words):
        return FeedbackRating.INSPIRE

    return None


class InboundRouter:
    """Routes and handles inbound interactions from Signal and Telegram."""

    def __init__(
        self,
        preferences_service: Optional[PreferencesService] = None,
        db: Optional[Any] = None,
        dispatcher: Optional[DeliveryDispatcher] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.preferences = preferences_service
        self.db = db
        self.dispatcher = dispatcher
        self.settings = settings or get_settings()

    async def resolve_breakthrough_id(
        self,
        channel: str,
        quoted_message_id: Optional[str] = None,
        quoted_text: Optional[str] = None,
        message_text: Optional[str] = None,
    ) -> Optional[int]:
        """Resolves the breakthrough ID using a prioritized 6-tier strategy:
        1. Explicit ID in message text ('hit 42', '#42', 'ID: 42')
        2. Explicit [#ID] tag in quoted card text ('[#42]')
        3. Dispatcher delivery cache lookup (channel, quoted_message_id)
        4. Quoted title match against database breakthroughs table
        5. Latest broadcast ID from Dispatcher
        6. Latest non-discard breakthrough in SQLite database
        """
        # Tier 1: Explicit ID in message text
        if message_text:
            m = re.search(r"(?:^|[\s,;:\[\(])(?:id[:\s]*|#|nr\.?\s*)(\d+)\b", message_text, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m2 = re.search(
                r"\b(?:hit|hype|known|inspire|inspiration|volltreffer|banal|bekannt|idee|genial|top)\s+(\d+)\b",
                message_text,
                re.IGNORECASE,
            )
            if m2:
                return int(m2.group(1))
            m3 = re.search(r"\b(\d+)\b", message_text)
            if m3:
                return int(m3.group(1))

        # Tier 2: Explicit [#ID] tag in quoted card text
        if quoted_text:
            m_tag = re.search(r"\[#?(\d+)\]|#(\d+)|\(ID:\s*(\d+)\)", quoted_text)
            if m_tag:
                val = m_tag.group(1) or m_tag.group(2) or m_tag.group(3)
                if val:
                    return int(val)

        # Tier 3: Dispatcher delivery cache lookup
        if self.dispatcher and quoted_message_id:
            cached_id = self.dispatcher.get_breakthrough_id_for_message(channel, quoted_message_id)
            if cached_id is not None:
                return cached_id

        # Tier 4: Quoted title match against database
        if quoted_text and self.db:
            lines = [ln.strip() for ln in quoted_text.splitlines() if ln.strip()]
            for line in lines[:3]:
                clean_line = re.sub(r"^[⚡\*#\s]+", "", line).strip()
                if len(clean_line) >= 5:
                    row = await self.db.fetch_one(
                        "SELECT id FROM breakthroughs WHERE title LIKE ? LIMIT 1",
                        (f"%{clean_line[:30]}%",),
                    )
                    if row:
                        return row["id"]

        # Tier 5: Latest broadcast ID from Dispatcher
        if self.dispatcher:
            latest_id = self.dispatcher.get_latest_breakthrough_id()
            if latest_id is not None:
                return latest_id

        # Tier 6: Latest non-discard breakthrough in SQLite database
        if self.db:
            row = await self.db.fetch_one(
                "SELECT id FROM breakthroughs WHERE category != 'discard' ORDER BY evaluated_at DESC LIMIT 1"
            )
            if row:
                return row["id"]

        return None

    async def handle_signal_webhook(self, payload: dict[str, Any]) -> InboundHandlingResult:
        """Parses a signal-cli-rest-api webhook / event payload."""
        envelope = payload.get("envelope", {}) or {}
        if not isinstance(envelope, dict):
            envelope = {}
        sender_id = (
            envelope.get("sourceNumber")
            or envelope.get("source")
            or envelope.get("sourceUuid")
            or ""
        )
        data_message = envelope.get("dataMessage", {}) or {}
        if not isinstance(data_message, dict):
            data_message = {}

        if not data_message:
            return InboundHandlingResult(action="ignored", success=True, channel="signal")

        # Reaction event
        reaction = data_message.get("reaction")
        if reaction:
            emoji = reaction.get("emoji", "") if isinstance(reaction, dict) else ""
            target_ts = reaction.get("targetSentTimestamp") if isinstance(reaction, dict) else None
            msg = InboundMessage(
                channel="signal",
                sender_id=sender_id,
                reaction_emoji=emoji,
                quoted_message_id=str(target_ts) if target_ts else None,
                raw_payload=payload,
            )
            return await self.process_message(msg)

        # Standard message with optional quote
        body = data_message.get("message", "")
        quote = data_message.get("quote", {}) or {}
        if not isinstance(quote, dict):
            quote = {}
        quoted_id = quote.get("id")
        quoted_text = quote.get("text")

        msg = InboundMessage(
            channel="signal",
            sender_id=sender_id,
            text=body,
            message_id=str(data_message.get("timestamp", "")),
            quoted_message_id=str(quoted_id) if quoted_id else None,
            quoted_text=quoted_text,
            raw_payload=payload,
        )
        return await self.process_message(msg)

    async def process_message(self, msg: InboundMessage) -> InboundHandlingResult:
        """Processes a normalized InboundMessage (reaction, slash command, or rating keyword)."""
        # 1. Reaction Handling
        if msg.reaction_emoji:
            rating = parse_feedback_rating(msg.reaction_emoji)
            if rating is None:
                return InboundHandlingResult(
                    action="unknown",
                    success=False,
                    channel=msg.channel,
                    error=f"Unrecognized reaction emoji: {msg.reaction_emoji}",
                )

            bid = await self.resolve_breakthrough_id(
                channel=msg.channel,
                quoted_message_id=msg.quoted_message_id,
                quoted_text=msg.quoted_text,
            )
            if bid is None:
                return InboundHandlingResult(
                    action="error",
                    success=False,
                    channel=msg.channel,
                    rating=rating,
                    error="Could not identify breakthrough for reaction.",
                )

            if self.preferences:
                await self.preferences.record_feedback(
                    breakthrough_id=bid,
                    rating=rating,
                    user_identifier=msg.sender_id,
                    source=msg.channel,
                    notes=f"Reaction {msg.reaction_emoji}",
                )

            reply = self._format_feedback_reply(msg.channel, rating)
            return InboundHandlingResult(
                action="feedback",
                success=True,
                channel=msg.channel,
                breakthrough_id=bid,
                rating=rating,
                reply_text=reply,
            )

        # 2. Text Message Handling
        text = (msg.text or "").strip()
        if not text:
            return InboundHandlingResult(action="empty", success=True, channel=msg.channel)

        # 2a. Slash Commands
        if text.startswith("/"):
            return await self._handle_slash_command(msg.channel, text, msg.sender_id)

        # 2b. Rating Keywords
        rating = parse_feedback_rating(text)
        if rating is not None:
            bid = await self.resolve_breakthrough_id(
                channel=msg.channel,
                quoted_message_id=msg.quoted_message_id,
                quoted_text=msg.quoted_text,
                message_text=text,
            )
            if bid is None:
                return InboundHandlingResult(
                    action="error",
                    success=False,
                    channel=msg.channel,
                    rating=rating,
                    error="Could not identify breakthrough for feedback text.",
                )

            if self.preferences:
                await self.preferences.record_feedback(
                    breakthrough_id=bid,
                    rating=rating,
                    user_identifier=msg.sender_id,
                    source=msg.channel,
                    notes=f"Keyword '{text}'",
                )

            reply = self._format_feedback_reply(msg.channel, rating)
            return InboundHandlingResult(
                action="feedback",
                success=True,
                channel=msg.channel,
                breakthrough_id=bid,
                rating=rating,
                reply_text=reply,
            )

        # 2c. Unrecognized text
        return InboundHandlingResult(
            action="unknown",
            success=False,
            channel=msg.channel,
            reply_text=self._format_help_reply(msg.channel),
        )

    async def _handle_slash_command(self, channel: str, text: str, sender_id: str) -> InboundHandlingResult:
        """Processes slash commands: /track, /interest, /radar, /help."""
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/help", "/start"):
            return InboundHandlingResult(
                action="command",
                command="help",
                success=True,
                channel=channel,
                reply_text=self._format_help_reply(channel),
            )

        elif cmd == "/track" and len(parts) >= 2:
            repo_slug = parts[1].strip()
            if self.db:
                from newsscout.dashboard.queries import add_tracked_repo
                res = await add_tracked_repo(self.db, repo_slug)
                if res.get("success"):
                    reply = self._format_track_success(channel, res.get("repo", repo_slug))
                else:
                    reply = f"❌ Fehler: {res.get('error', 'Ungültiges Repo')}"
            else:
                reply = self._format_track_success(channel, repo_slug)

            return InboundHandlingResult(
                action="command",
                command="track",
                success=True,
                channel=channel,
                reply_text=reply,
            )

        elif cmd == "/interest" and len(parts) >= 2:
            kw = " ".join(parts[1:]).strip().lower()
            if self.db:
                from newsscout.dashboard.queries import add_interest_keyword
                res = await add_interest_keyword(self.db, kw)
                if res.get("success"):
                    reply = self._format_interest_success(channel, kw)
                else:
                    reply = f"❌ Fehler: {res.get('error', 'Ungültiges Keyword')}"
            else:
                reply = self._format_interest_success(channel, kw)

            return InboundHandlingResult(
                action="command",
                command="interest",
                success=True,
                channel=channel,
                reply_text=reply,
            )

        elif cmd in ("/radar", "/topics"):
            if self.db:
                from newsscout.dashboard.queries import get_topic_radar
                radar = await get_topic_radar(self.db)
                repos = radar.get("tracked_repos", [])
                kws = radar.get("custom_keywords", [])
                reply = self._format_radar_reply(channel, repos, kws)
            else:
                reply = "📡 Themenradar aktiv (DB nicht verbunden)."

            return InboundHandlingResult(
                action="command",
                command="radar",
                success=True,
                channel=channel,
                reply_text=reply,
            )

        return InboundHandlingResult(
            action="unknown",
            command=cmd,
            success=False,
            channel=channel,
            reply_text=self._format_help_reply(channel),
        )

    def _format_feedback_reply(self, channel: str, rating: FeedbackRating) -> str:
        """Formats acknowledgment reply for rating."""
        if channel == "signal":
            return f"Dankeschön! {rating.emoji} **{rating.label_de}** gespeichert."
        return f"Dankeschön! {rating.emoji} <b>{rating.label_de}</b> gespeichert."

    def _format_track_success(self, channel: str, repo: str) -> str:
        if channel == "signal":
            return f"✅ **Repository hinzugefügt**\n`{repo}` wird jetzt im Release-Radar überwacht."
        return f"✅ <b>Repository hinzugefügt</b>\n<code>{repo}</code> wird jetzt im Release-Radar überwacht."

    def _format_interest_success(self, channel: str, kw: str) -> str:
        if channel == "signal":
            return f"🎯 **Keyword hinzugefügt**\n'**{kw}**' ist jetzt im Hacker News Radar aktiv."
        return f"🎯 <b>Keyword hinzugefügt</b>\n'<code>{kw}</code>' ist jetzt im Hacker News Radar aktiv."

    def _format_radar_reply(self, channel: str, repos: list[str], kws: list[str]) -> str:
        repos_str = "\n".join(f"• {r}" for r in repos) or "Keine"
        kws_str = ", ".join(kws) or "Standard"
        if channel == "signal":
            return f"📡 **Aktueller NewsScout Themenradar**\n\n**GitHub Repositories:**\n{repos_str}\n\n**Custom Keywords:**\n{kws_str}"
        return f"📡 <b>Aktueller NewsScout Themenradar</b>\n\n<b>GitHub Repositories:</b>\n{repos_str}\n\n<b>Custom Keywords:</b>\n{kws_str}"

    def _format_help_reply(self, channel: str) -> str:
        if channel == "signal":
            return (
                "🤖 **NewsScout Befehle & Feedback**\n\n"
                "• `/track <owner/repo>` — GitHub-Repo überwachen\n"
                "• `/interest <keyword>` — HN-Keyword überwachen\n"
                "• `/radar` — Aktive Themen & Repos anzeigen\n"
                "• `/help` — Diese Hilfe anzeigen\n\n"
                "**Feedback auf Karten:**\n"
                "Antworte auf eine Karte mit Reaction oder Text:\n"
                "🎯 Volltreffer | 💤 Zu banal / Hype | ✅ Kenne ich schon | 🚀 Geniale Inspiration"
            )
        return (
            "🤖 <b>NewsScout Befehle & Feedback</b>\n\n"
            "• <code>/track &lt;owner/repo&gt;</code> — GitHub-Repo überwachen\n"
            "• <code>/interest &lt;keyword&gt;</code> — HN-Keyword überwachen\n"
            "• <code>/radar</code> — Aktive Themen & Repos anzeigen\n"
            "• <code>/help</code> — Diese Hilfe anzeigen"
        )
