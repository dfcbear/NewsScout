"""newsscout.delivery.telegram_bot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Telegram Bot API client for NewsScout delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Optional

import httpx

from newsscout.config import Settings, get_settings
from newsscout.storage.models import (
    Breakthrough,
    Digest,
    FeedbackRating,
    Track,
)
from newsscout.storage.preferences import PreferencesService

logger = logging.getLogger("newsscout.delivery.telegram_bot")

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

def mask_telegram_url(url: str) -> str:
    """Masks secret bot token in Telegram URLs for secure logging."""
    return re.sub(r"/bot[^/]+/", "/bot<REDACTED>/", str(url))


FEEDBACK_BUTTONS: dict[FeedbackRating, str] = {
    FeedbackRating.HIT: "\U0001F3AF Volltreffer",
    FeedbackRating.HYPE: "\U0001F4A4 Zu banal/Hype",
    FeedbackRating.KNOWN: "\u2705 Kenne ich schon",
    FeedbackRating.INSPIRE: "\U0001F680 Geniale Inspiration",
}


class TelegramBot:
    """Telegram Bot API client for audio digest delivery and feedback."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        preferences_service: Optional[PreferencesService] = None,
        db: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._external_client = client is not None
        self._client: Optional[httpx.AsyncClient] = client
        self._preferences = preferences_service
        self._db = db
        self._base_url = TELEGRAM_API_BASE.format(
            token=self.settings.telegram_bot_token.get_secret_value()
        )

    async def __aenter__(self) -> "TelegramBot":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._external_client and self._client and not self._client.is_closed:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            raise RuntimeError("TelegramBot must be used as async context manager")
        return self._client

    # ------------------------------------------------------------------
    # Webhook management
    # ------------------------------------------------------------------

    async def set_webhook(self, webhook_url: str, secret: str = "") -> dict[str, Any]:
        """Registers a webhook URL with Telegram so updates are pushed to us."""
        payload: dict[str, Any] = {"url": webhook_url, "allowed_updates": ["message", "callback_query"]}
        if secret:
            payload["secret_token"] = secret
        response = await self._post_with_retry("/setWebhook", payload, max_retries=1)
        logger.info("Telegram webhook set to %s", webhook_url)
        return response

    async def delete_webhook(self) -> dict[str, Any]:
        """Removes the Telegram webhook (switches back to getUpdates polling)."""
        response = await self._post_with_retry("/deleteWebhook", {}, max_retries=1)
        logger.info("Telegram webhook deleted")
        return response

    async def get_webhook_info(self) -> dict[str, Any]:
        """Returns current webhook status from Telegram."""
        response = await self._post_with_retry("/getWebhookInfo", {}, max_retries=1)
        return response.get("result", {})

    async def send_decision_card(
        self,
        breakthrough: Breakthrough,
        chat_id: Optional[str] = None,
    ) -> int:
        chat_id = chat_id or self.settings.telegram_chat_id
        card = breakthrough.to_decision_card()
        html_text = card.render_telegram_html()
        keyboard = self._build_feedback_keyboard(breakthrough.id)

        payload = {
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "reply_markup": json.dumps(keyboard),
        }

        response = await self._post_with_retry("/sendMessage", payload)
        return response.get("result", {}).get("message_id", 0)

    def _build_feedback_keyboard(
        self, breakthrough_id: Optional[int]
    ) -> dict[str, Any]:
        buttons = []
        for rating in FeedbackRating:
            callback_data = f"feedback:{breakthrough_id}:{rating.value}"
            buttons.append({
                "text": FEEDBACK_BUTTONS[rating],
                "callback_data": callback_data,
            })

        return {
            "inline_keyboard": [
                [buttons[0], buttons[1]],
                [buttons[2], buttons[3]],
            ]
        }

    async def send_audio_track(
        self,
        track: Track,
        chat_id: Optional[str] = None,
    ) -> int:
        chat_id = chat_id or self.settings.telegram_chat_id
        audio_path = Path(track.audio_path)

        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            return 0

        files = {
            "audio": (audio_path.name, audio_path.read_bytes(), "audio/mpeg"),
        }
        data = {
            "chat_id": chat_id,
            "caption": f"\U0001F3B5 {track.title} ({track.duration_seconds // 60}:{track.duration_seconds % 60:02d})",
            "parse_mode": "HTML",
        }

        response = await self._post_with_retry_multipart("/sendAudio", data, files)
        return response.get("result", {}).get("message_id", 0)

    async def send_digest_menu(
        self,
        digest: Digest,
        chat_id: Optional[str] = None,
    ) -> int:
        chat_id = chat_id or self.settings.telegram_chat_id

        total_min = digest.total_duration_seconds // 60
        total_sec = digest.total_duration_seconds % 60

        text_lines = [
            "\U0001F4E3 <b>NewsScout Digest</b>",
            f"\U0001F4C5 {digest.digest_date} ({digest.slot.value})",
            f"\u23F1 Gesamtdauer: {total_min}:{total_sec:02d}",
            f"\u269B {len(digest.tracks)} Tracks",
            "",
            "<b>Tracks:</b>",
        ]

        for track in digest.tracks:
            mins = track.duration_seconds // 60
            secs = track.duration_seconds % 60
            text_lines.append(f"  \u25B8 {track.title} ({mins}:{secs:02d})")

        text_lines.append("")
        text_lines.append("Waehle einen Track oder hoere alles:")

        text = "\n".join(text_lines)
        keyboard = self._build_digest_menu_keyboard(digest)

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "reply_markup": json.dumps(keyboard),
        }

        response = await self._post_with_retry("/sendMessage", payload)
        return response.get("result", {}).get("message_id", 0)

    def _build_digest_menu_keyboard(self, digest: Digest) -> dict[str, Any]:
        rows: list[list[dict[str, Any]]] = []

        rows.append([{
            "text": "\U0001F3A7 Play All (Seamless Playlist)",
            "callback_data": f"playall:{digest.id}",
        }])

        for track in digest.tracks:
            mins = track.duration_seconds // 60
            secs = track.duration_seconds % 60
            rows.append([{
                "text": f"\u25B6 Track {track.track_number}: {track.title} ({mins}:{secs:02d})",
                "callback_data": f"play:{digest.id}:{track.track_number}",
            }])

        return {"inline_keyboard": rows}

    def is_authorized(self, chat_id: str) -> bool:
        """Verifies if a chat_id is authorized to interact with this bot."""
        allowed = str(self.settings.telegram_chat_id or "").strip()
        if not allowed or self.settings.environment == "test":
            return True
        return str(chat_id).strip() == allowed

    async def handle_callback_query(
        self,
        callback_data: str,
        callback_query_id: str,
        chat_id: str,
        telegram_user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.is_authorized(chat_id):
            logger.warning("Unauthorized callback from chat_id=%s", chat_id)
            await self._answer_callback(callback_query_id, "Nicht autorisiert")
            return {"action": "unauthorized"}

        parts = callback_data.split(":")

        if parts[0] == "feedback" and len(parts) == 3:
            return await self._handle_feedback_callback(
                parts[1], parts[2], callback_query_id, telegram_user_id
            )
        elif parts[0] == "play" and len(parts) == 3:
            return await self._handle_play_callback(
                parts[1], parts[2], callback_query_id, chat_id
            )
        elif parts[0] == "playall" and len(parts) == 2:
            return await self._handle_playall_callback(
                parts[1], callback_query_id, chat_id
            )
        else:
            logger.warning("Unknown callback format: %s", callback_data)
            await self._answer_callback(callback_query_id, "Unbekannte Aktion")
            return {"action": "unknown", "callback_data": callback_data}

    async def _handle_feedback_callback(
        self,
        breakthrough_id_str: str,
        rating_str: str,
        callback_query_id: str,
        telegram_user_id: Optional[int],
    ) -> dict[str, Any]:
        try:
            breakthrough_id = int(breakthrough_id_str)
            rating = FeedbackRating(rating_str)
        except (ValueError, KeyError):
            await self._answer_callback(callback_query_id, "Ungueltige Bewertung")
            return {"action": "error", "error": "invalid_rating"}

        if self._preferences:
            await self._preferences.record_feedback(
                breakthrough_id=breakthrough_id,
                rating=rating,
                telegram_user_id=telegram_user_id,
                source="telegram",
            )

        emoji_map = {
            FeedbackRating.HIT: "\U0001F3AF",
            FeedbackRating.HYPE: "\U0001F4A4",
            FeedbackRating.KNOWN: "\u2705",
            FeedbackRating.INSPIRE: "\U0001F680",
        }
        await self._answer_callback(
            callback_query_id, f"Dankeschoen! {emoji_map.get(rating, '')} Feedback gespeichert."
        )

        logger.info(
            "Feedback recorded: breakthrough_id=%d rating=%s user=%s",
            breakthrough_id,
            rating.value,
            telegram_user_id,
        )
        return {
            "action": "feedback",
            "breakthrough_id": breakthrough_id,
            "rating": rating,
        }

    async def _handle_play_callback(
        self,
        digest_id_str: str,
        track_number_str: str,
        callback_query_id: str,
        chat_id: str,
    ) -> dict[str, Any]:
        try:
            digest_id = int(digest_id_str)
            track_number = int(track_number_str)
        except ValueError:
            await self._answer_callback(callback_query_id, "Ungueltiger Track")
            return {"action": "error", "error": "invalid_track"}

        await self._answer_callback(
            callback_query_id, f"Spiele Track {track_number}..."
        )

        return {
            "action": "play_track",
            "digest_id": digest_id,
            "track_number": track_number,
            "chat_id": chat_id,
        }

    async def _handle_playall_callback(
        self,
        digest_id_str: str,
        callback_query_id: str,
        chat_id: str,
    ) -> dict[str, Any]:
        try:
            digest_id = int(digest_id_str)
        except ValueError:
            await self._answer_callback(callback_query_id, "Ungueltiger Digest")
            return {"action": "error", "error": "invalid_digest"}

        await self._answer_callback(callback_query_id, "Spiele kompletten Digest...")

        return {
            "action": "play_all",
            "digest_id": digest_id,
            "chat_id": chat_id,
        }

    async def _post_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        url = self._base_url + endpoint
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.post(url, json=payload)

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt > max_retries:
                        response.raise_for_status()
                    sleep_seconds = (1.5 ** attempt) + random.uniform(0.1, 0.5)
                    sleep_seconds = min(sleep_seconds, 10.0)
                    logger.warning(
                        "Telegram API %d, retrying in %.2fs (attempt %d/%d)",
                        response.status_code,
                        sleep_seconds,
                        attempt,
                        max_retries,
                    )
                    await asyncio.sleep(sleep_seconds)
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.RequestError, httpx.TimeoutException) as err:
                if attempt > max_retries:
                    raise
                sleep_seconds = (1.5 ** attempt) + random.uniform(0.1, 0.5)
                logger.warning(
                    "Telegram request error: %s, retrying in %.2fs",
                    mask_telegram_url(str(err)),
                    sleep_seconds,
                )
                await asyncio.sleep(sleep_seconds)

    async def _post_with_retry_multipart(
        self,
        endpoint: str,
        data: dict[str, Any],
        files: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        url = self._base_url + endpoint
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.post(url, data=data, files=files)

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt > max_retries:
                        response.raise_for_status()
                    sleep_seconds = (1.5 ** attempt) + random.uniform(0.1, 0.5)
                    sleep_seconds = min(sleep_seconds, 10.0)
                    await asyncio.sleep(sleep_seconds)
                    continue

                response.raise_for_status()
                return response.json()

            except (httpx.RequestError, httpx.TimeoutException) as err:
                if attempt > max_retries:
                    raise
                sleep_seconds = (1.5 ** attempt) + random.uniform(0.1, 0.5)
                await asyncio.sleep(sleep_seconds)

    async def _answer_callback(
        self,
        callback_query_id: str,
        text: str,
    ) -> None:
        try:
            await self._post_with_retry(
                "/answerCallbackQuery",
                {"callback_query_id": callback_query_id, "text": text},
                max_retries=1,
            )
        except Exception as e:
            logger.warning("Failed to answer callback query: %s", e)

    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> int:
        chat_id = chat_id or self.settings.telegram_chat_id
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        response = await self._post_with_retry("/sendMessage", payload)
        return response.get("result", {}).get("message_id", 0)

    async def handle_message_text(self, text: str, chat_id: str) -> dict[str, Any]:
        """Processes incoming Telegram text messages and slash commands."""
        if not self.is_authorized(chat_id):
            logger.warning("Unauthorized message attempt from chat_id=%s", chat_id)
            await self.send_message("⛔ Zugriff verweigert: Du bist nicht autorisiert, diesen Bot zu steuern.", chat_id=chat_id)
            return {"action": "unauthorized", "chat_id": chat_id}

        clean_text = text.strip()
        parts = clean_text.split()
        if not parts:
            return {"action": "empty"}

        cmd = parts[0].lower()
        if cmd in ("/help", "/start"):
            help_msg = (
                "🤖 <b>NewsScout Befehle</b>\n\n"
                "• <code>/track &lt;owner/repo&gt;</code> — Fügt ein GitHub-Repo zum Release-Radar hinzu\n"
                "• <code>/interest &lt;keyword&gt;</code> — Fügt ein Keyword zum Hacker News Radar hinzu\n"
                "• <code>/radar</code> — Zeigt alle aktuell überwachten Quellen & Keywords\n"
                "• <code>/help</code> — Zeigt diese Hilfe an"
            )
            await self.send_message(help_msg, chat_id=chat_id, parse_mode="HTML")
            return {"action": "help"}

        elif cmd == "/track" and len(parts) >= 2:
            repo_slug = parts[1].strip()
            if self._db:
                from newsscout.dashboard.queries import add_tracked_repo
                res = await add_tracked_repo(self._db, repo_slug)
                if res.get("success"):
                    reply = f"✅ <b>Repository hinzugefügt</b>\n<code>{res['repo']}</code> wird jetzt überwacht."
                else:
                    reply = f"❌ Fehler: {res.get('error', 'Ungültiges Repo')}"
            else:
                reply = f"✅ <code>{repo_slug}</code> erfasst (DB nicht direkt verbunden)."
            await self.send_message(reply, chat_id=chat_id, parse_mode="HTML")
            return {"action": "track", "repo": repo_slug}

        elif cmd == "/interest" and len(parts) >= 2:
            kw = " ".join(parts[1:]).strip().lower()
            if self._db:
                from newsscout.dashboard.queries import add_interest_keyword
                res = await add_interest_keyword(self._db, kw)
                if res.get("success"):
                    reply = f"🎯 <b>Keyword hinzugefügt</b>\n'<code>{kw}</code>' ist jetzt im Hacker News Radar aktiv."
                else:
                    reply = f"❌ Fehler: {res.get('error', 'Ungültiges Keyword')}"
            else:
                reply = f"🎯 Keyword '<code>{kw}</code>' erfasst."
            await self.send_message(reply, chat_id=chat_id, parse_mode="HTML")
            return {"action": "interest", "keyword": kw}

        elif cmd in ("/radar", "/topics"):
            if self._db:
                from newsscout.dashboard.queries import get_topic_radar
                radar = await get_topic_radar(self._db)
                repos = "\n".join(f"• <code>{r}</code>" for r in radar.get("tracked_repos", [])) or "<i>Keine</i>"
                kws = ", ".join(f"<code>{k}</code>" for k in radar.get("custom_keywords", [])) or "<i>Standard</i>"
                reply = (
                    "📡 <b>Aktueller NewsScout Themenradar</b>\n\n"
                    f"<b>GitHub Repositories:</b>\n{repos}\n\n"
                    f"<b>Custom Keywords:</b>\n{kws}"
                )
            else:
                reply = "📡 Themenradar aktiv."
            await self.send_message(reply, chat_id=chat_id, parse_mode="HTML")
            return {"action": "radar"}

        return {"action": "unknown", "text": clean_text}

    async def poll_updates(self, offset: int = 0, timeout: int = 5) -> tuple[int, list[dict[str, Any]]]:
        """Polls for new Telegram updates via getUpdates."""
        url = self._base_url + "/getUpdates"
        payload = {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }
        try:
            res = await self.client.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                updates = data.get("result", [])
                next_offset = offset
                for u in updates:
                    uid = u.get("update_id", 0)
                    if uid >= next_offset:
                        next_offset = uid + 1
                return next_offset, updates
        except Exception as err:
            logger.debug("poll_updates error: %s", err)
        return offset, []

    async def process_update(self, update: dict[str, Any]) -> dict[str, Any]:
        """Dispatches an incoming update to callback or message handlers."""
        if "callback_query" in update:
            cq = update["callback_query"]
            data = cq.get("data", "")
            cq_id = cq.get("id", "")
            user = cq.get("from", {})
            msg = cq.get("message", {})
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            return await self.handle_callback_query(
                callback_data=data,
                callback_query_id=cq_id,
                telegram_user_id=user.get("id"),
                chat_id=chat_id,
            )
        elif "message" in update:
            msg = update["message"]
            text = msg.get("text", "")
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            return await self.handle_message_text(text=text, chat_id=chat_id)
        return {"action": "ignored"}
