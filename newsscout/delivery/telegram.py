"""newsscout.delivery.telegram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Telegram Gateway adapter wrapping TelegramBot under the BaseMessengerGateway protocol.
Preserves 100% backward compatibility with inline keyboard callback buttons,
digest menus, and audio track delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from newsscout.config import Settings, get_settings
from newsscout.delivery.base import (
    BaseMessengerGateway,
    DeliveryReceipt,
    PairingQRResult,
    STATUS_CONNECTED,
    STATUS_DISABLED,
)
from newsscout.delivery.telegram_bot import FEEDBACK_BUTTONS, TelegramBot, mask_telegram_url
from newsscout.storage.models import Breakthrough, DecisionCard, Digest, Track
from newsscout.storage.preferences import PreferencesService

logger = logging.getLogger("newsscout.delivery.telegram")


class TelegramGateway(BaseMessengerGateway):
    """Messenger gateway adapter wrapping TelegramBot under BaseMessengerGateway.

    Provides complete interface compatibility for multi-channel dispatch while
    preserving all existing TelegramBot features (inline feedback keyboards,
    callback processing, digest menus, and command routing).
    """

    def __init__(
        self,
        bot: Optional[TelegramBot] = None,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        preferences_service: Optional[PreferencesService] = None,
        db: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bot = bot or TelegramBot(
            settings=self.settings,
            client=client,
            preferences_service=preferences_service,
            db=db,
        )
        self._internal_context: bool = False

    @property
    def channel_name(self) -> str:
        """Returns the canonical channel identifier."""
        return "telegram"

    def is_enabled(self) -> bool:
        """Returns True if the Telegram bot token is configured."""
        token = self.settings.telegram_bot_token.get_secret_value().strip()
        return bool(token)

    async def is_connected(self) -> bool:
        """Returns True if enabled."""
        return self.is_enabled()

    async def get_pairing_status(self) -> PairingQRResult:
        """Returns 'connected' if bot token is set, else 'disabled'.

        Telegram uses cloud bot tokens rather than local QR-code pairing.
        """
        if self.is_enabled():
            return PairingQRResult(
                channel="telegram",
                status=STATUS_CONNECTED,
                qr_data=None,
                qr_image_url=None,
                qr_ascii=None,
                message="Telegram bot token configured and active",
            )
        return PairingQRResult(
            channel="telegram",
            status=STATUS_DISABLED,
            qr_data=None,
            qr_image_url=None,
            qr_ascii=None,
            message="Telegram bot token not configured",
        )

    async def _ensure_client_open(self) -> bool:
        """Helper to ensure underlying TelegramBot client is open even if called outside context manager."""
        client = getattr(self.bot, "_client", None)
        if client is None or (isinstance(client, httpx.AsyncClient) and client.is_closed):
            self.bot._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
            return True
        return False

    async def _safe_close_bot(self) -> None:
        """Safely closes bot if close is async coroutine or callable."""
        close_fn = getattr(self.bot, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                await res

    async def send_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a 1-minute decision card formatted for Telegram with inline feedback buttons.

        Maintains 100% backward compatibility for callback button payload
        contracts ('feedback:{breakthrough_id}:{rating}').
        """
        chat_id = (recipient or self.settings.telegram_chat_id or "").strip()
        if not self.is_enabled():
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id or None,
                error="Telegram bot token not configured",
            )
        if not chat_id:
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=None,
                error="No recipient or telegram_chat_id configured",
            )

        # Normalize card input
        if isinstance(card, Breakthrough):
            breakthrough_id = card.id
            decision_card = card.to_decision_card()
        else:
            breakthrough_id = card.breakthrough_id
            decision_card = card

        created_client = await self._ensure_client_open()
        try:
            html_text = decision_card.render_telegram_html()
            keyboard = self.bot._build_feedback_keyboard(breakthrough_id)

            payload = {
                "chat_id": chat_id,
                "text": html_text,
                "parse_mode": "HTML",
                "reply_markup": json.dumps(keyboard),
            }

            response = await self.bot._post_with_retry("/sendMessage", payload)
            msg_id = response.get("result", {}).get("message_id")
            return DeliveryReceipt(
                channel="telegram",
                success=bool(msg_id),
                message_id=str(msg_id) if msg_id is not None else None,
                recipient=chat_id,
                error=None if msg_id else "No message_id returned by Telegram API",
                details=response,
            )
        except asyncio.TimeoutError:
            logger.warning("TelegramGateway send_card timed out for chat_id=%s", chat_id)
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                message_id=None,
                recipient=chat_id,
                error="Telegram API timeout",
            )
        except Exception as err:
            logger.error("TelegramGateway send_card error: %s", mask_telegram_url(str(err)))
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                message_id=None,
                recipient=chat_id,
                error=str(err),
            )
        finally:
            if created_client and not self._internal_context:
                await self._safe_close_bot()

    async def send_audio(
        self,
        file_path: Union[str, Path, Track],
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends an audio track or voice briefing to the recipient."""
        chat_id = (recipient or self.settings.telegram_chat_id or "").strip()
        if not self.is_enabled():
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id or None,
                error="Telegram bot token not configured",
            )
        if not chat_id:
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=None,
                error="No recipient or telegram_chat_id configured",
            )

        # Normalize Track or path
        if isinstance(file_path, Track):
            audio_path = Path(file_path.audio_path)
            if not caption:
                mins = file_path.duration_seconds // 60
                secs = file_path.duration_seconds % 60
                caption = f"🎵 {file_path.title} ({mins}:{secs:02d})"
        else:
            audio_path = Path(file_path)

        if not audio_path.exists() or not audio_path.is_file():
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id,
                error=f"Audio file not found: {audio_path}",
            )

        created_client = await self._ensure_client_open()
        try:
            files = {
                "audio": (audio_path.name, audio_path.read_bytes(), "audio/mpeg"),
            }
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML",
            }

            response = await self.bot._post_with_retry_multipart("/sendAudio", data, files)
            msg_id = response.get("result", {}).get("message_id")
            return DeliveryReceipt(
                channel="telegram",
                success=bool(msg_id),
                message_id=str(msg_id) if msg_id is not None else None,
                recipient=chat_id,
                error=None if msg_id else "No message_id returned by Telegram API",
                details=response,
            )
        except asyncio.TimeoutError:
            logger.warning("TelegramGateway send_audio timed out for chat_id=%s", chat_id)
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                message_id=None,
                recipient=chat_id,
                error="Telegram API timeout",
            )
        except Exception as err:
            logger.error("TelegramGateway send_audio error: %s", mask_telegram_url(str(err)))
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                message_id=None,
                recipient=chat_id,
                error=str(err),
            )
        finally:
            if created_client and not self._internal_context:
                await self._safe_close_bot()

    async def send_text(
        self,
        text: str,
        recipient: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> DeliveryReceipt:
        """Sends a plain or HTML message to the recipient."""
        chat_id = (recipient or self.settings.telegram_chat_id or "").strip()
        if not self.is_enabled():
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id or None,
                error="Telegram bot token not configured",
            )
        if not chat_id:
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=None,
                error="No recipient or telegram_chat_id configured",
            )

        created_client = await self._ensure_client_open()
        try:
            msg_id = await self.bot.send_message(text=text, chat_id=chat_id, parse_mode=parse_mode)
            return DeliveryReceipt(
                channel="telegram",
                success=bool(msg_id),
                message_id=str(msg_id) if msg_id else None,
                recipient=chat_id,
            )
        except asyncio.TimeoutError:
            logger.warning("TelegramGateway send_text timed out for chat_id=%s", chat_id)
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id,
                error="Telegram API timeout",
            )
        except Exception as err:
            logger.error("TelegramGateway send_text error: %s", mask_telegram_url(str(err)))
            return DeliveryReceipt(
                channel="telegram",
                success=False,
                recipient=chat_id,
                error=str(err),
            )
        finally:
            if created_client and not self._internal_context:
                await self._safe_close_bot()

    # ========================================================================
    # Backward Compatibility Forwarding Methods
    # ========================================================================

    async def send_decision_card(
        self,
        breakthrough: Breakthrough,
        chat_id: Optional[str] = None,
    ) -> int:
        """Direct backward-compatible wrapper returning int message_id."""
        return await self.bot.send_decision_card(breakthrough, chat_id=chat_id)

    async def send_audio_track(
        self,
        track: Track,
        chat_id: Optional[str] = None,
    ) -> int:
        """Direct backward-compatible wrapper returning int message_id."""
        return await self.bot.send_audio_track(track, chat_id=chat_id)

    async def send_digest_menu(
        self,
        digest: Digest,
        chat_id: Optional[str] = None,
    ) -> int:
        """Direct backward-compatible wrapper returning int message_id."""
        return await self.bot.send_digest_menu(digest, chat_id=chat_id)

    async def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> int:
        """Direct backward-compatible wrapper returning int message_id."""
        return await self.bot.send_message(text=text, chat_id=chat_id, parse_mode=parse_mode)

    async def handle_callback_query(
        self,
        callback_data: str,
        callback_query_id: str,
        chat_id: str,
        telegram_user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Direct backward-compatible callback handler."""
        return await self.bot.handle_callback_query(
            callback_data=callback_data,
            callback_query_id=callback_query_id,
            chat_id=chat_id,
            telegram_user_id=telegram_user_id,
        )

    async def handle_message_text(self, text: str, chat_id: str) -> dict[str, Any]:
        """Direct backward-compatible text command handler."""
        return await self.bot.handle_message_text(text=text, chat_id=chat_id)

    async def poll_updates(self, offset: int = 0, timeout: int = 5) -> tuple[int, list[dict[str, Any]]]:
        """Direct backward-compatible update polling."""
        return await self.bot.poll_updates(offset=offset, timeout=timeout)

    async def process_update(self, update: dict[str, Any]) -> dict[str, Any]:
        """Direct backward-compatible update processing."""
        return await self.bot.process_update(update)

    def is_authorized(self, chat_id: str) -> bool:
        """Direct backward-compatible authorization check."""
        return self.bot.is_authorized(chat_id)

    # Context Manager Protocol
    async def __aenter__(self) -> "TelegramGateway":
        self._internal_context = True
        await self.bot.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._internal_context = False
        await self.bot.__aexit__(exc_type, exc_val, exc_tb)

    async def close(self) -> None:
        await self._safe_close_bot()

    def __getattr__(self, item: str) -> Any:
        """Transparently delegates any other methods/properties to self.bot."""
        return getattr(self.bot, item)
