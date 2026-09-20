"""newsscout.delivery.whatsapp_gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WhatsApp Gateway client targeting containerized WAHA / Baileys REST bridge.
Formats 1-minute decision cards with WhatsApp-specific markdown and reply legends,
dispatches audio tracks, and retrieves QR pairing status.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
import random
import re
from typing import Any, Optional, Union

import httpx
from pydantic import SecretStr

from newsscout.config import Settings, get_settings
from newsscout.delivery.base import (
    BaseMessengerGateway,
    DeliveryReceipt,
    PairingQRResult,
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_PAIRING_REQUIRED,
)
from newsscout.storage.models import Breakthrough, DecisionCard, Stage2Category, Track

logger = logging.getLogger("newsscout.delivery.whatsapp")


class WhatsAppGateway(BaseMessengerGateway):
    """WhatsApp Gateway client targeting WAHA (WhatsApp HTTP API) / Baileys bridge."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        base_url: Optional[str] = None,
        session_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._base_url = (
            base_url
            or getattr(self.settings, "whatsapp_bridge_url", "http://localhost:3000")
        ).rstrip("/")
        self._session = (
            session_name
            or getattr(self.settings, "whatsapp_session", "default")
        )
        token_val = getattr(self.settings, "whatsapp_bridge_token", SecretStr(""))
        self._token = token_val.get_secret_value() if hasattr(token_val, "get_secret_value") else str(token_val)
        self._timeout = timeout or getattr(self.settings, "delivery_timeout_seconds", 15.0)
        self._external_client = client is not None
        self._client: Optional[httpx.AsyncClient] = client

    @property
    def channel_name(self) -> str:
        """Returns the canonical channel identifier."""
        return "whatsapp"

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["X-Api-Key"] = self._token
        return headers

    async def __aenter__(self) -> "WhatsAppGateway":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0, read=60.0),
                headers=self._get_headers(),
            )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if not self._external_client and self._client and not self._client.is_closed:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0, read=60.0),
                headers=self._get_headers(),
            )
        return self._client

    def is_enabled(self) -> bool:
        """Returns True if WhatsApp is enabled, has bridge URL and configured recipient."""
        enabled = getattr(self.settings, "whatsapp_enabled", False)
        recipients = getattr(self.settings, "effective_whatsapp_recipients", None)
        if recipients is None:
            recipients = getattr(self.settings, "whatsapp_recipients", [])
            single = getattr(self.settings, "whatsapp_recipient_id", "")
            if single:
                recipients = [single] + list(recipients)
        return bool(enabled and self._base_url and recipients)

    async def is_connected(self) -> bool:
        """Checks if the WAHA bridge session is running and authenticated."""
        try:
            url = f"{self._base_url}/api/sessions/{self._session}"
            resp = await self.client.get(url, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status", "").upper() in ("WORKING", "CONNECTED")
            return False
        except Exception:
            return False

    @staticmethod
    def normalize_chat_id(recipient: str) -> str:
        """Normalizes phone number to WhatsApp JID format (e.g. 491701234567@c.us)."""
        clean = recipient.strip()
        if "@" in clean:
            return clean
        digits = re.sub(r"[^\d]", "", clean)
        return f"{digits}@c.us"

    def _get_default_recipient(self) -> Optional[str]:
        recipients = getattr(self.settings, "effective_whatsapp_recipients", None)
        if recipients:
            return recipients[0]
        raw_recipients = getattr(self.settings, "whatsapp_recipients", [])
        if raw_recipients:
            return raw_recipients[0]
        single = getattr(self.settings, "whatsapp_recipient_id", "")
        return single.strip() if single.strip() else None

    @staticmethod
    def format_whatsapp_card(card: DecisionCard) -> str:
        """Renders DecisionCard into WhatsApp-flavored Markdown.

        WhatsApp uses *bold*, _italic_, ~strike~, ```code```.
        URLs must be bare (WhatsApp does not support named markdown links).
        Appends [#breakthrough_id] and interactive reply legend.
        """
        badge = (
            "*🎯 CORE BREAKTHROUGH*"
            if card.category == Stage2Category.CORE
            else "*🚀 SERENDIPITY HIT*"
        )
        repo_line = f"🔗 {card.repo_url}\n" if card.repo_url else ""
        ref_id = f" [#{card.breakthrough_id}]" if card.breakthrough_id is not None else ""

        return (
            f"⚡ *{card.title}*{ref_id}\n"
            f"{badge} | Score: *{card.breakthrough_score:.1f}/10* | ROI: *{card.roi_score:.1f}/10*\n"
            f"{repo_line}\n"
            f"💡 *TL;DR*: {card.data.tldr}\n\n"
            f"🛠 *Workflow Use Case*: {card.data.use_case}\n"
            f"📊 *Baseline Comparison*: {card.data.comparison}\n"
            f"🚀 *Quickstart*:\n"
            f"```{card.data.quickstart}```\n\n"
            f"💻 *Hardware & License*: {card.data.hardware_requirements} | _{card.data.license}_\n\n"
            f"──────────────\n"
            f"Antworte mit:\n"
            f"🎯 Hit | 💤 Hype | ✅ Bekannt | 🚀 Inspiration"
        )

    async def send_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a 1-minute decision card formatted for WhatsApp."""
        if isinstance(card, Breakthrough):
            card = card.to_decision_card()

        target = recipient or self._get_default_recipient()
        if not target:
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                error="No WhatsApp recipient configured or specified",
            )

        chat_id = self.normalize_chat_id(target)
        text = self.format_whatsapp_card(card)
        payload = {
            "session": self._session,
            "chatId": chat_id,
            "text": text,
        }

        try:
            resp_data = await self._post_with_retry(f"{self._base_url}/api/sendText", payload)
            msg_id = resp_data.get("id") or resp_data.get("message_id")
            return DeliveryReceipt(
                channel="whatsapp",
                success=True,
                recipient=chat_id,
                message_id=str(msg_id) if msg_id else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("WhatsApp send_card failed for %s: %s", chat_id, exc)
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                recipient=chat_id,
                error=str(exc),
            )

    async def send_audio(
        self,
        file_path: Union[str, Path, Track],
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends an audio track (MP3) or voice note to WhatsApp recipient."""
        target = recipient or self._get_default_recipient()
        if not target:
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                error="No WhatsApp recipient configured or specified",
            )

        chat_id = self.normalize_chat_id(target)

        # Normalize Track or path
        if isinstance(file_path, Track):
            p = Path(file_path.audio_path)
            if not caption:
                mins = file_path.duration_seconds // 60
                secs = file_path.duration_seconds % 60
                caption = f"🎵 {file_path.title} ({mins}:{secs:02d})"
        else:
            p = Path(file_path)

        if not p.exists() or not p.is_file():
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                recipient=chat_id,
                error=f"Audio file not found: {p}",
            )

        try:
            audio_bytes = p.read_bytes()
            b64_audio = base64.b64encode(audio_bytes).decode("ascii")

            payload = {
                "session": self._session,
                "chatId": chat_id,
                "file": {
                    "mimetype": "audio/mpeg",
                    "filename": p.name,
                    "data": b64_audio,
                },
                "caption": caption or f"🎵 {p.stem}",
            }

            resp_data = await self._post_with_retry(f"{self._base_url}/api/sendFile", payload)
            msg_id = resp_data.get("id") or resp_data.get("message_id")
            return DeliveryReceipt(
                channel="whatsapp",
                success=True,
                recipient=chat_id,
                message_id=str(msg_id) if msg_id else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("WhatsApp send_audio failed for %s: %s", chat_id, exc)
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                recipient=chat_id,
                error=str(exc),
            )

    async def send_text(
        self,
        text: str,
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a plain or markdown text message to WhatsApp recipient."""
        target = recipient or self._get_default_recipient()
        if not target:
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                error="No WhatsApp recipient configured or specified",
            )

        chat_id = self.normalize_chat_id(target)
        payload = {
            "session": self._session,
            "chatId": chat_id,
            "text": text,
        }

        try:
            resp_data = await self._post_with_retry(f"{self._base_url}/api/sendText", payload)
            msg_id = resp_data.get("id") or resp_data.get("message_id")
            return DeliveryReceipt(
                channel="whatsapp",
                success=True,
                recipient=chat_id,
                message_id=str(msg_id) if msg_id else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("WhatsApp send_text failed for %s: %s", chat_id, exc)
            return DeliveryReceipt(
                channel="whatsapp",
                success=False,
                recipient=chat_id,
                error=str(exc),
            )

    async def get_pairing_status(self) -> PairingQRResult:
        """Queries session status and retrieves QR code if pairing is needed."""
        if not getattr(self.settings, "whatsapp_enabled", False):
            return PairingQRResult(
                channel="whatsapp",
                status=STATUS_DISABLED,
                message="WhatsApp Gateway ist in der Konfiguration deaktiviert.",
            )

        try:
            status_url = f"{self._base_url}/api/sessions/{self._session}"
            resp = await self.client.get(status_url, timeout=5.0)

            if resp.status_code == 200:
                s_data = resp.json()
                if s_data.get("status", "").upper() in ("WORKING", "CONNECTED"):
                    return PairingQRResult(
                        channel="whatsapp",
                        status=STATUS_CONNECTED,
                        message=f"WhatsApp Session '{self._session}' ist verbunden.",
                        raw_details=s_data,
                    )

            # Fetch QR code image
            qr_url = f"{self._base_url}/api/{self._session}/auth/qr?format=image"
            qr_resp = await self.client.get(qr_url, timeout=5.0)
            if qr_resp.status_code == 200 and qr_resp.content:
                content_type = qr_resp.headers.get("content-type", "")
                if "image" in content_type:
                    b64_qr = base64.b64encode(qr_resp.content).decode("ascii")
                    return PairingQRResult(
                        channel="whatsapp",
                        status=STATUS_PAIRING_REQUIRED,
                        qr_image_url=f"data:image/png;base64,{b64_qr}",
                        message="Scanne diesen QR-Code in WhatsApp unter Verknüpfte Geräte.",
                    )
                else:
                    data = qr_resp.json()
                    return PairingQRResult(
                        channel="whatsapp",
                        status=STATUS_PAIRING_REQUIRED,
                        qr_data=data.get("qr") or data.get("qr_raw"),
                        qr_image_url=data.get("qr_data_url"),
                        message="Scanne diesen QR-Code in WhatsApp unter Verknüpfte Geräte.",
                        raw_details=data,
                    )

            # Fallback: screenshot
            ss_url = f"{self._base_url}/api/screenshot?session={self._session}"
            ss_resp = await self.client.get(ss_url, timeout=5.0)
            if ss_resp.status_code == 200 and ss_resp.content:
                b64_qr = base64.b64encode(ss_resp.content).decode("ascii")
                return PairingQRResult(
                    channel="whatsapp",
                    status=STATUS_PAIRING_REQUIRED,
                    qr_image_url=f"data:image/png;base64,{b64_qr}",
                    message="Scanne diesen QR-Code in WhatsApp unter Verknüpfte Geräte.",
                )

            return PairingQRResult(
                channel="whatsapp",
                status=STATUS_PAIRING_REQUIRED,
                message=f"Session '{self._session}' wartet auf Kopplung.",
            )
        except Exception as exc:
            logger.warning("WhatsApp pairing check failed: %s", exc)
            return PairingQRResult(
                channel="whatsapp",
                status=STATUS_DISABLED,
                message=f"WhatsApp Bridge nicht erreichbar: {exc}",
                error=str(exc),
            )

    async def _post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Posts payload with exponential backoff on HTTP 429 and >= 500."""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.post(url, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt > max_retries:
                        response.raise_for_status()
                    sleep_sec = min((0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2), 5.0)
                    logger.warning(
                        "WhatsApp bridge HTTP %d, retry in %.2fs (attempt %d/%d)",
                        response.status_code, sleep_sec, attempt, max_retries
                    )
                    await asyncio.sleep(sleep_sec)
                    continue

                response.raise_for_status()
                if response.content:
                    return response.json()
                return {}
            except (httpx.RequestError, httpx.TimeoutException) as err:
                if attempt > max_retries:
                    raise
                sleep_sec = min((0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2), 5.0)
                logger.warning(
                    "WhatsApp bridge error: %s, retry in %.2fs (attempt %d/%d)",
                    err, sleep_sec, attempt, max_retries
                )
                await asyncio.sleep(sleep_sec)
