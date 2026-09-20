"""newsscout.delivery.signal_gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Signal Gateway client targeting signal-cli-rest-api sidecar.
Formats 1-minute decision cards with CommonMark markdown and reply legends,
dispatches audio tracks as base64 attachments, and handles QR device linking.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
import random
from typing import Any, Optional, Union

import httpx

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

logger = logging.getLogger("newsscout.delivery.signal")


class SignalGateway(BaseMessengerGateway):
    """Signal Gateway client targeting signal-cli-rest-api."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
        api_url: Optional[str] = None,
        number: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._api_url = (
            api_url
            or getattr(self.settings, "signal_bridge_url", "")
            or getattr(self.settings, "signal_api_url", "http://localhost:8080")
        ).rstrip("/")
        self._number = (
            number
            or getattr(self.settings, "signal_sender_number", "")
            or getattr(self.settings, "signal_number", "")
        ).strip()
        self._timeout = timeout or getattr(self.settings, "delivery_timeout_seconds", 15.0)
        self._external_client = client is not None
        self._client: Optional[httpx.AsyncClient] = client

    @property
    def channel_name(self) -> str:
        """Returns the canonical channel identifier."""
        return "signal"

    async def __aenter__(self) -> "SignalGateway":
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0, read=60.0),
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
            )
        return self._client

    def is_enabled(self) -> bool:
        """Returns True if Signal is enabled, has bridge URL, sender number, and recipients."""
        enabled = getattr(self.settings, "signal_enabled", False)
        recipients = getattr(self.settings, "effective_signal_recipients", None)
        if recipients is None:
            recipients = getattr(self.settings, "signal_recipients", [])
            single = getattr(self.settings, "signal_recipient_id", "")
            if single:
                recipients = [single] + list(recipients)
        return bool(enabled and self._api_url and self._number and recipients)

    async def is_connected(self) -> bool:
        """Checks if signal-cli-rest-api is reachable and registered."""
        try:
            resp = await self.client.get(f"{self._api_url}/v1/about", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _get_recipients(self, recipient: Optional[str] = None) -> list[str]:
        if recipient:
            return [recipient.strip()]
        recipients = getattr(self.settings, "effective_signal_recipients", None)
        if recipients:
            return list(recipients)
        raw_recipients = getattr(self.settings, "signal_recipients", [])
        if raw_recipients:
            return list(raw_recipients)
        single = getattr(self.settings, "signal_recipient_id", "")
        return [single.strip()] if single.strip() else []

    @staticmethod
    def format_signal_card(card: DecisionCard) -> str:
        """Renders DecisionCard into standard clean Markdown for Signal."""
        badge = (
            "🎯 **CORE BREAKTHROUGH**"
            if card.category == Stage2Category.CORE
            else "🚀 **SERENDIPITY HIT**"
        )
        repo_line = f" ([Repository]({card.repo_url}))" if card.repo_url else ""
        ref_id = f" [#{card.breakthrough_id}]" if card.breakthrough_id is not None else ""

        return (
            f"⚡ **{card.title}**{ref_id}{repo_line}\n"
            f"{badge} — Score: `{card.breakthrough_score:.1f}/10` | ROI: `{card.roi_score:.1f}/10`\n\n"
            f"> **TL;DR**: {card.data.tldr}\n\n"
            f"* **Workflow Use Case**: {card.data.use_case}\n"
            f"* **Baseline Comparison**: {card.data.comparison}\n"
            f"* **Quickstart**:\n"
            f"```bash\n"
            f"{card.data.quickstart}\n"
            f"```\n"
            f"* **Hardware Requirements & License**: `{card.data.hardware_requirements}` | `{card.data.license}`\n\n"
            f"──────────────\n"
            f"Antworte mit:\n"
            f"🎯 Hit | 💤 Hype | ✅ Bekannt | 🚀 Inspiration"
        )

    async def send_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a 1-minute decision card formatted for Signal."""
        if isinstance(card, Breakthrough):
            card = card.to_decision_card()

        recipients = self._get_recipients(recipient)
        recipients_str = ",".join(recipients)
        if not recipients:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                error="No Signal recipient configured or specified",
            )
        if not self._number:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str or None,
                error="Signal sender number not configured",
            )

        text = self.format_signal_card(card)
        payload = {
            "number": self._number,
            "recipients": recipients,
            "message": text,
        }

        try:
            resp_data = await self._post_with_retry(f"{self._api_url}/v2/send", payload)
            timestamp = resp_data.get("timestamp")
            return DeliveryReceipt(
                channel="signal",
                success=True,
                recipient=recipients_str,
                message_id=str(timestamp) if timestamp else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("Signal send_card failed for %s: %s", recipients, exc)
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str,
                error=str(exc),
            )

    async def send_audio(
        self,
        file_path: Union[str, Path, Track],
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends an audio track (MP3) as base64 attachment to Signal."""
        recipients = self._get_recipients(recipient)
        recipients_str = ",".join(recipients)
        if not recipients:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                error="No Signal recipient configured or specified",
            )
        if not self._number:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str or None,
                error="Signal sender number not configured",
            )

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
                channel="signal",
                success=False,
                recipient=recipients_str,
                error=f"Audio file not found: {p}",
            )

        try:
            audio_bytes = p.read_bytes()
            b64_audio = base64.b64encode(audio_bytes).decode("ascii")

            payload = {
                "number": self._number,
                "recipients": recipients,
                "message": caption or f"🎵 {p.stem}",
                "base64_attachments": [
                    f"data:audio/mpeg;base64,{b64_audio}"
                ],
            }

            resp_data = await self._post_with_retry(f"{self._api_url}/v2/send", payload)
            timestamp = resp_data.get("timestamp")
            return DeliveryReceipt(
                channel="signal",
                success=True,
                recipient=recipients_str,
                message_id=str(timestamp) if timestamp else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("Signal send_audio failed for %s: %s", recipients, exc)
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str,
                error=str(exc),
            )

    async def send_text(
        self,
        text: str,
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a plain or markdown text message to Signal recipients."""
        recipients = self._get_recipients(recipient)
        recipients_str = ",".join(recipients)
        if not recipients:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                error="No Signal recipient configured or specified",
            )
        if not self._number:
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str or None,
                error="Signal sender number not configured",
            )

        payload = {
            "number": self._number,
            "recipients": recipients,
            "message": text,
        }

        try:
            resp_data = await self._post_with_retry(f"{self._api_url}/v2/send", payload)
            timestamp = resp_data.get("timestamp")
            return DeliveryReceipt(
                channel="signal",
                success=True,
                recipient=recipients_str,
                message_id=str(timestamp) if timestamp else None,
                details=resp_data,
            )
        except Exception as exc:
            logger.error("Signal send_text failed for %s: %s", recipients, exc)
            return DeliveryReceipt(
                channel="signal",
                success=False,
                recipient=recipients_str,
                error=str(exc),
            )

    async def get_pairing_status(self) -> PairingQRResult:
        """Fetches device linking QR code from signal-cli-rest-api."""
        if not getattr(self.settings, "signal_enabled", False):
            return PairingQRResult(
                channel="signal",
                status=STATUS_DISABLED,
                message="Signal Gateway ist in der Konfiguration deaktiviert.",
            )

        try:
            # Check reachability
            about_resp = await self.client.get(f"{self._api_url}/v1/about", timeout=5.0)
            if about_resp.status_code != 200:
                return PairingQRResult(
                    channel="signal",
                    status=STATUS_DISABLED,
                    message=f"Signal REST API meldet HTTP {about_resp.status_code}.",
                )

            # Check if account is already registered
            acc_resp = await self.client.get(f"{self._api_url}/v1/accounts", timeout=5.0)
            if acc_resp.status_code == 200:
                accounts = acc_resp.json()
                if isinstance(accounts, list) and self._number in accounts:
                    return PairingQRResult(
                        channel="signal",
                        status=STATUS_CONNECTED,
                        message=f"Signal-Konto {self._number} ist verknüpft und aktiv.",
                    )

            # Request device linking QR code
            device_name = "NewsScout"
            link_url = f"{self._api_url}/v1/qrcodelink?device_name={device_name}"
            qr_resp = await self.client.get(link_url, headers={"Accept": "image/png"}, timeout=10.0)

            if qr_resp.status_code == 200:
                content_type = qr_resp.headers.get("content-type", "")
                if "image" in content_type and qr_resp.content:
                    b64_qr = base64.b64encode(qr_resp.content).decode("ascii")
                    return PairingQRResult(
                        channel="signal",
                        status=STATUS_PAIRING_REQUIRED,
                        qr_image_url=f"data:image/png;base64,{b64_qr}",
                        message="Scanne diesen QR-Code in der Signal-App unter Einstellungen -> Verknüpfte Geräte -> Neues Gerät verknüpfen.",
                    )
                else:
                    data = qr_resp.json()
                    return PairingQRResult(
                        channel="signal",
                        status=STATUS_PAIRING_REQUIRED,
                        qr_data=data.get("uri"),
                        message="Scanne diesen QR-Code in der Signal-App unter Einstellungen -> Verknüpfte Geräte -> Neues Gerät verknüpfen.",
                        raw_details=data,
                    )

            return PairingQRResult(
                channel="signal",
                status=STATUS_PAIRING_REQUIRED,
                message="Signal-Bridge wartet auf Geräteverknüpfung.",
            )
        except Exception as exc:
            logger.warning("Signal pairing check failed: %s", exc)
            return PairingQRResult(
                channel="signal",
                status=STATUS_DISABLED,
                message=f"Signal REST Bridge nicht erreichbar: {exc}",
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
                        "Signal API HTTP %d, retry in %.2fs (attempt %d/%d)",
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
                    "Signal API error: %s, retry in %.2fs (attempt %d/%d)",
                    err, sleep_sec, attempt, max_retries
                )
                await asyncio.sleep(sleep_sec)
