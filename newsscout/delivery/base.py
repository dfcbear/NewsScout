"""newsscout.delivery.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified abstract protocol, data contracts, and exception hierarchy
for multi-messenger delivery gateways (Telegram, Signal, WhatsApp).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Optional, Union

from newsscout.storage.models import Breakthrough, DecisionCard, Track

logger = logging.getLogger(__name__)

# Canonical Pairing Status constants
STATUS_CONNECTED = "connected"
STATUS_PAIRING_REQUIRED = "pairing_required"
STATUS_DISABLED = "disabled"
STATUS_ERROR = "error"


# ============================================================================
# Outbound & Inbound Data Contracts
# ============================================================================

@dataclass
class DeliveryReceipt:
    """Standardized delivery receipt returned by messenger gateways upon dispatch."""

    channel: str
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    recipient: Optional[str] = None
    delivered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes receipt to a dictionary representation."""
        return {
            "channel": self.channel,
            "success": self.success,
            "message_id": self.message_id,
            "error": self.error,
            "recipient": self.recipient,
            "delivered_at": self.delivered_at.isoformat(),
            "details": self.details,
        }


@dataclass
class PairingQRResult:
    """Represents device pairing and QR code status for web UI / terminal display."""

    channel: str
    status: str  # "connected" | "pairing_required" | "disabled" | "error"
    qr_data: Optional[str] = None
    qr_image_url: Optional[str] = None
    qr_ascii: Optional[str] = None
    message: Optional[str] = None
    raw_details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes pairing result to a dictionary representation."""
        return {
            "channel": self.channel,
            "status": self.status,
            "qr_data": self.qr_data,
            "qr_image_url": self.qr_image_url,
            "qr_ascii": self.qr_ascii,
            "message": self.message,
            "error": self.error,
            "raw_details": self.raw_details,
        }


# ============================================================================
# Delivery Exception Hierarchy
# ============================================================================

class DeliveryError(Exception):
    """Base exception for all delivery gateway operations."""


class DeliveryConnectionError(DeliveryError):
    """Raised when connection to messenger API or sidecar fails."""


class DeliveryTimeoutError(DeliveryError):
    """Raised when outbound message or pairing request times out."""


class DeliveryAuthError(DeliveryError):
    """Raised on authentication failure, invalid bot token, or unlinked device."""


class DeliveryRateLimitError(DeliveryError):
    """Raised when outbound rate limit is reached (e.g. HTTP 429)."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class DeliveryChannelDisabledError(DeliveryError):
    """Raised when attempting to deliver on a disabled or unconfigured channel."""


class DeliveryPayloadError(DeliveryError):
    """Raised when payload formatting or validation fails (e.g. missing audio file)."""


# ============================================================================
# Abstract Messenger Gateway Protocol
# ============================================================================

class BaseMessengerGateway(abc.ABC):
    """Abstract interface for all NewsScout messenger delivery gateways."""

    @property
    @abc.abstractmethod
    def channel_name(self) -> str:
        """Returns the canonical channel identifier ('telegram', 'whatsapp', 'signal')."""
        ...

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """Returns True if minimum required credentials/endpoints are set for this gateway."""
        ...

    async def is_connected(self) -> bool:
        """Asynchronously checks if the gateway is authenticated and connected to its network."""
        return self.is_enabled()

    @abc.abstractmethod
    async def send_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a 1-minute decision card formatted for this channel.

        Must shield internal exceptions and return a DeliveryReceipt indicating
        success or error.
        """
        ...

    @abc.abstractmethod
    async def send_audio(
        self,
        file_path: Union[str, Path, Track],
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends an audio track (MP3 or voice note) to the recipient.

        Must shield internal exceptions and return a DeliveryReceipt.
        """
        ...

    @abc.abstractmethod
    async def get_pairing_status(self) -> PairingQRResult:
        """Queries current connection status and pairing QR code if applicable."""
        ...

    async def send_text(
        self,
        text: str,
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Sends a plain or formatted text message. Default implementation returns error receipt."""
        return DeliveryReceipt(
            channel=self.channel_name,
            success=False,
            recipient=recipient,
            error=f"{self.__class__.__name__} does not implement send_text",
        )

    async def close(self) -> None:
        """Closes any underlying network connections or HTTP clients."""
        pass

    async def __aenter__(self) -> "BaseMessengerGateway":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
