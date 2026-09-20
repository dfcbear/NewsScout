"""newsscout.delivery
~~~~~~~~~~~~~~~~~~~~~~
Unified delivery package: multi-messenger gateways (Telegram, WhatsApp, Signal),
delivery receipts, pairing QR status, delivery dispatcher, and inbound router.
"""

from newsscout.delivery.base import (
    STATUS_CONNECTED,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_PAIRING_REQUIRED,
    BaseMessengerGateway,
    DeliveryAuthError,
    DeliveryChannelDisabledError,
    DeliveryConnectionError,
    DeliveryError,
    DeliveryPayloadError,
    DeliveryRateLimitError,
    DeliveryReceipt,
    DeliveryTimeoutError,
    PairingQRResult,
)
from newsscout.delivery.dispatcher import (
    DeliveryDispatcher,
    create_default_dispatcher,
)
from newsscout.delivery.inbound import (
    InboundHandlingResult,
    InboundMessage,
    InboundRouter,
    parse_feedback_rating,
)
from newsscout.delivery.signal_gateway import SignalGateway
from newsscout.delivery.telegram import TelegramGateway
from newsscout.delivery.telegram_bot import FEEDBACK_BUTTONS, TelegramBot
from newsscout.delivery.whatsapp_gateway import WhatsAppGateway

__all__ = [
    # Status constants
    "STATUS_CONNECTED",
    "STATUS_PAIRING_REQUIRED",
    "STATUS_DISABLED",
    "STATUS_ERROR",
    # Base Protocol & Data Models
    "BaseMessengerGateway",
    "DeliveryReceipt",
    "PairingQRResult",
    # Exceptions
    "DeliveryError",
    "DeliveryConnectionError",
    "DeliveryTimeoutError",
    "DeliveryAuthError",
    "DeliveryRateLimitError",
    "DeliveryChannelDisabledError",
    "DeliveryPayloadError",
    # Gateways & Bot
    "TelegramGateway",
    "WhatsAppGateway",
    "SignalGateway",
    "TelegramBot",
    "FEEDBACK_BUTTONS",
    # Dispatcher & Inbound
    "DeliveryDispatcher",
    "create_default_dispatcher",
    "InboundRouter",
    "InboundMessage",
    "InboundHandlingResult",
    "parse_feedback_rating",
]
