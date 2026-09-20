"""newsscout.delivery.dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified multi-channel delivery dispatcher for NewsScout.
Manages registered messenger gateways (Telegram, WhatsApp, Signal) and orchestrates
concurrent broadcasting with per-channel fault isolation.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from newsscout.config import Settings, get_settings
from newsscout.delivery.base import (
    BaseMessengerGateway,
    DeliveryReceipt,
    PairingQRResult,
)
from newsscout.storage.models import Breakthrough, DecisionCard, Track
from newsscout.storage.preferences import PreferencesService

logger = logging.getLogger("newsscout.delivery.dispatcher")


class DeliveryDispatcher:
    """Dispatches decision cards, digests, and audio files across active messaging gateways."""

    def __init__(
        self,
        gateways: Optional[list[BaseMessengerGateway]] = None,
        settings: Optional[Settings] = None,
        preferences_service: Optional[PreferencesService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.preferences_service = preferences_service
        self._gateways: dict[str, BaseMessengerGateway] = {}
        self._recent_deliveries: OrderedDict[tuple[str, str], int] = OrderedDict()  # LRU, capped at 1000
        self._max_recent_deliveries: int = 1000
        self._latest_breakthrough_id: Optional[int] = None

        if gateways:
            for gw in gateways:
                self.register_gateway(gw)

    def register_gateway(self, gateway: BaseMessengerGateway) -> None:
        """Registers a messenger gateway under its channel_name."""
        self._gateways[gateway.channel_name] = gateway
        logger.info("Registered delivery gateway for channel: '%s'", gateway.channel_name)

    def unregister_gateway(self, channel_name: str) -> Optional[BaseMessengerGateway]:
        """Removes and returns a gateway by channel name."""
        return self._gateways.pop(channel_name, None)

    def get_gateway(self, channel_name: str) -> Optional[BaseMessengerGateway]:
        """Retrieves a registered gateway by channel name."""
        return self._gateways.get(channel_name)

    def get_all_gateways(self) -> list[BaseMessengerGateway]:
        """Returns all registered gateways."""
        return list(self._gateways.values())

    def get_enabled_gateways(self) -> list[BaseMessengerGateway]:
        """Returns all registered gateways where is_enabled() is True."""
        return [gw for gw in self._gateways.values() if gw.is_enabled()]

    def get_breakthrough_id_for_message(self, channel: str, message_id: str) -> Optional[int]:
        """Looks up the breakthrough_id for a previously delivered message."""
        return self._recent_deliveries.get((channel, str(message_id)))

    def get_latest_breakthrough_id(self) -> Optional[int]:
        """Returns the ID of the most recently broadcast breakthrough."""
        return self._latest_breakthrough_id

    async def broadcast_card(
        self,
        card: Union[DecisionCard, Breakthrough],
        recipients: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, DeliveryReceipt]:
        """Broadcasts a DecisionCard or Breakthrough to all enabled gateways concurrently.

        Guarantees:
        - Concurrent execution via asyncio.gather.
        - Strict per-channel fault isolation (exceptions are captured in DeliveryReceipt).
        - Per-gateway timeout enforcement (R4): unstable gateways (Signal, WhatsApp)
          are wrapped with asyncio.wait_for to prevent them from blocking the entire round.
        - Tracks (channel, message_id) -> breakthrough_id in delivery cache.
        """
        enabled = self.get_enabled_gateways()
        if not enabled:
            logger.warning("No enabled delivery gateways available for broadcast_card.")
            return {}

        recipients = recipients or {}

        # Extract breakthrough id
        if isinstance(card, Breakthrough):
            breakthrough_id = card.id
        else:
            breakthrough_id = card.breakthrough_id

        # Per-gateway timeout (R4): ALL gateways get timeout enforcement
        gateway_timeout = getattr(
            self.settings, "delivery_gateway_timeout_seconds", 10.0
        )

        async def _safe_send_card(gw: BaseMessengerGateway) -> tuple[str, DeliveryReceipt]:
            channel = gw.channel_name
            rec = recipients.get(channel)
            try:
                receipt = await asyncio.wait_for(
                    gw.send_card(card, recipient=rec),
                    timeout=gateway_timeout,
                )
                if not isinstance(receipt, DeliveryReceipt):
                    receipt = DeliveryReceipt(
                        channel=channel,
                        success=bool(receipt),
                        message_id=str(receipt) if receipt else None,
                    )
                return channel, receipt
            except asyncio.TimeoutError:
                logger.warning(
                    "[dispatcher] Gateway '%s' timed out after %.1fs (card delivery)",
                    channel,
                    gateway_timeout,
                )
                return channel, DeliveryReceipt(
                    channel=channel,
                    success=False,
                    error=f"Gateway timeout after {gateway_timeout:.1f}s",
                )
            except Exception as exc:
                logger.error("Delivery error on channel '%s': %s", channel, exc, exc_info=True)
                return channel, DeliveryReceipt(
                    channel=channel,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)}",
                )

        tasks = [_safe_send_card(gw) for gw in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        receipts: dict[str, DeliveryReceipt] = {}
        for channel, receipt in results:
            receipts[channel] = receipt
            if receipt.success and receipt.message_id and breakthrough_id is not None:
                key = (channel, str(receipt.message_id))
                self._recent_deliveries[key] = breakthrough_id
                # LRU eviction: remove oldest entries beyond cap
                while len(self._recent_deliveries) > self._max_recent_deliveries:
                    self._recent_deliveries.popitem(last=False)

        if breakthrough_id is not None:
            self._latest_breakthrough_id = breakthrough_id

        return receipts

    async def broadcast_audio(
        self,
        file_path: Union[str, Path, Track],
        caption: str = "",
        recipients: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, DeliveryReceipt]:
        """Broadcasts an audio file/digest to all enabled gateways concurrently.

        Per-gateway timeout enforcement (R4): unstable gateways (Signal, WhatsApp)
        are wrapped with asyncio.wait_for to prevent them from blocking the entire round.
        """
        enabled = self.get_enabled_gateways()
        if not enabled:
            logger.warning("No enabled delivery gateways available for broadcast_audio.")
            return {}

        recipients = recipients or {}

        # Per-gateway timeout (R4): ALL gateways get timeout enforcement
        gateway_timeout = getattr(
            self.settings, "delivery_gateway_timeout_seconds", 10.0
        )

        async def _safe_send_audio(gw: BaseMessengerGateway) -> tuple[str, DeliveryReceipt]:
            channel = gw.channel_name
            rec = recipients.get(channel)
            try:
                receipt = await asyncio.wait_for(
                    gw.send_audio(file_path=file_path, caption=caption, recipient=rec),
                    timeout=gateway_timeout,
                )
                if not isinstance(receipt, DeliveryReceipt):
                    receipt = DeliveryReceipt(
                        channel=channel,
                        success=bool(receipt),
                        message_id=str(receipt) if receipt else None,
                    )
                return channel, receipt
            except asyncio.TimeoutError:
                logger.warning(
                    "[dispatcher] Gateway '%s' timed out after %.1fs (audio delivery)",
                    channel,
                    gateway_timeout,
                )
                return channel, DeliveryReceipt(
                    channel=channel,
                    success=False,
                    error=f"Gateway timeout after {gateway_timeout:.1f}s",
                )
            except Exception as exc:
                logger.error("Audio delivery error on channel '%s': %s", channel, exc, exc_info=True)
                return channel, DeliveryReceipt(
                    channel=channel,
                    success=False,
                    error=f"{type(exc).__name__}: {str(exc)}",
                )

        tasks = [_safe_send_audio(gw) for gw in enabled]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)

    async def send_to_channel(
        self,
        channel: str,
        card: Optional[Union[DecisionCard, Breakthrough]] = None,
        audio_path: Optional[Union[str, Path, Track]] = None,
        caption: str = "",
        recipient: Optional[str] = None,
    ) -> DeliveryReceipt:
        """Dispatches a card or audio to a specific targeted channel."""
        gw = self.get_gateway(channel)
        if gw is None:
            return DeliveryReceipt(
                channel=channel,
                success=False,
                error=f"Gateway for channel '{channel}' is not registered.",
            )
        if not gw.is_enabled():
            return DeliveryReceipt(
                channel=channel,
                success=False,
                error=f"Gateway for channel '{channel}' is disabled.",
            )

        gateway_timeout = getattr(
            self.settings, "delivery_gateway_timeout_seconds", 10.0
        )

        try:
            if card is not None:
                receipt = await asyncio.wait_for(
                    gw.send_card(card, recipient=recipient),
                    timeout=gateway_timeout,
                )
                bid = card.id if isinstance(card, Breakthrough) else card.breakthrough_id
                if receipt.success and receipt.message_id and bid is not None:
                    key = (channel, str(receipt.message_id))
                    self._recent_deliveries[key] = bid
                    # LRU eviction: remove oldest entries beyond cap
                    while len(self._recent_deliveries) > self._max_recent_deliveries:
                        self._recent_deliveries.popitem(last=False)
                    self._latest_breakthrough_id = bid
                return receipt
            elif audio_path is not None:
                return await asyncio.wait_for(
                    gw.send_audio(audio_path, caption=caption, recipient=recipient),
                    timeout=gateway_timeout,
                )
            else:
                return DeliveryReceipt(
                    channel=channel,
                    success=False,
                    error="Neither card nor audio_path was provided for delivery.",
                )
        except asyncio.TimeoutError:
            logger.warning(
                "[dispatcher] Targeted send to '%s' timed out after %.1fs", channel, gateway_timeout
            )
            return DeliveryReceipt(
                channel=channel,
                success=False,
                error=f"Gateway timeout after {gateway_timeout:.1f}s",
            )
        except Exception as exc:
            logger.error("Targeted send error on channel '%s': %s", channel, exc, exc_info=True)
            return DeliveryReceipt(
                channel=channel,
                success=False,
                error=f"{type(exc).__name__}: {str(exc)}",
            )

    async def get_all_pairing_statuses(self) -> dict[str, PairingQRResult]:
        """Queries pairing and connectivity status from all registered gateways concurrently."""
        if not self._gateways:
            return {}

        async def _safe_status(gw: BaseMessengerGateway) -> tuple[str, PairingQRResult]:
            channel = gw.channel_name
            try:
                status = await gw.get_pairing_status()
                return channel, status
            except Exception as exc:
                logger.error("Failed to fetch pairing status for '%s': %s", channel, exc)
                return channel, PairingQRResult(
                    channel=channel,
                    status="error",
                    error=f"{type(exc).__name__}: {str(exc)}",
                )

        tasks = [_safe_status(gw) for gw in self._gateways.values()]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return dict(results)

    async def close(self) -> None:
        """Closes all gateways and releases resources with a 3s timeout per gateway."""
        for gw in self._gateways.values():
            if hasattr(gw, "close") and callable(gw.close):
                try:
                    res = gw.close()
                    if asyncio.iscoroutine(res):
                        await asyncio.wait_for(res, timeout=3.0)
                except asyncio.TimeoutError:
                    logger.warning("Gateway '%s' close() timed out after 3.0s", gw.channel_name)
                except Exception as err:
                    logger.debug("Error closing gateway '%s': %s", gw.channel_name, err)

    async def __aenter__(self) -> "DeliveryDispatcher":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


def create_default_dispatcher(
    settings: Optional[Settings] = None,
    db: Optional[Any] = None,
    preferences_service: Optional[PreferencesService] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> DeliveryDispatcher:
    """Factory function creating a DeliveryDispatcher with Telegram, WhatsApp, and Signal gateways."""
    settings = settings or get_settings()
    dispatcher = DeliveryDispatcher(settings=settings, preferences_service=preferences_service)

    # 1. Telegram Gateway
    try:
        from newsscout.delivery.telegram import TelegramGateway
        dispatcher.register_gateway(
            TelegramGateway(settings=settings, client=client, preferences_service=preferences_service, db=db)
        )
    except ImportError:
        logger.debug("TelegramGateway not available")

    # 2. WhatsApp Gateway
    try:
        from newsscout.delivery.whatsapp_gateway import WhatsAppGateway
        dispatcher.register_gateway(WhatsAppGateway(settings=settings, client=client))
    except ImportError:
        logger.debug("WhatsAppGateway not available")

    # 3. Signal Gateway
    try:
        from newsscout.delivery.signal_gateway import SignalGateway
        dispatcher.register_gateway(SignalGateway(settings=settings, client=client))
    except ImportError:
        logger.debug("SignalGateway not available")

    return dispatcher
