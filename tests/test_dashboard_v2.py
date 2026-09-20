"""tests.test_dashboard_v2
~~~~~~~~~~~~~~~~~~~~~~~
Integration and unit tests for Milestone 4 Dashboard extensions:
- GET /api/system/status (LLM, search, gateways status)
- GET /api/gateways/pairing/{channel} (WhatsApp and Signal pairing endpoints)
- Web dashboard HTML template verification for System & Gateways tab
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from newsscout.config import Settings
from newsscout.dashboard.app import create_app
from newsscout.delivery.base import PairingQRResult
from newsscout.storage.db import Database
from newsscout.storage.migrations import apply_migrations


@pytest.fixture
def v2_db(tmp_path: Path) -> Database:
    db_file = tmp_path / "dashboard_v2.db"
    db = Database(db_file)
    with db.get_sync_connection() as c:
        apply_migrations(conn=c)
    return db


@pytest.fixture
def v2_settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "dashboard_v2.db",
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3.1:8b",
        searxng_enabled=True,
        searxng_base_url="http://localhost:8080",
        duckduckgo_enabled=True,
        tavily_api_key=SecretStr("tvly-sample-key"),
        whatsapp_enabled=True,
        whatsapp_bridge_url="http://localhost:3000",
        whatsapp_recipient_id="491701234567@c.us",
        signal_enabled=True,
        signal_bridge_url="http://localhost:8085",
        signal_sender_number="+49170111222",
        signal_recipient_id="+49170333444",
    )


@pytest.fixture
async def v2_client(v2_db: Database, v2_settings: Settings):
    app = create_app(v2_settings)
    app.state.db = v2_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_system_status_endpoint(v2_client: AsyncClient):
    """GET /api/system/status returns complete operational status."""
    resp = await v2_client.get("/api/system/status")
    assert resp.status_code == 200
    data = resp.json()

    assert "llm" in data
    assert "search" in data
    assert "gateways" in data

    # Check LLM
    llm = data["llm"]
    assert llm["provider"] == "openai_compatible"
    assert llm["model"] == "llama3.1:8b"
    assert llm["base_url"] == "http://localhost:11434/v1"
    assert llm["configured"] is True

    # Check Search
    search = data["search"]
    assert search["searxng"]["enabled"] is True
    assert search["searxng"]["zero_key"] is True
    assert search["duckduckgo"]["enabled"] is True
    assert search["duckduckgo"]["zero_key"] is True
    assert search["tavily"]["enabled"] is True
    assert search["tavily"]["zero_key"] is False
    assert search["exa"]["enabled"] is False

    # Check Gateways
    gw = data["gateways"]
    assert gw["whatsapp"]["enabled"] is True
    assert gw["whatsapp"]["recipients_count"] == 1
    assert gw["signal"]["enabled"] is True
    assert gw["signal"]["recipients_count"] == 1


@pytest.mark.asyncio
async def test_gateway_pairing_unsupported_channel(v2_client: AsyncClient):
    """GET /api/gateways/pairing/discord returns 400 Bad Request."""
    resp = await v2_client.get("/api/gateways/pairing/discord")
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_gateway_pairing_whatsapp_status(v2_client: AsyncClient):
    """GET /api/gateways/pairing/whatsapp returns status object."""
    mock_qr = PairingQRResult(
        channel="whatsapp",
        status="pairing_required",
        qr_data="2@sample-whatsapp-qr-payload",
    )

    with patch("newsscout.delivery.whatsapp_gateway.WhatsAppGateway.get_pairing_status", new_callable=AsyncMock) as mock_status:
        mock_status.return_value = mock_qr
        resp = await v2_client.get("/api/gateways/pairing/whatsapp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channel"] == "whatsapp"
        assert data["status"] == "pairing_required"
        assert data["qr_data"] == "2@sample-whatsapp-qr-payload"


@pytest.mark.asyncio
async def test_gateway_pairing_signal_status(v2_client: AsyncClient):
    """GET /api/gateways/pairing/signal returns status object."""
    mock_qr = PairingQRResult(
        channel="signal",
        status="connected",
    )

    with patch("newsscout.delivery.signal_gateway.SignalGateway.get_pairing_status", new_callable=AsyncMock) as mock_status:
        mock_status.return_value = mock_qr
        resp = await v2_client.get("/api/gateways/pairing/signal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["channel"] == "signal"
        assert data["status"] == "connected"


@pytest.mark.asyncio
async def test_dashboard_html_contains_system_tab(v2_client: AsyncClient):
    """HTML dashboard includes System & Gateways tab and modal."""
    resp = await v2_client.get("/")
    assert resp.status_code == 200
    text = resp.text
    assert 'data-tab="system"' in text
    assert 'System & Gateways' in text
    assert 'tab-system' in text
    assert 'llm-status-content' in text
    assert 'search-status-content' in text
    assert 'gateways-status-content' in text
