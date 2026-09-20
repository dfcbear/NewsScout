"""newsscout.dashboard.security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
API key authentication and rate limiting for the dashboard.

Provides a lightweight in-memory fixed-window rate limiter (no external
dependencies) and a FastAPI dependency that validates the master API key
from ``X-API-Key`` header or ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class FixedWindowRateLimiter:
    """In-memory fixed-window rate limiter keyed by client IP.

    Each IP gets a counter that resets every ``window_seconds``. When the
    counter exceeds ``max_requests`` within the window, subsequent requests
    are denied until the window rolls over.  Entries older than twice the
    window are pruned automatically to prevent unbounded growth.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: dict[str, dict[str, float | int]] = defaultdict(dict)
        self._lock = Lock()

    def check(self, client_ip: str) -> bool:
        """Return *True* if the request is within the rate limit, else *False*.

        Side-effect: increments the counter for the current window.
        """
        now = time.monotonic()
        window_start = int(now // self._window_seconds) * self._window_seconds
        key = f"{client_ip}:{window_start}"

        with self._lock:
            # Prune stale entries (older than 2 windows)
            cutoff = now - 2 * self._window_seconds
            stale_keys = [
                k for k, v in self._buckets.items()
                if isinstance(v.get("ts"), (int, float)) and v["ts"] < cutoff
            ]
            for k in stale_keys:
                del self._buckets[k]

            bucket = self._buckets[key]
            bucket_ts = bucket.get("ts", 0)
            if isinstance(bucket_ts, (int, float)) and bucket_ts < window_start:
                bucket["count"] = 0
                bucket["ts"] = window_start

            count = int(bucket.get("count", 0))
            if count >= self._max_requests:
                return False

            bucket["count"] = count + 1
            bucket["ts"] = window_start
            return True


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------


def verify_api_key(request: Request) -> None:
    """FastAPI dependency that validates the API key for state-changing endpoints.

    Reads the master key from ``app.state.settings.api_secret_key``. When the
    key is empty, authentication is bypassed (local/Tailscale-only mode).
    Supports both ``X-API-Key`` header and ``Authorization: Bearer <token>``.
    """
    settings = request.app.state.settings
    master_key = settings.api_secret_key.get_secret_value().strip()

    # When no key is configured, auth is disabled
    if not master_key:
        return

    # Check X-API-Key header
    provided_key = request.headers.get("X-API-Key", "").strip()
    if provided_key and provided_key == master_key:
        return

    # Check Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token and token == master_key:
            return

    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Provide via 'X-API-Key' header or 'Authorization: Bearer <token>'.",
    )


def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces per-IP rate limiting on POST/DELETE."""
    limiter: FixedWindowRateLimiter = request.app.state.rate_limiter
    client_ip = request.client.host if request.client else "unknown"

    if not limiter.check(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
        )
