"""SQLite WAL Database Engine for NewsScout.

Provides thread-safe, non-blocking connection management, WAL pragmas,
and transactional context managers for both sync and async execution on Raspberry Pi 5.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import sqlite3
import time
from typing import Any, Final

DEFAULT_BUSY_TIMEOUT_MS: Final[int] = 5000
DEFAULT_CACHE_SIZE_KIB: Final[int] = -64000  # 64 MB
DEFAULT_MMAP_SIZE_BYTES: Final[int] = 268435456  # 256 MB


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Applies required per-connection performance and integrity pragmas."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute(f"PRAGMA busy_timeout = {DEFAULT_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(f"PRAGMA cache_size = {DEFAULT_CACHE_SIZE_KIB};")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute(f"PRAGMA mmap_size = {DEFAULT_MMAP_SIZE_BYTES};")


def create_connection(
    db_path: str | Path,
    timeout: float = 5.0,
    init_wal: bool = True,
) -> sqlite3.Connection:
    """Creates and configures a synchronous SQLite connection with WAL pragmas."""
    str_path = str(db_path)
    if str_path != ":memory:" and not str_path.startswith("file:"):
        path_obj = Path(db_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str_path,
        timeout=timeout,
        check_same_thread=False,
        isolation_level=None,  # Autocommit mode; transactions managed explicitly
    )

    # Apply pragmas (including busy_timeout) BEFORE setting journal mode
    apply_pragmas(conn)

    if init_wal and str_path != ":memory:":
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        current_mode = str(row[0]) if row else ""
        if current_mode.lower() != "wal":
            for attempt in range(5):
                try:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    break
                except sqlite3.OperationalError:
                    # Check if concurrent process completed WAL transition
                    row_check = conn.execute("PRAGMA journal_mode;").fetchone()
                    if row_check and str(row_check[0]).lower() == "wal":
                        break
                    if attempt == 4:
                        raise
                    time.sleep(0.05)

    return conn


@contextmanager
def get_connection(
    db_path: str | Path,
    timeout: float = 5.0,
    init_wal: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a synchronous SQLite connection and ensuring closure."""
    conn = create_connection(db_path, timeout=timeout, init_wal=init_wal)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def sync_transaction(
    conn: sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    """Synchronous transactional context manager using BEGIN IMMEDIATE."""
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


class Database:
    """Asynchronous SQLite WAL Database Manager.

    Wraps blocking SQLite operations in asyncio.to_thread and serializes
    write transactions with an asyncio.Lock to guarantee deadlock-free
    concurrency on Raspberry Pi 5.
    """

    def __init__(self, db_path: str | Path, timeout: float = 5.0) -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else db_path
        self.timeout = timeout
        self._write_lock = asyncio.Lock()
        self._is_initialized = False

    def get_sync_connection(self) -> sqlite3.Connection:
        """Returns a new synchronous connection configured with WAL pragmas."""
        return create_connection(self.db_path, timeout=self.timeout, init_wal=True)

    async def initialize(self) -> None:
        """Initializes database directory and ensures WAL mode is enabled."""
        if self._is_initialized:
            return

        def _init() -> None:
            with get_connection(self.db_path, timeout=self.timeout, init_wal=True) as conn:
                apply_pragmas(conn)

        await asyncio.to_thread(_init)
        self._is_initialized = True

    async def fetch_all(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> list[sqlite3.Row]:
        """Executes a non-blocking SELECT query returning all matching rows."""
        await self.initialize()

        def _read() -> list[sqlite3.Row]:
            with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()

        return await asyncio.to_thread(_read)

    async def fetch_one(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> sqlite3.Row | None:
        """Executes a non-blocking SELECT query returning the first matching row."""
        await self.initialize()

        def _read() -> sqlite3.Row | None:
            with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchone()

        return await asyncio.to_thread(_read)

    async def fetch_val(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
        column: int | str = 0,
    ) -> Any | None:
        """Executes a non-blocking query returning a single scalar value."""
        row = await self.fetch_one(query, params)
        if row is None:
            return None
        return row[column]

    async def execute(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> int:
        """Executes a single write query inside a serialized IMMEDIATE transaction.

        Returns:
            The last inserted row ID (for INSERT) or rows affected (for UPDATE/DELETE).
        """
        await self.initialize()
        async with self._write_lock:

            def _write() -> int:
                with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                    with sync_transaction(conn):
                        cursor = conn.execute(query, params)
                        if cursor.lastrowid is not None and cursor.lastrowid > 0:
                            return cursor.lastrowid
                        return cursor.rowcount

            return await asyncio.to_thread(_write)

    # Convenience alias for code referencing execute_write
    execute_write = execute

    async def execute_many(
        self,
        query: str,
        seq_of_params: Sequence[tuple[Any, ...] | dict[str, Any]],
    ) -> int:
        """Executes a batch write query inside a serialized IMMEDIATE transaction."""
        await self.initialize()
        async with self._write_lock:

            def _write_many() -> int:
                with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                    with sync_transaction(conn):
                        cursor = conn.executemany(query, seq_of_params)
                        return cursor.rowcount

            return await asyncio.to_thread(_write_many)

    async def execute_script(self, script: str) -> None:
        """Executes a multi-statement SQL script inside a serialized transaction."""
        await self.initialize()
        async with self._write_lock:

            def _run_script() -> None:
                with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                    try:
                        conn.executescript(f"BEGIN IMMEDIATE;\n{script}\nCOMMIT;")
                    except Exception:
                        try:
                            conn.execute("ROLLBACK;")
                        except Exception:
                            pass
                        raise

            await asyncio.to_thread(_run_script)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[sqlite3.Connection]:
        """Async context manager yielding a connection inside an IMMEDIATE transaction.

        Usage:
            async with db.transaction() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")
        """
        await self.initialize()
        async with self._write_lock:
            conn = create_connection(self.db_path, timeout=self.timeout, init_wal=False)
            try:
                await asyncio.to_thread(conn.execute, "BEGIN IMMEDIATE;")
                yield conn
                await asyncio.to_thread(conn.execute, "COMMIT;")
            except Exception:
                await asyncio.to_thread(conn.execute, "ROLLBACK;")
                raise
            finally:
                await asyncio.to_thread(conn.close)

    async def checkpoint(self, mode: str = "PASSIVE") -> tuple[int, int, int]:
        """Performs a WAL checkpoint to manage write-ahead log file size.

        Modes: 'PASSIVE', 'FULL', 'RESTART', 'TRUNCATE'.
        Returns: (busy_flag, log_pages, checkpointed_pages).
        """
        valid_modes = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
        target_mode = mode.upper()
        if target_mode not in valid_modes:
            raise ValueError(f"Invalid checkpoint mode '{mode}'. Choose from {valid_modes}")

        await self.initialize()
        async with self._write_lock:

            def _cp() -> tuple[int, int, int]:
                with get_connection(self.db_path, timeout=self.timeout, init_wal=False) as conn:
                    res = conn.execute(f"PRAGMA wal_checkpoint({target_mode});").fetchone()
                    return (int(res[0]), int(res[1]), int(res[2]))

            return await asyncio.to_thread(_cp)

    async def close(self) -> None:
        """Flushes WAL log and checkpoints database before shutdown."""
        if self._is_initialized:
            try:
                await self.checkpoint(mode="TRUNCATE")
            except Exception:
                pass
            self._is_initialized = False
