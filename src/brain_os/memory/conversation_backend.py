"""Conversation memory backends: SQLite (default), Postgres dual-write."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from brain_os.config import get_settings
from brain_os.data.conversations import (
    ConversationModel,
    MessageModel,
    PgConversationRepository,
)
from brain_os.data.crm import CRMDatabase

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("data/conversations.db")


class ConversationMemoryBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def should_start_new_conversation(self, user_id: str, channel: str) -> bool: ...

    @abstractmethod
    async def add_message(self, user_id: str, channel: str, role: str, content: str) -> None: ...

    @abstractmethod
    async def get_history(self, user_id: str, channel: str, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    @abstractmethod
    def sqlite_db_connection(self) -> aiosqlite.Connection | None: ...


class SqliteConversationMemoryBackend(ConversationMemoryBackend):
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._db

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path), timeout=30.0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_message_at TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_channel "
            "ON conversations(user_id, channel)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)"
        )
        await self._db.commit()
        logger.info("ConversationMemory (SQLite) initialised at %s", self._db_path)

    async def should_start_new_conversation(self, user_id: str, channel: str) -> bool:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT last_message_at FROM conversations
            WHERE user_id = ? AND channel = ?
            ORDER BY last_message_at DESC
            LIMIT 1
            """,
            (user_id, channel),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return True
        last_at = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        if datetime.now(UTC) - last_at > timedelta(minutes=30):
            return True
        return False

    async def add_message(self, user_id: str, channel: str, role: str, content: str) -> None:
        await self.add_message_with_ids(user_id, channel, role, content)

    async def add_message_with_ids(
        self, user_id: str, channel: str, role: str, content: str
    ) -> tuple[int, int, str]:
        assert self._db is not None
        now = datetime.now(UTC).isoformat()
        if await self.should_start_new_conversation(user_id, channel):
            cursor = await self._db.execute(
                """
                INSERT INTO conversations (user_id, channel, started_at, last_message_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, channel, now, now),
            )
            conversation_id = int(cursor.lastrowid or 0)
            await cursor.close()
        else:
            cursor = await self._db.execute(
                """
                SELECT id FROM conversations
                WHERE user_id = ? AND channel = ?
                ORDER BY last_message_at DESC
                LIMIT 1
                """,
                (user_id, channel),
            )
            row = await cursor.fetchone()
            await cursor.close()
            conversation_id = int(row[0])

        cursor = await self._db.execute(
            """
            INSERT INTO messages (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now),
        )
        message_id = int(cursor.lastrowid or 0)
        await cursor.close()
        await self._db.execute(
            "UPDATE conversations SET last_message_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await self._db.commit()
        return conversation_id, message_id, now

    async def get_history(self, user_id: str, channel: str, limit: int) -> list[dict[str, Any]]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT id FROM conversations
            WHERE user_id = ? AND channel = ?
            ORDER BY last_message_at DESC
            LIMIT 1
            """,
            (user_id, channel),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return []
        conversation_id = row[0]
        cursor = await self._db.execute(
            """
            SELECT role, content, timestamp FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class PgConversationMemoryBackend(ConversationMemoryBackend):
    def __init__(self, repo: PgConversationRepository | None = None) -> None:
        if repo is None:
            repo = PgConversationRepository(CRMDatabase().session_factory)
        self._repo = repo

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return None

    async def initialize(self) -> None:
        await ensure_conversation_tables()

    async def should_start_new_conversation(self, user_id: str, channel: str) -> bool:
        last = await self._repo.get_latest_last_message_at(user_id, channel)
        if last is None:
            return True
        last_at = datetime.fromisoformat(last.replace("Z", "+00:00"))
        if datetime.now(UTC) - last_at > timedelta(minutes=30):
            return True
        return False

    async def add_message(self, user_id: str, channel: str, role: str, content: str) -> None:
        now = datetime.now(UTC).isoformat()
        if await self.should_start_new_conversation(user_id, channel):
            conversation_id = await self._repo.create_conversation(user_id, channel, now, now)
        else:
            cid = await self._repo.get_latest_conversation_id(user_id, channel)
            if cid is None:
                conversation_id = await self._repo.create_conversation(user_id, channel, now, now)
            else:
                conversation_id = cid
        await self._repo.insert_message(conversation_id, role, content, now)
        await self._repo.update_conversation_last_message(conversation_id, now)

    async def shadow_add_message(
        self,
        user_id: str,
        channel: str,
        role: str,
        content: str,
        *,
        conversation_id: int,
        message_id: int,
        new_conversation: bool,
        now: str,
    ) -> None:
        if new_conversation:
            await self._repo.create_conversation(
                user_id,
                channel,
                now,
                now,
                row_id=conversation_id,
            )
        await self._repo.insert_message(
            conversation_id,
            role,
            content,
            now,
            row_id=message_id,
        )
        await self._repo.update_conversation_last_message(conversation_id, now)

    async def get_history(self, user_id: str, channel: str, limit: int) -> list[dict[str, Any]]:
        return await self._repo.get_history(user_id, channel, limit)

    async def close(self) -> None:
        return None


class DualWriteConversationMemoryBackend(ConversationMemoryBackend):
    def __init__(
        self,
        sqlite: SqliteConversationMemoryBackend,
        pg: PgConversationMemoryBackend,
    ) -> None:
        self._sqlite = sqlite
        self._pg = pg

    @property
    def sqlite_db_connection(self) -> aiosqlite.Connection | None:
        return self._sqlite.sqlite_db_connection

    def _read(self) -> ConversationMemoryBackend:
        if getattr(get_settings().app, "pg_store_conversation_read", False):
            return self._pg
        return self._sqlite

    async def initialize(self) -> None:
        await self._sqlite.initialize()
        await self._pg.initialize()

    async def should_start_new_conversation(self, user_id: str, channel: str) -> bool:
        return await self._read().should_start_new_conversation(user_id, channel)

    async def add_message(self, user_id: str, channel: str, role: str, content: str) -> None:
        new_conv = await self._sqlite.should_start_new_conversation(user_id, channel)
        conversation_id, message_id, now = await self._sqlite.add_message_with_ids(
            user_id, channel, role, content
        )
        try:
            await self._pg.shadow_add_message(
                user_id,
                channel,
                role,
                content,
                conversation_id=conversation_id,
                message_id=message_id,
                new_conversation=new_conv,
                now=now,
            )
        except Exception:
            logger.warning("Postgres conversation shadow-write failed", exc_info=True)

    async def get_history(self, user_id: str, channel: str, limit: int) -> list[dict[str, Any]]:
        return await self._read().get_history(user_id, channel, limit)

    async def close(self) -> None:
        await self._sqlite.close()
        await self._pg.close()


def build_conversation_backend(db_path: str | Path | None = None) -> ConversationMemoryBackend:
    path = Path(db_path) if db_path else _DEFAULT_DB_PATH
    sqlite = SqliteConversationMemoryBackend(path)
    if not getattr(get_settings().app, "pg_store_conversation_enabled", False):
        return sqlite
    return DualWriteConversationMemoryBackend(sqlite, PgConversationMemoryBackend())


async def ensure_conversation_tables() -> None:
    crm = CRMDatabase()
    async with crm._engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: ConversationModel.__table__.create(sync_conn, checkfirst=True)
        )
        await conn.run_sync(
            lambda sync_conn: MessageModel.__table__.create(sync_conn, checkfirst=True)
        )
