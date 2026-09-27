import sqlite3
from pathlib import Path
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4


TICKET_STATUSES = frozenset({"open", "in_progress", "resolved", "closed"})
TICKET_PRIORITIES = frozenset({"normal", "high", "urgent"})


class SupportStore:
    """Persist support tickets and customer feedback in SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = RLock()
        self._closed = False

        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_support_tickets_status
                ON support_tickets(status, created_at DESC)
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_customer_feedback_session
                ON customer_feedback(session_id, created_at DESC)
                """
            )
            # v2 之前使用纳秒时间戳。API 对外统一使用 Unix 秒，启动时
            # 自动迁移旧数据库，避免前端和第三方客户端自行猜测单位。
            self._connection.execute(
                """
                UPDATE support_tickets
                SET created_at = created_at / 1000000000,
                    updated_at = updated_at / 1000000000
                WHERE created_at > 1000000000000
                   OR updated_at > 1000000000000
                """
            )
            self._connection.execute(
                """
                UPDATE customer_feedback
                SET created_at = created_at / 1000000000
                WHERE created_at > 1000000000000
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_ticket(
        self,
        session_id: str,
        request_id: str,
        question: str,
        reason: str,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Create one idempotent ticket for a request ID."""
        if priority not in TICKET_PRIORITIES:
            raise ValueError(f"不支持的工单优先级：{priority}")

        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT * FROM support_tickets WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if existing is not None:
                return dict(existing)

            now = int(time())
            ticket_id = f"ticket_{uuid4().hex[:16]}"
            self._connection.execute(
                """
                INSERT INTO support_tickets (
                    ticket_id, session_id, request_id, question, reason,
                    priority, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?)
                """,
                (
                    ticket_id,
                    session_id,
                    request_id,
                    question,
                    reason,
                    priority,
                    now,
                    now,
                ),
            )
            row = self._connection.execute(
                "SELECT * FROM support_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return dict(row)

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM support_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def list_tickets(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in TICKET_STATUSES:
            raise ValueError(f"不支持的工单状态：{status}")
        safe_limit = max(1, min(limit, 100))
        with self._lock:
            if status is None:
                rows = self._connection.execute(
                    """
                    SELECT * FROM support_tickets
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT * FROM support_tickets
                    WHERE status = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (status, safe_limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def update_ticket_status(
        self,
        ticket_id: str,
        status: str,
    ) -> dict[str, Any] | None:
        if status not in TICKET_STATUSES:
            raise ValueError(f"不支持的工单状态：{status}")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE support_tickets
                SET status = ?, updated_at = ?
                WHERE ticket_id = ?
                """,
                (status, int(time()), ticket_id),
            )
            if cursor.rowcount == 0:
                return None
            row = self._connection.execute(
                "SELECT * FROM support_tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def add_feedback(
        self,
        session_id: str,
        request_id: str,
        rating: int,
        comment: str = "",
    ) -> dict[str, Any]:
        if rating < 1 or rating > 5:
            raise ValueError("评分必须在 1 到 5 之间。")
        feedback_id = f"feedback_{uuid4().hex[:16]}"
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO customer_feedback (
                    feedback_id, session_id, request_id,
                    rating, comment, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    session_id,
                    request_id,
                    rating,
                    comment.strip(),
                    int(time()),
                ),
            )
            row = self._connection.execute(
                """
                SELECT * FROM customer_feedback WHERE feedback_id = ?
                """,
                (feedback_id,),
            ).fetchone()
        return dict(row)

    def ping(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            return self._connection.execute("SELECT 1").fetchone()[0] == 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True
