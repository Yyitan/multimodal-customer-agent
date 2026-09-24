import sqlite3
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import TypedDict


class ConversationTurn(TypedDict):
    """一轮对话的数据结构。"""

    question: str
    answer: str


class ConversationStore:
    """使用 SQLite 保存数量受限的多轮对话。"""

    def __init__(
        self,
        max_turns: int = 6,
        max_sessions: int = 1000,
        db_path: str | Path = ":memory:",
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1。")
        if max_sessions < 1:
            raise ValueError("max_sessions 必须大于等于 1。")

        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.db_path = str(db_path)
        self._lock = RLock()

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
        self._closed = False
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")

        self._create_schema()

    def _create_schema(self) -> None:
        """首次启动时创建会话表、对话表和查询索引。"""
        with self._lock, self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    last_accessed_at INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES conversation_sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_conversation_turns_session_id
                ON conversation_turns(session_id, id)
                """
            )
  
    def get_history(
        self,
        session_id: str,
    ) -> list[ConversationTurn]:
        """按时间顺序返回指定会话最近的问答历史。"""
        clean_session_id = session_id.strip()
        if not clean_session_id:
            return []

        with self._lock, self._connection:
            session = self._connection.execute(
                """
                SELECT session_id
                FROM conversation_sessions
                WHERE session_id = ?
                """,
                (clean_session_id,),
            ).fetchone()
            if session is None:
                return []

            self._connection.execute(
                """
                UPDATE conversation_sessions
                SET last_accessed_at = ?
                WHERE session_id = ?
                """,
                (time_ns(), clean_session_id),
            )
            rows = self._connection.execute(
                """
                SELECT question, answer
                FROM (
                    SELECT id, question, answer
                    FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (clean_session_id, self.max_turns),
            ).fetchall()

        return [
            {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
            }
            for row in rows
        ]

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """新增一轮问答，并清理超出限制的旧数据。"""
        clean_session_id = session_id.strip()
        clean_question = question.strip()
        clean_answer = answer.strip()
        if not clean_session_id or not clean_question or not clean_answer:
            return

        accessed_at = time_ns()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversation_sessions (
                    session_id,
                    last_accessed_at
                )
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_accessed_at = excluded.last_accessed_at
                """,
                (clean_session_id, accessed_at),
            )
            self._connection.execute(
                """
                INSERT INTO conversation_turns (
                    session_id,
                    question,
                    answer,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    clean_session_id,
                    clean_question,
                    clean_answer,
                    accessed_at,
                ),
            )
            self._connection.execute(
                """
                DELETE FROM conversation_turns
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id
                      FROM conversation_turns
                      WHERE session_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (
                    clean_session_id,
                    clean_session_id,
                    self.max_turns,
                ),
            )
            self._remove_old_sessions()
 
    def _remove_old_sessions(self) -> None:
        """删除超过 max_sessions 限制的最久未访问会话。"""
        session_count = int(
            self._connection.execute(
                "SELECT COUNT(*) FROM conversation_sessions"
            ).fetchone()[0]
        )
        excess_count = session_count - self.max_sessions
        if excess_count <= 0:
            return

        self._connection.execute(
            """
            DELETE FROM conversation_sessions
            WHERE session_id IN (
                SELECT session_id
                FROM conversation_sessions
                ORDER BY last_accessed_at ASC, session_id ASC
                LIMIT ?
            )
            """,
            (excess_count,),
        )

    def clear(self, session_id: str) -> bool:
        """删除指定会话及其全部问答。"""
        clean_session_id = session_id.strip()
        if not clean_session_id:
            return False

        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM conversation_sessions
                WHERE session_id = ?
                """,
                (clean_session_id,),
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        """关闭数据库连接，测试或应用退出时可以主动调用。"""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

def format_conversation_history(
    history: list[ConversationTurn],
) -> str:
    """把结构化历史转换为模型可读的角色文本。"""
    lines: list[str] = []
    for turn in history:
        question = turn.get("question", "").strip()
        answer = turn.get("answer", "").strip()
        if question:
            lines.append(f"用户：{question}")
        if answer:
            lines.append(f"客服：{answer}")
    return "\n".join(lines)
