"""
SatQuery AI — MemPalace Persistent Memory Engine
Problem Statement ID: 26167 (ISRO / SAC)

Structured multi-wing episodic & conversational memory database built on PostgreSQL
with seamless SQLite fallback, supporting multi-turn dialogue sessions and precedent retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("satquery.mempalace")


class MemPalaceDB:
    """Manages long-term conversational memory, agent traces, and spatial metadata across MemPalace Wings."""
    _instance: Optional[MemPalaceDB] = None

    def __init__(self, db_path: str = "backend/database/satquery.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_url = os.getenv("DATABASE_URL", "")
        self.use_postgres = False
        self._init_db()

    def _get_pg_connection(self):
        import psycopg2
        return psycopg2.connect(
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "r@jhul@wd3"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "satquery")
        )

    def _init_db(self) -> None:
        # Try PostgreSQL first
        try:
            import psycopg2
            with self._get_pg_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM mempalace_wings")
                self.use_postgres = True
                logger.info("MemPalace connected to PostgreSQL database 'satquery'.")
                return
        except Exception as e:
            logger.info(f"PostgreSQL not reachable ({e}). Using local SQLite database at {self.db_path}.")

        # Fallback: SQLite
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata_json TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    image_id TEXT,
                    metadata_json TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mempalace_wings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wing_name TEXT,
                    record_key TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    tags TEXT,
                    created_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS merkle_audit_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    merkle_root TEXT,
                    computation_mode TEXT,
                    confidence REAL,
                    traces_json TEXT,
                    created_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_images (
                    image_id TEXT PRIMARY KEY,
                    payload_json TEXT,
                    created_at TIMESTAMP
                )
            """)
            conn.commit()

    # Session Management
    def create_or_get_session(self, session_id: str, title: str = "New Satellite Dialogue") -> str:
        now = datetime.now(timezone.utc).isoformat()
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO chat_sessions (session_id, title, created_at, updated_at, metadata_json) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
                            (session_id, title, now, now, json.dumps({}))
                        )
                    conn.commit()
                return session_id
            except Exception as e:
                logger.error(f"PostgreSQL create_session failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM chat_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                cursor.execute(
                    "INSERT INTO chat_sessions (session_id, title, created_at, updated_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (session_id, title, now, now, json.dumps({}))
                )
                conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str, image_id: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> None:
        self.create_or_get_session(session_id)
        now = datetime.now(timezone.utc).isoformat()

        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO chat_messages (session_id, role, content, image_id, metadata_json, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                            (session_id, role, content, image_id, json.dumps(metadata or {}), now)
                        )
                        cursor.execute("UPDATE chat_sessions SET updated_at = %s WHERE session_id = %s", (now, session_id))
                    conn.commit()
                return
            except Exception as e:
                logger.error(f"PostgreSQL add_message failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content, image_id, metadata_json, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, image_id, json.dumps(metadata or {}), now)
            )
            cursor.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))
            conn.commit()

    def get_conversation_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT role, content, image_id, metadata_json, timestamp FROM chat_messages WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                            (session_id, limit)
                        )
                        rows = cursor.fetchall()
                        messages = []
                        for r in reversed(rows):
                            messages.append({
                                "role": r[0],
                                "content": r[1],
                                "image_id": r[2],
                                "metadata": r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {}),
                                "timestamp": r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4])
                            })
                        return messages
            except Exception as e:
                logger.error(f"PostgreSQL get_conversation_history failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, image_id, metadata_json, timestamp FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            messages = []
            for r in reversed(rows):
                messages.append({
                    "role": r[0],
                    "content": r[1],
                    "image_id": r[2],
                    "metadata": json.loads(r[3]) if r[3] else {},
                    "timestamp": r[4]
                })
            return messages

    def save_analysis(self, session_id: str, query: str, result_payload: dict[str, Any]) -> None:
        """Persists full analysis payload into database across server restarts."""
        now = datetime.now(timezone.utc).isoformat()
        title = (query[:60] + "...") if len(query) > 60 else query
        payload_str = json.dumps(result_payload)

        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO chat_sessions (session_id, title, created_at, updated_at, metadata_json)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET updated_at = %s, metadata_json = %s, title = %s
                            """,
                            (session_id, title, now, now, payload_str, now, payload_str, title)
                        )
                        cursor.execute(
                            "INSERT INTO chat_messages (session_id, role, content, metadata_json, timestamp) VALUES (%s, %s, %s, %s, %s)",
                            (session_id, "user", query, json.dumps({}), now)
                        )
                        cursor.execute(
                            "INSERT INTO chat_messages (session_id, role, content, metadata_json, timestamp) VALUES (%s, %s, %s, %s, %s)",
                            (session_id, "assistant", result_payload.get("answer", ""), json.dumps({"confidence": result_payload.get("confidence", 0.0)}), now)
                        )
                        cursor.execute(
                            """
                            INSERT INTO merkle_audit_records (session_id, merkle_root, computation_mode, confidence, traces_json, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (session_id, result_payload.get("merkleRoot", ""), result_payload.get("computationMode", "rule_based"), float(result_payload.get("confidence", 0.0)), json.dumps(result_payload.get("steps", [])), now)
                        )
                    conn.commit()
                return
            except Exception as e:
                logger.error(f"PostgreSQL save_analysis failed: {e}")

        # SQLite fallback
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM chat_sessions WHERE session_id = ?", (session_id,))
            if cursor.fetchone():
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = ?, metadata_json = ?, title = ? WHERE session_id = ?",
                    (now, payload_str, title, session_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO chat_sessions (session_id, title, created_at, updated_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (session_id, title, now, now, payload_str)
                )
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content, metadata_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, "user", query, json.dumps({}), now)
            )
            cursor.execute(
                "INSERT INTO chat_messages (session_id, role, content, metadata_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, "assistant", result_payload.get("answer", ""), json.dumps({"confidence": result_payload.get("confidence", 0.0)}), now)
            )
            cursor.execute(
                "INSERT INTO merkle_audit_records (session_id, merkle_root, computation_mode, confidence, traces_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, result_payload.get("merkleRoot", ""), result_payload.get("computationMode", "rule_based"), float(result_payload.get("confidence", 0.0)), json.dumps(result_payload.get("steps", [])), now)
            )
            conn.commit()

    def get_analysis(self, session_id: str) -> Optional[dict[str, Any]]:
        """Loads a single analysis by session_id from database."""
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT metadata_json FROM chat_sessions WHERE session_id = %s", (session_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
            except Exception as e:
                logger.error(f"PostgreSQL get_analysis failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT metadata_json FROM chat_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except Exception:
                    pass
        return None

    def get_all_analyses(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns all persistent analysis records from database."""
        results = []
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT session_id, title, updated_at, metadata_json FROM chat_sessions ORDER BY updated_at DESC LIMIT %s", (limit,))
                        for row in cursor.fetchall():
                            data = row[3] if isinstance(row[3], dict) else (json.loads(row[3]) if row[3] else {})
                            if data:
                                results.append(data)
                        return results
            except Exception as e:
                logger.error(f"PostgreSQL get_all_analyses failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id, title, updated_at, metadata_json FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,))
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[3]) if row[3] else {}
                    if data:
                        results.append(data)
                except Exception:
                    pass
        return results

    def save_image_metadata(self, image_id: str, payload: dict[str, Any]) -> None:
        """Persists uploaded image metadata into database across server restarts."""
        now = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(payload)
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO uploaded_images (image_id, payload_json, created_at) VALUES (%s, %s, %s) ON CONFLICT (image_id) DO UPDATE SET payload_json = %s",
                            (image_id, payload_str, now, payload_str)
                        )
                    conn.commit()
                return
            except Exception as e:
                logger.error(f"PostgreSQL save_image_metadata failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO uploaded_images (image_id, payload_json, created_at) VALUES (?, ?, ?)",
                (image_id, payload_str, now)
            )
            conn.commit()

    def get_all_uploaded_images(self) -> dict[str, dict[str, Any]]:
        """Returns all persistent uploaded images indexed by image_id."""
        images = {}
        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT image_id, payload_json FROM uploaded_images")
                        for r in cursor.fetchall():
                            images[r[0]] = json.loads(r[1]) if isinstance(r[1], str) else r[1]
                return images
            except Exception as e:
                logger.error(f"PostgreSQL get_all_uploaded_images failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id, payload_json FROM uploaded_images")
            for r in cursor.fetchall():
                try:
                    images[r[0]] = json.loads(r[1])
                except Exception:
                    pass
        return images

    def search_mempalace_wings(self, query: str, wing_name: Optional[str] = None) -> list[dict[str, Any]]:
        """Finds precedents matching the user's disaster or domain context from MemPalace."""
        keywords = [k.lower().strip() for k in query.replace("?", "").split() if len(k) > 3]
        if not keywords:
            keywords = ["landslide", "flood", "cyclone", "drought", "water"]

        if self.use_postgres:
            try:
                with self._get_pg_connection() as conn:
                    with conn.cursor() as cursor:
                        results = []
                        for kw in keywords[:3]:
                            if wing_name:
                                cursor.execute(
                                    "SELECT wing_name, record_key, title, content FROM mempalace_wings WHERE wing_name = %s AND (content ILIKE %s OR %s = ANY(tags))",
                                    (wing_name, f"%{kw}%", kw)
                                )
                            else:
                                cursor.execute(
                                    "SELECT wing_name, record_key, title, content FROM mempalace_wings WHERE content ILIKE %s OR %s = ANY(tags)",
                                    (f"%{kw}%", kw)
                                )
                            for r in cursor.fetchall():
                                item = {"wing": r[0], "key": r[1], "title": r[2], "content": r[3]}
                                if item not in results:
                                    results.append(item)
                        return results[:3]
            except Exception as e:
                logger.error(f"PostgreSQL search_mempalace_wings failed: {e}")

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            results = []
            for kw in keywords[:3]:
                if wing_name:
                    cursor.execute(
                        "SELECT wing_name, record_key, title, content FROM mempalace_wings WHERE wing_name = ? AND (tags LIKE ? OR content LIKE ?)",
                        (wing_name, f"%{kw}%", f"%{kw}%")
                    )
                else:
                    cursor.execute(
                        "SELECT wing_name, record_key, title, content FROM mempalace_wings WHERE tags LIKE ? OR content LIKE ?",
                        (f"%{kw}%", f"%{kw}%")
                    )
                for r in cursor.fetchall():
                    item = {"wing": r[0], "key": r[1], "title": r[2], "content": r[3]}
                    if item not in results:
                        results.append(item)
            return results[:3]


mempalace = MemPalaceDB()
