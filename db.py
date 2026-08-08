"""SQLite persistence for analyzed shots (Phase 5) and match-context
tags (Phase 6).

A short-lived connection is opened per call rather than sharing one
handle across requests - sqlite3 connections aren't safe to share
across threads, and this matches the project's existing pattern of
not sharing stateful objects across requests (e.g. video_analysis.py's
fresh pose landmarker per call).
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

_DB_PATH = os.path.join("instance", "sessions.db")


def init_db(db_path: Optional[str] = None) -> None:
    """Create the sessions table if it doesn't exist. Call once at app startup."""
    global _DB_PATH
    if db_path:
        _DB_PATH = db_path
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)

    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                player_name TEXT NOT NULL DEFAULT '',
                shot_key TEXT NOT NULL,
                angle TEXT,
                is_video INTEGER NOT NULL DEFAULT 0,
                image_url TEXT,
                video_url TEXT,
                forward_pct INTEGER,
                back_pct INTEGER,
                biomechanics_score INTEGER,
                metrics_json TEXT,
                over_number INTEGER,
                bowler_type TEXT,
                line_length TEXT,
                outcome_runs INTEGER,
                outcome_dismissal INTEGER
            )
        """)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_session(player_name: str, shot_key: str, angle: str, is_video: bool,
                  image_url: Optional[str], video_url: Optional[str],
                  forward_pct: int, back_pct: int,
                  biomechanics_score: Optional[int], metrics: Optional[dict]) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions
                (created_at, player_name, shot_key, angle, is_video, image_url, video_url,
                 forward_pct, back_pct, biomechanics_score, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                player_name or "",
                shot_key,
                angle,
                1 if is_video else 0,
                image_url,
                video_url,
                forward_pct,
                back_pct,
                biomechanics_score,
                json.dumps(metrics or {}),
            ),
        )
        return cur.lastrowid


def update_shot_key(session_id: int, new_shot_key: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET shot_key = ? WHERE id = ?", (new_shot_key, session_id))


def save_match_context(session_id: int, over_number: Optional[int], bowler_type: Optional[str],
                        line_length: Optional[str], outcome_runs: Optional[int],
                        outcome_dismissal: bool) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET over_number = ?, bowler_type = ?, line_length = ?, outcome_runs = ?, outcome_dismissal = ?
            WHERE id = ?
            """,
            (over_number, bowler_type, line_length, outcome_runs, 1 if outcome_dismissal else 0, session_id),
        )


def get_recent_sessions(player_name: str, limit: int = 20) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE player_name = ? ORDER BY created_at DESC LIMIT ?",
            (player_name or "", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_shot_counts(player_name: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT shot_key, COUNT(*) AS count FROM sessions
            WHERE player_name = ? GROUP BY shot_key ORDER BY count DESC
            """,
            (player_name or "",),
        ).fetchall()
        return [dict(r) for r in rows]


def get_trend(player_name: str, limit: int = 20) -> list:
    """Biomechanics score over time, chronological (oldest first) - most recent `limit` sessions."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT created_at, biomechanics_score FROM sessions
            WHERE player_name = ? AND biomechanics_score IS NOT NULL
            ORDER BY created_at DESC LIMIT ?
            """,
            (player_name or "", limit),
        ).fetchall()
        return [{"created_at": r["created_at"], "score": r["biomechanics_score"]} for r in reversed(rows)]


def get_strongest_weakest(player_name: str) -> dict:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT shot_key, AVG(biomechanics_score) AS avg_score, COUNT(*) AS n
            FROM sessions
            WHERE player_name = ? AND biomechanics_score IS NOT NULL
            GROUP BY shot_key
            """,
            (player_name or "",),
        ).fetchall()
    if not rows:
        return {"strongest": None, "weakest": None}
    ranked = sorted(rows, key=lambda r: r["avg_score"], reverse=True)
    strongest = ranked[0]
    weakest = ranked[-1]
    return {
        "strongest": {"shot_key": strongest["shot_key"], "avg_score": round(strongest["avg_score"]), "n": strongest["n"]},
        "weakest": {"shot_key": weakest["shot_key"], "avg_score": round(weakest["avg_score"]), "n": weakest["n"]},
    }


def get_tagged_sessions(player_name: str, limit: int = 50) -> list:
    """Sessions with match-context outcomes recorded (for the Phase 6 shot map)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE player_name = ? AND (outcome_runs IS NOT NULL OR outcome_dismissal = 1)
            ORDER BY created_at DESC LIMIT ?
            """,
            (player_name or "", limit),
        ).fetchall()
        return [dict(r) for r in rows]
