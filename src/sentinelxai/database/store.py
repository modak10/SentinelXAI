"""SQLite store (Phase 7).

Minimal, dependency-free persistence for prediction history and model
metadata. Tables are created idempotently on :meth:`init`. Every query is
parameterized — no f-string interpolation of caller data — so uploaded
feature values can never inject SQL.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    id: int
    timestamp: str
    predicted_class: str
    confidence: float
    risk: str
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "risk": self.risk,
            "probabilities": self.probabilities,
        }


class SQLiteStore:
    """Thin wrapper around ``sqlite3`` with a fixed, documented schema."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init()

    # --- schema --------------------------------------------------------------

    def init(self) -> None:
        """Create tables if they do not exist. Idempotent."""
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk TEXT NOT NULL,
                input_features_json TEXT,
                probabilities_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # --- writes --------------------------------------------------------------

    def log_prediction(
        self,
        *,
        predicted_class: str,
        confidence: float,
        risk: str,
        input_features: dict | None = None,
        probabilities: dict[str, float] | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Insert one prediction row; returns its autoincrement id."""
        from datetime import datetime, timezone

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions
                (timestamp, predicted_class, confidence, risk, input_features_json, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                predicted_class,
                float(confidence),
                risk,
                json.dumps(input_features or {}, sort_keys=True),
                json.dumps(probabilities or {}, sort_keys=True),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_metadata(self, key: str, value: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO model_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    # --- reads ---------------------------------------------------------------

    def get_history(self, limit: int = 100) -> list[PredictionRecord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, timestamp, predicted_class, confidence, risk, probabilities_json "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall()
        return [
            PredictionRecord(
                id=row["id"],
                timestamp=row["timestamp"],
                predicted_class=row["predicted_class"],
                confidence=row["confidence"],
                risk=row["risk"],
                probabilities=json.loads(row["probabilities_json"] or "{}"),
            )
            for row in rows
        ]

    def get_metadata(self, key: str) -> str | None:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM model_metadata WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def close(self) -> None:
        self._conn.close()
