"""Database package (Phase 7).

SQLite-backed logging for prediction history, application logs, and model
metadata. Param-bound queries throughout (no string interpolation of user
data) to prevent SQL injection (per CLAUDE.md security rules).
"""

from __future__ import annotations

from .store import SQLiteStore

__all__ = ["SQLiteStore"]
