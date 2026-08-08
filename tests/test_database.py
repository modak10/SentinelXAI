"""Tests for the SQLite logging store (Phase 7)."""

from __future__ import annotations

from pathlib import Path

from sentinelxai.database.store import SQLiteStore


def test_log_and_read_history(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    row_id = store.log_prediction(
        predicted_class="Bot",
        confidence=0.96,
        risk="CRITICAL",
        input_features={"Flow Duration": 1.0},
        probabilities={"BENIGN": 0.02, "PortScan": 0.02, "Bot": 0.96},
    )
    assert row_id == 1
    history = store.get_history(limit=10)
    assert len(history) == 1
    rec = history[0]
    assert rec.predicted_class == "Bot"
    assert rec.confidence == 0.96
    assert rec.risk == "CRITICAL"
    assert rec.probabilities["Bot"] == 0.96


def test_metadata_roundtrip(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    store.set_metadata("model_name", "lightgbm")
    assert store.get_metadata("model_name") == "lightgbm"
    assert store.get_metadata("missing") is None


def test_history_limit(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    for i in range(5):
        store.log_prediction(predicted_class="BENIGN", confidence=0.9, risk="LOW")
    assert len(store.get_history(limit=3)) == 3
    assert len(store.get_history(limit=100)) == 5
