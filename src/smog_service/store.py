"""Prediction log (SQLite). Every served forecast is recorded with its model version,
so live accuracy can be measured later against what actually happened."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("PREDICTION_DB", "predictions.db"))
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    request_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model_version TEXT NOT NULL,
    city TEXT NOT NULL,
    issue_day TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_date TEXT NOT NULL,
    pm25 REAL NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_target ON predictions (city, target_date);
"""


class PredictionStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, check_same_thread=False)

    def log(self, request_id: str, endpoint: str, response, latency_ms: float) -> None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        rows = [(now, request_id, endpoint, response.model_version, response.city, str(response.issue_day),
                 f.horizon, str(f.date), f.pm25, latency_ms) for f in response.forecasts]
        with self.lock, self._connect() as db:
            db.executemany(
                "INSERT INTO predictions (logged_at, request_id, endpoint, model_version, city, issue_day, "
                "horizon, target_date, pm25, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def stats(self) -> dict:
        with self.lock, self._connect() as db:
            total, = db.execute("SELECT COUNT(DISTINCT request_id) FROM predictions").fetchone()
            latencies = [r[0] for r in db.execute(
                "SELECT latency_ms FROM predictions GROUP BY request_id ORDER BY id DESC LIMIT 1000")]
        latencies.sort()

        def pct(p):
            return round(latencies[min(len(latencies) - 1, int(p * len(latencies)))], 1) if latencies else None

        return {"requests_logged": total, "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95)}
