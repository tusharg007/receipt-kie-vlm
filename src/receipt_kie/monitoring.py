"""Privacy-conscious local audit storage, label feedback, and feature drift checks."""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def histogram(values: list[float], cuts: list[float]) -> list[int]:
    counts = [0] * (len(cuts) + 1)
    for value in values:
        counts[bisect.bisect_right(cuts, value)] += 1
    return counts


def population_stability(reference: list[int], observed: list[int]) -> float:
    """Smoothed PSI. This is a descriptive drift heuristic, not a hypothesis test."""
    if len(reference) != len(observed) or not reference or sum(reference) == 0:
        raise ValueError("Histogram bins must match and reference counts must be nonempty")
    smoothing = 0.5
    a_total, b_total = (
        sum(reference) + smoothing * len(reference),
        sum(observed) + smoothing * len(observed),
    )
    return sum(
        ((b + smoothing) / b_total - (a + smoothing) / a_total)
        * math.log(((b + smoothing) / b_total) / ((a + smoothing) / a_total))
        for a, b in zip(reference, observed, strict=True)
    )


class AuditStore:
    """One service replica, WAL connections per operation, no receipt text/images."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL, action TEXT NOT NULL,
                    score REAL NOT NULL, schema_valid INTEGER NOT NULL,
                    latency REAL NOT NULL, features TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    request_id TEXT PRIMARY KEY REFERENCES predictions(request_id),
                    created_at TEXT NOT NULL, correct_fields TEXT NOT NULL,
                    complete_correct INTEGER NOT NULL
                );
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        database = sqlite3.connect(self.path, timeout=10)
        try:
            database.execute("PRAGMA foreign_keys=ON")
            with database:
                yield database
        finally:
            database.close()

    def record(self, request_id: str, assessment: dict, features: dict, latency: float) -> None:
        with self.connect() as database:
            database.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    datetime.now(UTC).isoformat(),
                    assessment["quality_model_version"],
                    assessment["action"],
                    assessment["quality_score"],
                    int(assessment["schema_valid"]),
                    latency,
                    json.dumps(features, allow_nan=False),
                ),
            )

    def feedback(self, request_id: str, correct_fields: dict[str, bool]) -> str:
        """Immutable feedback: repeat identical labels is idempotent; conflicts rejected."""
        encoded = json.dumps(correct_fields, sort_keys=True)
        with self.connect() as database:
            if not database.execute(
                "SELECT 1 FROM predictions WHERE request_id=?", (request_id,)
            ).fetchone():
                raise KeyError(request_id)
            existing = database.execute(
                "SELECT correct_fields FROM feedback WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise ValueError("Feedback conflicts with an existing label")
                return "already_recorded"
            cursor = database.execute(
                "INSERT OR IGNORE INTO feedback VALUES (?,?,?,?)",
                (
                    request_id,
                    datetime.now(UTC).isoformat(),
                    encoded,
                    int(all(correct_fields.values())),
                ),
            )
            current = database.execute(
                "SELECT correct_fields FROM feedback WHERE request_id=?",
                (request_id,),
            ).fetchone()[0]
            if current != encoded:
                raise ValueError("Feedback conflicts with an existing label")
        return "recorded" if cursor.rowcount else "already_recorded"

    def summary(
        self,
        reference: dict[str, Any],
        *,
        window: int = 500,
        minimum: int = 30,
        model_version: str | None = None,
    ) -> dict:
        condition = " WHERE model_version=?" if model_version else ""
        parameters = (model_version, window) if model_version else (window,)
        with self.connect() as database:
            rows = database.execute(
                "SELECT request_id, action, schema_valid, latency, features "
                "FROM predictions" + condition + " ORDER BY rowid DESC LIMIT ?",
                parameters,
            ).fetchall()
            ids = [row[0] for row in rows]
            # Join through the current window, rather than mixing old feedback with new traffic.
            feedback = database.execute(
                "SELECT request_id, complete_correct FROM feedback WHERE request_id IN "
                "(SELECT request_id FROM predictions" + condition + " ORDER BY rowid DESC LIMIT ?)",
                parameters,
            ).fetchall()
        count = len(rows)
        features = [json.loads(row[4]) for row in rows]
        drift = {}
        if count >= minimum:
            for field, data in reference.items():
                observed = histogram([row[field] for row in features], data["cuts"])
                drift[field] = population_stability(data["counts"], observed)
        return {
            "window_size": window,
            "model_version": model_version,
            "sample_count": count,
            "schema_valid_rate": sum(row[2] for row in rows) / count if count else None,
            "review_rate": sum(row[1] == "review" for row in rows) / count if count else None,
            "average_latency_seconds": sum(row[3] for row in rows) / count if count else None,
            "feedback_count": len(feedback),
            "feedback_coverage": len(feedback) / len(ids) if ids else 0,
            "labeled_complete_accuracy": sum(row[1] for row in feedback) / len(feedback)
            if feedback
            else None,
            "drift_status": "measured" if count >= minimum else "insufficient_samples",
            "feature_psi": drift,
            "drift_alerts": [key for key, value in drift.items() if value > 0.2],
            "notes": (
                "PSI >0.2 is a heuristic. Voluntary feedback is selection-biased; "
                "it is not population accuracy."
            ),
        }

    def export_feedback(self) -> list[dict]:
        """Export numeric features and human labels for an explicitly reviewed retraining job."""
        with self.connect() as database:
            rows = database.execute(
                "SELECT p.request_id,p.model_version,p.features,"
                "f.correct_fields,f.complete_correct "
                "FROM predictions p JOIN feedback f USING(request_id) ORDER BY p.rowid",
            ).fetchall()
        return [
            {
                "request_id": row[0],
                "model_version": row[1],
                "features": json.loads(row[2]),
                "correct_fields": json.loads(row[3]),
                "complete_correct": bool(row[4]),
            }
            for row in rows
        ]
