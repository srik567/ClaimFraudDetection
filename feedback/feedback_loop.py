"""
Feedback Loop — Active Learning via SQLite

Stores every AI prediction alongside the human reviewer's final decision.
The retrain_thresholds() function analyses the override history and adjusts
matching thresholds to reduce future false positives, implementing a simple
but effective online learning mechanism.

Database schema (fraud_feedback.db):

    fraud_feedback
    ─────────────
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    invoice_id      TEXT    NOT NULL
    ai_risk_score   REAL    NOT NULL
    ai_decision     TEXT    NOT NULL   (APPROVED | FLAGGED | PENDING_REVIEW)
    human_decision  TEXT               (NULL until reviewed)
    pattern_flags   TEXT               (JSON array of flag strings)
    timestamp       TEXT    NOT NULL   (ISO-8601)
    notes           TEXT               (reviewer free-text)

    threshold_log
    ─────────────
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    pattern         TEXT    NOT NULL   (e.g. FUZZY_DUPLICATE)
    old_threshold   REAL    NOT NULL
    new_threshold   REAL    NOT NULL
    reason          TEXT
    timestamp       TEXT    NOT NULL
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from schemas.models import ClaimStatus, ReviewResult

logger = logging.getLogger(__name__)

# Minimum human-approval rate that triggers threshold adjustment.
APPROVAL_RATE_TRIGGER = 0.70

# Amount by which a threshold is raised per retrain cycle.
THRESHOLD_STEP = 5

# Minimum number of overrides required before retraining fires.
MIN_OVERRIDE_SAMPLE = 3

# Default DB path (in the project root).
DEFAULT_DB_PATH = Path(__file__).parent.parent / "fraud_feedback.db"

# Map pattern keywords → auditor threshold attribute names (for external callers).
PATTERN_TO_THRESHOLD: Dict[str, str] = {
    "FUZZY_DUPLICATE": "fuzzy_threshold",
    "ELA": "ela_threshold",
    "METADATA": "metadata_threshold",
    "CROSS_REFERENCE": "crossref_threshold",
    "EXACT_DUPLICATE": "exact_threshold",
}

# Default thresholds (mirroring agent defaults).
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "FUZZY_DUPLICATE": 85.0,
    "ELA": 12.0,
    "METADATA": 0.0,  # binary flag — raise to effectively disable
    "CROSS_REFERENCE": 0.0,
    "EXACT_DUPLICATE": 0.0,
}


class FeedbackLoop:
    """
    SQLite-backed store for AI predictions and human overrides.

    Usage:
        fb = FeedbackLoop()
        fb.store_ai_prediction(review_result)
        fb.submit_human_override("INV-002", "APPROVED", notes="Legitimate re-sub")
        updated = fb.retrain_thresholds()
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._thresholds: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_ai_prediction(self, result: ReviewResult) -> None:
        """Persist an AI ReviewResult into the feedback database."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fraud_feedback
                    (invoice_id, ai_risk_score, ai_decision, pattern_flags, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.claim_id,
                    result.risk_score,
                    result.status.value,
                    json.dumps(result.flags),
                    datetime.utcnow().isoformat(),
                ),
            )
        logger.debug("Stored AI prediction for %s", result.claim_id)

    def submit_human_override(
        self,
        invoice_id: str,
        human_decision: str,
        notes: str = "",
    ) -> bool:
        """
        Record a human reviewer's final decision for a claim.

        Args:
            invoice_id:     The claim's invoice ID.
            human_decision: One of 'APPROVED', 'FLAGGED', 'PENDING_REVIEW'.
            notes:          Optional free-text reviewer commentary.

        Returns:
            True if a row was updated, False if invoice_id was not found.
        """
        try:
            ClaimStatus(human_decision)
        except ValueError:
            raise ValueError(
                f"Invalid human_decision '{human_decision}'. "
                f"Must be one of: {[s.value for s in ClaimStatus]}"
            )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE fraud_feedback
                SET    human_decision = ?,
                       notes          = ?
                WHERE  invoice_id     = ?
                  AND  human_decision IS NULL
                """,
                (human_decision, notes, invoice_id),
            )
            updated = cursor.rowcount > 0

        if updated:
            logger.info(
                "Human override recorded: %s → %s", invoice_id, human_decision
            )
        else:
            logger.warning(
                "No pending row found for invoice '%s' to override.", invoice_id
            )
        return updated

    def retrain_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """
        Analyse human overrides and auto-adjust matching thresholds.

        Logic:
        - For each pattern (FUZZY_DUPLICATE, ELA, etc.), find all rows where:
            • The AI flagged the claim (FLAGGED or PENDING_REVIEW).
            • The human subsequently approved it (human_decision = 'APPROVED').
        - If ≥ MIN_OVERRIDE_SAMPLE overrides exist AND the human-approval rate
          is ≥ APPROVAL_RATE_TRIGGER, raise the pattern's threshold by THRESHOLD_STEP.

        Returns:
            Dict mapping pattern → (old_threshold, new_threshold) for changed patterns.
        """
        changes: Dict[str, Tuple[float, float]] = {}

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pattern_flags, ai_decision, human_decision
                FROM   fraud_feedback
                WHERE  human_decision IS NOT NULL
                  AND  ai_decision IN ('FLAGGED', 'PENDING_REVIEW')
                """
            ).fetchall()

        # Accumulate per-pattern override statistics
        pattern_stats: Dict[str, Dict[str, int]] = {
            p: {"total": 0, "approved": 0} for p in PATTERN_TO_THRESHOLD
        }

        for flags_json, ai_decision, human_decision in rows:
            flags: List[str] = json.loads(flags_json or "[]")
            for flag in flags:
                for pattern in PATTERN_TO_THRESHOLD:
                    if pattern in flag:
                        pattern_stats[pattern]["total"] += 1
                        if human_decision == ClaimStatus.APPROVED.value:
                            pattern_stats[pattern]["approved"] += 1

        # Apply threshold adjustments
        print("\n  [RETRAIN THRESHOLDS — ACTIVE LEARNING]")
        any_change = False
        for pattern, stats in pattern_stats.items():
            total = stats["total"]
            approved = stats["approved"]

            if total < MIN_OVERRIDE_SAMPLE:
                continue

            approval_rate = approved / total
            print(
                f"    {pattern:<22s}: {approved}/{total} human approvals "
                f"({approval_rate:.0%})",
                end="",
            )

            if approval_rate >= APPROVAL_RATE_TRIGGER:
                old = self._thresholds[pattern]
                new = old + THRESHOLD_STEP
                self._thresholds[pattern] = new
                changes[pattern] = (old, new)
                self._log_threshold_change(conn, pattern, old, new, approval_rate)
                print(f"  → threshold raised {old:.1f} → {new:.1f}")
                any_change = True
            else:
                print("  (no change)")

        if not any_change:
            print("    No patterns met the override threshold for adjustment.")
        print()

        return changes

    def get_current_thresholds(self) -> Dict[str, float]:
        """Return a copy of the current threshold values."""
        return dict(self._thresholds)

    def get_override_history(self) -> List[Dict]:
        """Return all threshold change log entries."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM threshold_log ORDER BY timestamp DESC"
            ).fetchall()
        columns = ["id", "pattern", "old_threshold", "new_threshold", "reason", "timestamp"]
        return [dict(zip(columns, row)) for row in rows]

    def get_all_predictions(self) -> List[Dict]:
        """Return every stored AI prediction (with human decisions where available)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fraud_feedback ORDER BY timestamp DESC"
            ).fetchall()
        columns = [
            "id", "invoice_id", "ai_risk_score", "ai_decision",
            "human_decision", "pattern_flags", "timestamp", "notes",
        ]
        return [dict(zip(columns, row)) for row in rows]

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS fraud_feedback (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id     TEXT    NOT NULL,
                    ai_risk_score  REAL    NOT NULL,
                    ai_decision    TEXT    NOT NULL,
                    human_decision TEXT,
                    pattern_flags  TEXT,
                    timestamp      TEXT    NOT NULL,
                    notes          TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS threshold_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern       TEXT NOT NULL,
                    old_threshold REAL NOT NULL,
                    new_threshold REAL NOT NULL,
                    reason        TEXT,
                    timestamp     TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_fb_invoice
                    ON fraud_feedback (invoice_id);
                CREATE INDEX IF NOT EXISTS idx_fb_decision
                    ON fraud_feedback (ai_decision, human_decision);
                """
            )
        logger.debug("Database initialised at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _log_threshold_change(
        self,
        conn: sqlite3.Connection,
        pattern: str,
        old: float,
        new: float,
        approval_rate: float,
    ) -> None:
        """Write a threshold change record to the log table."""
        conn.execute(
            """
            INSERT INTO threshold_log (pattern, old_threshold, new_threshold, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                pattern,
                old,
                new,
                f"Human approval rate {approval_rate:.0%} >= {APPROVAL_RATE_TRIGGER:.0%}",
                datetime.utcnow().isoformat(),
            ),
        )
