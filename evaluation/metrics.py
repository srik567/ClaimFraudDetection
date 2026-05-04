"""
Evaluation & Metrics Module

Computes standard classification metrics (Accuracy, Precision, Recall) and
renders an ASCII Confusion Matrix in the console using scikit-learn.

Also provides the grey-area identifier that surfaces PENDING_REVIEW claims
with human-readable explanations for the review queue.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from schemas.models import ClaimStatus, ReviewResult

logger = logging.getLogger(__name__)

# Risk score band for "grey area" — claims that need human verification.
GREY_AREA_LOW = 40
GREY_AREA_HIGH = 75

# Feature names used in the Error Analysis Report.
FEATURE_NAMES = [
    "Metadata Check",
    "ELA Analysis",
    "Exact Hash Match",
    "Fuzzy Match",
    "Cross-Reference",
]

# Map feature name to the flag keyword that identifies it.
FEATURE_FLAG_KEYWORDS: Dict[str, str] = {
    "Metadata Check": "METADATA",
    "ELA Analysis": "ELA",
    "Exact Hash Match": "EXACT_DUPLICATE",
    "Fuzzy Match": "FUZZY_DUPLICATE",
    "Cross-Reference": "CROSS_REFERENCE",
}


class MetricsEvaluator:
    """Evaluates a batch of ReviewResults against ground-truth fraud labels."""

    # ------------------------------------------------------------------
    # Core sklearn metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, y_true: List[int], y_pred: List[int]
    ) -> Dict[str, float]:
        """
        Calculate Accuracy, Precision, and Recall.

        Args:
            y_true: Ground-truth labels (1 = fraud, 0 = legitimate).
            y_pred: AI-predicted labels.

        Returns:
            Dict with keys 'accuracy', 'precision', 'recall'.
        """
        if not y_true or not y_pred:
            logger.warning("Empty label lists passed to compute_metrics.")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        }

    def print_confusion_matrix(
        self, y_true: List[int], y_pred: List[int]
    ) -> Tuple[int, int, int, int]:
        """
        Print a formatted ASCII confusion matrix to stdout and return the raw counts.

        Returns:
            (TN, FP, FN, TP)
        """
        if not y_true or not y_pred:
            print("  [No data to display confusion matrix]")
            return 0, 0, 0, 0

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        col_w = 8
        sep = "+" + ("-" * col_w + "+") * 3

        print()
        print("  [CONFUSION MATRIX]")
        print(f"  {'':20s} {'Predicted NEG':>{col_w}}  {'Predicted POS':>{col_w}}")
        print(f"  {sep}")
        print(
            f"  {'Actual NEG':20s} | {tn:>{col_w-2}} | {fp:>{col_w-2}} |"
        )
        print(f"  {sep}")
        print(
            f"  {'Actual POS':20s} | {fn:>{col_w-2}} | {tp:>{col_w-2}} |"
        )
        print(f"  {sep}")
        print()
        print(
            f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}  "
            f"| Precision={tp/(tp+fp) if (tp+fp) else 0:.2f}  "
            f"Recall={tp/(tp+fn) if (tp+fn) else 0:.2f}"
        )
        print()

        return int(tn), int(fp), int(fn), int(tp)

    # ------------------------------------------------------------------
    # Grey-area identification
    # ------------------------------------------------------------------

    def identify_grey_area(
        self, results: List[ReviewResult]
    ) -> List[ReviewResult]:
        """
        Return all ReviewResult objects whose risk_score falls in [40, 75].

        The reason field of each returned result is enriched with a
        human-readable explanation suitable for display in the review queue.
        """
        grey: List[ReviewResult] = []
        for result in results:
            if GREY_AREA_LOW <= result.risk_score <= GREY_AREA_HIGH:
                result = self._enrich_reason(result)
                grey.append(result)
        return grey

    # ------------------------------------------------------------------
    # Feature-level Error Analysis Report
    # ------------------------------------------------------------------

    def feature_report(
        self,
        results: List[ReviewResult],
        ground_truth: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Count True Positives and False Positives per feature across all results.

        Args:
            results:      All ReviewResult objects from a processing run.
            ground_truth: Optional mapping of claim_id → true label (1=fraud).
                          When provided, FP/FN counts are accurate.
                          When absent, every flagged claim is assumed a TP.

        Returns:
            Dict[feature_name, {"TP": n, "FP": n}]
        """
        report: Dict[str, Dict[str, int]] = {
            feat: {"TP": 0, "FP": 0} for feat in FEATURE_NAMES
        }

        for result in results:
            true_label = (ground_truth or {}).get(result.claim_id)
            for feat, keyword in FEATURE_FLAG_KEYWORDS.items():
                if self._flag_triggered(result.flags, keyword):
                    if true_label is None:
                        # No ground truth — conservatively count as TP
                        report[feat]["TP"] += 1
                    elif true_label == 1:
                        report[feat]["TP"] += 1
                    else:
                        report[feat]["FP"] += 1

        return report

    def print_feature_report(
        self,
        results: List[ReviewResult],
        ground_truth: Optional[Dict[str, int]] = None,
    ) -> None:
        """Print the per-feature Error Analysis Report to stdout."""
        report = self.feature_report(results, ground_truth)

        print("  [FEATURE REPORT]")
        for feat, counts in report.items():
            tp = counts["TP"]
            fp = counts["FP"]
            note = ""
            if fp > 0:
                note = f"  ← {fp} false positive(s) — consider raising threshold"
            print(f"    {feat:<20s}: {tp} TP, {fp} FP{note}")
        print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flag_triggered(flags: List[str], keyword: str) -> bool:
        return any(keyword in flag for flag in flags)

    @staticmethod
    def _enrich_reason(result: ReviewResult) -> ReviewResult:
        """Build a human-readable reason string for grey-area claims."""
        parts: List[str] = []

        if any("ELA" in f for f in result.flags):
            parts.append("High visual similarity detected via ELA analysis")
        if any("FUZZY" in f for f in result.flags):
            parts.append("Invoice ID closely resembles an existing record")
        if any("METADATA" in f for f in result.flags):
            parts.append("Document metadata indicates non-standard editing tool")
        if any("CROSS_REFERENCE" in f for f in result.flags):
            parts.append("Patient/date/amount combination previously seen")

        if not parts:
            parts.append("Multiple borderline signals below individual thresholds")

        reason = "; ".join(parts) + f" (risk score: {result.risk_score:.0f})"
        return result.model_copy(update={"reason": reason})

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def results_to_binary_labels(
        results: List[ReviewResult],
    ) -> List[int]:
        """
        Convert ReviewResult status to binary prediction.
        FLAGGED → 1, PENDING_REVIEW → 1 (conservative), APPROVED → 0.
        """
        mapping = {
            ClaimStatus.FLAGGED: 1,
            ClaimStatus.PENDING_REVIEW: 1,
            ClaimStatus.APPROVED: 0,
        }
        return [mapping[r.status] for r in results]
