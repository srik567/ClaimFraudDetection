"""
Operational efficiency metrics for executive reporting.

Compares automated triage against a fully-manual review baseline so leaders
can see time saved, STP rate, and throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from schemas.models import Claim, ClaimStatus, ReviewResult

# Conservative industry-style assumptions (minutes). Override via compute_efficiency().
DEFAULT_MANUAL_REVIEW_MINUTES = 12.0
DEFAULT_FLAGGED_TRIAGE_MINUTES = 5.0  # SIU / specialist brief review
DEFAULT_PENDING_REVIEW_MINUTES = 12.0  # full human verification


@dataclass
class EfficiencyMetrics:
    """Batch-level operational KPIs."""

    claims_processed: int
    approved: int
    pending: int
    flagged: int

    total_processing_seconds: float
    avg_processing_seconds: float
    p50_processing_seconds: float
    p95_processing_seconds: float
    throughput_claims_per_hour: float

    auto_decision_rate: float  # (approved + flagged) / total
    stp_rate: float  # approved / total — straight-through processing
    human_touch_rate: float  # pending / total

    baseline_manual_hours: float
    automated_human_hours: float
    hours_saved: float
    time_reduction_pct: float

    flagged_amount_total: float
    approved_amount_total: float
    pending_amount_total: float

    manual_review_minutes: float
    flagged_triage_minutes: float
    pending_review_minutes: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "claims_processed": float(self.claims_processed),
            "approved": float(self.approved),
            "pending": float(self.pending),
            "flagged": float(self.flagged),
            "total_processing_seconds": self.total_processing_seconds,
            "avg_processing_seconds": self.avg_processing_seconds,
            "p50_processing_seconds": self.p50_processing_seconds,
            "p95_processing_seconds": self.p95_processing_seconds,
            "throughput_claims_per_hour": self.throughput_claims_per_hour,
            "auto_decision_rate": self.auto_decision_rate,
            "stp_rate": self.stp_rate,
            "human_touch_rate": self.human_touch_rate,
            "baseline_manual_hours": self.baseline_manual_hours,
            "automated_human_hours": self.automated_human_hours,
            "hours_saved": self.hours_saved,
            "time_reduction_pct": self.time_reduction_pct,
            "flagged_amount_total": self.flagged_amount_total,
            "approved_amount_total": self.approved_amount_total,
            "pending_amount_total": self.pending_amount_total,
        }


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def compute_efficiency(
    claims: Sequence[Tuple[Claim, ReviewResult]],
    *,
    manual_review_minutes: float = DEFAULT_MANUAL_REVIEW_MINUTES,
    flagged_triage_minutes: float = DEFAULT_FLAGGED_TRIAGE_MINUTES,
    pending_review_minutes: float = DEFAULT_PENDING_REVIEW_MINUTES,
    batch_wall_seconds: Optional[float] = None,
) -> EfficiencyMetrics:
    """
    Derive executive efficiency KPIs from a processed claim batch.

    Baseline assumes every claim needs a full manual review.
    Automated path: APPROVED need no human time; FLAGGED get brief triage;
    PENDING_REVIEW get full human review.

    Throughput prefers batch_wall_seconds (true end-to-end wall clock) when
    provided; otherwise falls back to sum of per-claim processing times.
    """
    n = len(claims)
    approved = sum(1 for _, r in claims if r.status == ClaimStatus.APPROVED)
    pending = sum(1 for _, r in claims if r.status == ClaimStatus.PENDING_REVIEW)
    flagged = sum(1 for _, r in claims if r.status == ClaimStatus.FLAGGED)

    durations = [
        float(r.processing_time_ms or 0.0) / 1000.0
        for _, r in claims
    ]
    total_sec = sum(durations)
    avg_sec = (total_sec / n) if n else 0.0
    ordered = sorted(durations)
    p50 = _percentile(ordered, 50)
    p95 = _percentile(ordered, 95)

    wall = batch_wall_seconds if batch_wall_seconds and batch_wall_seconds > 0 else total_sec
    throughput = (n / wall * 3600.0) if wall > 0 else 0.0

    auto_rate = ((approved + flagged) / n) if n else 0.0
    stp = (approved / n) if n else 0.0
    human_touch = (pending / n) if n else 0.0

    baseline_hours = (n * manual_review_minutes) / 60.0
    automated_hours = (
        pending * pending_review_minutes + flagged * flagged_triage_minutes
    ) / 60.0
    hours_saved = max(0.0, baseline_hours - automated_hours)
    reduction = (hours_saved / baseline_hours * 100.0) if baseline_hours > 0 else 0.0

    flagged_amt = sum(c.amount for c, r in claims if r.status == ClaimStatus.FLAGGED)
    approved_amt = sum(c.amount for c, r in claims if r.status == ClaimStatus.APPROVED)
    pending_amt = sum(
        c.amount for c, r in claims if r.status == ClaimStatus.PENDING_REVIEW
    )

    return EfficiencyMetrics(
        claims_processed=n,
        approved=approved,
        pending=pending,
        flagged=flagged,
        total_processing_seconds=total_sec,
        avg_processing_seconds=avg_sec,
        p50_processing_seconds=p50,
        p95_processing_seconds=p95,
        throughput_claims_per_hour=throughput,
        auto_decision_rate=auto_rate,
        stp_rate=stp,
        human_touch_rate=human_touch,
        baseline_manual_hours=baseline_hours,
        automated_human_hours=automated_hours,
        hours_saved=hours_saved,
        time_reduction_pct=reduction,
        flagged_amount_total=flagged_amt,
        approved_amount_total=approved_amt,
        pending_amount_total=pending_amt,
        manual_review_minutes=manual_review_minutes,
        flagged_triage_minutes=flagged_triage_minutes,
        pending_review_minutes=pending_review_minutes,
    )


def format_efficiency_console(eff: EfficiencyMetrics) -> str:
    """Pretty-print efficiency block for stdout."""
    lines = [
        "",
        "=" * 64,
        "  OPERATIONAL EFFICIENCY — EXECUTIVE SUMMARY",
        "=" * 64,
        f"  Claims processed          : {eff.claims_processed}",
        f"  Total machine time        : {eff.total_processing_seconds:.2f}s",
        f"  Avg time / claim          : {eff.avg_processing_seconds*1000:.0f} ms"
        f"  (p50={eff.p50_processing_seconds*1000:.0f} ms,"
        f" p95={eff.p95_processing_seconds*1000:.0f} ms)",
        f"  Throughput                : {eff.throughput_claims_per_hour:.0f} claims/hour",
        "",
        f"  Straight-through (STP)    : {eff.stp_rate*100:.1f}%  (auto-approved)",
        f"  Auto-decision rate        : {eff.auto_decision_rate*100:.1f}%"
        f"  (approved + flagged, no full review)",
        f"  Human-touch rate          : {eff.human_touch_rate*100:.1f}%"
        f"  (pending review queue)",
        "",
        f"  Baseline manual effort    : {eff.baseline_manual_hours:.2f} hrs"
        f"  ({eff.manual_review_minutes:.0f} min × all claims)",
        f"  Effort with automation    : {eff.automated_human_hours:.2f} hrs"
        f"  (pending {eff.pending_review_minutes:.0f} min +"
        f" flagged triage {eff.flagged_triage_minutes:.0f} min)",
        f"  Estimated hours saved     : {eff.hours_saved:.2f} hrs"
        f"  ({eff.time_reduction_pct:.1f}% reduction)",
        "",
        f"  Amount auto-approved      : ${eff.approved_amount_total:,.2f}",
        f"  Amount in review queue    : ${eff.pending_amount_total:,.2f}",
        f"  Amount flagged (SIU)      : ${eff.flagged_amount_total:,.2f}",
        "=" * 64,
    ]
    return "\n".join(lines)
