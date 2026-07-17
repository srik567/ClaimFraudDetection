"""
HTML analysis report writer for fraud pipeline runs.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from schemas.models import Claim, ClaimStatus, ReviewResult


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _status_class(status: ClaimStatus) -> str:
    return {
        ClaimStatus.APPROVED: "status-approved",
        ClaimStatus.FLAGGED: "status-flagged",
        ClaimStatus.PENDING_REVIEW: "status-pending",
    }[status]


def write_analysis_html(
    path: Path,
    *,
    title: str,
    claims: Sequence[Tuple[Claim, ReviewResult]],
    summary: Optional[Dict[str, Any]] = None,
    failures: Optional[Sequence[Tuple[str, str]]] = None,
    metrics: Optional[Dict[str, float]] = None,
    confusion: Optional[Tuple[int, int, int, int]] = None,
    efficiency: Optional[Any] = None,
    source_note: str = "",
) -> Path:
    """
    Write a self-contained HTML analysis report.

    Args:
        path: Output .html file path.
        title: Report heading.
        claims: Ordered (Claim, ReviewResult) pairs.
        summary: Optional counts dict (processed/approved/pending/flagged).
        failures: Optional skipped document (name, error) pairs.
        metrics: Optional accuracy/precision/recall.
        confusion: Optional (tn, fp, fn, tp).
        efficiency: Optional EfficiencyMetrics for executive KPIs.
        source_note: Short subtitle (input dir, mode, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary = {
            "processed": len(claims),
            "approved": sum(1 for _, r in claims if r.status == ClaimStatus.APPROVED),
            "pending": sum(1 for _, r in claims if r.status == ClaimStatus.PENDING_REVIEW),
            "flagged": sum(1 for _, r in claims if r.status == ClaimStatus.FLAGGED),
        }

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [_claim_row(i, claim, result) for i, (claim, result) in enumerate(claims, start=1)]
    subtitle = f"{source_note} · Generated {generated}" if source_note else f"Generated {generated}"

    efficiency_html = _efficiency_section(efficiency)

    metrics_html = ""
    if metrics is not None:
        metrics_html = f"""
        <section class="panel">
          <h2>Model quality metrics</h2>
          <div class="metrics">
            <div><span>Accuracy</span><strong>{metrics.get('accuracy', 0):.4f}</strong></div>
            <div><span>Precision</span><strong>{metrics.get('precision', 0):.4f}</strong></div>
            <div><span>Recall</span><strong>{metrics.get('recall', 0):.4f}</strong></div>
          </div>
        </section>
        """

    confusion_html = ""
    if confusion is not None:
        tn, fp, fn, tp = confusion
        confusion_html = f"""
        <section class="panel">
          <h2>Confusion matrix</h2>
          <table class="matrix">
            <thead>
              <tr><th></th><th>Pred NEG</th><th>Pred POS</th></tr>
            </thead>
            <tbody>
              <tr><th>Actual NEG</th><td>{tn}</td><td>{fp}</td></tr>
              <tr><th>Actual POS</th><td>{fn}</td><td>{tp}</td></tr>
            </tbody>
          </table>
        </section>
        """

    failures_html = ""
    if failures:
        items = "".join(
            f"<li><code>{_esc(name)}</code> — {_esc(err)}</li>" for name, err in failures
        )
        failures_html = f"""
        <section class="panel warn">
          <h2>Skipped documents ({len(failures)})</h2>
          <ul>{items}</ul>
        </section>
        """

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --ink: #1c1915;
      --muted: #5c564c;
      --line: #d7d0c3;
      --card: #fffdf8;
      --approved: #1f6b4a;
      --approved-bg: #e5f4ec;
      --flagged: #9b1c1c;
      --flagged-bg: #fde8e8;
      --pending: #8a5a00;
      --pending-bg: #fff3d6;
      --accent: #0f4c5c;
      --hero: #143d4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #efe7d6 0%, transparent 40%),
        linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
      line-height: 1.45;
    }}
    header {{
      padding: 2.25rem 1.5rem 1.25rem;
      border-bottom: 1px solid var(--line);
      background: rgba(255,253,248,0.85);
    }}
    header h1 {{
      margin: 0 0 0.35rem;
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(1.6rem, 3vw, 2.2rem);
      letter-spacing: -0.02em;
    }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 1.5rem;
      display: grid;
      gap: 1.25rem;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.75rem;
    }}
    .summary-wide {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
    }}
    @media (max-width: 800px) {{
      .summary, .summary-wide {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 1rem;
    }}
    .stat.highlight {{
      border-color: var(--accent);
      background: #eef6f8;
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 0.85rem; }}
    .stat strong {{ font-size: 1.6rem; }}
    .stat em {{
      display: block;
      margin-top: 0.35rem;
      font-style: normal;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 1.1rem 1.2rem;
    }}
    .panel.warn {{ border-color: #e0b56a; background: #fff8ea; }}
    .panel.exec {{ border-color: var(--accent); }}
    .panel h2 {{
      margin: 0 0 0.85rem;
      font-size: 1.05rem;
      color: var(--accent);
    }}
    .panel p.note {{
      margin: 0.75rem 0 0;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
    }}
    .metrics div {{
      border: 1px solid var(--line);
      padding: 0.75rem;
    }}
    .metrics span {{ display: block; color: var(--muted); font-size: 0.8rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 0.7rem 0.55rem;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .matrix th, .matrix td {{ text-align: center; }}
    .badge {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .status-approved {{ color: var(--approved); background: var(--approved-bg); }}
    .status-flagged {{ color: var(--flagged); background: var(--flagged-bg); }}
    .status-pending {{ color: var(--pending); background: var(--pending-bg); }}
    .flags {{ margin: 0; padding-left: 1.1rem; color: var(--muted); }}
    .reason {{
      color: var(--ink);
      max-width: 36rem;
      line-height: 1.5;
    }}
    .llm {{ display: none; }}
    .flags {{ display: none; }}
    .timing {{ display: none; }}
    footer {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 0 1.5rem 2rem;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.86em; }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <p>{_esc(subtitle)}</p>
  </header>
  <main>
    <section class="summary">
      <div class="stat"><span>Processed</span><strong>{summary.get('processed', 0)}</strong></div>
      <div class="stat"><span>Approved</span><strong>{summary.get('approved', 0)}</strong></div>
      <div class="stat"><span>Pending</span><strong>{summary.get('pending', 0)}</strong></div>
      <div class="stat"><span>Flagged</span><strong>{summary.get('flagged', 0)}</strong></div>
    </section>
    {efficiency_html}
    {failures_html}
    {metrics_html}
    {confusion_html}
    <section class="panel">
      <h2>Claim analysis</h2>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Invoice</th>
            <th>Patient</th>
            <th>Amount</th>
            <th>Risk</th>
            <th>Time</th>
            <th>Status</th>
            <th>Findings</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </main>
  <footer>ClaimFraudDetection · automation reduces full manual review to exceptions only</footer>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def _efficiency_section(efficiency: Any) -> str:
    if efficiency is None:
        return ""
    eff = efficiency
    return f"""
    <section class="panel exec">
      <h2>Operational efficiency — executive view</h2>
      <div class="summary-wide">
        <div class="stat highlight">
          <span>Hours saved vs full manual</span>
          <strong>{eff.hours_saved:.2f} hrs</strong>
          <em>{eff.time_reduction_pct:.1f}% less human review effort</em>
        </div>
        <div class="stat highlight">
          <span>Straight-through processing</span>
          <strong>{eff.stp_rate*100:.1f}%</strong>
          <em>Claims auto-approved with no human touch</em>
        </div>
        <div class="stat highlight">
          <span>Auto-decision rate</span>
          <strong>{eff.auto_decision_rate*100:.1f}%</strong>
          <em>Approved + flagged without full review queue</em>
        </div>
        <div class="stat">
          <span>Machine throughput</span>
          <strong>{eff.throughput_claims_per_hour:.0f}/hr</strong>
          <em>Avg {eff.avg_processing_seconds*1000:.0f} ms/claim
          (p95 {eff.p95_processing_seconds*1000:.0f} ms)</em>
        </div>
        <div class="stat">
          <span>Human-touch rate</span>
          <strong>{eff.human_touch_rate*100:.1f}%</strong>
          <em>Only pending claims need analyst review</em>
        </div>
        <div class="stat">
          <span>Amount flagged for SIU</span>
          <strong>${eff.flagged_amount_total:,.0f}</strong>
          <em>Approved ${eff.approved_amount_total:,.0f} · Queue ${eff.pending_amount_total:,.0f}</em>
        </div>
      </div>
      <p class="note">
        Baseline assumes {eff.manual_review_minutes:.0f} min manual review per claim
        ({eff.baseline_manual_hours:.2f} hrs for this batch).
        With automation: pending @ {eff.pending_review_minutes:.0f} min +
        flagged triage @ {eff.flagged_triage_minutes:.0f} min
        = {eff.automated_human_hours:.2f} hrs.
        Total machine time for this run: {eff.total_processing_seconds:.2f}s.
      </p>
    </section>
    """


def _claim_row(index: int, claim: Claim, result: ReviewResult) -> str:
    findings = _esc(result.reason or "No findings recorded.")

    time_cell = "—"
    if result.processing_time_ms is not None:
        seconds = result.processing_time_ms / 1000.0
        if seconds < 1:
            time_cell = f"{result.processing_time_ms:.0f} ms"
        else:
            time_cell = f"{seconds:.1f} sec"

    status_label = {
        ClaimStatus.APPROVED: "Approved",
        ClaimStatus.FLAGGED: "Hold — investigate",
        ClaimStatus.PENDING_REVIEW: "Needs review",
    }[result.status]

    return f"""
    <tr>
      <td>{index:02d}</td>
      <td><code>{_esc(claim.invoice_id)}</code></td>
      <td>{_esc(claim.patient_name.title())}</td>
      <td>${claim.amount:,.2f}<br/><span style="color:var(--muted)">{_esc(claim.hospital_id)}</span></td>
      <td>{result.risk_score:.0f}/100</td>
      <td>{time_cell}</td>
      <td><span class="badge {_status_class(result.status)}">{_esc(status_label)}</span></td>
      <td><div class="reason">{findings}</div></td>
    </tr>
    """
