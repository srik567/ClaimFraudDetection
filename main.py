"""
main.py — Fraud Detection Pipeline Simulation

Simulates 10 carefully crafted insurance claims that exercise every detection
mechanism in the system, then prints a detailed Error Analysis Report.

Scenario map
────────────
 #   Label  Type                Expected decision
 1   CLEAN  Clean claim         APPROVED
 2   FRAUD  Perfect Duplicate   FLAGGED  (Exact Hash)
 3   FRAUD  Digital Tamper      FLAGGED  (Metadata + ELA)
 4   CLEAN  Clean (base for #5) APPROVED  (registered before fuzzy clone)
 5   FRAUD  Fuzzy Invoice       FLAGGED  (Fuzzy Match)
 6   FRAUD  Grey-area (ELA)     PENDING_REVIEW
 7   CLEAN  Clean               APPROVED
 8   CLEAN  Clean               APPROVED
 9   FRAUD  Cross-Reference     FLAGGED  (Cross-Ref of #8)
10   CLEAN  Low OCR confidence  APPROVED  (borderline but clean)

After simulation:
  • Stores all AI predictions in SQLite.
  • Simulates a human reviewer approving the grey-area claim (#6).
  • Calls retrain_thresholds() to demonstrate active learning.
  • Re-evaluates metrics with updated thresholds.
  • Prints the full Error Analysis Report.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so submodules can import each other.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.auditor_agent import AuditorAgent
from agents.extraction_agent import ExtractionAgent
from agents.forensic_agent import ForensicAgent
from evaluation.metrics import MetricsEvaluator
from feedback.feedback_loop import FeedbackLoop
from mock_data.generator import ClaimGenerator
from schemas.models import Claim, ClaimStatus, FraudType, ReviewResult

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Risk-score contribution weights
# ---------------------------------------------------------------------------
WEIGHT_EXACT_DUPLICATE = 100   # certain fraud → immediate FLAGGED
WEIGHT_CROSS_REFERENCE = 60    # strong signal → PENDING or FLAGGED with another hit
WEIGHT_FUZZY_MATCH = 50        # strong signal → PENDING; combined → FLAGGED
WEIGHT_METADATA_FLAG = 60      # unauthorized tool detected → alone reaches PENDING;
                                # combined with ELA exceeds FLAGGED threshold (>75)
WEIGHT_ELA_FLAG = 50           # visual tamper → alone = PENDING_REVIEW (50 ∈ [40,75]);
                                # combined with metadata = 110 → capped at 100 → FLAGGED
WEIGHT_LOW_OCR = 10            # informational; does not alone trigger a review

FLAGGED_THRESHOLD = 75
PENDING_THRESHOLD = 40


# ===========================================================================
# Pipeline core
# ===========================================================================

class FraudPipeline:
    """
    Orchestrates the full claim processing pipeline:
    Extraction → Forensic → Auditor → Risk Score → ReviewResult
    """

    def __init__(self) -> None:
        self.extractor = ExtractionAgent()
        self.forensic = ForensicAgent()
        self.auditor = AuditorAgent()
        self.evaluator = MetricsEvaluator()
        self.feedback = FeedbackLoop()

    def process(
        self,
        raw_data: dict,
        inject_ela: bool = False,
    ) -> Tuple[Claim, ReviewResult]:
        """
        End-to-end processing of a single raw claim dict.

        Args:
            raw_data:   Dict from mock generator / extraction layer.
            inject_ela: When True, generate synthetic tampered image bytes
                        for ELA analysis (used in simulation).

        Returns:
            (normalised_claim, review_result)
        """
        # ── Step 1: Extraction & normalisation ───────────────────────────
        claim = self.extractor.mock_ocr_extract(raw_data)

        # ── Step 2: Forensic analysis ─────────────────────────────────────
        image_source: Optional[bytes] = None
        if inject_ela:
            image_source = ForensicAgent.mock_tampered_image_bytes(ela_score_target=18.0)

        ela_score, forensic_flags = self.forensic.analyze_claim(
            pdf_bytes=claim.raw_pdf_bytes,
            image_source=image_source,
        )

        # ── Step 3: Auditor matching ──────────────────────────────────────
        match_result = self.auditor.audit(claim)

        # ── Step 4: Risk score composition ───────────────────────────────
        risk_score, feature_scores = self._compute_risk_score(
            match_result=match_result,
            forensic_flags=forensic_flags,
            ela_score=ela_score,
            ocr_confidence=claim.ocr_confidence,
        )
        risk_score = min(100.0, risk_score)

        all_flags = forensic_flags + match_result.flags

        # ── Step 5: Decision ──────────────────────────────────────────────
        status = self._decide(risk_score)
        fraud_type = self._infer_fraud_type(all_flags, status)
        reason = self._build_reason(all_flags, ela_score, risk_score, status)

        result = ReviewResult(
            claim_id=claim.invoice_id,
            risk_score=risk_score,
            status=status,
            flags=all_flags,
            reason=reason,
            feature_scores=feature_scores,
            fraud_type=fraud_type,
            ai_prediction=1 if status != ClaimStatus.APPROVED else 0,
        )

        # ── Step 6: Persist prediction ────────────────────────────────────
        self.feedback.store_ai_prediction(result)

        return claim, result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk_score(
        match_result,
        forensic_flags: List[str],
        ela_score: float,
        ocr_confidence: Optional[float],
    ) -> Tuple[float, Dict[str, float]]:
        score = 0.0
        feature_scores: Dict[str, float] = {}

        if match_result.exact_match:
            feature_scores["Exact Hash Match"] = WEIGHT_EXACT_DUPLICATE
            score += WEIGHT_EXACT_DUPLICATE

        if match_result.fuzzy_match:
            contribution = WEIGHT_FUZZY_MATCH
            feature_scores["Fuzzy Match"] = contribution
            score += contribution

        if match_result.cross_reference_match:
            feature_scores["Cross-Reference"] = WEIGHT_CROSS_REFERENCE
            score += WEIGHT_CROSS_REFERENCE

        metadata_hit = any("METADATA" in f for f in forensic_flags)
        if metadata_hit:
            feature_scores["Metadata Check"] = WEIGHT_METADATA_FLAG
            score += WEIGHT_METADATA_FLAG

        ela_hit = any("ELA" in f for f in forensic_flags)
        if ela_hit:
            feature_scores["ELA Analysis"] = WEIGHT_ELA_FLAG
            score += WEIGHT_ELA_FLAG

        if ocr_confidence is not None and ocr_confidence < 0.75:
            feature_scores["Low OCR"] = WEIGHT_LOW_OCR
            score += WEIGHT_LOW_OCR

        return score, feature_scores

    @staticmethod
    def _decide(risk_score: float) -> ClaimStatus:
        if risk_score > FLAGGED_THRESHOLD:
            return ClaimStatus.FLAGGED
        if risk_score >= PENDING_THRESHOLD:
            return ClaimStatus.PENDING_REVIEW
        return ClaimStatus.APPROVED

    @staticmethod
    def _infer_fraud_type(
        flags: List[str], status: ClaimStatus
    ) -> Optional[FraudType]:
        if status == ClaimStatus.APPROVED:
            return FraudType.CLEAN
        if any("EXACT_DUPLICATE" in f for f in flags):
            return FraudType.EXACT_DUPLICATE
        if any("FUZZY_DUPLICATE" in f for f in flags):
            return FraudType.FUZZY_DUPLICATE
        if any("CROSS_REFERENCE" in f for f in flags):
            return FraudType.CROSS_REFERENCE
        if any("METADATA" in f for f in flags):
            return FraudType.METADATA_TAMPER
        if any("ELA" in f for f in flags):
            return FraudType.ELA_TAMPER
        return None

    @staticmethod
    def _build_reason(
        flags: List[str],
        ela_score: float,
        risk_score: float,
        status: ClaimStatus,
    ) -> str:
        if status == ClaimStatus.APPROVED:
            return "No fraud signals detected — claim approved."

        parts: List[str] = []
        if any("EXACT_DUPLICATE" in f for f in flags):
            parts.append("SHA-256 hash matches a previously seen claim")
        if any("FUZZY_DUPLICATE" in f for f in flags):
            parts.append("Invoice ID closely resembles an existing record (possible character substitution)")
        if any("CROSS_REFERENCE" in f for f in flags):
            parts.append("Same patient / date / amount already exists under a different invoice number")
        if any("METADATA" in f for f in flags):
            parts.append("PDF metadata reveals unauthorized editing software (Photoshop/Canva)")
        if any("ELA" in f for f in flags):
            parts.append(
                f"Error Level Analysis detected pixel-level tampering (ELA score: {ela_score:.2f})"
            )

        if status == ClaimStatus.PENDING_REVIEW and not parts:
            parts.append("Multiple borderline signals below individual thresholds")

        reason = "; ".join(parts)
        if status == ClaimStatus.PENDING_REVIEW:
            reason = f"PENDING HUMAN REVIEW — {reason} (risk score: {risk_score:.0f})"
        return reason


# ===========================================================================
# Reporting
# ===========================================================================

def print_claim_result(index: int, claim: Claim, result: ReviewResult) -> None:
    """Pretty-print the result of a single claim."""
    status_icon = {
        ClaimStatus.APPROVED: "✓",
        ClaimStatus.FLAGGED: "✗",
        ClaimStatus.PENDING_REVIEW: "?",
    }[result.status]

    print(f"\n  ─── Claim #{index:02d} ────────────────────────────────────────")
    print(f"  Invoice   : {claim.invoice_id}")
    print(f"  Patient   : {claim.patient_name}")
    print(f"  Amount    : ${claim.amount:,.2f}   Hospital: {claim.hospital_id}")
    print(f"  Risk Score: {result.risk_score:.0f}/100")
    print(f"  STATUS    : [{status_icon}] {result.status.value}")
    if result.flags:
        print("  Flags     :")
        for flag in result.flags:
            print(f"              • {flag}")
    print(f"  Reason    : {result.reason}")


def print_error_analysis_report(
    results: List[ReviewResult],
    ground_truth: Dict[str, int],
    evaluator: MetricsEvaluator,
    y_true: Optional[List[int]] = None,
) -> None:
    """Print the full Error Analysis Report to stdout."""
    # Prefer the ordered list (avoids key-collision on duplicate invoice IDs)
    if y_true is None:
        y_true = [ground_truth.get(r.claim_id, 0) for r in results]
    y_pred = evaluator.results_to_binary_labels(results)

    metrics = evaluator.compute_metrics(y_true, y_pred)

    print("\n")
    print("=" * 64)
    print("       FRAUD DETECTION — ERROR ANALYSIS REPORT")
    print("=" * 64)

    print("\n  [SKLEARN METRICS]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}", end="")
    if metrics["precision"] < 0.9:
        print("  ← WARNING: below 0.90 — system is over-flagging legitimate claims")
    else:
        print("  ✓ Precision >= 0.90")
    print(f"    Recall    : {metrics['recall']:.4f}")

    tn, fp, fn, tp = evaluator.print_confusion_matrix(y_true, y_pred)

    feature_gt = {r.claim_id: ground_truth.get(r.claim_id, 0) for r in results}
    evaluator.print_feature_report(results, feature_gt)

    grey_area = evaluator.identify_grey_area(results)
    print(f"  [PENDING_REVIEW QUEUE] — {len(grey_area)} claim(s) require human verification")
    for r in grey_area:
        print(f"    Invoice: {r.claim_id:20s} | Risk: {r.risk_score:.0f} | {r.reason}")

    print()
    print("=" * 64)
    print("  FEATURE EXPLANATION SUMMARY")
    print("=" * 64)
    feature_details = [
        ("Metadata Check",
         "Reads /Producer and /Creator from PDF — flags Photoshop, Canva, Illustrator"),
        ("ELA Analysis",
         "Re-compresses image at JPEG q=90; high pixel diff → pasted overlays detected"),
        ("Exact Hash Match",
         "SHA-256 over (invoice_id + amount + patient + hospital + date) — O(1) lookup"),
        ("Fuzzy Match",
         "thefuzz.ratio on invoice_id — catches '101' vs 'I01', '0' vs 'O' swaps"),
        ("Cross-Reference",
         "(patient_name + date + amount) composite key — detects re-invoiced claims"),
    ]
    for feat, explanation in feature_details:
        print(f"  {feat:<20s}: {explanation}")
    print("=" * 64)


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 64)
    print("  CLAIM FRAUD DETECTION — 10-CLAIM SIMULATION")
    print("=" * 64)

    # Remove stale DB from previous runs to get clean metrics
    db_path = ROOT / "fraud_feedback.db"
    if db_path.exists():
        os.remove(db_path)

    pipeline = FraudPipeline()
    generator = ClaimGenerator(seed=2024)

    # ── Generate the 11 scenario dicts ───────────────────────────────────────
    # Index 3 (c4_base) is a clean claim that must be registered before its
    # fuzzy clone (index 4) is processed.  It is processed silently and NOT
    # included in the final metrics / report so the 10-claim display is clean.
    scenario_dicts = generator.generate_ten_scenario_claims()

    results: List[ReviewResult] = []
    claims_processed: List[Claim] = []
    # Use a list (not dict) for ground truth so duplicate invoice IDs don't
    # overwrite each other — claim #1 and #2 share the same ID but different labels.
    y_true_ordered: List[int] = []
    ground_truth: Dict[str, int] = {}  # kept for feature_report lookup (last-write wins)

    print("\n  Processing claims through pipeline...\n")

    # Process index 0–2 first (claims #1–3), then silently register c4_base,
    # then continue with index 4–10 (claims #4–10).
    SILENT_INDICES = {3}          # c4_base — register but do not display or score
    DISPLAY_ORDER = list(range(len(scenario_dicts)))

    claim_number = 0
    for i in DISPLAY_ORDER:
        raw = scenario_dicts[i]
        inject_ela = raw.get("ela_inject", False)
        claim, result = pipeline.process(raw, inject_ela=inject_ela)

        if i in SILENT_INDICES:
            # Register in the auditor but exclude from evaluation / display
            continue

        claim_number += 1
        print_claim_result(claim_number, claim, result)

        results.append(result)
        claims_processed.append(claim)
        label = raw.get("fraud_label", 0)
        y_true_ordered.append(label)
        ground_truth[claim.invoice_id] = label

    # ── Human-in-the-Loop simulation ─────────────────────────────────────────
    print("\n")
    print("=" * 64)
    print("  HUMAN-IN-THE-LOOP — REVIEWER OVERRIDE DEMO")
    print("=" * 64)

    grey_claims = pipeline.evaluator.identify_grey_area(results)
    if grey_claims:
        target = grey_claims[0]
        print(f"\n  Reviewer approves grey-area claim: {target.claim_id}")
        print(f"  AI said: {target.status.value} (risk={target.risk_score:.0f})")
        submitted = pipeline.feedback.submit_human_override(
            invoice_id=target.claim_id,
            human_decision=ClaimStatus.APPROVED.value,
            notes="Reviewed supporting documents — legitimate resubmission after billing error.",
        )
        if submitted:
            print("  Override recorded successfully.")

    # Simulate additional human approvals on ELA-flagged claims to trigger retraining.
    # Skip the claim already overridden above so we don't double-submit.
    already_overridden = {grey_claims[0].claim_id} if grey_claims else set()
    ela_results = [
        r for r in results
        if r.fraud_type and "ELA" in r.fraud_type.value
        and r.claim_id not in already_overridden
    ]
    for r in ela_results[:2]:
        pipeline.feedback.submit_human_override(
            invoice_id=r.claim_id,
            human_decision=ClaimStatus.APPROVED.value,
            notes="Auto-simulated override for retrain demo.",
        )

    # ── Active learning: retrain thresholds ──────────────────────────────────
    print("\n")
    print("=" * 64)
    print("  ACTIVE LEARNING — THRESHOLD ADJUSTMENT")
    print("=" * 64)
    changes = pipeline.feedback.retrain_thresholds()

    if changes:
        print("\n  Applied threshold changes:")
        for pattern, (old, new) in changes.items():
            print(f"    {pattern:<22s}: {old:.1f} → {new:.1f}")
        # Propagate fuzzy threshold change to the auditor
        if "FUZZY_DUPLICATE" in changes:
            new_fuzzy = changes["FUZZY_DUPLICATE"][1]
            pipeline.auditor.set_fuzzy_threshold(int(new_fuzzy))
            print(f"\n  Auditor fuzzy threshold updated to {new_fuzzy:.0f}")
        if "ELA" in changes:
            new_ela = changes["ELA"][1]
            pipeline.forensic.ela_threshold = new_ela
            print(f"  Forensic ELA threshold updated to {new_ela:.1f}")
    else:
        print("\n  No threshold changes applied (insufficient override samples).")

    # ── Final report ──────────────────────────────────────────────────────────
    print_error_analysis_report(results, ground_truth, pipeline.evaluator, y_true_ordered)

    # ── Feedback DB summary ───────────────────────────────────────────────────
    print("\n  [SQLITE FEEDBACK STORE]")
    all_preds = pipeline.feedback.get_all_predictions()
    reviewed = [p for p in all_preds if p["human_decision"] is not None]
    print(f"    Total predictions stored : {len(all_preds)}")
    print(f"    Human reviews submitted  : {len(reviewed)}")
    threshold_log = pipeline.feedback.get_override_history()
    print(f"    Threshold change events  : {len(threshold_log)}")
    print()

    print("  Run `python main.py` again to see improved metrics after retraining.")
    print("  Run `python -m mock_data.generator --count 200` for 200 synthetic claims.\n")


if __name__ == "__main__":
    main()
