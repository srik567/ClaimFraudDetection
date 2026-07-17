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

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so submodules can import each other.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.auditor_agent import AuditorAgent
from agents.document_extractor import DocumentExtractionError, DocumentExtractor
from agents.extraction_agent import ExtractionAgent
from agents.forensic_agent import ForensicAgent
from agents.llm_reviewer import LLMReviewer
from evaluation.efficiency import compute_efficiency, format_efficiency_console
from evaluation.findings import build_claim_manager_findings
from evaluation.html_report import write_analysis_html
from evaluation.metrics import MetricsEvaluator
from feedback.feedback_loop import FeedbackLoop
from mock_data.generator import ClaimGenerator
from schemas.models import Claim, ClaimStatus, FraudType, ProcessingTiming, ReviewResult
from sklearn.metrics import confusion_matrix

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

    def __init__(self, use_llm: Optional[bool] = None) -> None:
        self.extractor = ExtractionAgent()
        self.forensic = ForensicAgent()
        self.auditor = AuditorAgent()
        self.evaluator = MetricsEvaluator()
        self.feedback = FeedbackLoop()
        self.llm_reviewer = LLMReviewer(enabled=use_llm)

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
        t0 = time.perf_counter()
        timing = ProcessingTiming()

        # ── Step 1: Extraction & normalisation ───────────────────────────
        t = time.perf_counter()
        claim = self.extractor.mock_ocr_extract(raw_data)
        timing.extract_ms = (time.perf_counter() - t) * 1000.0

        # ── Step 2: Forensic analysis ─────────────────────────────────────
        t = time.perf_counter()
        image_source: Optional[bytes | str] = None
        if inject_ela:
            image_source = ForensicAgent.mock_tampered_image_bytes(ela_score_target=18.0)
        elif claim.image_path:
            image_source = claim.image_path

        ela_score, forensic_flags = self.forensic.analyze_claim(
            pdf_bytes=claim.raw_pdf_bytes,
            image_source=image_source,
        )
        timing.forensic_ms = (time.perf_counter() - t) * 1000.0

        # ── Step 3: Auditor matching ──────────────────────────────────────
        t = time.perf_counter()
        match_result = self.auditor.audit(claim)
        timing.audit_ms = (time.perf_counter() - t) * 1000.0

        # ── Step 4: Risk score composition ───────────────────────────────
        t = time.perf_counter()
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
        timing.scoring_ms = (time.perf_counter() - t) * 1000.0

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

        # ── Step 6: Advisory narrative (PENDING / FLAGGED only) ───────────
        t = time.perf_counter()
        llm_review = self.llm_reviewer.review(claim, result)
        timing.llm_ms = (time.perf_counter() - t) * 1000.0
        if llm_review is not None:
            result = result.model_copy(update={"llm_review": llm_review})

        # Plain-language findings for claim managers (rules + advisory text).
        manager_findings = build_claim_manager_findings(
            claim, result, ela_score=ela_score
        )
        result = result.model_copy(update={"reason": manager_findings})

        # ── Step 7: Persist prediction ────────────────────────────────────
        t = time.perf_counter()
        self.feedback.store_ai_prediction(result)
        timing.persist_ms = (time.perf_counter() - t) * 1000.0

        timing.total_ms = (time.perf_counter() - t0) * 1000.0
        result = result.model_copy(
            update={"processing_time_ms": timing.total_ms, "timing": timing}
        )

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
        """Placeholder reason; replaced by claim-manager findings after LLM step."""
        if status == ClaimStatus.APPROVED:
            return "No concerns found — ready for normal processing."
        if status == ClaimStatus.PENDING_REVIEW:
            return "Needs claim manager review before payment."
        return "Hold for investigation before payment."


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
    if result.processing_time_ms is not None:
        print(f"  Time      : {result.processing_time_ms:.0f} ms")
        if result.timing is not None:
            tm = result.timing
            print(
                f"              extract={tm.extract_ms:.0f}  forensic={tm.forensic_ms:.0f}  "
                f"audit={tm.audit_ms:.0f}  score={tm.scoring_ms:.0f}  "
                f"llm={tm.llm_ms:.0f}  persist={tm.persist_ms:.0f}"
            )
    print(f"  STATUS    : [{status_icon}] {result.status.value}")
    print(f"  Findings  : {result.reason}")


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
# Document ingestion mode
# ===========================================================================

def _json_safe_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON fields (bytes) before writing extracted claim JSON."""
    return {
        key: value
        for key, value in raw.items()
        if key != "raw_pdf_bytes" and not isinstance(value, (bytes, bytearray))
    }


def run_document_pipeline(
    input_dir: Path,
    json_out: Path,
    *,
    use_llm: Optional[bool] = None,
    html_out: Optional[Path] = None,
) -> None:
    """
    Read claim documents from disk → write extracted JSON → run FraudPipeline.
    """
    print("\n" + "=" * 64)
    print("  CLAIM FRAUD DETECTION — DOCUMENT OCR INGESTION")
    print("=" * 64)
    print(f"  Input dir : {input_dir.resolve()}")
    print(f"  JSON out  : {json_out.resolve()}")

    extractor = DocumentExtractor()
    pipeline = FraudPipeline(use_llm=use_llm)
    json_out.mkdir(parents=True, exist_ok=True)

    batch_t0 = time.perf_counter()
    extracted, failures = extractor.extract_directory(input_dir)
    if failures:
        print(f"\n  Skipped {len(failures)} document(s):")
        for path, err in failures:
            short = err.split("OCR text preview:")[0].strip()
            print(f"    • {path.name}: {short}")

    if not extracted:
        print("\n  No documents successfully extracted (pdf/png/jpg/tiff).")
        return

    print(f"\n  Extracted {len(extracted)} document(s)\n")
    results: List[ReviewResult] = []
    claim_pairs: List[Tuple[Claim, ReviewResult]] = []

    for index, (path, raw) in enumerate(extracted, start=1):
        out_path = json_out / f"{path.stem}.json"
        safe = _json_safe_payload(raw)
        safe.pop("_extract_ms", None)
        out_path.write_text(
            json.dumps(safe, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  JSON      : {out_path.name}")

        ocr_hint_ms = float(raw.pop("_extract_ms", 0.0) or 0.0)

        claim, result = pipeline.process(raw, inject_ela=False)
        if ocr_hint_ms > 0 and result.timing is not None:
            tm = result.timing.model_copy(
                update={
                    "extract_ms": result.timing.extract_ms + ocr_hint_ms,
                    "total_ms": result.timing.total_ms + ocr_hint_ms,
                }
            )
            result = result.model_copy(
                update={
                    "timing": tm,
                    "processing_time_ms": tm.total_ms,
                }
            )

        print_claim_result(index, claim, result)
        results.append(result)
        claim_pairs.append((claim, result))

    batch_wall = time.perf_counter() - batch_t0

    pending = [r for r in results if r.status == ClaimStatus.PENDING_REVIEW]
    flagged = [r for r in results if r.status == ClaimStatus.FLAGGED]
    approved = [r for r in results if r.status == ClaimStatus.APPROVED]
    print("\n" + "=" * 64)
    print("  DOCUMENT RUN SUMMARY")
    print("=" * 64)
    print(f"  Processed : {len(results)}")
    print(f"  APPROVED  : {len(approved)}")
    print(f"  PENDING   : {len(pending)}")
    print(f"  FLAGGED   : {len(flagged)}")
    print(f"  JSON dir  : {json_out.resolve()}")

    efficiency = compute_efficiency(claim_pairs, batch_wall_seconds=batch_wall)
    print(format_efficiency_console(efficiency))

    report_path = html_out or (ROOT / "samples" / "analysis_report.html")
    write_analysis_html(
        report_path,
        title="Claim Fraud Detection — Document Analysis",
        claims=claim_pairs,
        summary={
            "processed": len(results),
            "approved": len(approved),
            "pending": len(pending),
            "flagged": len(flagged),
        },
        failures=[
            (p.name, err.split("OCR text preview:")[0].strip())
            for p, err in failures
        ],
        efficiency=efficiency,
        source_note=f"Input: {input_dir.resolve()}",
    )
    print(f"  HTML report: {report_path.resolve()}")
    print()


def run_simulation(
    *,
    use_llm: Optional[bool] = None,
    html_out: Optional[Path] = None,
) -> None:
    print("\n" + "=" * 64)
    print("  CLAIM FRAUD DETECTION — 10-CLAIM SIMULATION")
    print("=" * 64)

    # Remove stale DB from previous runs to get clean metrics
    db_path = ROOT / "fraud_feedback.db"
    if db_path.exists():
        os.remove(db_path)

    pipeline = FraudPipeline(use_llm=use_llm)
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

    y_pred = pipeline.evaluator.results_to_binary_labels(results)
    metrics = pipeline.evaluator.compute_metrics(y_true_ordered, y_pred)

    cm = confusion_matrix(y_true_ordered, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1]))

    report_path = html_out or (ROOT / "samples" / "analysis_report.html")
    claim_pairs = list(zip(claims_processed, results))
    efficiency = compute_efficiency(claim_pairs)
    print(format_efficiency_console(efficiency))

    write_analysis_html(
        report_path,
        title="Claim Fraud Detection — Simulation Analysis",
        claims=claim_pairs,
        metrics=metrics,
        confusion=(tn, fp, fn, tp),
        efficiency=efficiency,
        source_note="10-claim mock simulation",
    )
    print(f"\n  HTML report: {report_path.resolve()}")

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
    print("  Run `python -m mock_data.generator --count 200` for 200 synthetic claims.")
    print("  Run `python main.py --input-dir samples/claims` for real OCR ingestion.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claim fraud detection — mock simulation or filesystem OCR ingestion",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of claim PDFs/images to OCR and process",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "samples" / "extracted",
        help="Directory to write extracted claim JSON (document mode)",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=ROOT / "samples" / "analysis_report.html",
        help="Path for the HTML analysis report",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Ollama LLM advisory reviews",
    )
    args = parser.parse_args()
    use_llm = False if args.no_llm else None

    if args.input_dir is not None:
        try:
            run_document_pipeline(
                args.input_dir,
                args.json_out,
                use_llm=use_llm,
                html_out=args.html_out,
            )
        except DocumentExtractionError as exc:
            logger.error("Document ingestion failed: %s", exc)
            sys.exit(1)
        return

    run_simulation(use_llm=use_llm, html_out=args.html_out)


if __name__ == "__main__":
    main()