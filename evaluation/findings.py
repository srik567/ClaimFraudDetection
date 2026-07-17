"""
Plain-language findings for claim managers (non-technical audience).
"""

from __future__ import annotations

import re
from typing import List, Optional

from schemas.models import Claim, ClaimStatus, ReviewResult


def build_claim_manager_findings(
    claim: Claim,
    result: ReviewResult,
    *,
    ela_score: Optional[float] = None,
) -> str:
    """
    Build one clear findings paragraph a claim manager can act on.

    Merges rule-based signals and any advisory model summary into normal
    business language — no technical codes, scores, or separate LLM labels.
    """
    if result.status == ClaimStatus.APPROVED:
        return (
            f"This claim for {claim.patient_name.title()} "
            f"(${claim.amount:,.2f}) looks consistent with prior clean activity. "
            "No duplicate, document-edit, or mismatch concerns were found. "
            "It can proceed for normal payment processing."
        )

    sentences: List[str] = []
    flags = result.flags or []

    # Opening decision in plain words
    if result.status == ClaimStatus.FLAGGED:
        sentences.append(
            f"This claim should be held and investigated before payment "
            f"(patient {claim.patient_name.title()}, amount ${claim.amount:,.2f})."
        )
    else:
        sentences.append(
            f"This claim needs a claim manager review before it is approved or denied "
            f"(patient {claim.patient_name.title()}, amount ${claim.amount:,.2f})."
        )

    matched_invoice = _extract_quoted(flags, r"(?:matches invoice|similar to existing invoice)\s+'([^']+)'")
    if any("EXACT_DUPLICATE" in f for f in flags):
        if matched_invoice:
            sentences.append(
                f"It appears to be an exact resubmission of an earlier claim "
                f"(same details as invoice {matched_invoice})."
            )
        else:
            sentences.append(
                "It appears to be an exact resubmission of a claim already on file."
            )

    if any("FUZZY_DUPLICATE" in f for f in flags) and not any(
        "EXACT_DUPLICATE" in f for f in flags
    ):
        if matched_invoice:
            sentences.append(
                f"The invoice / policy number is almost the same as an existing claim "
                f"({matched_invoice}). This often happens when a digit or letter was "
                "changed on purpose or by mistake."
            )
        else:
            sentences.append(
                "The invoice / policy number is nearly identical to one already on file, "
                "which can indicate a slightly altered resubmission."
            )

    if any("CROSS_REFERENCE" in f for f in flags):
        under = _extract_quoted(flags, r"already exists under invoice '([^']+)'")
        if under:
            sentences.append(
                f"The same patient, service date, and amount were already claimed under "
                f"invoice {under}, even though this form uses a different number."
            )
        else:
            sentences.append(
                "The same patient, service date, and amount already appear on another claim "
                "with a different invoice number."
            )

    if any("METADATA" in f for f in flags):
        sentences.append(
            "The document file shows it was created or edited with design software "
            "(such as Photoshop or Canva) rather than standard hospital billing software, "
            "so the PDF may have been altered."
        )

    if any("ELA" in f for f in flags):
        sentences.append(
            "The claim image looks like parts of it may have been digitally changed "
            "(for example, an amount or name pasted over the original). "
            "Please verify against the original hospital bill."
        )

    if claim.ocr_confidence is not None and claim.ocr_confidence < 0.75:
        sentences.append(
            "Some fields were hard to read from the document, so key details should be "
            "double-checked against the original paperwork."
        )

    # Fold model narrative in as normal prose (no "LLM" label).
    if result.llm_review is not None and result.llm_review.summary:
        summary = _plain_summary(result.llm_review.summary)
        if summary and not _is_redundant(summary, sentences):
            sentences.append(summary)

    if result.status == ClaimStatus.PENDING_REVIEW:
        sentences.append(
            "Recommended next step: compare this form with the original hospital documents "
            "and the earlier related claim, then approve, deny, or request more information."
        )
    elif result.status == ClaimStatus.FLAGGED:
        sentences.append(
            "Recommended next step: route to investigation / SIU, confirm whether this is "
            "a true duplicate or altered document, and do not release payment until cleared."
        )

    return " ".join(sentences)


def _extract_quoted(flags: List[str], pattern: str) -> Optional[str]:
    for flag in flags:
        match = re.search(pattern, flag, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _plain_summary(text: str) -> str:
    """Normalize advisory text into a single plain sentence."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    # Drop leading technical phrases if the model echoed them.
    cleaned = re.sub(
        r"(?i)^(the system|the pipeline|the model|llm)\s+(found|flagged|detected)\s+",
        "Review also noted that ",
        cleaned,
    )
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _is_redundant(summary: str, existing: List[str]) -> bool:
    """Skip advisory text that only repeats points already covered."""
    blob = " ".join(existing).lower()
    key_terms = (
        "duplicate",
        "resubmission",
        "photoshop",
        "altered",
        "same patient",
        "invoice",
    )
    hits = sum(1 for term in key_terms if term in summary.lower() and term in blob)
    return hits >= 2 and len(summary) < 180
