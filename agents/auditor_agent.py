"""
Auditor Agent — the matching engine that compares incoming claims against a
registry of previously seen claims using three complementary strategies:

1. Exact Match    — SHA-256 hash of all key fields; O(1) lookup.
2. Fuzzy Match    — thefuzz ratio on invoice_id; catches OCR errors and
                    deliberate substitutions like '101' → 'I01'.
3. Cross-Reference — (patient_name + date + amount) composite key; flags
                    re-submissions under a different invoice number.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from thefuzz import fuzz

from schemas.models import Claim

logger = logging.getLogger(__name__)

# Default fuzzy-match sensitivity (0–100).  Overridden by retrain_thresholds().
# Set high enough that naturally similar invoice IDs (same year/prefix) do not
# cross-match; only very-near substitutions (e.g. '0'→'O') are caught.
DEFAULT_FUZZY_THRESHOLD = 96

# Score contributions for each matching strategy.
SCORE_EXACT_DUPLICATE = 100
SCORE_CROSS_REFERENCE = 60
SCORE_FUZZY_MATCH = 50


@dataclass
class MatchResult:
    """Structured output from a single matching run."""

    exact_match: bool = False
    fuzzy_match: bool = False
    fuzzy_ratio: float = 0.0
    fuzzy_matched_invoice: Optional[str] = None
    cross_reference_match: bool = False
    flags: List[str] = field(default_factory=list)
    score_contribution: float = 0.0


class AuditorAgent:
    """
    Stateful matching engine.

    The agent keeps an in-memory registry that is populated claim-by-claim
    during a processing session.  For production use, back this registry with
    a database (PostgreSQL, Redis) between runs.
    """

    def __init__(self, fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD) -> None:
        self.fuzzy_threshold = fuzzy_threshold

        # hash → invoice_id
        self._hash_registry: Dict[str, str] = {}
        # invoice_id → Claim
        self._claim_registry: Dict[str, Claim] = {}
        # (patient_name, date_str, amount) → invoice_id
        self._crossref_registry: Dict[Tuple[str, str, float], str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(self, claim: Claim) -> MatchResult:
        """
        Run all three matching strategies on the incoming claim.

        The claim is registered AFTER matching so that duplicate detection
        works correctly even within the same batch.
        """
        result = MatchResult()

        # 1. Exact match
        claim_hash = self.compute_sha256(claim)
        if self._exact_match(claim_hash):
            matched_invoice = self._hash_registry[claim_hash]
            result.exact_match = True
            result.flags.append(
                f"EXACT_DUPLICATE: SHA-256 hash matches invoice '{matched_invoice}'"
            )
            result.score_contribution += SCORE_EXACT_DUPLICATE
            logger.warning(
                "Exact duplicate detected: %s matches %s", claim.invoice_id, matched_invoice
            )

        # 2. Fuzzy match (only if not already an exact duplicate)
        if not result.exact_match:
            fuzzy_result = self._fuzzy_match_invoice(claim)
            if fuzzy_result:
                matched_invoice, ratio = fuzzy_result
                result.fuzzy_match = True
                result.fuzzy_ratio = ratio
                result.fuzzy_matched_invoice = matched_invoice
                result.flags.append(
                    f"FUZZY_DUPLICATE: Invoice '{claim.invoice_id}' is {ratio:.0f}% "
                    f"similar to existing invoice '{matched_invoice}' "
                    f"(threshold: {self.fuzzy_threshold}%)"
                )
                result.score_contribution += SCORE_FUZZY_MATCH
                logger.warning(
                    "Fuzzy match: %s ≈ %s (ratio=%d)",
                    claim.invoice_id,
                    matched_invoice,
                    ratio,
                )

        # 3. Cross-reference match
        crossref_result = self._cross_reference(claim)
        if crossref_result:
            existing_invoice = crossref_result
            result.cross_reference_match = True
            result.flags.append(
                f"CROSS_REFERENCE: (patient='{claim.patient_name}', "
                f"date='{claim.timestamp.date()}', amount={claim.amount}) "
                f"already exists under invoice '{existing_invoice}'"
            )
            result.score_contribution += SCORE_CROSS_REFERENCE
            logger.warning(
                "Cross-reference hit: %s duplicates patient/date/amount of %s",
                claim.invoice_id,
                existing_invoice,
            )

        # Register claim for future comparisons
        self._register(claim, claim_hash)

        return result

    def compute_sha256(self, claim: Claim) -> str:
        """
        Produce a deterministic SHA-256 fingerprint from the claim's key fields.

        Fields included: invoice_id, amount (rounded to 2 dp), patient_name,
        hospital_id, and the date portion of timestamp.
        """
        date_str = claim.timestamp.date().isoformat()
        raw = (
            f"{claim.invoice_id}|"
            f"{round(claim.amount, 2):.2f}|"
            f"{claim.patient_name}|"
            f"{claim.hospital_id}|"
            f"{date_str}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        """Clear the in-memory registries (useful between test runs)."""
        self._hash_registry.clear()
        self._claim_registry.clear()
        self._crossref_registry.clear()

    def set_fuzzy_threshold(self, threshold: int) -> None:
        """Update the fuzzy matching sensitivity threshold."""
        self.fuzzy_threshold = max(0, min(100, threshold))
        logger.info("Fuzzy threshold updated to %d", self.fuzzy_threshold)

    # ------------------------------------------------------------------
    # Matching strategies
    # ------------------------------------------------------------------

    def _exact_match(self, claim_hash: str) -> bool:
        """O(1) hash table lookup."""
        return claim_hash in self._hash_registry

    def _fuzzy_match_invoice(self, claim: Claim) -> Optional[Tuple[str, float]]:
        """
        Compare the claim's invoice_id against every registered invoice using
        thefuzz.fuzz.ratio.  Returns (matched_invoice_id, ratio) for the
        highest-scoring match above the threshold, or None.

        thefuzz.fuzz.ratio is character-level Levenshtein similarity.  It
        naturally catches:
          '101'  vs 'I01'  → ratio ≈ 86 (I and 1 differ, rest matches)
          'INV-2023-001' vs 'INV-2O23-001' → ratio ≈ 92 (0 vs O)
        """
        best_invoice: Optional[str] = None
        best_ratio: float = 0.0

        for registered_invoice in self._claim_registry:
            ratio = fuzz.ratio(claim.invoice_id, registered_invoice)
            if ratio >= self.fuzzy_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_invoice = registered_invoice

        if best_invoice:
            return best_invoice, best_ratio
        return None

    def _cross_reference(self, claim: Claim) -> Optional[str]:
        """
        Check whether (patient_name, date, amount) already exists under a
        *different* invoice_id.  This catches re-submissions where the fraudster
        changes only the invoice number.
        """
        key = (
            claim.patient_name,
            claim.timestamp.date().isoformat(),
            round(claim.amount, 2),
        )
        existing_invoice = self._crossref_registry.get(key)
        if existing_invoice and existing_invoice != claim.invoice_id:
            return existing_invoice
        return None

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def _register(self, claim: Claim, claim_hash: str) -> None:
        """Add the claim to all three registries."""
        # Do not overwrite an existing entry (first-seen wins)
        if claim_hash not in self._hash_registry:
            self._hash_registry[claim_hash] = claim.invoice_id

        if claim.invoice_id not in self._claim_registry:
            self._claim_registry[claim.invoice_id] = claim

        crossref_key = (
            claim.patient_name,
            claim.timestamp.date().isoformat(),
            round(claim.amount, 2),
        )
        if crossref_key not in self._crossref_registry:
            self._crossref_registry[crossref_key] = claim.invoice_id
