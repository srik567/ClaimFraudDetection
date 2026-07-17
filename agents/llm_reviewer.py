"""
LLM Reviewer — advisory second-pass for PENDING_REVIEW / FLAGGED claims.

Uses a local Ollama model (default gemma4:latest) to produce a short narrative
summary and a recommended status.  Deterministic matching / forensic scores
remain authoritative; the LLM never overrides EXACT_DUPLICATE flags.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from agents.llm_client import OllamaClient
from schemas.models import Claim, ClaimStatus, LLMReview, ReviewResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an insurance fraud review assistant.
You receive a claim and the rule-based fraud pipeline's findings.
Respond with ONLY valid JSON matching this schema:
{
  "recommendation": "APPROVED" | "PENDING_REVIEW" | "FLAGGED",
  "summary": "2-4 sentences explaining the risk in plain language for a human reviewer",
  "confidence": 0.0 to 1.0
}
Rules:
- Be concise and factual; cite the signals you were given.
- Prefer PENDING_REVIEW when evidence is mixed or borderline.
- Prefer FLAGGED when duplicates or clear tampering are present.
- Prefer APPROVED only when signals look weak or likely false positives.
- Do not invent facts that are not in the input.
"""


def llm_enabled_from_env() -> bool:
    """CLAIM_FRAUD_LLM=0|false|off disables LLM reviews."""
    raw = os.getenv("CLAIM_FRAUD_LLM", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


class LLMReviewer:
    """
    Advisory LLM pass over pipeline ReviewResults.

    Results are cached by invoice_id for the lifetime of the instance so
    re-processing the same claim in one run does not re-hit the model.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.enabled = llm_enabled_from_env() if enabled is None else enabled
        self._cache: Dict[str, LLMReview] = {}
        self._availability_checked = False
        self._available = False

    def review(self, claim: Claim, result: ReviewResult) -> Optional[LLMReview]:
        """
        Produce an LLMReview for a non-APPROVED claim, or None if skipped/failed.

        Skips APPROVED claims, disabled mode, and unavailable Ollama.
        Never changes EXACT_DUPLICATE pipeline decisions in the returned review —
        recommendation is forced to FLAGGED when that flag is present.
        """
        if not self.enabled:
            return None
        if result.status == ClaimStatus.APPROVED:
            return None

        cached = self._cache.get(claim.invoice_id)
        if cached is not None:
            return cached

        if not self._ensure_available():
            logger.warning(
                "Ollama unavailable — skipping LLM review for %s", claim.invoice_id
            )
            return None

        try:
            raw = self.client.chat(
                system=SYSTEM_PROMPT,
                user=self._build_user_prompt(claim, result),
                json_mode=True,
            )
            review = self._parse_review(raw)
        except Exception as exc:  # noqa: BLE001 — advisory path must not break pipeline
            logger.warning(
                "LLM review failed for %s: %s", claim.invoice_id, exc
            )
            return None

        if any("EXACT_DUPLICATE" in f for f in result.flags):
            review = review.model_copy(
                update={"recommendation": ClaimStatus.FLAGGED}
            )

        self._cache[claim.invoice_id] = review
        logger.info(
            "LLM review %s → %s (confidence=%.2f)",
            claim.invoice_id,
            review.recommendation.value,
            review.confidence,
        )
        return review

    def _ensure_available(self) -> bool:
        if not self._availability_checked:
            self._available = self.client.is_available()
            self._availability_checked = True
            if self._available:
                logger.info(
                    "LLM reviewer ready (model=%s)", self.client.model
                )
            else:
                logger.warning(
                    "Ollama not reachable at %s — LLM reviews disabled for this run",
                    self.client.base_url,
                )
        return self._available

    @staticmethod
    def _build_user_prompt(claim: Claim, result: ReviewResult) -> str:
        payload = {
            "claim": {
                "invoice_id": claim.invoice_id,
                "amount": claim.amount,
                "patient_name": claim.patient_name,
                "hospital_id": claim.hospital_id,
                "timestamp": claim.timestamp.isoformat(),
                "ocr_confidence": claim.ocr_confidence,
            },
            "pipeline": {
                "risk_score": result.risk_score,
                "status": result.status.value,
                "flags": result.flags,
                "feature_scores": result.feature_scores,
                "fraud_type": result.fraud_type.value if result.fraud_type else None,
                "reason": result.reason,
            },
        }
        return (
            "Review this insurance claim fraud assessment and return JSON only.\n\n"
            + json.dumps(payload, indent=2)
        )

    @staticmethod
    def _parse_review(raw: str) -> LLMReview:
        data = json.loads(raw)
        recommendation = str(data.get("recommendation", "PENDING_REVIEW")).upper()
        if recommendation not in {s.value for s in ClaimStatus}:
            recommendation = ClaimStatus.PENDING_REVIEW.value
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        summary = str(data.get("summary", "")).strip() or "No summary provided."
        return LLMReview(
            recommendation=ClaimStatus(recommendation),
            summary=summary,
            confidence=confidence,
            model=os.getenv("OLLAMA_MODEL", "gemma4:latest"),
        )
