"""
Extraction Agent — normalises raw claim data and mocks OCR confidence scoring.

In production this module would wrap a real OCR engine (e.g., Tesseract, AWS
Textract).  Here we simulate it with structured dicts so the rest of the
pipeline can run without physical documents.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from schemas.models import Claim

logger = logging.getLogger(__name__)

# Fields that are OCR-sensitive; low confidence on any of them raises an alert.
OCR_SENSITIVE_FIELDS = {"invoice_id", "amount", "patient_name", "hospital_id"}

# Confidence below this value is treated as unreliable.
LOW_CONFIDENCE_THRESHOLD = 0.75


class ExtractionAgent:
    """Handles document extraction, field normalisation, and OCR confidence checks."""

    def normalize_claim(self, claim: Claim) -> Claim:
        """
        Return a new Claim with all string fields normalised:
        - patient_name  → lowercased, extra whitespace collapsed, accents stripped
        - invoice_id    → uppercase, non-alphanumeric characters removed
        - hospital_id   → uppercase, stripped
        """
        normalized_patient = self._normalize_text(claim.patient_name)
        normalized_invoice = self._normalize_invoice_id(claim.invoice_id)
        normalized_hospital = claim.hospital_id.upper().strip()

        # Pydantic models are immutable by default; use model_copy to update fields
        return claim.model_copy(
            update={
                "patient_name": normalized_patient,
                "invoice_id": normalized_invoice,
                "hospital_id": normalized_hospital,
            }
        )

    def mock_ocr_extract(self, raw_data: Dict[str, Any]) -> Claim:
        """
        Simulate OCR extraction from a raw dictionary payload.

        raw_data keys:
            invoice_id, amount, patient_name, hospital_id, timestamp,
            ocr_confidence (optional float or dict[field→float]),
            image_path (optional), raw_pdf_bytes (optional), fraud_label (optional)

        Returns a Claim with the ocr_confidence field populated.
        If per-field confidences are provided, the minimum is stored.
        """
        per_field_confidence: Optional[Dict[str, float]] = None
        confidence_raw = raw_data.get("ocr_confidence")

        if isinstance(confidence_raw, dict):
            per_field_confidence = confidence_raw
            overall_confidence = min(confidence_raw.values())
        elif isinstance(confidence_raw, (float, int)):
            overall_confidence = float(confidence_raw)
        else:
            overall_confidence = 1.0  # synthetic data — assume perfect confidence

        low_confidence_fields = self._detect_low_confidence_fields(
            per_field_confidence, overall_confidence
        )
        if low_confidence_fields:
            logger.warning(
                "Low OCR confidence on claim %s — fields: %s",
                raw_data.get("invoice_id", "UNKNOWN"),
                low_confidence_fields,
            )

        timestamp = raw_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()

        claim = Claim(
            invoice_id=str(raw_data.get("invoice_id", "")).strip(),
            amount=float(raw_data.get("amount", 0.0)),
            patient_name=str(raw_data.get("patient_name", "")).strip(),
            hospital_id=str(raw_data.get("hospital_id", "")).strip(),
            timestamp=timestamp,
            ocr_confidence=overall_confidence,
            raw_pdf_bytes=raw_data.get("raw_pdf_bytes"),
            image_path=raw_data.get("image_path"),
            fraud_label=raw_data.get("fraud_label"),
        )

        return self.normalize_claim(claim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, strip accents, collapse multiple spaces."""
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_text).strip().lower()

    @staticmethod
    def _normalize_invoice_id(invoice_id: str) -> str:
        """Uppercase and strip non-alphanumeric characters except hyphens."""
        cleaned = re.sub(r"[^A-Za-z0-9\-]", "", invoice_id)
        return cleaned.upper().strip()

    @staticmethod
    def _detect_low_confidence_fields(
        per_field: Optional[Dict[str, float]], overall: float
    ) -> List[str]:
        """Return field names whose OCR confidence falls below the threshold."""
        if per_field:
            return [
                f
                for f, score in per_field.items()
                if f in OCR_SENSITIVE_FIELDS and score < LOW_CONFIDENCE_THRESHOLD
            ]
        if overall < LOW_CONFIDENCE_THRESHOLD:
            return list(OCR_SENSITIVE_FIELDS)
        return []
