"""
Pydantic data schemas for the Claim Fraud Detection system.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ClaimStatus(str, enum.Enum):
    APPROVED = "APPROVED"
    FLAGGED = "FLAGGED"
    PENDING_REVIEW = "PENDING_REVIEW"


class FraudType(str, enum.Enum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    FUZZY_DUPLICATE = "FUZZY_DUPLICATE"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    METADATA_TAMPER = "METADATA_TAMPER"
    ELA_TAMPER = "ELA_TAMPER"
    CLEAN = "CLEAN"


class Claim(BaseModel):
    """Represents a single insurance claim extracted from an incoming document."""

    invoice_id: str = Field(..., description="Unique invoice identifier")
    amount: float = Field(..., gt=0, description="Claimed amount in USD")
    patient_name: str = Field(..., description="Full name of the patient")
    hospital_id: str = Field(..., description="Hospital or clinic identifier")
    timestamp: datetime = Field(..., description="Date/time the claim was submitted")

    # Optional fields populated during extraction
    ocr_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="OCR extraction confidence (0–1)"
    )
    raw_pdf_bytes: Optional[bytes] = Field(
        default=None, description="Raw PDF bytes for forensic analysis"
    )
    image_path: Optional[str] = Field(
        default=None, description="Path to claim image for ELA analysis"
    )
    fraud_label: Optional[int] = Field(
        default=None, description="Ground-truth label: 1=fraud, 0=legitimate"
    )

    @field_validator("patient_name", "invoice_id", "hospital_id", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    class Config:
        json_encoders = {bytes: lambda b: b.hex() if b else None}


class ReviewResult(BaseModel):
    """Result produced by the fraud pipeline for a single claim."""

    claim_id: str = Field(..., description="Invoice ID of the reviewed claim")
    risk_score: float = Field(
        ..., ge=0, le=100, description="Composite risk score (0–100)"
    )
    status: ClaimStatus = Field(..., description="System decision")
    flags: List[str] = Field(default_factory=list, description="Triggered fraud signals")
    reason: str = Field(default="", description="Human-readable explanation")
    feature_scores: Dict[str, float] = Field(
        default_factory=dict, description="Per-feature contribution to risk score"
    )
    fraud_type: Optional[FraudType] = Field(
        default=None, description="Primary fraud category (if flagged)"
    )
    ai_prediction: Optional[int] = Field(
        default=None, description="Binary prediction: 1=fraud, 0=legitimate"
    )


class FraudDecision(BaseModel):
    """Human override decision stored in the SQLite feedback database."""

    invoice_id: str
    ai_risk_score: float
    ai_decision: ClaimStatus
    human_decision: ClaimStatus
    pattern_flags: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: str = Field(default="", description="Reviewer notes")
