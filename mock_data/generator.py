"""
Mock Data Generator — produces synthetic insurance claims at any scale.

Usage (CLI):
    python -m mock_data.generator --count 200 > claims.json
    python -m mock_data.generator --count 200 --seed 42 > claims.json

Usage (Python):
    from mock_data.generator import ClaimGenerator
    gen = ClaimGenerator(seed=42)
    raw_claims = gen.generate_claims(100)

Fraud injection rates (configurable):
    - Perfect duplicates      :  8 %
    - Digital tamper (amount) :  6 %
    - Fuzzy invoice ID        :  5 %
    - Grey-area (borderline)  :  6 %
    - Clean claims            : 75 %
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

# Fraud injection rates (must sum to ≤ 1.0; remainder is clean claims).
FRAUD_RATES: Dict[str, float] = {
    "exact_duplicate": 0.08,
    "digital_tamper": 0.06,
    "fuzzy_invoice": 0.05,
    "grey_area": 0.06,
}

HOSPITAL_IDS = [
    "HOSP-001", "HOSP-002", "HOSP-003", "HOSP-004",
    "HOSP-005", "HOSP-006", "HOSP-007", "HOSP-008",
]

UNAUTHORIZED_SOFTWARE = [
    "Adobe Photoshop 23.0",
    "Canva 2.0",
    "Adobe Illustrator 26.0",
    "GIMP 2.10",
    "Inkscape 1.2",
]


class ClaimGenerator:
    """Generates synthetic insurance claim payloads with injected fraud patterns."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.fake = Faker("en_US")
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_claims(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Generate n claim dictionaries ready for ExtractionAgent.mock_ocr_extract().

        Returns a list where each dict has keys:
            invoice_id, amount, patient_name, hospital_id, timestamp,
            ocr_confidence, fraud_label, fraud_type,
            raw_pdf_bytes (optional), image_path (optional)
        """
        claims: List[Dict[str, Any]] = []
        source_pool: List[Dict[str, Any]] = []  # track clean claims for duplication

        # Determine how many of each type to generate
        counts = self._compute_counts(n)
        order = (
            ["clean"] * counts["clean"]
            + ["exact_duplicate"] * counts["exact_duplicate"]
            + ["digital_tamper"] * counts["digital_tamper"]
            + ["fuzzy_invoice"] * counts["fuzzy_invoice"]
            + ["grey_area"] * counts["grey_area"]
        )
        random.shuffle(order)

        for claim_type in order:
            if claim_type == "clean":
                claim = self._clean_claim()
                source_pool.append(claim)
            elif claim_type == "exact_duplicate" and source_pool:
                claim = self._exact_duplicate(random.choice(source_pool))
            elif claim_type == "digital_tamper" and source_pool:
                claim = self._digital_tamper(random.choice(source_pool))
            elif claim_type == "fuzzy_invoice" and source_pool:
                claim = self._fuzzy_invoice(random.choice(source_pool))
            elif claim_type == "grey_area":
                claim = self._grey_area_claim()
                source_pool.append(claim)
            else:
                claim = self._clean_claim()
                source_pool.append(claim)

            claims.append(claim)

        return claims

    def generate_ten_scenario_claims(self) -> List[Dict[str, Any]]:
        """
        Return exactly 11 dicts (10 displayed + 1 pre-registration base) that
        exercise every detection mechanism.

        Invoice ID scheme: CLAIM-{HOSP}-{PAT_CODE}-{YYYYMM}-{SEQ}
        Naturally different claims have ratio < 70%; the intended fuzzy pair
        shares all but one character, giving ratio ≥ 96.4% (above threshold=96).

        Order (11 items — c4_base is pre-registered before its fuzzy clone):
          #  Display  Fraud type        Expected status
          1    1      CLEAN             APPROVED
          2    2      EXACT_DUPLICATE   FLAGGED  (hash)
          3    3      METADATA_TAMPER   FLAGGED  (metadata + ELA)
          4    —      CLEAN base        APPROVED (registered so fuzzy pair can match)
          5    4      FUZZY_DUPLICATE   FLAGGED  (fuzzy match)
          6    5      ELA_TAMPER        PENDING_REVIEW (grey-area)
          7    6      CLEAN             APPROVED
          8    7      CLEAN             APPROVED
          9    8      CROSS_REFERENCE   FLAGGED  (cross-ref of #7)
         10    9      CLEAN             APPROVED
         11   10      CLEAN (low OCR)   APPROVED (borderline — score < 40)
        """
        base_date = datetime(2024, 3, 15, 9, 0, 0)

        # ── #1: Clean ─────────────────────────────────────────────────────
        c1 = self._make_raw(
            invoice_id="CLAIM-H03-AJOHN-202403-0001",
            amount=1250.00,
            patient_name="Alice Johnson",
            hospital_id="HOSP-003",
            timestamp=base_date,
            ocr_confidence=0.98,
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # ── #2: Perfect duplicate of #1 (same ID + all fields) ───────────
        c2 = self._make_raw(
            invoice_id="CLAIM-H03-AJOHN-202403-0001",
            amount=1250.00,
            patient_name="Alice Johnson",
            hospital_id="HOSP-003",
            timestamp=base_date,
            ocr_confidence=0.97,
            fraud_label=1,
            fraud_type="EXACT_DUPLICATE",
        )

        # ── #3: Digital tamper — Photoshop PDF + ELA image ───────────────
        c3 = self._make_raw(
            invoice_id="CLAIM-H01-BMART-202403-0031",
            amount=9999.00,
            patient_name="Bob Martinez",
            hospital_id="HOSP-001",
            timestamp=base_date + timedelta(hours=2),
            ocr_confidence=0.95,
            fraud_label=1,
            fraud_type="METADATA_TAMPER",
            raw_pdf_bytes=self._tampered_pdf_bytes("Adobe Photoshop 23.0"),
            ela_inject=True,
        )

        # ── Pre-registration base (not displayed as a numbered claim) ─────
        # ID: CLAIM-H05-CWHT-202403-0101
        #     fuzz.ratio vs fuzzy clone (O101 vs 0101) = 2*27/56 ≈ 96.4%  ✓
        c4_base = self._make_raw(
            invoice_id="CLAIM-H05-CWHT-202403-0101",
            amount=3400.00,
            patient_name="Carol White",
            hospital_id="HOSP-005",
            timestamp=base_date + timedelta(hours=3),
            ocr_confidence=0.96,
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # ── #4: Fuzzy invoice — digit '0' substituted with letter 'O' ────
        c4_fuzzy = dict(c4_base)
        c4_fuzzy["invoice_id"] = "CLAIM-H05-CWHT-202403-O101"
        c4_fuzzy["fraud_label"] = 1
        c4_fuzzy["fraud_type"] = "FUZZY_DUPLICATE"
        c4_fuzzy["ocr_confidence"] = 0.93

        # ── #5: Grey-area — ELA signal, fresh unique invoice ─────────────
        c5 = self._make_raw(
            invoice_id="CLAIM-H02-DKIM-202403-0055",
            amount=2100.00,
            patient_name="David Kim",
            hospital_id="HOSP-002",
            timestamp=base_date + timedelta(hours=4),
            ocr_confidence=0.88,
            fraud_label=1,
            fraud_type="ELA_TAMPER",
            ela_inject=True,
        )

        # ── #6: Clean ─────────────────────────────────────────────────────
        c6 = self._make_raw(
            invoice_id="CLAIM-H04-EBRWN-202403-0062",
            amount=780.50,
            patient_name="Eva Brown",
            hospital_id="HOSP-004",
            timestamp=base_date + timedelta(hours=5),
            ocr_confidence=0.99,
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # ── #7: Clean ─────────────────────────────────────────────────────
        c7 = self._make_raw(
            invoice_id="CLAIM-H06-FCHEN-202403-0077",
            amount=4500.00,
            patient_name="Frank Chen",
            hospital_id="HOSP-006",
            timestamp=base_date + timedelta(hours=6),
            ocr_confidence=0.97,
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # ── #8: Cross-reference — Eva Brown re-submits same amount ───────
        # Different seq number → no exact/fuzzy match; cross-ref fires instead
        c8 = self._make_raw(
            invoice_id="CLAIM-H04-EBRWN-202403-9988",
            amount=780.50,
            patient_name="Eva Brown",
            hospital_id="HOSP-004",
            timestamp=base_date + timedelta(hours=5),   # same date as #6
            ocr_confidence=0.96,
            fraud_label=1,
            fraud_type="CROSS_REFERENCE",
        )

        # ── #9: Clean ─────────────────────────────────────────────────────
        c9 = self._make_raw(
            invoice_id="CLAIM-H07-GLEE-202403-0091",
            amount=620.00,
            patient_name="Grace Lee",
            hospital_id="HOSP-007",
            timestamp=base_date + timedelta(hours=8),
            ocr_confidence=0.99,
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # ── #10: Low OCR confidence — borderline but ultimately clean ─────
        c10 = self._make_raw(
            invoice_id="CLAIM-H08-HPARK-202403-0102",
            amount=1890.00,
            patient_name="Henry Park",
            hospital_id="HOSP-008",
            timestamp=base_date + timedelta(hours=9),
            ocr_confidence=0.62,   # below LOW_CONFIDENCE_THRESHOLD (0.75) → +10 pts
            fraud_label=0,
            fraud_type="CLEAN",
        )

        # c4_base is returned before c4_fuzzy so the auditor registers it first
        return [c1, c2, c3, c4_base, c4_fuzzy, c5, c6, c7, c8, c9, c10]

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _clean_claim(self) -> Dict[str, Any]:
        return self._make_raw(
            invoice_id=self._random_invoice_id(),
            amount=round(random.uniform(100, 15000), 2),
            patient_name=self.fake.name(),
            hospital_id=random.choice(HOSPITAL_IDS),
            timestamp=self._random_timestamp(),
            ocr_confidence=round(random.uniform(0.85, 1.00), 3),
            fraud_label=0,
            fraud_type="CLEAN",
        )

    def _exact_duplicate(self, source: Dict[str, Any]) -> Dict[str, Any]:
        dupe = dict(source)
        dupe["fraud_label"] = 1
        dupe["fraud_type"] = "EXACT_DUPLICATE"
        dupe["ocr_confidence"] = round(random.uniform(0.90, 0.99), 3)
        return dupe

    def _digital_tamper(self, source: Dict[str, Any]) -> Dict[str, Any]:
        tampered = dict(source)
        tampered["invoice_id"] = self._random_invoice_id()
        tampered["amount"] = round(source["amount"] * random.uniform(1.5, 5.0), 2)
        tampered["fraud_label"] = 1
        tampered["fraud_type"] = "METADATA_TAMPER"
        tampered["raw_pdf_bytes"] = self._tampered_pdf_bytes(
            random.choice(UNAUTHORIZED_SOFTWARE)
        )
        tampered["ela_inject"] = True
        return tampered

    def _fuzzy_invoice(self, source: Dict[str, Any]) -> Dict[str, Any]:
        fuzzy = dict(source)
        fuzzy["invoice_id"] = self._mangle_invoice_id(source["invoice_id"])
        fuzzy["fraud_label"] = 1
        fuzzy["fraud_type"] = "FUZZY_DUPLICATE"
        return fuzzy

    def _grey_area_claim(self) -> Dict[str, Any]:
        """Claim with an ELA signal but a fresh invoice ID — intentionally borderline."""
        return self._make_raw(
            invoice_id=self._random_invoice_id(),
            amount=round(random.uniform(100, 15000), 2),
            patient_name=self.fake.name(),
            hospital_id=random.choice(HOSPITAL_IDS),
            timestamp=self._random_timestamp(),
            ocr_confidence=round(random.uniform(0.75, 0.88), 3),
            fraud_label=1,
            fraud_type="ELA_TAMPER",
            ela_inject=True,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_raw(
        invoice_id: str,
        amount: float,
        patient_name: str,
        hospital_id: str,
        timestamp: datetime,
        ocr_confidence: float,
        fraud_label: int,
        fraud_type: str,
        raw_pdf_bytes: Optional[bytes] = None,
        ela_inject: bool = False,
    ) -> Dict[str, Any]:
        return {
            "invoice_id": invoice_id,
            "amount": amount,
            "patient_name": patient_name,
            "hospital_id": hospital_id,
            "timestamp": timestamp.isoformat(),
            "ocr_confidence": ocr_confidence,
            "fraud_label": fraud_label,
            "fraud_type": fraud_type,
            "raw_pdf_bytes": raw_pdf_bytes,
            "ela_inject": ela_inject,  # flag consumed by main.py, not a Claim field
        }

    def _random_invoice_id(self) -> str:
        year = random.randint(2022, 2024)
        number = random.randint(1, 9999)
        return f"INV-{year}-{number:04d}"

    def _random_timestamp(self) -> datetime:
        start = datetime(2023, 1, 1)
        return start + timedelta(days=random.randint(0, 730), hours=random.randint(7, 18))

    @staticmethod
    def _mangle_invoice_id(invoice_id: str) -> str:
        """
        Introduce a single-character substitution that resembles the original
        but differs visually — the canonical fuzzy-fraud pattern.
        """
        substitutions = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
        chars = list(invoice_id)
        for i, ch in enumerate(chars):
            if ch in substitutions:
                chars[i] = substitutions[ch]
                return "".join(chars)
        # Fallback: swap two adjacent characters
        if len(chars) >= 2:
            chars[-1], chars[-2] = chars[-2], chars[-1]
        return "".join(chars)

    @staticmethod
    def _tampered_pdf_bytes(software: str) -> bytes:
        pdf_content = (
            f"%PDF-1.4\n"
            f"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            f"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            f"3 0 obj\n"
            f"<< /Producer ({software}) /Creator ({software}) >>\n"
            f"endobj\n"
            f"xref\n0 4\n"
            f"0000000000 65535 f \n"
            f"0000000009 00000 n \n"
            f"0000000058 00000 n \n"
            f"0000000115 00000 n \n"
            f"trailer\n<< /Size 4 /Root 1 0 R /Info 3 0 R >>\n"
            f"startxref\n199\n%%EOF"
        )
        return pdf_content.encode("latin-1")

    @staticmethod
    def _compute_counts(n: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        total_fraud = 0
        for kind, rate in FRAUD_RATES.items():
            c = max(1, round(n * rate))
            counts[kind] = c
            total_fraud += c
        counts["clean"] = max(1, n - total_fraud)
        return counts


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """JSON serialiser for bytes and datetime objects."""
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic insurance claims for fraud detection testing."
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of claims to generate (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    gen = ClaimGenerator(seed=args.seed)
    claims = gen.generate_claims(args.count)
    print(json.dumps(claims, indent=2, default=_serialize))


if __name__ == "__main__":
    main()
