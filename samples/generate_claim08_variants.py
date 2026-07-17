"""
Generate claim_09+ PDF variants derived from claim_08_original.pdf.

Scenarios
─────────
 claim_09_exact_dup.pdf       Exact byte copy of claim_08 → EXACT_DUPLICATE
 claim_10_fuzzy_base.pdf      Same PQR/11000 identity, long Policy No (fuzzy base)
 claim_11_fuzzy.pdf           1-char Policy edit vs claim_10 → FUZZY (≥96)
 claim_12_amount_tamper.pdf   Same Policy 12345678, amount 99000 + Photoshop meta
 claim_13_low_ocr.pdf         Noisy image PDF derived from claim_08 fields
 claim_14_crossref.pdf        Same patient+amount+date, different Policy No

Usage:
    python -m samples.generate_claim08_variants
    python -m samples.generate_claim08_variants --base samples/claims/claim_08_original.pdf
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_CLAIMS = ROOT / "claims"
DEFAULT_BASE = DEFAULT_CLAIMS / "claim_08_original.pdf"

# Fields extracted / derived from claim_08_original.pdf (Paramount TPA sample).
BASE_FIELDS: Dict[str, str] = {
    "policy_no": "12345678",
    "patient_name": "PQR",
    "phs_id": "LMN1234",
    "insurer": "ABC Insurance Company",
    "amount": "11000",
    "date": "2024-08-22",
    "hospital_name": "City Hospital",
}

# Longer policy form so a 1-char edit still reaches fuzzy threshold (≥96).
FUZZY_BASE_POLICY = "CLAIM-PHS-12345678-AUG2024"
FUZZY_TWIN_POLICY = "CLAIM-PHS-1234567B-AUG2024"


def _font(size: int = 22) -> ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(name)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _paramount_lines(fields: Dict[str, str], *, scenario: str = "") -> List[str]:
    """Build labeled lines that DocumentExtractor can parse."""
    header = [
        "PARAMOUNT HEALTH SERVICES & INSURANCE TPA PRIVATE LIMITED",
        "CLAIM ACKNOWLEDGMENT SHEET",
    ]
    if scenario:
        header.append(f"Scenario: {scenario}")
    header.extend(
        [
            "",
            f"Name of Insurer: {fields['insurer']} PHS ID: {fields['phs_id']}",
            f"Patient Name: {fields['patient_name']}",
            f"Policy No: {fields['policy_no']}",
            f"Hospital ID: {fields.get('hospital_id', fields['phs_id'])}",
            f"Name of Hospital where Admitted: {fields.get('hospital_name', 'City Hospital')}",
            f"Date: {fields['date']}",
            "",
            "DETAILS OF CLAIM:",
            f"Hospital main Bill { ' '.join(fields['amount']) }",
            f"Amount: Rs. {fields['amount']}",
            f"Total claimed amount: {fields['amount']}",
        ]
    )
    return header


def write_text_pdf(
    path: Path,
    fields: Dict[str, str],
    *,
    scenario: str = "",
    producer: str = "ClaimFraudDetection Sample Generator",
) -> None:
    """Write a digital text PDF with Paramount-style labels (+ optional metadata)."""
    lines = _paramount_lines(fields, scenario=scenario)
    safe = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        for line in lines
    ]

    content = ["BT", "/F1 11 Tf", "40 760 Td", "14 TL"]
    for i, line in enumerate(safe):
        if i == 0:
            content.append(f"({line}) Tj")
        else:
            content.append("T*")
            content.append(f"({line}) Tj")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")

    producer_esc = producer.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects: List[bytes] = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
        ),
        (
            f"4 0 obj<< /Length {len(stream)} >>stream\n".encode("ascii")
            + stream
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n",
        (
            f"6 0 obj<< /Producer ({producer_esc}) "
            f"/Creator ({producer_esc}) >>endobj\n"
        ).encode("latin-1", errors="replace"),
    ]

    # Point catalog Info to metadata object 6.
    objects[0] = b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(buffer.tell())
        buffer.write(obj)

    xref_pos = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buffer.write(f"{off:010d} 00000 n \n".encode("ascii"))
    buffer.write(
        (
            f"trailer<< /Size {len(offsets)} /Root 1 0 R /Info 6 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(buffer.getvalue())


def write_noisy_image_pdf(path: Path, fields: Dict[str, str], *, scenario: str) -> None:
    """Render form as a noisy image and wrap in a single-page PDF (low OCR)."""
    lines = _paramount_lines(fields, scenario=scenario)
    width, height = 900, 700
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _font(20)
    y = 36
    for line in lines:
        draw.text((40, y), line, fill=(25, 25, 25), font=font)
        y += 28

    # Heavy noise / low contrast → lower Tesseract confidence + ELA signal.
    img = ImageEnhance.Contrast(img).enhance(0.45)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    noise = Image.effect_noise((width, height), 28).convert("RGB")
    img = Image.blend(img, noise, 0.22)

    # Paste a small overlay rectangle (simulates amount tamper for ELA).
    overlay = Image.new("RGB", (180, 36), color=(245, 245, 245))
    ImageDraw.Draw(overlay).text((8, 6), f"Rs. {fields['amount']}", fill=(10, 10, 10), font=font)
    img.paste(overlay, (40, 430))

    img.save(path, "PDF", resolution=150.0)


def generate_variants(
    out_dir: Path,
    base_pdf: Path = DEFAULT_BASE,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if not base_pdf.is_file():
        raise FileNotFoundError(
            f"Base PDF not found: {base_pdf}. Place claim_08_original.pdf first."
        )

    # 09 — exact duplicate (byte copy of original)
    p09 = out_dir / "claim_09_exact_dup.pdf"
    shutil.copy2(base_pdf, p09)
    written.append(p09)
    print(f"  wrote {p09.name}  [EXACT DUP of {base_pdf.name}]")

    # 10 — fuzzy base (same people/amount as claim_08, longer Policy No)
    fuzzy_base = dict(BASE_FIELDS)
    fuzzy_base["policy_no"] = FUZZY_BASE_POLICY
    p10 = out_dir / "claim_10_fuzzy_base.pdf"
    write_text_pdf(p10, fuzzy_base, scenario="FUZZY_BASE")
    written.append(p10)
    print(f"  wrote {p10.name}  [FUZZY BASE Policy No {FUZZY_BASE_POLICY}]")

    # 11 — fuzzy twin (1-char edit → ratio 96)
    fuzzy_twin = dict(BASE_FIELDS)
    fuzzy_twin["policy_no"] = FUZZY_TWIN_POLICY
    p11 = out_dir / "claim_11_fuzzy.pdf"
    write_text_pdf(p11, fuzzy_twin, scenario="FUZZY_TWIN")
    written.append(p11)
    print(f"  wrote {p11.name}  [FUZZY twin Policy No {FUZZY_TWIN_POLICY}]")

    # 12 — amount tamper + Photoshop metadata (same policy as claim_08)
    tamper = dict(BASE_FIELDS)
    tamper["amount"] = "99000"
    p12 = out_dir / "claim_12_amount_tamper.pdf"
    write_text_pdf(
        p12,
        tamper,
        scenario="AMOUNT_TAMPER",
        producer="Adobe Photoshop 23.0",
    )
    written.append(p12)
    print(f"  wrote {p12.name}  [AMOUNT TAMPER + Photoshop metadata]")

    # 13 — low OCR noisy image PDF (unique policy so not exact-hash vs 08)
    low = dict(BASE_FIELDS)
    low["policy_no"] = "CLAIM-PHS-12345679-LOWOCR"
    p13 = out_dir / "claim_13_low_ocr.pdf"
    write_noisy_image_pdf(p13, low, scenario="LOW_OCR")
    written.append(p13)
    print(f"  wrote {p13.name}  [LOW OCR noisy image PDF]")

    # 14 — cross-reference: same patient + amount + date as fuzzy base, new policy
    xref = dict(BASE_FIELDS)
    xref["policy_no"] = "CLAIM-PHS-87654321-XREF"
    xref["date"] = fuzzy_base["date"]
    p14 = out_dir / "claim_14_crossref.pdf"
    write_text_pdf(p14, xref, scenario="CROSS_REFERENCE")
    written.append(p14)
    print(f"  wrote {p14.name}  [CROSS-REF same patient/amount/date]")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate claim_09+ fraud/fuzzy/low-OCR PDFs from claim_08"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Source PDF (default: samples/claims/claim_08_original.pdf)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CLAIMS,
        help="Output directory (default: samples/claims)",
    )
    args = parser.parse_args()
    print(f"Base : {args.base.resolve()}")
    print(f"Out  : {args.out.resolve()}")
    paths = generate_variants(args.out, args.base)
    print(f"Done — {len(paths)} variant PDFs")


if __name__ == "__main__":
    main()
