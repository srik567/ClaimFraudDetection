"""
Generate sample insurance claim documents for OCR demos.

Usage:
    python -m samples.generate_samples
    python -m samples.generate_samples --out samples/claims
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "claims"


def _font(size: int = 28) -> ImageFont.ImageFont:
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


def _claim_lines(fields: Dict[str, str]) -> List[str]:
    return [
        "INSURANCE CLAIM FORM",
        "",
        f"Invoice ID: {fields['invoice_id']}",
        f"Patient Name: {fields['patient_name']}",
        f"Hospital ID: {fields['hospital_id']}",
        f"Amount: ${fields['amount']}",
        f"Date: {fields['date']}",
    ]


def render_claim_image(
    fields: Dict[str, str],
    *,
    noisy: bool = False,
    size: Tuple[int, int] = (900, 520),
) -> Image.Image:
    """Render a labeled claim form image suitable for Tesseract OCR."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _font(36)
    body_font = _font(28)

    y = 40
    for i, line in enumerate(_claim_lines(fields)):
        font = title_font if i == 0 else body_font
        draw.text((48, y), line, fill=(20, 20, 20), font=font)
        y += 58 if i == 0 else 48

    if noisy:
        # Slight blur / contrast drop to reduce OCR confidence.
        img = ImageEnhance.Contrast(img).enhance(0.55)
        img = ImageEnhance.Brightness(img).enhance(1.15)
        noise = Image.effect_noise(size, 18).convert("RGB")
        img = Image.blend(img, noise, 0.12)
    return img


def write_minimal_pdf(path: Path, fields: Dict[str, str]) -> None:
    """
    Write a simple text PDF without reportlab.

    Uses a minimal PDF content stream so PyPDF2 can extract embedded text.
    """
    lines = _claim_lines(fields)
    # Escape parentheses for PDF string literals.
    safe_lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]

    content_lines = ["BT", "/F1 16 Tf", "50 750 Td", "18 TL"]
    for i, line in enumerate(safe_lines):
        if i == 0:
            content_lines.append(f"({line}) Tj")
        else:
            content_lines.append("T*")
            content_lines.append(f"({line}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    objects.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objects.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objects.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
    )
    objects.append(
        f"4 0 obj<< /Length {len(stream)} >>stream\n".encode("ascii")
        + stream
        + b"\nendstream\nendobj\n"
    )
    objects.append(
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
    )

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
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(buffer.getvalue())


SAMPLE_SPECS: List[Dict[str, object]] = [
    {
        "filename": "claim_01_clean.png",
        "kind": "image",
        "fields": {
            "invoice_id": "CLAIM-H001-ABC-202403-001",
            "patient_name": "Alice Johnson",
            "hospital_id": "HOSP-001",
            "amount": "1250.00",
            "date": "2024-03-01",
        },
    },
    {
        "filename": "claim_02_clean_pdf.pdf",
        "kind": "pdf",
        "fields": {
            "invoice_id": "CLAIM-H002-DEF-202403-010",
            "patient_name": "Robert Smith",
            "hospital_id": "HOSP-002",
            "amount": "890.50",
            "date": "2024-03-05",
        },
    },
    {
        "filename": "claim_03_base.png",
        "kind": "image",
        "fields": {
            "invoice_id": "CLAIM-H003-XYZ-202403-100",
            "patient_name": "Maria Garcia",
            "hospital_id": "HOSP-003",
            "amount": "2400.00",
            "date": "2024-03-10",
        },
    },
    {
        # Near-duplicate invoice ID (I vs 1) for fuzzy matching demos.
        "filename": "claim_04_fuzzy.png",
        "kind": "image",
        "fields": {
            "invoice_id": "CLAIM-H003-XYZ-202403-I00",
            "patient_name": "Maria Garcia",
            "hospital_id": "HOSP-003",
            "amount": "2400.00",
            "date": "2024-03-10",
        },
    },
    {
        "filename": "claim_05_dup_a.png",
        "kind": "image",
        "fields": {
            "invoice_id": "CLAIM-H004-DUP-202403-200",
            "patient_name": "James Wilson",
            "hospital_id": "HOSP-004",
            "amount": "1750.25",
            "date": "2024-03-12",
        },
    },
    {
        # Exact content twin of claim_05 for exact-hash detection.
        "filename": "claim_06_dup_b.png",
        "kind": "image",
        "fields": {
            "invoice_id": "CLAIM-H004-DUP-202403-200",
            "patient_name": "James Wilson",
            "hospital_id": "HOSP-004",
            "amount": "1750.25",
            "date": "2024-03-12",
        },
    },
    {
        "filename": "claim_07_low_ocr.png",
        "kind": "image",
        "noisy": True,
        "fields": {
            "invoice_id": "CLAIM-H005-LOW-202403-300",
            "patient_name": "Emily Chen",
            "hospital_id": "HOSP-005",
            "amount": "640.00",
            "date": "2024-03-15",
        },
    },
]


def generate_samples(out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for spec in SAMPLE_SPECS:
        fields = spec["fields"]  # type: ignore[assignment]
        path = out_dir / str(spec["filename"])
        if spec["kind"] == "pdf":
            write_minimal_pdf(path, fields)  # type: ignore[arg-type]
        else:
            img = render_claim_image(
                fields,  # type: ignore[arg-type]
                noisy=bool(spec.get("noisy", False)),
            )
            img.save(path)
        written.append(path)
        print(f"  wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample claim documents")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: samples/claims)",
    )
    args = parser.parse_args()
    print(f"Generating sample claims in {args.out.resolve()}")
    paths = generate_samples(args.out)
    print(f"Done — {len(paths)} files")


if __name__ == "__main__":
    main()
