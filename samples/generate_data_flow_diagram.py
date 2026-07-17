"""
Render the ClaimFraudDetection data-flow diagram.

Usage:
    python -m samples.generate_data_flow_diagram
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PNG = ROOT / "data_flow_diagram.png"
SVG = ROOT / "data_flow_diagram.svg"

W, H = 1400, 980
BG = (247, 243, 234)
INK = (28, 25, 21)
MUTED = (92, 86, 76)
LINE = (180, 170, 150)
CARD = (255, 253, 248)
ACCENT = (15, 76, 92)
APPROVED = (31, 107, 74)
FLAGGED = (155, 28, 28)
PENDING = (138, 90, 0)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_png(path: Path = PNG) -> Path:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_f = _font(28, True)
    section_f = _font(14, True)
    body_f = _font(13)
    small_f = _font(11)

    def rounded_rect(xy, fill, outline=LINE, radius=10, width=2):
        draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

    def box(x, y, w, h, title, lines, fill=CARD, title_color=ACCENT):
        rounded_rect((x, y, x + w, y + h), fill=fill)
        draw.text((x + 14, y + 10), title, font=section_f, fill=title_color)
        ty = y + 34
        for line in lines:
            draw.text((x + 14, ty), line, font=body_f, fill=INK)
            ty += 18

    def arrow(x1, y1, x2, y2):
        draw.line((x1, y1, x2, y2), fill=ACCENT, width=2)
        if abs(y2 - y1) >= abs(x2 - x1):
            draw.polygon([(x2, y2), (x2 - 6, y2 - 10), (x2 + 6, y2 - 10)], fill=ACCENT)
        else:
            draw.polygon([(x2, y2), (x2 - 10, y2 - 6), (x2 - 10, y2 + 6)], fill=ACCENT)

    draw.text((40, 28), "Claim Fraud Detection — Data Flow", font=title_f, fill=INK)
    draw.text(
        (40, 66),
        "Documents → Extract → Analyze → Decide → Advisory → Outputs",
        font=body_f,
        fill=MUTED,
    )

    box(40, 110, 280, 90, "1. Input", [
        "PDF / PNG / JPG claims",
        "or mock synthetic claims",
    ], fill=(238, 246, 248))
    box(360, 110, 300, 90, "2. Document Extractor", [
        "Digital PDF → PyPDF2 text",
        "Images / scans → Tesseract",
        "Parse fields → claim JSON",
    ])
    box(700, 110, 300, 90, "3. Extraction Agent", [
        "Normalize patient / invoice",
        "OCR confidence scoring",
        "Build Claim model",
    ])
    box(1040, 110, 300, 90, "Extracted JSON", [
        "samples/extracted/*.json",
        "Ready for pipeline steps",
    ], fill=(238, 246, 248))

    arrow(320, 155, 360, 155)
    arrow(660, 155, 700, 155)
    arrow(1000, 155, 1040, 155)

    box(200, 280, 280, 100, "4. Forensic Agent", [
        "PDF metadata (edit tools)",
        "Image tamper check (ELA)",
        "Document authenticity flags",
    ])
    box(560, 280, 280, 100, "5. Auditor Agent", [
        "Exact duplicate match",
        "Near-match invoice / policy",
        "Same patient+date+amount",
    ])
    box(920, 280, 280, 100, "6. Risk Scorer", [
        "Weighted fraud signals",
        "Score 0–100",
        "Drive claim decision",
    ])

    arrow(850, 200, 340, 280)
    arrow(850, 200, 700, 280)
    arrow(480, 380, 560, 330)
    arrow(840, 330, 920, 330)

    box(200, 460, 200, 88, "Approved", [
        "Score under 40",
        "Pay / process normally",
    ], fill=(229, 244, 236), title_color=APPROVED)
    box(460, 460, 220, 88, "Needs review", [
        "Score 40–75",
        "Claim manager checks",
    ], fill=(255, 243, 214), title_color=PENDING)
    box(740, 460, 220, 88, "Hold — investigate", [
        "Score over 75",
        "SIU / investigation",
    ], fill=(253, 232, 232), title_color=FLAGGED)

    arrow(1060, 380, 300, 460)
    arrow(1060, 380, 570, 460)
    arrow(1060, 380, 850, 460)

    box(200, 620, 340, 100, "7. Advisory narrative (optional)", [
        "Local model for held / review claims",
        "Plain-language findings for managers",
        "Does not override exact duplicates",
    ], fill=(238, 246, 248))
    box(600, 620, 300, 100, "8. Feedback store", [
        "SQLite predictions + overrides",
        "retrain_thresholds() learning",
        "Improve future sensitivity",
    ])
    box(960, 620, 340, 100, "9. Reports", [
        "HTML analysis for executives",
        "Timing + efficiency KPIs",
        "Console summary",
    ], fill=(238, 246, 248))

    arrow(570, 548, 370, 620)
    arrow(850, 548, 370, 620)
    arrow(370, 720, 600, 670)
    arrow(900, 670, 960, 670)

    draw.text(
        (40, 780),
        "Authority: rule-based score decides status. Advisory text only explains findings for claim managers.",
        font=body_f,
        fill=MUTED,
    )
    draw.text(
        (40, 810),
        "Outputs: samples/extracted/*.json  ·  samples/analysis_report.html  ·  fraud_feedback.db",
        font=small_f,
        fill=MUTED,
    )
    draw.text((40, 900), "ClaimFraudDetection", font=section_f, fill=ACCENT)
    draw.text((40, 925), "Data flow diagram", font=small_f, fill=MUTED)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    return path


def render_svg(path: Path = SVG) -> Path:
    def box(x, y, w, h, title, lines, fill="#fffdf8", title_fill="#0f4c5c"):
        text_lines = "\n".join(
            f'<text x="{x+14}" y="{y+52+i*18}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="13" fill="#1c1915">{line}</text>'
            for i, line in enumerate(lines)
        )
        return f'''
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#b4aa96" stroke-width="2"/>
  <text x="{x+14}" y="{y+28}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" fill="{title_fill}">{title}</text>
  {text_lines}
'''

    def arrow(x1, y1, x2, y2):
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#0f4c5c" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    parts = [
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0f4c5c"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f7f3ea"/>',
        '<text x="40" y="50" font-family="Arial, Helvetica, sans-serif" font-size="28" '
        'font-weight="bold" fill="#1c1915">Claim Fraud Detection — Data Flow</text>',
        '<text x="40" y="78" font-family="Arial, Helvetica, sans-serif" font-size="14" '
        'fill="#5c564c">Documents → Extract → Analyze → Decide → Advisory → Outputs</text>',
        box(40, 110, 280, 90, "1. Input", ["PDF / PNG / JPG claims", "or mock synthetic claims"], fill="#eef6f8"),
        box(360, 110, 300, 90, "2. Document Extractor", ["Digital PDF → PyPDF2 text", "Images / scans → Tesseract", "Parse fields → claim JSON"]),
        box(700, 110, 300, 90, "3. Extraction Agent", ["Normalize patient / invoice", "OCR confidence scoring", "Build Claim model"]),
        box(1040, 110, 300, 90, "Extracted JSON", ["samples/extracted/*.json", "Ready for pipeline steps"], fill="#eef6f8"),
        arrow(320, 155, 355, 155),
        arrow(660, 155, 695, 155),
        arrow(1000, 155, 1035, 155),
        box(200, 280, 280, 100, "4. Forensic Agent", ["PDF metadata (edit tools)", "Image tamper check (ELA)", "Document authenticity flags"]),
        box(560, 280, 280, 100, "5. Auditor Agent", ["Exact duplicate match", "Near-match invoice / policy", "Same patient+date+amount"]),
        box(920, 280, 280, 100, "6. Risk Scorer", ["Weighted fraud signals", "Score 0–100", "Drive claim decision"]),
        arrow(850, 200, 340, 280),
        arrow(850, 200, 700, 280),
        arrow(480, 380, 555, 330),
        arrow(840, 330, 915, 330),
        box(200, 460, 200, 88, "Approved", ["Score under 40", "Pay / process normally"], fill="#e5f4ec", title_fill="#1f6b4a"),
        box(460, 460, 220, 88, "Needs review", ["Score 40–75", "Claim manager checks"], fill="#fff3d6", title_fill="#8a5a00"),
        box(740, 460, 220, 88, "Hold — investigate", ["Score over 75", "SIU / investigation"], fill="#fde8e8", title_fill="#9b1c1c"),
        arrow(1060, 380, 300, 460),
        arrow(1060, 380, 570, 460),
        arrow(1060, 380, 850, 460),
        box(200, 620, 340, 100, "7. Advisory narrative (optional)", ["Local model for held / review claims", "Plain-language findings for managers", "Does not override exact duplicates"], fill="#eef6f8"),
        box(600, 620, 300, 100, "8. Feedback store", ["SQLite predictions + overrides", "retrain_thresholds() learning", "Improve future sensitivity"]),
        box(960, 620, 340, 100, "9. Reports", ["HTML analysis for executives", "Timing + efficiency KPIs", "Console summary"], fill="#eef6f8"),
        arrow(570, 548, 370, 620),
        arrow(850, 548, 370, 620),
        arrow(540, 670, 595, 670),
        arrow(900, 670, 955, 670),
        '<text x="40" y="800" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#5c564c">'
        "Authority: rule-based score decides status. Advisory text only explains findings for claim managers.</text>",
        '<text x="40" y="830" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#5c564c">'
        "Outputs: samples/extracted/*.json  ·  samples/analysis_report.html  ·  fraud_feedback.db</text>",
        '<text x="40" y="920" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="bold" fill="#0f4c5c">ClaimFraudDetection</text>',
        '<text x="40" y="945" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#5c564c">Data flow diagram</text>',
    ]
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        + "\n".join(parts)
        + "\n</svg>\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    png = render_png()
    svg = render_svg()
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
