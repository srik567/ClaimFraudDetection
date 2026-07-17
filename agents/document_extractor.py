"""
Document Extractor — hybrid filesystem OCR for claim documents.

Strategy:
  1. Digital PDFs  → extract embedded text with PyPDF2
  2. Empty/scanned PDFs → rasterize via pdf2image (Poppler) then Tesseract
  3. Images (PNG/JPG/TIFF) → Tesseract OCR with word-level confidence

Parses both demo claim forms (Invoice ID / Amount) and common Indian TPA
labels (Policy No, PHS ID, Patient Name, Hospital main Bill, etc.).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | SUPPORTED_PDF_SUFFIXES

# Minimum characters of embedded PDF text before we treat the PDF as digital.
MIN_DIGITAL_PDF_CHARS = 40

# Stop instructional / filler text that often follows sample values.
_VALUE_CUTOFF = re.compile(
    r"\s+---+|\s+[—–-]+\s+|\s{2,}|\n|(?i)\b(?:enter|name of person|indicate|as allotted)\b"
)


class DocumentExtractionError(RuntimeError):
    """Raised when a document cannot be read or required fields are missing."""


class DocumentExtractor:
    """Read claim documents from disk and produce pipeline-ready dicts."""

    def extract_from_path(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract claim fields from a PDF or image file.

        Returns a dict compatible with ExtractionAgent.mock_ocr_extract /
        FraudPipeline.process, including source bytes/path for forensics.
        """
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            raise DocumentExtractionError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise DocumentExtractionError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
            )

        if suffix in SUPPORTED_PDF_SUFFIXES:
            text, confidence, image_path, pdf_bytes = self._extract_pdf(file_path)
        else:
            text, confidence, image_path, pdf_bytes = self._extract_image(file_path)

        t_parse = time.perf_counter()
        fields = self.parse_claim_text(text)
        parse_ms = (time.perf_counter() - t_parse) * 1000.0
        missing = [
            key
            for key in ("invoice_id", "amount", "patient_name", "hospital_id")
            if not fields.get(key)
        ]
        if missing:
            raise DocumentExtractionError(
                f"Could not parse required fields {missing} from {file_path.name}. "
                f"OCR text preview: {text[:240]!r}"
            )

        timestamp = fields.get("timestamp") or datetime.utcnow()
        if isinstance(timestamp, str):
            timestamp = self._parse_date(timestamp) or datetime.utcnow()

        # Wall time for this file is finalized by extract_directory.
        payload: Dict[str, Any] = {
            "invoice_id": str(fields["invoice_id"]).strip(),
            "amount": float(fields["amount"]),
            "patient_name": str(fields["patient_name"]).strip(),
            "hospital_id": str(fields["hospital_id"]).strip(),
            "timestamp": timestamp.isoformat(),
            "ocr_confidence": float(confidence),
            "source_file": str(file_path),
            "raw_text": text,
            "_extract_ms": parse_ms,
        }
        if image_path:
            payload["image_path"] = image_path
        if pdf_bytes is not None:
            payload["raw_pdf_bytes"] = pdf_bytes
        return payload

    def extract_directory(
        self, directory: Union[str, Path]
    ) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[Path, str]]]:
        """
        Extract all supported documents in a directory (non-recursive).

        Returns:
            (successes, failures) where failures are (path, error_message).
        """
        dir_path = Path(directory).expanduser().resolve()
        if not dir_path.is_dir():
            raise DocumentExtractionError(f"Not a directory: {dir_path}")

        successes: List[Tuple[Path, Dict[str, Any]]] = []
        failures: List[Tuple[Path, str]] = []
        files = sorted(
            p
            for p in dir_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_SUFFIXES
            and not p.name.startswith(".")
            and ".ocr_page" not in p.name
        )
        for file_path in files:
            try:
                t0 = time.perf_counter()
                payload = self.extract_from_path(file_path)
                # overwrite with full wall clock for this file
                payload["_extract_ms"] = (time.perf_counter() - t0) * 1000.0
                successes.append((file_path, payload))
            except DocumentExtractionError as exc:
                logger.error("Skipping %s: %s", file_path.name, exc)
                failures.append((file_path, str(exc)))
        return successes, failures

    # ------------------------------------------------------------------
    # PDF / image readers
    # ------------------------------------------------------------------

    def _extract_pdf(
        self, file_path: Path
    ) -> Tuple[str, float, Optional[str], bytes]:
        pdf_bytes = file_path.read_bytes()
        text = self._pdf_embedded_text(pdf_bytes)
        if len(text.strip()) >= MIN_DIGITAL_PDF_CHARS:
            logger.info("Digital PDF text extracted from %s", file_path.name)
            return text, 0.98, None, pdf_bytes

        logger.info(
            "PDF %s has little embedded text — attempting scan OCR via pdf2image",
            file_path.name,
        )
        images = self._rasterize_pdf(file_path)
        if not images:
            raise DocumentExtractionError(
                f"Scanned PDF '{file_path.name}' could not be rasterized. "
                "Install Poppler (`brew install poppler`) or provide a PNG/JPG."
            )

        texts: List[str] = []
        confidences: List[float] = []
        first_image_path: Optional[str] = None
        for idx, image in enumerate(images):
            page_text, page_conf = self._tesseract_ocr(image)
            texts.append(page_text)
            confidences.append(page_conf)
            if idx == 0:
                cache_dir = file_path.parent / ".ocr_cache"
                cache_dir.mkdir(exist_ok=True)
                cache_path = cache_dir / f"{file_path.stem}_page0.png"
                image.save(cache_path)
                first_image_path = str(cache_path)

        combined = "\n".join(texts).strip()
        confidence = sum(confidences) / len(confidences) if confidences else 0.5
        if not combined:
            raise DocumentExtractionError(
                f"Tesseract returned empty text for scanned PDF {file_path.name}"
            )
        return combined, confidence, first_image_path, pdf_bytes

    def _extract_image(
        self, file_path: Path
    ) -> Tuple[str, float, str, None]:
        from PIL import Image

        with Image.open(file_path) as img:
            image = img.convert("RGB")
            text, confidence = self._tesseract_ocr(image)
        if not text.strip():
            raise DocumentExtractionError(
                f"Tesseract returned empty text for image {file_path.name}"
            )
        return text, confidence, str(file_path), None

    @staticmethod
    def _pdf_embedded_text(pdf_bytes: bytes) -> str:
        import io

        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()

    @staticmethod
    def _rasterize_pdf(file_path: Path):
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise DocumentExtractionError(
                "pdf2image is required for scanned PDFs. "
                "pip install pdf2image and brew install poppler"
            ) from exc

        try:
            return convert_from_path(str(file_path), dpi=200)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdf2image failed for %s: %s", file_path.name, exc)
            return []

    @staticmethod
    def _tesseract_ocr(image) -> Tuple[str, float]:
        try:
            import pytesseract
        except ImportError as exc:
            raise DocumentExtractionError(
                "pytesseract is required for image OCR. pip install pytesseract "
                "and brew install tesseract"
            ) from exc

        try:
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise DocumentExtractionError(
                "Tesseract binary not found. Install with: brew install tesseract"
            ) from exc

        words: List[str] = []
        confidences: List[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            token = (text or "").strip()
            if not token:
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = -1.0
            if conf_val < 0:
                continue
            words.append(token)
            confidences.append(conf_val)

        joined = " ".join(words)
        page_text = pytesseract.image_to_string(image) or joined
        mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.5
        return page_text.strip(), round(mean_conf, 4)

    # ------------------------------------------------------------------
    # Field parsing
    # ------------------------------------------------------------------

    @classmethod
    def parse_claim_text(cls, text: str) -> Dict[str, Any]:
        """Parse labeled claim fields from OCR / PDF text (demo + TPA forms)."""
        cleaned = text.replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        fields: Dict[str, Any] = {}

        invoice = cls._first_match(
            cleaned,
            [
                r"(?i)invoice\s*(?:id|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})",
                r"(?i)policy\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})",
                r"(?i)claim\s*(?:no|number|id)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})",
                r"(?i)phs\s*id\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{3,})",
            ],
        )
        if invoice:
            fields["invoice_id"] = cls._clean_value(invoice)

        patient = cls._first_match(
            cleaned,
            [
                r"(?i)patient\s*name\s*[:\-]?\s*([A-Za-z][A-Za-z .'\-]{0,80})",
                r"(?i)name\s+of\s+patient\s*[:\-]?\s*([A-Za-z][A-Za-z .'\-]{0,80})",
            ],
        )
        if patient:
            fields["patient_name"] = cls._clean_value(patient)

        hospital = cls._first_nonempty(
            cleaned,
            [
                # Prefer TPA / insurer IDs over blank "Hospital ID Enter..." prompts.
                r"(?i)phs\s*id\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{2,})",
                r"(?i)hospital\s+(?:id|code)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{2,})",
                r"(?i)name\s+of\s+insurer\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 .&\-]{2,40})",
            ],
        )
        if hospital:
            fields["hospital_id"] = hospital

        amount = cls._parse_amount(cleaned)
        if amount is not None:
            fields["amount"] = amount

        timestamp = cls._first_match(
            cleaned,
            [
                r"(?i)(?:date|submitted|timestamp)\s*[:\-]?\s*"
                r"([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}"
                r"|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})",
            ],
        )
        if timestamp:
            fields["timestamp"] = timestamp.strip()

        return fields

    @classmethod
    def _parse_amount(cls, text: str) -> Optional[float]:
        """Extract claim amount from demo labels or TPA bill layouts."""
        m = re.search(
            r"(?i)(?:amount|claim\s*amount|total\s*claimed\s*amount)\s*[:\-]?\s*"
            r"(?:rs\.?|inr|₹|\$)?\s*"
            r"([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|[0-9]+\.[0-9]{2}|[0-9]{3,})",
            text,
        )
        if m:
            return float(m.group(1).replace(",", ""))

        # Demo forms: Amount: $1250.00 (also allow shorter totals)
        m = re.search(
            r"(?i)\bamount\s*[:\-]?\s*\$?\s*"
            r"([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?|[0-9]+\.[0-9]{2}|[0-9]+)",
            text,
        )
        if m:
            return float(m.group(1).replace(",", ""))

        # TPA boxed digits on one line: "Hospital main Bill 1 1 0 0 0"
        # Use [ \t] (not \s) so we do not pull the next line's leading "2.".
        m = re.search(
            r"(?i)hospital\s+main\s+bill\s*[:\-]?\s*([0-9](?:[ \t]+[0-9]){2,})",
            text,
        )
        if m:
            return float("".join(m.group(1).split()))

        spaced_total = 0.0
        found = False
        for label in (
            r"hospital\s+main\s+bill",
            r"pre-?hospitalization\s+bills?",
            r"post-?hospitalization\s+bills?",
        ):
            m = re.search(
                rf"(?i){label}[^0-9\n]{{0,40}}([0-9](?:[ \t]+[0-9]){{2,}})",
                text,
            )
            if m:
                spaced_total += float("".join(m.group(1).split()))
                found = True
        if found and spaced_total > 0:
            return spaced_total

        m = re.search(r"(?i)(?:rs\.?|inr|₹)\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})", text)
        if m:
            return float(m.group(1).replace(",", ""))
        return None

    @staticmethod
    def _first_match(text: str, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    @classmethod
    def _first_nonempty(cls, text: str, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            cleaned = cls._clean_value(match.group(1))
            if cleaned and cleaned.lower() not in {"enter", "the", "id", "number", "yes", "no"}:
                return cleaned
        return None

    @staticmethod
    def _clean_value(raw: str) -> str:
        truncated = _VALUE_CUTOFF.split(raw, maxsplit=1)[0]
        return truncated.strip(" .-:\t")

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        candidates = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%m/%d/%y",
        ]
        for fmt in candidates:
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
        return None
