"""
Forensic Agent — detects digital tampering via two complementary techniques:

1. PDF Metadata Scrutiny
   Reads the /Producer and /Creator fields embedded in a PDF.  Tools like
   Photoshop, Canva, and Illustrator leave distinctive signatures that expose
   documents built outside standard hospital billing software.

2. Error Level Analysis (ELA)
   Saves a JPEG at reduced quality and computes the pixel-difference between
   the original and the re-compressed version using Pillow.  Regions that were
   digitally altered (e.g., a number pasted on top) retain more compression
   artefacts than the surrounding unchanged pixels, producing a visibly higher
   mean ELA score.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

logger = logging.getLogger(__name__)

# Software names whose presence in PDF metadata signals non-billing origin.
UNAUTHORIZED_PRODUCERS = {
    "photoshop",
    "canva",
    "illustrator",
    "inkscape",
    "gimp",
    "affinity",
    "coreldraw",
    "paint.net",
}

# ELA score (mean absolute pixel difference, 0–255) above which tampering is flagged.
DEFAULT_ELA_THRESHOLD = 8.0

# JPEG quality used for the re-compression step.
DEFAULT_RECOMPRESS_QUALITY = 90


class ForensicAgent:
    """Runs metadata and image-level forensic checks on claim documents."""

    def __init__(
        self,
        ela_threshold: float = DEFAULT_ELA_THRESHOLD,
        recompress_quality: int = DEFAULT_RECOMPRESS_QUALITY,
    ) -> None:
        self.ela_threshold = ela_threshold
        self.recompress_quality = recompress_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_pdf_metadata(self, pdf_bytes: bytes) -> List[str]:
        """
        Parse PDF bytes and return a list of forensic flags.

        Each flag is a human-readable string such as:
            "METADATA: Document created with Photoshop (unauthorized)"
        """
        flags: List[str] = []

        try:
            import PyPDF2  # imported lazily to avoid hard dependency at module load

            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            metadata = reader.metadata or {}

            suspicious = self._scan_metadata_fields(metadata)
            for field_name, software in suspicious:
                flag = (
                    f"METADATA: Document {field_name} field contains "
                    f"'{software}' — unauthorized editing tool detected"
                )
                flags.append(flag)
                logger.warning("Forensic flag: %s", flag)

        except Exception as exc:  # noqa: BLE001
            logger.debug("PDF metadata read failed: %s", exc)
            # Non-fatal: return no metadata flags rather than crashing the pipeline.

        return flags

    def run_ela_analysis(
        self,
        image_source: str | bytes | None,
        quality: int | None = None,
    ) -> Tuple[float, List[str]]:
        """
        Perform Error Level Analysis on the given image.

        Args:
            image_source: File path (str) or raw image bytes.  Pass None to
                          simulate a clean claim (returns score 0.0).
            quality:      JPEG re-compression quality.  Lower → larger
                          differences for pristine images; default 90.

        Returns:
            (ela_score, flags)
            ela_score — mean absolute difference across all pixels (0–255).
            flags     — list of human-readable tamper strings if above threshold.
        """
        if image_source is None:
            return 0.0, []

        recompress_quality = quality or self.recompress_quality
        flags: List[str] = []

        try:
            ela_score = self._compute_ela_score(image_source, recompress_quality)
            logger.debug("ELA score: %.4f (threshold: %.4f)", ela_score, self.ela_threshold)

            if ela_score > self.ela_threshold:
                flag = (
                    f"ELA: Pixel-level tampering detected — ELA score {ela_score:.2f} "
                    f"exceeds threshold {self.ela_threshold:.2f} "
                    f"(possible font overlay or altered amount field)"
                )
                flags.append(flag)
                logger.warning("Forensic flag: %s", flag)

        except Exception as exc:  # noqa: BLE001
            logger.debug("ELA analysis failed: %s", exc)

        return ela_score if "ela_score" in dir() else 0.0, flags

    def analyze_claim(
        self, pdf_bytes: bytes | None, image_source: str | bytes | None
    ) -> Tuple[float, List[str]]:
        """
        Convenience method that runs both metadata and ELA checks and merges results.

        Returns:
            (ela_score, all_flags)
        """
        metadata_flags: List[str] = []
        if pdf_bytes:
            metadata_flags = self.check_pdf_metadata(pdf_bytes)

        ela_score, ela_flags = self.run_ela_analysis(image_source)
        return ela_score, metadata_flags + ela_flags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ela_score(
        self, image_source: str | bytes, quality: int
    ) -> float:
        """
        Core ELA computation:
        1. Open the original image (convert to RGB).
        2. Re-save at reduced JPEG quality into a BytesIO buffer.
        3. Compute absolute pixel difference with ImageChops.difference.
        4. Enhance contrast ×10 to amplify subtle edits (standard ELA technique).
        5. Return the mean absolute difference across all channels.
        """
        if isinstance(image_source, (str, os.PathLike)):
            original = Image.open(image_source).convert("RGB")
        else:
            original = Image.open(io.BytesIO(image_source)).convert("RGB")

        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        diff = ImageChops.difference(original, recompressed)
        diff_enhanced = ImageEnhance.Brightness(diff).enhance(10)

        diff_array = np.array(diff_enhanced, dtype=np.float32)
        ela_score = float(np.mean(np.abs(diff_array)))
        return ela_score

    @staticmethod
    def _scan_metadata_fields(
        metadata: dict,
    ) -> List[Tuple[str, str]]:
        """
        Return (field_name, matched_software) pairs for any suspicious value found
        in the PDF metadata dictionary.
        """
        hits: List[Tuple[str, str]] = []
        fields_of_interest = {
            "/Producer": "producer",
            "/Creator": "creator",
            "/Author": "author",
        }

        for pdf_key, label in fields_of_interest.items():
            raw_value = metadata.get(pdf_key, "") or ""
            lower_value = str(raw_value).lower()
            for software in UNAUTHORIZED_PRODUCERS:
                if software in lower_value:
                    hits.append((label, software.capitalize()))
                    break  # one match per field is enough

        return hits

    # ------------------------------------------------------------------
    # Mock helpers — used when no real document is available
    # ------------------------------------------------------------------

    @staticmethod
    def mock_tampered_pdf_bytes(software: str = "Adobe Photoshop 23.0") -> bytes:
        """
        Build a minimal well-formed PDF with a suspicious /Producer field.
        Used in tests and the 10-claim main.py simulation.
        """
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
    def mock_tampered_image_bytes(ela_score_target: float = 16.0) -> bytes:
        """
        Generate synthetic image bytes guaranteed to produce an ELA score well
        above DEFAULT_ELA_THRESHOLD (8.0).

        Technique:
        - Start with a uniform gray background (compresses cleanly → low ELA).
        - Overwrite the 'amount field' region with random pixel noise.  Random
          high-frequency content does NOT compress stably between quality levels,
          so recompressing the image produces a consistently high mean diff.

        In production, real scanned PDFs are passed — ELA then detects genuine
        paste artefacts at quality boundaries rather than synthetic noise.
        """
        width, height = 400, 200
        rng = np.random.default_rng(seed=2024)

        # Uniform gray base → ELA ≈ 0 for the background
        arr = np.full((height, width, 3), 200, dtype=np.uint8)

        # Random noise patch in the 'amount' region → ELA ≈ 15–20 for this region
        arr[70:130, 80:320] = rng.integers(
            0, 256, size=(60, 240, 3), dtype=np.uint8
        )

        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
