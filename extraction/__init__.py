"""
extraction package
==================
Independent report extraction module for HealthVault AI.

Exposes:
- extract_health_values: End-to-end extraction from a PDF or image file.
- detect_report_type: Keyword-based report type suggestion.
- merge_with_manual_input: Conflict resolution prioritizing manual patient entry.
- extract_text_from_file: Direct PDF text extraction with OCR fallback.
- extract_text_from_pdf: Direct digital PDF text parser.
- extract_text_with_ocr: Tesseract OCR text parser for scanned reports and images.
"""

from extraction.extract import (
    extract_health_values,
    detect_report_type,
    merge_with_manual_input,
    extract_text_from_file,
)
from extraction.pdf_parser import extract_text_from_pdf
from extraction.ocr_parser import extract_text_with_ocr

__all__ = [
    "extract_health_values",
    "detect_report_type",
    "merge_with_manual_input",
    "extract_text_from_file",
    "extract_text_from_pdf",
    "extract_text_with_ocr",
]
