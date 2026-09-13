"""
extraction/pdf_parser.py
========================
Extracts text directly from digital (non-scanned) PDF documents.

Concept: Digital PDF vs. Scanned PDF
------------------------------------
- Digital PDFs: Created directly by a computer (e.g., exported from medical software,
  Word, or Google Docs). The text is embedded as vector character codes and font
  instructions. Extracting text from these files is fast, exact, and 100% accurate.
- Scanned PDFs: Created by scanning a physical sheet of paper with a scanner or taking
  a smartphone photo. The PDF actually contains only an image (bitmap of pixels) wrapped
  in a PDF container. To standard PDF text parsers, a scanned page looks completely empty!
  Therefore, if we find little or no text across the pages, it is a strong signal that
  the document is scanned and requires Optical Character Recognition (OCR).
"""

import os
from typing import Optional
import pdfplumber

# Minimum character count across the entire PDF to consider direct extraction successful.
# If an entire document yields fewer than 30 characters, it is almost certainly a scanned
# image or blank document rather than a readable digital text PDF.
MIN_TEXT_LENGTH_THRESHOLD = 30


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from all pages of a digital PDF file using pdfplumber.

    Parameters:
        file_path (str): The path to the PDF document.

    Returns:
        str: Concatenated text from all pages. If the document has little or no
             extractable text (indicating a scanned image), returns an empty string
             as a clear signal to fall back to OCR.

    Behavior:
        - Validates file existence.
        - Loops through EVERY page in the PDF (multi-page support).
        - Prints page count and processing status.
        - Signals scanned documents clearly when text is below threshold.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[PDF Parser] File not found: {file_path}")

    extracted_pages = []
    total_pages = 0

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"[PDF Parser] Found {total_pages} page(s) in '{os.path.basename(file_path)}'.")

            for page_index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())

        full_text = "\n\n".join(extracted_pages).strip()
        print(f"[PDF Parser] Processed {total_pages} page(s). Total characters extracted: {len(full_text)}.")

        # Check if the extracted text meets the minimum threshold
        if len(full_text) < MIN_TEXT_LENGTH_THRESHOLD:
            print(
                f"[PDF Parser] Signal: Document has little or no extractable text "
                f"({len(full_text)} chars found). This indicates a scanned or image-only PDF."
            )
            return ""

        return full_text

    except Exception as e:
        print(f"[PDF Parser] Error reading PDF '{file_path}': {e}")
        return ""
