"""
extraction/ocr_parser.py
========================
Extracts text from scanned PDF documents and image files using OCR (Optical Character Recognition).

Concept: What is OCR and why is it less reliable than direct PDF text?
----------------------------------------------------------------------
OCR (Optical Character Recognition) analyzes image pixels to recognize letter and number shapes:
1. Pixels vs. Characters: In a digital PDF, the letter 'A' is stored as an exact digital character code.
   In a scanned document, 'A' is merely a cluster of dark pixels on a white background.
2. The OCR Process: The OCR engine (Tesseract) inspects the shapes, strokes, and contours of the pixels
   and compares them against learned statistical models to guess the corresponding character.
3. Why OCR is less reliable:
   - Image Quality: Low scan resolution, camera blur, shadows, and page tilt can distort letter shapes.
   - Ambiguous Characters: OCR commonly misreads '0' (zero) as 'O' (letter O), '8' as 'B', or misses
     faint decimal points (e.g. reading '5.8' as '58').
   - Processing Speed: OCR requires substantial visual computation compared to direct text parsing.
   For these reasons, HealthVault AI always attempts direct digital PDF extraction first, and only
   falls back to OCR when direct extraction fails or yields insufficient text.
"""

import os
from typing import List
from PIL import Image
import pytesseract

# Standard Windows installation paths for Tesseract-OCR
COMMON_TESSERACT_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def configure_tesseract_path() -> bool:
    """
    Checks if Tesseract is available in system PATH or at standard Windows install paths.
    If found at a standard path, updates pytesseract.pytesseract.tesseract_cmd.

    Returns:
        bool: True if Tesseract executable is located and ready, False otherwise.
    """
    # 1. Check if standard command works directly (i.e. in system PATH)
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass

    # 2. Check common Windows installation directories
    for path in COMMON_TESSERACT_WINDOWS_PATHS:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                continue

    return False


def _load_images_from_file(file_path: str) -> List[Image.Image]:
    """
    Converts a file (PDF or image) into a list of PIL Images for OCR processing.

    For PDFs:
    - First attempts conversion via `pypdfium2` (which runs without external Poppler binaries).
    - Falls back to `pdf2image` if available.
    For standard image files (.png, .jpg, .jpeg, .tiff, .bmp):
    - Opens directly using Pillow.
    """
    ext = os.path.splitext(file_path)[1].lower()
    images: List[Image.Image] = []

    if ext == ".pdf":
        # Multi-page PDF: render pages to images
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_path)
            for page in pdf:
                # Render at 2x scale for higher OCR character recognition clarity
                pil_img = page.render(scale=2.0).to_pil()
                images.append(pil_img)
            return images
        except Exception as e:
            print(f"[OCR Parser] pypdfium2 render failed ({e}), trying pdf2image...")

        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            return images
        except Exception as e:
            print(f"[OCR Parser] pdf2image render failed ({e}).")
            return []

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
        try:
            img = Image.open(file_path)
            images.append(img)
            return images
        except Exception as e:
            print(f"[OCR Parser] Failed to open image file '{file_path}': {e}")
            return []
    else:
        print(f"[OCR Parser] Unsupported file format for OCR: {ext}")
        return []


def extract_text_with_ocr(file_path: str) -> str:
    """
    Extracts text from a scanned PDF or image file using Tesseract OCR.

    Parameters:
        file_path (str): Path to the image or scanned PDF file.

    Returns:
        str: Extracted text from all pages/images combined, or empty string if OCR fails.

    Notice on Reliability:
        OCR output depends heavily on scan resolution, lighting, and document clarity.
        Always inspect extracted values against the original report snippet.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[OCR Parser] File not found: {file_path}")

    # Ensure Tesseract binary is located
    is_configured = configure_tesseract_path()
    if not is_configured:
        print(
            f"[OCR Parser] Warning: Tesseract OCR engine is not installed or not in PATH.\n"
            f"             On Windows, download the installer from:\n"
            f"             https://github.com/UB-Mannheim/tesseract/wiki\n"
            f"             and install to C:\\Program Files\\Tesseract-OCR."
        )
        return ""

    images = _load_images_from_file(file_path)
    if not images:
        print(f"[OCR Parser] No readable pages or images could be extracted from '{file_path}'.")
        return ""

    print(f"[OCR Parser] Processing {len(images)} page(s)/image(s) with Tesseract OCR...")
    extracted_text_blocks = []

    for index, img in enumerate(images, start=1):
        try:
            # Perform OCR on image
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                extracted_text_blocks.append(page_text.strip())
            print(f"[OCR Parser] Page {index}/{len(images)} OCR completed.")
        except Exception as e:
            print(f"[OCR Parser] OCR failed on page {index}: {e}")

    full_text = "\n\n".join(extracted_text_blocks).strip()
    print(f"[OCR Parser] OCR complete. Total characters extracted: {len(full_text)}.")
    return full_text
