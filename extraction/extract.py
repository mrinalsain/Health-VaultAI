"""
extraction/extract.py
=====================
Orchestrator for the HealthVault AI Report Extraction Module.

Architecture Overview:
----------------------
This module accepts a medical report file (PDF or image) and executes a
transparent, 4-step pipeline:
1. Text Extraction: Extracts text via direct PDF parsing; falls back to OCR if scanned/image.
2. Report Type Suggestion: Deterministically detects report type via keyword matching.
3. Health Value Extraction: Extracts target lab values (cholesterol, glucose, BP, heart rate).
4. Manual Merging: Merges extracted findings with patient-entered manual data.

Core Safety Rules:
------------------
- NO Fabrication: Never invents or assumes values not found in the text.
- NO Risk Model Coupling: Does not import or reference risk models. Plain data only.
- Suggestions Only: Report types are presented as suggestions for patient confirmation.
- Manual Authority: Patient entries always override automated extractions.
"""

import os
from typing import Any, Dict, Optional, Tuple

try:
    from extraction.pdf_parser import extract_text_from_pdf
    from extraction.ocr_parser import extract_text_with_ocr
    from extraction.field_patterns import extract_all_fields, TARGET_FIELDS
    from extraction.report_type_patterns import find_matched_keywords
except ImportError:
    from pdf_parser import extract_text_from_pdf
    from ocr_parser import extract_text_with_ocr
    from field_patterns import extract_all_fields, TARGET_FIELDS
    from report_type_patterns import find_matched_keywords


def extract_text_from_file(file_path: str) -> Tuple[str, str]:
    """
    Extracts text from a given file path.

    Strategy:
    - If file is a PDF, tries fast, lossless direct text extraction first (pdfplumber).
    - If direct extraction yields insufficient text (< 30 chars, indicating a scanned PDF),
      or if the file is an image (.png, .jpg), falls back to OCR (pytesseract).

    Parameters:
        file_path (str): Path to the target report file.

    Returns:
        Tuple[str, str]: (extracted_text, extraction_method_used)
                         where method is either "direct_pdf" or "ocr".
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Extractor] File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # If it is a PDF, try direct text extraction first
    if ext == ".pdf":
        print(f"[Extractor] Attempting direct PDF text extraction on '{os.path.basename(file_path)}'...")
        text = extract_text_from_pdf(file_path)
        if len(text.strip()) >= 30:
            print("[Extractor] Direct PDF text extraction successful.")
            return text, "direct_pdf"

        # Direct text extraction returned empty or negligible text (scanned PDF signal)
        print("[Extractor] Direct PDF text extraction returned minimal text. Falling back to OCR...")
        ocr_text = extract_text_with_ocr(file_path)
        return ocr_text, "ocr"

    # For image formats, directly route to OCR
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
        print(f"[Extractor] Image file detected ('{ext}'). Routing directly to OCR...")
        ocr_text = extract_text_with_ocr(file_path)
        return ocr_text, "ocr"

    else:
        raise ValueError(f"[Extractor] Unsupported file extension: '{ext}'. Expected PDF or image.")


def detect_report_type(text: str) -> Dict[str, Any]:
    """
    Counts keyword matches per report type from the text. Returns the 
    type with the highest match count, plus a confidence level 
    ("high" for multiple strong matches, "low" for one weak match, 
    "unknown" if nothing meaningful matches). NEVER force a confident 
    guess when evidence is weak — return "unknown"/low confidence 
    instead, since this is only ever shown to the patient as a 
    suggestion to confirm or correct, and a wrong confident-looking 
    guess is worse than an honest "not sure."

    Why Keyword Counting instead of a Trained Classifier?
    -----------------------------------------------------
    1. Explainability: We can show the user the exact keywords found (e.g. ['LDL', 'HDL']).
    2. Zero Hallucination: A statistical model might guess 'lipid_panel' for an unclear note
       simply because of vocabulary overlap. Keyword counting only matches genuine terms.
    3. Honesty: When evidence is insufficient, returning 'unknown' empowers the patient
       to manually select their report type without feeling misled by an AI guess.

    Return format:
    {
        "detected_type": "lipid_panel",
        "confidence": "high", 
        "matched_keywords": ["Lipid Panel", "LDL", "HDL"]
    }
    """
    if not text or not text.strip():
        return {
            "detected_type": "unknown",
            "confidence": "unknown",
            "matched_keywords": [],
        }

    # Gather matched keywords per report type
    matches_per_type = find_matched_keywords(text)

    # Calculate match count per type
    type_counts = {rtype: len(kws) for rtype, kws in matches_per_type.items()}

    # Find highest match count
    max_count = max(type_counts.values()) if type_counts else 0

    # If no keywords matched at all:
    if max_count == 0:
        return {
            "detected_type": "unknown",
            "confidence": "unknown",
            "matched_keywords": [],
        }

    # Check for ties among top match counts
    top_types = [rtype for rtype, count in type_counts.items() if count == max_count]

    # If there is a tie between multiple types:
    if len(top_types) > 1:
        # Ambiguous evidence: cannot confidently distinguish between competing types
        return {
            "detected_type": "unknown",
            "confidence": "low",
            "matched_keywords": [
                kw for rtype in top_types for kw in matches_per_type[rtype]
            ],
        }

    best_type = top_types[0]
    matched_kws = matches_per_type[best_type]

    # Confidence assessment:
    # - "high": 2 or more distinct strong keyword matches
    # - "low": exactly 1 keyword match (weak signal, needs user confirmation)
    confidence = "high" if len(matched_kws) >= 2 else "low"

    return {
        "detected_type": best_type,
        "confidence": confidence,
        "matched_keywords": matched_kws,
    }


def extract_health_values(file_path: str) -> Dict[str, Any]:
    """
    Orchestrates end-to-end extraction from a single report file:
    1. Extracts raw text via PDF parser or OCR fallback.
    2. Suggests the report type using detect_report_type().
    3. Runs regex patterns for each target clinical metric:
       - cholesterol (Total Cholesterol in mg/dL)
       - glucose (Blood Glucose in mg/dL)
       - heart_rate (Heart Rate in bpm)
       - systolic_bp (Systolic BP in mmHg)
       - diastolic_bp (Diastolic BP in mmHg)
    4. For each field, formats the result as either:
       {"value": <number>, "unit": "mg/dL", "source": "extracted", 
        "confidence": "high"/"medium", "raw_snippet": "<matched text>"}
       or:
       {"value": None, "source": "not_found"}
    5. Returns a structured dictionary containing field results, report type
       detection, and the extraction method used.

    Parameters:
        file_path (str): File path to PDF or image.

    Returns:
        Dict[str, Any]: Complete extraction result dictionary.
    """
    # 1. Text extraction with automated fallback
    text, method = extract_text_from_file(file_path)

    # 2. Report type detection (suggestion only)
    report_type_result = detect_report_type(text)

    # 3. Field value extraction
    fields = extract_all_fields(text)

    return {
        "fields": fields,
        "report_type_detection": report_type_result,
        "extraction_method": method,
    }


def merge_with_manual_input(
    extracted_dict: Dict[str, Any],
    manual_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Takes the output of extract_health_values() and an optional dict 
    of manually-entered values from the patient. For each field:
    - A manually entered value ALWAYS takes priority over an extracted one.
    - If no manual value exists but extraction succeeded, use the extracted value.
    - If neither exists, leave the field as None — do NOT default it here, 
      that's the risk model's own job later.

    Parameters:
        extracted_dict (dict): Dictionary produced by extract_health_values().
        manual_overrides (dict, optional): Patient-entered values, e.g.
                                           {"cholesterol": 195.0, "glucose": 95.0}

    Returns:
        Dict[str, Dict[str, Any]]: Final merged dictionary where each field is tagged with
        its origin: "manual", "extracted", or "missing".

    Example Return:
        {
            "cholesterol": {"value": 195.0, "source": "manual", "unit": "mg/dL"},
            "glucose": {"value": 92.0, "source": "extracted", "unit": "mg/dL"},
            "heart_rate": {"value": None, "source": "missing", "unit": None},
            ...
        }
    """
    if manual_overrides is None:
        manual_overrides = {}

    merged_results: Dict[str, Dict[str, Any]] = {}
    extracted_fields = extracted_dict.get("fields", {})

    for field in TARGET_FIELDS:
        extracted_item = extracted_fields.get(field, {"value": None, "source": "not_found"})

        # Check if patient provided a manual value for this field
        if field in manual_overrides and manual_overrides[field] is not None:
            # Rule 1: Manual entry ALWAYS wins
            # Use extracted unit if known, or fallback to standard unit for that field
            default_units = {
                "cholesterol": "mg/dL",
                "glucose": "mg/dL",
                "heart_rate": "bpm",
                "systolic_bp": "mmHg",
                "diastolic_bp": "mmHg",
            }
            unit = extracted_item.get("unit") or default_units.get(field)
            merged_results[field] = {
                "value": manual_overrides[field],
                "source": "manual",
                "unit": unit,
            }

        elif extracted_item.get("value") is not None:
            # Rule 2: No manual value, but extraction succeeded
            merged_results[field] = {
                "value": extracted_item["value"],
                "source": "extracted",
                "unit": extracted_item.get("unit", "unspecified"),
            }

        else:
            # Rule 3: Neither exists -> mark as missing, NEVER default
            merged_results[field] = {
                "value": None,
                "source": "missing",
                "unit": None,
            }

    return merged_results
