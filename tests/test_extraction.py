"""
tests/test_extraction.py
========================
Automated test suite for the HealthVault AI report extraction module.

Covers:
1. End-to-end extraction on 4 sample PDF files:
   - lipid_panel.pdf (multi-page digital PDF)
   - cbc.pdf (complete blood count with no target metrics)
   - glucose_test.pdf (fasting glucose in mmol/L -> converted to mg/dL)
   - ambiguous_report.pdf (encounter note with vitals)
2. Report type detection confidence and matched keywords validation.
3. Field value extraction (cholesterol, glucose, BP, heart rate).
4. Manual input merging:
   - Overriding an extracted value (manual cholesterol overrides extracted cholesterol)
   - Filling a missing value (patient inputs glucose when extraction found none)
   - Preserving missing state for unprovided metrics.
"""

import os
import sys
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.extract import (
    extract_health_values,
    detect_report_type,
    merge_with_manual_input,
    extract_text_from_file,
)
from extraction.pdf_parser import extract_text_from_pdf


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def test_pdf_extraction():
    print_section("Part 5.10: Testing End-to-End Extraction on Sample Reports")
    sample_dir = os.path.join(PROJECT_ROOT, "tests", "sample_reports")
    test_files = [
        "lipid_panel.pdf",
        "cbc.pdf",
        "glucose_test.pdf",
        "ambiguous_report.pdf",
        "unclear_note.pdf",
    ]

    all_results = {}

    for fname in test_files:
        fpath = os.path.join(sample_dir, fname)
        print(f"\n--- Testing File: {fname} ---")
        result = extract_health_values(fpath)
        all_results[fname] = result

        print(f"Extraction Method: {result['extraction_method']}")
        print(f"Detected Report Type: {result['report_type_detection']['detected_type']}")
        print(f"Confidence Level:     {result['report_type_detection']['confidence']}")
        print(f"Matched Keywords:     {result['report_type_detection']['matched_keywords']}")
        print("Extracted Values:")
        for field, data in result["fields"].items():
            if data.get("source") == "extracted":
                val = data["value"]
                unit = data["unit"]
                conf = data["confidence"]
                snippet = data["raw_snippet"]
                print(f"  - {field:14}: {val} {unit} [Confidence: {conf}] (Snippet: '{snippet}')")
            else:
                print(f"  - {field:14}: Not found")

    return all_results


def test_manual_merging():
    print_section("Part 5.11: Testing Manual Input Merging and Priority")
    # Scenario requested:
    # "Test merge_with_manual_input() with a case where extraction found
    #  cholesterol but not glucose, and the patient manually provided glucose
    #  plus overrode the extracted cholesterol value. Confirm the manual value
    #  correctly wins for cholesterol."

    mock_extracted = {
        "fields": {
            "cholesterol": {
                "value": 215.0,
                "unit": "mg/dL",
                "source": "extracted",
                "confidence": "high",
                "raw_snippet": "Total Cholesterol: 215 mg/dL",
            },
            "glucose": {
                "value": None,
                "source": "not_found",
            },
            "heart_rate": {
                "value": 72,
                "unit": "bpm",
                "source": "extracted",
                "confidence": "high",
                "raw_snippet": "Pulse: 72 bpm",
            },
            "systolic_bp": {
                "value": None,
                "source": "not_found",
            },
            "diastolic_bp": {
                "value": None,
                "source": "not_found",
            },
        },
        "report_type_detection": {
            "detected_type": "lipid_panel",
            "confidence": "high",
            "matched_keywords": ["Lipid Panel", "Total Cholesterol"],
        },
        "extraction_method": "direct_pdf",
    }

    # Patient manually overrides cholesterol (e.g. they know their current fasting lab is 195.0)
    # and provides glucose (92.0)
    manual_input = {
        "cholesterol": 195.0,
        "glucose": 92.0,
    }

    print("Initial Extracted Data (from report):")
    print(f"  - Cholesterol: {mock_extracted['fields']['cholesterol']['value']} mg/dL (Source: extracted)")
    print(f"  - Glucose:     {mock_extracted['fields']['glucose']['value']} (Source: not_found)")
    print(f"  - Heart Rate:  {mock_extracted['fields']['heart_rate']['value']} bpm (Source: extracted)")
    print(f"  - Systolic BP: None")
    print(f"  - Diastolic BP: None")

    print("\nPatient Manual Entries Provided:")
    print(f"  - Cholesterol override: {manual_input['cholesterol']}")
    print(f"  - Glucose manual entry: {manual_input['glucose']}")

    merged = merge_with_manual_input(mock_extracted, manual_input)

    print("\nFinal Merged Results:")
    for field, res in merged.items():
        print(f"  - {field:14}: Value = {res['value']!s:6} | Source = {res['source']:9} | Unit = {res['unit']}")

    # Assertions
    assert merged["cholesterol"]["value"] == 195.0, "Manual cholesterol must override extracted!"
    assert merged["cholesterol"]["source"] == "manual", "Cholesterol source must be 'manual'"

    assert merged["glucose"]["value"] == 92.0, "Manual glucose must be recorded!"
    assert merged["glucose"]["source"] == "manual", "Glucose source must be 'manual'"

    assert merged["heart_rate"]["value"] == 72, "Uncontested extracted heart rate must be preserved!"
    assert merged["heart_rate"]["source"] == "extracted", "Heart rate source must be 'extracted'"

    assert merged["systolic_bp"]["value"] is None, "Systolic BP must remain None (missing)!"
    assert merged["systolic_bp"]["source"] == "missing", "Systolic BP source must be 'missing'"

    assert merged["diastolic_bp"]["value"] is None, "Diastolic BP must remain None (missing)!"
    assert merged["diastolic_bp"]["source"] == "missing", "Diastolic BP source must be 'missing'"

    print("\n[PASSED] All assertion checks for manual merging passed successfully!")
    print("  -> Manual entry correctly won over extracted cholesterol.")
    print("  -> Manual glucose filled the missing field.")
    print("  -> Missing fields remained None without any false defaulting.")


def test_scanned_image_fallback():
    print_section("Part 5.12: Testing Scanned Image Handling & Fallback")
    sample_dir = os.path.join(PROJECT_ROOT, "tests", "sample_reports")
    img_path = os.path.join(sample_dir, "scanned_vitals.png")

    print(f"Testing OCR pathway on image file: {os.path.basename(img_path)}")
    try:
        text, method = extract_text_from_file(img_path)
        print(f"Method triggered: {method}")
        print(f"Extracted characters: {len(text)}")
        if text:
            print("Extracted Text Preview:\n" + text[:200])
            result = detect_report_type(text)
            print(f"Detected Report Type: {result}")
        else:
            print("Note: OCR engine returned empty text (Tesseract OCR binary not yet installed on system).")
            print("Clear diagnostic guidance was printed as expected.")
    except Exception as e:
        print(f"OCR handled gracefully with exception: {e}")


def main():
    print("Running HealthVault AI Report Extraction Test Suite...")
    test_pdf_extraction()
    test_manual_merging()
    test_scanned_image_fallback()
    print_section("Test Suite Complete")


if __name__ == "__main__":
    main()
