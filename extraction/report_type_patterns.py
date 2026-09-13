"""
extraction/report_type_patterns.py
==================================
Defines the keyword patterns used to detect the type of medical report.

Supported report types:
- "lipid_panel": Cholesterol, triglycerides, LDL, HDL profiles.
- "glucose_test": Blood sugar, fasting blood glucose, HbA1c tests.
- "cbc": Complete Blood Count (WBC, RBC, Hemoglobin, Platelets).
- "vitals": Clinical vital signs (Blood Pressure, Heart Rate/Pulse).
- "unknown": Default fallback when no meaningful keywords match.

Design Note: Why keyword sets instead of a Machine Learning Classifier?
------------------------------------------------------------------------
1. Explainability: Clinicians and patients need to understand *why* a document
   was categorized as a certain report type. Keyword matching lets us report the
   exact words found in the document (e.g. "Matched keywords: LDL, HDL").
2. No Overfitting or Hallucination: A statistical or ML classifier trained on a
   small set of lab reports can easily make unpredictable guesses on unseen formats.
   Keyword counting is deterministic and never hallucinates categories.
3. Lightweight & Zero Extra Overhead: No large model weights or GPU requirements.
"""

import re
from typing import Dict, List

# Dictionary mapping report types to their signature medical keywords.
# Each entry is a list of strings commonly found on lab reports.
REPORT_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "lipid_panel": [
        "Lipid Panel",
        "Lipid Profile",
        "Total Cholesterol",
        "LDL",
        "HDL",
        "Triglycerides",
        "Cholesterol",
    ],
    "glucose_test": [
        "Fasting Glucose",
        "Blood Sugar",
        "HbA1c",
        "Glucose",
        "Fasting Blood Sugar",
        "Oral Glucose",
    ],
    "cbc": [
        "Complete Blood Count",
        "CBC",
        "WBC",
        "RBC",
        "Hemoglobin",
        "Platelet",
        "Hematocrit",
        "Platelets",
    ],
    "vitals": [
        "Blood Pressure",
        "Heart Rate",
        "Pulse",
        "Vitals",
        "Vital Signs",
        "Pulse Rate",
    ],
}


def find_matched_keywords(text: str) -> Dict[str, List[str]]:
    """
    Scans the extracted text for signature keywords associated with each report type.

    Why word boundaries (\\b) are important:
    Short medical acronyms like 'CBC', 'WBC', 'RBC', 'LDL', 'HDL' must only match
    whole words. Without word boundaries, a word like 'building' could trigger false
    matches.

    Parameters:
        text (str): The raw text extracted from the report.

    Returns:
        Dict[str, List[str]]: Mapping from report type name to the list of unique
                              keywords that were found in the text.
    """
    matches_per_type: Dict[str, List[str]] = {}

    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():
        matched_for_this_type: List[str] = []
        for kw in keywords:
            # Escape regex special characters in the keyword, and enforce word boundaries.
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text, re.IGNORECASE):
                matched_for_this_type.append(kw)

        matches_per_type[report_type] = matched_for_this_type

    return matches_per_type
