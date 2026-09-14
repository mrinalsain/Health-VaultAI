"""
extraction/report_type_patterns.py
==================================
Defines the keyword patterns used to detect the type of medical report.

Supported report types:
- "lipid_panel": Cholesterol, triglycerides, LDL, HDL profiles.
- "glucose_test": Blood sugar, fasting blood glucose, HbA1c tests.
- "cbc": Complete Blood Count (WBC, RBC, Hemoglobin, Platelets).
- "vitals": Clinical vital signs (Blood Pressure, Heart Rate/Pulse).
- "xray": Radiography / X-ray imaging reports (Chest, bone, plain film).
- "mri": Magnetic Resonance Imaging scans.
- "ct_scan": Computed Tomography / CAT scans.
- "sonography": Ultrasound and Doppler imaging studies.
- "ecg": Electrocardiogram (ECG/EKG) tracings.
- "echo": Echocardiogram / 2D Echo cardiac ultrasound reports.
- "prescription": Prescriptions, pharmacy orders, Rx dosages, directions.
- "biopsy": Biopsy, surgical pathology, and histopathology reports.
- "allergy_panel": Allergen testing, IgE profiles, skin prick tests.
- "hormone_panel": Endocrine tests (thyroid TSH/T3/T4, testosterone, estrogen).
- "liver_function": Hepatic function panel (LFT, SGPT, SGOT, bilirubin, ALT, AST).
- "kidney_function": Renal function panel (KFT, creatinine, BUN, urea).
- "urinalysis": Urine routine examination and microscopic analysis.
- "vaccination_record": Immunization records, vaccines, and booster history.
- "discharge_summary": Inpatient hospital discharge summaries and course.
- "consultation_note": Clinical outpatient encounter, specialist, and progress notes.
- "other": General medical documents without specific panel matches.
- "unknown": Default fallback when no meaningful keywords match or ambiguous.

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

# Note on non-exhaustive scope:
# This is not an exhaustive list — real-world medical reports include types
# not covered here, and this is intentional. detect_report_type() in extract.py
# will return "unknown" with low confidence when nothing matches well, rather
# than forcing an incorrect guess into one of these categories. This list exists
# to catch common clinical cases and reduce how often patients need to manually
# enter their report type, not to cover every possible specialty or document type.

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
    "xray": [
        "X-Ray",
        "Radiograph",
        "Chest X-Ray",
        "Bone X-Ray",
        "Radiography",
        "X-Ray Chest",
        "Plain Radiograph",
        "Roentgenogram",
        "X Ray",
    ],
    "mri": [
        "MRI",
        "Magnetic Resonance Imaging",
        "MRI Scan",
        "MR Imaging",
        "Brain MRI",
        "Spine MRI",
        "Magnetic Resonance",
    ],
    "ct_scan": [
        "CT Scan",
        "CAT Scan",
        "Computed Tomography",
        "CT Angiography",
        "Computed Axial Tomography",
        "High-Resolution CT",
        "HRCT",
    ],
    "sonography": [
        "Ultrasound",
        "Sonography",
        "USG",
        "Doppler",
        "Ultrasonography",
        "Color Doppler",
        "Ultrasound Scan",
        "Sonogram",
    ],
    "ecg": [
        "ECG",
        "Electrocardiogram",
        "EKG",
        "12-Lead ECG",
        "12-Lead EKG",
        "Electrocardiograph",
        "Electrocardiography",
    ],
    "echo": [
        "Echocardiogram",
        "Echo Report",
        "2D Echo",
        "Echocardiography",
        "Transthoracic Echocardiogram",
        "2D Echocardiography",
        "TTE",
        "Stress Echo",
    ],
    "prescription": [
        "Prescription",
        "Rx",
        "Sig",
        "Sig:",
        "Dosage",
        "Take as directed",
        "Dispense",
        "Refills",
        "Medication Order",
    ],
    "biopsy": [
        "Biopsy",
        "Histopathology",
        "Pathology Report",
        "Tissue Sample",
        "Histological Examination",
        "Histopathological Examination",
        "Surgical Pathology",
        "Fine Needle Aspiration",
    ],
    "allergy_panel": [
        "Allergy Panel",
        "Allergen Test",
        "IgE",
        "Skin Prick Test",
        "Allergy Test",
        "Specific IgE",
        "Total IgE",
        "Allergen Panel",
        "RAST",
    ],
    "hormone_panel": [
        "Hormone Panel",
        "Thyroid Panel",
        "TSH",
        "T3",
        "T4",
        "Testosterone",
        "Estrogen",
        "Free T3",
        "Free T4",
        "Estradiol",
        "Progesterone",
        "Prolactin",
        "Thyroid Function Test",
    ],
    "liver_function": [
        "Liver Function Test",
        "LFT",
        "SGPT",
        "SGOT",
        "Bilirubin",
        "ALT",
        "AST",
        "Alkaline Phosphatase",
        "Hepatic Function Panel",
        "Serum Bilirubin",
    ],
    "kidney_function": [
        "Kidney Function Test",
        "KFT",
        "Renal Panel",
        "Creatinine",
        "Urea",
        "Renal Function Test",
        "RFT",
        "Blood Urea Nitrogen",
        "Serum Creatinine",
        "eGFR",
    ],
    "urinalysis": [
        "Urinalysis",
        "Urine Test",
        "Urine Routine",
        "Urine Examination",
        "Urine Analysis",
        "Microscopic Urinalysis",
        "Urine R/M",
    ],
    "vaccination_record": [
        "Vaccination",
        "Immunization Record",
        "Vaccine",
        "Vaccination Record",
        "Immunization",
        "Vaccine Administration",
        "Immunization History",
        "Booster Dose",
    ],
    "discharge_summary": [
        "Discharge Summary",
        "Hospital Discharge",
        "Admission and Discharge",
        "Discharge Instructions",
        "Discharge Note",
        "Hospital Course",
        "Condition at Discharge",
    ],
    "consultation_note": [
        "Consultation Note",
        "Doctor's Note",
        "Clinical Note",
        "Progress Note",
        "Clinic Note",
        "Consultation Report",
        "Physician Note",
    ],
    "other": [
        "Medical Report",
        "Medical Record",
        "Clinical Report",
        "Health Summary",
        "Medical Assessment",
        "Clinical Assessment",
        "General Medical Examination",
        "Medical Evaluation",
        "Medical Certificate",
        "Medical Clearance",
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
