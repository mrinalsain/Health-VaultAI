"""
tests/test_new_report_types.py
==============================
Automated test suite verifying the expanded report-type detection logic
in report_type_patterns.py and detect_report_type() in extract.py.

Tests:
1. Newly added types:
   - xray (X-ray / Radiograph / Chest X-Ray)
   - prescription (Rx / Sig / Dosage / Take as directed)
   - biopsy (Histopathology / Pathology Report / Tissue Sample)
   - hormone_panel (Thyroid Panel / TSH / Free T3 / Free T4 / Testosterone / Estrogen)
   - other (General medical document with no specific lab panel)
2. Genuinely ambiguous / non-medical document fallback:
   - Plain text tenancy lease agreement -> "unknown" (confidence: "unknown")
3. Ambiguous tie fallback:
   - Document with equal 1-keyword match across two distinct categories -> "unknown" (confidence: "low")
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extraction.extract import detect_report_type


TEST_CASES = [
    {
        "id": "xray_sample",
        "description": "Chest X-Ray Imaging Report",
        "text": """
DEPARTMENT OF RADIOLOGY & IMAGING SCIENCES
PATIENT: Michael Scott | AGE: 45 | DATE: 2026-09-10
EXAMINATION: Chest X-Ray PA and Lateral Views

CLINICAL INDICATION: Chronic cough, evaluate for infiltrates.

FINDINGS:
Lungs are clear bilaterally. No focal alveolar consolidation, pneumothorax, or pleural effusion.
Cardiomediastinal silhouette and pulmonary vascularity are within normal limits.
Visualized osseous structures demonstrate no acute fracture on this radiograph.

IMPRESSION:
Normal Chest X-Ray. No acute cardiopulmonary abnormality.
""",
        "expected_type": "xray",
        "expected_confidence": "high",
    },
    {
        "id": "prescription_sample",
        "description": "Outpatient Clinic Prescription",
        "text": """
METROPOLITAN HEALTH CLINIC - OUTPATIENT PRESCRIPTION
DATE: 2026-09-12
PATIENT: Sarah Connor | DOB: 1985-05-14
PRESCRIBER: Dr. Emily Hayes, MD | LIC: NY-882109

Rx: Amoxicillin-Potassium Clavulanate 875-125 mg
Dosage: 875 mg oral tablet
Sig: Take 1 tablet by mouth twice daily with meals for 10 days
Dispense: 20 (twenty) tablets
Refills: 0 (zero)

Instructions: Take as directed. Complete the entire course of medication.
""",
        "expected_type": "prescription",
        "expected_confidence": "high",
    },
    {
        "id": "biopsy_sample",
        "description": "Surgical Pathology & Histopathology Report",
        "text": """
PATHOLOGY & HISTOPATHOLOGY LABORATORY REPORT
SPECIMEN: Skin lesion biopsy, right upper back
PATIENT ID: PATH-2026-8812 | DATE OF SERVICE: 2026-09-08

CLINICAL HISTORY: Pigmented lesion, irregular borders.
PROCEDURE: Punch biopsy of skin lesion.

GROSS DESCRIPTION:
Received in formalin labeled with patient name is a tissue sample measuring 0.5 x 0.4 x 0.3 cm.

HISTOPATHOLOGICAL EXAMINATION:
Sections reveal stratified squamous epithelium with benign dermal melanocytic proliferation.
No cellular atypia, architectural disorder, or invasive malignancy identified.
Surgical pathology margins are free of lesion.

FINAL PATHOLOGY REPORT:
Skin punch biopsy, right upper back: Benign intradermal melanocytic nevus.
""",
        "expected_type": "biopsy",
        "expected_confidence": "high",
    },
    {
        "id": "hormone_panel_sample",
        "description": "Endocrine Laboratory Hormone & Thyroid Panel",
        "text": """
LABCORP SPECIALTY TESTING - COMPREHENSIVE HORMONE PANEL
PATIENT: David Miller | DOB: 1982-11-20 | COLLECTION DATE: 2026-09-01
ORDERING PHYSICIAN: Dr. Robert Vance, Endocrinology

TEST NAME                      RESULT      REFERENCE INTERVAL    UNITS
----------------------------------------------------------------------
Thyroid Panel:
  TSH                          2.10        0.450 - 4.500         uIU/mL
  Free T3                      3.2         2.0 - 4.4             pg/mL
  Free T4                      1.35        0.82 - 1.77           ng/dL
Reproductive Hormones:
  Total Testosterone           612         264 - 916             ng/dL
  Estrogen (Estradiol)         26.4        7.6 - 42.6            pg/mL

COMMENT: Thyroid function test results and testosterone levels are within normal physiological ranges.
""",
        "expected_type": "hormone_panel",
        "expected_confidence": "high",
    },
    {
        "id": "other_medical_sample",
        "description": "General Occupational Medical Evaluation ('other')",
        "text": """
OCCUPATIONAL HEALTH SERVICES
PATIENT: Kevin Malone | ID: OH-5541 | DATE: 2026-09-03

GENERAL MEDICAL EXAMINATION & HEALTH SUMMARY
Reason for visit: Annual workplace fitness and medical clearance assessment.

CLINICAL EVALUATION:
Subject is asymptomatic with no active physical limitations or functional impairments.
Musculoskeletal and neurological exams within normal limits.

CONCLUSION:
Medical Clearance granted for unrestricted occupational duties.
Official Medical Certificate issued to employee health records.
""",
        "expected_type": "other",
        "expected_confidence": "high",
    },
    {
        "id": "non_medical_sample",
        "description": "Plain Non-Medical Text (Tenancy Lease Agreement)",
        "text": """
RESIDENTIAL APARTMENT LEASE AGREEMENT
This agreement is made on September 1, 2026, by and between Oakridge Apartments (Landlord)
and Alex Johnson (Tenant).

TERMS AND CONDITIONS:
1. Rent: The monthly rental amount of $1,650 is payable in advance on the first day of each month.
2. Term: The initial lease term commences on October 1, 2026, and expires on September 30, 2027.
3. Security Deposit: Tenant shall deposit $1,650 as security for performance of obligations.
4. Maintenance: Tenant shall keep the premises in good order and clean condition.
5. Utilities: Electricity and internet shall be paid directly by the tenant.
""",
        "expected_type": "unknown",
        "expected_confidence": "unknown",
    },
    {
        "id": "ambiguous_tie_sample",
        "description": "Ambiguous Encounter with Tied Matches (Vitals vs Ultrasound)",
        "text": """
CLINICAL MEMORANDUM
Patient was seen in clinic today.
Blood Pressure was checked during triage.
Reviewed recent Ultrasound from external clinic.
Follow up as needed.
""",
        "expected_type": "unknown",
        "expected_confidence": "low",
    },
]


def run_tests():
    print("=" * 75)
    print(" HEALTHVAULT AI - EXPANDED REPORT TYPE DETECTION TEST SUITE")
    print("=" * 75)

    all_passed = True

    for tc in TEST_CASES:
        print(f"\n[Test Case] {tc['id']}: {tc['description']}")
        result = detect_report_type(tc["text"])

        detected_type = result["detected_type"]
        confidence = result["confidence"]
        matched_kws = result["matched_keywords"]

        print(f"  -> Detected Type   : {detected_type}")
        print(f"  -> Confidence Level : {confidence}")
        print(f"  -> Matched Keywords : {matched_kws}")

        # Verification
        type_match = detected_type == tc["expected_type"]
        conf_match = confidence == tc["expected_confidence"]

        if type_match and conf_match:
            print("  -> Status           : PASS [OK]")
        else:
            print(f"  -> Status           : FAIL [Expected {tc['expected_type']} ({tc['expected_confidence']})]")
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print(" ALL 7 TEST CASES PASSED SUCCESSFULLY!")
    else:
        print(" SOME TEST CASES FAILED!")
    print("=" * 75)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
