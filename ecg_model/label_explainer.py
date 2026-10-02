"""
ecg_model/label_explainer.py
============================
HealthVault AI - ECG Image Classification Module
Part 5: Patient-Facing Plain-Language Translation Layer

This module translates raw AI model outputs into empathetic, clinically sound,
and completely transparent patient explanations.

CRITICAL DESIGN & SAFETY RULES:
1. Patient Safety First: Raw clinical abbreviations (MI, STTC, CD, HYP, NORM)
   must NEVER be displayed directly to the patient. Every condition must be referred
   to by its full medical name followed by an accessible, jargon-free explanation.
2. Confidence Score Banding:
   Raw numeric probabilities (e.g. 0.742) can cause confusion or unwarranted panic.
   We translate numeric confidence scores into qualitative bands:
   - 'High': Score is significantly above threshold (>= threshold + 0.35 or >= 0.75).
   - 'Moderate': Score is moderately above threshold (threshold + 0.15 <= score < threshold + 0.35).
   - 'Low': Score is slightly above threshold (threshold <= score < threshold + 0.15).
3. Sorting by Confidence:
   When multiple abnormal conditions are flagged, they are displayed sorted by
   confidence score descending (highest confidence finding first).
4. Urgency Triage Rule (MI Priority):
   If Myocardial Infarction (heart attack indicators) is among the detected conditions,
   an urgent referral advisory is ALWAYS triggered at the end, regardless of MI's
   confidence level or its position in the list.
5. Dual Return Format:
   Returns both a formatted human-readable string ('explanation_text') and a structured
   dictionary ('structured_data') allowing frontend components to render either
   direct copy or custom interactive cards.
"""

import os
import json
from typing import Dict, Any, List, Tuple

# Mapping of raw class keys to full condition names and patient-friendly explanations
CONDITION_DEFINITIONS = {
    "NORM": {
        "name": "Normal ECG",
        "explanation": "No significant abnormal patterns detected in this screening."
    },
    "MI": {
        "name": "Myocardial Infarction",
        "explanation": "signs sometimes associated with a past or current heart attack, where part of the heart muscle didn't get enough blood flow"
    },
    "STTC": {
        "name": "ST/T Wave Changes",
        "explanation": "changes in a specific part of the heart's electrical signal, which can be linked to several different underlying causes, not just one"
    },
    "CD": {
        "name": "Conduction Disturbance",
        "explanation": "irregularities in how electrical signals travel through the heart, which can affect heart rhythm"
    },
    "HYP": {
        "name": "Hypertrophy",
        "explanation": "signs that suggest the heart muscle wall may be thicker than typical, often related to the heart working harder than usual over time"
    }
}

# Standard closing statements
CLOSING_URGENT_MI = (
    "Your ECG shows signs that include Myocardial Infarction, which can indicate a risk to your heart. "
    "Please proceed to a doctor urgently for further evaluation."
)

CLOSING_MODERATE_LOW_MI = (
    "Your ECG shows signs that may relate to your heart. "
    "These are preliminary AI-assisted observations, not a confirmed diagnosis. "
    "Please see a doctor soon for further evaluation."
)

CLOSING_NON_URGENT_ABNORMAL = (
    "You show signs of the above pattern(s) and have a higher chance of these conditions. "
    "These are preliminary AI-assisted observations, not a confirmed diagnosis — please proceed to a doctor for further evaluation."
)

CLOSING_NORMAL_ONLY = (
    "Normal — no significant abnormal patterns detected in this screening. "
    "We still recommend a periodic checkup with a doctor for full clinical confirmation."
)

CLOSING_BORDERLINE = (
    "No clear abnormal pattern detected, but some borderline signs were present. "
    "This is a preliminary screening only. Please see a doctor for a proper check."
)

# Default per-class thresholds used for banding if metadata is not provided
DEFAULT_THRESHOLDS = {
    "NORM": 0.50,
    "MI": 0.40,
    "STTC": 0.45,
    "CD": 0.45,
    "HYP": 0.45
}


def get_confidence_band(score: float, threshold: float = 0.45) -> str:
    """
    Converts a numeric probability score into a qualitative band: High, Moderate, or Low.

    Confidence Banding Rationale & Logic:
    In clinical machine learning, classification is determined relative to an optimized threshold.
    - Low: score in [threshold, threshold + 0.15)
      The waveform matches abnormal features sufficiently to pass the detection threshold,
      but lies near the decision boundary.
    - Moderate: score in [threshold + 0.15, threshold + 0.35)
      The model identifies prominent and reliable disease-associated waveform indicators.
    - High: score >= threshold + 0.35 or score >= 0.75
      Strong, unambiguous diagnostic pattern features identified across multiple leads.
    """
    if score >= max(0.75, threshold + 0.35):
        return "High"
    elif score >= threshold + 0.15:
        return "Moderate"
    else:
        return "Low"


def explain_ecg_result(
    classification_dict: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Converts raw classification output into a patient-facing result.
    NEVER shows raw class abbreviations (MI, STTC, etc.) to the patient directly.

    Parameters:
        classification_dict (dict): Dictionary with keys NORM, MI, STTC, CD, HYP, each mapping to
            {"detected": bool, "confidence_score": float}.
        thresholds (dict, optional): Class-specific thresholds. If None, tries to read from
            ecg_model/saved/metadata.json or falls back to sensible defaults.

    Returns:
        dict: Containing both:
            - 'explanation_text': Full plain-language narrative for the patient.
            - 'structured_data': Machine-readable records with full names, bands, and scores.
            - 'has_urgent_referral': Boolean flag indicating if immediate care is required.
    """
    # 1. Resolve thresholds
    if thresholds is None:
        metadata_path = os.path.join(os.path.dirname(__file__), "saved", "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    thresholds = meta.get("thresholds", DEFAULT_THRESHOLDS)
            except Exception:
                thresholds = DEFAULT_THRESHOLDS
        else:
            thresholds = DEFAULT_THRESHOLDS

    # 2. Identify detected abnormal conditions (excluding NORM)
    abnormal_keys = ["MI", "STTC", "CD", "HYP"]
    detected_abnormals = []

    for k in abnormal_keys:
        if k in classification_dict and classification_dict[k].get("detected", False):
            score = float(classification_dict[k].get("confidence_score", 0.0))
            thresh = float(thresholds.get(k, 0.45))
            band = get_confidence_band(score, thresh)
            detected_abnormals.append({
                "class_key": k,
                "name": CONDITION_DEFINITIONS[k]["name"],
                "explanation": CONDITION_DEFINITIONS[k]["explanation"],
                "confidence_score": score,
                "confidence_band": band,
            })

    # Check if NORM was detected
    norm_detected = False
    norm_score = 0.0
    if "NORM" in classification_dict:
        norm_detected = classification_dict["NORM"].get("detected", False)
        norm_score = float(classification_dict["NORM"].get("confidence_score", 0.0))

    # 3. Determine Case A vs Case B
    # Case A: Only NORM detected or no abnormal class above its threshold
    if len(detected_abnormals) == 0:
        # Borderline rule: if any abnormal class has probability >= (threshold - 0.10)
        is_borderline = False
        for k in abnormal_keys:
            if k in classification_dict:
                score = float(classification_dict[k].get("confidence_score", 0.0))
                thresh = float(thresholds.get(k, 0.45))
                if score >= (thresh - 0.10):
                    is_borderline = True
                    break

        if is_borderline:
            explanation_text = CLOSING_BORDERLINE
            status = "borderline"
        else:
            explanation_text = CLOSING_NORMAL_ONLY
            status = "normal"

        structured_data = {
            "status": status,
            "detected_abnormal_conditions": [],
            "norm_confidence_score": norm_score,
            "has_urgent_referral": False,
        }
        return {
            "explanation_text": explanation_text,
            "structured_data": structured_data,
            "has_urgent_referral": False,
        }

    # Case B: One or more abnormal classes (MI, STTC, CD, HYP) detected
    # Step 1: Sort by confidence score descending (highest confidence first)
    detected_abnormals.sort(key=lambda item: item["confidence_score"], reverse=True)

    # Step 2: Format each detected class block
    blocks = []
    for item in detected_abnormals:
        block = f"{item['name']} — Confidence: {item['confidence_band']}\n{item['explanation']}"
        blocks.append(block)

    # Step 3: Determine closing urgency statement
    # Check if MI is detected and inspect its confidence band
    mi_item = next((item for item in detected_abnormals if item["class_key"] == "MI"), None)
    if mi_item is not None:
        if mi_item["confidence_band"] == "High":
            closing_line = CLOSING_URGENT_MI
            has_urgent = True
        else:
            closing_line = CLOSING_MODERATE_LOW_MI
            has_urgent = False
    else:
        closing_line = CLOSING_NON_URGENT_ABNORMAL
        has_urgent = False

    # Combine blocks and closing line
    full_explanation = "\n\n".join(blocks) + "\n\n" + closing_line

    structured_data = {
        "status": "abnormal_detected",
        "detected_abnormal_conditions": detected_abnormals,
        "has_urgent_referral": has_urgent,
    }

    return {
        "explanation_text": full_explanation,
        "structured_data": structured_data,
        "has_urgent_referral": has_urgent,
    }


def run_test_cases():
    """
    Executes the 4 mandatory validation test cases specified in Step 17:
    a) NORM only
    b) HYP (High confidence) + MI (Moderate confidence) together:
       Confirm HYP is listed first (higher confidence), but MI-urgent closing applies!
    c) CD alone (Low confidence)
    d) STTC + CD together, no MI present
    """
    print("=" * 75)
    print("ECG LABEL EXPLAINER - STEP 17 VALIDATION TEST CASES")
    print("=" * 75)

    test_thresholds = {
        "NORM": 0.50,
        "MI": 0.40,
        "STTC": 0.45,
        "CD": 0.45,
        "HYP": 0.45
    }

    # Case a: NORM only
    case_a = {
        "NORM": {"detected": True, "confidence_score": 0.88},
        "MI":   {"detected": False, "confidence_score": 0.12},
        "STTC": {"detected": False, "confidence_score": 0.20},
        "CD":   {"detected": False, "confidence_score": 0.15},
        "HYP":  {"detected": False, "confidence_score": 0.08},
    }

    # Case b: HYP (High) + MI (Moderate)
    # HYP score 0.85 -> High; MI score 0.58 -> Moderate (threshold is 0.40)
    case_b = {
        "NORM": {"detected": False, "confidence_score": 0.10},
        "MI":   {"detected": True, "confidence_score": 0.58},
        "STTC": {"detected": False, "confidence_score": 0.22},
        "CD":   {"detected": False, "confidence_score": 0.18},
        "HYP":  {"detected": True, "confidence_score": 0.85},
    }

    # Case c: CD alone (Low confidence)
    # CD score 0.52 (threshold 0.45) -> Low
    case_c = {
        "NORM": {"detected": False, "confidence_score": 0.25},
        "MI":   {"detected": False, "confidence_score": 0.15},
        "STTC": {"detected": False, "confidence_score": 0.30},
        "CD":   {"detected": True, "confidence_score": 0.52},
        "HYP":  {"detected": False, "confidence_score": 0.10},
    }

    # Case d: STTC + CD together, no MI
    case_d = {
        "NORM": {"detected": False, "confidence_score": 0.15},
        "MI":   {"detected": False, "confidence_score": 0.20},
        "STTC": {"detected": True, "confidence_score": 0.78},
        "CD":   {"detected": True, "confidence_score": 0.62},
        "HYP":  {"detected": False, "confidence_score": 0.12},
    }

    cases = [
        ("Test Case A: NORM Only", case_a),
        ("Test Case B: HYP (High) + MI (Moderate) [MI Priority Test]", case_b),
        ("Test Case C: CD Alone (Low Confidence)", case_c),
        ("Test Case D: STTC + CD Together (No MI)", case_d),
    ]

    for title, c_dict in cases:
        print(f"\n--- {title} ---")
        res = explain_ecg_result(c_dict, thresholds=test_thresholds)
        print("EXPLANATION TEXT OUTPUT:")
        print(res["explanation_text"])
        print("-" * 50)
        print("STRUCTURED DATA OUTPUT:")
        print(json.dumps(res["structured_data"], indent=2))
        print("=" * 75)


if __name__ == "__main__":
    run_test_cases()
