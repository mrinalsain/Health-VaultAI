"""
extraction/field_patterns.py
============================
Regex patterns and value parsing for target clinical health metrics:
1. Total Cholesterol (mg/dL)
2. Blood Glucose (mg/dL)
3. Heart Rate (bpm)
4. Systolic Blood Pressure (mmHg)
5. Diastolic Blood Pressure (mmHg)

Medical Concept: Unit Variations (mg/dL vs mmol/L)
--------------------------------------------------
In different regions and laboratories, biochemical values are measured in different units:
- US / Standard American labs commonly use milligrams per deciliter (mg/dL).
- UK, Canada, Australia, and European labs frequently use millimoles per liter (mmol/L).

To maintain consistency for downstream analysis:
- Cholesterol: 1 mmol/L = 38.67 mg/dL
- Glucose: 1 mmol/L = 18.018 mg/dL

Crucial Clinical Rule:
If a unit is missing or ambiguous, we DO NOT silently assume a unit or blindly convert.
Instead, we flag the unit as 'unspecified', lower the extraction confidence to 'medium',
and preserve the raw value as-is for human review.
"""

import re
from typing import Any, Dict, Optional, Tuple

# Standard conversion constants from mmol/L to mg/dL:
CHOLESTEROL_MMOL_TO_MGDL = 38.67
GLUCOSE_MMOL_TO_MGDL = 18.018

# Regex pattern fragments for units:
# Matches variations like: mg/dL, mg/dl, mg/100ml, mmol/L, mmol/l
UNIT_PATTERN = r"(?P<unit>mg\s*/\s*d[lL]|mmol\s*/\s*[lL]|mg\s*%)"

# Target fields to extract:
TARGET_FIELDS = [
    "cholesterol",
    "glucose",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
]


def _normalize_unit(raw_unit: Optional[str]) -> Optional[str]:
    """
    Standardizes raw unit strings into either 'mg/dL' or 'mmol/L'.
    Returns None if no recognizable unit was found.
    """
    if not raw_unit:
        return None
    cleaned = raw_unit.strip().lower().replace(" ", "")
    if "mmol" in cleaned:
        return "mmol/L"
    if "mg" in cleaned:
        return "mg/dL"
    return None


def extract_cholesterol(text: str) -> Dict[str, Any]:
    """
    Extracts Total Cholesterol from report text.
    Handles phrasings such as:
    - 'Total Cholesterol: 215 mg/dL'
    - 'Cholesterol, Total: 5.5 mmol/L'
    - 'TC: 190'
    - 'Cholesterol: 200'

    Converts mmol/L to mg/dL if needed.
    """
    patterns = [
        # 1. Explicit 'Total Cholesterol' or 'Cholesterol, Total'
        rf"(?:Total\s+Cholesterol|Cholesterol\s*,\s*Total)\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
        # 2. Abbreviated 'TC' followed by number and optional unit
        rf"\bTC\b\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
        # 3. General 'Cholesterol' (avoiding HDL/LDL prefixes)
        rf"(?<!HDL\s)(?<!LDL\s)(?<!Non-HDL\s)\bCholesterol\b\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            raw_val = float(match.group("val"))
            raw_unit = match.group("unit") if "unit" in match.groupdict() else None
            normalized_unit = _normalize_unit(raw_unit)

            raw_snippet = match.group(0).strip()

            if normalized_unit == "mmol/L":
                # Converted from mmol/L to mg/dL
                converted_val = round(raw_val * CHOLESTEROL_MMOL_TO_MGDL, 1)
                return {
                    "value": converted_val,
                    "unit": "mg/dL",
                    "source": "extracted",
                    "confidence": "high",
                    "raw_snippet": f"{raw_snippet} (converted from {raw_val} mmol/L)",
                }
            elif normalized_unit == "mg/dL":
                return {
                    "value": round(raw_val, 1),
                    "unit": "mg/dL",
                    "source": "extracted",
                    "confidence": "high",
                    "raw_snippet": raw_snippet,
                }
            else:
                # Unit was missing or ambiguous: do not guess!
                return {
                    "value": round(raw_val, 1),
                    "unit": "unspecified",
                    "source": "extracted",
                    "confidence": "medium",
                    "raw_snippet": f"{raw_snippet} [Warning: Missing or ambiguous unit]",
                }

    return {"value": None, "source": "not_found"}


def extract_glucose(text: str) -> Dict[str, Any]:
    """
    Extracts Blood Glucose from report text.
    Handles phrasings such as:
    - 'Fasting Glucose: 95 mg/dL'
    - 'Fasting Blood Sugar: 5.2 mmol/L'
    - 'Blood Sugar: 110'
    - 'Glucose: 90'
    - 'FBS: 102'

    Converts mmol/L to mg/dL if needed.
    """
    patterns = [
        # 1. Fasting Glucose / Blood Sugar / Fasting Blood Sugar / Blood Glucose
        rf"(?:Fasting\s+Glucose|Fasting\s+Blood\s+Sugar|Blood\s+Glucose|Blood\s+Sugar)\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
        # 2. Acronym 'FBS'
        rf"\bFBS\b\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
        # 3. Plain 'Glucose'
        rf"\bGlucose\b\s*[:=\-]?\s*(?P<val>\d+(?:\.\d+)?)\s*{UNIT_PATTERN}?",
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            raw_val = float(match.group("val"))
            raw_unit = match.group("unit") if "unit" in match.groupdict() else None
            normalized_unit = _normalize_unit(raw_unit)

            raw_snippet = match.group(0).strip()

            if normalized_unit == "mmol/L":
                converted_val = round(raw_val * GLUCOSE_MMOL_TO_MGDL, 1)
                return {
                    "value": converted_val,
                    "unit": "mg/dL",
                    "source": "extracted",
                    "confidence": "high",
                    "raw_snippet": f"{raw_snippet} (converted from {raw_val} mmol/L)",
                }
            elif normalized_unit == "mg/dL":
                return {
                    "value": round(raw_val, 1),
                    "unit": "mg/dL",
                    "source": "extracted",
                    "confidence": "high",
                    "raw_snippet": raw_snippet,
                }
            else:
                # Unit was missing or ambiguous
                return {
                    "value": round(raw_val, 1),
                    "unit": "unspecified",
                    "source": "extracted",
                    "confidence": "medium",
                    "raw_snippet": f"{raw_snippet} [Warning: Missing or ambiguous unit]",
                }

    return {"value": None, "source": "not_found"}


def extract_heart_rate(text: str) -> Dict[str, Any]:
    """
    Extracts Heart Rate / Pulse from report text.
    Handles phrasings such as:
    - 'Heart Rate: 72 bpm'
    - 'Pulse: 68 beats/min'
    - 'Pulse Rate: 75'
    - 'HR: 80'
    """
    patterns = [
        # Explicit Heart Rate or Pulse Rate
        r"(?:Heart\s+Rate|Pulse\s+Rate)\s*[:=\-]?\s*(?P<val>\d{2,3})\s*(?:bpm|beats\s*/\s*min(?:ute)?)?",
        # Pulse
        r"\bPulse\b\s*[:=\-]?\s*(?P<val>\d{2,3})\s*(?:bpm|beats\s*/\s*min(?:ute)?)?",
        # HR
        r"\bHR\b\s*[:=\-]?\s*(?P<val>\d{2,3})\s*(?:bpm)?",
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            val = int(match.group("val"))
            # Plausibility check: physiological heart rate is typically 30 - 250 bpm
            confidence = "high" if 35 <= val <= 220 else "medium"
            return {
                "value": val,
                "unit": "bpm",
                "source": "extracted",
                "confidence": confidence,
                "raw_snippet": match.group(0).strip(),
            }

    return {"value": None, "source": "not_found"}


def extract_blood_pressure(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extracts Systolic and Diastolic Blood Pressure from report text.
    Blood pressure commonly appears as:
    - Combined format: 'Blood Pressure: 120/80 mmHg', 'BP: 135/85', '120/80 mmHg'
    - Separate format: 'Systolic: 120 mmHg', 'Diastolic: 80 mmHg'

    Returns:
        Tuple of (systolic_dict, diastolic_dict)
    """
    # 1. Check for combined slashed format: e.g. "BP: 120/80 mmHg" or "120/80"
    slashed_patterns = [
        r"(?:Blood\s+Pressure|BP)\s*[:=\-]?\s*(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})\s*(?:mm\s*Hg)?",
        r"\b(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})\s*mm\s*Hg\b",
    ]

    for pat in slashed_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            sys_val = int(match.group("sys"))
            dia_val = int(match.group("dia"))
            snippet = match.group(0).strip()

            # Medical plausibility: systolic is typically 70-240, diastolic 40-150, sys > dia
            is_plausible = (60 <= sys_val <= 260) and (40 <= dia_val <= 160) and (sys_val > dia_val)
            conf = "high" if is_plausible else "medium"

            sys_dict = {
                "value": sys_val,
                "unit": "mmHg",
                "source": "extracted",
                "confidence": conf,
                "raw_snippet": snippet,
            }
            dia_dict = {
                "value": dia_val,
                "unit": "mmHg",
                "source": "extracted",
                "confidence": conf,
                "raw_snippet": snippet,
            }
            return sys_dict, dia_dict

    # 2. Check for separate systolic and diastolic mentions:
    sys_match = re.search(
        r"(?:Systolic(?:\s+BP)?)\s*[:=\-]?\s*(?P<sys>\d{2,3})\s*(?:mm\s*Hg)?",
        text,
        re.IGNORECASE,
    )
    dia_match = re.search(
        r"(?:Diastolic(?:\s+BP)?)\s*[:=\-]?\s*(?P<dia>\d{2,3})\s*(?:mm\s*Hg)?",
        text,
        re.IGNORECASE,
    )

    if sys_match:
        sys_val = int(sys_match.group("sys"))
        sys_dict = {
            "value": sys_val,
            "unit": "mmHg",
            "source": "extracted",
            "confidence": "high" if 60 <= sys_val <= 260 else "medium",
            "raw_snippet": sys_match.group(0).strip(),
        }
    else:
        sys_dict = {"value": None, "source": "not_found"}

    if dia_match:
        dia_val = int(dia_match.group("dia"))
        dia_dict = {
            "value": dia_val,
            "unit": "mmHg",
            "source": "extracted",
            "confidence": "high" if 40 <= dia_val <= 160 else "medium",
            "raw_snippet": dia_match.group(0).strip(),
        }
    else:
        dia_dict = {"value": None, "source": "not_found"}

    return sys_dict, dia_dict


def extract_all_fields(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Executes regex extraction for all target clinical values against the report text.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing extraction results for:
        - cholesterol
        - glucose
        - heart_rate
        - systolic_bp
        - diastolic_bp
    """
    sys_bp, dia_bp = extract_blood_pressure(text)

    return {
        "cholesterol": extract_cholesterol(text),
        "glucose": extract_glucose(text),
        "heart_rate": extract_heart_rate(text),
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
    }
