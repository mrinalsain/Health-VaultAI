# ==============================================================================
# HealthVault AI - Cardiovascular Risk Inference Module
#
# This module provides a reusable, production-ready inference function:
# `predict_risk(input_dict)`
#
# Designed to be called by backend services, web APIs, or user interfaces.
# Handles missing/optional health parameters gracefully using calibrated
# population medians from the training cohort.
# ==============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

# Configure Keras backend to match the trained model environment
os.environ["KERAS_BACKEND"] = "torch"
import keras


# Directory containing trained artifacts
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")

# Module-level caches for model artifacts to optimize backend performance
_CACHED_MODEL = None
_CACHED_SCALER = None
_CACHED_MEDIANS = None
_CACHED_METADATA = None


def load_artifacts():
    """
    Load and cache trained artifacts (model, scaler, medians, metadata) from disk.
    
    Caching prevents redundant disk reads across repeated API requests.
    """
    global _CACHED_MODEL, _CACHED_SCALER, _CACHED_MEDIANS, _CACHED_METADATA
    
    if _CACHED_MODEL is None:
        metadata_path = os.path.join(SAVED_DIR, "model_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Model metadata not found at '{metadata_path}'. "
                "Please execute 'python risk_model/train.py' first to train and save the model."
            )
        with open(metadata_path, "r") as f:
            _CACHED_METADATA = json.load(f)
            
        medians_path = os.path.join(SAVED_DIR, "medians.json")
        with open(medians_path, "r") as f:
            _CACHED_MEDIANS = json.load(f)
            
        scaler_path = os.path.join(SAVED_DIR, "scaler.joblib")
        _CACHED_SCALER = joblib.load(scaler_path)
        
        model_file = _CACHED_METADATA["model_file"]
        model_path = os.path.join(SAVED_DIR, model_file)
        if _CACHED_METADATA.get("model_type") == "keras":
            _CACHED_MODEL = keras.models.load_model(model_path)
        else:
            _CACHED_MODEL = joblib.load(model_path)
            
    return _CACHED_MODEL, _CACHED_SCALER, _CACHED_MEDIANS, _CACHED_METADATA


def parse_boolean(val, field_name: str) -> int:
    """
    Safely converts flexible boolean/binary representations into 0 or 1.
    Accepts True/False, 1/0, 'yes'/'no', 'true'/'false'.
    """
    if isinstance(val, (bool, np.bool_)):
        return 1 if val else 0
    if isinstance(val, (int, float, np.number)):
        return 1 if val > 0 else 0
    if isinstance(val, str):
        clean = val.strip().lower()
        if clean in ("1", "true", "yes", "y"):
            return 1
        if clean in ("0", "false", "no", "n"):
            return 0
    raise ValueError(f"Field '{field_name}' must be a boolean or 0/1 indicator, received: {repr(val)}")


def parse_sex(val) -> int:
    """
    Converts sex / gender representation to dataset's 'male' binary column (1 = Male, 0 = Female).
    Accepts: 'male', 'female', 'm', 'f', 1, 0.
    """
    if isinstance(val, (int, float, np.number)):
        if int(val) in (0, 1):
            return int(val)
    elif isinstance(val, str):
        clean = val.strip().lower()
        if clean in ("m", "male", "1"):
            return 1
        if clean in ("f", "female", "0"):
            return 0
    raise ValueError(
        f"Invalid value for 'sex': {repr(val)}. Expected 'male', 'female', 'M', 'F', 1, or 0."
    )


def predict_risk(input_dict: dict) -> dict:
    """
    Predict 10-year cardiovascular disease risk from patient-reported health data.

    Required fields in input_dict (raise a clear error if any are missing,
    since these are always patient-known without clinical devices or lab tests):
        - age (int/float): Patient age in years.
        - sex (str/int): Biological sex ('male'/'female', 'M'/'F', or 1/0).
        - is_smoker (bool/int): Current smoking status.
        - has_hypertension (bool/int): Clinically diagnosed hypertension.
        - has_diabetes (bool/int): Diagnosed diabetes.

    Optional fields (if missing or None, populated using the training cohort median):
        - cigs_per_day (int/float): Cigarettes smoked per day (defaults to 0 if non-smoker,
          or cohort smoker median if smoker).
        - on_bp_meds (bool/int): Currently taking blood pressure medication (defaults to 0).
        - systolic_bp (float): Systolic blood pressure in mmHg (defaults to 128.0).
        - diastolic_bp (float): Diastolic blood pressure in mmHg (defaults to 82.0).
        - cholesterol (float): Total serum cholesterol in mg/dL (defaults to 234.0).
        - bmi (float): Body Mass Index in kg/m^2 (defaults to 25.38).
        - heart_rate (float): Resting heart rate in bpm (defaults to 75.0).
        - glucose (float): Fasting blood glucose in mg/dL (defaults to 78.0).

    Behavior & Processing Stages:
    1. Validates all required fields are provided and non-null.
    2. Identifies missing/None optional fields, replaces them with saved training medians,
       and logs which parameters were defaulted.
    3. Maps patient input keys to the model's exact 13 training feature names and order.
    4. Applies the saved StandardScaler to continuous numeric variables.
    5. Feeds the standardized input vector into the trained model to calculate risk probability.
    6. Compares probability against the calibrated medical screening threshold (0.47)
       derived via ROC/Youden's J optimization to maximize clinical sensitivity.
    7. Returns a structured clinical response dictionary.

    Returns:
        dict: {
            "risk_label": "Elevated" or "Normal",
            "risk_score": float (probability between 0.0000 and 1.0000),
            "threshold_used": float,
            "defaulted_fields": list of str (fields that were defaulted to median),
            "note": str (plain-language clinical guidance regarding data completeness)
        }
    """
    model, scaler, medians, metadata = load_artifacts()

    # Step 1: Validate Required Fields
    required_keys = ["age", "sex", "is_smoker", "has_hypertension", "has_diabetes"]
    missing_required = [k for k in required_keys if k not in input_dict or input_dict[k] is None]
    if missing_required:
        raise ValueError(
            f"Missing required patient fields: {missing_required}. "
            "These fields are mandatory because they are self-reported and known by the patient."
        )

    # Parse required inputs
    try:
        age_val = float(input_dict["age"])
        if not (18 <= age_val <= 120):
            raise ValueError(f"Age must be between 18 and 120, got: {age_val}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid age value: {input_dict['age']}. Must be a numeric value.") from e

    male_val = parse_sex(input_dict["sex"])
    is_smoker_val = parse_boolean(input_dict["is_smoker"], "is_smoker")
    has_hyp_val = parse_boolean(input_dict["has_hypertension"], "has_hypertension")
    has_diab_val = parse_boolean(input_dict["has_diabetes"], "has_diabetes")

    # Step 2: Handle Optional Fields with Fallback to Saved Medians
    defaulted_fields = []
    
    # Optional field: cigs_per_day
    if "cigs_per_day" in input_dict and input_dict["cigs_per_day"] is not None:
        cigs_val = float(input_dict["cigs_per_day"])
    else:
        # Clinical logic: If non-smoker, cigarettes per day is 0.
        # If smoker but quantity unspecified, use smoker median from training set.
        if is_smoker_val == 0:
            cigs_val = 0.0
        else:
            cigs_val = float(medians["cigs_per_day"])
            defaulted_fields.append("cigs_per_day")

    # Helper to resolve an optional numeric field
    def resolve_optional(key_name, median_key):
        if key_name in input_dict and input_dict[key_name] is not None:
            return float(input_dict[key_name])
        defaulted_fields.append(key_name)
        return float(medians[median_key])

    # Optional field: on_bp_meds
    if "on_bp_meds" in input_dict and input_dict["on_bp_meds"] is not None:
        bp_meds_val = float(parse_boolean(input_dict["on_bp_meds"], "on_bp_meds"))
    else:
        defaulted_fields.append("on_bp_meds")
        bp_meds_val = float(medians["on_bp_meds"])  # 0.0

    sys_bp_val = resolve_optional("systolic_bp", "systolic_bp")
    dia_bp_val = resolve_optional("diastolic_bp", "diastolic_bp")
    chol_val = resolve_optional("cholesterol", "cholesterol")
    bmi_val = resolve_optional("bmi", "bmi")
    hr_val = resolve_optional("heart_rate", "heart_rate")
    glucose_val = resolve_optional("glucose", "glucose")

    # Step 3: Map to Model's Feature Ordering
    feature_dict = {
        "age": age_val,
        "male": male_val,
        "currentSmoker": is_smoker_val,
        "cigsPerDay": cigs_val,
        "BPMeds": bp_meds_val,
        "prevalentHyp": has_hyp_val,
        "diabetes": has_diab_val,
        "totChol": chol_val,
        "sysBP": sys_bp_val,
        "diaBP": dia_bp_val,
        "BMI": bmi_val,
        "heartRate": hr_val,
        "glucose": glucose_val
    }

    feature_order = metadata["feature_order"]
    numeric_features = metadata["numeric_features"]

    # Assemble single-row DataFrame matching training layout
    df_patient = pd.DataFrame([feature_dict])[feature_order]

    # Step 4: Apply StandardScaler to Numeric Features
    df_patient_scaled = df_patient.copy()
    df_patient_scaled[numeric_features] = scaler.transform(df_patient[numeric_features])

    # Step 5: Execute Model Inference
    if metadata.get("model_type") == "keras":
        input_array = df_patient_scaled.values.astype(np.float32)
        raw_prob = float(model.predict(input_array, verbose=0)[0][0])
    else:
        raw_prob = float(model.predict_proba(df_patient_scaled)[:, 1][0])

    risk_score = round(raw_prob, 4)

    # Step 6: Apply Calibrated Medical Screening Threshold
    threshold = float(metadata.get("threshold", 0.47))
    risk_label = "Elevated" if risk_score >= threshold else "Normal"

    # Step 7: Construct Plain-Language Transparency Note
    if len(defaulted_fields) == 0:
        note = (
            "Complete patient profile provided. Risk estimate calculated using all specific "
            "clinical measurements."
        )
    else:
        defaulted_readable = [f.replace("_", " ") for f in defaulted_fields]
        primary_missing = defaulted_readable[0]
        note = (
            f"Estimate calculated with partial data ({len(defaulted_fields)} fields defaulted to population "
            f"medians: {', '.join(defaulted_readable)}). Adding your {primary_missing} will provide a "
            "more personalized clinical assessment."
        )

    return {
        "risk_label": risk_label,
        "risk_score": risk_score,
        "threshold_used": threshold,
        "defaulted_fields": defaulted_fields,
        "note": note
    }


# ==============================================================================
# Self-Verification Test Routine (Task 12)
# ==============================================================================
# if __name__ == "__main__":
#     print("=" * 80)
#     print("HEALTHVAULT AI - PREDICT_RISK FUNCTION VERIFICATION")
#     print("=" * 80)

#     # Test Case 1: Complete Patient Profile
#     # High-risk profile: 58-year-old male smoker with elevated BP and high cholesterol
#     complete_patient = {
#         "age": 58,
#         "sex": "male",
#         "is_smoker": True,
#         "cigs_per_day": 25,
#         "has_hypertension": True,
#         "on_bp_meds": False,
#         "systolic_bp": 158.0,
#         "diastolic_bp": 98.0,
#         "has_diabetes": False,
#         "cholesterol": 285.0,
#         "bmi": 29.4,
#         "heart_rate": 82.0,
#         "glucose": 95.0
#     }

#     print("\n--- TEST CASE 1: All Fields Provided (Full Clinical Profile) ---")
#     print("Input Profile:")
#     for k, v in complete_patient.items():
#         print(f"  {k:18s}: {v}")
    
#     result_1 = predict_risk(complete_patient)
#     print("\nModel Output:")
#     print(json.dumps(result_1, indent=4))

#     # Test Case 2: Partial Patient Profile (Missing Several Optional Fields)
#     # Young non-smoker patient entering data from home without recent blood tests
#     partial_patient = {
#         "age": 42,
#         "sex": "female",
#         "is_smoker": True,
#         "has_hypertension": True,
#         "has_diabetes": False,
#         # Missing optional fields: cholesterol, glucose, bmi, blood pressures, heart rate
#     }

#     print("\n" + "-" * 80)
#     print("--- TEST CASE 2: Partial Profile (Optional Fields Missing / Defaulted) ---")
#     print("Input Profile:")
#     for k, v in partial_patient.items():
#         print(f"  {k:18s}: {v}")

#     result_2 = predict_risk(partial_patient)
#     print("\nModel Output:")
#     print(json.dumps(result_2, indent=4))

#     # Test Case 3: Error Handling Demonstration (Missing Required Field)
#     print("\n" + "-" * 80)
#     print("--- TEST CASE 3: Error Handling (Missing Required Field 'age') ---")
#     invalid_patient = {
#         "sex": "male",
#         "is_smoker": True,
#         "has_hypertension": False,
#         "has_diabetes": False
#     }
#     try:
#         predict_risk(invalid_patient)
#     except ValueError as err:
#         print(f"Successfully caught expected ValueError:\n  --> '{err}'")

#     print("\n" + "=" * 80)
#     print("ALL VERIFICATION TESTS COMPLETED")
#     print("=" * 80)


# low_risk_patient = {
#    "age": 58,
#     "sex": "female",
#     "is_smoker": False,
#     "cigs_per_day": 0,
#     "has_hypertension": False,
#     "on_bp_meds": False,
#     "systolic_bp": 115.0,
#     "diastolic_bp": 75.0,
#     "has_diabetes": False,
#     "cholesterol": 170.0,
#     "bmi": 23.0,
#     "heart_rate": 70.0,
#     "glucose": 85.0
# }

# result1 = predict_risk(low_risk_patient)
# print(result1)
# print(json.dumps(result1, indent=4)) # asking about the json.dumps and also asking that the data was in the csv but how it got converted into the json 

for test_age in [30, 40, 50, 55, 58, 60, 65, 70]:
    profile = {
        "age": test_age, "sex": "female", "is_smoker": False, "cigs_per_day": 0,
        "has_hypertension": False, "on_bp_meds": False, "systolic_bp": 115.0,
        "diastolic_bp": 75.0, "has_diabetes": False, "cholesterol": 170.0,
        "bmi": 23.0, "heart_rate": 70.0, "glucose": 85.0
    }
    result = predict_risk(profile)
    print(f"Age {test_age}: risk_score = {result['risk_score']}, label = {result['risk_label']}")

# base = {
#     "age": 58, "sex": "female", "is_smoker": False, "cigs_per_day": 0,
#     "has_hypertension": False, "on_bp_meds": False, "systolic_bp": 115.0,
#     "diastolic_bp": 75.0, "has_diabetes": False, "cholesterol": 170.0,
#     "bmi": 23.0, "heart_rate": 70.0, "glucose": 85.0
# }
# print("Baseline:", predict_risk(base)['risk_score'])

# smoker_only = {**base, "is_smoker": True, "cigs_per_day": 30}
# print("Smoker only:", predict_risk(smoker_only)['risk_score'])

# hyper_only = {**base, "has_hypertension": True, "systolic_bp": 160.0, "diastolic_bp": 100.0}
# print("Hypertension only:", predict_risk(hyper_only)['risk_score'])

# chol_only = {**base, "cholesterol": 300.0}
# print("High cholesterol only:", predict_risk(chol_only)['risk_score'])

# baselines = [
#     {"age": 58, "sex": "female", "has_hypertension": False, "on_bp_meds": False,
#      "systolic_bp": 115.0, "diastolic_bp": 75.0, "has_diabetes": False,
#      "cholesterol": 170.0, "bmi": 23.0, "heart_rate": 70.0, "glucose": 85.0},
#     {"age": 45, "sex": "male", "has_hypertension": True, "on_bp_meds": False,
#      "systolic_bp": 140.0, "diastolic_bp": 90.0, "has_diabetes": False,
#      "cholesterol": 220.0, "bmi": 27.0, "heart_rate": 78.0, "glucose": 90.0},
#     {"age": 65, "sex": "male", "has_hypertension": False, "on_bp_meds": False,
#      "systolic_bp": 125.0, "diastolic_bp": 80.0, "has_diabetes": True,
#      "cholesterol": 190.0, "bmi": 25.0, "heart_rate": 72.0, "glucose": 110.0},
# ]

# for i, base in enumerate(baselines):
#     non_smoker = {**base, "is_smoker": False, "cigs_per_day": 0}
#     smoker = {**base, "is_smoker": True, "cigs_per_day": 30}
#     r1 = predict_risk(non_smoker)['risk_score']
#     r2 = predict_risk(smoker)['risk_score']
#     print(f"Baseline {i+1}: non-smoker={r1:.4f}, smoker={r2:.4f}, diff={r2-r1:+.4f}")