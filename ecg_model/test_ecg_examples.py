"""
tests/test_ecg_examples.py
==========================
Test suite to evaluate 8 representative ECG test-split images across key diagnostic categories:
- 2 NORM-only (clean & distorted)
- 2 with MI (clean & distorted)
- 2 with multiple abnormal labels (clean & distorted)
- 2 with HYP or CD only (clean & distorted)

For each image:
1. Runs ecg_model/predict.py (model inference with saved thresholds).
2. Runs ecg_model/label_explainer.py (patient-facing explanation translation).
3. Evaluates output and flags any case where "Normal" is shown together with an abnormal finding.
4. Formats results into a clean evaluation table.
"""

import os
import sys
import json
import pandas as pd

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add workspace and ecg_model to path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECG_MODEL_DIR = os.path.join(WORKSPACE_DIR, "ecg_model")
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(0, ECG_MODEL_DIR)

from predict import classify_ecg_image
from label_explainer import explain_ecg_result

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMAL_CLASSES = ["MI", "STTC", "CD", "HYP"]
ABNORMAL_TERMS = [
    "Myocardial Infarction",
    "ST/T Wave Changes",
    "Conduction Disturbance",
    "Hypertrophy"
]

DATA_DIR = os.path.join(ECG_MODEL_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
RECORDS_CSV_PATH = os.path.join(IMAGES_DIR, "image_records.csv")
SAVED_DIR = os.path.join(ECG_MODEL_DIR, "saved")
METADATA_PATH = os.path.join(SAVED_DIR, "metadata.json")


def pick_test_examples(df_records: pd.DataFrame) -> list:
    """
    Picks 8 representative test images:
    - 2 NORM-only (1 clean, 1 distorted)
    - 2 with MI (1 clean, 1 distorted)
    - 2 with multiple abnormal labels (1 clean, 1 distorted)
    - 2 with HYP or CD only (1 clean, 1 distorted)
    """
    test_df = df_records[df_records["split"] == "test"].copy()
    used_ecg_ids = set()
    picks = []

    # 1. 2 NORM-only
    norm_c = test_df[
        (test_df["NORM"] == 1) &
        (test_df[ABNORMAL_CLASSES].sum(axis=1) == 0) &
        (test_df["distortion_type"] == "clean")
    ].iloc[0]
    used_ecg_ids.add(norm_c["ecg_id"])
    picks.append(norm_c)

    norm_d = test_df[
        (test_df["NORM"] == 1) &
        (test_df[ABNORMAL_CLASSES].sum(axis=1) == 0) &
        (test_df["distortion_type"] == "distorted") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(norm_d["ecg_id"])
    picks.append(norm_d)

    # 2. 2 with MI
    mi_c = test_df[
        (test_df["MI"] == 1) &
        (test_df["distortion_type"] == "clean") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(mi_c["ecg_id"])
    picks.append(mi_c)

    mi_d = test_df[
        (test_df["MI"] == 1) &
        (test_df["distortion_type"] == "distorted") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(mi_d["ecg_id"])
    picks.append(mi_d)

    # 3. 2 with multiple abnormal labels (>=2 of MI, STTC, CD, HYP)
    mul_c = test_df[
        (test_df[ABNORMAL_CLASSES].sum(axis=1) >= 2) &
        (test_df["distortion_type"] == "clean") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(mul_c["ecg_id"])
    picks.append(mul_c)

    mul_d = test_df[
        (test_df[ABNORMAL_CLASSES].sum(axis=1) >= 2) &
        (test_df["distortion_type"] == "distorted") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(mul_d["ecg_id"])
    picks.append(mul_d)

    # 4. 2 with HYP or CD only (1 clean CD, 1 distorted HYP)
    cd_c = test_df[
        (test_df["NORM"] == 0) &
        (test_df["MI"] == 0) &
        (test_df["STTC"] == 0) &
        (test_df["CD"] == 1) &
        (test_df["HYP"] == 0) &
        (test_df["distortion_type"] == "clean") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(cd_c["ecg_id"])
    picks.append(cd_c)

    hyp_d = test_df[
        (test_df["NORM"] == 0) &
        (test_df["MI"] == 0) &
        (test_df["STTC"] == 0) &
        (test_df["CD"] == 0) &
        (test_df["HYP"] == 1) &
        (test_df["distortion_type"] == "distorted") &
        (~test_df["ecg_id"].isin(used_ecg_ids))
    ].iloc[0]
    used_ecg_ids.add(hyp_d["ecg_id"])
    picks.append(hyp_d)

    return picks


def run_evaluation():
    # 1. Load saved thresholds from metadata.json
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Missing {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    thresholds = meta.get("thresholds", {})

    # 2. Load records catalog
    if not os.path.exists(RECORDS_CSV_PATH):
        raise FileNotFoundError(f"Missing {RECORDS_CSV_PATH}")
    df_records = pd.read_csv(RECORDS_CSV_PATH)

    sample_rows = pick_test_examples(df_records)

    table_rows = []
    conflict_detected_any = False

    for row in sample_rows:
        filename = os.path.basename(row["image_path"])
        full_path = os.path.join(IMAGES_DIR, row["image_path"])

        # True labels
        true_labels = [sc for sc in SUPERCLASSES if row[sc] == 1]
        true_labels_str = ", ".join(true_labels) if true_labels else "None"

        # Model prediction & patient explanation
        pred = classify_ecg_image(full_path)
        expl = explain_ecg_result(pred, thresholds=thresholds)

        # Probabilities
        probs_str = ", ".join([f"{sc}: {pred[sc]['confidence_score']:.3f}" for sc in SUPERCLASSES])

        # Flagged classes (above threshold)
        flagged_classes = [sc for sc in SUPERCLASSES if pred[sc]["detected"]]
        flagged_str = ", ".join(flagged_classes) if flagged_classes else "None"

        # Patient-facing text (flatten newlines for clean table display)
        explanation_raw = expl.get("explanation_text", "").strip()
        explanation_inline = explanation_raw.replace("\r\n", " ").replace("\n", " ")

        # Check for conflict: "Normal" shown together with an abnormal finding
        has_normal_in_text = "Normal" in explanation_raw
        has_abnormal_in_text = any(t in explanation_raw for t in ABNORMAL_TERMS)
        has_normal_flagged = "NORM" in flagged_classes
        has_abnormal_flagged = any(c in flagged_classes for c in ABNORMAL_CLASSES)

        is_conflict = (has_normal_in_text and has_abnormal_in_text) or (has_normal_flagged and has_abnormal_flagged)
        if is_conflict:
            conflict_detected_any = True
            conflict_str = "FLAGGED (Normal + Abnormal conflict)"
        else:
            conflict_str = "None (No conflict)"

        table_rows.append({
            "filename": filename,
            "true_labels": true_labels_str,
            "probs": probs_str,
            "flagged": flagged_str,
            "text": explanation_inline,
            "conflict": conflict_str
        })

    # Print markdown table
    print("| Filename | True Labels | Predicted Probabilities | Flagged Classes | Final Patient-Facing Text | Normal + Abnormal Conflict |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in table_rows:
        print(f"| {r['filename']} | {r['true_labels']} | {r['probs']} | {r['flagged']} | {r['text']} | {r['conflict']} |")

    if conflict_detected_any:
        print("\nWARNING: One or more test examples showed 'Normal' together with an abnormal finding!", file=sys.stderr)


if __name__ == "__main__":
    run_evaluation()
