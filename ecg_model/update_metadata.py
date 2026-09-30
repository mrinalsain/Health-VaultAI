"""
update_metadata.py
Update metadata.json with the new per-class thresholds and updated test metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from train import compute_multilabel_metrics, SUPERCLASSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved")
METADATA_PATH = os.path.join(SAVED_DIR, "metadata.json")
RECORDS_CSV = os.path.join(BASE_DIR, "data", "images", "image_records.csv")

test_probs = np.load(os.path.join(SAVED_DIR, "test_probs.npy"))
y_test = np.load(os.path.join(SAVED_DIR, "test_true.npy"))
df_records = pd.read_csv(RECORDS_CSV)
test_df = df_records[df_records["split"] == "test"].copy()

test_clean_indices = np.where(test_df["distortion_type"].values == "clean")[0]
test_distorted_indices = np.where(test_df["distortion_type"].values == "distorted")[0]

new_thresholds = {
    "NORM": 0.55,
    "MI": 0.35,
    "STTC": 0.55,
    "CD": 0.50,
    "HYP": 0.35
}

overall_metrics = compute_multilabel_metrics(y_test, test_probs, new_thresholds)
clean_metrics = compute_multilabel_metrics(y_test[test_clean_indices], test_probs[test_clean_indices], new_thresholds)
distorted_metrics = compute_multilabel_metrics(y_test[test_distorted_indices], test_probs[test_distorted_indices], new_thresholds)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata["thresholds"] = new_thresholds
metadata["metrics_overall"] = overall_metrics
metadata["metrics_clean"] = clean_metrics
metadata["metrics_distorted"] = distorted_metrics

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("metadata.json updated successfully with new thresholds and test metrics!")
