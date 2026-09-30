import os
import json
import numpy as np
import pandas as pd
from train import compute_multilabel_metrics, SUPERCLASSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved")
RECORDS_CSV = os.path.join(BASE_DIR, "data", "images", "image_records.csv")

test_probs = np.load(os.path.join(SAVED_DIR, "test_probs.npy"))
y_test = np.load(os.path.join(SAVED_DIR, "test_true.npy"))
df_records = pd.read_csv(RECORDS_CSV)
test_df = df_records[df_records["split"] == "test"].copy()

test_clean_indices = np.where(test_df["distortion_type"].values == "clean")[0]
test_distorted_indices = np.where(test_df["distortion_type"].values == "distorted")[0]

th_35 = {"NORM": 0.55, "MI": 0.35, "STTC": 0.55, "CD": 0.50, "HYP": 0.35}

m_overall = compute_multilabel_metrics(y_test, test_probs, th_35)
m_clean = compute_multilabel_metrics(y_test[test_clean_indices], test_probs[test_clean_indices], th_35)
m_dist = compute_multilabel_metrics(y_test[test_distorted_indices], test_probs[test_distorted_indices], th_35)

print("=" * 75)
print("CLEAN VS DISTORTED EVALUATION (TEST SET WITH NEW THRESHOLDS MI=0.35)")
print("=" * 75)
print(f"{'Class':<8} | {'Clean Recall':<14} | {'Distorted Recall':<18} | {'Clean F1':<12} | {'Distorted F1':<14}")
print("-" * 75)
for sc in SUPERCLASSES:
    c_rec = m_clean["per_class"][sc]["recall"]
    d_rec = m_dist["per_class"][sc]["recall"]
    c_f1 = m_clean["per_class"][sc]["f1"]
    d_f1 = m_dist["per_class"][sc]["f1"]
    print(f"{sc:<8} | {c_rec:<14.3f} | {d_rec:<18.3f} | {c_f1:<12.3f} | {d_f1:<14.3f}")
print("-" * 75)
print(f"Clean Subset Accuracy:     {m_clean['subset_accuracy']:.4f} | Hamming Loss: {m_clean['hamming_loss']:.4f}")
print(f"Distorted Subset Accuracy: {m_dist['subset_accuracy']:.4f} | Hamming Loss: {m_dist['hamming_loss']:.4f}")
print("=" * 75)

# Also check for MI=0.40
th_40 = {"NORM": 0.55, "MI": 0.40, "STTC": 0.55, "CD": 0.50, "HYP": 0.35}
m_clean_40 = compute_multilabel_metrics(y_test[test_clean_indices], test_probs[test_clean_indices], th_40)
m_dist_40 = compute_multilabel_metrics(y_test[test_distorted_indices], test_probs[test_distorted_indices], th_40)
print("\n" + "=" * 75)
print("CLEAN VS DISTORTED EVALUATION (TEST SET WITH NEW THRESHOLDS MI=0.40)")
print("=" * 75)
print(f"{'Class':<8} | {'Clean Recall':<14} | {'Distorted Recall':<18} | {'Clean F1':<12} | {'Distorted F1':<14}")
print("-" * 75)
for sc in SUPERCLASSES:
    c_rec = m_clean_40["per_class"][sc]["recall"]
    d_rec = m_dist_40["per_class"][sc]["recall"]
    c_f1 = m_clean_40["per_class"][sc]["f1"]
    d_f1 = m_dist_40["per_class"][sc]["f1"]
    print(f"{sc:<8} | {c_rec:<14.3f} | {d_rec:<18.3f} | {c_f1:<12.3f} | {d_f1:<14.3f}")
print("-" * 75)
print(f"Clean Subset Accuracy:     {m_clean_40['subset_accuracy']:.4f} | Hamming Loss: {m_clean_40['hamming_loss']:.4f}")
print(f"Distorted Subset Accuracy: {m_dist_40['subset_accuracy']:.4f} | Hamming Loss: {m_dist_40['hamming_loss']:.4f}")
print("=" * 75)
