"""
ecg_model/evaluate_auc.py
=========================
Read-only evaluation script to compute per-class ROC-AUC and macro-AUC
on the PTB-XL ECG test split (overall, clean, and distorted).
Saves results to ecg_model/saved/auc_results.json.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch

# Constants
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
SUBSET_METADATA_PATH = os.path.join(DATA_DIR, "subset_metadata.csv")
RECORDS_CSV_PATH = os.path.join(IMAGES_DIR, "image_records.csv")
SAVED_DIR = os.path.join(BASE_DIR, "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "ecg_model.keras")
OUTPUT_JSON_PATH = os.path.join(SAVED_DIR, "auc_results.json")


def load_single_image(path: str, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Loads and normalizes an ECG image identical to train.py."""
    with Image.open(path) as img:
        img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        return np.array(img_rgb, dtype=np.float32) / 255.0


def main():
    torch.set_num_threads(os.cpu_count() or 8)

    # 1. Load test split from subset_metadata.csv & image catalog
    if not os.path.exists(SUBSET_METADATA_PATH):
        raise FileNotFoundError(f"Missing {SUBSET_METADATA_PATH}")

    subset_df = pd.read_csv(SUBSET_METADATA_PATH)
    test_meta = subset_df[subset_df["split"] == "test"].copy()

    if os.path.exists(RECORDS_CSV_PATH):
        records_df = pd.read_csv(RECORDS_CSV_PATH)
        test_df = records_df[(records_df["split"] == "test") & (records_df["ecg_id"].isin(test_meta["ecg_id"]))].copy().reset_index(drop=True)
    else:
        rows = []
        for _, row in test_meta.iterrows():
            eid = int(row["ecg_id"])
            labels = {sc: row[sc] for sc in SUPERCLASSES}
            rows.append({
                "image_path": os.path.join("test", f"{eid}_clean.jpg"),
                "ecg_id": eid,
                "distortion_type": "clean",
                **labels
            })
            rows.append({
                "image_path": os.path.join("test", f"{eid}_distorted.jpg"),
                "ecg_id": eid,
                "distortion_type": "distorted",
                **labels
            })
        test_df = pd.DataFrame(rows)

    paths = [os.path.join(IMAGES_DIR, p) for p in test_df["image_path"]]
    y_test = test_df[SUPERCLASSES].values.astype(np.float32)

    with ThreadPoolExecutor(max_workers=10) as executor:
        X_test = np.stack(list(executor.map(load_single_image, paths)), axis=0)

    # 2. Load model and predict once
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing trained model at {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    # 3 & 4. Compute ROC-AUC (all, clean, distorted)
    clean_mask = (test_df["distortion_type"].values == "clean")
    distorted_mask = (test_df["distortion_type"].values == "distorted")

    y_clean, probs_clean = y_test[clean_mask], test_probs[clean_mask]
    y_dist, probs_dist = y_test[distorted_mask], test_probs[distorted_mask]

    auc_all = []
    auc_clean = []
    auc_dist = []
    prevalence = []

    results = {
        "classes": {},
        "macro_avg": {},
        "total_test_samples": len(y_test),
        "clean_samples": int(clean_mask.sum()),
        "distorted_samples": int(distorted_mask.sum())
    }

    for i, sc in enumerate(SUPERCLASSES):
        a_all = float(roc_auc_score(y_test[:, i], test_probs[:, i]))
        a_cln = float(roc_auc_score(y_clean[:, i], probs_clean[:, i]))
        a_dst = float(roc_auc_score(y_dist[:, i], probs_dist[:, i]))
        prev = float(np.mean(y_test[:, i]))

        auc_all.append(a_all)
        auc_clean.append(a_cln)
        auc_dist.append(a_dst)
        prevalence.append(prev)

        results["classes"][sc] = {
            "auc_all": round(a_all, 4),
            "auc_clean": round(a_cln, 4),
            "auc_distorted": round(a_dst, 4),
            "test_prevalence": round(prev, 4),
            "positive_count": int(np.sum(y_test[:, i]))
        }

    macro_all = float(np.mean(auc_all))
    macro_cln = float(np.mean(auc_clean))
    macro_dst = float(np.mean(auc_dist))

    results["macro_avg"] = {
        "auc_all": round(macro_all, 4),
        "auc_clean": round(macro_cln, 4),
        "auc_distorted": round(macro_dst, 4)
    }

    # 5. Print table: class | AUC all | AUC clean | AUC distorted | test prevalence
    col_w = {"class": 9, "all": 9, "clean": 11, "dist": 15, "prev": 15}
    header = f"{'class':<{col_w['class']}} | {'AUC all':<{col_w['all']}} | {'AUC clean':<{col_w['clean']}} | {'AUC distorted':<{col_w['dist']}} | {'test prevalence':<{col_w['prev']}}"
    divider = f"{'-'*col_w['class']}-+-{'-'*col_w['all']}-+-{'-'*col_w['clean']}-+-{'-'*col_w['dist']}-+-{'-'*col_w['prev']}"
    
    print(header)
    print(divider)
    for i, sc in enumerate(SUPERCLASSES):
        prev_str = f"{prevalence[i]:.4f} ({prevalence[i]*100:.1f}%)"
        print(f"{sc:<{col_w['class']}} | {auc_all[i]:<{col_w['all']}.4f} | {auc_clean[i]:<{col_w['clean']}.4f} | {auc_dist[i]:<{col_w['dist']}.4f} | {prev_str:<{col_w['prev']}}")
    print(divider)
    print(f"{'Macro Avg':<{col_w['class']}} | {macro_all:<{col_w['all']}.4f} | {macro_cln:<{col_w['clean']}.4f} | {macro_dst:<{col_w['dist']}.4f} | {'-':<{col_w['prev']}}")

    # 6. Save results to JSON
    os.makedirs(SAVED_DIR, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
