"""
redo_threshold_search.py
Run validation inference (or load cached validation probabilities) and sweep thresholds
from 0.15 to 0.75 with step 0.05 for all 5 classes.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

os.environ["KERAS_BACKEND"] = "torch"
import torch
torch.set_num_threads(os.cpu_count() or 8)
import keras

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
IMAGE_SIZE = (96, 96)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
RECORDS_CSV = os.path.join(IMAGES_DIR, "image_records.csv")
SAVED_DIR = os.path.join(BASE_DIR, "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "ecg_model.keras")
METADATA_PATH = os.path.join(SAVED_DIR, "metadata.json")

VAL_PROBS_PATH = os.path.join(SAVED_DIR, "val_probs.npy")
VAL_TRUE_PATH = os.path.join(SAVED_DIR, "val_true.npy")
TEST_PROBS_PATH = os.path.join(SAVED_DIR, "test_probs.npy")
TEST_TRUE_PATH = os.path.join(SAVED_DIR, "test_true.npy")

def load_single_image(path: str, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        return arr

def load_split(df_split: pd.DataFrame, max_workers: int = 10):
    paths = [os.path.join(IMAGES_DIR, p) for p in df_split["image_path"]]
    labels = df_split[SUPERCLASSES].values.astype(np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images_list = list(executor.map(load_single_image, paths))
    X = np.stack(images_list, axis=0)
    y = labels
    return X, y

def get_val_probs():
    if os.path.exists(VAL_PROBS_PATH) and os.path.exists(VAL_TRUE_PATH):
        print(f"Loading cached validation probabilities from {VAL_PROBS_PATH}...")
        val_probs = np.load(VAL_PROBS_PATH)
        y_val = np.load(VAL_TRUE_PATH)
        return val_probs, y_val

    print("Computing validation probabilities using saved model...")
    df_records = pd.read_csv(RECORDS_CSV)
    val_df = df_records[df_records["split"] == "val"].copy()
    X_val, y_val = load_split(val_df)

    model = keras.models.load_model(MODEL_PATH, compile=False)
    val_probs = model.predict(X_val, batch_size=64, verbose=1)

    np.save(VAL_PROBS_PATH, val_probs)
    np.save(VAL_TRUE_PATH, y_val)
    print("Saved validation probabilities to cache.")
    return val_probs, y_val

def get_test_probs():
    if os.path.exists(TEST_PROBS_PATH) and os.path.exists(TEST_TRUE_PATH):
        test_probs = np.load(TEST_PROBS_PATH)
        y_test = np.load(TEST_TRUE_PATH)
        return test_probs, y_test

    print("Computing test probabilities using saved model...")
    df_records = pd.read_csv(RECORDS_CSV)
    test_df = df_records[df_records["split"] == "test"].copy()
    X_test, y_test = load_split(test_df)

    model = keras.models.load_model(MODEL_PATH, compile=False)
    test_probs = model.predict(X_test, batch_size=64, verbose=1)

    np.save(TEST_PROBS_PATH, test_probs)
    np.save(TEST_TRUE_PATH, y_test)
    return test_probs, y_test

def sweep():
    val_probs, y_val = get_val_probs()

    # Step of 0.05 from 0.15 to 0.75
    # [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    threshold_values = [round(0.15 + i * 0.05, 2) for i in range(13)]

    results = {}
    print(f"Total Validation Samples: {len(y_val)}")
    for idx, sc in enumerate(SUPERCLASSES):
        yt = y_val[:, idx].astype(int)
        yp = val_probs[:, idx]
        support = int(np.sum(yt))

        class_rows = []
        for t in threshold_values:
            pred = (yp >= t).astype(int)
            tp = int(np.sum((yt == 1) & (pred == 1)))
            fp = int(np.sum((yt == 0) & (pred == 1)))
            fn = int(np.sum((yt == 1) & (pred == 0)))
            tn = int(np.sum((yt == 0) & (pred == 0)))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            # F_beta with beta=1.5 and beta=2 for reference
            f1_5 = ((1 + 1.5**2) * p * r) / (1.5**2 * p + r) if (1.5**2 * p + r) > 0 else 0.0
            f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0

            class_rows.append({
                "threshold": t,
                "precision": p,
                "recall": r,
                "f1": f1,
                "f1.5": f1_5,
                "f2": f2,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total_flagged": tp + fp
            })
        results[sc] = {
            "support": support,
            "rows": class_rows
        }

    # Print markdown-formatted / structured tables
    for sc, data in results.items():
        print("\n" + "=" * 95)
        print(f"CLASS: {sc} (Validation Support: {data['support']} / {len(y_val)} positive samples, {data['support']/len(y_val)*100:.1f}%)")
        print("=" * 95)
        print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'F1.5':<10} | {'F2':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Flagged':<8}")
        print("-" * 95)
        for row in data["rows"]:
            print(f"{row['threshold']:<10.2f} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1']:<10.4f} | {row['f1.5']:<10.4f} | {row['f2']:<10.4f} | {row['tp']:<6} | {row['fp']:<6} | {row['fn']:<6} | {row['total_flagged']:<8}")

    # Save to a json for detailed programmatic inspection
    with open(os.path.join(SAVED_DIR, "threshold_sweep_val.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    sweep()
