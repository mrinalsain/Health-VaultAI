"""
ecg_model/predict.py
====================
HealthVault AI - ECG Image Classification Module
Part 4: Standalone Inference Function

This module provides the core inference entry point:
    classify_ecg_image(image_path: str) -> dict

How it works:
1. Validates that the input image path exists on disk.
2. Preprocesses the image:
   - Opens via PIL and ensures 3-channel RGB format.
   - Resizes to the model's expected input resolution (128x128).
   - Normalizes pixel values from [0, 255] to [0.0, 1.0].
3. Loads the trained multi-label CNN model and metadata:
   - Caches both in memory so subsequent calls avoid repeated disk I/O.
   - Sets KERAS_BACKEND='torch' to ensure native PyTorch execution.
4. Generates predictions:
   - Passes the preprocessed image through the CNN.
   - Receives 5 independent sigmoid probabilities [0.0, 1.0].
5. Applies learned per-class decision thresholds:
   - Compares each probability against the class-specific threshold stored in metadata.json.
   - Sets 'detected': True if probability >= threshold, else False.
6. Returns the standardized classification dictionary.
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Any

# Ensure Keras runs with the PyTorch backend on Windows Python 3.14
os.environ["KERAS_BACKEND"] = "torch"
import keras

# Define module paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "ecg_model.keras")
METADATA_PATH = os.path.join(SAVED_DIR, "metadata.json")

# Default superclasses and fallback thresholds in case metadata is not yet generated
DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DEFAULT_THRESHOLDS = {
    "NORM": 0.50,
    "MI": 0.40,
    "STTC": 0.45,
    "CD": 0.45,
    "HYP": 0.45
}
DEFAULT_INPUT_SHAPE = (128, 128)

# Module-level cache to keep model and metadata resident in memory across calls
_CACHED_MODEL = None
_CACHED_METADATA = None


def load_model_and_metadata():
    """
    Loads and caches the trained Keras CNN and metadata dictionary.
    Uses compile=False since only forward inference is performed, eliminating
    the need to deserialize custom training loss functions.
    """
    global _CACHED_MODEL, _CACHED_METADATA

    if _CACHED_MODEL is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL, _CACHED_METADATA

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained ECG model not found at {MODEL_PATH}. "
            "Please run 'python ecg_model/train.py' first."
        )

    # Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "superclasses": DEFAULT_SUPERCLASSES,
            "thresholds": DEFAULT_THRESHOLDS,
            "input_shape": [DEFAULT_INPUT_SHAPE[0], DEFAULT_INPUT_SHAPE[1], 3]
        }

    # Load model weights without compilation
    model = keras.models.load_model(MODEL_PATH, compile=False)

    _CACHED_MODEL = model
    _CACHED_METADATA = metadata
    return _CACHED_MODEL, _CACHED_METADATA


def preprocess_ecg_image(image_path: str, target_size: tuple = DEFAULT_INPUT_SHAPE) -> np.ndarray:
    """
    Loads an image from image_path, validates it, and preprocesses it for model input:
    1. Converts image to RGB (stripping alpha or grayscale channels).
    2. Resizes to target_size (e.g. 128x128) using bilinear interpolation.
    3. Normalizes pixel values to [0.0, 1.0] as float32.
    4. Adds a leading batch dimension, returning shape (1, height, width, 3).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"ECG image not found at path: '{image_path}'")

    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
            img_array = np.array(img_rgb, dtype=np.float32) / 255.0
            # Add batch dimension: (H, W, C) -> (1, H, W, C)
            batched_array = np.expand_dims(img_array, axis=0)
            return batched_array
    except Exception as e:
        raise ValueError(f"Failed to load and process image at '{image_path}': {e}")


def classify_ecg_image(image_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Classifies a 12-lead ECG paper image into 5 diagnostic superclasses.

    Parameters:
        image_path (str): Filepath to the ECG image (e.g., .jpg, .png).

    Returns:
        dict: Standardized multi-label prediction output:
        {
          "NORM": {"detected": bool, "confidence_score": float},
          "MI":   {"detected": bool, "confidence_score": float},
          "STTC": {"detected": bool, "confidence_score": float},
          "CD":   {"detected": bool, "confidence_score": float},
          "HYP":  {"detected": bool, "confidence_score": float}
        }
    """
    # 1. Load model and metadata
    model, metadata = load_model_and_metadata()

    superclasses = metadata.get("superclasses", DEFAULT_SUPERCLASSES)
    thresholds = metadata.get("thresholds", DEFAULT_THRESHOLDS)
    input_shape = metadata.get("input_shape", [128, 128, 3])
    target_size = (input_shape[0], input_shape[1])

    # 2. Preprocess input image
    input_tensor = preprocess_ecg_image(image_path, target_size=target_size)

    # 3. Model forward pass (returns sigmoid probabilities in [0.0, 1.0])
    raw_preds = model.predict(input_tensor, verbose=0)
    probabilities = raw_preds[0]  # First sample in batch

    # 4. Construct result dictionary applying class thresholds
    result: Dict[str, Dict[str, Any]] = {}
    for idx, sc in enumerate(superclasses):
        prob = float(probabilities[idx])
        # Look up class threshold
        thresh = float(thresholds.get(sc, 0.50))
        is_detected = bool(prob >= thresh)

        result[sc] = {
            "detected": is_detected,
            "confidence_score": round(prob, 4)
        }

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = os.path.join(BASE_DIR, "data", "images", "test", "10006_clean.jpg")

    print(f"Testing inference on: {test_path}")
    if os.path.exists(test_path):
        res = classify_ecg_image(test_path)
        print("\nClassification Result:")
        print(json.dumps(res, indent=2))
    else:
        print(f"File not found: {test_path}")
