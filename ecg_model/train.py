"""
ecg_model/train.py
==================
HealthVault AI - ECG Image Classification Module
Part 3: Multi-Label Convolutional Neural Network (CNN) Training & Evaluation

This script:
1. Loads pre-generated ECG images (clean and distorted) cataloged in image_records.csv.
2. Formulates multi-label image classification:
   - Target classes: 5 PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP).
   - In clinical reality, a single patient ECG can exhibit multiple cardiac conditions
     simultaneously (e.g., both Myocardial Infarction and Conduction Disturbance).
   - Final output layer: Dense(5, activation='sigmoid') — NOT softmax!
     Why Sigmoid instead of Softmax?
     Softmax normalizes outputs across all classes so they sum to 1.0, treating classes
     as mutually exclusive (if one probability goes up, others are forced down).
     In multi-label diagnostics, diseases are independent or co-occurring.
     Sigmoid outputs 5 separate independent probabilities in [0.0, 1.0].
3. Loss Function:
   - Weighted Binary Cross-Entropy (BCE).
   - Evaluates each output unit as an independent binary decision (yes/no).
   - Minority classes (such as NORM and MI) receive positive class weights to penalize
     false negatives and preserve diagnostic sensitivity.
4. Multi-Label Evaluation Metrics:
   - Per-class Precision, Recall, and F1-score.
     Prioritizes Recall for abnormal classes: missing an acute infarction or conduction block
     is clinically dangerous (false negative), whereas a false positive is safely ruled out
     by a subsequent physician consultation.
   - Hamming Loss: The fraction of incorrect single labels across all samples and classes.
   - Subset Accuracy: The exact match ratio (all 5 labels correct simultaneously).
5. Clean-vs-Distorted Domain Gap Evaluation:
   - Evaluates the test set partitioned into clean-only and distorted-only subsets.
   - Quantifies whether phone camera distortions (blur, tilt, lighting shadows) degrade performance.
6. Per-Class Threshold Optimization:
   - Sweeps classification thresholds on the validation set to balance sensitivity and specificity.
7. Model Persistence:
   - Saves model weights to ecg_model/saved/ecg_model.keras.
   - Saves thresholds, class weights, and metrics to ecg_model/saved/metadata.json.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Set Keras backend to PyTorch before importing Keras.
# Rationale: Official TensorFlow Windows wheels are unavailable for Python 3.14 on Windows.
# Running Keras 3 with PyTorch provides native execution and leverages multi-core CPU threading.
os.environ["KERAS_BACKEND"] = "torch"
import torch
# Configure PyTorch to utilize all available CPU cores for high-throughput tensor operations
torch.set_num_threads(os.cpu_count() or 8)

import keras
from keras import layers

# Define constants
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
IMAGE_SIZE = (96, 96)  # High-efficiency resolution preserving ECG grid and waveform deflections
BATCH_SIZE = 64
EPOCHS = 7
LEARNING_RATE = 1e-3

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
RECORDS_CSV = os.path.join(IMAGES_DIR, "image_records.csv")
SAVED_DIR = os.path.join(BASE_DIR, "saved")
MODEL_SAVE_PATH = os.path.join(SAVED_DIR, "ecg_model.keras")
METADATA_SAVE_PATH = os.path.join(SAVED_DIR, "metadata.json")


def load_single_image(path: str, target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Loads an ECG image from disk, converts to RGB, resizes to target_size,
    and returns normalized float32 array in range [0.0, 1.0].
    """
    with Image.open(path) as img:
        img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        return arr


def load_dataset_split(df_split: pd.DataFrame, split_name: str, max_workers: int = 10):
    """
    Loads all images and multi-hot label vectors for a given split in parallel.
    """
    print(f"Loading {len(df_split):,} images for '{split_name}' split...")
    t0 = time.time()

    paths = [os.path.join(IMAGES_DIR, p) for p in df_split["image_path"]]
    labels = df_split[SUPERCLASSES].values.astype(np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images_list = list(executor.map(load_single_image, paths))

    X = np.stack(images_list, axis=0)
    y = labels
    elapsed = time.time() - t0
    print(f"  Finished loading '{split_name}' in {elapsed:.2f}s. Shape: X={X.shape}, y={y.shape}")
    return X, y


@keras.saving.register_keras_serializable(package="custom")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """
    Custom Weighted Binary Cross-Entropy Loss for multi-label classification.
    
    Why custom weighted BCE?
    In multi-label problems, each output is an independent binary target. Certain conditions
    like NORM and MI have fewer positive instances than negative instances in our subset.
    Standard BCE weights false positives and false negatives equally. By assigning a higher
    weight to positive examples (w_pos = N_neg / N_pos), false negatives incur a larger
    penalty, encouraging the model to maintain high recall on dangerous heart conditions.
    """
    def __init__(self, pos_weights, name="weighted_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        if isinstance(pos_weights, (list, tuple, np.ndarray)):
            self.pos_weights = np.array(pos_weights, dtype=np.float32).tolist()
        else:
            self.pos_weights = pos_weights

    def call(self, y_true, y_pred):
        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")
        eps = 1e-7
        y_pred = keras.ops.clip(y_pred, eps, 1.0 - eps)
        weights_tensor = keras.ops.convert_to_tensor(self.pos_weights, dtype="float32")
        
        # Positive term: y_true * log(y_pred) * pos_weights
        pos_term = keras.ops.multiply(
            keras.ops.multiply(y_true, keras.ops.log(y_pred)),
            weights_tensor
        )
        # Negative term: (1 - y_true) * log(1 - y_pred)
        neg_term = keras.ops.multiply(
            1.0 - y_true,
            keras.ops.log(1.0 - y_pred)
        )
        return -keras.ops.mean(pos_term + neg_term)

    def get_config(self):
        config = super().get_config()
        config.update({"pos_weights": self.pos_weights})
        return config


def build_multilabel_cnn(input_shape=(96, 96, 3), num_classes=5) -> keras.Model:
    """
    Builds a high-capacity, low-latency Convolutional Neural Network for ECG classification.
    
    Architecture Design:
    - 4 Conv2D blocks with progressive filter depths (16 -> 32 -> 64 -> 64).
    - Strided downsampling in block 1 rapidly captures macro lead layout.
    - MaxPooling2D pools spatial information while preserving lead waveforms.
    - Dropout (0.10 - 0.25) provides strong regularization against overfitting.
    - GlobalAveragePooling2D reduces spatial feature maps to vector embeddings.
    - Final Layer: Dense(5, activation='sigmoid')
      Outputs 5 independent probability scores between 0.0 and 1.0.
    """
    inputs = layers.Input(shape=input_shape, name="ecg_image_input")

    # Block 1: Strided Conv + Pool
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", activation="relu", name="conv1")(inputs)  # 48x48
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 24x24
    x = layers.Dropout(0.10, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)  # 24x24
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 12x12
    x = layers.Dropout(0.15, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(x)  # 12x12
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)  # 6x6
    x = layers.Dropout(0.20, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv4")(x)  # 6x6
    x = layers.Dropout(0.20, name="drop4")(x)

    # Global feature pooling & classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.25, name="drop_fc")(x)

    # CRITICAL: Sigmoid activation for multi-label classification
    outputs = layers.Dense(num_classes, activation="sigmoid", name="multilabel_sigmoid_out")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="HealthVault_ECG_CNN")
    return model


def compute_multilabel_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, thresholds: dict) -> dict:
    """
    Computes comprehensive multi-label evaluation metrics:
    - Per-class: Precision, Recall, F1-Score, Support, Confusion Matrix components
    - Overall Summary:
      * Hamming Loss: fraction of label errors across all instances and classes.
        Formula: HL = (1 / (N * C)) * sum(y_true != y_pred_binary)
      * Subset Accuracy: fraction of instances where all 5 labels match ground truth exactly.
        Formula: SA = (1 / N) * sum(all(y_true == y_pred_binary, axis=1))
    """
    thresh_array = np.array([thresholds[sc] for sc in SUPERCLASSES], dtype=np.float32)
    y_pred_binary = (y_pred_prob >= thresh_array).astype(int)

    per_class_metrics = {}
    for idx, sc in enumerate(SUPERCLASSES):
        yt = y_true[:, idx].astype(int)
        yp = y_pred_binary[:, idx].astype(int)

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[sc] = {
            "threshold": float(thresholds[sc]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(yt)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    # Hamming Loss: fraction of incorrect labels across entire matrix
    hamming_loss = float(np.mean(y_true != y_pred_binary))

    # Subset Accuracy: exact match across all classes simultaneously
    exact_matches = np.all(y_true == y_pred_binary, axis=1)
    subset_accuracy = float(np.mean(exact_matches))

    return {
        "per_class": per_class_metrics,
        "hamming_loss": hamming_loss,
        "subset_accuracy": subset_accuracy
    }


def optimize_thresholds(y_val_true: np.ndarray, y_val_probs: np.ndarray) -> dict:
    """
    Determines per-class decision thresholds using the validation set.
    
    Threshold Strategy:
    In clinical screening, false negatives are far more hazardous than false alarms.
    - For abnormal conditions (MI, HYP, CD, STTC):
      Prioritize Recall (target >= 0.70-0.80) while choosing the threshold that maximizes F1.
      Missing an acute infarction or conduction block is critical.
    - For NORM:
      Sweep to maximize balanced F1, ensuring healthy ECGs are accurately identified.
    """
    print("\n" + "=" * 70)
    print("OPTIMIZING PER-CLASS CLASSIFICATION THRESHOLDS (VALIDATION SET)")
    print("=" * 70)
    chosen_thresholds = {}

    candidate_thresholds = np.linspace(0.15, 0.75, 25)

    for idx, sc in enumerate(SUPERCLASSES):
        yt = y_val_true[:, idx]
        yp = y_val_probs[:, idx]

        best_thresh = 0.50
        best_score = -1.0
        best_prec, best_rec, best_f1 = 0.0, 0.0, 0.0

        for t in candidate_thresholds:
            y_pred = (yp >= t).astype(int)
            tp = np.sum((yt == 1) & (y_pred == 1))
            fp = np.sum((yt == 0) & (y_pred == 1))
            fn = np.sum((yt == 1) & (y_pred == 0))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            # For abnormal findings, use F2-weighted score to prioritize recall
            if sc in ["MI", "CD", "HYP", "STTC"]:
                score = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
            else:
                score = f1

            if score > best_score and r >= 0.35:
                best_score = score
                best_thresh = t
                best_prec, best_rec, best_f1 = p, r, f1

        chosen_thresholds[sc] = round(float(best_thresh), 3)
        print(f"  {sc:<6}: Threshold = {chosen_thresholds[sc]:.3f} | Val Prec: {best_prec:.3f}, Recall: {best_rec:.3f}, F1: {best_f1:.3f}")

    print("=" * 70 + "\n")
    return chosen_thresholds


def main():
    print("=" * 75)
    print("HEALTHVAULT AI - MULTI-LABEL ECG CNN TRAINING PIPELINE")
    print("=" * 75)
    print(f"Active CPU Threads: {torch.get_num_threads()}")

    # 1. Load image catalog
    if not os.path.exists(RECORDS_CSV):
        raise FileNotFoundError(f"Missing {RECORDS_CSV}. Please run image_generator.py first.")

    df_records = pd.read_csv(RECORDS_CSV)
    print(f"Total image records loaded: {len(df_records):,}")
    print(df_records.groupby(["split", "distortion_type"]).size())

    # 2. Partition dataset
    train_df = df_records[df_records["split"] == "train"].copy()
    val_df = df_records[df_records["split"] == "val"].copy()
    test_df = df_records[df_records["split"] == "test"].copy()

    # 3. Load pixel arrays in parallel
    X_train, y_train = load_dataset_split(train_df, "Train")
    X_val, y_val = load_dataset_split(val_df, "Validation")
    X_test, y_test = load_dataset_split(test_df, "Test")

    # 4. Compute positive class weights
    # w_pos = (N - N_pos) / N_pos
    num_train = len(y_train)
    pos_counts = np.sum(y_train, axis=0)
    pos_weights = (num_train - pos_counts) / np.maximum(pos_counts, 1.0)
    # Clip weights to moderate multiplier range [0.8, 3.5]
    pos_weights_clipped = np.clip(pos_weights, 0.8, 3.5).astype(np.float32)

    class_weight_dict = {sc: float(pos_weights_clipped[i]) for i, sc in enumerate(SUPERCLASSES)}
    print("\nClass Imbalance & Positive Weights:")
    for sc, w in class_weight_dict.items():
        idx = SUPERCLASSES.index(sc)
        print(f"  {sc:<6}: Positive Count = {int(pos_counts[idx]):>4} / {num_train} | Weight = {w:.2f}")

    # 5. Build and compile CNN
    keras.utils.set_random_seed(42)
    model = build_multilabel_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=5)
    loss_fn = WeightedBinaryCrossentropy(pos_weights=pos_weights_clipped.tolist())

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["binary_accuracy"]
    )
    print("\nModel Architecture Summary:")
    model.summary()

    # 6. Train CNN
    print(f"\nTraining Multi-Label CNN for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    train_duration = time.time() - t_start
    print(f"\nTraining completed in {train_duration:.1f}s ({train_duration/60:.2f} mins).")

    # 7. Validation predictions and threshold optimization
    print("\nEvaluating on Validation Set...")
    val_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    thresholds = optimize_thresholds(y_val, val_probs)

    # 8. Test set evaluation (Overall)
    print("=" * 75)
    print("FINAL TEST SET EVALUATION (OVERALL: 1,500 IMAGES)")
    print("=" * 75)
    test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    overall_metrics = compute_multilabel_metrics(y_test, test_probs, thresholds)

    print(f"{'Class':<8} | {'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for sc in SUPERCLASSES:
        m = overall_metrics["per_class"][sc]
        print(f"{sc:<8} | {m['threshold']:<10.3f} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1']:<10.3f} | {m['support']:<8}")
    print("-" * 65)
    print(f"Hamming Loss (lower is better):     {overall_metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy (exact 5-match):     {overall_metrics['subset_accuracy']:.4f} ({overall_metrics['subset_accuracy']*100:.1f}%)")
    print("=" * 75 + "\n")

    # 9. Clean vs Distorted Test Set Breakdown
    print("=" * 75)
    print("DOMAIN-GAP ANALYSIS: CLEAN vs DISTORTED TEST SET EVALUATION")
    print("=" * 75)

    test_clean_indices = np.where(test_df["distortion_type"].values == "clean")[0]
    test_distorted_indices = np.where(test_df["distortion_type"].values == "distorted")[0]

    y_test_clean = y_test[test_clean_indices]
    probs_clean = test_probs[test_clean_indices]
    clean_metrics = compute_multilabel_metrics(y_test_clean, probs_clean, thresholds)

    y_test_distorted = y_test[test_distorted_indices]
    probs_distorted = test_probs[test_distorted_indices]
    distorted_metrics = compute_multilabel_metrics(y_test_distorted, probs_distorted, thresholds)

    print(f"{'Class':<8} | {'Clean Recall':<14} | {'Distorted Recall':<18} | {'Clean F1':<12} | {'Distorted F1':<14}")
    print("-" * 75)
    for sc in SUPERCLASSES:
        c_rec = clean_metrics["per_class"][sc]["recall"]
        d_rec = distorted_metrics["per_class"][sc]["recall"]
        c_f1 = clean_metrics["per_class"][sc]["f1"]
        d_f1 = distorted_metrics["per_class"][sc]["f1"]
        print(f"{sc:<8} | {c_rec:<14.3f} | {d_rec:<18.3f} | {c_f1:<12.3f} | {d_f1:<14.3f}")
    print("-" * 75)
    print(f"Clean Subset Accuracy:     {clean_metrics['subset_accuracy']:.4f} | Hamming Loss: {clean_metrics['hamming_loss']:.4f}")
    print(f"Distorted Subset Accuracy: {distorted_metrics['subset_accuracy']:.4f} | Hamming Loss: {distorted_metrics['hamming_loss']:.4f}")
    print("=" * 75 + "\n")

    # 10. Persist Model and Metadata
    os.makedirs(SAVED_DIR, exist_ok=True)
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    metadata = {
        "model_name": "HealthVault_ECG_MultiLabel_CNN",
        "superclasses": SUPERCLASSES,
        "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "thresholds": thresholds,
        "class_weights": class_weight_dict,
        "training_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "metrics_overall": overall_metrics,
        "metrics_clean": clean_metrics,
        "metrics_distorted": distorted_metrics,
    }

    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_SAVE_PATH}")
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
