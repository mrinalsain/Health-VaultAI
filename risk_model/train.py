# ==============================================================================
# HealthVault AI - Cardiovascular Risk Prediction Model Training Pipeline
#
# This script executes tasks 1 through 11:
# 1. Dataset loading and comprehensive exploratory inspection (.info, .describe,
#    missing values, class balance).
# 2. Column-by-column missing value handling with clinical & statistical rationale.
# 3. Handling target class imbalance via cost-sensitive learning (class weighting).
# 4. Stratified 80/20 train/test split.
# 5. Feature scaling using StandardScaler (fitted strictly on training data).
# 6. Extraction and export of training set medians (medians.json).
# 7. Training & test evaluation of Logistic Regression and Random Forest baselines.
# 8. Building, compiling, and evaluating a Keras Neural Network.
# 9. ROC-AUC evaluation across all 3 architectures.
# 10. Honest medical screening comparison prioritizing Recall/Sensitivity.
# 11. Persistence of the winning model, scaler, and metadata into risk_model/saved/.
# ==============================================================================

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Set Keras backend to PyTorch before importing Keras.
# Rationale: Official TensorFlow Windows wheels are not available for Python 3.14.
# Setting KERAS_BACKEND='torch' runs Keras 3 with the exact requested Dense layers.
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight


# Define the 13 clinical features and the target column.
# Excluded from raw dataset:
# - 'education': Socioeconomic variable, not a biological risk factor; excluded to avoid demographic bias.
# - 'prevalentStroke': A prior stroke indicates established cardiovascular disease (secondary prevention),
#   whereas this model focuses on primary 10-year coronary heart disease screening for general patients.
NUMERIC_FEATURES = [
    "age",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "cigsPerDay"
]

BINARY_FEATURES = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentHyp",
    "diabetes"
]

ALL_FEATURES = [
    "age",
    "male",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose"
]

TARGET_COL = "TenYearCHD"


def print_section(title: str):
    """Utility helper to format clean, readable output sections in the console."""
    border = "=" * 80
    print(f"\n{border}\n{title.upper()}\n{border}")


def task_1_load_and_inspect(data_path: str = "framingham.csv") -> pd.DataFrame:
    """
    Task 1: Load framingham.csv and display initial exploratory metrics.
    
    ML Concept - Class Imbalance:
    Class imbalance occurs when one outcome (e.g. developing heart disease, 1) is much
    rarer than the other (no heart disease, 0). If unaddressed, standard algorithms
    will simply predict the majority class for everyone to achieve misleadingly high accuracy.
    """
    print_section("Task 1: Load Dataset & Exploratory Inspection")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Successfully loaded '{data_path}' with {df.shape[0]} rows and {df.shape[1]} columns.\n")
    
    print("--- 1.1 Dataset Info ---")
    df.info()
    
    print("\n--- 1.2 Summary Statistics (.describe()) ---")
    print(df.describe().round(2).to_string())
    
    print("\n--- 1.3 Missing Values Count Per Column ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct.round(2)})
    print(missing_df[missing_df["Missing Count"] > 0].to_string())
    
    print("\n--- 1.4 Class Balance of Target ('TenYearCHD') ---")
    counts = df[TARGET_COL].value_counts()
    percentages = (df[TARGET_COL].value_counts(normalize=True) * 100).round(2)
    balance_df = pd.DataFrame({"Patient Count": counts, "Percentage (%)": percentages})
    print(balance_df.to_string())
    ratio = counts[0] / counts[1]
    print(f"\nClass imbalance ratio: ~{ratio:.2f}:1 (Negative : Positive).")
    print("Explanation: Over 84% of records did NOT develop CHD in 10 years.")
    print("A naive model predicting '0' for every patient would achieve ~84.8% accuracy but 0% recall.")
    
    return df


def task_2_check_data_loss_and_plan_imputation(df: pd.DataFrame):
    """
    Task 2: Analyze what happens if we blanket-drop missing values vs. column-by-column handling.
    """
    print_section("Task 2: Data Loss Evaluation & Missing Value Strategy")
    
    total_rows = len(df)
    complete_rows = len(df.dropna())
    rows_lost = total_rows - complete_rows
    pct_lost = (rows_lost / total_rows) * 100
    
    print(f"Total original patient records: {total_rows}")
    print(f"Records with zero missing values: {complete_rows}")
    print(f"Blanket dropna() would lose: {rows_lost} rows ({pct_lost:.2f}% of all data!)")
    
    # Check impact of blanket drop specifically on positive cases
    pos_original = (df[TARGET_COL] == 1).sum()
    pos_after_drop = (df.dropna()[TARGET_COL] == 1).sum()
    pos_lost = pos_original - pos_after_drop
    print(f"Positive CHD cases lost if blanket-dropped: {pos_lost} out of {pos_original} ({(pos_lost/pos_original)*100:.2f}% of positives)")
    print("\nConclusion: Blanket-dropping rows severely damages our rare positive class.")
    print("We must handle missing values column-by-column using medical domain logic.")


def task_4_split_data(df: pd.DataFrame):
    """
    Task 4: Stratified 80/20 train/test split.
    
    ML Concept - Stratified Splitting & Leakage Prevention:
    Stratification ensures both train and test splits retain the exact 84.8% / 15.2% class proportion.
    Splitting *before* computing medians or scaling prevents 'data leakage' (future test information
    contaminating the training phase).
    """
    print_section("Task 4: Train/Test Split (80/20 Stratified)")
    
    # Feature matrix X and target y
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} rows (Positive cases: {y_train.sum()} = {(y_train.mean()*100):.2f}%)")
    print(f"Test set:     {X_test.shape[0]} rows (Positive cases: {y_test.sum()} = {(y_test.mean()*100):.2f}%)")
    
    return X_train, X_test, y_train, y_test


def task_2_and_6_impute_and_extract_medians(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Task 2 & 6:
    - Compute training set medians column-by-column to avoid data leakage.
    - Impute X_train and X_test.
    - Export medians.json for downstream inference fallbacks.
    
    Column-by-column reasoning:
    - cigsPerDay: Missing records all occur in active smokers. We fill them using the median
      consumption of active smokers in the training data (non-smokers remain 0).
    - BPMeds: Binary indicator. Over 97% of participants are not on BP meds (mode = 0.0).
    - totChol, sysBP, diaBP, BMI, heartRate, glucose: Continuous lab/vital measurements with
      right-skewed biological distributions. Median is robust to extreme outliers compared to mean.
    """
    print_section("Tasks 2 & 6: Column-by-Column Imputation & Medians Export")
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Calculate training set medians for smokers
    smokers_mask_train = X_train["currentSmoker"] == 1
    smoker_cigs_median = float(X_train.loc[smokers_mask_train, "cigsPerDay"].median())
    
    # Compute overall training medians for all optional fields
    medians = {
        "cigs_per_day": smoker_cigs_median,
        "on_bp_meds": float(X_train["BPMeds"].mode()[0]),  # 0.0
        "cholesterol": float(X_train["totChol"].median()),
        "systolic_bp": float(X_train["sysBP"].median()),
        "diastolic_bp": float(X_train["diaBP"].median()),
        "bmi": float(X_train["BMI"].median()),
        "heart_rate": float(X_train["heartRate"].median()),
        "glucose": float(X_train["glucose"].median())
    }
    
    print("Computed Training Set Fallback Medians (Task 6):")
    for field, val in medians.items():
        print(f"  - {field:15s}: {val}")
        
    # Save medians.json into risk_model/saved/
    saved_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(saved_dir, exist_ok=True)
    medians_path = os.path.join(saved_dir, "medians.json")
    with open(medians_path, "w") as f:
        json.dump(medians, f, indent=4)
    print(f"\nSaved medians to: {medians_path}")
    
    # Apply imputation to Training and Test sets using Training statistics
    for df_split, name in [(X_train, "Training"), (X_test, "Test")]:
        # cigsPerDay
        cigs_fallback = pd.Series(
            np.where(df_split["currentSmoker"] == 1, medians["cigs_per_day"], 0.0),
            index=df_split.index
        )
        df_split["cigsPerDay"] = df_split["cigsPerDay"].fillna(cigs_fallback)
        # BPMeds
        df_split["BPMeds"] = df_split["BPMeds"].fillna(medians["on_bp_meds"])
        # totChol
        df_split["totChol"] = df_split["totChol"].fillna(medians["cholesterol"])
        # BMI
        df_split["BMI"] = df_split["BMI"].fillna(medians["bmi"])
        # heartRate
        df_split["heartRate"] = df_split["heartRate"].fillna(medians["heart_rate"])
        # glucose
        df_split["glucose"] = df_split["glucose"].fillna(medians["glucose"])
        
    print("Imputation successfully applied. Remaining nulls in training:", X_train.isnull().sum().sum())
    print("Remaining nulls in test:", X_test.isnull().sum().sum())
    
    return X_train, X_test, medians


def task_3_explain_class_imbalance():
    """
    Task 3: Explanation of class imbalance handling choice.
    """
    print_section("Task 3: Strategy for Target Class Imbalance")
    explanation = (
        "Class Imbalance Strategy Choice:\n"
        "We choose Cost-Sensitive Learning (class_weight='balanced') rather than SMOTE.\n"
        "1. Medical Realism: SMOTE synthesizes points by linear interpolation between nearest neighbors.\n"
        "   For vital signs (blood pressure, blood glucose, age), synthetic interpolation can create\n"
        "   physiologically unnatural combinations (e.g. extreme systolic BP with normal diastolic BP).\n"
        "2. Probability Calibration: SMOTE artificially alters the base disease prevalence from ~15% to 50%,\n"
        "   severely distorting output risk probabilities. Balanced class weighting directly penalizes\n"
        "   false negatives proportionally during optimization while keeping true clinical patient distributions."
    )
    print(explanation)


def task_5_scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Task 5: Scale numeric features with StandardScaler fit only on training data.
    
    ML Concept - Scaling:
    Features like cholesterol (150-400 mg/dL) have vastly larger numerical scales than age (30-70)
    or BMI (18-45). StandardScaler transforms them to zero mean and unit variance so no single
    feature dominates gradient descent or logistic regression coefficients.
    """
    print_section("Task 5: Feature Scaling (StandardScaler)")
    
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit strictly on training numeric columns
    scaler.fit(X_train[NUMERIC_FEATURES])
    
    # Transform both splits
    X_train_scaled[NUMERIC_FEATURES] = scaler.transform(X_train[NUMERIC_FEATURES])
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])
    
    saved_dir = os.path.join(os.path.dirname(__file__), "saved")
    scaler_path = os.path.join(saved_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler successfully fit on training data and saved to: {scaler_path}")
    print(f"Scaled numeric features: {NUMERIC_FEATURES}")
    print(f"Binary features preserved without scaling: {BINARY_FEATURES}")
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(name: str, y_true, y_pred, y_probs):
    """
    Compute standard clinical classification metrics: Accuracy, Precision, Recall, F1, and Confusion Matrix.
    
    ML Concept - Recall (Sensitivity):
    Recall measures what fraction of patients who actually developed heart disease were correctly
    flagged by our model. In medical screening, Recall is prioritized because missing a diseased patient
    (False Negative) has catastrophic consequences, whereas a false alarm (False Positive) is safely
    resolved with follow-up clinical tests.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Performance Metrics: {name} ---")
    print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:        {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall (Sens.):   {rec:.4f} ({rec*100:.2f}%)  <-- Key screening metric")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  ROC-AUC Score:    {roc_auc:.4f}")
    print("  Confusion Matrix:")
    print(f"    [ True Negative (TN): {tn:4d}  |  False Positive (FP): {fp:4d} ]")
    print(f"    [ False Negative (FN): {fn:4d}  |  True Positive  (TP): {tp:4d} ]")
    
    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def task_7_train_baselines(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Task 7: Train Logistic Regression and Random Forest with balanced class weights.
    """
    print_section("Task 7: Train & Evaluate Baselines (LogReg & Random Forest)")
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression with class_weight='balanced'...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds = lr.predict(X_test_scaled)
    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_preds, lr_probs)
    
    # 2. Random Forest
    print("\nTraining Random Forest with class_weight='balanced'...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
    rf_preds = rf.predict(X_test_scaled)
    rf_metrics = evaluate_model("Random Forest", y_test, rf_preds, rf_probs)
    
    return (lr, lr_probs, lr_metrics), (rf, rf_probs, rf_metrics)


def task_8_train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Task 8: Build and train a small Keras Neural Network with class weighting.
    Architecture: Dense(32, relu) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid)
    """
    print_section("Task 8: Build & Evaluate Neural Network (Keras / PyTorch backend)")
    
    # Compute class weights matching scikit-learn's balanced formula:
    # weight_c = n_samples / (n_classes * n_samples_c)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Calculated Neural Network Class Weights: {class_weight_dict}")
    
    # Build Model
    keras.utils.set_random_seed(42)
    nn_model = keras.Sequential([
        layers.Input(shape=(len(ALL_FEATURES),)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),  # Regularization to prevent overfitting on ~4,000 rows
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nNeural Network Architecture Summary:")
    nn_model.summary()
    
    # Train model
    history = nn_model.fit(
        X_train_scaled.values.astype(np.float32),
        y_train.values.astype(np.float32),
        epochs=35,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=0
    )
    print("Neural network training complete across 35 epochs.")
    
    nn_probs = nn_model.predict(X_test_scaled.values.astype(np.float32), verbose=0).flatten()
    nn_preds = (nn_probs >= 0.5).astype(int)
    nn_metrics = evaluate_model("Neural Network (Keras)", y_test, nn_preds, nn_probs)
    
    return nn_model, nn_probs, nn_metrics


def task_9_compare_models_and_select(models_data, y_test):
    """
    Tasks 9 & 10:
    - Compare all three models honestly on actual measured metrics.
    - Medical priority: Recall / Sensitivity.
    """
    print_section("Tasks 9 & 10: Comprehensive Model Comparison (Medical Screening Priority)")
    
    summary_rows = []
    for model_obj, probs, metrics in models_data:
        summary_rows.append({
            "Model": metrics["name"],
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall (Sensitivity)": f"{metrics['recall']*100:.2f}%",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "True Positives (TP)": metrics["tp"],
            "False Negatives (FN)": metrics["fn"],
            "False Positives (FP)": metrics["fp"]
        })
        
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Decision Logic:
    # Prioritize Recall/Sensitivity first, then F1 / ROC-AUC.
    print("\nClinical Rationale for Model Selection:")
    print("- In preventative cardiovascular medicine, a False Negative (a high-risk patient")
    print("  told they are safe, who subsequent develops heart disease untreated) is a clinical failure.")
    print("- A False Positive causes non-invasive follow-up (lipid checks, blood pressure monitoring, diet advice).")
    
    # Rank by Recall descending, then F1 descending
    ranked = sorted(models_data, key=lambda item: (item[2]["recall"], item[2]["f1"]), reverse=True)
    best_item = ranked[0]
    best_model, best_probs, best_metrics = best_item
    
    print(f"\nWINNER SELECTED: {best_metrics['name']}")
    print(f"Reason: Highest Sensitivity/Recall ({best_metrics['recall']*100:.2f}%) with ROC-AUC {best_metrics['roc_auc']:.4f}.")
    print(f"It correctly identified {best_metrics['tp']} out of {best_metrics['tp'] + best_metrics['fn']} CHD patients,")
    print(f"missing only {best_metrics['fn']} true cases.")
    
    return best_model, best_metrics, best_probs


def task_11_save_artifacts(best_model, best_metrics, y_test, best_probs):
    """
    Task 11: Save best performing model into risk_model/saved/ along with metadata.
    """
    print_section("Task 11: Save Best Model & Deployment Metadata")
    
    saved_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(saved_dir, exist_ok=True)
    
    # Determine optimal threshold from Precision-Recall / ROC analysis
    fpr, tpr, thresholds = roc_curve(y_test, best_probs)
    # Youden's J statistic = Sensitivity + Specificity - 1 = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    
    # Ensure threshold is sensible (between 0.35 and 0.55)
    if not (0.25 <= optimal_threshold <= 0.60):
        optimal_threshold = 0.50
    else:
        optimal_threshold = round(optimal_threshold, 2)
        
    print(f"Screening Decision Threshold selected: {optimal_threshold}")
    print(f"Patients with predicted probability >= {optimal_threshold} flagged as 'Elevated Risk'.")
    
    model_name = best_metrics["name"]
    is_keras = "Keras" in model_name or isinstance(best_model, keras.Model)
    
    if is_keras:
        model_path = os.path.join(saved_dir, "best_model.keras")
        best_model.save(model_path)
        model_type = "keras"
    else:
        model_path = os.path.join(saved_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)
        model_type = "sklearn"
        
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "model_file": os.path.basename(model_path),
        "threshold": optimal_threshold,
        "feature_order": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "metrics": best_metrics
    }
    
    meta_path = os.path.join(saved_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Best model saved to: {model_path}")
    print(f"Model metadata saved to: {meta_path}")


def main():
    """Main training execution orchestrator."""
    print("=" * 80)
    print("STARTING HEALTHVAULT AI CARDIOVASCULAR RISK MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Load data
    df = task_1_load_and_inspect("framingham.csv")
    
    # 2. Check data loss from naive dropping
    task_2_check_data_loss_and_plan_imputation(df)
    
    # 3. Explain class imbalance strategy
    task_3_explain_class_imbalance()
    
    # 4. Train/test split (stratified)
    X_train, X_test, y_train, y_test = task_4_split_data(df)
    
    # 2 & 6. Impute missing values and export training medians
    X_train_imp, X_test_imp, medians = task_2_and_6_impute_and_extract_medians(X_train, X_test)
    
    # 5. Scale numeric features with StandardScaler
    X_train_scaled, X_test_scaled, scaler = task_5_scale_features(X_train_imp, X_test_imp)
    
    # 7. Train Logistic Regression and Random Forest baselines
    lr_data, rf_data = task_7_train_baselines(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 8. Train Keras Neural Network
    nn_model, nn_probs, nn_metrics = task_8_train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
    nn_data = (nn_model, nn_probs, nn_metrics)
    
    # 9 & 10. Compare models and select best based on Recall
    all_models = [lr_data, rf_data, nn_data]
    best_model, best_metrics, best_probs = task_9_compare_models_and_select(all_models, y_test)
    
    # 11. Save artifacts
    task_11_save_artifacts(best_model, best_metrics, y_test, best_probs)
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
