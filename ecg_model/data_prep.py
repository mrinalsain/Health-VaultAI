"""
ecg_model/data_prep.py
======================
HealthVault AI - ECG Image Classification Module
Part 1: Data Subset Selection and Patient-Stratified Splitting

This script:
1. Loads the PTB-XL database records and diagnostic SCP statements.
2. Maps each ECG record to its applicable diagnostic superclasses (NORM, MI, STTC, CD, HYP).
   Note: This is a MULTI-LABEL problem, meaning one ECG can exhibit multiple cardiac conditions
   simultaneously (e.g., both Myocardial Infarction and Conduction Disturbance).
3. Computes and displays the distribution of all 5 superclasses across the FULL dataset (21,837 records).
4. Selects a representative subset of approximately 5,000 records using patient-level stratification.
   - Core Clinical ML Constraint: A patient's records must NEVER be divided across different splits.
     If a patient appears in both training and testing, the model might "cheat" by memorizing
     that individual patient's anatomical waveform quirks rather than learning generalized disease features.
5. Evaluates class representation against an explicit usability threshold (>= 500 records in subset).
6. Splits the subset into Train (70%), Validation (15%), and Test (15%) splits strictly by patient_id,
   while maintaining balanced class representation across splits.
7. Validates and asserts that patient overlap across all splits is strictly zero (zero data leakage).
8. Saves the resulting subset dataset and split assignments to disk.
"""

import os
import ast
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set

# Define the 5 diagnostic superclasses defined in PTB-XL
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ptbxl")
OUTPUT_METADATA_PATH = os.path.join(BASE_DIR, "data", "subset_metadata.csv")


def load_and_map_ptbxl(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ptbxl_database.csv and scp_statements.csv, then maps each record's
    diagnostic statements to one or more of the 5 diagnostic superclasses:
    - NORM: Normal ECG
    - MI:   Myocardial Infarction (heart attack)
    - STTC: ST/T Changes (ischemia, repolarization changes)
    - CD:   Conduction Disturbance (bundle branch blocks, AV blocks)
    - HYP:  Hypertrophy (ventricular/atrial enlargement)

    In PTB-XL, scp_codes contains a dictionary of {statement_code: likelihood_score}.
    scp_statements.csv maps these codes to diagnostic classes when diagnostic == 1.0.
    """
    db_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(data_dir, "scp_statements.csv")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cannot find PTB-XL database file at {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"Cannot find SCP statements file at {scp_path}")

    # Load database and parse scp_codes string representation into python dictionaries
    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Load SCP statements and filter for diagnostic statements
    scp_df = pd.read_csv(scp_path, index_col=0)
    diagnostic_scp = scp_df[scp_df["diagnostic"] == 1.0]

    # Map statements to superclasses
    def get_superclasses(scp_dict: Dict[str, float]) -> List[str]:
        """
        Maps a record's scp_codes to its diagnostic superclasses.
        Matches the standard PhysioNet benchmark aggregation.
        """
        classes = set()
        for statement_code, likelihood in scp_dict.items():
            if statement_code in diagnostic_scp.index:
                d_class = diagnostic_scp.loc[statement_code, "diagnostic_class"]
                if pd.notna(d_class) and d_class in SUPERCLASSES:
                    classes.add(d_class)
        return sorted(list(classes))

    df["diagnostic_superclass"] = df["scp_codes"].apply(get_superclasses)

    # Create binary multi-hot indicator columns for each superclass
    for sc in SUPERCLASSES:
        df[sc] = df["diagnostic_superclass"].apply(lambda labels: 1 if sc in labels else 0)

    return df, diagnostic_scp


def display_full_dataset_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes and displays the count of records for each superclass across the entire dataset.
    Note: Since this is multi-label, a record can belong to multiple superclasses simultaneously,
    so the sum of counts across classes exceeds the total number of records.
    """
    total_records = len(df)
    total_patients = df["patient_id"].nunique()

    print("\n" + "=" * 70)
    print("PTB-XL FULL DATASET SUMMARY")
    print("=" * 70)
    print(f"Total ECG Records:  {total_records:,}")
    print(f"Total Unique Patients: {total_patients:,}")
    print("\nRecord Counts Per Superclass (Full Dataset):")
    print("-" * 50)

    counts = {}
    for sc in SUPERCLASSES:
        count = int(df[sc].sum())
        percentage = (count / total_records) * 100.0
        counts[sc] = count
        print(f"  {sc:<6} : {count:>6,} records ({percentage:>5.1f}%)")

    # Count multi-label complexity
    num_labels_per_record = df[SUPERCLASSES].sum(axis=1)
    print("\nDiagnostic Label Count Distribution per Record:")
    for num_labels, freq in num_labels_per_record.value_counts().sort_index().items():
        print(f"  {num_labels} labels: {freq:>6,} records ({(freq/total_records)*100:.1f}%)")
    print("=" * 70 + "\n")

    return counts


def select_stratified_subset(
    df: pd.DataFrame,
    target_records: int = 5000,
    min_class_threshold: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Selects approximately target_records (5,000) records strictly by patient_id,
    using multi-label stratification to ensure all 5 superclasses are well represented.

    Why multi-label stratification by patient?
    1. If we sampled individual records randomly, the same patient could have one record in
       the subset and another record outside, or later split between train and test.
    2. Certain conditions like HYP (Hypertrophy) appear in only ~12% of the dataset.
       Without stratification, rare classes can become too thin to train a robust CNN head.
    3. Usability Threshold (min_class_threshold = 500):
       In a 5,000-record subset split 70/15/15, 500 records yields ~350 training records
       (700 clean+distorted images). If any class has fewer than 500 positive records,
       the gradient updates for that class's independent sigmoid output head are sparse,
       risking unstable training or failure to converge.
    """
    print(f"Selecting ~{target_records:,} records with patient-level stratification...")
    rng = np.random.RandomState(random_state)

    # Group all records by patient_id
    patient_records = {}
    patient_labels = {}

    for pid, group in df.groupby("patient_id"):
        patient_records[pid] = list(group.index)
        # Patient-level label is the logical OR of all records for that patient
        patient_labels[pid] = {sc: int(group[sc].max()) for sc in SUPERCLASSES}

    all_patients = list(patient_records.keys())
    rng.shuffle(all_patients)

    # Strategy: Prioritize patients who possess rarer classes (HYP first, then CD, STTC, MI, NORM)
    # until balanced representation is achieved, accumulating complete patient record bundles.
    selected_patients: Set[float] = set()
    current_record_count = 0

    # Sort classes from rarest to most common
    class_scarcity = sorted(SUPERCLASSES, key=lambda c: df[c].sum())

    # Step A: Seed subset with patients having rare classes (HYP, CD, STTC)
    for target_class in class_scarcity:
        class_patients = [
            p for p in all_patients
            if patient_labels[p][target_class] == 1 and p not in selected_patients
        ]
        rng.shuffle(class_patients)

        # Allocate patients until we have a healthy floor for this class
        for p in class_patients:
            if current_record_count >= target_records * 0.85:
                break
            selected_patients.add(p)
            current_record_count += len(patient_records[p])

    # Step B: Fill remaining budget with remaining patients (including NORM)
    remaining_patients = [p for p in all_patients if p not in selected_patients]
    rng.shuffle(remaining_patients)

    for p in remaining_patients:
        if current_record_count >= target_records:
            break
        selected_patients.add(p)
        current_record_count += len(patient_records[p])

    subset_df = df[df["patient_id"].isin(selected_patients)].copy()

    # Step C: Show per-class counts in final subset and verify against usability threshold
    print("\n" + "=" * 70)
    print(f"FINAL SUBSET SELECTION: {len(subset_df):,} records from {subset_df['patient_id'].nunique():,} patients")
    print("=" * 70)
    print(f"Usability Evaluation (Threshold = >= {min_class_threshold} records per class):")
    print("-" * 50)

    subset_counts = {}
    for sc in SUPERCLASSES:
        cnt = int(subset_df[sc].sum())
        pct = (cnt / len(subset_df)) * 100.0
        subset_counts[sc] = cnt
        status = "PASSED (Sufficient)" if cnt >= min_class_threshold else f"WARNING (Below {min_class_threshold})"
        print(f"  {sc:<6} : {cnt:>5,} records ({pct:>5.1f}%) -> {status}")

    # Explicit threshold check
    thin_classes = [sc for sc, cnt in subset_counts.items() if cnt < min_class_threshold]
    if thin_classes:
        print(f"\n[EXPLICIT THRESHOLD NOTICE]: The following classes have fewer than {min_class_threshold} examples: {thin_classes}.")
        print("Reasoning: Multi-label classification trains an independent binary classifier for each class.")
        print(f"Fewer than {min_class_threshold} positives in the subset leaves fewer than {int(min_class_threshold*0.7)} training examples,")
        print("which risks high variance and poor recall. Multi-label stratification was applied to prevent this.")
    else:
        print(f"\nAll 5 superclasses exceed the {min_class_threshold}-record threshold and are usable for robust multi-label training.")
    print("=" * 70 + "\n")

    return subset_df


def split_train_val_test_by_patient(
    subset_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Splits the subset into Train (70%), Val (15%), and Test (15%) by patient_id.
    Ensures that:
    1. No patient_id appears in more than one split (Zero Data Leakage).
    2. Class distributions are reasonably balanced across all three splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = np.random.RandomState(random_state)

    # Group records by patient
    patient_groups = list(subset_df.groupby("patient_id"))
    rng.shuffle(patient_groups)

    # Track target record counts for each split
    total_records = len(subset_df)
    target_train = int(total_records * train_ratio)
    target_val = int(total_records * val_ratio)

    train_pids, val_pids, test_pids = set(), set(), set()
    train_count, val_count, test_count = 0, 0, 0

    # Sort patient groups by rare class presence to evenly distribute them
    def patient_priority(item):
        pid, grp = item
        # Priority score: weighted sum favoring HYP (4) and CD (3) and STTC (2)
        score = grp["HYP"].max() * 4 + grp["CD"].max() * 3 + grp["STTC"].max() * 2 + grp["MI"].max() * 1
        return score

    patient_groups.sort(key=patient_priority, reverse=True)

    # Assign patients cyclically to train/val/test with 70/15/15 target proportions
    for pid, group in patient_groups:
        n_rec = len(group)
        # Proportional assignment
        train_deficit = max(0, target_train - train_count) / target_train
        val_deficit = max(0, target_val - val_count) / target_val
        test_deficit = max(0, (total_records - target_train - target_val) - test_count) / (total_records - target_train - target_val)

        best_split = max(
            [("train", train_deficit), ("val", val_deficit), ("test", test_deficit)],
            key=lambda x: x[1]
        )[0]

        if best_split == "train":
            train_pids.add(pid)
            train_count += n_rec
        elif best_split == "val":
            val_pids.add(pid)
            val_count += n_rec
        else:
            test_pids.add(pid)
            test_count += n_rec

    # Assign split column
    def get_split(pid):
        if pid in train_pids:
            return "train"
        elif pid in val_pids:
            return "val"
        else:
            return "test"

    subset_df["split"] = subset_df["patient_id"].apply(get_split)

    # CONFIRM AND ASSERT NO PATIENT OVERLAP
    overlap_train_val = train_pids.intersection(val_pids)
    overlap_train_test = train_pids.intersection(test_pids)
    overlap_val_test = val_pids.intersection(test_pids)

    print("=" * 70)
    print("PATIENT-LEVEL SPLIT VERIFICATION (LEAKAGE CHECK)")
    print("=" * 70)
    print(f"Train unique patients: {len(train_pids):,}")
    print(f"Val unique patients:   {len(val_pids):,}")
    print(f"Test unique patients:  {len(test_pids):,}")
    print(f"Overlap Train & Val:   {len(overlap_train_val)} patients (Must be 0)")
    print(f"Overlap Train & Test:  {len(overlap_train_test)} patients (Must be 0)")
    print(f"Overlap Val & Test:    {len(overlap_val_test)} patients (Must be 0)")

    assert len(overlap_train_val) == 0, "CRITICAL ERROR: Patient overlap detected between train and val!"
    assert len(overlap_train_test) == 0, "CRITICAL ERROR: Patient overlap detected between train and test!"
    assert len(overlap_val_test) == 0, "CRITICAL ERROR: Patient overlap detected between val and test!"
    print(">>> VERIFIED: Zero patient overlap across splits. No data leakage.\n")

    # Show per-class counts across each split
    print("Per-Class Record Distribution Across Splits:")
    print("-" * 65)
    header = f"{'Class':<8} | {'Train (' + str(train_count) + ')':<16} | {'Val (' + str(val_count) + ')':<14} | {'Test (' + str(test_count) + ')':<14}"
    print(header)
    print("-" * 65)
    for sc in SUPERCLASSES:
        tr_c = int(subset_df[subset_df["split"] == "train"][sc].sum())
        va_c = int(subset_df[subset_df["split"] == "val"][sc].sum())
        te_c = int(subset_df[subset_df["split"] == "test"][sc].sum())
        tr_p = (tr_c / train_count) * 100
        va_p = (va_c / val_count) * 100
        te_p = (te_c / test_count) * 100
        print(f"{sc:<8} | {tr_c:>5} ({tr_p:>4.1f}%)     | {va_c:>4} ({va_p:>4.1f}%)   | {te_c:>4} ({te_p:>4.1f}%)")
    print("=" * 70 + "\n")

    return subset_df


def main():
    print("Running ECG Data Preparation (ecg_model/data_prep.py)...")
    # 1. Load and map
    df, _ = load_and_map_ptbxl(DATA_DIR)

    # 2. Show full dataset distribution
    display_full_dataset_counts(df)

    # 3. Select stratified ~5,000 subset
    subset_df = select_stratified_subset(
        df,
        target_records=5000,
        min_class_threshold=500,
        random_state=42
    )

    # 4. Split by patient_id into train/val/test
    subset_df = split_train_val_test_by_patient(
        subset_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    # Save metadata to disk
    os.makedirs(os.path.dirname(OUTPUT_METADATA_PATH), exist_ok=True)
    subset_df.to_csv(OUTPUT_METADATA_PATH)
    print(f"Saved subset metadata to {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
