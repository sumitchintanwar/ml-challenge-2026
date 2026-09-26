"""
Training and Validation Module for Entity Resolution LightGBM Matcher.
Amazon ML Challenge 2026.

Performs:
  1. 80/20 Entity-level stratified split (Source 1 entity_id, stratified by country & singleton status).
  2. Training LightGBM classifier on train-split candidate pairs with scale_pos_weight.
  3. Validation on held-out val-split candidate pairs.
  4. Calculation and reporting of:
     - AUC-PR (Average Precision)
     - AUC-ROC
     - Confusion Matrix at threshold 0.5 (TN, FP, FN, TP)
     - Precision, Recall, F1, and F0.5 (Competition Metric)
     - Feature Importances (Gain and Split)
  5. Persistence of trained model to models/lgbm_matcher.pkl.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import json
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_COLUMNS


def get_or_create_entity_split(
    s1_entities: List[str],
    s1_parquet_path: str = "data/processed/train_source1.parquet",
    gt_tsv_path: str = "data/raw/train_ground_truth.tsv",
    split_output_path: str = "data/processed/model_train_val_split.json",
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Create an 80/20 entity-level split of unique Source 1 entities,
    stratified by country and singleton status.
    """
    unique_entities = sorted(list(set(s1_entities)))
    print(f"Total unique Source 1 entities to split: {len(unique_entities):,}")

    # Load country mapping
    print(f"Loading country metadata from {s1_parquet_path}...")
    s1_df = pd.read_parquet(s1_parquet_path, columns=["entity_id", "country"])
    country_map = dict(zip(s1_df["entity_id"], s1_df["country"]))

    # Load singleton status from ground truth
    print(f"Loading ground truth singleton status from {gt_tsv_path}...")
    gt_df = pd.read_csv(gt_tsv_path, sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt_set = set(gt_df[gt_df["matched_entity_ids"].notna() & (gt_df["matched_entity_ids"].astype(str).str.strip() != "")]["source1_entity_id"])

    # Build strata: country + is_singleton
    strata = []
    for eid in unique_entities:
        c = country_map.get(eid, "UNKNOWN")
        is_matched = eid in gt_set
        strata.append(f"{c}_{'matched' if is_matched else 'singleton'}")

    # Stratified train/val split
    train_eids, val_eids = train_test_split(
        unique_entities,
        test_size=test_size,
        random_state=random_seed,
        stratify=strata,
    )

    train_set = set(train_eids)
    val_set = set(val_eids)

    # Sanity check: zero overlap
    assert len(train_set.intersection(val_set)) == 0, "Leakage detected: overlap between train and val entities!"

    metadata = {
        "random_seed": random_seed,
        "test_size": test_size,
        "total_entities": len(unique_entities),
        "train_count": len(train_eids),
        "val_count": len(val_eids),
        "train_pct": round(len(train_eids) / len(unique_entities) * 100, 2),
        "val_pct": round(len(val_eids) / len(unique_entities) * 100, 2),
    }

    print(f"Entity split completed:")
    print(f"  Train entities: {len(train_eids):,} ({metadata['train_pct']}%)")
    print(f"  Val entities:   {len(val_eids):,} ({metadata['val_pct']}%)")

    # Persist split metadata
    out_file = Path(split_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": metadata,
            "train_entity_ids": train_eids,
            "val_entity_ids": val_eids,
        }, f, indent=2)
    print(f"Saved entity split to {split_output_path}")

    return train_eids, val_eids, metadata


def train_and_validate_lgbm(
    feature_matrix_path: str = "data/processed/feature_matrix_train.parquet",
    model_output_path: str = "models/lgbm_matcher.pkl",
    s1_parquet_path: str = "data/processed/train_source1.parquet",
    gt_tsv_path: str = "data/raw/train_ground_truth.tsv",
    split_path: str = "data/processed/model_train_val_split.json",
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train LightGBM binary classifier on train-split entities and evaluate on val-split entities.
    """
    print("=" * 70)
    print("LIGHTGBM ENTITY RESOLUTION MATCHER: TRAINING & VALIDATION")
    print("=" * 70)

    # 1. Load Feature Matrix
    print(f"Loading feature matrix from {feature_matrix_path}...")
    t0 = time.time()
    df = pd.read_parquet(feature_matrix_path)
    print(f"  Loaded {len(df):,} pairs ({len(df.columns)} columns) in {time.time()-t0:.2f}s")

    # Check available features
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    # 2. Entity-level split
    train_eids, val_eids, split_meta = get_or_create_entity_split(
        s1_entities=df["source1_entity_id"].tolist(),
        s1_parquet_path=s1_parquet_path,
        gt_tsv_path=gt_tsv_path,
        split_output_path=split_path,
        test_size=test_size,
        random_seed=random_seed,
    )

    train_set = set(train_eids)
    val_set = set(val_eids)

    print("\nPartitioning feature matrix by entity-level split...")
    train_mask = df["source1_entity_id"].isin(train_set)
    val_mask = df["source1_entity_id"].isin(val_set)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    print(f"  Train pairs: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val pairs:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")

    # 3. Class Imbalance and scale_pos_weight
    n_pos_train = int(train_df["label"].sum())
    n_neg_train = len(train_df) - n_pos_train
    pos_ratio_train = n_pos_train / len(train_df)
    scale_pos_weight = float(n_neg_train / n_pos_train) if n_pos_train > 0 else 1.0

    n_pos_val = int(val_df["label"].sum())
    n_neg_val = len(val_df) - n_pos_val
    pos_ratio_val = n_pos_val / len(val_df)

    print("\nClass Distribution:")
    print(f"  Train: Positives={n_pos_train:,} ({pos_ratio_train*100:.2f}%), Negatives={n_neg_train:,} ({100-pos_ratio_train*100:.2f}%)")
    print(f"         Class Imbalance = 1 : {scale_pos_weight:.2f}")
    print(f"         Configured scale_pos_weight = {scale_pos_weight:.4f}")
    print(f"  Val:   Positives={n_pos_val:,} ({pos_ratio_val*100:.2f}%), Negatives={n_neg_val:,} ({100-pos_ratio_val*100:.2f}%)")

    # 4. Prepare Feature Arrays
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(np.int32)

    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values.astype(np.int32)

    # 5. Initialize and Train LightGBM Classifier
    print("\nTraining LightGBM binary classifier...")
    lgbm_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**lgbm_params)

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric=["binary_logloss", "average_precision"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t0
    best_iter = model.best_iteration_
    print(f"Model trained in {train_time:.2f}s (Best iteration: {best_iter})")

    # 6. Evaluation on Held-Out Validation Pairs
    print("\nEvaluating model on held-out validation pairs...")
    t0 = time.time()
    val_probs = model.predict_proba(X_val)[:, 1]
    eval_time = time.time() - t0
    print(f"Scored {len(val_df):,} validation pairs in {eval_time:.2f}s ({len(val_df)/eval_time:,.0f} pairs/sec)")

    # Metrics
    auc_pr = average_precision_score(y_val, val_probs)
    auc_roc = roc_auc_score(y_val, val_probs)

    # Confusion matrix and metrics at threshold 0.5
    threshold = 0.5
    val_preds = (val_probs >= threshold).astype(np.int32)
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_val, val_preds, zero_division=0)
    recall = recall_score(y_val, val_preds, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # Competition metric: F0.5
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

    # Feature Importances
    importances_gain = model.booster_.feature_importance(importance_type="gain")
    importances_split = model.booster_.feature_importance(importance_type="split")
    total_gain = sum(importances_gain)
    feat_imp = []
    for col, gain, split in zip(feature_cols, importances_gain, importances_split):
        pct_gain = (gain / total_gain * 100) if total_gain > 0 else 0.0
        feat_imp.append({"feature": col, "gain": round(gain, 2), "gain_pct": round(pct_gain, 2), "split": split})
    feat_imp = sorted(feat_imp, key=lambda x: x["gain"], reverse=True)

    # 7. Print Comprehensive Report
    print("\n" + "=" * 70)
    print("VALIDATION METRICS REPORT (HELD-OUT VAL SPLIT)")
    print("=" * 70)
    print(f"Validation Pairs Evaluated: {len(val_df):,}")
    print(f"Positive Validation Pairs:  {n_pos_val:,} ({pos_ratio_val*100:.2f}%)")
    print(f"Negative Validation Pairs:  {n_neg_val:,} ({100-pos_ratio_val*100:.2f}%)")
    print("-" * 70)
    print(f"AUC-PR (Average Precision): {auc_pr:.4f}  <-- PRIMARY PRECISION-RECALL AREA")
    print(f"AUC-ROC:                    {auc_roc:.4f}")
    print("-" * 70)
    print("Confusion Matrix (at threshold = 0.5):")
    print(f"                    Predicted Negative    Predicted Positive")
    print(f"  Actual Negative   TN = {tn:9,d}        FP = {fp:9,d}    (Total Neg: {n_neg_val:,})")
    print(f"  Actual Positive   FN = {fn:9,d}        TP = {tp:9,d}    (Total Pos: {n_pos_val:,})")
    print("-" * 70)
    print(f"Precision (at 0.5):         {precision*100:.2f}%")
    print(f"Recall (at 0.5):            {recall*100:.2f}%")
    print(f"F1-Score (at 0.5):          {f1*100:.2f}%")
    print(f"F0.5-Score (at 0.5):        {f05*100:.2f}%  <-- COMPETITION OBJECTIVE")
    print("-" * 70)
    print("\nTop Feature Importances (by Gain):")
    for i, fi in enumerate(feat_imp, 1):
        print(f"  {i:2d}. {fi['feature']:<26} Gain %: {fi['gain_pct']:5.2f}%  (Split: {fi['split']:4d})")

    # 8. Save Trained Model
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feature_cols,
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": best_iter,
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "f05": f05,
            "params": lgbm_params,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nTrained model successfully saved to {model_path} ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")

    results = {
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": best_iter,
        "auc_pr": round(auc_pr, 4),
        "auc_roc": round(auc_roc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": threshold,
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "f1_pct": round(f1 * 100, 2),
        "f05_pct": round(f05 * 100, 2),
        "feature_importances": feat_imp,
        "model_path": str(model_path),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and validate LightGBM matcher.")
    parser.add_argument("--features-train", type=str, default="data/processed/feature_matrix_train.parquet", help="Path to train feature matrix")
    parser.add_argument("--model-out", type=str, default="models/lgbm_matcher.pkl", help="Path to save trained model")
    parser.add_argument("--s1-data", type=str, default="data/processed/train_source1.parquet", help="Source 1 parquet for country metadata")
    parser.add_argument("--ground-truth", type=str, default="data/raw/train_ground_truth.tsv", help="Ground truth TSV")
    parser.add_argument("--split-out", type=str, default="data/processed/model_train_val_split.json", help="Output split metadata")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction for validation split")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_and_validate_lgbm(
        feature_matrix_path=args.features_train,
        model_output_path=args.model_out,
        s1_parquet_path=args.s1_data,
        gt_tsv_path=args.ground_truth,
        split_path=args.split_out,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
