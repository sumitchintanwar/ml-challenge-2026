"""
Comprehensive Post-Fix Validation & Error Analysis Script.
Amazon ML Challenge 2026.

1. Fine-grained threshold sweep (0.50 to 0.99, step 0.01).
2. Ultra-fine sweep (step 0.001) around peak.
3. Macro-averaged entity F0.5, precision, recall, singleton accuracy, country breakdown.
4. Direct comparison against 0.8026 baseline.
5. Population-level FP/FN failure mode breakdown:
   - Adjacent/different street-number FP rate
   - Blocker recall ceiling / missed true pairs
   - Threshold-conservatism FN rate
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_entity_f05, evaluate_entity_predictions
from src.address_normalization import parse_street_and_unit_numbers, compare_street_and_unit_numbers


def main():
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION & THRESHOLD OPTIMIZATION (POST-FIX)")
    print("=" * 80)

    # 1. Load model
    model_path = "models/lgbm_matcher.pkl"
    print(f"Loading trained LightGBM model from {model_path}...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    print(f"  Model loaded: {len(feature_cols)} features")

    # 2. Load validation split
    split_path = "data/processed/model_train_val_split.json"
    print(f"Loading validation split from {split_path}...")
    with open(split_path, "r") as f:
        split_data = json.load(f)
    val_entity_ids = split_data["val_entity_ids"]
    val_id_set = set(val_entity_ids)
    print(f"  Validation Source 1 entities: {len(val_entity_ids):,}")

    # 3. Load country mapping for validation entities
    print("Loading country metadata from data/processed/train_source1.parquet...")
    s1_df = pd.read_parquet("data/processed/train_source1.parquet", columns=["entity_id", "country"])
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_id_set)]
    country_map = dict(zip(val_s1_df["entity_id"], val_s1_df["country"]))

    # 4. Load full ground truth mapping
    print("Loading ground truth from data/raw/train_ground_truth.tsv...")
    gt_df = pd.read_csv("data/raw/train_ground_truth.tsv", sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        if m_str:
            gt_dict[s1] = set(x.strip() for x in m_str.split(",") if x.strip())
        else:
            gt_dict[s1] = set()

    # 5. Load feature matrix and filter to validation entities
    print("Loading validation candidate pairs from data/processed/feature_matrix_train.parquet...")
    t0 = time.time()
    cols_to_load = ["source1_entity_id", "candidate_entity_id", "label"] + feature_cols
    df_all = pd.read_parquet("data/processed/feature_matrix_train.parquet", columns=cols_to_load)
    val_mask = df_all["source1_entity_id"].isin(val_id_set)
    df_val = df_all[val_mask].copy().reset_index(drop=True)
    print(f"  Loaded {len(df_val):,} validation candidate pairs in {time.time()-t0:.2f}s")

    # 6. Score validation pairs
    print("Predicting probabilities on validation candidate pairs...")
    t0 = time.time()
    X_val = df_val[feature_cols].values
    probs = model.predict_proba(X_val)[:, 1]
    df_val["prob"] = probs.astype(np.float32)
    print(f"  Scoring completed in {time.time()-t0:.2f}s ({len(df_val)/(time.time()-t0):,.0f} pairs/sec)")

    # -----------------------------------------------------------------------
    # Step 1: Coarse Sweep (0.50 to 0.99, step 0.01)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: COARSE THRESHOLD SWEEP (0.50 to 0.99, step 0.01)")
    print("=" * 80)

    coarse_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.01)]
    coarse_results = []

    best_coarse_t = 0.50
    best_coarse_f05 = -1.0

    for t in coarse_thresholds:
        above = df_val[df_val["prob"] >= t]
        preds_dict = above.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict() if len(above) > 0 else {}
        m = evaluate_entity_predictions(gt_dict, preds_dict, val_entity_ids, country_map)
        coarse_results.append({
            "threshold": t,
            "macro_f05": m["macro_f05"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "singletons_accuracy": m["singletons_accuracy"],
            "matched_macro_f05": m["matched_macro_f05"],
        })
        if m["macro_f05"] > best_coarse_f05:
            best_coarse_f05 = m["macro_f05"]
            best_coarse_t = t

    # Sort and print top 5 coarse thresholds
    sorted_coarse = sorted(coarse_results, key=lambda x: x["macro_f05"], reverse=True)
    print("\nTop 5 Coarse Thresholds (by Macro F0.5):")
    for i, r in enumerate(sorted_coarse[:5], 1):
        print(f"  {i}. Threshold {r['threshold']:.2f}: Macro F0.5 = {r['macro_f05']:.4f} | Prec = {r['macro_precision']:.4f} | Rec = {r['macro_recall']:.4f} | Singleton Acc = {r['singletons_accuracy']*100:.2f}%")

    # -----------------------------------------------------------------------
    # Step 2: Fine-grained Sweep (step 0.001) around peak
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STEP 2: ULTRA-FINE SWEEP (step 0.001) around peak {best_coarse_t:.2f}")
    print("=" * 80)

    fine_min = max(0.50, round(best_coarse_t - 0.03, 3))
    fine_max = min(0.999, round(best_coarse_t + 0.03, 3))
    fine_thresholds = [round(t, 3) for t in np.arange(fine_min, fine_max + 0.0005, 0.001)]

    fine_results = []
    best_opt_t = best_coarse_t
    best_opt_f05 = -1.0
    best_opt_metrics = None

    for t in fine_thresholds:
        above = df_val[df_val["prob"] >= t]
        preds_dict = above.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict() if len(above) > 0 else {}
        m = evaluate_entity_predictions(gt_dict, preds_dict, val_entity_ids, country_map)
        res_entry = {
            "threshold": t,
            "macro_f05": m["macro_f05"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "singletons_accuracy": m["singletons_accuracy"],
            "matched_macro_f05": m["matched_macro_f05"],
            "by_country": m.get("by_country", {}),
        }
        fine_results.append(res_entry)
        if m["macro_f05"] > best_opt_f05:
            best_opt_f05 = m["macro_f05"]
            best_opt_t = t
            best_opt_metrics = m

    print(f"\nTRUE OPTIMAL THRESHOLD FOUND: {best_opt_t:.3f}")
    print(f"  Macro F0.5:        {best_opt_metrics['macro_f05']:.4f} ({best_opt_metrics['macro_f05']*100:.2f}%)")
    print(f"  Macro Precision:   {best_opt_metrics['macro_precision']:.4f} ({best_opt_metrics['macro_precision']*100:.2f}%)")
    print(f"  Macro Recall:      {best_opt_metrics['macro_recall']:.4f} ({best_opt_metrics['macro_recall']*100:.2f}%)")
    print(f"  Singleton Acc:     {best_opt_metrics['singletons_accuracy']*100:.2f}%")
    print(f"  Matched F0.5:      {best_opt_metrics['matched_macro_f05']:.4f}")

    if "by_country" in best_opt_metrics:
        print("\nCountry Breakdown at Optimum:")
        for c, c_info in best_opt_metrics["by_country"].items():
            print(f"  {c} ({c_info['entities']} entities): F0.5 = {c_info['macro_f05']:.4f} | Prec = {c_info['macro_precision']:.4f} | Rec = {c_info['macro_recall']:.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Direct Baseline Comparison
    # -----------------------------------------------------------------------
    baseline_f05 = 0.8017  # Baseline at 0.90 threshold
    gain = (best_opt_metrics["macro_f05"] - baseline_f05) * 100
    print("\n" + "=" * 80)
    print("COMPARISON AGAINST BASELINE (0.8017 / 0.8026)")
    print("=" * 80)
    print(f"  Baseline Macro F0.5:       {baseline_f05:.4f} (80.17%)")
    print(f"  New Optimal Macro F0.5:    {best_opt_metrics['macro_f05']:.4f} ({best_opt_metrics['macro_f05']*100:.2f}%)")
    print(f"  Net Absolute Gain:         {gain:+.2f}%")

    # -----------------------------------------------------------------------
    # Step 4: Population-Level FP/FN Failure Mode Breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POPULATION-LEVEL FAILURE MODE RE-EVALUATION")
    print("=" * 80)

    # A. Adjacent/Different Street Number FP Breakdown
    # At the optimal threshold, examine all validation FP pairs
    val_preds_opt = df_val[df_val["prob"] >= best_opt_t]
    fp_pairs = val_preds_opt[val_preds_opt["label"] == 0]
    total_fps = len(fp_pairs)
    print(f"Total Validation FP Pairs at threshold {best_opt_t}: {total_fps:,}")

    # How many FP pairs have street_number_numeric_distance > 0?
    diff_street_fps = len(fp_pairs[fp_pairs["street_number_numeric_distance"] > 0])
    diff_street_pct = (diff_street_fps / total_fps * 100) if total_fps > 0 else 0
    print(f"  Adjacent/Different Street Number FPs remaining: {diff_street_fps:,} ({diff_street_pct:.2f}% of FPs)")
    print(f"  (Previously, adjacent street numbers comprised ~64% of top high-confidence FPs!)")

    # B. Blocker Recall / Missed True Pairs
    # Total true pairs in validation split
    val_true_pairs_total = sum(len(gt_dict.get(eid, set())) for eid in val_entity_ids)
    val_pos_in_candidates = int(df_val["label"].sum())
    blocker_missed_count = val_true_pairs_total - val_pos_in_candidates
    blocker_miss_pct = (blocker_missed_count / val_true_pairs_total * 100) if val_true_pairs_total > 0 else 0

    print(f"\nValidation True Pairs Total: {val_true_pairs_total:,}")
    print(f"  Candidate Generator Retrieved: {val_pos_in_candidates:,} ({val_pos_in_candidates/val_true_pairs_total*100:.2f}%)")
    print(f"  Blocker-Missed True Pairs:     {blocker_missed_count:,} ({blocker_miss_pct:.2f}% of true pairs)")
    print(f"  (In the blocker recovery evaluation with transliteration + prefix-stripping, recall ceiling increased to 99.53%)")

    # C. Threshold Conservatism FN Rate
    # True pairs in candidates that were rejected because prob < best_opt_t
    cand_pos = df_val[df_val["label"] == 1]
    borderline_fns = len(cand_pos[(cand_pos["prob"] >= 0.50) & (cand_pos["prob"] < best_opt_t)])
    deep_miss_fns = len(cand_pos[cand_pos["prob"] < 0.50])
    total_cand_fns = len(cand_pos[cand_pos["prob"] < best_opt_t])
    borderline_fn_pct = (borderline_fns / total_cand_fns * 100) if total_cand_fns > 0 else 0

    print(f"\nModel-Rejected True Pairs (Candidate FNs at threshold {best_opt_t}): {total_cand_fns:,}")
    print(f"  Borderline (0.50 <= prob < {best_opt_t}): {borderline_fns:,} ({borderline_fn_pct:.2f}%)")
    print(f"  Deep misses (prob < 0.50):               {deep_miss_fns:,} ({100-borderline_fn_pct:.2f}%)")

    # -----------------------------------------------------------------------
    # Step 5: Save JSON & Markdown Reports
    # -----------------------------------------------------------------------
    output_report = {
        "best_threshold": best_opt_t,
        "best_macro_f05": best_opt_metrics["macro_f05"],
        "best_macro_precision": best_opt_metrics["macro_precision"],
        "best_macro_recall": best_opt_metrics["macro_recall"],
        "singletons_accuracy": best_opt_metrics["singletons_accuracy"],
        "matched_macro_f05": best_opt_metrics["matched_macro_f05"],
        "by_country": best_opt_metrics.get("by_country", {}),
        "coarse_top_5": sorted_coarse[:5],
        "fine_sweep": fine_results,
        "failure_mode_breakdown": {
            "total_val_fps": total_fps,
            "adjacent_street_fps": diff_street_fps,
            "adjacent_street_fp_pct": round(diff_street_pct, 2),
            "val_true_pairs_total": val_true_pairs_total,
            "candidate_positives": val_pos_in_candidates,
            "blocker_missed_count": blocker_missed_count,
            "blocker_miss_pct": round(blocker_miss_pct, 2),
            "candidate_fns_total": total_cand_fns,
            "threshold_conservatism_fns": borderline_fns,
            "threshold_conservatism_pct": round(borderline_fn_pct, 2),
            "deep_miss_fns": deep_miss_fns,
        },
    }

    with open("notebooks/post_fix_validation_report.json", "w") as f:
        json.dump(output_report, f, indent=2)
    print("\nSaved full validation report to notebooks/post_fix_validation_report.json")


if __name__ == "__main__":
    main()
