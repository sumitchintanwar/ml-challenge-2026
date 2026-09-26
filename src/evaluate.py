"""
Entity-Level Evaluation Module for Amazon ML Challenge 2026.

Implements the official competition macro-averaged F0.5 metric:
  - F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
  - Calculated per Source 1 entity, then macro-averaged across ALL entities.
  - Singletons (no true matches) score 1.0 if predicted empty, 0.0 if any match is predicted.
  - Non-singletons score 0.0 if predicted empty or if true/predicted intersection is empty.
  - Threshold sweep over model prediction probabilities to maximize macro F0.5.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------

def compute_entity_f05(
    true_set: Set[str],
    pred_set: Set[str],
) -> Tuple[float, float, float]:
    """
    Compute entity-level (F0.5, Precision, Recall) according to the official
    competition evaluation specification.

    Rules:
      1. Singletons (true_set is empty):
         - pred_set is empty: F0.5 = 1.0, Precision = 1.0, Recall = 1.0 (correctly identified singleton)
         - pred_set is non-empty: F0.5 = 0.0, Precision = 0.0, Recall = 1.0 (false merge)
      2. Non-singletons (true_set is non-empty):
         - pred_set is empty: F0.5 = 0.0, Precision = 0.0, Recall = 0.0 (missed match)
         - pred_set is non-empty:
           Let TP = |true_set ∩ pred_set|
           If TP == 0: F0.5 = 0.0, Precision = 0.0, Recall = 0.0
           Else:
             Precision = TP / |pred_set|
             Recall = TP / |true_set|
             F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Returns:
        (f05, precision, recall)
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    # Case 1: Ground truth singleton
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0

    # Case 2: Ground truth non-singleton
    if n_pred == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    prec = tp / n_pred
    rec = tp / n_true
    denom = 0.25 * prec + rec
    f05 = (1.25 * prec * rec) / denom if denom > 0 else 0.0
    return f05, prec, rec


def evaluate_entity_predictions(
    gt_dict: Dict[str, Set[str]],
    preds_dict: Dict[str, Set[str]],
    eval_entity_ids: List[str],
    country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate predicted matches across a set of Source 1 entities and compute:
      - Overall Macro F0.5, Macro Precision, Macro Recall
      - Singleton accuracy (fraction of true singletons predicted as empty)
      - Matched entity Macro F0.5
      - Country-level breakdowns (if country_map provided)
    """
    f05_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []

    singleton_scores: List[float] = []
    matched_scores: List[float] = []

    by_country: Dict[str, Dict[str, List[float]]] = {}

    for eid in eval_entity_ids:
        t_set = gt_dict.get(eid, set())
        p_set = preds_dict.get(eid, set())

        f05, prec, rec = compute_entity_f05(t_set, p_set)

        f05_list.append(f05)
        prec_list.append(prec)
        rec_list.append(rec)

        if len(t_set) == 0:
            singleton_scores.append(f05)
        else:
            matched_scores.append(f05)

        if country_map:
            c = country_map.get(eid, "Other")
            if c not in by_country:
                by_country[c] = {"f05": [], "prec": [], "rec": []}
            by_country[c]["f05"].append(f05)
            by_country[c]["prec"].append(prec)
            by_country[c]["rec"].append(rec)

    total_n = len(eval_entity_ids)
    metrics = {
        "macro_f05": round(float(np.mean(f05_list)), 4) if f05_list else 0.0,
        "macro_precision": round(float(np.mean(prec_list)), 4) if prec_list else 0.0,
        "macro_recall": round(float(np.mean(rec_list)), 4) if rec_list else 0.0,
        "total_entities": total_n,
        "singletons_total": len(singleton_scores),
        "singletons_accuracy": round(float(np.mean(singleton_scores)), 4) if singleton_scores else 1.0,
        "matched_total": len(matched_scores),
        "matched_macro_f05": round(float(np.mean(matched_scores)), 4) if matched_scores else 0.0,
    }

    if country_map:
        metrics["by_country"] = {}
        for c, c_vals in by_country.items():
            metrics["by_country"][c] = {
                "entities": len(c_vals["f05"]),
                "macro_f05": round(float(np.mean(c_vals["f05"])), 4),
                "macro_precision": round(float(np.mean(c_vals["prec"])), 4),
                "macro_recall": round(float(np.mean(c_vals["rec"])), 4),
            }

    return metrics


# ---------------------------------------------------------------------------
# Threshold Sweep Pipeline
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    df_pairs: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    val_entity_ids: List[str],
    thresholds: List[float],
    country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Sweep probability thresholds over validation pairs and evaluate macro F0.5.
    Returns table of results and details at the optimal threshold.
    """
    sweep_results: List[Dict[str, Any]] = []
    best_threshold = thresholds[0]
    best_macro_f05 = -1.0
    best_metrics: Dict[str, Any] = {}

    print(f"\nEvaluating {len(thresholds)} thresholds across {len(val_entity_ids):,} validation entities...")
    print(f"{'Threshold':>10} | {'Macro F0.5':>11} | {'Macro Prec':>11} | {'Macro Rec':>11} | {'Singleton Acc':>14} | {'Matched F0.5':>13}")
    print("-" * 80)

    for thresh in thresholds:
        thresh = round(float(thresh), 4)

        # Filter pairs exceeding threshold
        above_df = df_pairs[df_pairs["prob"] >= thresh]

        # Group into predicted match sets
        if len(above_df) > 0:
            preds_dict = above_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
        else:
            preds_dict = {}

        metrics = evaluate_entity_predictions(
            gt_dict=gt_dict,
            preds_dict=preds_dict,
            eval_entity_ids=val_entity_ids,
            country_map=country_map,
        )

        res_entry = {
            "threshold": thresh,
            "macro_f05": metrics["macro_f05"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "singletons_accuracy": metrics["singletons_accuracy"],
            "matched_macro_f05": metrics["matched_macro_f05"],
        }
        sweep_results.append(res_entry)

        marker = " <-- OPTIMAL" if metrics["macro_f05"] > best_macro_f05 else ""
        print(f"{thresh:10.2f} | {metrics['macro_f05']:11.4f} | {metrics['macro_precision']:11.4f} | {metrics['macro_recall']:11.4f} | {metrics['singletons_accuracy']*100:13.2f}% | {metrics['matched_macro_f05']:13.4f}{marker}")

        if metrics["macro_f05"] > best_macro_f05:
            best_macro_f05 = metrics["macro_f05"]
            best_threshold = thresh
            best_metrics = metrics

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "best_metrics": best_metrics,
        "sweep_table": sweep_results,
    }


# ---------------------------------------------------------------------------
# Markdown Report Generation
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    sweep_data: Dict[str, Any],
    output_path: str = "notebooks/entity_evaluation_report.md",
) -> None:
    """Generate comprehensive markdown report of the entity-level threshold sweep."""
    best_t = sweep_data["best_threshold"]
    best_f05 = sweep_data["best_macro_f05"]
    m = sweep_data["best_metrics"]

    md = f"""# Entity-Level Evaluation & Threshold Sweep Report
Amazon ML Challenge 2026

## Target Metric: Validation Macro-Averaged $F_{{0.5}}$

The competition evaluates submissions using **Macro-Averaged $F_{{0.5}}$** across all Source 1 entities, including singletons:
$$F_{{0.5}} = \\frac{{1.25 \\cdot \\text{{Precision}} \\cdot \\text{{Recall}}}}{{0.25 \\cdot \\text{{Precision}} + \\text{{Recall}}}}$$

- **Optimal Classification Threshold**: **`{best_t:.2f}`**
- **Optimal Macro-Averaged $F_{{0.5}}$**: **`{best_f05:.4f}`** ({best_f05*100:.2f}%)
- **Macro Precision at Optimal Threshold**: **`{m['macro_precision']:.4f}`** ({m['macro_precision']*100:.2f}%)
- **Macro Recall at Optimal Threshold**: **`{m['macro_recall']:.4f}`** ({m['macro_recall']*100:.2f}%)
- **Singleton Accuracy**: **`{m['singletons_accuracy']*100:.2f}%`** ({m['singletons_total']:,} singletons evaluated)
- **Matched Entities $F_{{0.5}}$**: **`{m['matched_macro_f05']:.4f}`** ({m['matched_total']:,} matched entities evaluated)

---

## Threshold Sweep Trajectory (0.50 to 0.95, step = 0.05)

| Threshold | **Macro $F_{{0.5}}$** | Macro Precision | Macro Recall | Singleton Accuracy | Matched $F_{{0.5}}$ | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for row in sweep_data["sweep_table"]:
        t = row["threshold"]
        star = " **(OPTIMAL)**" if abs(t - best_t) < 1e-4 else ""
        md += f"| **{t:.2f}** | **{row['macro_f05']:.4f}** | {row['macro_precision']:.4f} | {row['macro_recall']:.4f} | {row['singletons_accuracy']*100:.2f}% | {row['matched_macro_f05']:.4f} |{star} |\n"

    if "by_country" in m:
        md += """
---

## Country-Level Performance at Optimal Threshold

| Country | Entities Evaluated | **Macro $F_{0.5}$** | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: |
"""
        for country, c_data in m["by_country"].items():
            md += f"| **{country}** | {c_data['entities']:,} | **{c_data['macro_f05']:.4f}** | {c_data['macro_precision']:.4f} | {c_data['macro_recall']:.4f} |\n"

    md += f"""
---

## Key Strategic Takeaways

1. **Precision-Recall Asymmetry**:
   - Because $F_{{0.5}}$ weights precision 2× over recall, raising the threshold from 0.50 to **{best_t:.2f}** yields a significant jump in overall Macro $F_{{0.5}}$ (from {sweep_data['sweep_table'][0]['macro_f05']:.4f} to **{best_f05:.4f}**, a **+{((best_f05 - sweep_data['sweep_table'][0]['macro_f05'])*100):.2f}% absolute improvement**).
2. **Singleton Protection**:
   - At threshold {best_t:.2f}, singleton accuracy reaches **{m['singletons_accuracy']*100:.2f}%**, eliminating false positive matches on unlinked entities.
3. **Threshold Calibration**:
   - The threshold of **{best_t:.2f}** should be used for all test set predictions to generate `matching_results.tsv`.
"""

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSaved evaluation report to {out_file}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate entity-level Macro F0.5 across threshold sweep.")
    parser.add_argument("--model", type=str, default="models/lgbm_matcher.pkl", help="Path to trained model pickle")
    parser.add_argument("--feature-matrix", type=str, default="data/processed/feature_matrix_train.parquet", help="Path to feature matrix")
    parser.add_argument("--split-file", type=str, default="data/processed/model_train_val_split.json", help="Train/Val split JSON")
    parser.add_argument("--ground-truth", type=str, default="data/raw/train_ground_truth.tsv", help="Ground truth TSV")
    parser.add_argument("--s1-data", type=str, default="data/processed/train_source1.parquet", help="Source 1 parquet for country metadata")
    parser.add_argument("--min-thresh", type=float, default=0.50, help="Min sweep threshold (default: 0.50)")
    parser.add_argument("--max-thresh", type=float, default=0.95, help="Max sweep threshold (default: 0.95)")
    parser.add_argument("--step-thresh", type=float, default=0.05, help="Sweep step (default: 0.05)")
    parser.add_argument("--report-path", type=str, default="notebooks/entity_evaluation_report.md", help="Path for markdown report")
    args = parser.parse_args()

    print("=" * 80)
    print("ENTITY-LEVEL EVALUATION & THRESHOLD OPTIMIZATION (MACRO F0.5)")
    print("=" * 80)

    # 1. Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as f:
        model_artifact = pickle.load(f)
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    print(f"  Loaded model: {type(model).__name__} ({len(feature_cols)} features)")

    # 2. Load validation entity split
    print(f"Loading validation split from {args.split_file}...")
    with open(args.split_file, "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    val_entity_ids = split_meta["val_entity_ids"]
    val_id_set = set(val_entity_ids)
    print(f"  Total validation Source 1 entities: {len(val_entity_ids):,}")

    # 3. Load feature matrix and filter to validation pairs
    print(f"Loading feature matrix from {args.feature_matrix}...")
    t0 = time.time()
    df_all = pd.read_parquet(args.feature_matrix, columns=["source1_entity_id", "candidate_entity_id"] + feature_cols)
    val_mask = df_all["source1_entity_id"].isin(val_id_set)
    df_val = df_all[val_mask].reset_index(drop=True)
    print(f"  Loaded {len(df_val):,} validation candidate pairs in {time.time()-t0:.2f}s")

    # 4. Predict probabilities
    print(f"Scoring {len(df_val):,} validation pairs with model...")
    t0 = time.time()
    probs = model.predict_proba(df_val[feature_cols].values)[:, 1]
    df_val["prob"] = probs.astype(np.float32)
    score_time = time.time() - t0
    print(f"  Scoring completed in {score_time:.2f}s ({len(df_val)/score_time:,.0f} pairs/sec)")

    # 5. Load Ground Truth mapping
    print(f"Loading ground truth mapping from {args.ground_truth}...")
    t0 = time.time()
    gt_df = pd.read_csv(args.ground_truth, sep="\t", usecols=["source1_entity_id", "matched_entity_ids"])
    gt_dict: Dict[str, Set[str]] = {}
    for s1, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if s1 in val_id_set:
            if pd.notna(m) and str(m).strip() != "":
                gt_dict[s1] = set(x.strip() for x in str(m).split(",") if x.strip())
            else:
                gt_dict[s1] = set()
    print(f"  Mapped ground truth for {len(gt_dict):,} validation entities in {time.time()-t0:.2f}s")

    # 6. Load Country Map
    print(f"Loading country metadata from {args.s1_data}...")
    s1_df = pd.read_parquet(args.s1_data, columns=["entity_id", "country"])
    country_map = dict(zip(s1_df["entity_id"], s1_df["country"]))

    # 7. Generate threshold grid (inclusive of max_thresh)
    steps = int(round((args.max_thresh - args.min_thresh) / args.step_thresh)) + 1
    thresholds = [round(args.min_thresh + i * args.step_thresh, 4) for i in range(steps)]

    # 8. Run Threshold Sweep
    sweep_data = run_threshold_sweep(
        df_pairs=df_val[["source1_entity_id", "candidate_entity_id", "prob"]],
        gt_dict=gt_dict,
        val_entity_ids=val_entity_ids,
        thresholds=thresholds,
        country_map=country_map,
    )

    # 9. Generate Report
    generate_evaluation_report(sweep_data, output_path=args.report_path)

    # 10. Update saved model artifact with optimal threshold
    best_t = sweep_data["best_threshold"]
    best_f05 = sweep_data["best_macro_f05"]
    model_artifact["optimal_threshold"] = best_t
    model_artifact["val_macro_f05"] = best_f05
    model_artifact["evaluation_sweep"] = sweep_data["sweep_table"]
    with open(args.model, "wb") as f:
        pickle.dump(model_artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Updated {args.model} with optimal threshold ({best_t:.2f}) and macro F0.5 ({best_f05:.4f})")

    print("\n" + "=" * 80)
    print("FINAL VALIDATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Target Metric (Validation Macro F0.5): {best_f05:.4f} ({best_f05*100:.2f}%)")
    print(f"Optimal Decision Threshold:            {best_t:.2f}")
    print(f"Macro Precision at Optimal Threshold:  {sweep_data['best_metrics']['macro_precision']:.4f}")
    print(f"Macro Recall at Optimal Threshold:     {sweep_data['best_metrics']['macro_recall']:.4f}")
    print(f"Singleton Accuracy:                    {sweep_data['best_metrics']['singletons_accuracy']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
