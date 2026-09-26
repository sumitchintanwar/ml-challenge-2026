"""
Test Set Inference and Submission Generator for Amazon ML Challenge 2026.

Generates the two official competition submission files in output/:
  1. output/matching_results.tsv:
     - Header: source1_entity_id <TAB> matched_entity_ids
     - One row per Source 1 test entity
     - Empty string for singletons
     - Comma-joined entity IDs with no quoting, no duplicates
  2. output/candidate_pairs.tsv:
     - Header: source1_entity_id <TAB> candidate_entity_ids
     - Exactly the candidate set fed into the model
     - Every matched entity ID is guaranteed to be a subset of candidate IDs
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_inference(
    model_path: str = "models/lgbm_matcher.pkl",
    feature_matrix_path: str = "data/processed/feature_matrix_test.parquet",
    candidate_pairs_path: str = "data/processed/candidate_pairs_test.parquet",
    test_source1_path: str = "data/raw/student_resource/dataset/test/test_source1.tsv",
    output_dir: str = "output",
    threshold: Optional[float] = None,
    train_gt_path: Optional[str] = "data/raw/train_ground_truth.tsv",
) -> Dict[str, Any]:
    """
    Run full test inference and generate submission files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    # 1. Load model artifact
    print(f"Loading model artifact from {model_path}...")
    t0 = time.time()
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    
    if threshold is None:
        threshold = float(artifact.get("optimal_threshold", 0.90))
    print(f"  Loaded model: {type(model).__name__} ({len(feature_cols)} features)")
    print(f"  Using decision threshold: {threshold:.4f}")

    # 2. Load test candidate pairs
    print(f"\nLoading test candidate pairs from {candidate_pairs_path}...")
    t0 = time.time()
    df_cp = pd.read_parquet(candidate_pairs_path, columns=["source1_entity_id", "candidate_entity_id"])
    print(f"  Loaded {len(df_cp):,} candidate pairs in {time.time()-t0:.2f}s")

    # Build candidate lookup dictionary: s1_id -> list of candidate_ids (deduped, order preserved)
    print("  Aggregating candidates per Source 1 entity...")
    t0 = time.time()
    cand_dict: Dict[str, List[str]] = {}
    for s1, c in zip(df_cp["source1_entity_id"], df_cp["candidate_entity_id"]):
        if s1 not in cand_dict:
            cand_dict[s1] = []
        cand_dict[s1].append(c)
    # Deduplicate while preserving order
    for s1, clist in cand_dict.items():
        cand_dict[s1] = list(dict.fromkeys(clist))
    print(f"  Found {len(cand_dict):,} unique Source 1 entities with candidates in {time.time()-t0:.2f}s")

    # 3. Load feature matrix and score pairs
    print(f"\nLoading feature matrix from {feature_matrix_path}...")
    t0 = time.time()
    df_fm = pd.read_parquet(feature_matrix_path, columns=["source1_entity_id", "candidate_entity_id"] + feature_cols)
    print(f"  Loaded {len(df_fm):,} feature rows in {time.time()-t0:.2f}s")

    print(f"Scoring candidate pairs with model...")
    t0 = time.time()
    probs = model.predict_proba(df_fm[feature_cols].values)[:, 1]
    df_fm["prob"] = probs.astype(np.float32)
    score_time = time.time() - t0
    print(f"  Scored {len(df_fm):,} pairs in {score_time:.2f}s ({len(df_fm)/score_time:,.0f} pairs/sec)")

    # 4. Filter matches exceeding threshold
    print(f"\nFiltering matches at threshold >= {threshold:.4f}...")
    df_matched = df_fm[df_fm["prob"] >= threshold]
    print(f"  Accepted {len(df_matched):,} / {len(df_fm):,} pairs ({len(df_matched)/len(df_fm)*100:.2f}%)")

    # Build matched lookup dictionary: s1_id -> list of matched_ids (sorted by prob desc, deduped)
    print("  Aggregating predictions per Source 1 entity...")
    t0 = time.time()
    matched_dict: Dict[str, List[str]] = {}
    # Sort matched pairs by prob descending so best matches appear first
    df_matched_sorted = df_matched.sort_values(["source1_entity_id", "prob"], ascending=[True, False])
    for s1, c in zip(df_matched_sorted["source1_entity_id"], df_matched_sorted["candidate_entity_id"]):
        if s1 not in matched_dict:
            matched_dict[s1] = []
        if c not in matched_dict[s1]:
            matched_dict[s1].append(c)
    print(f"  Found {len(matched_dict):,} unique Source 1 entities with predicted matches in {time.time()-t0:.2f}s")

    # 5. Stream test_source1 to produce output files
    print(f"\nGenerating submission files from {test_source1_path}...")
    t0 = time.time()
    total_test_entities = 0
    full_singletons = 0
    full_matched = 0

    with open(test_source1_path, "r", encoding="utf-8") as f_in, \
         open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:
        
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        # Skip header of test_source1
        header = f_in.readline()

        for line in f_in:
            if not line.strip():
                continue
            s1_id = line.split("\t", 1)[0].strip()
            total_test_entities += 1

            # Get predicted matches
            m_list = matched_dict.get(s1_id, [])
            m_str = ",".join(m_list) if m_list else ""
            f_match.write(f"{s1_id}\t{m_str}\n")

            # Get candidate list
            c_list = cand_dict.get(s1_id, [])
            c_str = ",".join(c_list) if c_list else ""
            f_cand.write(f"{s1_id}\t{c_str}\n")

            if m_str:
                full_matched += 1
            else:
                full_singletons += 1

    gen_time = time.time() - t0
    print(f"  Successfully wrote {total_test_entities:,} rows to {matching_file} and {candidate_file} in {gen_time:.2f}s")

    # 6. Compute Statistics & Comparison with Ground Truth
    scored_s1_count = len(cand_dict)
    scored_matched_count = len(matched_dict)
    scored_singletons_count = scored_s1_count - scored_matched_count
    scored_singleton_rate = (scored_singletons_count / scored_s1_count) * 100 if scored_s1_count > 0 else 0.0

    full_singleton_rate = (full_singletons / total_test_entities) * 100 if total_test_entities > 0 else 0.0

    # Ground truth singleton rate
    gt_singleton_rate = None
    gt_total = None
    gt_singletons = None
    if train_gt_path and os.path.isfile(train_gt_path):
        print(f"\nComputing ground truth singleton rate from {train_gt_path}...")
        t0 = time.time()
        gt_df = pd.read_csv(train_gt_path, sep="\t", usecols=["matched_entity_ids"])
        gt_total = len(gt_df)
        gt_singletons = int((gt_df["matched_entity_ids"].isna() | (gt_df["matched_entity_ids"].str.strip() == "")).sum())
        gt_singleton_rate = (gt_singletons / gt_total) * 100
        print(f"  Loaded GT in {time.time()-t0:.2f}s: {gt_singletons:,} singletons / {gt_total:,} entities ({gt_singleton_rate:.2f}%)")

    stats = {
        "threshold": threshold,
        "matching_file": str(matching_file),
        "candidate_file": str(candidate_file),
        "total_test_entities": total_test_entities,
        "full_singletons": full_singletons,
        "full_singleton_rate": full_singleton_rate,
        "full_matched": full_matched,
        "scored_s1_count": scored_s1_count,
        "scored_singletons_count": scored_singletons_count,
        "scored_singleton_rate": scored_singleton_rate,
        "scored_matched_count": scored_matched_count,
        "gt_total": gt_total,
        "gt_singletons": gt_singletons,
        "gt_singleton_rate": gt_singleton_rate,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("SUBMISSION INFERENCE & SANITY CHECK SUMMARY")
    print("=" * 80)
    print(f"Trained Model:               {model_path}")
    print(f"Classification Threshold:    {threshold:.2f}")
    print(f"Total Test Source 1 Rows:    {total_test_entities:,}")
    print(f"Entities Scored via Blocker: {scored_s1_count:,}")
    print(f"Entities with Matches Found: {full_matched:,} ({full_matched/total_test_entities*100:.2f}%)")
    print("-" * 80)
    print(f"Sanity Check: Singleton Rate Comparison:")
    if gt_singleton_rate is not None:
        print(f"  - Train Ground Truth Singleton Rate:        {gt_singleton_rate:.2f}% ({gt_singletons:,} / {gt_total:,})")
    print(f"  - Test (Among Scored Queries) Singleton Rate: {scored_singleton_rate:.2f}% ({scored_singletons_count:,} / {scored_s1_count:,})")
    print(f"  - Test (Full Set, All 1.73M) Singleton Rate:  {full_singleton_rate:.2f}% ({full_singletons:,} / {total_test_entities:,})")
    print("=" * 80)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run test inference and produce competition submission files.")
    parser.add_argument("--model", type=str, default="models/lgbm_matcher.pkl", help="Path to trained LightGBM model")
    parser.add_argument("--feature-matrix", type=str, default="data/processed/feature_matrix_test.parquet", help="Path to test feature matrix")
    parser.add_argument("--candidate-pairs", type=str, default="data/processed/candidate_pairs_test.parquet", help="Path to test candidate pairs")
    parser.add_argument("--test-source1", type=str, default="data/raw/student_resource/dataset/test/test_source1.tsv", help="Path to test_source1.tsv")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save submission files")
    parser.add_argument("--threshold", type=float, default=None, help="Decision probability threshold (default: optimal from model)")
    parser.add_argument("--train-gt", type=str, default="data/raw/train_ground_truth.tsv", help="Path to train ground truth TSV")
    parser.add_argument("--validate", action="store_true", default=True, help="Run utils/validate_submission.py check")
    args = parser.parse_args()

    stats = run_inference(
        model_path=args.model,
        feature_matrix_path=args.feature_matrix,
        candidate_pairs_path=args.candidate_pairs,
        test_source1_path=args.test_source1,
        output_dir=args.output_dir,
        threshold=args.threshold,
        train_gt_path=args.train_gt,
    )

    # Run official validator script
    validator_script = Path("data/raw/student_resource/utils/validate_submission.py")
    if args.validate and validator_script.is_file():
        print(f"\nRunning official submission validator ({validator_script})...")
        cmd = (
            f"python3 {validator_script} "
            f"--matching {stats['matching_file']} "
            f"--candidate {stats['candidate_file']} "
            f"--test-dir data/raw/student_resource/dataset/test"
        )
        print(f"Command: {cmd}")
        res = os.system(cmd)
        if res == 0:
            print("\n>>> VALIDATION RESULT: PASSED ALL CHECKS! READY FOR LEADERBOARD SUBMISSION <<<")
        else:
            print(f"\n>>> VALIDATION WARNING / ERROR: Exit code {res} <<<")


if __name__ == "__main__":
    main()
