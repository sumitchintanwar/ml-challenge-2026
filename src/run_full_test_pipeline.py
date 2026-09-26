"""
Full-Scale End-to-End Test Inference Pipeline for Amazon ML Challenge 2026.

Processes ALL 1,732,544 test Source 1 entities against the 9.9M Source 2/3 corpus:
  1. Multi-strategy candidate blocking across all 7 CPU cores (zero IPC serialization overhead).
  2. In-memory Rapidfuzz + TF-IDF feature extraction.
  3. LightGBM scoring at optimal threshold theta* = 0.90.
  4. Direct streaming to output/matching_results.tsv and output/candidate_pairs.tsv.
  5. Full compliance with official competition validator rules.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import multiprocessing as mp
import pickle
import sys
import time
import traceback

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureExtractor, FEATURE_COLUMNS, compute_street_unit_pair_metrics


# Global variables for shared COW memory across forked processes
_BLOCKER = None
_MODEL = None
_EXTRACTOR = None
_THRESHOLD = 0.879
_S1_DF = None
_CORPUS_NAMES = None
_CORPUS_ADDRS = None
_CORPUS_COUNTRIES = None


def _worker_process_shard(args: Tuple[int, int, int, str]) -> Dict[str, Any]:
    """
    Worker function to process a contiguous shard of test queries end-to-end:
    Blocking -> Feature Extraction -> Model Scoring -> Output Writing.
    """
    try:
        shard_id, start_idx, end_idx, parts_dir = args
        n_queries = end_idx - start_idx

        match_part_path = Path(parts_dir) / f"matching_part_{shard_id:02d}.tsv"
        cand_part_path = Path(parts_dir) / f"candidate_part_{shard_id:02d}.tsv"

        # Vectorizer for character 3-gram TF-IDF cosine
        vec = _EXTRACTOR.vectorizer

        t0 = time.time()
        batch_size = 2000
        total_matches_found = 0
        total_singletons = 0
        total_candidates = 0

        with open(match_part_path, "w", encoding="utf-8") as f_match, \
             open(cand_part_path, "w", encoding="utf-8") as f_cand:

            for b_start in range(start_idx, end_idx, batch_size):
                b_end = min(b_start + batch_size, end_idx)
                sub_s1 = _S1_DF.iloc[b_start:b_end]

                # 1. Blocking
                cands_by_q = _BLOCKER.block_queries(sub_s1)

                # Pre-extract S1 fields
                s1_ids = sub_s1["entity_id"].values
                s1_names = sub_s1["norm_name"].fillna("").values
                s1_addrs = sub_s1["business_address"].fillna("").values
                s1_countries = sub_s1["country"].fillna("").values

                # Collect candidate pairs for the entire micro-batch
                query_spans = []
                b_s1_names = []
                b_cand_names = []
                b_s1_addrs = []
                b_cand_addrs = []
                b_s1_countries = []
                b_cand_countries = []

                query_out: Dict[str, Tuple[str, str]] = {}

                for q_idx, q_id in enumerate(s1_ids):
                    cand_ids = cands_by_q.get(q_id, [])

                    if not cand_ids:
                        query_out[q_id] = ("", "")
                        total_singletons += 1
                        continue

                    s1_n = s1_names[q_idx]
                    s1_a = s1_addrs[q_idx]
                    s1_c = s1_countries[q_idx]

                    start_idx_pair = len(b_cand_names)
                    for cid in cand_ids:
                        b_s1_names.append(s1_n)
                        b_cand_names.append(_CORPUS_NAMES.get(cid, ""))
                        b_s1_addrs.append(s1_a)
                        b_cand_addrs.append(_CORPUS_ADDRS.get(cid, ""))
                        b_s1_countries.append(s1_c)
                        b_cand_countries.append(_CORPUS_COUNTRIES.get(cid, ""))
                    end_idx_pair = len(b_cand_names)

                    query_spans.append((q_id, cand_ids, start_idx_pair, end_idx_pair))

                total_candidates += len(b_cand_names)

                if b_cand_names:
                    # 2. Vectorized Feature Extraction across all micro-batch pairs
                    # Name similarities
                    name_lev = np.array([Levenshtein.normalized_similarity(s1, c) for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.float32)
                    name_jw = np.array([JaroWinkler.similarity(s1, c) for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.float32)
                    name_tsort = np.array([fuzz.token_sort_ratio(s1, c) / 100.0 for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.float32)
                    name_tset = np.array([fuzz.token_set_ratio(s1, c) / 100.0 for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.float32)

                    # Address similarities
                    addr_lev = np.array([Levenshtein.normalized_similarity(s1, c) for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.float32)
                    addr_jw = np.array([JaroWinkler.similarity(s1, c) for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.float32)
                    addr_tsort = np.array([fuzz.token_sort_ratio(s1, c) / 100.0 for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.float32)
                    addr_tset = np.array([fuzz.token_set_ratio(s1, c) / 100.0 for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.float32)

                    # Length diff ratios & exact matches
                    name_len_diff = np.array([abs(len(s1) - len(c)) / max(len(s1), len(c), 1) for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.float32)
                    name_exact = np.array([1 if s1 == c and s1 != "" else 0 for s1, c in zip(b_s1_names, b_cand_names)], dtype=np.int8)

                    addr_len_diff = np.array([abs(len(s1) - len(c)) / max(len(s1), len(c), 1) for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.float32)
                    addr_exact = np.array([1 if s1 == c and s1 != "" else 0 for s1, c in zip(b_s1_addrs, b_cand_addrs)], dtype=np.int8)

                    country_match = np.array([1 if s1 == c and s1 != "" else 0 for s1, c in zip(b_s1_countries, b_cand_countries)], dtype=np.int8)

                    # TF-IDF cosine similarity on name
                    v_s1 = vec.transform(b_s1_names)
                    v_cands = vec.transform(b_cand_names)
                    tfidf_sim = np.asarray(v_s1.multiply(v_cands).sum(axis=1), dtype=np.float32).ravel()
                    np.clip(tfidf_sim, 0.0, 1.0, out=tfidf_sim)

                    # Street number & unit number features
                    street_unit_feats = compute_street_unit_pair_metrics(b_s1_addrs, b_cand_addrs)

                    # Assemble 17 features
                    X_batch = np.column_stack([
                        name_lev,
                        name_jw,
                        name_tsort,
                        name_tset,
                        addr_lev,
                        addr_jw,
                        addr_tsort,
                        addr_tset,
                        tfidf_sim,
                        country_match,
                        name_len_diff,
                        addr_len_diff,
                        name_exact,
                        addr_exact,
                        street_unit_feats["exact_street_number_match"],
                        street_unit_feats["street_number_numeric_distance"],
                        street_unit_feats["unit_number_match"],
                    ])

                    # 3. Model Scoring
                    probs = _MODEL.predict_proba(X_batch)[:, 1]

                    # Slice predictions per query
                    for q_id, cand_ids, p_start, p_end in query_spans:
                        q_probs = probs[p_start:p_end]
                        matched_indices = [i for i, p in enumerate(q_probs) if p >= _THRESHOLD]

                        if matched_indices:
                            matched_indices.sort(key=lambda idx: q_probs[idx], reverse=True)
                            matched_cands = list(dict.fromkeys(cand_ids[idx] for idx in matched_indices))
                            query_out[q_id] = (",".join(matched_cands), ",".join(dict.fromkeys(cand_ids)))
                            total_matches_found += 1
                        else:
                            query_out[q_id] = ("", ",".join(dict.fromkeys(cand_ids)))
                            total_singletons += 1

                # Write all queries in strict input order
                for q_id in s1_ids:
                    m_str, c_str = query_out[q_id]
                    f_match.write(f"{q_id}\t{m_str}\n")
                    f_cand.write(f"{q_id}\t{c_str}\n")

                done = b_end - start_idx
                if done % 10000 == 0 or done == n_queries:
                    elapsed = time.time() - t0
                    qps = done / elapsed if elapsed > 0 else 0
                    print(f"[Shard {shard_id:02d}] Progress: {done:,} / {n_queries:,} ({done/n_queries*100:.1f}%) "
                          f"[{qps:,.0f} queries/sec] Matches: {total_matches_found:,}, Singletons: {total_singletons:,}", flush=True)

        elapsed = time.time() - t0
        return {
            "shard_id": shard_id,
            "n_queries": n_queries,
            "total_matches_found": total_matches_found,
            "total_singletons": total_singletons,
            "total_candidates": total_candidates,
            "elapsed": elapsed,
            "match_part_path": str(match_part_path),
            "cand_part_path": str(cand_part_path),
        }
    except Exception as e:
        print(f"[Shard {shard_id:02d}] FATAL ERROR: {e}\n{traceback.format_exc()}", flush=True)
        raise e


def main():
    parser = argparse.ArgumentParser(description="Full-scale test inference across all 1.73M entities.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.879, help="Classification probability threshold (default: 0.879)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed Parquet files")
    parser.add_argument("--model-path", type=str, default="models/lgbm_matcher.pkl", help="Path to trained model artifact")
    parser.add_argument("--blocker-cache", type=str, default="data/processed/blocker_test_index.pkl", help="Path to test blocker index")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    print("=" * 80)
    print("FULL-SCALE TEST INFERENCE & LEADERBOARD PIPELINE (1.73M ENTITIES)")
    print("=" * 80)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model & Threshold
    print(f"\n1. Loading trained model from {args.model_path}...")
    t0 = time.time()
    with open(args.model_path, "rb") as f:
        model_artifact = pickle.load(f)
    model = model_artifact["model"]
    threshold = float(args.threshold or model_artifact.get("optimal_threshold", 0.90))
    print(f"   Model loaded in {time.time()-t0:.2f}s | Using threshold theta* = {threshold:.2f}")

    # 2. Fit TF-IDF Extractor on train_source1 (exact same representation as model training)
    print("\n2. Fitting TF-IDF vectorizer on train_source1 names...")
    t0 = time.time()
    train_s1 = pd.read_parquet(f"{args.data_dir}/train_source1.parquet", columns=["norm_name"])
    fit_samples = train_s1["norm_name"].dropna().sample(n=100000, random_state=42).tolist()
    extractor = FeatureExtractor().fit(fit_samples)
    print(f"   TF-IDF vectorizer fitted in {time.time()-t0:.2f}s ({len(extractor.vectorizer.vocabulary_):,} features)")
    del train_s1, fit_samples

    # 3. Load Blocker Index
    print(f"\n3. Loading cached test blocker index from {args.blocker_cache}...")
    t0 = time.time()
    with open(args.blocker_cache, "rb") as f:
        blocker = pickle.load(f)
    print(f"   Blocker index loaded in {time.time()-t0:.2f}s")

    # 4. Load Corpus Lookups (Source 2 + Source 3)
    print(f"\n4. Loading test corpus metadata lookups (Source 2 + Source 3)...")
    t0 = time.time()
    s2 = pd.read_parquet(f"{args.data_dir}/test_source2.parquet", columns=["entity_id", "norm_name", "business_address", "country"])
    s3 = pd.read_parquet(f"{args.data_dir}/test_source3.parquet", columns=["entity_id", "norm_name", "business_address", "country"])
    corpus = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    print("   Building in-memory string lookup dictionaries...")
    corpus_names = dict(zip(corpus["entity_id"], corpus["norm_name"].fillna("")))
    corpus_addrs = dict(zip(corpus["entity_id"], corpus["business_address"].fillna("")))
    corpus_countries = dict(zip(corpus["entity_id"], corpus["country"].fillna("")))
    del corpus
    print(f"   Lookups built for {len(corpus_names):,} records in {time.time()-t0:.2f}s")

    # 5. Load Test Source 1 Queries
    print(f"\n5. Loading full test Source 1 queries...")
    t0 = time.time()
    s1_cols = [
        "entity_id", "country", "business_name", "business_address",
        "norm_name", "norm_name_no_legal", "norm_address",
        "addr_postal_code", "addr_city", "addr_state",
        "addr_street_number", "addr_street_name",
    ]
    s1_df = pd.read_parquet(f"{args.data_dir}/test_source1.parquet", columns=s1_cols)
    total_queries = len(s1_df)
    print(f"   Loaded {total_queries:,} queries in {time.time()-t0:.2f}s")

    # 6. Assign Shards to Workers
    n_workers = args.workers
    shard_size = int(np.ceil(total_queries / n_workers))
    shards = []
    for w in range(n_workers):
        start = w * shard_size
        end = min((w + 1) * shard_size, total_queries)
        if start < total_queries:
            shards.append((w, start, end, str(parts_dir)))

    print(f"\n6. Partitioning {total_queries:,} queries across {len(shards)} workers (~{shard_size:,} queries/worker)...")

    # Set globals directly in worker function namespace for shared copy-on-write memory
    _worker_process_shard.__globals__["_BLOCKER"] = blocker
    _worker_process_shard.__globals__["_MODEL"] = model
    _worker_process_shard.__globals__["_EXTRACTOR"] = extractor
    _worker_process_shard.__globals__["_THRESHOLD"] = threshold
    _worker_process_shard.__globals__["_S1_DF"] = s1_df
    _worker_process_shard.__globals__["_CORPUS_NAMES"] = corpus_names
    _worker_process_shard.__globals__["_CORPUS_ADDRS"] = corpus_addrs
    _worker_process_shard.__globals__["_CORPUS_COUNTRIES"] = corpus_countries

    # Use fork start method to share memory pages with zero IPC transfer
    mp.set_start_method("fork", force=True)

    print(f"\n7. Launching {n_workers} worker processes...")
    t_start_workers = time.time()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_process_shard, shards)

    t_worker_elapsed = time.time() - t_start_workers
    print(f"\nAll {len(shards)} shards finished processing in {t_worker_elapsed:.1f}s ({total_queries/t_worker_elapsed:,.0f} queries/sec)!")

    # 7. Merge Shards into Final Output Files
    print("\n8. Merging shard outputs into final submission TSVs...")
    t0 = time.time()
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    total_matches = 0
    total_singletons = 0
    total_cands = 0

    with open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:
        
        # Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for r in sorted(results, key=lambda x: x["shard_id"]):
            total_matches += r["total_matches_found"]
            total_singletons += r["total_singletons"]
            total_cands += r["total_candidates"]

            # Stream matching part
            with open(r["match_part_path"], "r", encoding="utf-8") as f_p:
                for line in f_p:
                    f_match.write(line)

            # Stream candidate part
            with open(r["cand_part_path"], "r", encoding="utf-8") as f_p:
                for line in f_p:
                    f_cand.write(line)

    print(f"   Merged final files in {time.time()-t0:.2f}s:")
    print(f"   - {matching_file} ({matching_file.stat().st_size / 1e6:.1f} MB)")
    print(f"   - {candidate_file} ({candidate_file.stat().st_size / 1e6:.1f} MB)")

    # 8. Report Summary
    singleton_rate = (total_singletons / total_queries) * 100
    matched_rate = (total_matches / total_queries) * 100
    print("\n" + "=" * 80)
    print("FULL-SCALE SUBMISSION SUMMARY")
    print("=" * 80)
    print(f"Total Test Source 1 Entities:   {total_queries:,}")
    print(f"Entities with Matches Found:    {total_matches:,} ({matched_rate:.2f}%)")
    print(f"Singletons (Empty Predictions): {total_singletons:,} ({singleton_rate:.2f}%)")
    print(f"Total Candidate Pairs Evaluated:{total_cands:,}")
    print(f"Avg Candidates per Entity:      {total_cands/total_queries:.1f}")
    print(f"Total Pipeline Runtime:         {t_worker_elapsed:.1f}s ({t_worker_elapsed/60:.2f} mins)")
    print("=" * 80)

    # 9. Run Official Submission Validator
    print("\n9. Running official submission validator...")
    cmd = (
        f"python3 utils/validate_submission.py "
        f"--matching {matching_file} "
        f"--candidate {candidate_file} "
        f"--test-dir data/raw"
    )
    print(f"Command: {cmd}")
    res = os.system(cmd)
    if res == 0:
        print("\n>>> VALIDATION RESULT: PASSED ALL CHECKS! READY FOR LEADERBOARD SUBMISSION <<<")
    else:
        print(f"\n>>> VALIDATION FAILED with exit code {res} <<<")


if __name__ == "__main__":
    main()
