"""
Candidate Generation Script for Amazon ML Challenge 2026.

Uses MultiStrategyBlocker with parallel multiprocessing to generate candidate pairs:
  - For Train: train_source1 against train_source2 + train_source3
  - For Test: test_source1 against test_source2 + test_source3 (including France)

Outputs pairwise Parquet files:
  - candidate_pairs_train.parquet
  - candidate_pairs_test.parquet
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import json
import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking import BlockingConfig, MultiStrategyBlocker


CORPUS_COLUMNS = [
    "entity_id",
    "country",
    "business_name",
    "business_address",
    "norm_name",
    "norm_name_no_legal",
    "addr_postal_code",
    "addr_city",
    "addr_state",
    "addr_street_number",
    "addr_street_name",
]


def load_corpus(
    source2_path: str,
    source3_path: str,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """Load and concatenate Source 2 and Source 3 corpus."""
    print(f"Loading Source 2 from {source2_path}...")
    t0 = time.time()
    s2 = pd.read_parquet(source2_path, columns=CORPUS_COLUMNS)
    print(f"  Loaded {len(s2):,} records in {time.time()-t0:.1f}s")

    print(f"Loading Source 3 from {source3_path}...")
    t0 = time.time()
    s3 = pd.read_parquet(source3_path, columns=CORPUS_COLUMNS)
    print(f"  Loaded {len(s3):,} records in {time.time()-t0:.1f}s")

    corpus = pd.concat([s2, s3], ignore_index=True)
    if max_records and max_records < len(corpus):
        corpus = corpus.iloc[:max_records]
    print(f"Combined corpus total: {len(corpus):,} records")
    return corpus


def load_queries(
    source1_path: str,
    split_path: Optional[str] = None,
    split_name: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load Source 1 queries with optional split filtering and sampling."""
    print(f"Loading Source 1 queries from {source1_path}...")
    t0 = time.time()
    s1 = pd.read_parquet(source1_path, columns=CORPUS_COLUMNS)
    print(f"  Loaded {len(s1):,} records in {time.time()-t0:.1f}s")

    if split_path and split_name and split_name in ("train", "val"):
        with open(split_path, "r", encoding="utf-8") as f:
            split_meta = json.load(f)
        target_ids = set(split_meta[f"{split_name}_entity_ids"])
        s1 = s1[s1["entity_id"].isin(target_ids)].reset_index(drop=True)
        print(f"  Filtered to {len(s1):,} queries for split '{split_name}'")

    if sample_size and sample_size < len(s1):
        # Stratify by country
        print(f"  Sampling {sample_size:,} queries stratified by country...")
        countries = s1["country"].unique()
        samples = []
        for c in countries:
            sub = s1[s1["country"] == c]
            n_c = int(round(sample_size * (len(sub) / len(s1))))
            n_c = max(1, min(n_c, len(sub)))
            samples.append(sub.sample(n=n_c, random_state=random_seed))
        s1 = pd.concat(samples, ignore_index=True)
        if len(s1) > sample_size:
            s1 = s1.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        print(f"  Sampled query set: {len(s1):,} entities: {s1['country'].value_counts().to_dict()}")

    return s1


_GLOBAL_BLOCKER: Optional[MultiStrategyBlocker] = None

def _init_worker(blocker: MultiStrategyBlocker):
    global _GLOBAL_BLOCKER
    _GLOBAL_BLOCKER = blocker


def _worker_block_chunk(chunk_df: pd.DataFrame) -> Dict[str, List[str]]:
    global _GLOBAL_BLOCKER
    assert _GLOBAL_BLOCKER is not None
    return _GLOBAL_BLOCKER.block_queries(chunk_df)


def parallel_block_queries(
    blocker: MultiStrategyBlocker,
    query_df: pd.DataFrame,
    n_workers: int = 6,
    batch_size: int = 5000,
) -> Dict[str, List[str]]:
    """Block queries in parallel batches using ProcessPoolExecutor."""
    total_queries = len(query_df)
    chunks = [query_df.iloc[i : i + batch_size] for i in range(0, total_queries, batch_size)]
    print(f"Blocking {total_queries:,} queries across {len(chunks)} batches using {n_workers} workers...")

    t0 = time.time()
    all_candidates: Dict[str, List[str]] = {}

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(blocker,),
    ) as executor:
        for batch_res in executor.map(_worker_block_chunk, chunks):
            all_candidates.update(batch_res)
            done = len(all_candidates)
            elapsed = time.time() - t0
            qps = done / elapsed if elapsed > 0 else 0
            if done % (batch_size * 2) == 0 or done == total_queries:
                print(f"  Progress: {done:,} / {total_queries:,} ({done/total_queries*100:.1f}%) [{qps:,.0f} queries/sec]")

    elapsed = time.time() - t0
    print(f"Finished candidate blocking in {elapsed:.1f}s ({total_queries/elapsed:,.0f} queries/sec)")
    return all_candidates


def candidates_to_long_parquet(
    candidates_dict: Dict[str, List[str]],
    output_parquet: str,
    chunk_size: int = 500000,
) -> int:
    """Save candidates dictionary as long DataFrame parquet with schema (source1_entity_id, candidate_entity_id)."""
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("candidate_entity_id", pa.string()),
    ])

    writer = pq.ParquetWriter(out_path, schema, compression="SNAPPY")
    total_pairs = 0

    cur_s1 = []
    cur_cand = []

    for s1_id, c_list in candidates_dict.items():
        for c_id in c_list:
            cur_s1.append(s1_id)
            cur_cand.append(c_id)
            if len(cur_s1) >= chunk_size:
                table = pa.Table.from_arrays([
                    pa.array(cur_s1, type=pa.string()),
                    pa.array(cur_cand, type=pa.string()),
                ], schema=schema)
                writer.write_table(table)
                total_pairs += len(cur_s1)
                cur_s1.clear()
                cur_cand.clear()

    if cur_s1:
        table = pa.Table.from_arrays([
            pa.array(cur_s1, type=pa.string()),
            pa.array(cur_cand, type=pa.string()),
        ], schema=schema)
        writer.write_table(table)
        total_pairs += len(cur_s1)

    writer.close()
    print(f"Saved {total_pairs:,} candidate pairs to {out_path}")
    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate candidate pairs using MultiStrategyBlocker.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"], help="Split for train mode")
    parser.add_argument("--sample", type=int, default=0, help="Number of queries to sample (0 for full)")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per query entity")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker processes")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--out-parquet", type=str, default=None, help="Output parquet path")
    parser.add_argument("--index-cache", type=str, default=None, help="Path to cache/load fitted blocker pickle")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    proc_dir = data_dir / "processed"

    if args.mode == "train":
        s1_path = str(proc_dir / "train_source1.parquet")
        s2_path = str(proc_dir / "train_source2.parquet")
        s3_path = str(proc_dir / "train_source3.parquet")
        split_path = str(proc_dir / "train_val_split.json")
        default_out = str(proc_dir / "candidate_pairs_train.parquet")
        split_arg = args.split if args.split != "all" else None
        default_cache = str(proc_dir / "blocker_train_index.pkl")
    else:
        s1_path = str(proc_dir / "test_source1.parquet")
        s2_path = str(proc_dir / "test_source2.parquet")
        s3_path = str(proc_dir / "test_source3.parquet")
        split_path = None
        default_out = str(proc_dir / "candidate_pairs_test.parquet")
        split_arg = None
        default_cache = str(proc_dir / "blocker_test_index.pkl")

    out_parquet = args.out_parquet or default_out
    sample_size = None if args.sample == 0 else args.sample
    cache_path = args.index_cache or default_cache

    print("=" * 70, flush=True)
    print(f"CANDIDATE GENERATION: MODE = {args.mode.upper()}, MAX_CANDS = {args.max_candidates}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load or Fit Blocker
    import pickle
    blocker = None
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached blocker index from {cache_path}...", flush=True)
        t0 = time.time()
        with open(cache_path, "rb") as f:
            blocker = pickle.load(f)
        # Update config max candidates if changed
        blocker.config.max_candidates_per_query = args.max_candidates
        print(f"Loaded cached blocker index in {time.time()-t0:.1f}s", flush=True)
    else:
        corpus = load_corpus(s2_path, s3_path)
        cfg = BlockingConfig(
            enable_tfidf=False,
            enable_token_overlap=True,
            enable_phonetic=True,
            enable_address=True,
            max_block_size=1200,
            max_candidates_per_query=args.max_candidates,
        )
        t0 = time.time()
        blocker = MultiStrategyBlocker(cfg).fit(corpus)
        print(f"Blocker fitted in {time.time()-t0:.1f}s", flush=True)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving fitted blocker index to {cache_path}...", flush=True)
            t0 = time.time()
            with open(cache_path, "wb") as f:
                pickle.dump(blocker, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved blocker index in {time.time()-t0:.1f}s", flush=True)

    # 2. Load Queries
    queries = load_queries(
        source1_path=s1_path,
        split_path=split_path,
        split_name=split_arg,
        sample_size=sample_size,
    )

    # 3. Block Queries in Parallel
    candidates = parallel_block_queries(
        blocker=blocker,
        query_df=queries,
        n_workers=args.workers,
        batch_size=5000,
    )

    # 4. Save long Parquet
    total_pairs = candidates_to_long_parquet(candidates, out_parquet)

    # Also link or copy to project root if out_parquet is in data/processed/
    root_out = Path(out_parquet).name
    if not Path(root_out).exists() and Path(out_parquet).exists():
        try:
            os.symlink(out_parquet, root_out)
            print(f"Created symlink at project root: {root_out} -> {out_parquet}", flush=True)
        except Exception:
            pass

    print(f"Candidate generation completed successfully! Total pairs: {total_pairs:,}", flush=True)


if __name__ == "__main__":
    main()
