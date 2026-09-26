"""
High-Speed Feature Extraction Module for Entity Resolution.
Amazon ML Challenge 2026.

Computes a compact, discriminative ~12-15 feature set for candidate entity pairs:
  1. Levenshtein ratio on normalized business_name (rapidfuzz)
  2. Jaro-Winkler similarity on normalized business_name (rapidfuzz)
  3. Token sort ratio on normalized business_name (rapidfuzz)
  4. Token set ratio on normalized business_name (rapidfuzz)
  5. Levenshtein ratio on normalized business_address (rapidfuzz)
  6. Jaro-Winkler similarity on normalized business_address (rapidfuzz)
  7. Token sort ratio on normalized business_address (rapidfuzz)
  8. Token set ratio on normalized business_address (rapidfuzz)
  9. TF-IDF character 3-gram cosine similarity on business_name
 10. Country exact-match flag (int8 1/0)
 11. Name length-difference ratio (|L1 - L2| / max(L1, L2, 1))
 12. Address length-difference ratio (|L1 - L2| / max(L1, L2, 1))
 13. Exact name match flag (int8 1/0)
 14. Exact address match flag (int8 1/0)

Optimized for high-throughput batch evaluation (>200,000 pairs/sec) and chunked Parquet I/O.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
import argparse
import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from src.address_normalization import (
    compare_street_and_unit_numbers,
    parse_street_and_unit_numbers,
)


# ---------------------------------------------------------------------------
# Feature Column Names & Schema Definition
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "feat_name_levenshtein",
    "feat_name_jaro_winkler",
    "feat_name_token_sort",
    "feat_name_token_set",
    "feat_addr_levenshtein",
    "feat_addr_jaro_winkler",
    "feat_addr_token_sort",
    "feat_addr_token_set",
    "feat_name_tfidf_cosine",
    "feat_country_match",
    "feat_name_len_diff_ratio",
    "feat_addr_len_diff_ratio",
    "feat_name_exact_match",
    "feat_addr_exact_match",
    "exact_street_number_match",
    "street_number_numeric_distance",
    "unit_number_match",
]

OUTPUT_SCHEMA_TRAIN = pa.schema([
    ("source1_entity_id", pa.string()),
    ("candidate_entity_id", pa.string()),
    ("feat_name_levenshtein", pa.float32()),
    ("feat_name_jaro_winkler", pa.float32()),
    ("feat_name_token_sort", pa.float32()),
    ("feat_name_token_set", pa.float32()),
    ("feat_addr_levenshtein", pa.float32()),
    ("feat_addr_jaro_winkler", pa.float32()),
    ("feat_addr_token_sort", pa.float32()),
    ("feat_addr_token_set", pa.float32()),
    ("feat_name_tfidf_cosine", pa.float32()),
    ("feat_country_match", pa.int8()),
    ("feat_name_len_diff_ratio", pa.float32()),
    ("feat_addr_len_diff_ratio", pa.float32()),
    ("feat_name_exact_match", pa.int8()),
    ("feat_addr_exact_match", pa.int8()),
    ("exact_street_number_match", pa.int8()),
    ("street_number_numeric_distance", pa.float32()),
    ("unit_number_match", pa.float32()),
    ("label", pa.int8()),
])

OUTPUT_SCHEMA_TEST = pa.schema([
    ("source1_entity_id", pa.string()),
    ("candidate_entity_id", pa.string()),
    ("feat_name_levenshtein", pa.float32()),
    ("feat_name_jaro_winkler", pa.float32()),
    ("feat_name_token_sort", pa.float32()),
    ("feat_name_token_set", pa.float32()),
    ("feat_addr_levenshtein", pa.float32()),
    ("feat_addr_jaro_winkler", pa.float32()),
    ("feat_addr_token_sort", pa.float32()),
    ("feat_addr_token_set", pa.float32()),
    ("feat_name_tfidf_cosine", pa.float32()),
    ("feat_country_match", pa.int8()),
    ("feat_name_len_diff_ratio", pa.float32()),
    ("feat_addr_len_diff_ratio", pa.float32()),
    ("feat_name_exact_match", pa.int8()),
    ("feat_addr_exact_match", pa.int8()),
    ("exact_street_number_match", pa.int8()),
    ("street_number_numeric_distance", pa.float32()),
    ("unit_number_match", pa.float32()),
])


# ---------------------------------------------------------------------------
# High-Speed Vectorized Metric Functions
# ---------------------------------------------------------------------------

def compute_string_pair_metrics(
    str1_list: List[str],
    str2_list: List[str],
    prefix: str,
) -> Dict[str, np.ndarray]:
    """
    Compute 4 rapidfuzz similarity metrics + length diff ratio + exact match
    for paired lists of strings.
    """
    n = len(str1_list)
    if n == 0:
        return {
            f"feat_{prefix}_levenshtein": np.empty(0, dtype=np.float32),
            f"feat_{prefix}_jaro_winkler": np.empty(0, dtype=np.float32),
            f"feat_{prefix}_token_sort": np.empty(0, dtype=np.float32),
            f"feat_{prefix}_token_set": np.empty(0, dtype=np.float32),
            f"feat_{prefix}_len_diff_ratio": np.empty(0, dtype=np.float32),
            f"feat_{prefix}_exact_match": np.empty(0, dtype=np.int8),
        }

    lev = np.empty(n, dtype=np.float32)
    jw = np.empty(n, dtype=np.float32)
    tsort = np.empty(n, dtype=np.float32)
    tset = np.empty(n, dtype=np.float32)
    len_diff = np.empty(n, dtype=np.float32)
    exact = np.empty(n, dtype=np.int8)

    for i in range(n):
        s1 = str1_list[i]
        s2 = str2_list[i]

        # Exact match
        if s1 == s2:
            lev[i] = 1.0
            jw[i] = 1.0
            tsort[i] = 1.0
            tset[i] = 1.0
            len_diff[i] = 0.0
            exact[i] = 1
            continue

        exact[i] = 0

        # Rapidfuzz similarities
        lev[i] = Levenshtein.normalized_similarity(s1, s2)
        jw[i] = JaroWinkler.similarity(s1, s2)
        tsort[i] = fuzz.token_sort_ratio(s1, s2) / 100.0
        tset[i] = fuzz.token_set_ratio(s1, s2) / 100.0

        # Length difference ratio: |L1 - L2| / max(L1, L2, 1)
        l1 = len(s1)
        l2 = len(s2)
        max_l = max(l1, l2, 1)
        len_diff[i] = abs(l1 - l2) / max_l

    return {
        f"feat_{prefix}_levenshtein": lev,
        f"feat_{prefix}_jaro_winkler": jw,
        f"feat_{prefix}_token_sort": tsort,
        f"feat_{prefix}_token_set": tset,
        f"feat_{prefix}_len_diff_ratio": len_diff,
        f"feat_{prefix}_exact_match": exact,
    }


def compute_tfidf_cosine_similarity(
    s1_names: List[str],
    cand_names: List[str],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """
    Compute character 3-gram TF-IDF cosine similarity for paired string lists.
    Uses sparse row-wise dot product of L2-normalized TF-IDF vectors.
    """
    n = len(s1_names)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    # Transform paired batches
    v1 = vectorizer.transform(s1_names)
    v2 = vectorizer.transform(cand_names)

    # Row-wise dot product of normalized vectors equals cosine similarity
    dot = v1.multiply(v2).sum(axis=1)
    cos_sim = np.asarray(dot, dtype=np.float32).ravel()
    # Clip numerical precision artifacts to [0.0, 1.0]
    np.clip(cos_sim, 0.0, 1.0, out=cos_sim)
    return cos_sim


def compute_street_unit_pair_metrics(
    addr1_list: List[str],
    addr2_list: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute street number and unit number match features for paired addresses:
      - exact_street_number_match: 1 if both parsed and match, 0 otherwise (pa.int8)
      - street_number_numeric_distance: absolute difference if both are numbers, else -1.0 (pa.float32)
      - unit_number_match: 1.0 if both match, 0.0 if mismatch, -1.0 if not applicable (neither has unit)
    """
    n = len(addr1_list)
    exact_match = np.empty(n, dtype=np.int8)
    num_dist = np.empty(n, dtype=np.float32)
    unit_match = np.empty(n, dtype=np.float32)

    cache: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}

    for i in range(n):
        a1 = addr1_list[i]
        a2 = addr2_list[i]

        p1 = cache.get(a1)
        if p1 is None:
            p1 = parse_street_and_unit_numbers(a1)
            cache[a1] = p1

        p2 = cache.get(a2)
        if p2 is None:
            p2 = parse_street_and_unit_numbers(a2)
            cache[a2] = p2

        s1_str, s1_num, u1 = p1
        s2_str, s2_num, u2 = p2

        # 1. exact_street_number_match
        exact_match[i] = 1 if (s1_str is not None and s2_str is not None and s1_str == s2_str) else 0

        # 2. street_number_numeric_distance
        if s1_num is not None and s2_num is not None:
            num_dist[i] = float(abs(s1_num - s2_num))
        else:
            num_dist[i] = -1.0

        # 3. unit_number_match
        if u1 is not None and u2 is not None:
            unit_match[i] = 1.0 if u1 == u2 else 0.0
        elif u1 is None and u2 is None:
            unit_match[i] = -1.0  # Not applicable
        else:
            unit_match[i] = 0.0

    return {
        "exact_street_number_match": exact_match,
        "street_number_numeric_distance": num_dist,
        "unit_number_match": unit_match,
    }


# ---------------------------------------------------------------------------
# Feature Extractor Class
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    High-performance feature extraction pipeline for entity resolution pairs.
    """

    def __init__(
        self,
        tfidf_max_features: int = 40000,
        tfidf_ngram_range: Tuple[int, int] = (3, 3),
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tfidf_ngram_range,
            max_features=tfidf_max_features,
            norm="l2",
            dtype=np.float32,
        )
        self.is_fitted = False

    def fit(self, text_samples: Union[List[str], pd.Series]) -> "FeatureExtractor":
        """
        Fit character n-gram TF-IDF vectorizer on business names.
        """
        if isinstance(text_samples, pd.Series):
            text_samples = text_samples.dropna().tolist()
        self.vectorizer.fit(text_samples)
        self.is_fitted = True
        return self

    def extract_features(
        self,
        s1_names: List[str],
        cand_names: List[str],
        s1_addrs: List[str],
        cand_addrs: List[str],
        s1_countries: List[str],
        cand_countries: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Extract the full feature dictionary from aligned string lists.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted with .fit() before extracting features.")

        n = len(s1_names)

        # 1. Name metrics
        name_feats = compute_string_pair_metrics(s1_names, cand_names, prefix="name")

        # 2. Address metrics
        addr_feats = compute_string_pair_metrics(s1_addrs, cand_addrs, prefix="addr")

        # 3. TF-IDF cosine similarity on name
        tfidf_sim = compute_tfidf_cosine_similarity(s1_names, cand_names, self.vectorizer)

        # 4. Country exact match
        country_match = np.empty(n, dtype=np.int8)
        for i in range(n):
            country_match[i] = 1 if s1_countries[i] == cand_countries[i] else 0

        # 5. Street number and unit number metrics
        street_unit_feats = compute_street_unit_pair_metrics(s1_addrs, cand_addrs)

        # Combine all features
        features = {
            "feat_name_levenshtein": name_feats["feat_name_levenshtein"],
            "feat_name_jaro_winkler": name_feats["feat_name_jaro_winkler"],
            "feat_name_token_sort": name_feats["feat_name_token_sort"],
            "feat_name_token_set": name_feats["feat_name_token_set"],
            "feat_addr_levenshtein": addr_feats["feat_addr_levenshtein"],
            "feat_addr_jaro_winkler": addr_feats["feat_addr_jaro_winkler"],
            "feat_addr_token_sort": addr_feats["feat_addr_token_sort"],
            "feat_addr_token_set": addr_feats["feat_addr_token_set"],
            "feat_name_tfidf_cosine": tfidf_sim,
            "feat_country_match": country_match,
            "feat_name_len_diff_ratio": name_feats["feat_name_len_diff_ratio"],
            "feat_addr_len_diff_ratio": addr_feats["feat_addr_len_diff_ratio"],
            "feat_name_exact_match": name_feats["feat_name_exact_match"],
            "feat_addr_exact_match": addr_feats["feat_addr_exact_match"],
            "exact_street_number_match": street_unit_feats["exact_street_number_match"],
            "street_number_numeric_distance": street_unit_feats["street_number_numeric_distance"],
            "unit_number_match": street_unit_feats["unit_number_match"],
        }
        return features

    def extract_features_df(
        self,
        pairs_df: pd.DataFrame,
        s1_lookup: Dict[str, Tuple[str, str, str]],
        corpus_lookup: Dict[str, Tuple[str, str, str]],
        ground_truth_dict: Optional[Dict[str, Set[str]]] = None,
    ) -> pd.DataFrame:
        """
        Extract features for pairs DataFrame with optional ground-truth labeling.

        Args:
            pairs_df: DataFrame with 'source1_entity_id' and 'candidate_entity_id'
            s1_lookup: Dict mapping s1_id -> (norm_name, norm_addr, country)
            corpus_lookup: Dict mapping cand_id -> (norm_name, norm_addr, country)
            ground_truth_dict: Optional mapping s1_id -> set of matched candidate IDs

        Returns:
            DataFrame with IDs, features, and optional 'label' column.
        """
        s1_ids = pairs_df["source1_entity_id"].tolist()
        cand_ids = pairs_df["candidate_entity_id"].tolist()
        n = len(s1_ids)

        s1_names: List[str] = []
        s1_addrs: List[str] = []
        s1_countries: List[str] = []
        cand_names: List[str] = []
        cand_addrs: List[str] = []
        cand_countries: List[str] = []

        empty_record = ("", "", "")

        for s1_id, c_id in zip(s1_ids, cand_ids):
            s1_rec = s1_lookup.get(s1_id, empty_record)
            c_rec = corpus_lookup.get(c_id, empty_record)

            s1_names.append(s1_rec[0])
            s1_addrs.append(s1_rec[1])
            s1_countries.append(s1_rec[2])

            cand_names.append(c_rec[0])
            cand_addrs.append(c_rec[1])
            cand_countries.append(c_rec[2])

        feats = self.extract_features(
            s1_names=s1_names,
            cand_names=cand_names,
            s1_addrs=s1_addrs,
            cand_addrs=cand_addrs,
            s1_countries=s1_countries,
            cand_countries=cand_countries,
        )

        result_dict = {
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
        }
        result_dict.update(feats)

        if ground_truth_dict is not None:
            labels = np.empty(n, dtype=np.int8)
            for i in range(n):
                q = s1_ids[i]
                c = cand_ids[i]
                labels[i] = 1 if (q in ground_truth_dict and c in ground_truth_dict[q]) else 0
            result_dict["label"] = labels

        return pd.DataFrame(result_dict)


# ---------------------------------------------------------------------------
# Data Loading Helpers & In-Memory Lookups
# ---------------------------------------------------------------------------

def load_ground_truth_dict(gt_tsv_path: Union[str, Path]) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping: source1_entity_id -> set(matched_entity_ids).
    """
    gt_df = pd.read_csv(gt_tsv_path, sep="\t")
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
        if m_str:
            gt_dict[s1] = set(x.strip() for x in m_str.split(",") if x.strip())
        else:
            gt_dict[s1] = set()
    return gt_dict


def build_entity_lookup(
    df: pd.DataFrame,
    name_col: str = "norm_name",
    addr_col: str = "business_address",
    country_col: str = "country",
) -> Dict[str, Tuple[str, str, str]]:
    """
    Build fast memory dictionary: entity_id -> (norm_name, norm_addr, country).
    """
    lookup: Dict[str, Tuple[str, str, str]] = {}
    e_ids = df["entity_id"].values
    names = df[name_col].fillna("").astype(str).values
    addrs = df[addr_col].fillna("").astype(str).values
    countries = df[country_col].fillna("").astype(str).values

    for i in range(len(e_ids)):
        lookup[e_ids[i]] = (names[i], addrs[i], countries[i])
    return lookup


# ---------------------------------------------------------------------------
# Streamed Chunk-Based Parquet Pipeline
# ---------------------------------------------------------------------------

def compute_and_save_feature_matrix(
    candidate_pairs_path: Union[str, Path],
    output_path: Union[str, Path],
    s1_lookup: Dict[str, Tuple[str, str, str]],
    corpus_lookup: Dict[str, Tuple[str, str, str]],
    feature_extractor: FeatureExtractor,
    ground_truth_dict: Optional[Dict[str, Set[str]]] = None,
    chunk_size: int = 250000,
) -> Dict[str, Any]:
    """
    Process candidate pairs in chunks and write to Parquet table incrementally.
    Avoids high memory peaks and logs progress.
    """
    candidate_pairs_path = Path(candidate_pairs_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading candidate pairs from {candidate_pairs_path}...")
    parquet_file = pq.ParquetFile(candidate_pairs_path)
    total_pairs = parquet_file.metadata.num_rows
    print(f"  Total candidate pairs to process: {total_pairs:,}")

    schema = OUTPUT_SCHEMA_TRAIN if ground_truth_dict is not None else OUTPUT_SCHEMA_TEST
    writer = pq.ParquetWriter(output_path, schema, compression="SNAPPY")

    total_processed = 0
    total_positives = 0
    t0 = time.time()

    # Iterate over record batches or row groups
    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=["source1_entity_id", "candidate_entity_id"]):
        chunk_df = batch.to_pandas()
        feat_df = feature_extractor.extract_features_df(
            pairs_df=chunk_df,
            s1_lookup=s1_lookup,
            corpus_lookup=corpus_lookup,
            ground_truth_dict=ground_truth_dict,
        )

        if "label" in feat_df.columns:
            total_positives += int(feat_df["label"].sum())

        table = pa.Table.from_pandas(feat_df, schema=schema, preserve_index=False)
        writer.write_table(table)

        total_processed += len(feat_df)
        elapsed = time.time() - t0
        speed = total_processed / elapsed if elapsed > 0 else 0
        pos_info = f", Positives: {total_positives:,}" if ground_truth_dict is not None else ""
        print(f"  Processed {total_processed:,} / {total_pairs:,} pairs ({total_processed/total_pairs*100:.1f}%) "
              f"[{speed:,.0f} pairs/sec]{pos_info}")

    writer.close()
    elapsed = time.time() - t0

    stats = {
        "output_path": str(output_path),
        "total_pairs": total_processed,
        "elapsed_sec": round(elapsed, 2),
        "pairs_per_sec": round(total_processed / elapsed if elapsed > 0 else 0, 1),
    }

    if ground_truth_dict is not None:
        negatives = total_processed - total_positives
        pos_ratio = (total_positives / total_processed * 100) if total_processed > 0 else 0.0
        neg_pos_ratio = (negatives / total_positives) if total_positives > 0 else 0.0
        stats.update({
            "positives": total_positives,
            "negatives": negatives,
            "positive_pct": round(pos_ratio, 2),
            "class_imbalance_ratio": f"1:{neg_pos_ratio:.2f}",
        })

    return stats


# ---------------------------------------------------------------------------
# Command-Line Interface (CLI)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract features for ER candidate pairs.")
    parser.add_argument("--candidate-pairs", type=str, required=True, help="Input candidate pairs parquet")
    parser.add_argument("--output", type=str, required=True, help="Output feature matrix parquet")
    parser.add_argument("--s1-data", type=str, required=True, help="Processed Source 1 parquet")
    parser.add_argument("--s2-data", type=str, required=True, help="Processed Source 2 parquet")
    parser.add_argument("--s3-data", type=str, required=True, help="Processed Source 3 parquet")
    parser.add_argument("--ground-truth", type=str, default=None, help="Optional train_ground_truth.tsv for labeling")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Chunk size for batch processing")
    parser.add_argument("--sample-fit-size", type=int, default=100000, help="Number of business names to fit TF-IDF")
    args = parser.parse_args()

    print("=" * 70)
    print(f"FEATURE EXTRACTION: {args.candidate_pairs} -> {args.output}")
    print("=" * 70)

    # 1. Load source data for lookups
    print("Loading entity records for lookups...")
    cols = ["entity_id", "country", "business_name", "business_address", "norm_name"]
    s1_df = pd.read_parquet(args.s1_data, columns=cols)
    s2_df = pd.read_parquet(args.s2_data, columns=cols)
    s3_df = pd.read_parquet(args.s3_data, columns=cols)

    corpus_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"  Source 1 records: {len(s1_df):,}")
    print(f"  Corpus records:   {len(corpus_df):,}")

    print("Building in-memory entity lookups...")
    t0 = time.time()
    s1_lookup = build_entity_lookup(s1_df)
    corpus_lookup = build_entity_lookup(corpus_df)
    print(f"  Lookups built in {time.time()-t0:.2f}s")

    # 2. Fit TF-IDF Vectorizer
    print(f"Fitting TF-IDF vectorizer on sample of {args.sample_fit_size:,} names...")
    t0 = time.time()
    fit_samples = s1_df["norm_name"].dropna().sample(
        n=min(args.sample_fit_size, len(s1_df)), random_state=42
    ).tolist()
    extractor = FeatureExtractor().fit(fit_samples)
    print(f"  TF-IDF vectorizer fitted in {time.time()-t0:.2f}s ({len(extractor.vectorizer.vocabulary_):,} features)")

    # 3. Load ground truth if provided
    gt_dict = None
    if args.ground_truth:
        print(f"Loading ground truth labels from {args.ground_truth}...")
        gt_dict = load_ground_truth_dict(args.ground_truth)
        print(f"  Loaded ground truth for {len(gt_dict):,} Source 1 entities")

    # 4. Streamed Feature Computation
    stats = compute_and_save_feature_matrix(
        candidate_pairs_path=args.candidate_pairs,
        output_path=args.output,
        s1_lookup=s1_lookup,
        corpus_lookup=corpus_lookup,
        feature_extractor=extractor,
        ground_truth_dict=gt_dict,
        chunk_size=args.chunk_size,
    )

    # Also link or copy to project root if output is in data/processed/
    root_out = Path(args.output).name
    if not Path(root_out).exists() and Path(args.output).exists():
        try:
            os.symlink(args.output, root_out)
            print(f"Created symlink at project root: {root_out} -> {args.output}")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

