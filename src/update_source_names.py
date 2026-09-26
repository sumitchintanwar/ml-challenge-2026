"""
Fast Upstream Parquet Normalization Updater.
Amazon ML Challenge 2026.

Re-normalizes business_name in data/processed/*.parquet using multiprocessing
with the updated normalize_name() (domain splitting, credential extraction,
and OCR homoglyph corrections) while preserving all existing address normalization fields.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_name


UPDATED_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("business_name", pa.string()),
    ("business_address", pa.string()),
    ("norm_name", pa.string()),
    ("norm_name_no_legal", pa.string()),
    ("legal_type", pa.string()),
    ("credential", pa.string()),
    ("norm_address", pa.string()),
    ("addr_street_number", pa.string()),
    ("addr_street_name", pa.string()),
    ("addr_unit", pa.string()),
    ("addr_landmark", pa.string()),
    ("addr_city", pa.string()),
    ("addr_state", pa.string()),
    ("addr_postal_code", pa.string()),
])


def _worker_norm_names(batch: Tuple[List[str], List[str]]) -> Dict[str, List[Any]]:
    names, countries = batch
    norm_names = []
    norm_names_no_legal = []
    legal_types = []
    credentials = []

    for name, country in zip(names, countries):
        res = normalize_name(name, country=country)
        norm_names.append(res.norm_name)
        norm_names_no_legal.append(res.norm_name_no_legal)
        legal_types.append(res.legal_type)
        credentials.append(res.credential)

    return {
        "norm_name": norm_names,
        "norm_name_no_legal": norm_names_no_legal,
        "legal_type": legal_types,
        "credential": credentials,
    }


def update_single_parquet(
    parquet_path: Path,
    executor: ProcessPoolExecutor,
    chunk_size: int = 100_000,
    sub_batch_size: int = 12_500,
):
    print(f"\n=======================================================")
    print(f"Updating: {parquet_path.name}")
    print(f"=======================================================")
    t0 = time.time()

    temp_path = parquet_path.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(temp_path, schema=UPDATED_SCHEMA, compression="snappy")

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"  Total rows: {total_rows:,}")

    rows_processed = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        n = len(df_chunk)

        # Worker sub-batches
        sub_batches = []
        for i in range(0, n, sub_batch_size):
            sub = df_chunk.iloc[i : i + sub_batch_size]
            sub_batches.append((
                sub["business_name"].fillna("").astype(str).tolist(),
                sub["country"].fillna("").astype(str).tolist(),
            ))

        results = list(executor.map(_worker_norm_names, sub_batches))

        # Combine worker results
        up_norm_names = []
        up_norm_no_legal = []
        up_legal_types = []
        up_credentials = []

        for r in results:
            up_norm_names.extend(r["norm_name"])
            up_norm_no_legal.extend(r["norm_name_no_legal"])
            up_legal_types.extend(r["legal_type"])
            up_credentials.extend(r["credential"])

        # Construct final dict with UPDATED_SCHEMA
        chunk_dict = {
            "entity_id": df_chunk["entity_id"].tolist(),
            "country": df_chunk["country"].tolist(),
            "business_name": df_chunk["business_name"].tolist(),
            "business_address": df_chunk["business_address"].tolist(),
            "norm_name": up_norm_names,
            "norm_name_no_legal": up_norm_no_legal,
            "legal_type": up_legal_types,
            "credential": up_credentials,
            "norm_address": df_chunk["norm_address"].tolist(),
            "addr_street_number": df_chunk["addr_street_number"].tolist(),
            "addr_street_name": df_chunk["addr_street_name"].tolist(),
            "addr_unit": df_chunk["addr_unit"].tolist(),
            "addr_landmark": df_chunk["addr_landmark"].tolist(),
            "addr_city": df_chunk["addr_city"].tolist(),
            "addr_state": df_chunk["addr_state"].tolist(),
            "addr_postal_code": df_chunk["addr_postal_code"].tolist(),
        }

        table = pa.Table.from_pydict(chunk_dict, schema=UPDATED_SCHEMA)
        writer.write_table(table)

        rows_processed += n
        elapsed = time.time() - t0
        rate = rows_processed / elapsed if elapsed > 0 else 0
        print(f"  Processed {rows_processed:,} / {total_rows:,} rows ({rows_processed/total_rows*100:.1f}%) [{rate:.0f} rows/s]")

    writer.close()
    temp_path.replace(parquet_path)
    print(f"  Completed {parquet_path.name} in {time.time()-t0:.1f}s")


def main():
    processed_dir = Path("data/processed")
    parquet_files = [
        processed_dir / "train_source1.parquet",
        processed_dir / "train_source2.parquet",
        processed_dir / "train_source3.parquet",
        processed_dir / "test_source1.parquet",
        processed_dir / "test_source2.parquet",
        processed_dir / "test_source3.parquet",
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        for p in parquet_files:
            if not p.exists():
                print(f"Warning: {p} does not exist, skipping.")
                continue
            update_single_parquet(p, executor)

    print("\nAll 6 parquet files successfully updated with latest normalization!")


if __name__ == "__main__":
    main()
