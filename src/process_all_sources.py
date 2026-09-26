"""
Batch Normalization Pipeline for Amazon ML Challenge 2026.

Streams all 6 source TSVs, normalizes business names and addresses using
multiprocessing, writes optimized Parquet files to data/processed/,
and collects random before/after sample records for verification.
"""

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.address_normalization import normalize_address
from src.normalization import normalize_name


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_REPORT_DIR = Path("notebooks")
SAMPLE_REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = [
    ("train_source1", "train_source1.tsv", "train_source1.parquet"),
    ("train_source2", "train_source2.tsv", "train_source2.parquet"),
    ("train_source3", "train_source3.tsv", "train_source3.parquet"),
    ("test_source1", "test_source1.tsv", "test_source1.parquet"),
    ("test_source2", "test_source2.tsv", "test_source2.parquet"),
    ("test_source3", "test_source3.tsv", "test_source3.parquet"),
]

PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("business_name", pa.string()),
    ("business_address", pa.string()),
    ("norm_name", pa.string()),
    ("norm_name_no_legal", pa.string()),
    ("legal_type", pa.string()),
    ("norm_address", pa.string()),
    ("addr_street_number", pa.string()),
    ("addr_street_name", pa.string()),
    ("addr_unit", pa.string()),
    ("addr_landmark", pa.string()),
    ("addr_city", pa.string()),
    ("addr_state", pa.string()),
    ("addr_postal_code", pa.string()),
])


def _worker_normalize_batch(
    batch: Tuple[List[str], List[Any], List[Any], List[str]],
) -> Dict[str, List[Any]]:
    """Worker function to process a batch of records in parallel."""
    entity_ids, names, addrs, countries = batch

    norm_names = []
    norm_names_no_legal = []
    legal_types = []

    norm_addrs = []
    street_nums = []
    street_names = []
    units = []
    landmarks = []
    cities = []
    states = []
    postals = []

    for name, addr, country in zip(names, addrs, countries):
        n_res = normalize_name(name, country=country)
        norm_names.append(n_res.norm_name)
        norm_names_no_legal.append(n_res.norm_name_no_legal)
        legal_types.append(n_res.legal_type)

        a_res = normalize_address(addr, country=country)
        norm_addrs.append(a_res.cleaned_address)
        street_nums.append(a_res.street_number)
        street_names.append(a_res.street_name)
        units.append(a_res.unit)
        landmarks.append(a_res.landmark)
        cities.append(a_res.city)
        states.append(a_res.state_or_region)
        postals.append(a_res.postal_code)

    return {
        "entity_id": entity_ids,
        "country": countries,
        "business_name": [str(n) if pd.notna(n) else "" for n in names],
        "business_address": [str(a) if pd.notna(a) else "" for a in addrs],
        "norm_name": norm_names,
        "norm_name_no_legal": norm_names_no_legal,
        "legal_type": legal_types,
        "norm_address": norm_addrs,
        "addr_street_number": street_nums,
        "addr_street_name": street_names,
        "addr_unit": units,
        "addr_landmark": landmarks,
        "addr_city": cities,
        "addr_state": states,
        "addr_postal_code": postals,
    }


def process_single_source(
    name: str,
    tsv_filename: str,
    parquet_filename: str,
    executor: ProcessPoolExecutor,
    chunk_size: int = 50_000,
    sub_batch_size: int = 6_250,
    sample_k: int = 20,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Process one source TSV into parquet using streaming and parallel execution."""
    input_path = RAW_DIR / tsv_filename
    output_path = PROCESSED_DIR / parquet_filename

    print(f"\n[{name}] Starting normalization: {input_path} -> {output_path}")
    t0 = time.time()

    # Pre-select indices to sample for before/after comparison
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1  # exclude header

    random.seed(42)
    sample_indices = set(random.sample(range(total_lines), min(sample_k, total_lines)))
    sampled_records: List[Dict[str, Any]] = []

    writer = pq.ParquetWriter(
        output_path,
        schema=PARQUET_SCHEMA,
        compression="snappy",
    )

    rows_processed = 0
    try:
        reader = pd.read_csv(
            input_path,
            sep="\t",
            chunksize=chunk_size,
            dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
            keep_default_na=False,
        )

        for chunk_df in reader:
            chunk_len = len(chunk_df)
            global_start_idx = rows_processed

            # Split chunk into sub-batches for parallel worker mapping
            sub_batches = []
            for i in range(0, chunk_len, sub_batch_size):
                sub = chunk_df.iloc[i : i + sub_batch_size]
                sub_batches.append((
                    sub["entity_id"].tolist(),
                    sub["business_name"].tolist(),
                    sub["business_address"].tolist(),
                    sub["country"].tolist(),
                ))

            # Run in worker pool
            results = list(executor.map(_worker_normalize_batch, sub_batches))

            # Concatenate results for the chunk
            chunk_dict: Dict[str, List[Any]] = {col: [] for col in PARQUET_SCHEMA.names}
            for res in results:
                for col in PARQUET_SCHEMA.names:
                    chunk_dict[col].extend(res[col])

            # Check if any sampled index falls in this chunk
            for local_idx in range(chunk_len):
                global_idx = global_start_idx + local_idx
                if global_idx in sample_indices:
                    sampled_records.append({
                        "source": name,
                        "global_index": global_idx,
                        "entity_id": chunk_dict["entity_id"][local_idx],
                        "country": chunk_dict["country"][local_idx],
                        "orig_name": chunk_dict["business_name"][local_idx],
                        "norm_name": chunk_dict["norm_name"][local_idx],
                        "norm_name_no_legal": chunk_dict["norm_name_no_legal"][local_idx],
                        "legal_type": chunk_dict["legal_type"][local_idx],
                        "orig_address": chunk_dict["business_address"][local_idx],
                        "norm_address": chunk_dict["norm_address"][local_idx],
                        "addr_street_number": chunk_dict["addr_street_number"][local_idx],
                        "addr_street_name": chunk_dict["addr_street_name"][local_idx],
                        "addr_unit": chunk_dict["addr_unit"][local_idx],
                        "addr_landmark": chunk_dict["addr_landmark"][local_idx],
                        "addr_city": chunk_dict["addr_city"][local_idx],
                        "addr_state": chunk_dict["addr_state"][local_idx],
                        "addr_postal_code": chunk_dict["addr_postal_code"][local_idx],
                    })

            # Convert to PyArrow Table and write batch
            table = pa.Table.from_pydict(chunk_dict, schema=PARQUET_SCHEMA)
            writer.write_table(table)

            rows_processed += chunk_len
            elapsed = time.time() - t0
            print(f"[{name}] {rows_processed:,}/{total_lines:,} rows ({rows_processed/elapsed:.1f} r/s)", end="\r")

    finally:
        writer.close()

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    summary = {
        "source": name,
        "rows": rows_processed,
        "time_seconds": round(elapsed, 2),
        "rows_per_second": round(rows_processed / elapsed, 1) if elapsed > 0 else 0,
        "parquet_size_mb": round(file_size_mb, 2),
        "parquet_path": str(output_path),
    }
    print(f"\n[{name}] Complete! {rows_processed:,} rows written in {elapsed:.1f}s ({file_size_mb:.1f} MB)")
    return summary, sampled_records


def generate_sample_report(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Write markdown and JSON reports containing before/after sample records."""
    json_path = SAMPLE_REPORT_DIR / "normalization_samples.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    md_path = SAMPLE_REPORT_DIR / "normalization_samples.md"
    lines = [
        "# Normalization Sanity Check: Before & After Samples",
        "",
        "This document contains 20 randomly sampled records from each of the 6 normalized source files.",
        "",
    ]

    for source_name, records in all_samples.items():
        lines.extend([
            f"## Source: `{source_name}` (20 Random Samples)",
            "",
            "| # | ID & Country | Original Business Name | Normalized Name (`norm_name`) | Legal Type | Original Address | Normalized Address | Key Extracted Address Fields |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for idx, r in enumerate(records, 1):
            e_id = r["entity_id"]
            country = r["country"]
            o_name = r["orig_name"].replace("|", "\\|")
            n_name = r["norm_name"].replace("|", "\\|")
            lt = r["legal_type"] or "-"
            o_addr = r["orig_address"].replace("|", "\\|") if r["orig_address"] else "*[NULL]*"
            n_addr = r["norm_address"].replace("|", "\\|") if r["norm_address"] else "*[EMPTY]*"

            extracted_fields = []
            if r["addr_street_number"]:
                extracted_fields.append(f"num: {r['addr_street_number']}")
            if r["addr_street_name"]:
                extracted_fields.append(f"street: {r['addr_street_name']}")
            if r["addr_unit"]:
                extracted_fields.append(f"unit: {r['addr_unit']}")
            if r["addr_landmark"]:
                extracted_fields.append(f"landmark: {r['addr_landmark']}")
            if r["addr_city"]:
                extracted_fields.append(f"city: {r['addr_city']}")
            if r["addr_state"]:
                extracted_fields.append(f"state: {r['addr_state']}")
            if r["addr_postal_code"]:
                extracted_fields.append(f"zip: {r['addr_postal_code']}")
            ext_str = ", ".join(extracted_fields) if extracted_fields else "-"

            lines.append(
                f"| {idx} | `{e_id}` ({country}) | {o_name} | `{n_name}` | `{lt}` | {o_addr} | {n_addr} | {ext_str} |"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Sample report written to {md_path} and {json_path}")


def main():
    print("Initializing parallel normalization pipeline...")
    all_summaries = []
    all_samples: Dict[str, List[Dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=8) as executor:
        for name, tsv_file, parquet_file in SOURCE_FILES:
            summary, samples = process_single_source(
                name=name,
                tsv_filename=tsv_file,
                parquet_filename=parquet_file,
                executor=executor,
            )
            all_summaries.append(summary)
            all_samples[name] = samples

    generate_sample_report(all_samples)

    print("\n" + "=" * 80)
    print("NORMALIZATION PIPELINE SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
