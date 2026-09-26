"""
Stratified Train/Validation Split for Amazon ML Challenge 2026.

Splits Source 1 entities into Train (80%) and Validation (20%) using a fixed
random seed, stratified by country and singleton status.
Persists split entity_id lists and metadata to data/processed/train_val_split.json.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
OUTPUT_JSON_PATH = PROCESSED_DIR / "train_val_split.json"
RANDOM_SEED = 42
TEST_SIZE = 0.20


def create_train_val_split() -> Dict[str, Any]:
    print("Loading train_source1 and train_ground_truth...")
    s1_path = PROCESSED_DIR / "train_source1.parquet"
    gt_path = Path("data/raw/train_ground_truth.tsv")

    s1 = pd.read_parquet(s1_path, columns=["entity_id", "country"])
    gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)

    print(f"Loaded {len(s1):,} Source 1 entities and {len(gt):,} ground truth rows.")
    df = s1.merge(gt, left_on="entity_id", right_on="source1_entity_id")

    # Define stratification criteria: country and singleton status
    df["is_singleton"] = (df["matched_entity_ids"] == "")
    df["stratum"] = df["country"] + "_" + df["is_singleton"].apply(lambda x: "singleton" if x else "matched")

    print("\nStrata distribution across all Source 1 entities:")
    strata_counts = df["stratum"].value_counts()
    for stratum, count in strata_counts.items():
        print(f"  - {stratum:<25}: {count:,} ({count / len(df) * 100:.2f}%)")

    print(f"\nSplitting into Train ({(1-TEST_SIZE)*100:.0f}%) and Validation ({TEST_SIZE*100:.0f}%) with seed={RANDOM_SEED}...")
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["stratum"],
    )

    # Sanity checks
    assert len(set(train_df["entity_id"]).intersection(set(val_df["entity_id"]))) == 0, "Data leakage! Entities overlap between train and val."
    assert len(train_df) + len(val_df) == len(df), "Row count mismatch after split."

    train_ids = train_df["entity_id"].tolist()
    val_ids = val_df["entity_id"].tolist()

    # Strata breakdown
    train_strata = train_df["stratum"].value_counts().to_dict()
    val_strata = val_df["stratum"].value_counts().to_dict()

    summary_stats = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "total_entities": len(df),
        "train_count": len(train_ids),
        "train_percentage": round(len(train_ids) / len(df) * 100, 4),
        "val_count": len(val_ids),
        "val_percentage": round(len(val_ids) / len(df) * 100, 4),
        "strata_breakdown": {
            s: {
                "total": int(strata_counts[s]),
                "train_count": int(train_strata.get(s, 0)),
                "train_pct": round(train_strata.get(s, 0) / len(train_df) * 100, 4),
                "val_count": int(val_strata.get(s, 0)),
                "val_pct": round(val_strata.get(s, 0) / len(val_df) * 100, 4),
            }
            for s in sorted(strata_counts.keys())
        },
    }

    print("\nSplit verification:")
    print(f"  Train: {summary_stats['train_count']:,} entities ({summary_stats['train_percentage']}%)")
    print(f"  Val:   {summary_stats['val_count']:,} entities ({summary_stats['val_percentage']}%)")

    print("\nStrata proportions comparison (Train vs Val):")
    for s, info in summary_stats["strata_breakdown"].items():
        print(f"  - {s:<25}: Train={info['train_pct']:.2f}% ({info['train_count']:,}) | Val={info['val_pct']:.2f}% ({info['val_count']:,})")

    # Persist JSON file
    payload = {
        "metadata": summary_stats,
        "train_entity_ids": train_ids,
        "val_entity_ids": val_ids,
    }

    print(f"\nWriting split to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_mb = OUTPUT_JSON_PATH.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {OUTPUT_JSON_PATH} ({file_size_mb:.2f} MB)")
    return summary_stats


if __name__ == "__main__":
    create_train_val_split()
