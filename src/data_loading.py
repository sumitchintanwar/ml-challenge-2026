"""
Reusable Data Loading and Inspection Module for Amazon ML Challenge 2026.

Provides functions to load the 6 source TSVs and ground truth TSV,
and verify schema, shapes, uniqueness, prefixes, and null counts.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


def get_default_data_dir() -> Path:
    """Resolve the default data directory containing TSV files."""
    # Preferred location: data/raw/
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "raw",
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent.parent / "data" / "raw" / "student_resource" / "dataset",
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "train_source1.tsv").exists():
            return candidate
    return candidates[0]


def get_tsv_path(
    filename: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the full path to a TSV file, checking standard subfolders if needed."""
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()

    # Direct match (e.g., data/raw/train_source1.tsv)
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path

    # Check nested student_resource subdirectories (dataset/train, dataset/test)
    split = "train" if "train" in filename else "test"
    nested_path = base_dir / "student_resource" / "dataset" / split / filename
    if nested_path.exists():
        return nested_path

    subfolder_path = base_dir / split / filename
    if subfolder_path.exists():
        return subfolder_path

    return direct_path


def load_source(
    split: str,
    source: Union[int, str],
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a source TSV for a given split ('train' or 'test') and source (1, 2, or 3).

    Args:
        split: 'train' or 'test'
        source: 1, 2, or 3 (int or str)
        data_dir: Optional path to data directory
        **kwargs: Additional keyword arguments passed to pd.read_csv

    Returns:
        pd.DataFrame containing the source records.
    """
    filename = f"{split.lower()}_source{source}.tsv"
    path = get_tsv_path(filename, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {path}")

    # Set sep='\t' by default
    read_kwargs = {"sep": "\t", **kwargs}
    return pd.read_csv(path, **read_kwargs)


def load_ground_truth(
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load the train ground truth TSV.

    Args:
        data_dir: Optional path to data directory
        **kwargs: Additional keyword arguments passed to pd.read_csv

    Returns:
        pd.DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
    """
    path = get_tsv_path("train_ground_truth.tsv", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    # keep_default_na=False can be used if empty matched_entity_ids should remain ""
    # but standard pandas read produces NaN for missing values
    read_kwargs = {"sep": "\t", **kwargs}
    return pd.read_csv(path, **read_kwargs)


def load_all_sources(
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    """
    Load all three sources for a given split.

    Args:
        split: 'train' or 'test'
        data_dir: Optional path to data directory
        **kwargs: Additional keyword arguments passed to pd.read_csv

    Returns:
        Dictionary mapping 'source1', 'source2', 'source3' to DataFrames.
    """
    return {
        f"source{i}": load_source(split, i, data_dir=data_dir, **kwargs)
        for i in (1, 2, 3)
    }


def inspect_dataframe(
    df: pd.DataFrame,
    name: str,
    id_col: str = "entity_id",
) -> Dict[str, Any]:
    """
    Compute and print summary statistics for a dataset:
    - shape
    - dtypes
    - first 5 rows
    - number of nulls per column
    - number of unique entity_id prefixes (e.g. S1-, S2-, S3-)
    - entity_id uniqueness

    Args:
        df: DataFrame to inspect
        name: Name / label of the dataset
        id_col: Column name of the primary entity identifier

    Returns:
        Dictionary of computed summary statistics.
    """
    shape = df.shape
    dtypes = df.dtypes.to_dict()
    null_counts = df.isnull().sum().to_dict()
    is_unique = bool(df[id_col].is_unique)

    # Prefix extraction: prefix is defined by text up to and including the first hyphen (e.g., 'S1-')
    prefix_series = df[id_col].dropna().astype(str).str.extract(r"^([A-Za-z0-9]+-)", expand=False)
    prefix_counts = prefix_series.value_counts().to_dict()
    num_unique_prefixes = len(prefix_counts)

    print("=" * 80)
    print(f"DATASET: {name}")
    print("=" * 80)
    print(f"Shape: {shape[0]:,} rows x {shape[1]} columns")
    print("\nData Types:")
    for col, dtype in dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\nNull Values per Column:")
    for col, nulls in null_counts.items():
        pct = (nulls / shape[0]) * 100 if shape[0] > 0 else 0.0
        print(f"  - {col}: {nulls:,} nulls ({pct:.2f}%)")

    print(f"\nEntity ID Uniqueness ({id_col}):")
    print(f"  - Is unique: {is_unique} ({df[id_col].nunique():,} unique values out of {shape[0]:,} rows)")

    print(f"\nPrefix Counts ({id_col}):")
    print(f"  - Number of unique prefixes: {num_unique_prefixes}")
    for prefix, count in prefix_counts.items():
        print(f"  - Prefix '{prefix}': {count:,} records")

    print("\nFirst 5 Rows:")
    print(df.head(5).to_string())
    print("=" * 80 + "\n")

    return {
        "name": name,
        "shape": shape,
        "dtypes": {str(k): str(v) for k, v in dtypes.items()},
        "null_counts": null_counts,
        "is_unique": is_unique,
        "prefix_counts": prefix_counts,
        "num_unique_prefixes": num_unique_prefixes,
    }


def inspect_all_datasets(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load each of the 6 source TSVs and the ground truth TSV and inspect them.

    Returns:
        Dictionary of inspection results keyed by dataset name.
    """
    specs = [
        ("train_source1", lambda: load_source("train", 1, data_dir=data_dir), "entity_id"),
        ("train_source2", lambda: load_source("train", 2, data_dir=data_dir), "entity_id"),
        ("train_source3", lambda: load_source("train", 3, data_dir=data_dir), "entity_id"),
        ("train_ground_truth", lambda: load_ground_truth(data_dir=data_dir), "source1_entity_id"),
        ("test_source1", lambda: load_source("test", 1, data_dir=data_dir), "entity_id"),
        ("test_source2", lambda: load_source("test", 2, data_dir=data_dir), "entity_id"),
        ("test_source3", lambda: load_source("test", 3, data_dir=data_dir), "entity_id"),
    ]

    results = {}
    for name, loader, id_col in specs:
        print(f"Loading {name}...")
        df = loader()
        results[name] = inspect_dataframe(df, name=name, id_col=id_col)

    return results


if __name__ == "__main__":
    inspect_all_datasets()
