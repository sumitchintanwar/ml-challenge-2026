"""
Unit tests for LightGBM training and validation pipeline.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import pickle

from src.train import get_or_create_entity_split, train_and_validate_lgbm
from src.features import FEATURE_COLUMNS


class TestTrain(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create synthetic source1 parquet
        s1_data = pd.DataFrame([
            {"entity_id": f"S1-{i}", "country": "US" if i % 2 == 0 else "India"}
            for i in range(100)
        ])
        self.s1_path = self.tmp_path / "s1.parquet"
        s1_data.to_parquet(self.s1_path)

        # Create synthetic ground truth
        gt_data = pd.DataFrame([
            {"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i},S3-{i}" if i % 5 != 0 else ""}
            for i in range(100)
        ])
        self.gt_path = self.tmp_path / "gt.tsv"
        gt_data.to_csv(self.gt_path, sep="\t", index=False)

        # Create synthetic feature matrix
        rows = []
        for i in range(100):
            # 5 candidates per entity
            s1_id = f"S1-{i}"
            for c in range(5):
                cand_id = f"S2-{i}_{c}"
                is_match = 1 if c == 0 and (i % 5 != 0) else 0
                row = {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "label": is_match,
                }
                for f_col in FEATURE_COLUMNS:
                    row[f_col] = float(np.random.uniform(0.7, 1.0) if is_match else np.random.uniform(0.0, 0.4))
                rows.append(row)

        self.feat_df = pd.DataFrame(rows)
        self.feat_path = self.tmp_path / "feature_matrix.parquet"
        self.feat_df.to_parquet(self.feat_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_entity_split(self):
        eids = [f"S1-{i}" for i in range(100)]
        train_e, val_e, meta = get_or_create_entity_split(
            s1_entities=eids,
            s1_parquet_path=str(self.s1_path),
            gt_tsv_path=str(self.gt_path),
            split_output_path=str(self.tmp_path / "split.json"),
            test_size=0.20,
            random_seed=42,
        )

        self.assertEqual(len(train_e), 80)
        self.assertEqual(len(val_e), 20)
        self.assertEqual(len(set(train_e).intersection(set(val_e))), 0)

    def test_train_and_validate_pipeline(self):
        model_out = self.tmp_path / "model.pkl"
        split_out = self.tmp_path / "split.json"

        results = train_and_validate_lgbm(
            feature_matrix_path=str(self.feat_path),
            model_output_path=str(model_out),
            s1_parquet_path=str(self.s1_path),
            gt_tsv_path=str(self.gt_path),
            split_path=str(split_out),
            test_size=0.20,
            random_seed=42,
        )

        self.assertIn("auc_pr", results)
        self.assertGreater(results["auc_pr"], 0.5)
        self.assertIn("confusion_matrix", results)
        self.assertTrue(model_out.exists())

        # Check loaded model
        with open(model_out, "rb") as f:
            saved = pickle.load(f)
        self.assertIn("model", saved)
        self.assertIn("feature_cols", saved)


if __name__ == "__main__":
    unittest.main()
