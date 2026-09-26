"""
Unit tests for test inference and submission generator in src/inference.py.
"""

import unittest
from pathlib import Path
import tempfile
import pickle
import pandas as pd
import numpy as np

from src.inference import run_inference


class DummyModel:
    def predict_proba(self, X):
        # Return probability based on first feature
        p1 = np.clip(X[:, 0], 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class TestInference(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # 1. Save dummy model
        self.model_path = self.dir_path / "model.pkl"
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "model": DummyModel(),
                "feature_cols": ["f1", "f2"],
                "optimal_threshold": 0.80,
            }, f)

        # 2. Save candidate pairs parquet
        self.cp_path = self.dir_path / "candidate_pairs.parquet"
        cp_data = pd.DataFrame({
            "source1_entity_id": ["S1-01", "S1-01", "S1-02", "S1-03"],
            "candidate_entity_id": ["S2-01", "S3-01", "S2-02", "S2-03"],
        })
        cp_data.to_parquet(self.cp_path)

        # 3. Save feature matrix parquet
        self.fm_path = self.dir_path / "feature_matrix.parquet"
        fm_data = pd.DataFrame({
            "source1_entity_id": ["S1-01", "S1-01", "S1-02", "S1-03"],
            "candidate_entity_id": ["S2-01", "S3-01", "S2-02", "S2-03"],
            "f1": [0.95, 0.85, 0.40, 0.90],  # probs: 0.95, 0.85, 0.40, 0.90
            "f2": [0.1, 0.2, 0.3, 0.4],
        })
        fm_data.to_parquet(self.fm_path)

        # 4. Save test source1 TSV (includes S1-01, S1-02, S1-03, S1-04)
        self.s1_path = self.dir_path / "test_source1.tsv"
        with open(self.s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-01\tName 1\tAddr 1\tUS\n")
            f.write("S1-02\tName 2\tAddr 2\tUS\n")
            f.write("S1-03\tName 3\tAddr 3\tIndia\n")
            f.write("S1-04\tName 4\tAddr 4\tFrance\n")  # singleton with no candidates

        # 5. Output dir
        self.out_dir = self.dir_path / "output"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_inference_format_and_logic(self):
        stats = run_inference(
            model_path=str(self.model_path),
            feature_matrix_path=str(self.fm_path),
            candidate_pairs_path=str(self.cp_path),
            test_source1_path=str(self.s1_path),
            output_dir=str(self.out_dir),
            threshold=0.80,
            train_gt_path=None,
        )

        match_file = Path(stats["matching_file"])
        cand_file = Path(stats["candidate_file"])

        self.assertTrue(match_file.is_file())
        self.assertTrue(cand_file.is_file())

        # Check matching_results.tsv content
        match_lines = [l for l in match_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(match_lines[0], "source1_entity_id\tmatched_entity_ids")
        self.assertEqual(len(match_lines), 5)  # 1 header + 4 entities

        m_dict = dict(line.split("\t", 1) for line in match_lines[1:])
        # S1-01 has 2 matches >= 0.80 (S2-01: 0.95, S3-01: 0.85)
        self.assertEqual(m_dict["S1-01"], "S2-01,S3-01")
        # S1-02 candidate scored 0.40 < 0.80 -> empty string singleton
        self.assertEqual(m_dict["S1-02"], "")
        # S1-03 candidate scored 0.90 >= 0.80 -> S2-03
        self.assertEqual(m_dict["S1-03"], "S2-03")
        # S1-04 had no candidates -> empty string singleton
        self.assertEqual(m_dict["S1-04"], "")

        # Check candidate_pairs.tsv content
        cand_lines = [l for l in cand_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(cand_lines[0], "source1_entity_id\tcandidate_entity_ids")
        self.assertEqual(len(cand_lines), 5)

        c_dict = dict(line.split("\t", 1) for line in cand_lines[1:])
        self.assertEqual(c_dict["S1-01"], "S2-01,S3-01")
        self.assertEqual(c_dict["S1-02"], "S2-02")
        self.assertEqual(c_dict["S1-03"], "S2-03")
        self.assertEqual(c_dict["S1-04"], "")

        # Check stats
        self.assertEqual(stats["total_test_entities"], 4)
        self.assertEqual(stats["full_matched"], 2)
        self.assertEqual(stats["full_singletons"], 2)
        self.assertEqual(stats["scored_s1_count"], 3)
        self.assertEqual(stats["scored_singletons_count"], 1)
        self.assertAlmostEqual(stats["scored_singleton_rate"], 33.3333, places=2)


if __name__ == "__main__":
    unittest.main()
