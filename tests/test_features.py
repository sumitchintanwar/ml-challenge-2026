"""
Unit tests for FeatureExtractor in src/features.py.
"""

import unittest
import numpy as np
import pandas as pd
from src.features import (
    FeatureExtractor,
    compute_string_pair_metrics,
    compute_tfidf_cosine_similarity,
    load_ground_truth_dict,
    build_entity_lookup,
)


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.sample_names = [
            "apple inc",
            "microsoft corporation",
            "tata consultancy services",
            "walmart stores",
            "reliance industries limited",
        ]
        self.extractor = FeatureExtractor(tfidf_max_features=1000).fit(self.sample_names)

    def test_compute_string_pair_metrics_identical(self):
        s1 = ["apple inc", "walmart store"]
        s2 = ["apple inc", "walmart store"]
        res = compute_string_pair_metrics(s1, s2, prefix="name")

        self.assertTrue(np.allclose(res["feat_name_levenshtein"], [1.0, 1.0]))
        self.assertTrue(np.allclose(res["feat_name_jaro_winkler"], [1.0, 1.0]))
        self.assertTrue(np.allclose(res["feat_name_token_sort"], [1.0, 1.0]))
        self.assertTrue(np.allclose(res["feat_name_token_set"], [1.0, 1.0]))
        self.assertTrue(np.allclose(res["feat_name_len_diff_ratio"], [0.0, 0.0]))
        self.assertTrue(np.array_equal(res["feat_name_exact_match"], [1, 1]))

    def test_compute_string_pair_metrics_partial(self):
        s1 = ["apple inc"]
        s2 = ["apple incorporated"]
        res = compute_string_pair_metrics(s1, s2, prefix="name")

        self.assertGreater(res["feat_name_jaro_winkler"][0], 0.8)
        self.assertGreater(res["feat_name_token_set"][0], 0.6)
        self.assertEqual(res["feat_name_exact_match"][0], 0)
        self.assertGreater(res["feat_name_len_diff_ratio"][0], 0.0)

    def test_compute_tfidf_cosine_similarity(self):
        s1 = ["apple inc", "tata motors"]
        s2 = ["apple inc", "reliance fresh"]
        sim = compute_tfidf_cosine_similarity(s1, s2, self.extractor.vectorizer)

        self.assertAlmostEqual(sim[0], 1.0, places=4)
        self.assertLess(sim[1], 0.3)

    def test_extract_features_end_to_end(self):
        pairs_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-2"},
        ])
        s1_lookup = {
            "S1-1": ("apple computer", "1 infinite loop cupertino ca", "US"),
        }
        corpus_lookup = {
            "S2-1": ("apple computer inc", "1 infinite loop cupertino ca 95014", "US"),
            "S3-2": ("walmart stores", "702 sw 8th st bentonville ar", "US"),
        }
        gt_dict = {
            "S1-1": {"S2-1"},
        }

        res_df = self.extractor.extract_features_df(
            pairs_df=pairs_df,
            s1_lookup=s1_lookup,
            corpus_lookup=corpus_lookup,
            ground_truth_dict=gt_dict,
        )

        self.assertEqual(len(res_df), 2)
        self.assertIn("label", res_df.columns)
        self.assertEqual(res_df.loc[0, "label"], 1)
        self.assertEqual(res_df.loc[1, "label"], 0)
        self.assertGreater(res_df.loc[0, "feat_name_jaro_winkler"], res_df.loc[1, "feat_name_jaro_winkler"])
        self.assertEqual(res_df.loc[0, "feat_country_match"], 1)

    def test_empty_strings(self):
        s1 = [""]
        s2 = [""]
        res = compute_string_pair_metrics(s1, s2, prefix="name")
        self.assertEqual(res["feat_name_exact_match"][0], 1)
        self.assertEqual(res["feat_name_len_diff_ratio"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
