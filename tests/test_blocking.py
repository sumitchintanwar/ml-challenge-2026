"""
Unit tests for multi-strategy blocking module.
"""

import unittest
from pathlib import Path
import pandas as pd
from src.blocking import (
    BlockingConfig,
    MultiStrategyBlocker,
    extract_address_blocking_keys,
    extract_phonetic_blocking_keys,
    extract_token_blocking_keys,
)


class TestBlocking(unittest.TestCase):

    def test_token_blocking_keys(self):
        """Test exact and fuzzy token key extraction."""
        name = "johnson electric motors"
        keys = extract_token_blocking_keys(name)
        # Order invariant: electric_johnson
        self.assertIn("TOK2:electric_johnson", keys)
        self.assertIn("LONG:electric", keys)
        self.assertTrue(any(k.startswith("PRE6:") for k in keys))

    def test_phonetic_blocking_keys(self):
        """Test phonetic Metaphone and Soundex key extraction."""
        name = "johnson electric"
        keys = extract_phonetic_blocking_keys(name)
        # Metaphone of johnson is JNSN
        self.assertIn("META1:JNSN", keys)
        self.assertTrue(any(k.startswith("SND1:") for k in keys))

    def test_address_blocking_keys(self):
        """Test postal code and city/state address key generation."""
        keys = extract_address_blocking_keys(
            name="Apex Tools",
            postal_code="90210",
            city="Beverly Hills",
            state="CA",
        )
        self.assertIn("POST:90210_ap", keys)
        self.assertIn("GEO:CA_beverly hills_ap", keys)

    def test_blocker_sklearn_engine(self):
        """Test end-to-end blocking with sklearn NearestNeighbors engine."""
        corpus_data = [
            {"entity_id": "S2-1", "country": "US", "norm_name": "apple computer inc", "norm_name_no_legal": "apple computer", "addr_postal_code": "95014", "addr_city": "cupertino", "addr_state": "CA"},
            {"entity_id": "S2-2", "country": "US", "norm_name": "general electric company", "norm_name_no_legal": "general electric", "addr_postal_code": "02210", "addr_city": "boston", "addr_state": "MA"},
            {"entity_id": "S3-1", "country": "US", "norm_name": "apple computers co", "norm_name_no_legal": "apple computers", "addr_postal_code": "95014", "addr_city": "cupertino", "addr_state": "CA"},
            {"entity_id": "S2-3", "country": "India", "norm_name": "tata motors limited", "norm_name_no_legal": "tata motors", "addr_postal_code": "400001", "addr_city": "mumbai", "addr_state": "Maharashtra"},
        ]
        corpus_df = pd.DataFrame(corpus_data)

        query_data = [
            {"entity_id": "S1-1", "country": "US", "norm_name": "apple computer", "norm_name_no_legal": "apple computer", "addr_postal_code": "95014", "addr_city": "cupertino", "addr_state": "CA"},
            {"entity_id": "S1-2", "country": "India", "norm_name": "tata motors pvt ltd", "norm_name_no_legal": "tata motors", "addr_postal_code": "400001", "addr_city": "mumbai", "addr_state": "Maharashtra"},
        ]
        query_df = pd.DataFrame(query_data)

        config = BlockingConfig(knn_engine="sklearn", top_k_tfidf=5, min_tfidf_sim=0.1)
        blocker = MultiStrategyBlocker(config).fit(corpus_df)
        candidates = blocker.block_queries(query_df)

        self.assertIn("S1-1", candidates)
        self.assertIn("S2-1", candidates["S1-1"])
        self.assertIn("S3-1", candidates["S1-1"])
        # Cross-country isolation: India entity S2-3 should not be in S1-1 candidates
        self.assertNotIn("S2-3", candidates["S1-1"])

        # Check long DataFrame conversion
        pairs_df = blocker.to_pairs_long_df(candidates)
        self.assertEqual(list(pairs_df.columns), ["source1_entity_id", "candidate_entity_id"])
        self.assertTrue(len(pairs_df) > 0)

    def test_blocker_faiss_engine(self):
        """Test end-to-end blocking with FAISS engine."""
        corpus_data = [
            {"entity_id": "S2-1", "country": "US", "norm_name": "johnson electric inc", "norm_name_no_legal": "johnson electric", "addr_postal_code": "10001", "addr_city": "new york", "addr_state": "NY"},
            {"entity_id": "S3-1", "country": "US", "norm_name": "johnson electric motors", "norm_name_no_legal": "johnson electric motors", "addr_postal_code": "10001", "addr_city": "new york", "addr_state": "NY"},
            {"entity_id": "S2-2", "country": "US", "norm_name": "walmart supercenter", "norm_name_no_legal": "walmart supercenter", "addr_postal_code": "72716", "addr_city": "bentonville", "addr_state": "AR"},
        ]
        corpus_df = pd.DataFrame(corpus_data)

        query_data = [
            {"entity_id": "S1-1", "country": "US", "norm_name": "johnson electric company", "norm_name_no_legal": "johnson electric", "addr_postal_code": "10001", "addr_city": "new york", "addr_state": "NY"},
        ]
        query_df = pd.DataFrame(query_data)

        config = BlockingConfig(knn_engine="faiss", faiss_dim=16, top_k_tfidf=5, min_tfidf_sim=0.01)
        blocker = MultiStrategyBlocker(config).fit(corpus_df)
        candidates = blocker.block_queries(query_df)

        self.assertIn("S1-1", candidates)
        self.assertIn("S2-1", candidates["S1-1"])

    def test_evaluate_blocking_quality(self):
        """Test precision and recall metrics calculation."""
        candidates = {
            "S1-1": ["S2-10", "S2-11", "S3-12"],
            "S1-2": ["S2-20"],
            "S1-3": [],  # singleton
        }
        gt_df = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-12"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-20,S3-99"},  # S3-99 missed
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},  # true singleton
        ])

        stats = MultiStrategyBlocker.evaluate_blocking_quality(candidates, gt_df, total_corpus_entities=1000)
        # S1-1 has 2/2 hits, S1-2 has 1/2 hits -> 3/4 pair recall = 75.0%
        self.assertEqual(stats["pair_recall"], 75.0)
        # Both matched entities had at least 1 hit -> 2/2 entity recall = 100.0%
        self.assertEqual(stats["entity_recall"], 100.0)
        self.assertTrue(stats["reduction_ratio"] > 99.0)


if __name__ == "__main__":
    unittest.main()
