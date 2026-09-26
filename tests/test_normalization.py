"""
Unit tests for business name normalization module.
"""

import unittest
from src.normalization import (
    clean_name_text,
    extract_legal_form,
    normalize_name,
)


class TestNameNormalization(unittest.TestCase):

    def test_standard_us_company(self):
        """Test standard US company with suffix."""
        res = normalize_name("Custom Wealth Services LLC", country="US")
        self.assertEqual(res.norm_name, "custom wealth services llc")
        self.assertEqual(res.norm_name_no_legal, "custom wealth services")
        self.assertEqual(res.legal_type, "llc")

    def test_displaced_legal_prefix_us(self):
        """Test displaced legal prefix (LLC at start)."""
        res = normalize_name("LLC Moncada Léarning Center", country="US")
        self.assertEqual(res.norm_name, "moncada learning center llc")
        self.assertEqual(res.norm_name_no_legal, "moncada learning center")
        self.assertEqual(res.legal_type, "llc")

    def test_displaced_legal_prefix_india(self):
        """Test displaced legal prefix in Indian company (Pvt. at start, Ltd at end)."""
        res = normalize_name("Pvt. EFS Print Ventures Ltd.", country="India")
        self.assertEqual(res.norm_name, "efs print ventures limited")
        self.assertEqual(res.norm_name_no_legal, "efs print ventures")
        self.assertEqual(res.legal_type, "limited")

    def test_noise_prefix_symbols(self):
        """Test stripping << and -- noise symbols."""
        res1 = normalize_name("<< Team Ecole", country="France")
        self.assertEqual(res1.norm_name, "team ecole")
        self.assertEqual(res1.norm_name_no_legal, "team ecole")

        res2 = normalize_name("-- Holloway Peak Inc Seafood", country="US")
        self.assertEqual(res2.legal_type, "inc")
        self.assertIn("holloway peak seafood", res2.norm_name_no_legal)

    def test_trade_symbols_expansion(self):
        """Test expansion of & symbol and French legal form."""
        res = normalize_name("Thermal & Fils SASU", country="France")
        self.assertEqual(res.norm_name, "thermal and fils sasu")
        self.assertEqual(res.norm_name_no_legal, "thermal and fils")
        self.assertEqual(res.legal_type, "sasu")

    def test_apostrophes_and_possessives(self):
        """Test apostrophe handling."""
        res = normalize_name("Orelee's Barbershop", country="US")
        self.assertEqual(res.norm_name, "orelees barbershop")

    def test_empty_and_null(self):
        """Test edge cases with None and empty strings."""
        res_none = normalize_name(None, country="US")
        self.assertEqual(res_none.norm_name, "")
        self.assertIsNone(res_none.legal_type)

        res_empty = normalize_name("   ", country="India")
        self.assertEqual(res_empty.norm_name, "")


if __name__ == "__main__":
    unittest.main()
