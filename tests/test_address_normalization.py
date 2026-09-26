"""
Unit tests for address normalization module with real dataset examples.
"""

import unittest
from src.address_normalization import (
    NormalizedAddress,
    clean_text,
    expand_abbreviations,
    extract_landmarks,
    extract_unit_info,
    normalize_address,
)


class TestAddressNormalization(unittest.TestCase):

    def test_abbreviation_expansion(self):
        """Test expansion of common road, unit, and regional abbreviations."""
        sample = "100 Main Rd, Apt 4B, 2nd Fl, West Blvd, North Pkwy"
        expanded = expand_abbreviations(sample)
        self.assertIn("road", expanded.lower())
        self.assertIn("apartment", expanded.lower())
        self.assertIn("floor", expanded.lower())
        self.assertIn("boulevard", expanded.lower())
        self.assertIn("parkway", expanded.lower())

    def test_po_box_and_suite(self):
        """Test PO Box and Suite normalization."""
        sample = "P.O. Box 1234, Ste 400"
        expanded = expand_abbreviations(sample)
        self.assertIn("po box", expanded.lower())
        self.assertIn("suite", expanded.lower())

    def test_landmark_extraction_english_and_parentheses(self):
        """Test extracting landmarks into a dedicated field."""
        raw = "Mulund Link Road, Near Fortis Hospital, Bhandup West, Mumbai"
        rem_addr, landmark = extract_landmarks(raw)
        self.assertIsNotNone(landmark)
        self.assertIn("Near Fortis Hospital", landmark)
        self.assertNotIn("Near Fortis Hospital", rem_addr)

        # Parenthesized landmark
        raw_paren = "Mahodadhi Bhawan (Next To Iter College) Panchasakha Nagar"
        rem_paren, landmark_paren = extract_landmarks(raw_paren)
        self.assertIsNotNone(landmark_paren)
        self.assertIn("Next To Iter College", landmark_paren)
        self.assertNotIn("Next To Iter College", rem_paren)

    def test_landmark_extraction_hindi(self):
        """Test extracting Indian landmark phrases like 'ke pas'."""
        raw = "Karauli, Rajasthan, Pani Ki Tanki Ke Pas Choubepada"
        rem_addr, landmark = extract_landmarks(raw)
        self.assertIsNotNone(landmark)
        self.assertIn("Pani Ki Tanki Ke Pas", landmark)

    def test_us_standard_address(self):
        """Test US standard address parsing."""
        raw = "1795 Westchester Drive, High Point, NC"
        norm = normalize_address(raw, country="US")
        self.assertEqual(norm.country, "US")
        self.assertEqual(norm.street_number, "1795")
        self.assertIn("westchester", norm.street_name.lower())
        self.assertEqual(norm.state_or_region, "NC")
        self.assertEqual(norm.city, "High Point")

    def test_us_inverted_with_unit(self):
        """Test US inverted address with unit."""
        raw = "IA, Iowa City, 1064 Newton Rd, Unit 11"
        norm = normalize_address(raw, country="US")
        self.assertEqual(norm.state_or_region, "IA")
        self.assertEqual(norm.city, "Iowa City")
        self.assertEqual(norm.street_number, "1064")
        self.assertIn("newton", norm.street_name.lower())
        self.assertIn("unit 11", norm.unit.lower())

    def test_us_with_po_box(self):
        """Test US address with PO Box."""
        raw = "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio"
        norm = normalize_address(raw, country="US")
        self.assertEqual(norm.state_or_region, "OH")
        self.assertEqual(norm.city, "Cincinnati")
        self.assertIn("po box 6009", norm.unit.lower())
        self.assertEqual(norm.street_number, "1")

    def test_us_punctuation_noise(self):
        """Test US address with noise characters."""
        raw = "##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri 63101"
        norm = normalize_address(raw, country="US")
        self.assertEqual(norm.street_number, "8")
        self.assertEqual(norm.state_or_region, "MO")
        self.assertEqual(norm.postal_code, "63101")
        self.assertIn("floor 0", norm.unit.lower())

    def test_india_with_landmark_and_pin(self):
        """Test India address with landmark, PIN, and state."""
        raw = "2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra 400078"
        norm = normalize_address(raw, country="India")
        self.assertEqual(norm.country, "India")
        self.assertEqual(norm.postal_code, "400078")
        self.assertEqual(norm.state_or_region, "Maharashtra")
        self.assertIsNotNone(norm.landmark)
        self.assertIn("Near Fortis Hospital", norm.landmark)
        self.assertEqual(norm.street_number, "2505")

    def test_india_door_and_cross(self):
        """Test Indian Door No and Cross/Main pattern."""
        raw = "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, Karnataka 560041"
        norm = normalize_address(raw, country="India")
        self.assertIn("183", norm.street_number)
        self.assertEqual(norm.postal_code, "560041")
        self.assertEqual(norm.state_or_region, "Karnataka")

    def test_india_khasra_no(self):
        """Test Khasra number pattern in Delhi."""
        raw = "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041"
        norm = normalize_address(raw, country="India")
        self.assertEqual(norm.postal_code, "110041")
        self.assertEqual(norm.state_or_region, "Delhi")
        self.assertIn("570/13", norm.street_number)

    def test_france_unseen_country_degradation(self):
        """Test graceful degradation on unseen country label (France)."""
        raw = "175 Boulevard du President Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine 33000"
        norm = normalize_address(raw, country="France")
        self.assertEqual(norm.country, "France")
        self.assertEqual(norm.street_number, "175")
        self.assertEqual(norm.postal_code, "33000")
        self.assertIn("Nouvelle-Aquitaine", norm.state_or_region)
        self.assertEqual(norm.city, "Bordeaux")
        self.assertTrue(len(norm.tokens) > 0)

    def test_france_french_abbreviation_and_bis(self):
        """Test French address with '5 bis' and abbreviation."""
        raw = "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac 33260"
        norm = normalize_address(raw, country="France")
        self.assertEqual(norm.street_number, "5 bis")
        self.assertEqual(norm.postal_code, "33260")
        self.assertIn("Nouvelle-Aquitaine", norm.state_or_region)
        self.assertEqual(norm.city, "La Teste-de-Buch")

    def test_french_street_abbreviation(self):
        """Test French '63 R. DE DIEPPE' expands to 'rue'."""
        raw = "63 R. DE DIEPPE, LILLE, Hauts-de-France 59000"
        norm = normalize_address(raw, country="France")
        self.assertIn("rue", norm.cleaned_address.lower())
        self.assertEqual(norm.street_number, "63")
        self.assertEqual(norm.postal_code, "59000")
        self.assertEqual(norm.city, "LILLE")

    def test_edge_cases_null_and_empty(self):
        """Test handling of None, NaN, and empty strings."""
        norm_none = normalize_address(None, country="US")
        self.assertEqual(norm_none.cleaned_address, "")
        self.assertIsNone(norm_none.street_number)

        norm_empty = normalize_address("", country="India")
        self.assertEqual(norm_empty.cleaned_address, "")

        norm_nan = normalize_address("nan", country="France")
        self.assertEqual(norm_nan.cleaned_address, "")

        # Verify feature dict generation doesn't crash on empty
        feat_dict = norm_none.as_feature_dict()
        self.assertIsInstance(feat_dict, dict)
        self.assertEqual(feat_dict["cleaned_address"], "")


if __name__ == "__main__":
    unittest.main()
