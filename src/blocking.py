"""
Multi-Strategy Blocking Module for Amazon ML Challenge 2026.

Generates candidate (source1_entity_id, candidate_entity_id) pairs using:
  (a) TF-IDF character n-gram cosine similarity with top-K search (sklearn or FAISS)
  (b) Exact/fuzzy token-overlap inverted index keys
  (c) Phonetic blocking keys (Soundex/Metaphone via jellyfish)
  (d) Address-based blocking keys (postal code / city token + name prefix)

Combines all candidate sets via union with configurable thresholds and limits.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import faiss
import jellyfish
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import unidecode



# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    """Configurable parameters for the multi-strategy blocking pipeline."""
    # (a) TF-IDF k-NN parameters
    enable_tfidf: bool = True
    top_k_tfidf: int = 20
    min_tfidf_sim: float = 0.20
    tfidf_ngram_range: Tuple[int, int] = (3, 3)
    tfidf_max_features: int = 40000
    knn_engine: str = "sklearn"  # 'sklearn' (sparse brute force) or 'faiss' (dense SVD)
    faiss_dim: int = 128

    # (b) Token overlap parameters
    enable_token_overlap: bool = True
    min_token_len: int = 3

    # (c) Phonetic parameters
    enable_phonetic: bool = True
    min_phonetic_len: int = 3

    # (d) Address blocking parameters
    enable_address: bool = True

    # General constraints
    max_block_size: int = 1200  # drop keys containing > max_block_size items to prevent mega-blocks
    max_candidates_per_query: int = 200  # upper limit on unioned candidates per S1 entity
    country_partition: bool = True  # block within country to prevent invalid cross-country pairs


# ---------------------------------------------------------------------------
# Key Generation Utilities
# ---------------------------------------------------------------------------

COMMON_STOP_WORDS = {
    "and", "the", "of", "for", "in", "on", "at", "to", "a", "an",
    "co", "ltd", "inc", "corp", "llc", "lp", "company", "corporation",
    "limited", "pvt", "services", "enterprises", "group", "holdings",
}

DOMAIN_REGEX = re.compile(r"https?://|www\.|@|\.(com|org|net|in|co\.in|edu|gov|io|ai)\b", re.IGNORECASE)
PUNCT_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
INDIAN_PIN_REGEX = re.compile(r"\b[1-9][0-9]{5}\b")
PLOT_DOOR_REGEX = re.compile(r"\b(?:door|flat|flt|h\.?no|plot|d|b|c|sector|sec|kh\.?no|shop|shp)?[-.\s]*([a-zA-Z]?[-/]?\d+[/a-zA-Z0-9-]*)\b", re.IGNORECASE)

US_STATE_CODES = {
    "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS",
    "CA": "CALIFORNIA", "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE",
    "FL": "FLORIDA", "GA": "GEORGIA", "HI": "HAWAII", "ID": "IDAHO",
    "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA", "KS": "KANSAS",
    "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND",
    "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI",
    "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA",
    "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK",
    "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA",
    "OR": "OREGON", "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA", "TN": "TENNESSEE", "TX": "TEXAS", "UT": "UTAH",
    "VT": "VERMONT", "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST VIRGINIA",
    "WI": "WISCONSIN", "WY": "WYOMING", "DC": "DISTRICT OF COLUMBIA",
}

INDIAN_STATE_ABBRS = {
    "UP": "UTTAR PRADESH", "MH": "MAHARASHTRA", "KA": "KARNATAKA",
    "TN": "TAMIL NADU", "TG": "TELANGANA", "TS": "TELANGANA",
    "AP": "ANDHRA PRADESH", "DL": "DELHI", "WB": "WEST BENGAL",
    "GJ": "GUJARAT", "RJ": "RAJASTHAN", "MP": "MADHYA PRADESH",
    "KL": "KERALA", "PB": "PUNJAB", "HR": "HARYANA", "BR": "BIHAR",
    "OR": "ODISHA", "OD": "ODISHA", "AS": "ASSAM", "JH": "JHARKHAND",
    "UT": "UTTARAKHAND", "UK": "UTTARAKHAND", "HP": "HIMACHAL PRADESH",
    "GA": "GOA", "TR": "TRIPURA", "ML": "MEGHALAYA", "MN": "MANIPUR",
    "MZ": "MIZORAM", "NL": "NAGALAND", "SK": "SIKKIM", "AR": "ARUNACHAL PRADESH",
    "CH": "CHANDIGARH", "PY": "PUDUCHERRY",
    # Indic script state names
    "পশ্চিমবঙ্গ": "WEST BENGAL",
    "తెలంగాణ": "TELANGANA",
    "दिल्ली": "DELHI",
    "ગુજરાત": "GUJARAT",
    "महाराष्ट्र": "MAHARASHTRA",
    "தமிழ்நாடு": "TAMIL NADU",
    "കേരളം": "KERALA",
    "കേരള": "KERALA",
    "ಕರ್ನಾಟಕ": "KARNATAKA",
    "उत्तर प्रदेश": "UTTAR PRADESH",
    "राजस्थान": "RAJASTHAN",
    "हरियाणा": "HARYANA",
    "मध्य प्रदेश": "MADHYA PRADESH",
    "बिहार": "BIHAR",
    "पंजाब": "PUNJAB",
    "आंध्र प्रदेश": "ANDHRA PRADESH",
    "ଓଡ଼ିଶା": "ODISHA",
    "অসম": "ASSAM",
}

US_STATE_LOOKUP = {}
for code, name in US_STATE_CODES.items():
    US_STATE_LOOKUP[name.lower()] = name
    US_STATE_LOOKUP[code.lower()] = name

_us_sorted_keys = sorted(US_STATE_LOOKUP.keys(), key=len, reverse=True)
US_STATE_COMBINED_REGEX = re.compile(r"\b(" + "|".join(re.escape(k) for k in _us_sorted_keys) + r")\b", re.IGNORECASE)

IN_STATE_LOOKUP = {}
for abbr, name in INDIAN_STATE_ABBRS.items():
    IN_STATE_LOOKUP[name.lower()] = name
    IN_STATE_LOOKUP[abbr.lower()] = name

_in_sorted_keys = sorted(IN_STATE_LOOKUP.keys(), key=len, reverse=True)
IN_STATE_COMBINED_REGEX = re.compile(r"\b(" + "|".join(re.escape(k) for k in _in_sorted_keys) + r")\b", re.IGNORECASE)

TOP_INDIAN_CITIES = [
    "mumbai", "delhi", "bengaluru", "bangalore", "hyderabad", "ahmedabad", "chennai",
    "kolkata", "pune", "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore",
    "thane", "bhopal", "visakhapatnam", "patna", "vadodara", "ghaziabad", "ludhiana",
    "agra", "nashik", "ranchi", "faridabad", "meerut", "rajkot", "varanasi", "srinagar",
    "aurangabad", "dhanbad", "amritsar", "navi mumbai", "allahabad", "prayagraj",
    "howrah", "gwalior", "jabalpur", "coimbatore", "vijayawada", "jodhpur", "madurai",
    "raipur", "kota", "guwahati", "chandigarh", "solapur", "hubli", "dharwad",
    "bareilly", "moradabad", "mysore", "mysuru", "gurgaon", "gurugram", "aligarh",
    "jalandhar", "tiruchirappalli", "bhubaneswar", "salem", "warangal", "mira bhayandar",
    "thiruvananthapuram", "bhiwandi", "saharanpur", "guntur", "amravati", "bikaner",
    "noida", "jamshedpur", "bhilai", "cuttack", "firozabad", "kochi", "nellore",
    "bhavnagar", "dehradun", "durgapur", "asansol", "rourkela", "nanded", "kolhapur",
    "ajmer", "akola", "gulbarga", "jamnagar", "ujjain", "loni", "siliguri", "jhansi",
    "ulhasnagar", "jammu", "sangli", "mangalore", "erode", "belgaum", "ambattur",
    "tirunelveli", "malegaon", "gaya", "jalgaon", "udaipur", "maheshtala", "kozhikode",
    "panchkula", "khordha", "palghar", "gorakhpur", "sahajanwa"
]
IN_CITY_REGEX = re.compile(rf"\b({'|'.join(TOP_INDIAN_CITIES)})\b", re.IGNORECASE)

ADDRESS_STOPWORDS = {
    "street", "avenue", "boulevard", "colony", "complex", "building", "district",
    "railway", "station", "sector", "karnataka", "maharashtra", "delhi", "telangana",
    "bengal", "gujarat", "pradesh", "floor", "house", "nagar", "road", "near",
    "floorflat", "estate", "limited", "private", "opphotel", "opprailway", "block",
    "rd", "st", "lane", "ln", "dr", "drive", "court", "ct", "circle", "cir", "suite",
    "ste", "unit", "apt", "apartment", "apartments", "bldg", "hno", "opp", "opposite",
    "behind", "beside", "above", "cross", "main", "phase", "layout", "enclave",
    "bazaar", "market", "tower", "towers", "park", "plaza", "centre", "center",
    "industrial", "area", "dist", "taluk", "mandal", "post", "village", "city", "state",
    "india", "pvt", "ltd", "corp", "llc", "null", "none",
    "east", "west", "north", "south", "central", "middle", "old", "new",
    "ground", "first", "second", "third", "front", "back", "side", "upper", "lower",
    "room", "rooms", "hall", "point", "view", "garden", "gardens", "residency", "heights",
    "square", "gate", "bridge", "bypass", "national", "highway", "expressway",
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
    "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur", "nagpur", "indore",
    "thane", "bhopal", "patna", "vadodara", "noida", "gurgaon", "gurugram",
}


def canonicalize_state(state: Optional[str], country: str = "US") -> str:
    """Normalize state representation across full names and abbreviations."""
    if not state or pd.isna(state) or str(state).lower() in ("none", "nan", "<null>", ""):
        return ""
    st = str(state).strip().upper().replace(".", "")
    if country == "India":
        return INDIAN_STATE_ABBRS.get(st, st)
    return US_STATE_CODES.get(st, st)


COMMON_HONORIFIC_PREFIXES = re.compile(
    r"^(?:m\s*/\s*s\.?|shri|shree|sri|smt|dr\.?|mr\.?|ms\.?|private|pvt\.?|limited|ltd\.?|the)\b[\s\.\-\/\:]*",
    re.IGNORECASE,
)


def strip_honorific_prefixes(text: str) -> str:
    """Recursively strip leading honorific and legal prefix tokens."""
    s = text.strip()
    while True:
        sub = COMMON_HONORIFIC_PREFIXES.sub("", s).strip()
        if sub == s or not sub:
            break
        s = sub
    return s


def _generate_token_keys_for_string(text: str, prefix_tag: str = "", min_len: int = 2) -> List[str]:
    raw_str = text.lower()
    de_domain = DOMAIN_REGEX.sub(" ", raw_str)
    raw_words = [w for w in PUNCT_SPLIT.split(de_domain) if len(w) >= min_len]
    if not raw_words:
        return []

    content_words = [w for w in raw_words if w not in COMMON_STOP_WORDS]
    target_words = content_words if content_words else raw_words

    keys = []
    # Index all content words >= 3 chars
    for w in target_words:
        if len(w) >= 3:
            keys.append(f"{prefix_tag}TOK:{w}")

    # 1. Sorted first two tokens (order-invariant)
    sorted_words = sorted(target_words)
    if len(sorted_words) >= 2:
        w0, w1 = sorted_words[0], sorted_words[1]
        keys.append(f"{prefix_tag}TOK2:{w0}_{w1}")
        if len(w0) >= 3 and len(w1) >= 3:
            keys.append(f"{prefix_tag}TOK2_PRE3:{w0[:3]}_{w1[:3]}")
    keys.append(f"{prefix_tag}TOK1:{sorted_words[0]}")

    # 2. Acronym of content tokens
    if len(target_words) >= 2:
        acro = "".join(w[0] for w in target_words[:4])
        if len(acro) >= 2:
            keys.append(f"{prefix_tag}ACRO:{acro}")

    # 3. Longest distinctive token (if length >= 4)
    longest = max(target_words, key=len)
    if len(longest) >= 5:
        keys.append(f"{prefix_tag}LONG:{longest}")
        for i in range(min(3, len(longest) - 3)):
            keys.append(f"{prefix_tag}G4:{longest[i:i+4]}")
    if len(longest) >= 4:
        for i in range(min(4, len(longest) - 2)):
            keys.append(f"{prefix_tag}G3:{longest[i:i+3]}")

    # 4. First 6 characters prefix of full string
    clean_str = "".join(raw_words)
    if len(clean_str) >= 6:
        keys.append(f"{prefix_tag}PRE6:{clean_str[:6]}")
    clean_target = "".join(target_words)
    if len(clean_target) >= 5:
        keys.append(f"{prefix_tag}PRE5:{clean_target[:5]}")

    return keys


def extract_token_blocking_keys(name: Optional[str], min_len: int = 2) -> List[str]:
    """
    Generate exact and fuzzy token-overlap blocking keys with:
      - Domain stripping
      - Transliteration-normalization for non-Latin / Indic scripts
      - Prefix-stripped keys for honorific/legal variance
    """
    if not name:
        return []
    raw_str = str(name).strip()
    if not raw_str:
        return []

    keys = []
    # 1. Base keys on original string
    keys.extend(_generate_token_keys_for_string(raw_str, prefix_tag="", min_len=min_len))

    # 2. Transliteration-normalized keys if non-ascii (Devanagari, Telugu, Gujarati, Gurmukhi, etc.)
    if not raw_str.isascii():
        translit_str = unidecode.unidecode(raw_str)
        if translit_str and translit_str != raw_str:
            keys.extend(_generate_token_keys_for_string(translit_str, prefix_tag="", min_len=min_len))
            keys.extend(_generate_token_keys_for_string(translit_str, prefix_tag="XLAT_", min_len=min_len))

    # 3. Prefix-stripped blocking keys (strips Sri, Shree, Shri, Private, Pvt, Ltd, M/s, etc.)
    stripped = strip_honorific_prefixes(raw_str)
    if stripped and stripped.lower() != raw_str.lower():
        keys.extend(_generate_token_keys_for_string(stripped, prefix_tag="PFX_", min_len=min_len))

    if not raw_str.isascii():
        stripped_translit = strip_honorific_prefixes(unidecode.unidecode(raw_str))
        if stripped_translit:
            keys.extend(_generate_token_keys_for_string(stripped_translit, prefix_tag="PFX_", min_len=min_len))

    return list(dict.fromkeys(keys))


def _generate_phonetic_keys_for_string(text: str, prefix_tag: str = "", min_len: int = 2) -> List[str]:
    raw_str = text.lower()
    de_domain = DOMAIN_REGEX.sub(" ", raw_str)
    words = [w for w in PUNCT_SPLIT.split(de_domain) if len(w) >= min_len and w.isalpha()]
    if not words:
        return []

    content_words = [w for w in words if w not in COMMON_STOP_WORDS]
    target_words = content_words if content_words else words

    keys = []
    # 1. Metaphone of first word
    meta0 = jellyfish.metaphone(target_words[0])
    if meta0 and len(meta0) >= min_len:
        keys.append(f"{prefix_tag}META1:{meta0}")

    # 2. Metaphone of first two words
    if len(target_words) >= 2:
        meta1 = jellyfish.metaphone(target_words[1])
        if meta1 and len(meta1) >= 2:
            sorted_meta = sorted([meta0, meta1])
            keys.append(f"{prefix_tag}META2:{sorted_meta[0]}_{sorted_meta[1]}")

    # 3. Soundex of first word
    snd0 = jellyfish.soundex(target_words[0])
    if snd0:
        keys.append(f"{prefix_tag}SND1:{snd0}")

    return keys


def extract_phonetic_blocking_keys(name: Optional[str], min_len: int = 2) -> List[str]:
    """
    Generate phonetic (Metaphone and Soundex) blocking keys with:
      - Transliteration normalization for non-Latin / Indic scripts
      - Prefix-stripped phonetic keys for honorific/legal variance
    """
    if not name:
        return []
    raw_str = str(name).strip()
    if not raw_str:
        return []

    keys = []
    # 1. Base phonetic keys
    keys.extend(_generate_phonetic_keys_for_string(raw_str, prefix_tag="", min_len=min_len))

    # 2. Transliteration phonetic keys (enables phonetic matching for Devanagari, Telugu, Gujarati, Gurmukhi, etc.)
    if not raw_str.isascii():
        translit_str = unidecode.unidecode(raw_str)
        if translit_str:
            keys.extend(_generate_phonetic_keys_for_string(translit_str, prefix_tag="", min_len=min_len))

    # 3. Prefix-stripped phonetic keys
    stripped = strip_honorific_prefixes(raw_str)
    if stripped and stripped.lower() != raw_str.lower():
        keys.extend(_generate_phonetic_keys_for_string(stripped, prefix_tag="PFX_", min_len=min_len))

    if not raw_str.isascii():
        stripped_translit = strip_honorific_prefixes(unidecode.unidecode(raw_str))
        if stripped_translit:
            keys.extend(_generate_phonetic_keys_for_string(stripped_translit, prefix_tag="PFX_", min_len=min_len))

    return list(dict.fromkeys(keys))



def extract_address_blocking_keys(
    name: Optional[str],
    postal_code: Optional[str],
    city: Optional[str],
    state: Optional[str],
    street_number: Optional[str] = None,
    street_name: Optional[str] = None,
    raw_address: Optional[str] = None,
    country: str = "US",
) -> List[str]:
    # 0. Transliteration for non-ASCII / Indic address components
    if raw_address and not str(raw_address).isascii():
        raw_address = f"{raw_address} {unidecode.unidecode(str(raw_address))}"
    if city and not str(city).isascii():
        city = unidecode.unidecode(str(city))
    if state and not str(state).isascii():
        state = unidecode.unidecode(str(state))
    clean_name = str(name or "").lower().strip()
    if clean_name and not clean_name.isascii():
        clean_name = unidecode.unidecode(clean_name)

    prefix = clean_name[:2] if len(clean_name) >= 2 else "xx"
    clean_city = str(city).lower().strip() if city and pd.notna(city) and str(city).lower() not in ("none", "nan", "<null>") else ""
    raw_addr_str = str(raw_address or "").lower()

    # 1. State resolution
    clean_state = canonicalize_state(state, country)

    # Check if city actually contains a state name (frequent in Indian data)
    if not clean_state and clean_city:
        cand_state = canonicalize_state(clean_city, country)
        if cand_state and cand_state != clean_city.upper():
            clean_state = cand_state
            clean_city = ""
    elif clean_city:
        cand_state = canonicalize_state(clean_city, country)
        if cand_state and cand_state == clean_state:
            clean_city = ""

    # Fallback state extraction from raw address
    if not clean_state and raw_addr_str:
        if country == "India":
            m_st = IN_STATE_COMBINED_REGEX.search(raw_addr_str)
            if m_st:
                clean_state = IN_STATE_LOOKUP.get(m_st.group(1).lower(), "")
        elif country == "US":
            m_st = US_STATE_COMBINED_REGEX.search(raw_addr_str)
            if m_st:
                clean_state = US_STATE_LOOKUP.get(m_st.group(1).lower(), "")

    # Fallback city extraction from raw address
    if not clean_city and raw_addr_str and country == "India":
        m_city = IN_CITY_REGEX.search(raw_addr_str)
        if m_city:
            clean_city = m_city.group(1).lower()

    INDIAN_CITY_ALIASES = {
        "bombay": "mumbai", "bangalore": "bengaluru", "calcutta": "kolkata",
        "madras": "chennai", "gurgaon": "gurugram", "allahabad": "prayagraj",
        "baroda": "vadodara", "trivandrum": "thiruvananthapuram", "cochin": "kochi",
        "poona": "pune", "pondicherry": "puducherry", "mysore": "mysuru",
    }
    if country == "India" and clean_city in INDIAN_CITY_ALIASES:
        clean_city = INDIAN_CITY_ALIASES[clean_city]

    # State aliasing for historical reorganizations (Telangana / Andhra Pradesh)
    states_to_index = [clean_state] if clean_state else []
    if country == "India":
        if clean_state == "TELANGANA":
            states_to_index.append("ANDHRA PRADESH")
        elif clean_state == "ANDHRA PRADESH":
            states_to_index.append("TELANGANA")

    keys = []
    # 1. Postal code keys
    clean_post = str(postal_code).strip() if postal_code and pd.notna(postal_code) else ""
    if clean_post and clean_post.lower() not in ("none", "nan", ""):
        keys.append(f"POST:{clean_post}_{prefix}")
        for st in states_to_index:
            keys.append(f"POST_STATE:{st}_{clean_post}")
        if country == "India" and INDIAN_PIN_REGEX.match(clean_post):
            keys.append(f"IN_PIN:{clean_post}")
            if clean_city:
                keys.append(f"IN_PIN_CITY:{clean_post}_{clean_city}")

    # 2. City + State + name prefix
    raw_state = str(state).upper().strip() if state and pd.notna(state) else ""
    if clean_city and clean_state:
        keys.append(f"GEO:{clean_state}_{clean_city}_{prefix}")
        if raw_state and raw_state != clean_state:
            keys.append(f"GEO:{raw_state}_{clean_city}_{prefix}")

    # 3. Geographically conditioned tokens (GEO_TOK)
    words = [w for w in clean_name.split() if w not in COMMON_STOP_WORDS and len(w) >= 3]
    if clean_state and clean_city:
        for tok in words[:3]:
            keys.append(f"GEO_TOK:{clean_state}_{clean_city}_{tok}")

    # 4. Street / Plot number + State (using robust digit extraction)
    num_str = ""
    if street_number and pd.notna(street_number):
        m_dig = re.search(r"\d+", str(street_number))
        if m_dig:
            num_str = m_dig.group(0)

    s_tok = ""
    if street_name and pd.notna(street_name):
        s_toks = [w for w in PUNCT_SPLIT.split(str(street_name).lower()) if len(w) >= 3 and w not in ("street", "road", "drive", "avenue", "lane", "boulevard", "blvd", "rd", "st", "ave", "dr", "way", "court", "ct")]
        if s_toks:
            s_tok = s_toks[0]

    if num_str and len(num_str) >= 2 and clean_state:
        for st in states_to_index:
            keys.append(f"NUM_ST:{st}_{num_str}")
        if clean_city:
            keys.append(f"ST_NUM:{clean_state}_{clean_city}_{num_str}")
        if s_tok and clean_city:
            keys.append(f"STREET_ADDR:{clean_state}_{clean_city}_{num_str}_{s_tok}")

    # Pure distinctive street name + city (when street number is missing or divergent)
    if s_tok and clean_state and clean_city and len(s_tok) >= 4:
        keys.append(f"STREET_NAME:{clean_state}_{clean_city}_{s_tok}")

    # 5. Indian-Specific Script-Agnostic Address Keys
    if country == "India" and raw_addr_str:
        if not clean_post:
            pins = INDIAN_PIN_REGEX.findall(raw_addr_str)
            if pins:
                keys.append(f"IN_PIN:{pins[0]}")
                if clean_city:
                    keys.append(f"IN_PIN_CITY:{pins[0]}_{clean_city}")

        plots = PLOT_DOOR_REGEX.findall(raw_addr_str)
        for p in plots[:3]:
            m_p = re.search(r"\d+", p)
            if m_p:
                clean_p = str(int(m_p.group(0)))
                if clean_p and clean_city:
                    keys.append(f"ADDR_PLOT_CITY:{clean_city}_{clean_p}")
                if len(clean_p) >= 2:
                    for st in states_to_index:
                        keys.append(f"ADDR_PLOT_STATE:{st}_{clean_p}")

        # All distinctive numbers (door, flat, shop, sector) >= 2 digits
        all_nums = [m.group(0) for m in re.finditer(r"\b\d{2,5}\b", raw_addr_str)]
        for dig in all_nums[:4]:
            if clean_city:
                keys.append(f"ADDR_NUM_CITY:{clean_city}_{dig}")

        # ADDR_PLOT_PIN
        all_pins = INDIAN_PIN_REGEX.findall(raw_addr_str)
        if clean_post and INDIAN_PIN_REGEX.match(clean_post):
            all_pins.append(clean_post)
        if all_pins and plots:
            m_p = re.search(r"\d+", plots[0])
            if m_p and len(m_p.group(0)) >= 2:
                keys.append(f"ADDR_PLOT_PIN:{all_pins[0]}_{m_p.group(0)}")

        # Distinctive address locality words
        addr_words = [w for w in PUNCT_SPLIT.split(raw_addr_str) if len(w) >= 5 and w not in ADDRESS_STOPWORDS]
        for aw in addr_words[:5]:
            if clean_city and len(clean_city) >= 3:
                keys.append(f"ADDR_WORD_CITY:{clean_city}_{aw}")
            for st in states_to_index:
                keys.append(f"ADDR_WORD_STATE:{st}_{aw}")
        if len(addr_words) >= 2:
            s_pair = sorted(addr_words[:4])
            for st in states_to_index:
                keys.append(f"ADDR_PAIR_STATE:{st}_{s_pair[0]}_{s_pair[1]}")

    return keys


# ---------------------------------------------------------------------------
# Multi-Strategy Blocker Class
# ---------------------------------------------------------------------------

class MultiStrategyBlocker:
    """
    Implements multi-strategy candidate generation combining TF-IDF cosine k-NN,
    exact/fuzzy token overlap, phonetic encoding, and address blocking.
    """

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()
        # Per-country index structures
        self.country_indices: Dict[str, Dict[str, Any]] = {}
        self.is_fitted = False

    def fit(self, corpus_df: pd.DataFrame) -> "MultiStrategyBlocker":
        """
        Build indexing structures over the candidate corpus (Source 2 and Source 3).

        Args:
            corpus_df: DataFrame with columns:
                ['entity_id', 'country', 'norm_name', 'norm_name_no_legal',
                 'addr_postal_code', 'addr_city', 'addr_state']
        """
        print(f"Fitting MultiStrategyBlocker on {len(corpus_df):,} corpus entities...")
        countries = corpus_df["country"].unique() if self.config.country_partition else ["ALL"]

        for country in countries:
            print(f"  Indexing partition: {country}...")
            if self.config.country_partition:
                sub_df = corpus_df[corpus_df["country"] == country].reset_index(drop=True)
            else:
                sub_df = corpus_df.reset_index(drop=True)

            if len(sub_df) == 0:
                continue

            entity_ids = sub_df["entity_id"].values
            partition_data: Dict[str, Any] = {"entity_ids": entity_ids, "n_records": len(sub_df)}

            # (a) TF-IDF Model & Index
            if self.config.enable_tfidf:
                min_df_val = min(2, len(sub_df))
                vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=self.config.tfidf_ngram_range,
                    max_features=self.config.tfidf_max_features,
                    min_df=min_df_val,
                )
                text_series = sub_df["norm_name"].fillna("").astype(str)
                text_series = text_series.apply(lambda s: f"{s} {unidecode.unidecode(s)}" if not s.isascii() else s)
                X_corpus = vec.fit_transform(text_series)
                partition_data["tfidf_vec"] = vec

                if self.config.knn_engine == "faiss":
                    # Dense TruncatedSVD + FAISS IndexFlatIP
                    if X_corpus.shape[1] <= self.config.faiss_dim:
                        X_dense = X_corpus.toarray().astype("float32")
                        faiss.normalize_L2(X_dense)
                        faiss_index = faiss.IndexFlatIP(X_dense.shape[1])
                        faiss_index.add(X_dense)
                        partition_data["faiss_svd"] = None
                        partition_data["faiss_index"] = faiss_index
                    else:
                        n_comp = min(self.config.faiss_dim, X_corpus.shape[1] - 1)
                        svd = TruncatedSVD(n_components=n_comp, random_state=42)
                        X_dense = svd.fit_transform(X_corpus).astype("float32")
                        faiss.normalize_L2(X_dense)
                        faiss_index = faiss.IndexFlatIP(X_dense.shape[1])
                        faiss_index.add(X_dense)
                        partition_data["faiss_svd"] = svd
                        partition_data["faiss_index"] = faiss_index
                else:
                    # Sparse sklearn NearestNeighbors
                    k_neighbors = min(self.config.top_k_tfidf, len(sub_df))
                    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine", algorithm="brute")
                    nn.fit(X_corpus)
                    partition_data["sklearn_nn"] = nn

            # Combined single-pass inverted index generation
            token_inv_index = defaultdict(list) if self.config.enable_token_overlap else None
            phone_inv_index = defaultdict(list) if self.config.enable_phonetic else None
            addr_inv_index = defaultdict(list) if self.config.enable_address else None

            names = sub_df["norm_name_no_legal"].fillna("").astype(str).tolist()
            posts = sub_df["addr_postal_code"].tolist() if "addr_postal_code" in sub_df.columns else [None] * len(sub_df)
            cities = sub_df["addr_city"].tolist() if "addr_city" in sub_df.columns else [None] * len(sub_df)
            states = sub_df["addr_state"].tolist() if "addr_state" in sub_df.columns else [None] * len(sub_df)
            street_nums = sub_df["addr_street_number"].tolist() if "addr_street_number" in sub_df.columns else [None] * len(sub_df)
            street_names = sub_df["addr_street_name"].tolist() if "addr_street_name" in sub_df.columns else [None] * len(sub_df)
            raw_addrs = sub_df["business_address"].tolist() if "business_address" in sub_df.columns else [None] * len(sub_df)
            country_str = str(country)

            do_tok = self.config.enable_token_overlap
            do_phn = self.config.enable_phonetic
            do_adr = self.config.enable_address
            min_t = self.config.min_token_len
            min_p = self.config.min_phonetic_len

            for idx in range(len(entity_ids)):
                eid = entity_ids[idx]
                nm = names[idx]

                if do_tok:
                    for key in extract_token_blocking_keys(nm, min_len=min_t):
                        token_inv_index[key].append(eid)

                if do_phn:
                    for key in extract_phonetic_blocking_keys(nm, min_len=min_p):
                        phone_inv_index[key].append(eid)

                if do_adr:
                    keys = extract_address_blocking_keys(
                        name=nm,
                        postal_code=posts[idx],
                        city=cities[idx],
                        state=states[idx],
                        street_number=street_nums[idx],
                        street_name=street_names[idx],
                        raw_address=raw_addrs[idx],
                        country=country_str,
                    )
                    for key in keys:
                        addr_inv_index[key].append(eid)

            if do_tok:
                partition_data["token_index"] = {
                    k: v for k, v in token_inv_index.items()
                    if len(v) <= self.config.max_block_size
                }
            if do_phn:
                partition_data["phone_index"] = {
                    k: v for k, v in phone_inv_index.items()
                    if len(v) <= self.config.max_block_size
                }
            if do_adr:
                partition_data["addr_index"] = {
                    k: v for k, v in addr_inv_index.items()
                    if len(v) <= self.config.max_block_size
                }

            self.country_indices[str(country)] = partition_data

        self.is_fitted = True
        print("Blocker fitting complete!")
        return self

    def block_queries(
        self,
        query_df: pd.DataFrame,
        batch_size: int = 5000,
    ) -> Dict[str, List[str]]:
        """
        Generate candidate pairs for all query entities in query_df.

        Args:
            query_df: DataFrame of Source 1 entities with columns:
                ['entity_id', 'country', 'norm_name', 'norm_name_no_legal',
                 'addr_postal_code', 'addr_city', 'addr_state']

        Returns:
            Dictionary mapping source1_entity_id -> list of candidate entity_ids.
        """
        if not self.is_fitted:
            raise RuntimeError("MultiStrategyBlocker must be fitted before calling block_queries.")

        candidates_by_query: Dict[str, Dict[str, int]] = {q_id: defaultdict(int) for q_id in query_df["entity_id"]}

        countries = query_df["country"].unique() if self.config.country_partition else ["ALL"]

        for country in countries:
            country_key = str(country) if self.config.country_partition else "ALL"
            if country_key not in self.country_indices:
                continue

            part = self.country_indices[country_key]
            corpus_entity_ids = part["entity_ids"]

            if self.config.country_partition:
                sub_query = query_df[query_df["country"] == country].reset_index(drop=True)
            else:
                sub_query = query_df.reset_index(drop=True)

            if len(sub_query) == 0:
                continue

            query_ids = sub_query["entity_id"].values

            # --- Strategy (a): TF-IDF k-NN ---
            if self.config.enable_tfidf and "tfidf_vec" in part:
                vec = part["tfidf_vec"]
                text_series = sub_query["norm_name"].fillna("").astype(str)
                text_series = text_series.apply(lambda s: f"{s} {unidecode.unidecode(s)}" if not s.isascii() else s)

                # Process in batches to limit peak memory
                for i in range(0, len(sub_query), batch_size):
                    batch_texts = text_series.iloc[i : i + batch_size]
                    batch_q_ids = query_ids[i : i + batch_size]
                    X_q = vec.transform(batch_texts)

                    if self.config.knn_engine == "faiss" and "faiss_index" in part:
                        svd = part["faiss_svd"]
                        if svd is not None:
                            X_q_dense = svd.transform(X_q).astype("float32")
                        else:
                            X_q_dense = X_q.toarray().astype("float32")
                        faiss.normalize_L2(X_q_dense)
                        k = min(self.config.top_k_tfidf, part["n_records"])
                        dists, indices = part["faiss_index"].search(X_q_dense, k)
                        for q_local_idx, q_id in enumerate(batch_q_ids):
                            for d, idx in zip(dists[q_local_idx], indices[q_local_idx]):
                                if idx >= 0 and d >= self.config.min_tfidf_sim:
                                    candidates_by_query[q_id][corpus_entity_ids[idx]] += 3
                    elif "sklearn_nn" in part:
                        nn = part["sklearn_nn"]
                        dists, indices = nn.kneighbors(X_q)
                        for q_local_idx, q_id in enumerate(batch_q_ids):
                            for d, idx in zip(dists[q_local_idx], indices[q_local_idx]):
                                sim = 1.0 - d
                                if sim >= self.config.min_tfidf_sim:
                                    candidates_by_query[q_id][corpus_entity_ids[idx]] += 3

            # --- Strategy (b): Token Overlap Inverted Index ---
            if self.config.enable_token_overlap and "token_index" in part:
                t_idx = part["token_index"]
                for q_id, name in zip(query_ids, sub_query["norm_name_no_legal"]):
                    for key in extract_token_blocking_keys(name, min_len=self.config.min_token_len):
                        if key in t_idx:
                            weight = 4 if key.startswith("TOK2") else (3 if key.startswith(("TOK1", "LONG", "TOK")) else 1)
                            for c_id in t_idx[key]:
                                candidates_by_query[q_id][c_id] += weight

            # --- Strategy (c): Phonetic Inverted Index ---
            if self.config.enable_phonetic and "phone_index" in part:
                p_idx = part["phone_index"]
                for q_id, name in zip(query_ids, sub_query["norm_name_no_legal"]):
                    for key in extract_phonetic_blocking_keys(name, min_len=self.config.min_phonetic_len):
                        if key in p_idx:
                            weight = 2 if key.startswith("META2") else 1
                            for c_id in p_idx[key]:
                                candidates_by_query[q_id][c_id] += weight

            # --- Strategy (d): Address Inverted Index ---
            if self.config.enable_address and "addr_index" in part:
                a_idx = part["addr_index"]
                names = sub_query["norm_name_no_legal"].tolist()
                posts = sub_query["addr_postal_code"].tolist()
                cities = sub_query["addr_city"].tolist()
                states = sub_query["addr_state"].tolist()
                street_nums = sub_query["addr_street_number"].tolist() if "addr_street_number" in sub_query.columns else [None] * len(sub_query)
                street_names = sub_query["addr_street_name"].tolist() if "addr_street_name" in sub_query.columns else [None] * len(sub_query)
                raw_addrs = sub_query["business_address"].tolist() if "business_address" in sub_query.columns else [None] * len(sub_query)
                country_str = str(country)
                for q_id, name, post, city, state, s_num, s_name, r_addr in zip(query_ids, names, posts, cities, states, street_nums, street_names, raw_addrs):
                    keys = extract_address_blocking_keys(
                        name=name,
                        postal_code=post,
                        city=city,
                        state=state,
                        street_number=s_num,
                        street_name=s_name,
                        raw_address=r_addr,
                        country=country_str,
                    )
                    for key in keys:
                        if key in a_idx:
                            weight = 8 if key.startswith("ADDR_PLOT_PIN") else (
                                7 if key.startswith(("IN_PIN", "IN_PIN_CITY")) else (
                                    6 if key.startswith(("STREET_ADDR", "STREET_NAME", "ADDR_PLOT_CITY", "ADDR_PLOT_STATE")) else (
                                        5 if key.startswith(("ADDR_WORD_CITY", "ADDR_WORD_STATE", "ADDR_PAIR_STATE", "ADDR_NUM_CITY")) else (
                                            3 if key.startswith(("POST", "NUM_ST", "ST_NUM", "GEO_TOK")) else 2
                                        )
                                    )
                                )
                            )
                            for c_id in a_idx[key]:
                                candidates_by_query[q_id][c_id] += weight

        # Enforce maximum candidate cap per entity with priority ranking
        result: Dict[str, List[str]] = {}
        for q_id, cand_scores in candidates_by_query.items():
            if len(cand_scores) > self.config.max_candidates_per_query:
                # Rank candidates by match score descending
                sorted_cands = sorted(cand_scores.keys(), key=lambda c: cand_scores[c], reverse=True)
                result[q_id] = sorted_cands[: self.config.max_candidates_per_query]
            else:
                result[q_id] = list(cand_scores.keys())

        return result

    @staticmethod
    def to_candidate_pairs_tsv(
        candidates_dict: Dict[str, List[str]],
        output_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Persist candidate pairs to candidate_pairs.tsv in exact competition format.
        Header: source1_entity_id\tcandidate_entity_ids
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {"source1_entity_id": q_id, "candidate_entity_ids": ",".join(c_list)}
            for q_id, c_list in candidates_dict.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved candidate_pairs to {output_path} ({len(df):,} rows)")
        return df

    @staticmethod
    def to_pairs_long_df(candidates_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Convert candidate dictionary into pairwise long DataFrame for ML scoring.
        Columns: source1_entity_id, candidate_entity_id
        """
        pairs = []
        for q_id, c_list in candidates_dict.items():
            for c_id in c_list:
                pairs.append((q_id, c_id))
        return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])

    @staticmethod
    def evaluate_blocking_quality(
        candidates_dict: Dict[str, List[str]],
        ground_truth_df: pd.DataFrame,
        total_corpus_entities: int,
    ) -> Dict[str, float]:
        """
        Evaluate candidate recall ceiling, average candidates per entity,
        and reduction ratio against ground truth labels.
        """
        # Build mapping of true matches
        true_matches: Dict[str, Set[str]] = {}
        for _, row in ground_truth_df.iterrows():
            s1_id = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""
            if m_str:
                true_matches[s1_id] = set(x.strip() for x in m_str.split(",") if x.strip())
            else:
                true_matches[s1_id] = set()

        total_eval_entities = 0
        total_matched_entities = 0
        recalled_entities_any = 0
        total_true_pairs = 0
        total_recalled_pairs = 0
        total_candidate_pairs = 0

        for q_id, c_list in candidates_dict.items():
            c_set = set(c_list)
            total_candidate_pairs += len(c_set)
            total_eval_entities += 1

            if q_id in true_matches:
                t_set = true_matches[q_id]
                if t_set:
                    total_matched_entities += 1
                    total_true_pairs += len(t_set)
                    hits = len(t_set.intersection(c_set))
                    total_recalled_pairs += hits
                    if hits > 0:
                        recalled_entities_any += 1

        pair_recall = total_recalled_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
        entity_recall = recalled_entities_any / total_matched_entities if total_matched_entities > 0 else 0.0
        avg_candidates = total_candidate_pairs / total_eval_entities if total_eval_entities > 0 else 0.0
        total_possible = total_eval_entities * total_corpus_entities
        reduction_ratio = 1.0 - (total_candidate_pairs / total_possible) if total_possible > 0 else 1.0

        return {
            "pair_recall": round(pair_recall * 100, 2),
            "entity_recall": round(entity_recall * 100, 2),
            "avg_candidates_per_entity": round(avg_candidates, 1),
            "total_candidate_pairs": total_candidate_pairs,
            "reduction_ratio": round(reduction_ratio * 100, 4),
        }
