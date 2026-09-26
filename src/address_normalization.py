"""
Address Normalization Module for Amazon ML Challenge 2026.

Standardizes abbreviations, extracts landmark phrases into a separate field,
and extracts structural address components (street number, street name, unit,
city, state/region, postal code) without external geocoding services.
Handles US and India specifically, while gracefully degrading to generic
fallback tokenization for unseen countries (e.g., France).
"""

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NormalizedAddress:
    """Structured representation of a normalized address."""
    raw_address: str
    country: str
    cleaned_address: str = ""
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    unit: Optional[str] = None
    landmark: Optional[str] = None
    city: Optional[str] = None
    state_or_region: Optional[str] = None
    postal_code: Optional[str] = None
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    def as_feature_dict(self) -> Dict[str, str]:
        """Convert to dictionary suitable for tabular feature extraction."""
        return {
            "cleaned_address": self.cleaned_address or "",
            "street_number": self.street_number or "",
            "street_name": self.street_name or "",
            "unit": self.unit or "",
            "landmark": self.landmark or "",
            "city": self.city or "",
            "state_or_region": self.state_or_region or "",
            "postal_code": self.postal_code or "",
        }


# ---------------------------------------------------------------------------
# Lexicons & Regular Expressions
# ---------------------------------------------------------------------------

# US States mapping: 2-letter codes and lower-case full names to standard code
US_STATES = {
    "AL": "AL", "AK": "AK", "AZ": "AZ", "AR": "AR", "CA": "CA", "CO": "CO",
    "CT": "CT", "DE": "DE", "FL": "FL", "GA": "GA", "HI": "HI", "ID": "ID",
    "IL": "IL", "IN": "IN", "IA": "IA", "KS": "KS", "KY": "KY", "LA": "LA",
    "ME": "ME", "MD": "MD", "MA": "MA", "MI": "MI", "MN": "MN", "MS": "MS",
    "MO": "MO", "MT": "MT", "NE": "NE", "NV": "NV", "NH": "NH", "NJ": "NJ",
    "NM": "NM", "NY": "NY", "NC": "NC", "ND": "ND", "OH": "OH", "OK": "OK",
    "OR": "OR", "PA": "PA", "RI": "RI", "SC": "SC", "SD": "SD", "TN": "TN",
    "TX": "TX", "UT": "UT", "VT": "VT", "VA": "VA", "WA": "WA", "WV": "WV",
    "WI": "WI", "WY": "WY", "DC": "DC", "PR": "PR",
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "puerto rico": "PR",
}

# Indian States and UTs (canonical names and codes)
INDIA_STATES = {
    "andhra pradesh": "Andhra Pradesh", "ap": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh", "ar": "Arunachal Pradesh",
    "assam": "Assam", "as": "Assam",
    "bihar": "Bihar", "br": "Bihar",
    "chhattisgarh": "Chhattisgarh", "cg": "Chhattisgarh",
    "goa": "Goa", "ga": "Goa",
    "gujarat": "Gujarat", "gj": "Gujarat",
    "haryana": "Haryana", "hr": "Haryana",
    "himachal pradesh": "Himachal Pradesh", "hp": "Himachal Pradesh",
    "jharkhand": "Jharkhand", "jh": "Jharkhand",
    "karnataka": "Karnataka", "ka": "Karnataka",
    "kerala": "Kerala", "kl": "Kerala",
    "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "maharashtra": "Maharashtra", "mh": "Maharashtra",
    "manipur": "Manipur", "mn": "Manipur",
    "meghalaya": "Meghalaya", "ml": "Meghalaya",
    "mizoram": "Mizoram", "mz": "Mizoram",
    "nagaland": "Nagaland", "nl": "Nagaland",
    "odisha": "Odisha", "orissa": "Odisha", "od": "Odisha", "or": "Odisha",
    "punjab": "Punjab", "pb": "Punjab",
    "rajasthan": "Rajasthan", "rj": "Rajasthan",
    "sikkim": "Sikkim", "sk": "Sikkim",
    "tamil nadu": "Tamil Nadu", "tn": "Tamil Nadu",
    "telangana": "Telangana", "tg": "Telangana", "ts": "Telangana",
    "tripura": "Tripura", "tr": "Tripura",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand", "uttaranchal": "Uttarakhand", "uk": "Uttarakhand", "ua": "Uttarakhand",
    "west bengal": "West Bengal", "wb": "West Bengal",
    "delhi": "Delhi", "new delhi": "Delhi", "dl": "Delhi",
    "chandigarh": "Chandigarh", "ch": "Chandigarh",
    "puducherry": "Puducherry", "pondicherry": "Puducherry", "py": "Puducherry",
    "jammu and kashmir": "Jammu and Kashmir", "jk": "Jammu and Kashmir",
    "ladakh": "Ladakh", "la": "Ladakh",
}

# French Regions and common departments
FRANCE_REGIONS = [
    "nouvelle-aquitaine", "nouvelle aquitaine", "hauts-de-france", "hauts de france",
    "ile-de-france", "ile de france", "occitanie", "auvergne-rhone-alpes",
    "provence-alpes-cote d'azur", "paca", "grand est", "bretagne", "normandie",
    "pays de la loire", "bourgogne-franche-comte", "centre-val de loire", "corse",
    "gironde", "nord", "paris", "rhone", "bouches-du-rhone", "haute-garonne"
]

# Standard abbreviations mapping: lowercase regex token to expanded standard form
STREET_ABBREVIATIONS = {
    r"\brd\b\.?": "road",
    r"\bst\b\.?": "street",
    r"\bave\b\.?": "avenue",
    r"\bav\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bbvd\b\.?": "boulevard",
    r"\bdr\b\.?": "drive",
    r"\bct\b\.?": "court",
    r"\bln\b\.?": "lane",
    r"\bpkwy\b\.?": "parkway",
    r"\bhwy\b\.?": "highway",
    r"\bcir\b\.?": "circle",
    r"\bsq\b\.?": "square",
    r"\bpl\b\.?": "place",
    r"\bter\b\.?": "terrace",
    r"\bterr\b\.?": "terrace",
    r"\btrl\b\.?": "trail",
    r"\bexpy\b\.?": "expressway",
    r"\brte\b\.?": "route",
    r"\brt\b\.?": "route",
    # French
    r"\br\b\.?": "rue",
    r"\bbd\b\.?": "boulevard",
    r"\bch\b\.?": "chemin",
    r"\bchem\b\.?": "chemin",
    r"\ball\b\.?": "allee",
    r"\bimp\b\.?": "impasse",
}

UNIT_ABBREVIATIONS = {
    r"\bapt\b\.?": "apartment",
    r"\bste\b\.?": "suite",
    r"\bfl\b\.?": "floor",
    r"\bflr\b\.?": "floor",
    r"\bunt\b\.?": "unit",
    r"\bbldg\b\.?": "building",
    r"\brm\b\.?": "room",
    r"\bdept\b\.?": "department",
    r"\bbsmt\b\.?": "basement",
    r"\bp\.?\s*o\.?\s*box\b": "po box",
    r"\bpobox\b": "po box",
}

INDIAN_PREFIX_ABBREVIATIONS = {
    r"\bh\.?\s*no\b\.?": "house no",
    r"\bhn\b\.?": "house no",
    r"\bkh\.?\s*no\b\.?": "khasra no",
    r"\bdoor\s*no\b\.?": "door no",
    r"\bplot\s*no\b\.?": "plot no",
    r"\bshop\s*no\b\.?": "shop no",
    r"\bflat\s*no\b\.?": "flat no",
    r"\btq\b\.?": "taluk",
    r"\bdist\b\.?": "district",
}


# ---------------------------------------------------------------------------
# Core Normalization & Parsing Functions
# ---------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """Normalize unicode, strip extraneous punctuation and normalize whitespace."""
    if text is None:
        return ""
    text = str(text).strip()
    if text.lower() in ("nan", "none", "null"):
        return ""

    # Normalize unicode to decomposed NFKD form and strip unprintable characters
    text = unicodedata.normalize("NFKD", text)

    # Strip noise prefix symbols like '##', '--', '<<', '>>', '//'
    text = re.sub(r"^[\s\<\>\-\+\#\*\/\\]+", "", text)
    text = re.sub(r"[\s\<\>\-\+\#\*\/\\]+$", "", text)

    # Replace tabs and multiple spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# Precompiled regex patterns
PAREN_LANDMARK_RE = re.compile(
    r"\(\s*(?:next to|near|opp|opposite|behind|beside|adjacent to|adj to)\s+([^)]+)\)",
    re.IGNORECASE,
)
PREFIX_LANDMARK_RE = re.compile(
    r"\b(near|opposite|opp\.?|behind|beside|adjacent to|adj\.? to|next to|in front of|close to|across from)\s+([^\,\;\(\)]+)",
    re.IGNORECASE,
)
POSTFIX_LANDMARK_RE = re.compile(
    r"([^\,\;\(\)]+?)\s+(ke pas|ke pass|ke samne)\b",
    re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"\b(apartment|suite|floor|unit|room|po box|building)\s+([a-zA-Z0-9\-\/]+)\b",
    re.IGNORECASE,
)
ORDINAL_FLOOR_RE = re.compile(
    r"\b((?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+floor)\b",
    re.IGNORECASE,
)
US_ZIP_RE = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
INDIA_PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
GENERIC_POST_RE = re.compile(r"\b(\d{5})\b")
INDIA_DOOR_RE = re.compile(
    r"\b((?:door no|house no|flat no|plot no|shop no|khasra no|no)\s*[\-\.\:]?\s*[A-Za-z0-9\/\-]+|\d+[\-\/][A-Za-z0-9\-\/]+|\d+)\b",
    re.IGNORECASE,
)
STREET_NUM_RE = re.compile(r"^(\d+(?:\s+1\/2|\s*-\s*[A-Za-z0-9]+)?)\s+(.+)")
GENERIC_STREET_NUM_RE = re.compile(r"^(\d+(?:\s+bis|\s+ter)?)\s+(.+)", re.IGNORECASE)

COMPILED_UNIT_ABBREVIATIONS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in UNIT_ABBREVIATIONS.items()]
COMPILED_STREET_ABBREVIATIONS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in STREET_ABBREVIATIONS.items()]
COMPILED_INDIAN_PREFIX_ABBREVIATIONS = [(re.compile(p, re.IGNORECASE), repl) for p, repl in INDIAN_PREFIX_ABBREVIATIONS.items()]


def expand_abbreviations(text: str) -> str:
    """Expand road, unit, and local abbreviations into standardized forms."""
    result = text
    for pattern_re, replacement in COMPILED_UNIT_ABBREVIATIONS:
        result = pattern_re.sub(replacement, result)

    for pattern_re, replacement in COMPILED_STREET_ABBREVIATIONS:
        result = pattern_re.sub(replacement, result)

    for pattern_re, replacement in COMPILED_INDIAN_PREFIX_ABBREVIATIONS:
        result = pattern_re.sub(replacement, result)

    return result


def extract_landmarks(address: str) -> Tuple[str, Optional[str]]:
    """
    Extract landmark phrases (e.g., 'Near Fortis Hospital', 'Opp. RTA Office',
    'Next To Iter College', 'Beside Krupa Petrol Pump', 'Pani Ki Tanki Ke Pas')
    into a separate landmark string rather than discarding them.

    Returns:
        Tuple of (address_without_landmark, extracted_landmark)
    """
    landmarks_found: List[str] = []
    cleaned_address = address

    # 1. Parenthesized landmark phrases
    for m in PAREN_LANDMARK_RE.finditer(cleaned_address):
        landmarks_found.append(m.group(0).strip("()"))
    cleaned_address = PAREN_LANDMARK_RE.sub(" ", cleaned_address)

    # 2. English prefix landmark phrases
    for m in PREFIX_LANDMARK_RE.finditer(cleaned_address):
        landmarks_found.append(m.group(0).strip())
    cleaned_address = PREFIX_LANDMARK_RE.sub(" ", cleaned_address)

    # 3. Hindi/Indian postfix landmark phrases
    for m in POSTFIX_LANDMARK_RE.finditer(cleaned_address):
        landmarks_found.append(m.group(0).strip())
    cleaned_address = POSTFIX_LANDMARK_RE.sub(" ", cleaned_address)

    cleaned_address = re.sub(r",\s*,+", ",", cleaned_address)
    cleaned_address = re.sub(r"^[\s,]+|[\s,]+$", "", cleaned_address)
    cleaned_address = re.sub(r"\s+", " ", cleaned_address).strip()

    combined_landmark = "; ".join(landmarks_found) if landmarks_found else None
    return cleaned_address, combined_landmark


def extract_unit_info(address: str) -> Tuple[str, Optional[str]]:
    """
    Extract apartment, suite, floor, unit, or PO Box information.

    Returns:
        Tuple of (address_without_unit, extracted_unit)
    """
    match = UNIT_RE.search(address)
    if match:
        extracted = match.group(0).strip()
        rem_addr = UNIT_RE.sub(" ", address)
        rem_addr = re.sub(r",\s*,+", ",", rem_addr)
        rem_addr = re.sub(r"^[\s,]+|[\s,]+$", "", rem_addr).strip()
        return rem_addr, extracted

    m_floor = ORDINAL_FLOOR_RE.search(address)
    if m_floor:
        extracted = m_floor.group(0).strip()
        rem_addr = ORDINAL_FLOOR_RE.sub(" ", address)
        rem_addr = re.sub(r",\s*,+", ",", rem_addr)
        rem_addr = re.sub(r"^[\s,]+|[\s,]+$", "", rem_addr).strip()
        return rem_addr, extracted

    return address, None


def tokenize_address(text: str) -> List[str]:
    """Tokenize normalized address into alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


# ---------------------------------------------------------------------------
# Country-Specific Parsers
# ---------------------------------------------------------------------------

def normalize_us_address(
    raw_address: str,
    cleaned_text: str,
    landmark: Optional[str],
) -> NormalizedAddress:
    """Parse and extract structural components for United States addresses."""
    working_text, unit = extract_unit_info(cleaned_text)

    # 1. Postal Code (5-digit ZIP or ZIP+4)
    m_zip = US_ZIP_RE.search(working_text)
    postal_code = None
    if m_zip:
        postal_code = m_zip.group(1)
        working_text = US_ZIP_RE.sub(" ", working_text)

    # 2. State Identification
    state = None
    # Check comma-separated tokens first
    parts = [p.strip() for p in working_text.split(",") if p.strip()]

    # Match state from segments or words
    matched_part_idx = None
    for idx, part in enumerate(parts):
        # Look for 2-letter state code or full state name in segment
        words = part.lower().split()
        if len(words) == 1 and words[0].upper() in US_STATES:
            state = US_STATES[words[0].upper()]
            matched_part_idx = idx
            break
        elif part.lower() in US_STATES:
            state = US_STATES[part.lower()]
            matched_part_idx = idx
            break

    # If state not found in exact segment, search word tokens
    if not state:
        for word in re.findall(r"\b[A-Za-z]{2}\b", working_text):
            if word.upper() in US_STATES:
                state = US_STATES[word.upper()]
                break

    if not state:
        for full_name, code in US_STATES.items():
            if len(full_name) > 2 and re.search(r"\b" + re.escape(full_name) + r"\b", working_text, re.IGNORECASE):
                state = code
                break

    # Remove matched state part if it was a standalone segment
    if matched_part_idx is not None:
        parts.pop(matched_part_idx)

    # 3. Identify Street and City components
    street_number = None
    street_name = None
    city = None

    # Check for street number at the beginning of parts
    street_part_idx = None
    for idx, part in enumerate(parts):
        m_num = re.match(r"^(\d+(?:\s+1\/2|\s*-\s*[A-Za-z0-9]+)?)\s+(.+)", part)
        if m_num:
            street_number = m_num.group(1).replace(" ", "")
            street_name = m_num.group(2).strip()
            street_part_idx = idx
            break

    # If street number wasn't found at segment start, check for road keyword in parts
    if street_name is None:
        road_keywords = ["road", "street", "avenue", "drive", "court", "lane", "boulevard", "parkway", "highway", "trail", "way"]
        for idx, part in enumerate(parts):
            if any(re.search(r"\b" + kw + r"\b", part, re.IGNORECASE) for kw in road_keywords):
                m_num = re.search(r"\b(\d+)\b", part)
                if m_num:
                    street_number = m_num.group(1)
                    street_name = re.sub(r"\b" + m_num.group(1) + r"\b", "", part).strip(" ,-")
                else:
                    street_name = part.strip()
                street_part_idx = idx
                break

    # Remaining parts designate City / Locality
    remaining_parts = [p for i, p in enumerate(parts) if i != street_part_idx]
    if remaining_parts:
        city = remaining_parts[0].strip()

    tokens = tokenize_address(f"{cleaned_text} {landmark or ''}")

    return NormalizedAddress(
        raw_address=raw_address,
        country="US",
        cleaned_address=cleaned_text,
        street_number=street_number,
        street_name=street_name,
        unit=unit,
        landmark=landmark,
        city=city,
        state_or_region=state,
        postal_code=postal_code,
        tokens=tokens,
    )


def normalize_india_address(
    raw_address: str,
    cleaned_text: str,
    landmark: Optional[str],
) -> NormalizedAddress:
    """Parse and extract structural components for India addresses."""
    working_text, unit = extract_unit_info(cleaned_text)

    # 1. Postal Code (Indian 6-digit PIN)
    m_pin = INDIA_PIN_RE.search(working_text)
    postal_code = None
    if m_pin:
        postal_code = m_pin.group(1)
        working_text = INDIA_PIN_RE.sub(" ", working_text)

    # 2. State Identification
    state = None
    parts = [p.strip() for p in working_text.split(",") if p.strip()]

    matched_state_idx = None
    for idx, part in enumerate(parts):
        clean_p = part.lower().strip()
        if clean_p in INDIA_STATES:
            state = INDIA_STATES[clean_p]
            matched_state_idx = idx
            break

    if not state:
        for state_key, canonical_name in INDIA_STATES.items():
            if re.search(r"\b" + re.escape(state_key) + r"\b", working_text, re.IGNORECASE):
                state = canonical_name
                break

    if matched_state_idx is not None:
        parts.pop(matched_state_idx)

    # 3. Door / House / Flat / Plot / Khasra number
    street_number = None
    door_re = re.compile(
        r"\b((?:door no|house no|flat no|plot no|shop no|khasra no|no)\s*[\-\.\:]?\s*[A-Za-z0-9\/\-]+|\d+[\-\/][A-Za-z0-9\-\/]+|\d+)\b",
        re.IGNORECASE,
    )

    street_name = None
    city = None

    # Search for door/house number in first two segments
    for idx in range(min(2, len(parts))):
        m_door = door_re.search(parts[idx])
        if m_door:
            street_number = m_door.group(0).strip()
            # If the segment contains more text, that could be street / locality
            rem = door_re.sub("", parts[idx]).strip(" ,-")
            if rem:
                parts[idx] = rem
            else:
                parts.pop(idx)
            break

    # Look for road / cross / main / marg in remaining parts
    road_indicators = ["cross", "main", "road", "marg", "street", "lane", "colony", "nagar", "block", "sector", "phase"]
    street_idx = None
    for idx, part in enumerate(parts):
        if any(re.search(r"\b" + kw + r"\b", part, re.IGNORECASE) for kw in road_indicators):
            street_name = part.strip()
            street_idx = idx
            break

    remaining_parts = [p for i, p in enumerate(parts) if i != street_idx]
    if remaining_parts:
        # Typically the last remaining segment is the city/district
        city = remaining_parts[-1].strip()

    tokens = tokenize_address(f"{cleaned_text} {landmark or ''}")

    return NormalizedAddress(
        raw_address=raw_address,
        country="India",
        cleaned_address=cleaned_text,
        street_number=street_number,
        street_name=street_name,
        unit=unit,
        landmark=landmark,
        city=city,
        state_or_region=state,
        postal_code=postal_code,
        tokens=tokens,
    )


def normalize_generic_address(
    raw_address: str,
    country: str,
    cleaned_text: str,
    landmark: Optional[str],
) -> NormalizedAddress:
    """
    Generic fallback address normalizer for unseen countries (e.g. France).
    Gracefully handles international address formats via tokenization and
    heuristic component extraction.
    """
    working_text, unit = extract_unit_info(cleaned_text)

    # 1. Postal code (5 digits for France or European formats)
    m_post = GENERIC_POST_RE.search(working_text)
    postal_code = None
    if m_post:
        postal_code = m_post.group(1)
        working_text = GENERIC_POST_RE.sub(" ", working_text)

    parts = [p.strip() for p in working_text.split(",") if p.strip()]

    # 2. Region / Department (e.g. French regions)
    region = None
    matched_region_idx = None
    for idx, part in enumerate(parts):
        clean_p = part.lower().strip()
        if any(clean_p == r or r in clean_p for r in FRANCE_REGIONS):
            region = part.strip()
            matched_region_idx = idx
            break

    if matched_region_idx is not None:
        parts.pop(matched_region_idx)

    # 3. Street number & name (e.g., '175 Boulevard ...', '5 bis Rue Pierre Dignac')
    street_number = None
    street_name = None
    city = None

    street_keywords = ["rue", "boulevard", "avenue", "chemin", "allee", "place", "route", "road", "street"]
    street_idx = None
    for idx, part in enumerate(parts):
        m_num = re.match(r"^(\d+(?:\s+bis|\s+ter)?)\s+(.+)", part, re.IGNORECASE)
        if m_num:
            street_number = m_num.group(1).strip()
            street_name = m_num.group(2).strip()
            street_idx = idx
            break
        elif any(re.search(r"\b" + kw + r"\b", part, re.IGNORECASE) for kw in street_keywords):
            street_name = part.strip()
            street_idx = idx
            break

    remaining_parts = [p for i, p in enumerate(parts) if i != street_idx]
    if remaining_parts:
        city = remaining_parts[-1].strip()

    tokens = tokenize_address(f"{cleaned_text} {landmark or ''}")

    return NormalizedAddress(
        raw_address=raw_address,
        country=country,
        cleaned_address=cleaned_text,
        street_number=street_number,
        street_name=street_name,
        unit=unit,
        landmark=landmark,
        city=city,
        state_or_region=region,
        postal_code=postal_code,
        tokens=tokens,
    )


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def normalize_address(
    address: Optional[str],
    country: Optional[str] = None,
) -> NormalizedAddress:
    """
    Normalize an address string:
    - Standardize common abbreviations
    - Extract landmark phrases into a separate landmark field
    - Extract structural components (street number, street name, unit, city, state/region, postal code)
    - Handle US and India specifically, degrading gracefully for unseen countries (e.g. France)

    Args:
        address: Raw address string (can be None, empty, or nan)
        country: Country label ('US', 'India', 'France', or None)

    Returns:
        NormalizedAddress dataclass instance.
    """
    country_str = str(country).strip() if country is not None else ""
    if not country_str or country_str.lower() in ("nan", "none", "null"):
        country_str = "Unknown"

    cleaned = clean_text(address)
    if not cleaned:
        return NormalizedAddress(
            raw_address=str(address or ""),
            country=country_str,
            cleaned_address="",
            tokens=[],
        )

    # 1. Expand abbreviations
    expanded = expand_abbreviations(cleaned)

    # 2. Extract landmarks
    addr_without_landmark, landmark = extract_landmarks(expanded)

    # 3. Country-specific dispatch
    country_upper = country_str.upper()
    if country_upper in ("US", "USA", "UNITED STATES"):
        return normalize_us_address(
            raw_address=address,
            cleaned_text=addr_without_landmark,
            landmark=landmark,
        )
    elif country_upper in ("INDIA", "IN", "IND"):
        return normalize_india_address(
            raw_address=address,
            cleaned_text=addr_without_landmark,
            landmark=landmark,
        )
    else:
        # Fallback for France or any unseen country label
        return normalize_generic_address(
            raw_address=address,
            country=country_str,
            cleaned_text=addr_without_landmark,
            landmark=landmark,
        )


def batch_normalize_addresses(
    addresses: List[Optional[str]],
    countries: List[Optional[str]],
) -> List[NormalizedAddress]:
    """Normalize a batch of addresses with corresponding country labels."""
    return [
        normalize_address(addr, ctry)
        for addr, ctry in zip(addresses, countries)
    ]
