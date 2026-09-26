"""
Business Name Normalization Module for Amazon ML Challenge 2026.

Standardizes trade terminology, extracts and canonicalizes legal entity affixes,
handles displaced legal prefixes (moving them or extracting them), strips legal
designations for core brand matching, and normalizes multilingual/accented text.
Supports US, India, and unseen countries (e.g., France).
"""

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NormalizedName:
    """Structured representation of a normalized business name."""
    raw_name: str
    country: str
    norm_name: str = ""
    norm_name_no_legal: str = ""
    legal_type: Optional[str] = None
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Canonical legal forms mapping (lowercase pattern -> canonical legal form)
LEGAL_FORMS = [
    # India / Commonwealth multi-word
    (r"\b(pvt\.?\s*ltd\.?|private\s*limited|pvt\.?\s*limited|private\s*ltd\.?)\b", "private limited"),
    (r"\b(public\s*limited|pub\.?\s*ltd\.?)\b", "public limited"),
    # US & General
    (r"\b(limited\s*liability\s*company|l\.?l\.?c\.?)\b", "llc"),
    (r"\b(incorporated|inc\.?)\b", "inc"),
    (r"\b(corporation|corp\.?)\b", "corporation"),
    (r"\b(limited\s*liability\s*partnership|l\.?l\.?p\.?)\b", "llp"),
    (r"\b(professional\s*llc|p\.?l\.?l\.?c\.?)\b", "pllc"),
    (r"\b(limited\s*partnership|l\.?p\.?)\b", "lp"),
    (r"\b(professional\s*corporation|p\.?c\.?)\b", "pc"),
    (r"\b(limited|ltd\.?)\b", "limited"),
    (r"\b(company|co\.?)\b", "company"),
    # France
    (r"\b(s\.?a\.?r\.?l\.?|societe\s*a\s*responsabilite\s*limitee)\b", "sarl"),
    (r"\b(s\.?a\.?s\.?u\.?)\b", "sasu"),
    (r"\b(s\.?a\.?s\.?)\b", "sas"),
    (r"\b(s\.?a\.?)\b", "sa"),
    (r"\b(s\.?c\.?i\.?|societe\s*civile\s*immobiliere)\b", "sci"),
    (r"\b(e\.?u\.?r\.?l\.?)\b", "eurl"),
    (r"\b(s\.?n\.?c\.?)\b", "snc"),
]

# Prefixes often placed at the start in Indian & French corporate records
LEADING_AFFIXES = [
    (r"^(?:m\/s\.?|messrs\.?|shri|shree|smt\.?)\s+", None),
    (r"^(?:pvt\.?\s*ltd\.?|private\s*limited|pvt\.?|private)\s+", "private limited"),
    (r"^(?:llc|limited\s*liability\s*company)\s+", "llc"),
    (r"^(?:inc\.?|incorporated)\s+", "inc"),
    (r"^(?:corp\.?|corporation)\s+", "corporation"),
    (r"^(?:ltd\.?|limited)\s+", "limited"),
    (r"^(?:sarl)\s+", "sarl"),
    (r"^(?:sas|sasu)\s+", "sas"),
    (r"^(?:sci)\s+", "sci"),
    (r"^(?:sa)\s+", "sa"),
]

# Trade symbol replacements
TRADE_SYMBOLS = {
    r"&": " and ",
    r"@": " at ",
    r"\+": " plus ",
    r"\bco\.?\b": "company",
    r"\bmfg\.?\b": "manufacturing",
    r"\bintl\.?\b": "international",
    r"\bdept\.?\b": "department",
    r"\btech\.?\b": "technology",
    r"\bsvc\.?\b|\bsvcs\.?\b": "services",
    r"\bassn\.?\b": "association",
    r"\bctr\.?\b": "center",
    r"\bdba\b|\bd\/b\/a\b": "dba",
}


def clean_name_text(text: Optional[str]) -> str:
    """Normalize unicode, strip accents, outer noise symbols and collapse spaces."""
    if text is None:
        return ""
    text = str(text).strip()
    if text.lower() in ("nan", "none", "null"):
        return ""

    # Normalize unicode to decomposed NFKD form and strip combining accents
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

    # Strip noise prefix and suffix characters: <<, >>, --, ##, **, //
    text = re.sub(r"^[\s\<\>\-\+\#\*\/\\]+", "", text)
    text = re.sub(r"[\s\<\>\-\+\#\*\/\\]+$", "", text)

    # Strip apostrophes (e.g. Orelee's -> Orelees)
    text = re.sub(r"['’`]", "", text)

    # Replace tabs and multiple spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text



def extract_legal_form(text: str) -> Tuple[str, Optional[str]]:
    """
    Identify and extract legal entity form (e.g. 'llc', 'private limited', 'corporation').
    Returns (cleaned_text_without_legal_form, legal_type)
    """
    detected_legal: Optional[str] = None
    stripped_text = text

    for pattern, canonical in LEGAL_FORMS:
        m = re.search(pattern, stripped_text, flags=re.IGNORECASE)
        if m:
            detected_legal = canonical
            # Replace the legal term with space
            stripped_text = re.sub(pattern, " ", stripped_text, count=1, flags=re.IGNORECASE)
            break

    # Clean residual punctuation like trailing commas, hyphens, and whitespace
    stripped_text = re.sub(r"[\,\.\-\/\(\)]+", " ", stripped_text)
    stripped_text = re.sub(r"\s+", " ", stripped_text).strip()

    return stripped_text, detected_legal


def normalize_name(
    name: Optional[str],
    country: Optional[str] = None,
) -> NormalizedName:
    """
    Normalize a business entity name:
    - Unicode decomposition and accent removal
    - Strip noisy bounding symbols (<<, >>, --, ##)
    - Expand trade abbreviations (& -> and, co -> company, mfg -> manufacturing)
    - Extract and canonicalize legal entity types (LLC, Inc, Pvt Ltd, SARL)
    - Handle displaced legal prefixes (e.g. LLC at start)
    - Generate both full normalized name and core name without legal designations

    Args:
        name: Raw business name
        country: Country label ('US', 'India', 'France', or None)

    Returns:
        NormalizedName dataclass instance.
    """
    country_str = str(country).strip() if country is not None else ""
    if not country_str or country_str.lower() in ("nan", "none", "null"):
        country_str = "Unknown"

    cleaned = clean_name_text(name)
    if not cleaned:
        return NormalizedName(
            raw_name=str(name or ""),
            country=country_str,
            norm_name="",
            norm_name_no_legal="",
            legal_type=None,
            tokens=[],
        )

    # Check for leading honorific / displaced legal prefixes (e.g., M/S, LLC, Pvt Ltd, SCI)
    displaced_legal = None
    working_name = cleaned
    for pattern, canonical in LEADING_AFFIXES:
        m = re.search(pattern, working_name, flags=re.IGNORECASE)
        if m:
            if canonical:
                displaced_legal = canonical
            working_name = re.sub(pattern, "", working_name, count=1, flags=re.IGNORECASE)
            break

    for pattern, canonical in LEGAL_FORMS:
        prefix_pattern = r"^" + pattern + r"\s+"
        if re.search(prefix_pattern, working_name, flags=re.IGNORECASE):
            displaced_legal = canonical
            working_name = re.sub(prefix_pattern, "", working_name, count=1, flags=re.IGNORECASE)
            break

    # Expand trade symbols
    for pattern, replacement in TRADE_SYMBOLS.items():
        working_name = re.sub(pattern, replacement, working_name, flags=re.IGNORECASE)

    # Extract legal entity form (from anywhere, primarily suffix)
    name_no_legal, detected_legal = extract_legal_form(working_name)
    legal_type = detected_legal or displaced_legal

    # Construct canonical normalized name: core name + canonical legal form
    norm_name_no_legal = re.sub(r"[^\w\s]", " ", name_no_legal.lower())
    norm_name_no_legal = re.sub(r"\s+", " ", norm_name_no_legal).strip()

    if legal_type:
        norm_name = f"{norm_name_no_legal} {legal_type}".strip()
    else:
        norm_name = norm_name_no_legal

    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", norm_name)

    return NormalizedName(
        raw_name=cleaned,
        country=country_str,
        norm_name=norm_name,
        norm_name_no_legal=norm_name_no_legal,
        legal_type=legal_type,
        tokens=tokens,
    )
