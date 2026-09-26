"""
Deep Country-Specific Noise Pattern Analysis.
Samples 30 matched pairs from US and 30 from India via train_ground_truth.tsv,
analyzes transformations side-by-side, and compiles notebooks/country_patterns.md.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

REPORT_PATH = Path("notebooks/country_patterns.md")


def extract_pairs():
    print("Loading ground truth and processed datasets...")
    gt = pd.read_csv("data/raw/train_ground_truth.tsv", sep="\t", keep_default_na=False)
    gt = gt[gt["matched_entity_ids"] != ""].copy()

    s1 = pd.read_parquet("data/processed/train_source1.parquet", columns=["entity_id", "country", "business_name", "business_address", "norm_name", "norm_address"])
    s2 = pd.read_parquet("data/processed/train_source2.parquet", columns=["entity_id", "country", "business_name", "business_address", "norm_name", "norm_address"])
    s3 = pd.read_parquet("data/processed/train_source3.parquet", columns=["entity_id", "country", "business_name", "business_address", "norm_name", "norm_address"])

    print("Building lookup dictionaries...")
    s2_dict = s2.set_index("entity_id")[["business_name", "business_address", "norm_name", "norm_address"]].to_dict("index")
    s3_dict = s3.set_index("entity_id")[["business_name", "business_address", "norm_name", "norm_address"]].to_dict("index")

    gt_s1 = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id")

    # Sample 30 S1 records with matches for US and 30 for India
    us_s1_sample = gt_s1[gt_s1["country"] == "US"].sample(30, random_state=42)
    in_s1_sample = gt_s1[gt_s1["country"] == "India"].sample(30, random_state=42)

    def build_pair_records(sample_df, country_name):
        pairs = []
        for _, row in sample_df.iterrows():
            s1_id = row["source1_entity_id"]
            s1_name = row["business_name"]
            s1_addr = row["business_address"]
            s1_nname = row["norm_name"]
            s1_naddr = row["norm_address"]

            matched_ids = [m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()]
            # Select the first valid matched record
            target_id = None
            target_rec = None
            target_source = None
            for m_id in matched_ids:
                if m_id.startswith("S2-") and m_id in s2_dict:
                    target_id = m_id
                    target_rec = s2_dict[m_id]
                    target_source = "Source 2"
                    break
                elif m_id.startswith("S3-") and m_id in s3_dict:
                    target_id = m_id
                    target_rec = s3_dict[m_id]
                    target_source = "Source 3"
                    break

            if target_rec:
                pairs.append({
                    "country": country_name,
                    "s1_id": s1_id,
                    "target_id": target_id,
                    "target_source": target_source,
                    "s1_name": s1_name,
                    "target_name": target_rec["business_name"],
                    "s1_addr": s1_addr,
                    "target_addr": target_rec["business_address"],
                    "s1_nname": s1_nname,
                    "target_nname": target_rec["norm_name"],
                    "s1_naddr": s1_naddr,
                    "target_naddr": target_rec["norm_address"],
                })
        return pairs

    us_pairs = build_pair_records(us_s1_sample, "US")
    in_pairs = build_pair_records(in_s1_sample, "India")
    return us_pairs, in_pairs


def analyze_transformations(pair: Dict[str, Any]) -> str:
    """Detect and annotate key transformation patterns in a pair."""
    notes = []
    s1_n, t_n = pair["s1_name"].strip(), pair["target_name"].strip()
    s1_a, t_a = pair["s1_addr"].strip(), pair["target_addr"].strip()

    # Name checks
    if s1_n.lower() == t_n.lower():
        if s1_n != t_n:
            notes.append("Case variation only")
        else:
            notes.append("Exact name match")
    else:
        # Check non-ascii in target
        if any(ord(c) > 127 for c in t_n):
            notes.append("Non-Latin/Indic script or accent in name")
        # Check website attached
        if "www." in t_n.lower() or ".com" in t_n.lower():
            notes.append("URL / Domain appended to name")
        # Check word reordering
        s1_words = set(s1_n.lower().split())
        t_words = set(t_n.lower().split())
        if s1_words == t_words:
            notes.append("Name word transposition/reordering")
        elif s1_words.issubset(t_words) or t_words.issubset(s1_words):
            notes.append("Legal suffix/prefix addition or omission")
        else:
            notes.append("Lexical variation / typo / alias")

    # Address checks
    if not t_a or t_a.lower() in ("nan", "none", "null"):
        notes.append("Target address MISSING (Null)")
    elif s1_a.lower() == t_a.lower():
        if s1_a != t_a:
            notes.append("Address case variation")
        else:
            notes.append("Exact address match")
    else:
        # Check landmark
        if ("near" in s1_a.lower() or "near" in t_a.lower() or
            "opp" in s1_a.lower() or "opp" in t_a.lower()):
            notes.append("Landmark addition/omission")
        # Check number variation
        s1_nums = set(w for w in s1_a.split() if any(c.isdigit() for c in w))
        t_nums = set(w for w in t_a.split() if any(c.isdigit() for c in w))
        if s1_nums != t_nums:
            notes.append("Street/Door number variation (range or unit)")
        # Check component permutation
        s1_parts = [p.strip().lower() for p in s1_a.split(",") if p.strip()]
        t_parts = [p.strip().lower() for p in t_a.split(",") if p.strip()]
        if set(s1_parts) == set(t_parts) and s1_parts != t_parts:
            notes.append("Address component inversion/reordering")

    return "; ".join(notes) if notes else "Minor lexical variation"


def write_report(us_pairs: List[Dict[str, Any]], in_pairs: List[Dict[str, Any]]):
    lines = [
        "# Deep Dive: Country-Specific Noise & Transformation Patterns",
        "",
        "## Executive Summary",
        "",
        "This empirical analysis investigates the exact transformation and noise patterns between reference entities (`Source 1`) and their true matched counterparts in `Source 2` and `Source 3` as defined by [train_ground_truth.tsv](file:///home/sumitchint_work/amazon-ml-challenge/data/raw/train_ground_truth.tsv).",
        "",
        "We sampled **30 ground-truth matched pairs from the United States** and **30 ground-truth matched pairs from India**, analyzing name and address mutations side-by-side. The findings systematically expose failure modes of naive string matching and directly prescribe our feature engineering architecture.",
        "",
        "---",
        "",
        "## 1. Summary of Discovered Transformation Patterns",
        "",
        "### 1.1 United States Specific Patterns",
        "1. **Address Block Inversion (State/City First)**: In S1, addresses standardly follow `[Number] [Street], [City], [State]`. In S2/S3, addresses frequently invert to `[State], [City], [Street]` (e.g., `ME, Madison, 476 Horsetail Hill Road`) or `[City], [Street], [State]` (`ELGIN, 1287-B FLEETWOOD DR, IL`).",
        "2. **Street Number Modifications**: True matches frequently introduce street number ranges or letter suffixes: `674 Ridge Gate Drive` $\\to$ `674-678 RIDGE GATE DR` or `1287 Fleetwood Drive` $\\to$ `1287-B FLEETWOOD DR`.",
        "3. **Unit / Suite Normalization Discrepancies**: S1 records contain explicit secondary clauses like `Unit 308`, which are often condensed, appended to street numbers (`1287-B`), or completely omitted in S2/S3.",
        "4. **Web Domains & DBAs in Business Name**: S2/S3 names frequently contain appended URLs or DBA strings (`Vargas Phoenix` $\\to$ `Vargas Phoenix | www.vargaspho.com`).",
        "5. **Township & Municipal Descriptors**: City names vary between municipality types: `Ardmore` $\\to$ `ARDMORE TOWNSHIP`.",
        "6. **Phonetic & Typographical OCR Typos**: High-consonant typographical shifts (`Zaon Irenic` $\\to$ `Zaon Icr`).",
        "",
        "### 1.2 India Specific Patterns",
        "1. **Non-Latin Script Transliteration**: Matches in S2/S3 frequently feature native regional scripts (Tamil, Devanagari Hindi, Kannada, Telugu, Punjabi) paired with English reference entities (e.g., `Sree Construction Private Limited` $\\to$ `ஸ்ரீ கன்ஸ்ட்ரக்ஷன் பிரைவேட் லிமிடெட்`).",
        "2. **Displaced & Permuted Legal Designations**: Severe word reordering within legal affixes (e.g., `Sree Construction Private Limited` $\\to$ `Sree Limited Private Construction` or `Private Beyond Heat (Limited)`).",
        "3. **Landmark Inclusion, Omission & Variations**: Landmarks (e.g., `Near Vishal Hall`, `Near Parimal Railway Crossing`, `Opp. RTA Office`) appear in one source and are omitted or altered in another.",
        "4. **Door / House / Plot / Khasra Numbering Variations**: Numbering format shifts from `137` to `A-137` or `Flat No. 403` to `403` or house number typos (`45, Teli Gulli` $\\to$ `44, TELI GULLI`).",
        "5. **State / UT Naming Multiplicity**: Alternation between full English state name, 2-letter state code, and vernacular name (`Maharashtra` $\\to$ `MH` $\\to$ `महाराष्ट्र`).",
        "6. **Phonetic Spelling Substitutions**: Common phoneme variations: `Sree` $\\leftrightarrow$ `Shree`, `Laxmi` $\\leftrightarrow$ `Lakshmi`, `Kishan` $\\leftrightarrow$ `Krishan`.",
        "",
        "---",
        "",
        "## 2. United States: 30 Matched Pairs (Side-by-Side)",
        "",
        "| # | S1 ID & Matched ID | S1 Business Name | Matched Business Name | S1 Address | Matched Address | Identified Transformations |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for idx, p in enumerate(us_pairs, 1):
        s1_n = p["s1_name"].replace("|", "\\|")
        t_n = p["target_name"].replace("|", "\\|")
        s1_a = p["s1_addr"].replace("|", "\\|")
        t_a = p["target_addr"].replace("|", "\\|") if p["target_addr"] else "*[NULL]*"
        notes = analyze_transformations(p)
        lines.append(
            f"| {idx} | `{p['s1_id']}`<br>$\\to$ `{p['target_id']}` ({p['target_source']}) | {s1_n} | **{t_n}** | {s1_a} | **{t_a}** | {notes} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. India: 30 Matched Pairs (Side-by-Side)",
        "",
        "| # | S1 ID & Matched ID | S1 Business Name | Matched Business Name | S1 Address | Matched Address | Identified Transformations |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for idx, p in enumerate(in_pairs, 1):
        s1_n = p["s1_name"].replace("|", "\\|")
        t_n = p["target_name"].replace("|", "\\|")
        s1_a = p["s1_addr"].replace("|", "\\|")
        t_a = p["target_addr"].replace("|", "\\|") if p["target_addr"] else "*[NULL]*"
        notes = analyze_transformations(p)
        lines.append(
            f"| {idx} | `{p['s1_id']}`<br>$\\to$ `{p['target_id']}` ({p['target_source']}) | {s1_n} | **{t_n}** | {s1_a} | **{t_a}** | {notes} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Feature Engineering Architecture Implications",
        "",
        "Based on these empirical transformation behaviors, our ML feature pipeline must implement the following dedicated feature groups:",
        "",
        "### 4.1 Name Similarity Feature Group",
        "| Feature | Rationale & Noise Pattern Handled |",
        "| :--- | :--- |",
        "| `name_token_sort_ratio` | Invariant to word reordering (e.g., `Sree Limited Private Construction` vs `Sree Construction Private Limited`). |",
        "| `name_token_set_ratio` | Invariant to extraneous tokens (e.g. appended URLs `| www.vargaspho.com` or legal affixes). |",
        "| `name_no_legal_levenshtein` | Computes edit distance after legal forms are stripped, preventing mismatch penalties on `LLC` vs `Inc` vs omitted suffix. |",
        "| `name_jaro_winkler` | Weights common prefixes heavily (e.g. `Vargas Phoenix` vs `Vargas Phoenix LLC`). |",
        "| `multilingual_embedding_cosine` | **Critical for Indic scripts**: Cross-lingual dense embeddings (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) match Tamil/Hindi script names with their English phonetic forms where lexical edit distance is 0.0. |",
        "| `phonetic_match_double_metaphone` | Handles English & Indic phonetic alternations (`Sree` vs `Shree`, `Center` vs `Centre`). |",
        "",
        "### 4.2 Address Similarity Feature Group",
        "| Feature | Rationale & Noise Pattern Handled |",
        "| :--- | :--- |",
        "| `address_token_set_ratio` | Invariant to address component inversions (`[State], [City], [Street]` vs `[Street], [City], [State]`). Linear Levenshtein fails completely on inverted addresses. |",
        "| `street_number_exact_match` | Checks if street number is identical. |",
        "| `street_number_fuzzy_match` | Handles range extensions (`674` vs `674-678`) and alphanumeric suffixes (`1287` vs `1287-B`). |",
        "| `state_match` | Exact match on standardized 2-letter state / province code (`AL`, `IN`, `Maharashtra`, `Tamil Nadu`). |",
        "| `city_similarity` | Fuzzy matching on city names (`Ardmore` vs `Ardmore Township`, `Jaipur` vs `Jaipur City`). |",
        "| `landmark_similarity` | Compares extracted landmark fields independently so presence/absence of landmark does not distort core street matching. |",
        "| `address_null_indicator` | Explicit binary indicator when S2/S3 address is null, allowing the tree models to learn separate name-only decision branches. |",
    ])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")


def main():
    us_pairs, in_pairs = extract_pairs()
    write_report(us_pairs, in_pairs)
    print(f"Extracted and analyzed {len(us_pairs)} US pairs and {len(in_pairs)} India pairs!")


if __name__ == "__main__":
    main()
