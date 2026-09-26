"""
Comprehensive EDA Script for Amazon ML Challenge Training Sources.
Computes distributions, missingness, length statistics, token frequencies,
formatting patterns, and generates publication-grade figures.
"""

import re
from collections import Counter
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loading import load_source

OUTPUT_DIR = Path("notebooks/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("notebooks/eda_report.md")

# Suffixes and patterns to detect
COMMON_LEGAL_SUFFIXES = [
    "llc", "inc", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt ltd", "private limited", "pvt", "llp", "pllc", "gmbh", "sarl", "sa", "sas"
]

def analyze_source(source_num: int):
    print(f"Loading and analyzing train_source{source_num}...")
    df = load_source("train", source_num)
    n_rows = len(df)

    # 1. Country distribution
    country_counts = df["country"].value_counts(dropna=False).to_dict()
    country_pcts = {k: (v / n_rows) * 100 for k, v in country_counts.items()}

    # 2. Missing/Empty counts
    name_na = int(df["business_name"].isna().sum())
    name_empty = int((df["business_name"].fillna("").str.strip() == "").sum())

    addr_na = int(df["business_address"].isna().sum())
    addr_empty = int((df["business_address"].fillna("").str.strip() == "").sum())

    # 3. Length distributions
    # Clean strings for length calculation
    name_clean = df["business_name"].dropna().astype(str)
    addr_clean = df["business_address"].dropna().astype(str)

    name_char_len = name_clean.str.len()
    name_word_len = name_clean.str.split().str.len()

    addr_char_len = addr_clean.str.len()
    addr_word_len = addr_clean.str.split().str.len()

    def get_stats(series: pd.Series):
        desc = series.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_dict()
        return {
            "count": int(desc.get("count", 0)),
            "mean": float(desc.get("mean", 0)),
            "std": float(desc.get("std", 0)),
            "min": float(desc.get("min", 0)),
            "1%": float(desc.get("1%", 0)),
            "5%": float(desc.get("5%", 0)),
            "25%": float(desc.get("25%", 0)),
            "50%": float(desc.get("50%", 0)),
            "75%": float(desc.get("75%", 0)),
            "95%": float(desc.get("95%", 0)),
            "99%": float(desc.get("99%", 0)),
            "max": float(desc.get("max", 0)),
        }

    stats = {
        "source": f"train_source{source_num}",
        "total_rows": n_rows,
        "country_counts": country_counts,
        "country_pcts": country_pcts,
        "name_na": name_na,
        "name_empty": name_empty,
        "name_missing_pct": (name_empty / n_rows) * 100,
        "addr_na": addr_na,
        "addr_empty": addr_empty,
        "addr_missing_pct": (addr_empty / n_rows) * 100,
        "name_char_stats": get_stats(name_char_len),
        "name_word_stats": get_stats(name_word_len),
        "addr_char_stats": get_stats(addr_char_len),
        "addr_word_stats": get_stats(addr_word_len),
    }

    # 4. Top tokens by country
    print("Computing top tokens by country...")
    tokens_by_country = {}
    for country in ["US", "India"]:
        sub_names = df.loc[df["country"] == country, "business_name"].dropna().astype(str)
        # Sample if very large to speed up tokenization while retaining robust distribution
        sample_size = min(len(sub_names), 500_000)
        sample = sub_names.sample(sample_size, random_state=42)
        words = []
        for text in sample:
            tokens = re.findall(r"\b[\w\'-]+\b", text.lower())
            words.extend(tokens)
        counter = Counter(words)
        tokens_by_country[country] = counter.most_common(40)

    # 5. Formatting patterns
    print("Checking formatting patterns...")
    sample_names = df["business_name"].dropna().sample(min(len(name_clean), 200_000), random_state=42).astype(str)
    sample_addrs = df["business_address"].dropna().sample(min(len(addr_clean), 200_000), random_state=42).astype(str)

    # Non-ASCII / Non-Latin scripts check (e.g. Devanagari, Kannada, etc.)
    non_ascii_names = int(sample_names.str.contains(r"[^\x00-\x7F]").sum())
    non_ascii_addrs = int(sample_addrs.str.contains(r"[^\x00-\x7F]").sum())

    # Leading legal prefixes (e.g., 'LLC ...', 'Pvt ...')
    leading_legal = int(sample_names.str.contains(r"^(pvt|private|llc|inc|corp|ltd|m\/s|shri)\b", case=False).sum())

    # Trailing legal suffixes
    trailing_legal = int(sample_names.str.contains(r"\b(llc|inc|corp|corporation|ltd|limited|llp|co)\.?$", case=False).sum())

    # Punctuation prefixes like '<<', '--', '>>'
    punct_prefix = int(sample_names.str.contains(r"^[\s\<\>\-\+\#\*]+").sum())

    # All-caps names / addresses
    all_caps_names = int(sample_names.str.isupper().sum())
    all_caps_addrs = int(sample_addrs.str.isupper().sum())

    stats["tokens_by_country"] = tokens_by_country
    stats["patterns"] = {
        "sample_size": len(sample_names),
        "non_ascii_name_pct": (non_ascii_names / len(sample_names)) * 100,
        "non_ascii_addr_pct": (non_ascii_addrs / len(sample_addrs)) * 100,
        "leading_legal_pct": (leading_legal / len(sample_names)) * 100,
        "trailing_legal_pct": (trailing_legal / len(sample_names)) * 100,
        "punct_prefix_pct": (punct_prefix / len(sample_names)) * 100,
        "all_caps_names_pct": (all_caps_names / len(sample_names)) * 100,
        "all_caps_addrs_pct": (all_caps_addrs / len(sample_addrs)) * 100,
    }

    # Store raw series sample for plotting (downsample to 50,000 for fast high-res density plots)
    stats["name_char_sample"] = name_char_len.sample(min(len(name_char_len), 50_000), random_state=42).tolist()
    stats["addr_char_sample"] = addr_char_len.sample(min(len(addr_char_len), 50_000), random_state=42).tolist()

    return stats


def generate_figures(all_stats):
    print("Generating figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Country distribution across sources
    fig, ax = plt.subplots(figsize=(8, 5))
    sources = [s["source"] for s in all_stats]
    us_counts = [s["country_counts"].get("US", 0) for s in all_stats]
    india_counts = [s["country_counts"].get("India", 0) for s in all_stats]

    x = np.arange(len(sources))
    width = 0.35

    rects1 = ax.bar(x - width/2, us_counts, width, label="US", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, india_counts, width, label="India", color="#e27c38")

    ax.set_ylabel("Record Count")
    ax.set_title("Country Distribution Across Training Sources")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    # Add text labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height/1e6:.2f}M",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "country_distribution.png", dpi=200)
    plt.close(fig)

    # Figure 2: Missingness of business_address
    fig, ax = plt.subplots(figsize=(7, 4.5))
    missing_pcts = [s["addr_missing_pct"] for s in all_stats]
    bars = ax.bar(sources, missing_pcts, color=["#4c72b0", "#55a868", "#c44e52"], width=0.45)
    ax.set_ylabel("Missing Address (%)")
    ax.set_title("Percentage of Missing/Empty Business Address by Source")
    ax.set_ylim(0, max(missing_pcts) * 1.35 if max(missing_pcts) > 0 else 5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "address_missingness.png", dpi=200)
    plt.close(fig)

    # Figure 3: Business Name character length distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2b5c8f", "#2ca02c", "#d62728"]
    for s, c in zip(all_stats, colors):
        sample = np.array(s["name_char_sample"])
        ax.hist(sample, bins=np.arange(0, 100, 2), density=True, alpha=0.45, label=f"{s['source']} (mean={s['name_char_stats']['mean']:.1f})", color=c)
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Business Name Character Length")
    ax.set_xlim(0, 90)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "name_length_distribution.png", dpi=200)
    plt.close(fig)

    # Figure 4: Business Address character length distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for s, c in zip(all_stats, colors):
        sample = np.array(s["addr_char_sample"])
        ax.hist(sample, bins=np.arange(0, 200, 4), density=True, alpha=0.45, label=f"{s['source']} (mean={s['addr_char_stats']['mean']:.1f})", color=c)
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Business Address Character Length (Non-Null)")
    ax.set_xlim(0, 180)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "address_length_distribution.png", dpi=200)
    plt.close(fig)

    # Figure 5: Top tokens by country (US vs India) from Source 1
    s1 = all_stats[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    us_tokens = s1["tokens_by_country"]["US"][:15]
    in_tokens = s1["tokens_by_country"]["India"][:15]

    w_us, c_us = zip(*us_tokens[::-1])
    w_in, c_in = zip(*in_tokens[::-1])

    ax1.barh(w_us, c_us, color="#2b5c8f")
    ax1.set_title("Top 15 Tokens in US Business Names (Source 1)")
    ax1.set_xlabel("Frequency (in 500k sample)")

    ax2.barh(w_in, c_in, color="#e27c38")
    ax2.set_title("Top 15 Tokens in India Business Names (Source 1)")
    ax2.set_xlabel("Frequency (in 500k sample)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_tokens_by_country.png", dpi=200)
    plt.close(fig)
    print("All figures successfully saved to notebooks/figures/.")


def generate_markdown_report(all_stats):
    print("Compiling markdown report...")
    lines = [
        "# Exploratory Data Analysis: Training Sources",
        "",
        "## Executive Summary",
        "",
        "This report provides an in-depth exploratory data analysis (EDA) across the three training sources (`train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`) for the Amazon ML Challenge 2026 Business Entity Resolution task.",
        "",
        "### Key Findings at a Glance:",
        "1. **Scale & Reference Role**: Source 1 serves as the clean reference dataset with **2,206,821** records, with zero null business names and zero null addresses. Sources 2 and 3 represent large external sources with **5,034,616** and **5,285,603** records respectively.",
        "2. **Country Distribution**: Training data covers two countries: `US` and `India`. Across all sources, the US comprises ~54% of records and India comprises ~46%.",
        "3. **Address Missingness**: While Source 1 has complete addresses, Sources 2 and 3 have substantial missing addresses (~**3.36%** in S2, ~**3.33%** in S3, over 168k records each). Entity resolution models must handle missing addresses gracefully using fallback name matching.",
        "4. **Length Profiles**: Business names average ~22 to 24 characters (3 to 3.5 words). Addresses average ~47 to 57 characters (6 to 8 words).",
        "5. **High Lexical Specificity & Legal Tokens**: Country-specific vocabularies show distinct legal entity suffixes: `llc`, `inc`, `corp`, `co` in the US vs. `pvt`, `ltd`, `private`, `limited`, `llp` in India.",
        "6. **Crucial Formatting & Inversion Noise**:",
        "   - **Legal token displacement**: Suffixes often appear as *prefixes* (e.g., `LLC Moncada Learning Center`, `Pvt. EFS Print Ventures Ltd.`).",
        "   - **Address component permutation**: Components are frequently inverted (e.g., `State, City, Street, Apt` in S1 vs. `City, State, Street` in S2).",
        "   - **Multilingual / Non-ASCII Scripts**: Indian records contain regional scripts (Devanagari, Kannada, Hindi, etc.) and transliteration variants.",
        "   - **Extraneous punctuation**: Leading symbols like `<<`, `--`, `>>` and internal quotes.",
        "",
        "---",
        "",
        "## 1. Source Overview & Volume",
        "",
        "| Source | Total Records | Columns | Primary Key Prefix | Missing Names | Missing Addresses |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in all_stats:
        lines.append(
            f"| `{s['source']}` | {s['total_rows']:,} | 4 | `{s['source'][-1].upper()}...` | {s['name_na']:,} ({s['name_na']/s['total_rows']*100:.4f}%) | {s['addr_na']:,} ({s['addr_missing_pct']:.2f}%) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Country Distribution",
        "",
        "The training datasets cover entities from the United States (`US`) and India (`India`).",
        "",
        "| Source | US Records | US % | India Records | India % | Total |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in all_stats:
        us = s["country_counts"].get("US", 0)
        us_pct = s["country_pcts"].get("US", 0.0)
        ind = s["country_counts"].get("India", 0)
        ind_pct = s["country_pcts"].get("India", 0.0)
        lines.append(f"| `{s['source']}` | {us:,} | {us_pct:.2f}% | {ind:,} | {ind_pct:.2f}% | {s['total_rows']:,} |")

    lines.extend([
        "",
        "![Country Distribution](figures/country_distribution.png)",
        "",
        "> [!IMPORTANT]",
        "> Note: In the test set, an unseen country (`France`) is introduced. Preprocessing, tokenization, and blocking logic must remain country-agnostic and avoid hardcoded filters for `{US, India}`.",
        "",
        "---",
        "",
        "## 3. Missing Data Analysis",
        "",
        "| Source | Missing Business Name | Empty Name String | Missing Address (`NaN`) | Empty Address String | Address Missing % |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in all_stats:
        lines.append(
            f"| `{s['source']}` | {s['name_na']:,} | {s['name_empty']:,} | {s['addr_na']:,} | {s['addr_empty']:,} | **{s['addr_missing_pct']:.2f}%** |"
        )

    lines.extend([
        "",
        "![Address Missingness](figures/address_missingness.png)",
        "",
        "> [!NOTE]",
        "> Source 1 has complete address coverage (0% missing). Sources 2 and 3 exhibit missing addresses on ~169k and ~176k records respectively. Entity resolution algorithms must avoid penalizing address similarity when address information is absent in one or both records.",
        "",
        "---",
        "",
        "## 4. Length Distributions",
        "",
        "### 4.1 Business Name Lengths",
        "",
        "| Source | Metric | Min | 5% | 25% | Median (50%) | 75% | 95% | 99% | Max | Mean ± Std |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in all_stats:
        c = s["name_char_stats"]
        w = s["name_word_stats"]
        lines.append(
            f"| `{s['source']}` | **Chars** | {c['min']:.0f} | {c['5%']:.0f} | {c['25%']:.0f} | {c['50%']:.0f} | {c['75%']:.0f} | {c['95%']:.0f} | {c['99%']:.0f} | {c['max']:.0f} | {c['mean']:.1f} ± {c['std']:.1f} |"
        )
        lines.append(
            f"| `{s['source']}` | **Words** | {w['min']:.0f} | {w['5%']:.0f} | {w['25%']:.0f} | {w['50%']:.0f} | {w['75%']:.0f} | {w['95%']:.0f} | {w['99%']:.0f} | {w['max']:.0f} | {w['mean']:.1f} ± {w['std']:.1f} |"
        )

    lines.extend([
        "",
        "![Business Name Length Distribution](figures/name_length_distribution.png)",
        "",
        "### 4.2 Business Address Lengths (Non-Null)",
        "",
        "| Source | Metric | Min | 5% | 25% | Median (50%) | 75% | 95% | 99% | Max | Mean ± Std |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for s in all_stats:
        c = s["addr_char_stats"]
        w = s["addr_word_stats"]
        lines.append(
            f"| `{s['source']}` | **Chars** | {c['min']:.0f} | {c['5%']:.0f} | {c['25%']:.0f} | {c['50%']:.0f} | {c['75%']:.0f} | {c['95%']:.0f} | {c['99%']:.0f} | {c['max']:.0f} | {c['mean']:.1f} ± {c['std']:.1f} |"
        )
        lines.append(
            f"| `{s['source']}` | **Words** | {w['min']:.0f} | {w['5%']:.0f} | {w['25%']:.0f} | {w['50%']:.0f} | {w['75%']:.0f} | {w['95%']:.0f} | {w['99%']:.0f} | {w['max']:.0f} | {w['mean']:.1f} ± {w['std']:.1f} |"
        )

    lines.extend([
        "",
        "![Business Address Length Distribution](figures/address_length_distribution.png)",
        "",
        "---",
        "",
        "## 5. Most Common Words/Tokens in Business Names Across Countries",
        "",
        "Token frequency analysis reveals a stark contrast between entity structures in the United States and India:",
        "",
        "### 5.1 Top 25 Tokens in US vs. India (Source 1 Sample)",
        "",
        "| Rank | US Token | Frequency | India Token | Frequency |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    us_tokens = all_stats[0]["tokens_by_country"]["US"][:25]
    in_tokens = all_stats[0]["tokens_by_country"]["India"][:25]
    for r in range(25):
        u_tok, u_freq = us_tokens[r]
        i_tok, i_freq = in_tokens[r]
        lines.append(f"| {r+1} | `{u_tok}` | {u_freq:,} | `{i_tok}` | {i_freq:,} |")

    lines.extend([
        "",
        "![Top Tokens by Country](figures/top_tokens_by_country.png)",
        "",
        "#### Key Lexical Observations:",
        "- **US**: Heavily dominated by corporate entity forms (`llc`, `inc`, `corp`, `co`, `company`, `services`, `group`) and sector descriptors (`medical`, `auto`, `center`, `care`, `dental`, `real`, `estate`).",
        "- **India**: Overwhelmingly dominated by Indian corporate law designations (`pvt`, `ltd`, `private`, `limited`, `enterprises`, `services`, `technologies`, `solutions`, `india`, `associates`, `traders`, `industries`, `llp`).",
        "",
        "---",
        "",
        "## 6. Formatting Patterns, Noise Characteristics & Anomalies",
        "",
        "Analysis of character casing, legal affix positions, and punctuation noise shows specific systemic patterns:",
        "",
        "| Pattern / Anomaly | Source 1 | Source 2 | Source 3 | Impact on Entity Resolution |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for metric, desc in [
        ("non_ascii_name_pct", "Non-ASCII / Regional Script in Name (%)"),
        ("non_ascii_addr_pct", "Non-ASCII / Regional Script in Address (%)"),
        ("leading_legal_pct", "Displaced Legal Affix as Prefix (%)"),
        ("trailing_legal_pct", "Standard Trailing Legal Suffix (%)"),
        ("punct_prefix_pct", "Extraneous Punctuation Prefix (`<<`, `--`) (%)"),
        ("all_caps_names_pct", "All-Uppercase Business Name (%)"),
        ("all_caps_addrs_pct", "All-Uppercase Address (%)"),
    ]:
        s1_val = all_stats[0]["patterns"][metric]
        s2_val = all_stats[1]["patterns"][metric]
        s3_val = all_stats[2]["patterns"][metric]
        lines.append(f"| **{desc}** | {s1_val:.2f}% | {s2_val:.2f}% | {s3_val:.2f}% | Detailed Below |")

    lines.extend([
        "",
        "### 6.1 Major Noise Types & Preprocessing Implications",
        "",
        "1. **Displaced Legal Affixes (Prefix vs Suffix)**:",
        "   - In standard corporate names, legal designations appear at the end: `Moncada Learning Center LLC`.",
        "   - In Sources 2 and 3, legal tokens are frequently prepended: `LLC Moncada Learning Center` or `Pvt. EFS Print Ventures Ltd.`.",
        "   - **Solution**: Normalization rules must identify legal tokens anywhere in the token string, standardize them (e.g. `pvt ltd` -> `private limited`), and compute name similarity with and without legal designations.",
        "",
        "2. **Address Component Permutations**:",
        "   - Standard US: `[Number] [Street], [City], [State] [Zip]`.",
        "   - Inverted noisy format: `[State], [City], [Number] [Street]` (e.g. `IA, Iowa City, 1064 Newton Rd, Unit 11` or `GREENSBORO, NC, 19 1/2 STARDUST TRAIL`).",
        "   - **Solution**: Address token set Jaccard / token sort ratio / bag-of-words / TF-IDF matching are critical because linear Levenshtein edit distance fails when comma-delimited blocks are swapped.",
        "",
        "3. **Transliteration & Multilingual Text**:",
        "   - Indian entities frequently have names or addresses written in Devanagari (e.g. `राम मार्केटिंग प्राइवेट लिमिटेड`, `मॉडर्न फाइनेंस`) or Kannada (ಕರ್ನಾಟಕ).",
        "   - **Solution**: Use Unicode normalization (NFKD) and consider character n-gram matching or multilingual dense sentence embeddings (`sentence-transformers`).",
        "",
        "4. **Punctuation Artifacts**:",
        "   - Noise tokens like `<<`, `--`, `//`, `>>` precede business names (e.g., `<< Team Ecole`, `-- Holloway Peak Inc Seafood`).",
        "   - **Solution**: Strip leading/trailing non-alphanumeric punctuation in pre-cleaning.",
        "",
        "---",
        "",
        "## 7. Strategic Recommendations for Pipeline Development",
        "",
        "1. **Blocking Strategy**:",
        "   - Block strictly by `country` (partition candidates within country).",
        "   - Generate multiple blocking keys: first significant word of business name, phonetic hash (Double Metaphone / Soundex), and postal code / city token.",
        "   - Complement lexical blocking with FAISS vector similarity search using sentence embeddings.",
        "2. **Feature Engineering**:",
        "   - Exact token intersection count, Jaccard similarity, Levenshtein ratio, Token Sort Ratio, and Token Set Ratio.",
        "   - Suffix-stripped name similarity.",
        "   - Address component matching (street number matching, state match, city match).",
        "   - Dense semantic embedding cosine similarity.",
        "3. **Classifier & Post-Processing**:",
        "   - Gradient boosted decision trees (LightGBM / XGBoost) trained to optimize F_0.5 score.",
        "   - Threshold tuning with F_0.5 emphasis on precision.",
    ])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {REPORT_PATH}")


def main():
    all_stats = []
    for src_id in [1, 2, 3]:
        stats = analyze_source(src_id)
        all_stats.append(stats)

    generate_figures(all_stats)
    generate_markdown_report(all_stats)
    print("EDA completed successfully!")


if __name__ == "__main__":
    main()
