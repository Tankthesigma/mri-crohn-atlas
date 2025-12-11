#!/usr/bin/env python3
"""
Clean master_cases.json by removing non-clinical cases.

Removes cases matching:
- Non-clinical keywords (reviews, animal studies, etc.)
- Short/missing findings text
- No MRI findings

Outputs clean_cases.json with detailed removal report.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "data" / "training" / "master_cases.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "clean_cases.json"

# Trash keywords (case-insensitive)
TRASH_KEYWORDS = [
    "bibliometric",
    "meta-analysis",
    "systematic review",
    "social media",
    "facebook",
    "instagram",
    "rat model",
    "murine",
    "mice",
    "rabbit",
    "porcine",
    "canine",
    "calf",
    "bovine",
    "animal model",
    "questionnaire",
    "survey",
    "cadaver",
    "cadaveric",
    "narrative review",
    "in vitro",
    "guideline",
    "protocol",
    "editorial",
    "consensus",
]

# Compile regex for faster matching
TRASH_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in TRASH_KEYWORDS),
    re.IGNORECASE
)


def contains_trash_keyword(text: str) -> Optional[str]:
    """Check if text contains any trash keyword. Returns matched keyword or None."""
    if not text:
        return None
    match = TRASH_PATTERN.search(text)
    if match:
        return match.group().lower()
    return None


def clean_cases(cases: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Clean cases and return (clean_cases, removal_stats).

    removal_stats contains:
    - keyword_counts: dict mapping keyword -> count
    - short_text_count: int
    - null_text_count: int
    - no_mri_count: int
    - removed_ids: list of removed case_ids
    """
    clean = []
    removal_stats = {
        "keyword_counts": defaultdict(int),
        "short_text_count": 0,
        "null_text_count": 0,
        "no_mri_count": 0,
        "removed_ids": [],
    }

    for case in cases:
        case_id = case.get("case_id", case.get("id", "unknown"))
        title = case.get("title", "") or ""
        findings = case.get("findings_text", "") or ""
        has_mri = case.get("has_mri_findings", True)  # Default True if missing

        removed = False

        # Check 1: Null or short findings_text
        if findings is None or findings == "":
            removal_stats["null_text_count"] += 1
            removal_stats["removed_ids"].append(case_id)
            removed = True
        elif len(findings.strip()) < 50:
            removal_stats["short_text_count"] += 1
            removal_stats["removed_ids"].append(case_id)
            removed = True

        # Check 2: Trash keywords in title or findings
        if not removed:
            keyword_in_title = contains_trash_keyword(title)
            keyword_in_findings = contains_trash_keyword(findings)
            matched_keyword = keyword_in_title or keyword_in_findings

            if matched_keyword:
                removal_stats["keyword_counts"][matched_keyword] += 1
                removal_stats["removed_ids"].append(case_id)
                removed = True

        # Check 3: has_mri_findings is False
        if not removed and has_mri is False:
            removal_stats["no_mri_count"] += 1
            removal_stats["removed_ids"].append(case_id)
            removed = True

        if not removed:
            clean.append(case)

    return clean, removal_stats


def count_with_ground_truth(cases: List[Dict]) -> int:
    """Count cases that have scored_vai (ground truth)."""
    count = 0
    for case in cases:
        scored_vai = case.get("scored_vai")
        if scored_vai is not None:
            count += 1
    return count


def print_report(original_count: int, clean_cases: List[Dict], removal_stats: Dict):
    """Print detailed cleaning report."""
    clean_count = len(clean_cases)
    removed_count = original_count - clean_count
    ground_truth_count = count_with_ground_truth(clean_cases)

    print("=" * 60)
    print("MASTER CASES CLEANING REPORT")
    print("=" * 60)
    print()
    print(f"Original count:      {original_count:,}")
    print(f"Final clean count:   {clean_count:,}")
    print(f"Removed count:       {removed_count:,} ({removed_count/original_count*100:.1f}%)" if original_count > 0 else "Removed count:       0")
    print()
    print(f"Clean cases with ground truth (scored_vai): {ground_truth_count:,}")
    print()

    print("-" * 60)
    print("REMOVAL BREAKDOWN")
    print("-" * 60)
    print()

    # Text issues
    print(f"Null/empty findings_text:  {removal_stats['null_text_count']:,}")
    print(f"Short findings (<50 char): {removal_stats['short_text_count']:,}")
    print(f"No MRI findings:           {removal_stats['no_mri_count']:,}")
    print()

    # Keyword breakdown
    keyword_counts = removal_stats["keyword_counts"]
    if keyword_counts:
        print("Keyword removals:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
            print(f"  - '{keyword}': {count:,}")
        print()
        total_keyword = sum(keyword_counts.values())
        print(f"Total keyword removals: {total_keyword:,}")
    else:
        print("Keyword removals: 0")
    print()

    # Sample of removed case IDs
    print("-" * 60)
    print("SAMPLE REMOVED CASE IDS (first 10)")
    print("-" * 60)
    removed_ids = removal_stats["removed_ids"][:10]
    if removed_ids:
        for case_id in removed_ids:
            print(f"  - {case_id}")
    else:
        print("  (none removed)")
    print()

    print("=" * 60)
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 60)


def main():
    # Check input file exists
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print()
        print("Please ensure data/training/master_cases.json exists.")
        return 1

    # Load input
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Handle if it's wrapped in a dict
    if isinstance(cases, dict):
        if "cases" in cases:
            cases = cases["cases"]
        else:
            print("ERROR: Expected list or dict with 'cases' key")
            return 1

    original_count = len(cases)
    print(f"Loaded {original_count:,} cases")
    print()

    # Clean
    print("Cleaning cases...")
    clean, removal_stats = clean_cases(cases)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save output
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    # Print report
    print()
    print_report(original_count, clean, removal_stats)

    return 0


if __name__ == "__main__":
    exit(main())
