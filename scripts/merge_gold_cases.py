#!/usr/bin/env python3
"""
Merge gold standard cases from clean_cases.json and mega_test_cases.json.

Creates data/calibration/gold_cases.json with deduplicated cases.
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

CLEAN_CASES_PATH = PROJECT_ROOT / "data" / "training" / "clean_cases.json"
MEGA_TEST_PATH = PROJECT_ROOT / "data" / "parser_validation" / "mega_test_cases.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "calibration" / "gold_cases.json"

SIMILARITY_THRESHOLD = 0.90


def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def normalize_case(case: dict, source_file: str) -> dict:
    """Normalize case to consistent format."""
    # Handle mega_test_cases format (nested ground_truth)
    if "ground_truth" in case:
        gt = case["ground_truth"]
        return {
            "case_id": case.get("id", case.get("case_id")),
            "source": case.get("source", "mega_test"),
            "source_file": source_file,
            "title": case.get("title", ""),
            "findings_text": case.get("report_text", ""),
            "case_type": case.get("case_type"),
            "severity": case.get("severity"),
            "scored_vai": gt.get("expected_vai_score"),
            "scored_magnifi": gt.get("expected_magnifi_score"),
            "ground_truth": gt
        }
    # Handle clean_cases format
    return {
        "case_id": case.get("case_id"),
        "source": case.get("source", "clean_cases"),
        "source_file": source_file,
        "title": case.get("title", ""),
        "findings_text": case.get("findings_text", ""),
        "case_type": case.get("case_type"),
        "severity": case.get("severity"),
        "scored_vai": case.get("scored_vai"),
        "scored_magnifi": case.get("scored_magnifi"),
    }


def is_duplicate(new_case: dict, existing_cases: list) -> bool:
    """Check if case is duplicate based on findings_text similarity."""
    new_text = new_case.get("findings_text", "")
    if not new_text:
        return False

    for existing in existing_cases:
        existing_text = existing.get("findings_text", "")
        if text_similarity(new_text, existing_text) > SIMILARITY_THRESHOLD:
            return True
    return False


def main():
    # Load clean_cases.json
    print(f"Loading {CLEAN_CASES_PATH}...")
    with open(CLEAN_CASES_PATH, "r", encoding="utf-8") as f:
        clean_cases = json.load(f)

    # Filter to only scored cases
    scored_clean = [c for c in clean_cases if c.get("scored_vai") is not None]
    print(f"  Found {len(scored_clean)} cases with ground truth (scored_vai)")

    # Load mega_test_cases.json
    print(f"Loading {MEGA_TEST_PATH}...")
    with open(MEGA_TEST_PATH, "r", encoding="utf-8") as f:
        mega_data = json.load(f)

    mega_cases = mega_data.get("test_cases", [])
    print(f"  Found {len(mega_cases)} test cases")

    # Normalize and merge
    gold_cases = []
    duplicates_removed = 0

    # Add scored clean cases first
    for case in scored_clean:
        normalized = normalize_case(case, "clean_cases.json")
        if not is_duplicate(normalized, gold_cases):
            gold_cases.append(normalized)
        else:
            duplicates_removed += 1

    clean_count = len(gold_cases)

    # Add mega test cases
    for case in mega_cases:
        normalized = normalize_case(case, "mega_test_cases.json")
        if not is_duplicate(normalized, gold_cases):
            gold_cases.append(normalized)
        else:
            duplicates_removed += 1

    mega_count = len(gold_cases) - clean_count

    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(gold_cases, f, indent=2, ensure_ascii=False)

    # Report
    print("\n" + "=" * 50)
    print("GOLD CASES MERGE REPORT")
    print("=" * 50)
    print(f"Cases from clean_cases:    {clean_count}")
    print(f"Cases from mega_test:      {mega_count}")
    print(f"Duplicates removed:        {duplicates_removed}")
    print("-" * 50)
    print(f"TOTAL GOLD CASES:          {len(gold_cases)}")
    print("=" * 50)
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
