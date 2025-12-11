#!/usr/bin/env python3
"""
Fix Ground Truth Data

Based on the LLM Judge audit, apply corrections to mislabeled cases:
- case_0024: scored_vai 9 → 16 (LABEL_WRONG, high confidence)
- case_0053: scored_vai 14 → 18 (LABEL_WRONG, high confidence)
- case_0049: scored_vai 14 → 18 (audit noted parser was correct)
"""

import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# We need to fix gold_cases.json (used for validation) and optionally master_cases.json
GOLD_CASES_PATH = PROJECT_ROOT / "data" / "calibration" / "gold_cases.json"
MASTER_CASES_PATH = Path.home() / "Desktop" / "Antigrav crohns trial" / "data" / "training" / "master_cases.json"

# Corrections to apply based on audit
CORRECTIONS = {
    "case_0024": {"scored_vai": 16, "reason": "LABEL_WRONG - text supports VAI=16 (high confidence)"},
    "case_0053": {"scored_vai": 18, "reason": "LABEL_WRONG - text supports VAI=18 (high confidence)"},
    "case_0049": {"scored_vai": 18, "reason": "Audit noted parser prediction of 18 was correct"},
}


def fix_gold_cases():
    """Fix the gold_cases.json file."""
    print("=" * 60)
    print("FIXING GOLD CASES")
    print("=" * 60)

    if not GOLD_CASES_PATH.exists():
        print(f"ERROR: {GOLD_CASES_PATH} not found")
        return False

    with open(GOLD_CASES_PATH, "r") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} gold cases")
    print()

    fixes_applied = 0
    for case in cases:
        case_id = case.get("case_id")
        if case_id in CORRECTIONS:
            old_vai = case.get("scored_vai")
            new_vai = CORRECTIONS[case_id]["scored_vai"]
            reason = CORRECTIONS[case_id]["reason"]

            print(f"  {case_id}:")
            print(f"    Old VAI: {old_vai}")
            print(f"    New VAI: {new_vai}")
            print(f"    Reason: {reason}")
            print()

            case["scored_vai"] = new_vai
            case["_correction_note"] = f"Fixed {datetime.now().isoformat()}: {reason}"
            fixes_applied += 1

    # Save back
    with open(GOLD_CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Applied {fixes_applied} corrections to {GOLD_CASES_PATH}")
    return True


def fix_master_cases():
    """Fix the master_cases.json file if it exists."""
    print()
    print("=" * 60)
    print("FIXING MASTER CASES")
    print("=" * 60)

    if not MASTER_CASES_PATH.exists():
        print(f"WARNING: {MASTER_CASES_PATH} not found - skipping")
        return False

    with open(MASTER_CASES_PATH, "r") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} master cases")
    print()

    fixes_applied = 0
    for case in cases:
        case_id = case.get("case_id")
        if case_id in CORRECTIONS:
            old_vai = case.get("scored_vai")
            if old_vai is None:
                continue  # No score to fix

            new_vai = CORRECTIONS[case_id]["scored_vai"]
            reason = CORRECTIONS[case_id]["reason"]

            print(f"  {case_id}:")
            print(f"    Old VAI: {old_vai}")
            print(f"    New VAI: {new_vai}")
            print(f"    Reason: {reason}")
            print()

            case["scored_vai"] = new_vai
            case["_correction_note"] = f"Fixed {datetime.now().isoformat()}: {reason}"
            fixes_applied += 1

    # Save back
    with open(MASTER_CASES_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Applied {fixes_applied} corrections to {MASTER_CASES_PATH}")
    return True


def main():
    print()
    print("GROUND TRUTH CORRECTION SCRIPT")
    print("Based on LLM Judge Audit Results")
    print()

    print("Corrections to apply:")
    for case_id, correction in CORRECTIONS.items():
        print(f"  - {case_id}: VAI → {correction['scored_vai']}")
    print()

    fix_gold_cases()
    fix_master_cases()

    print()
    print("=" * 60)
    print("DONE - Ground truth corrections applied")
    print("=" * 60)


if __name__ == "__main__":
    main()
