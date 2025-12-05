#!/usr/bin/env python3
"""
Adversarial Test Runner for MRI Report Parser
Tests parser robustness against edge cases, OCR errors, and unusual inputs

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from validate_parser import MRIReportParser

# API Configuration - use environment variable
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def load_adversarial_cases():
    """Load adversarial test cases"""
    project_root = Path(__file__).parent.parent.parent
    test_path = project_root / "data" / "parser_tests" / "adversarial_cases.json"

    if not test_path.exists():
        print(f"ERROR: Test file not found at {test_path}")
        return []

    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("test_cases", [])


def run_adversarial_validation():
    """Run adversarial validation suite"""
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "parser_tests" / "adversarial_results.json"

    test_cases = load_adversarial_cases()
    if not test_cases:
        print("No test cases found!")
        return

    parser = MRIReportParser(API_KEY, delay=2.0)  # Longer delay for complex cases

    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST SUITE - Parser Robustness Testing")
    print("=" * 70)
    print(f"\nRunning {len(test_cases)} adversarial test cases\n")

    results = []
    passed = 0
    failed = 0

    for i, case in enumerate(test_cases):
        case_id = case["id"]
        case_name = case["name"]

        print(f"\n[{i+1}/{len(test_cases)}] {case_id}: {case_name}")
        print(f"    Testing: {case['what_it_tests'][:60]}...")

        try:
            # Extract features
            features, extraction_time = parser.extract_features(case["report_text"])

            if features is None:
                print(f"    ERROR: Extraction failed completely")
                failed += 1
                results.append({
                    "case_id": case_id,
                    "name": case_name,
                    "passed": False,
                    "error": "Extraction returned None",
                    "extraction_time": extraction_time
                })
                continue

            # Calculate scores
            vai = parser.calculate_vai(features)
            magnifi = parser.calculate_magnifi(features)
            confidence = parser.calculate_confidence(features, case["report_text"])

            # Get expected values
            expected = case.get("ground_truth", {})
            expected_vai = expected.get("expected_vai_score", 0)
            expected_magnifi = expected.get("expected_magnifi_score", 0)

            # Calculate errors
            vai_error = abs(vai - expected_vai)
            magnifi_error = abs(magnifi - expected_magnifi)

            # Pass criteria: within 3 points on both scores OR low confidence correctly flags uncertainty
            is_within_tolerance = vai_error <= 3 and magnifi_error <= 3

            # For contradiction/ambiguous cases, low confidence is a pass
            is_uncertainty_case = case_name in ["CONTRADICTORY", "AMBIGUOUS", "DIFFERENTIAL_HEAVY", "GARBLED_SECTIONS"]
            correctly_flagged_uncertainty = is_uncertainty_case and confidence < 0.55

            case_passed = is_within_tolerance or correctly_flagged_uncertainty

            if case_passed:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"    VAI: {vai} (expected {expected_vai}, error {vai_error})")
            print(f"    MAGNIFI: {magnifi} (expected {expected_magnifi}, error {magnifi_error})")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Status: {status}")

            results.append({
                "case_id": case_id,
                "name": case_name,
                "what_it_tests": case["what_it_tests"],
                "passed": case_passed,
                "vai_extracted": vai,
                "vai_expected": expected_vai,
                "vai_error": vai_error,
                "magnifi_extracted": magnifi,
                "magnifi_expected": expected_magnifi,
                "magnifi_error": magnifi_error,
                "confidence": confidence,
                "is_uncertainty_case": is_uncertainty_case,
                "correctly_flagged": correctly_flagged_uncertainty if is_uncertainty_case else None,
                "features_extracted": {
                    "fistula_count": features.get("fistula_count"),
                    "fistula_type": features.get("fistula_type"),
                    "t2_hyperintensity": features.get("t2_hyperintensity"),
                    "collections_abscesses": features.get("collections_abscesses")
                },
                "extraction_time": extraction_time
            })

        except Exception as e:
            print(f"    EXCEPTION: {str(e)}")
            failed += 1
            results.append({
                "case_id": case_id,
                "name": case_name,
                "passed": False,
                "error": str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST SUMMARY")
    print("=" * 70)

    total = len(test_cases)
    pass_rate = passed / total * 100 if total > 0 else 0

    print(f"\n  Total Cases: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Pass Rate: {pass_rate:.1f}%")

    # Save results
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate / 100
        },
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    run_adversarial_validation()
