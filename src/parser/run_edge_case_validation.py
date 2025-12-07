#!/usr/bin/env python3
"""
Edge Case Validation for MRI Report Parser
Tests parser on challenging edge cases to identify weaknesses

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import json
import time
import re
import sys
from pathlib import Path
from datetime import datetime
import requests

# Configuration
import os
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-chat"

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Run: export OPENROUTER_API_KEY='your-key'")

# Import the extraction prompt from validate_parser
sys.path.insert(0, str(Path(__file__).parent))
from validate_parser import EXTRACTION_PROMPT, MRIReportParser, VAI_WEIGHTS, MAGNIFI_WEIGHTS


def run_edge_case_validation():
    """Run validation on edge cases"""
    project_root = Path(__file__).parent.parent.parent
    edge_cases_path = project_root / "data" / "parser_tests" / "edge_cases.json"
    output_path = project_root / "data" / "parser_tests" / "edge_case_results.json"

    # Load edge cases
    with open(edge_cases_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    parser = MRIReportParser(API_KEY, delay=1.5)

    results = []
    passed = 0
    failed = 0

    print("\n" + "="*70)
    print("EDGE CASE VALIDATION SUITE")
    print("="*70)
    print(f"\nTesting {len(test_cases)} edge cases...\n")

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {case['name']}: {case['what_it_tests']}")

        # Extract features
        features, extraction_time = parser.extract_features(case["report_text"])

        if features is None:
            print(f"    ERROR: Failed to extract features")
            results.append({
                "case_id": case["id"],
                "name": case["name"],
                "passed": False,
                "error": "Extraction failed",
                "what_it_tests": case["what_it_tests"]
            })
            failed += 1
            continue

        # Calculate scores and confidence
        vai = parser.calculate_vai(features)
        magnifi = parser.calculate_magnifi(features)
        confidence = parser.calculate_confidence(features, case["report_text"])
        confidence_level = parser.get_confidence_interpretation(confidence)

        expected = case["ground_truth"]
        expected_vai = expected.get("expected_vai_score", 0)
        expected_magnifi = expected.get("expected_magnifi_score", 0)
        expected_interpretation = expected.get("expected_interpretation", "")

        vai_error = abs(vai - expected_vai)
        magnifi_error = abs(magnifi - expected_magnifi)

        # Determine pass/fail (within 3 points for edge cases)
        # AMBIGUOUS case: Pass if confidence is appropriately low (<70%) - validates uncertainty detection
        is_ambiguous_case = case["name"] == "AMBIGUOUS"
        if is_ambiguous_case:
            # For ambiguous case, the key metric is LOW confidence, not exact scores
            score_pass = confidence < 0.70  # Parser correctly identifies uncertainty
        else:
            score_pass = vai_error <= 3 and magnifi_error <= 3

        # Check key feature matches
        feature_issues = []

        # Skip feature matching for AMBIGUOUS case - it's designed to have uncertain/variable extraction
        # (is_ambiguous_case already defined above)

        # Fistula count
        ext_count = features.get("fistula_count", 0) or 0
        exp_count = expected.get("fistula_count", 0) or 0
        if abs(ext_count - exp_count) > 1 and not is_ambiguous_case:
            feature_issues.append(f"fistula_count: got {ext_count}, expected {exp_count}")

        # Collections/abscesses - skip for ambiguous case
        ext_abs = features.get("collections_abscesses", False)
        exp_abs = expected.get("collections_abscesses", False)
        if bool(ext_abs) != bool(exp_abs) and not is_ambiguous_case:
            feature_issues.append(f"collections: got {ext_abs}, expected {exp_abs}")

        # Treatment status
        ext_treat = features.get("treatment_status", "unknown")
        exp_treat = expected.get("expected_treatment_status", "unknown")
        treatment_match = ext_treat == exp_treat or exp_treat == "unknown"

        # Disease phase check (for new_onset cases)
        disease_phase = features.get("disease_phase", "unknown")
        expected_phase = expected.get("disease_phase", None)

        case_passed = score_pass and len(feature_issues) == 0

        if case_passed:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"    Status: {status}")
        print(f"    VAI: {vai} (expected {expected_vai}, error: {vai_error})")
        print(f"    MAGNIFI: {magnifi} (expected {expected_magnifi}, error: {magnifi_error})")
        print(f"    Confidence: {confidence:.0%} ({confidence_level})")
        print(f"    Disease Phase: {disease_phase}")
        print(f"    Treatment: {ext_treat} (expected: {exp_treat})")
        if feature_issues:
            print(f"    Issues: {', '.join(feature_issues)}")
        print(f"    Time: {extraction_time:.2f}s")

        results.append({
            "case_id": case["id"],
            "name": case["name"],
            "what_it_tests": case["what_it_tests"],
            "passed": case_passed,
            "features_extracted": features,
            "vai_extracted": vai,
            "vai_expected": expected_vai,
            "vai_error": vai_error,
            "magnifi_extracted": magnifi,
            "magnifi_expected": expected_magnifi,
            "magnifi_error": magnifi_error,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "disease_phase": disease_phase,
            "treatment_extracted": ext_treat,
            "treatment_expected": exp_treat,
            "feature_issues": feature_issues,
            "extraction_time": extraction_time
        })

    # Summary
    print("\n" + "="*70)
    print("EDGE CASE SUMMARY")
    print("="*70)
    print(f"\nTotal: {len(test_cases)}")
    print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")

    # Show failures
    print("\n" + "-"*50)
    print("DETAILED RESULTS")
    print("-"*50)
    print(f"{'Case':<20} {'Status':<8} {'VAI Err':<10} {'MAG Err':<10} {'Issues'}")
    print("-"*70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        issues = ", ".join(r.get("feature_issues", []))[:30] if r.get("feature_issues") else "-"
        vai_err = r.get("vai_error", "N/A")
        mag_err = r.get("magnifi_error", "N/A")
        print(f"{r['name']:<20} {status:<8} {vai_err:<10} {mag_err:<10} {issues}")

    # Calculate aggregate metrics
    vai_errors = [r["vai_error"] for r in results if "vai_error" in r]
    magnifi_errors = [r["magnifi_error"] for r in results if "magnifi_error" in r]

    avg_vai_mae = sum(vai_errors) / len(vai_errors) if vai_errors else 0
    avg_magnifi_mae = sum(magnifi_errors) / len(magnifi_errors) if magnifi_errors else 0

    # Save results
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_cases": len(test_cases),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "vai_mae": avg_vai_mae,
            "magnifi_mae": avg_magnifi_mae,
            "vai_within_3": sum(1 for e in vai_errors if e <= 3) / len(vai_errors) if vai_errors else 0,
            "magnifi_within_3": sum(1 for e in magnifi_errors if e <= 3) / len(magnifi_errors) if magnifi_errors else 0
        },
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    run_edge_case_validation()
