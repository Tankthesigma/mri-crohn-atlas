#!/usr/bin/env python3
"""
Expanded Validation for MRI Report Parser
Runs validation on all collected real reports (v1 + v2)

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"
MODEL = "deepseek/deepseek-chat"

# Import from validate_parser
sys.path.insert(0, str(Path(__file__).parent))
from validate_parser import MRIReportParser, VAI_WEIGHTS, MAGNIFI_WEIGHTS


def run_expanded_validation():
    """Run validation on expanded report collection"""
    project_root = Path(__file__).parent.parent.parent

    # Load v2 cases
    v2_path = project_root / "data" / "real_reports" / "collected_reports_v2.json"
    output_path = project_root / "data" / "parser_tests" / "expanded_validation_results.json"

    with open(v2_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = data["new_cases"]
    parser = MRIReportParser(API_KEY, delay=1.5)

    results = []
    passed = 0
    failed = 0

    print("\n" + "="*70)
    print("EXPANDED VALIDATION SUITE (Phase 2)")
    print("="*70)
    print(f"\nTesting {len(test_cases)} new cases from collected_reports_v2.json\n")

    for i, case in enumerate(test_cases):
        case_id = case.get("id", f"case_{i+1}")
        source = case.get("source", "unknown")
        patient = case.get("patient", {})

        print(f"[{i+1}/{len(test_cases)}] {case_id} ({source})")
        print(f"    Patient: {patient.get('age', '?')}y {patient.get('gender', '?')}")

        # Extract features
        features, extraction_time = parser.extract_features(case["report_text"])

        if features is None:
            print(f"    ERROR: Failed to extract features")
            results.append({
                "case_id": case_id,
                "source": source,
                "passed": False,
                "error": "Extraction failed"
            })
            failed += 1
            continue

        # Calculate scores
        vai = parser.calculate_vai(features)
        magnifi = parser.calculate_magnifi(features)

        expected = case.get("ground_truth", {})
        expected_vai = expected.get("expected_vai_score", 0)
        expected_magnifi = expected.get("expected_magnifi_score", 0)

        vai_error = abs(vai - expected_vai)
        magnifi_error = abs(magnifi - expected_magnifi)

        # Determine pass/fail (within 3 points)
        score_pass = vai_error <= 3 and magnifi_error <= 3

        # Check key feature matches
        feature_issues = []

        # Fistula count
        ext_count = features.get("fistula_count", 0) or 0
        exp_count = expected.get("fistula_count", 0) or 0
        if abs(ext_count - exp_count) > 1:
            feature_issues.append(f"fistula_count: got {ext_count}, expected {exp_count}")

        # Collections/abscesses
        ext_abs = features.get("collections_abscesses", False)
        exp_abs = expected.get("collections_abscesses", False)
        if bool(ext_abs) != bool(exp_abs):
            feature_issues.append(f"collections: got {ext_abs}, expected {exp_abs}")

        # Seton detection - check treatment_status field
        ext_seton = features.get("treatment_status") == "seton_in_place"
        exp_seton = expected.get("seton_in_place", False)
        # Also pass if treatment_status is seton even if expected uses on_biologics with seton_in_place flag
        if bool(ext_seton) != bool(exp_seton):
            feature_issues.append(f"seton: got {ext_seton}, expected {exp_seton}")

        # Treatment status
        ext_treat = features.get("treatment_status", "unknown")
        exp_treat = expected.get("treatment_status", "unknown")
        treatment_match = ext_treat == exp_treat or exp_treat == "unknown"

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
        if features.get("treatment_status") == "seton_in_place":
            print(f"    Seton detected: Yes (treatment-aware scoring applied)")
        if feature_issues:
            print(f"    Issues: {', '.join(feature_issues)}")
        print(f"    Time: {extraction_time:.2f}s")

        results.append({
            "case_id": case_id,
            "source": source,
            "patient": patient,
            "passed": case_passed,
            "features_extracted": features,
            "vai_extracted": vai,
            "vai_expected": expected_vai,
            "vai_error": vai_error,
            "magnifi_extracted": magnifi,
            "magnifi_expected": expected_magnifi,
            "magnifi_error": magnifi_error,
            "treatment_extracted": ext_treat,
            "treatment_expected": exp_treat,
            "seton_detected": features.get("treatment_status") == "seton_in_place",
            "feature_issues": feature_issues,
            "extraction_time": extraction_time
        })

    # Summary
    print("\n" + "="*70)
    print("EXPANDED VALIDATION SUMMARY")
    print("="*70)
    print(f"\nTotal: {len(test_cases)}")
    print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")

    # Calculate aggregate metrics
    vai_errors = [r["vai_error"] for r in results if "vai_error" in r]
    magnifi_errors = [r["magnifi_error"] for r in results if "magnifi_error" in r]

    avg_vai_mae = sum(vai_errors) / len(vai_errors) if vai_errors else 0
    avg_magnifi_mae = sum(magnifi_errors) / len(magnifi_errors) if magnifi_errors else 0

    print(f"\nVAI MAE: {avg_vai_mae:.2f}")
    print(f"MAGNIFI MAE: {avg_magnifi_mae:.2f}")

    # By source breakdown
    print("\n" + "-"*50)
    print("RESULTS BY SOURCE")
    print("-"*50)
    sources = {}
    for r in results:
        src = r.get("source", "unknown")
        if src not in sources:
            sources[src] = {"total": 0, "passed": 0}
        sources[src]["total"] += 1
        if r.get("passed"):
            sources[src]["passed"] += 1

    for src, counts in sources.items():
        pct = counts["passed"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"{src}: {counts['passed']}/{counts['total']} ({pct:.1f}%)")

    # Seton detection accuracy
    seton_cases = [r for r in results if r.get("seton_detected") is not None]
    seton_correct = sum(1 for r in results
                       if r.get("seton_detected", False) ==
                          (r.get("features_extracted", {}).get("seton_in_place", False)))

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
        "by_source": sources,
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    run_expanded_validation()
