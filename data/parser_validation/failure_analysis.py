#!/usr/bin/env python3
"""
Analyze failures from real API validation.
Categorize errors and identify root causes.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_FILE = Path(__file__).parent / "real_validation_results.json"
CASES_FILE = Path(__file__).parent / "mega_test_cases.json"
OUTPUT_FILE = Path(__file__).parent / "failure_analysis.json"


def load_data():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    with open(CASES_FILE) as f:
        cases_data = json.load(f)

    # Create case lookup
    case_lookup = {}
    for case in cases_data.get("test_cases", []):
        case_lookup[case["id"]] = case

    return results, case_lookup


def categorize_failure(result, case_data):
    """Categorize the likely cause of the failure"""
    vai_err = result.get("vai_error", 0)
    mag_err = result.get("magnifi_error", 0)
    case_type = result.get("case_type", "")
    source = result.get("source", "")
    report = case_data.get("report_text", "")[:200] if case_data else ""

    categories = []

    # Check for ambiguous language
    ambiguous_keywords = ["possible", "uncertain", "equivocal", "cannot exclude",
                          "may represent", "suspicious", "likely", "probable",
                          "subtle", "questionable", "indeterminate"]
    if any(kw in report.lower() for kw in ambiguous_keywords):
        categories.append("ambiguous_findings")

    # Check for complex anatomy
    complex_keywords = ["horseshoe", "multiple", "branching", "complex",
                        "bilateral", "interconnecting", "supralevator"]
    if any(kw in report.lower() for kw in complex_keywords):
        categories.append("complex_anatomy")

    # Check if synthetic
    if "synth" in source.lower() or "synthetic" in source.lower():
        categories.append("synthetic_case")

    # Check for severe cases (harder to score precisely)
    if result.get("severity", "").lower() == "severe":
        categories.append("severe_complexity")

    # Check for scoring boundary issues (off by 1-2 in same severity category)
    if abs(vai_err) <= 3 and abs(mag_err) <= 3:
        categories.append("boundary_disagreement")

    # Check for direction of error
    if vai_err > 0:
        categories.append("parser_overestimate")
    elif vai_err < 0:
        categories.append("parser_underestimate")

    # Multi-fistula
    if "multiple" in report.lower() or "fistulas" in report.lower():
        categories.append("multi_fistula")

    if not categories:
        categories.append("unknown")

    return categories


def main():
    print("=" * 70)
    print("FAILURE ANALYSIS - Real API Validation")
    print("=" * 70)

    results, case_lookup = load_data()

    # Find failures (|VAI error| > 2 OR |MAGNIFI error| > 3)
    vai_failures = []
    mag_failures = []
    combined_failures = []

    for r in results:
        vai_err = abs(r.get("vai_error", 0) or 0)
        mag_err = abs(r.get("magnifi_error", 0) or 0)

        if vai_err > 2 or mag_err > 3:
            combined_failures.append(r)
        if vai_err > 2:
            vai_failures.append(r)
        if mag_err > 3:
            mag_failures.append(r)

    print(f"\nTotal cases: {len(results)}")
    print(f"VAI failures (|error| > 2): {len(vai_failures)} ({len(vai_failures)/len(results)*100:.1f}%)")
    print(f"MAGNIFI failures (|error| > 3): {len(mag_failures)} ({len(mag_failures)/len(results)*100:.1f}%)")
    print(f"Combined failures: {len(combined_failures)} ({len(combined_failures)/len(results)*100:.1f}%)")

    # Detailed failure analysis
    failure_details = []
    category_counts = defaultdict(int)

    print("\n" + "=" * 70)
    print("FAILURE DETAILS")
    print("=" * 70)

    for r in sorted(combined_failures, key=lambda x: abs(x.get("vai_error", 0)), reverse=True):
        case_id = r["case_id"]
        case_data = case_lookup.get(case_id, {})
        report_text = case_data.get("report_text", "N/A")[:150]

        categories = categorize_failure(r, case_data)
        for cat in categories:
            category_counts[cat] += 1

        detail = {
            "case_id": case_id,
            "source": r.get("source", "unknown"),
            "case_type": r.get("case_type", "unknown"),
            "severity": r.get("severity", "unknown"),
            "expected_vai": r.get("expected_vai"),
            "predicted_vai": r.get("predicted_vai"),
            "vai_error": r.get("vai_error"),
            "expected_magnifi": r.get("expected_magnifi"),
            "predicted_magnifi": r.get("predicted_magnifi"),
            "magnifi_error": r.get("magnifi_error"),
            "categories": categories,
            "findings_excerpt": report_text
        }
        failure_details.append(detail)

        print(f"\n{case_id}")
        print(f"  Source: {r.get('source')} | Type: {r.get('case_type')} | Severity: {r.get('severity')}")
        print(f"  VAI: {r.get('expected_vai')} -> {r.get('predicted_vai')} (error: {r.get('vai_error'):+d})")
        print(f"  MAGNIFI: {r.get('expected_magnifi')} -> {r.get('predicted_magnifi')} (error: {r.get('magnifi_error'):+d})")
        print(f"  Categories: {', '.join(categories)}")
        print(f"  Report: {report_text}...")

    # Category summary
    print("\n" + "=" * 70)
    print("FAILURE CATEGORY SUMMARY")
    print("=" * 70)

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} cases")

    # Synthetic vs Real comparison
    print("\n" + "=" * 70)
    print("SYNTHETIC VS REAL PERFORMANCE")
    print("=" * 70)

    synthetic_results = [r for r in results if "synth" in r.get("source", "").lower()]
    real_results = [r for r in results if "synth" not in r.get("source", "").lower()]

    def calc_accuracy(res_list, threshold):
        vai_acc = sum(1 for r in res_list if abs(r.get("vai_error", 0) or 0) <= threshold) / len(res_list) if res_list else 0
        return vai_acc

    print(f"\nReal cases (n={len(real_results)}):")
    print(f"  VAI Accuracy (±2): {calc_accuracy(real_results, 2)*100:.1f}%")
    print(f"  VAI Accuracy (±3): {calc_accuracy(real_results, 3)*100:.1f}%")

    real_vai_errors = [abs(r.get("vai_error", 0) or 0) for r in real_results]
    print(f"  VAI MAE: {sum(real_vai_errors)/len(real_vai_errors):.2f}")

    print(f"\nSynthetic cases (n={len(synthetic_results)}):")
    print(f"  VAI Accuracy (±2): {calc_accuracy(synthetic_results, 2)*100:.1f}%")
    print(f"  VAI Accuracy (±3): {calc_accuracy(synthetic_results, 3)*100:.1f}%")

    synth_vai_errors = [abs(r.get("vai_error", 0) or 0) for r in synthetic_results]
    print(f"  VAI MAE: {sum(synth_vai_errors)/len(synth_vai_errors):.2f}")

    # By source breakdown
    print("\n" + "=" * 70)
    print("BREAKDOWN BY SOURCE")
    print("=" * 70)

    sources = defaultdict(list)
    for r in results:
        sources[r.get("source", "unknown")].append(r)

    source_stats = {}
    for source, res_list in sorted(sources.items()):
        vai_errs = [abs(r.get("vai_error", 0) or 0) for r in res_list]
        acc_2 = sum(1 for e in vai_errs if e <= 2) / len(vai_errs) * 100
        acc_3 = sum(1 for e in vai_errs if e <= 3) / len(vai_errs) * 100
        mae = sum(vai_errs) / len(vai_errs)

        source_stats[source] = {
            "n": len(res_list),
            "vai_acc_2": acc_2,
            "vai_acc_3": acc_3,
            "vai_mae": mae
        }

        print(f"  {source} (n={len(res_list)}): Acc(±2)={acc_2:.1f}%, Acc(±3)={acc_3:.1f}%, MAE={mae:.2f}")

    # Largest errors
    print("\n" + "=" * 70)
    print("LARGEST ERRORS (|VAI error| >= 4)")
    print("=" * 70)

    large_errors = [r for r in results if abs(r.get("vai_error", 0) or 0) >= 4]
    for r in sorted(large_errors, key=lambda x: abs(x.get("vai_error", 0)), reverse=True):
        print(f"  {r['case_id']}: VAI {r['expected_vai']} -> {r['predicted_vai']} (error: {r['vai_error']:+d})")
        print(f"    Type: {r['case_type']} | Source: {r['source']}")

    # Save analysis
    analysis = {
        "summary": {
            "total_cases": len(results),
            "vai_failures_count": len(vai_failures),
            "vai_failure_rate": len(vai_failures) / len(results),
            "magnifi_failures_count": len(mag_failures),
            "magnifi_failure_rate": len(mag_failures) / len(results),
            "combined_failures_count": len(combined_failures),
            "combined_failure_rate": len(combined_failures) / len(results)
        },
        "category_counts": dict(category_counts),
        "synthetic_vs_real": {
            "real": {
                "n": len(real_results),
                "vai_acc_2": calc_accuracy(real_results, 2),
                "vai_acc_3": calc_accuracy(real_results, 3),
                "vai_mae": sum(real_vai_errors) / len(real_vai_errors) if real_vai_errors else 0
            },
            "synthetic": {
                "n": len(synthetic_results),
                "vai_acc_2": calc_accuracy(synthetic_results, 2),
                "vai_acc_3": calc_accuracy(synthetic_results, 3),
                "vai_mae": sum(synth_vai_errors) / len(synth_vai_errors) if synth_vai_errors else 0
            }
        },
        "source_breakdown": source_stats,
        "failure_details": failure_details
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n\nAnalysis saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
