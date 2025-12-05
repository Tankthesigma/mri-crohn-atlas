#!/usr/bin/env python3
"""
Confidence Calibration Analysis for MRI Report Parser
Analyzes relationship between LLM confidence and prediction accuracy

Part of MRI-Crohn Atlas ISEF 2026 Project
Phase 4: Confidence calibration
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from validate_parser import MRIReportParser

# API Configuration - use environment variable
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def load_test_cases():
    """Load all test cases from various sources"""
    project_root = Path(__file__).parent.parent.parent
    test_cases = []

    # Load edge cases
    edge_path = project_root / "data" / "parser_tests" / "edge_cases.json"
    if edge_path.exists():
        with open(edge_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for case in data.get("test_cases", []):
                test_cases.append({
                    "id": case["id"],
                    "source": "edge_cases",
                    "report_text": case["report_text"],
                    "ground_truth": case["ground_truth"],
                    "difficulty": case.get("difficulty", "unknown")
                })

    # Load v2 cases
    v2_path = project_root / "data" / "real_reports" / "collected_reports_v2.json"
    if v2_path.exists():
        with open(v2_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for case in data.get("new_cases", []):
                test_cases.append({
                    "id": case["id"],
                    "source": case.get("source", "real_reports"),
                    "report_text": case["report_text"],
                    "ground_truth": case.get("ground_truth", {}),
                    "difficulty": "real_case"
                })

    return test_cases


def analyze_confidence_calibration():
    """Run confidence calibration analysis"""
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "data" / "parser_tests" / "confidence_calibration.json"

    test_cases = load_test_cases()
    parser = MRIReportParser(API_KEY, delay=1.5)

    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION ANALYSIS (Phase 4)")
    print("=" * 70)
    print(f"\nAnalyzing {len(test_cases)} cases for confidence-accuracy relationship\n")

    results = []
    confidence_bins = {
        "very_high": {"range": (0.9, 1.0), "cases": [], "correct": 0, "total": 0},
        "high": {"range": (0.75, 0.9), "cases": [], "correct": 0, "total": 0},
        "moderate": {"range": (0.5, 0.75), "cases": [], "correct": 0, "total": 0},
        "low": {"range": (0.0, 0.5), "cases": [], "correct": 0, "total": 0}
    }

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {case['id']} ({case['source']})")

        # Extract features with confidence
        features, extraction_time = parser.extract_features(case["report_text"])

        if features is None:
            print(f"    ERROR: Extraction failed")
            continue

        # Use our NEW calibrated confidence calculation (not raw LLM confidence)
        confidence = parser.calculate_confidence(features, case["report_text"])

        # Calculate scores
        vai = parser.calculate_vai(features)
        magnifi = parser.calculate_magnifi(features)

        expected = case.get("ground_truth", {})
        expected_vai = expected.get("expected_vai_score", 0)
        expected_magnifi = expected.get("expected_magnifi_score", 0)

        vai_error = abs(vai - expected_vai)
        magnifi_error = abs(magnifi - expected_magnifi)

        # Consider "correct" if within 3 points on both scores
        is_correct = vai_error <= 3 and magnifi_error <= 3

        # Bin by confidence
        for bin_name, bin_data in confidence_bins.items():
            low, high = bin_data["range"]
            if low <= confidence < high or (high == 1.0 and confidence == 1.0):
                bin_data["cases"].append({
                    "id": case["id"],
                    "confidence": confidence,
                    "vai_error": vai_error,
                    "magnifi_error": magnifi_error,
                    "is_correct": is_correct
                })
                bin_data["total"] += 1
                if is_correct:
                    bin_data["correct"] += 1
                break

        results.append({
            "case_id": case["id"],
            "source": case["source"],
            "difficulty": case.get("difficulty"),
            "confidence": confidence,
            "vai_extracted": vai,
            "vai_expected": expected_vai,
            "vai_error": vai_error,
            "magnifi_extracted": magnifi,
            "magnifi_expected": expected_magnifi,
            "magnifi_error": magnifi_error,
            "is_correct": is_correct,
            "extraction_time": extraction_time
        })

        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"    Confidence: {confidence:.2f}, VAI err: {vai_error}, MAG err: {magnifi_error} -> {status}")

    # Calibration analysis
    print("\n" + "-" * 50)
    print("CONFIDENCE CALIBRATION RESULTS")
    print("-" * 50)

    calibration_data = {}
    print(f"\n{'Confidence Bin':<20} {'Accuracy':<15} {'Expected':<15} {'Calibration'}")
    print("-" * 70)

    for bin_name, bin_data in confidence_bins.items():
        if bin_data["total"] > 0:
            accuracy = bin_data["correct"] / bin_data["total"]
            expected_acc = (bin_data["range"][0] + bin_data["range"][1]) / 2
            calibration_error = accuracy - expected_acc

            calibration_data[bin_name] = {
                "range": bin_data["range"],
                "n_cases": bin_data["total"],
                "n_correct": bin_data["correct"],
                "accuracy": accuracy,
                "expected_accuracy": expected_acc,
                "calibration_error": calibration_error
            }

            calib_status = "Well-calibrated" if abs(calibration_error) < 0.15 else (
                "Overconfident" if calibration_error < 0 else "Underconfident"
            )

            print(f"{bin_name:<20} {accuracy*100:.1f}% ({bin_data['correct']}/{bin_data['total']})  "
                  f"{expected_acc*100:.1f}%          {calib_status}")
        else:
            print(f"{bin_name:<20} No cases")

    # Overall calibration metrics
    print("\n" + "-" * 50)
    print("OVERALL CALIBRATION METRICS")
    print("-" * 50)

    all_confidences = [r["confidence"] for r in results]
    all_correct = [1 if r["is_correct"] else 0 for r in results]
    all_vai_errors = [r["vai_error"] for r in results]
    all_magnifi_errors = [r["magnifi_error"] for r in results]

    if len(results) > 0:
        mean_confidence = np.mean(all_confidences)
        overall_accuracy = np.mean(all_correct)
        calibration_gap = mean_confidence - overall_accuracy

        print(f"\n  Mean Confidence: {mean_confidence:.2f}")
        print(f"  Overall Accuracy: {overall_accuracy*100:.1f}%")
        print(f"  Calibration Gap: {calibration_gap:+.2f}")

        if calibration_gap > 0.15:
            print(f"  STATUS: OVERCONFIDENT (should reduce confidence by ~{calibration_gap:.2f})")
        elif calibration_gap < -0.15:
            print(f"  STATUS: UNDERCONFIDENT (should increase confidence by ~{abs(calibration_gap):.2f})")
        else:
            print(f"  STATUS: WELL-CALIBRATED")

        # Correlation between confidence and error
        if len(all_confidences) > 3:
            from scipy.stats import pearsonr, spearmanr

            # Inverse: high confidence should correlate with low error
            avg_errors = [(v + m) / 2 for v, m in zip(all_vai_errors, all_magnifi_errors)]

            try:
                corr_pearson, p_pearson = pearsonr(all_confidences, avg_errors)
                corr_spearman, p_spearman = spearmanr(all_confidences, avg_errors)

                print(f"\n  Confidence-Error Correlation:")
                print(f"    Pearson r: {corr_pearson:.3f} (p={p_pearson:.3f})")
                print(f"    Spearman rho: {corr_spearman:.3f} (p={p_spearman:.3f})")

                if corr_pearson < -0.2:
                    print(f"    INTERPRETATION: Confidence is informative (higher = lower error)")
                elif corr_pearson > 0.2:
                    print(f"    INTERPRETATION: Confidence is INVERSELY predictive (problem!)")
                else:
                    print(f"    INTERPRETATION: Confidence has weak relationship to accuracy")
            except Exception as e:
                print(f"    Could not compute correlation: {e}")

    # By difficulty analysis
    print("\n" + "-" * 50)
    print("ACCURACY BY DIFFICULTY")
    print("-" * 50)

    difficulty_stats = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"correct": 0, "total": 0, "errors": []}
        difficulty_stats[diff]["total"] += 1
        if r["is_correct"]:
            difficulty_stats[diff]["correct"] += 1
        difficulty_stats[diff]["errors"].append((r["vai_error"] + r["magnifi_error"]) / 2)

    for diff, stats in sorted(difficulty_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_err = np.mean(stats["errors"]) if stats["errors"] else 0
        print(f"  {diff}: {acc:.1f}% accuracy, avg error: {avg_err:.1f} (n={stats['total']})")

    # Save results
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_cases": len(results),
            "mean_confidence": float(mean_confidence) if results else None,
            "overall_accuracy": float(overall_accuracy) if results else None,
            "calibration_gap": float(calibration_gap) if results else None,
            "vai_mae": float(np.mean(all_vai_errors)) if all_vai_errors else None,
            "magnifi_mae": float(np.mean(all_magnifi_errors)) if all_magnifi_errors else None
        },
        "calibration_by_bin": calibration_data,
        "difficulty_breakdown": {
            k: {"accuracy": v["correct"]/v["total"] if v["total"] > 0 else 0, "n": v["total"]}
            for k, v in difficulty_stats.items()
        },
        "detailed_results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    analyze_confidence_calibration()
