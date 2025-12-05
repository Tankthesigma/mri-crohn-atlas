#!/usr/bin/env python3
"""
Generate Poster Statistics for MRI Report Parser
Creates formatted stats for ISEF poster presentation

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_poster_stats(results_path: str, output_path: str):
    """Generate poster-ready statistics from validation results"""

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    s = data['summary']
    results = data['results']

    # Calculate additional statistics
    total = s['total_cases']
    passed = s['passed_cases']

    # Feature accuracy rankings
    sorted_features = sorted(s['feature_accuracy'].items(), key=lambda x: -x[1])
    top_features = sorted_features[:3]
    bottom_features = sorted_features[-3:]

    # Difficulty analysis
    diff_acc = s['accuracy_by_difficulty']

    # Generate output
    output = []
    output.append("=" * 60)
    output.append("MRI REPORT PARSER - ISEF POSTER STATISTICS")
    output.append("=" * 60)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output.append("")

    output.append("-" * 40)
    output.append("HEADLINE METRICS")
    output.append("-" * 40)
    output.append(f"Test Cases Evaluated:     {total}")
    output.append(f"Overall Pass Rate:        {passed}/{total} ({s['pass_rate']*100:.0f}%)")
    output.append(f"Feature Extraction Acc:   {s['overall_feature_accuracy']*100:.1f}%")
    output.append(f"VAI Score MAE:            {s['vai_mae']:.2f} points")
    output.append(f"MAGNIFI-CD Score MAE:     {s['magnifi_mae']:.2f} points")
    output.append(f"VAI Within +/-2 Points:   {s['vai_within_2']*100:.0f}%")
    output.append(f"MAGNIFI Within +/-2:      {s['magnifi_within_2']*100:.0f}%")
    output.append(f"Avg Traceability Score:   {s['avg_traceability']*100:.0f}%")
    output.append("")

    output.append("-" * 40)
    output.append("FEATURE EXTRACTION ACCURACY")
    output.append("-" * 40)
    for feature, acc in sorted_features:
        bar = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
        output.append(f"{feature:<28} [{bar}] {acc*100:5.1f}%")
    output.append("")

    output.append("-" * 40)
    output.append("ACCURACY BY DIFFICULTY")
    output.append("-" * 40)
    for diff, acc in sorted(diff_acc.items()):
        count = sum(1 for r in results if r['difficulty'] == diff)
        passed_count = sum(1 for r in results if r['difficulty'] == diff and r['passed'])
        output.append(f"{diff.replace('_', ' ').title():<15} {passed_count}/{count} ({acc*100:.0f}%)")
    output.append("")

    output.append("-" * 40)
    output.append("KEY FINDINGS FOR POSTER")
    output.append("-" * 40)
    output.append("")
    output.append("1. NEURO-SYMBOLIC AI EXTRACTION:")
    output.append(f"   - Achieves {s['overall_feature_accuracy']*100:.0f}% accuracy across 8 MRI features")
    output.append(f"   - T2 hyperintensity detection: {s['feature_accuracy'].get('t2_hyperintensity', 0)*100:.0f}% accurate")
    output.append(f"   - Fistula classification: {s['feature_accuracy'].get('fistula_type', 0)*100:.0f}% accurate")
    output.append("")
    output.append("2. SCORE CALCULATION ACCURACY:")
    output.append(f"   - VAI scores within +/-2 points: {s['vai_within_2']*100:.0f}% of cases")
    output.append(f"   - MAGNIFI-CD within +/-2 points: {s['magnifi_within_2']*100:.0f}% of cases")
    output.append(f"   - Mean Absolute Error: {(s['vai_mae'] + s['magnifi_mae'])/2:.2f} points average")
    output.append("")
    output.append("3. EVIDENCE TRACEABILITY:")
    output.append(f"   - {s['avg_traceability']*100:.0f}% of key findings linked to source text")
    output.append("   - Enables clinical validation and trust")
    output.append("")
    output.append("4. ROBUSTNESS:")
    if 'edge_case' in diff_acc:
        output.append(f"   - Edge cases: {diff_acc['edge_case']*100:.0f}% pass rate")
    if 'complex' in diff_acc:
        output.append(f"   - Complex reports: {diff_acc['complex']*100:.0f}% pass rate")
    output.append("")

    output.append("-" * 40)
    output.append("POSTER-READY STATEMENTS")
    output.append("-" * 40)
    output.append("")

    # Generate poster-ready statements
    statements = [
        f"Our MRI Report Parser extracts {len(sorted_features)} key clinical features with {s['overall_feature_accuracy']*100:.0f}% accuracy.",
        f"The parser calculates both VAI and MAGNIFI-CD scores with an average error of only {(s['vai_mae'] + s['magnifi_mae'])/2:.1f} points.",
        f"{s['vai_within_2']*100:.0f}% of calculated scores fall within the clinically acceptable +/-2 point range.",
        f"Evidence traceability links {s['avg_traceability']*100:.0f}% of extracted features to their source text.",
        f"Validated on {total} ground truth test cases across {len(diff_acc)} difficulty levels.",
    ]

    for i, stmt in enumerate(statements, 1):
        output.append(f"{i}. {stmt}")
    output.append("")

    output.append("-" * 40)
    output.append("COMPARISON METRICS")
    output.append("-" * 40)
    output.append("For context with crosswalk formula validation:")
    output.append(f"  - Crosswalk Formula RMSE:  1.10 points (R^2 = 0.96)")
    output.append(f"  - Parser VAI MAE:          {s['vai_mae']:.2f} points")
    output.append(f"  - Parser MAGNIFI MAE:      {s['magnifi_mae']:.2f} points")
    output.append(f"  - Combined Error:          {(s['vai_mae'] + s['magnifi_mae'])/2:.2f} points")
    output.append("")

    output.append("-" * 40)
    output.append("TECHNICAL SPECIFICATIONS")
    output.append("-" * 40)
    output.append(f"Avg Extraction Time:       {s['avg_extraction_time']:.2f}s per report")
    output.append(f"Total Validation Time:     {s['total_time']:.1f}s")
    output.append(f"LLM Model:                 DeepSeek V3.2 (via OpenRouter)")
    output.append(f"Features Extracted:        8 clinical features")
    output.append(f"Scoring Systems:           VAI (0-22), MAGNIFI-CD (0-25)")
    output.append("")

    output.append("=" * 60)
    output.append("END OF POSTER STATISTICS")
    output.append("=" * 60)

    # Write output
    output_text = "\n".join(output)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(output_text)
    print(f"\nSaved to: {output_path}")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / "data" / "parser_tests" / "validation_results.json"
    output_path = project_root / "data" / "parser_tests" / "poster_stats.txt"

    if not results_path.exists():
        print(f"Error: Validation results not found at {results_path}")
        print("Run validate_parser.py first to generate results.")
        sys.exit(1)

    generate_poster_stats(results_path, output_path)


if __name__ == "__main__":
    main()
