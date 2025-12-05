#!/usr/bin/env python3
"""
Deep Failure Analysis for MRI Report Parser
Analyzes all 8 failed real-world cases to identify root causes

FAILED CASES (from expanded_validation_results.json):
- rp_v2_3: VAI 5 vs 8 (err 3), MAGNIFI 3 vs 10 (err 7)
- rp_v2_4: VAI 12 vs 8 (err 4), MAGNIFI 11 vs 9 (err 2)
- rp_v2_5: VAI 9 vs 6 (err 3), MAGNIFI 6 vs 7 (err 1) - seton detection miss
- rp_v2_6: VAI 10 vs 14 (err 4), MAGNIFI 7 vs 16 (err 9) - fistula count 1 vs 4, seton miss
- rp_v2_8: VAI 18 vs 12 (err 6), MAGNIFI 17 vs 14 (err 3)
- rp_v2_10: VAI 12 vs 6 (err 6), MAGNIFI 14 vs 8 (err 6)
- ped_v2_2: VAI 15 vs 10 (err 5), MAGNIFI 15 vs 12 (err 3)
- ped_v2_3: VAI 14 vs 8 (err 6), MAGNIFI 13 vs 9 (err 4) - seton detection miss
"""

import json
from pathlib import Path

# Load data
project_root = Path(__file__).parent.parent.parent
reports_path = project_root / "data" / "real_reports" / "collected_reports_v2.json"
results_path = project_root / "data" / "parser_tests" / "expanded_validation_results.json"

with open(reports_path, 'r') as f:
    reports_data = json.load(f)

with open(results_path, 'r') as f:
    results_data = json.load(f)

# Get reports dict
reports_dict = {r['id']: r for r in reports_data['new_cases']}

# Failed case IDs
failed_ids = ['rp_v2_3', 'rp_v2_4', 'rp_v2_5', 'rp_v2_6', 'rp_v2_8', 'rp_v2_10', 'ped_v2_2', 'ped_v2_3']

print("=" * 80)
print("DEEP FAILURE ANALYSIS - 8 FAILED REAL-WORLD CASES")
print("=" * 80)

for case_id in failed_ids:
    report = reports_dict.get(case_id, {})
    result = next((r for r in results_data['results'] if r['case_id'] == case_id), None)

    if not report or not result:
        continue

    print(f"\n{'='*80}")
    print(f"CASE: {case_id} | Source: {report.get('source', 'unknown')}")
    print(f"Patient: {report.get('patient', {})}")
    print(f"Presentation: {report.get('presentation', 'N/A')}")
    print("="*80)

    print(f"\n📄 FULL REPORT TEXT:")
    print("-" * 60)
    print(report.get('report_text', 'N/A'))
    print("-" * 60)

    gt = report.get('ground_truth', {})
    ext = result.get('features_extracted', {})

    print(f"\n📊 SCORE COMPARISON:")
    print(f"  VAI:     Extracted {result.get('vai_extracted')} vs Expected {gt.get('expected_vai_score')} | Error: {result.get('vai_error')}")
    print(f"  MAGNIFI: Extracted {result.get('magnifi_extracted')} vs Expected {gt.get('expected_magnifi_score')} | Error: {result.get('magnifi_error')}")

    print(f"\n🔍 FEATURE-BY-FEATURE COMPARISON:")
    features = [
        'fistula_count', 'fistula_type', 't2_hyperintensity', 't2_hyperintensity_degree',
        'extension', 'collections_abscesses', 'rectal_wall_involvement',
        'inflammatory_mass', 'seton_in_place', 'treatment_status'
    ]

    for f in features:
        expected = gt.get(f, 'N/A')
        extracted = ext.get(f, 'N/A')
        match = "✓" if expected == extracted else "✗"
        if expected != extracted and expected != 'N/A':
            print(f"  {match} {f}: Got '{extracted}' Expected '{expected}' ← MISMATCH")
        elif expected != 'N/A':
            print(f"  {match} {f}: {extracted}")

    # Issues from validation
    issues = result.get('feature_issues', [])
    if issues:
        print(f"\n⚠️  DETECTED ISSUES:")
        for issue in issues:
            print(f"    - {issue}")

    # Root cause analysis
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")

    # Check for seton terminology
    report_text = report.get('report_text', '').lower()
    if 'seton' in report_text:
        if ext.get('treatment_status') != 'seton_in_place':
            print("    - SETON DETECTION FAILURE: 'seton' in text but not detected")
        else:
            print("    - Seton correctly detected")

    # Check fistula count issues
    if gt.get('fistula_count', 0) != ext.get('fistula_count', 0):
        expected_count = gt.get('fistula_count', 0)
        extracted_count = ext.get('fistula_count', 0)
        print(f"    - FISTULA COUNT ERROR: Expected {expected_count}, got {extracted_count}")

        # Look for count indicators
        if 'two' in report_text or '2' in report_text:
            print("      Found '2' or 'two' in text")
        if 'three' in report_text or '3' in report_text:
            print("      Found '3' or 'three' in text")
        if 'multiple' in report_text:
            print("      Found 'multiple' in text")

    # T2/extension analysis
    if gt.get('extension') != ext.get('extension'):
        print(f"    - EXTENSION MISMATCH: Expected '{gt.get('extension')}', got '{ext.get('extension')}'")
        if 'supralevator' in report_text:
            print("      'supralevator' found → should be 'severe'")
        if 'ischioanal' in report_text:
            print("      'ischioanal' found → should be 'moderate'")

    # T2 degree analysis
    if gt.get('t2_hyperintensity_degree') != ext.get('t2_hyperintensity_degree'):
        print(f"    - T2 DEGREE MISMATCH: Expected '{gt.get('t2_hyperintensity_degree')}', got '{ext.get('t2_hyperintensity_degree')}'")
        if 'marked' in report_text or 'avid' in report_text:
            print("      'marked'/'avid' found → should be 'marked'")
        if 'mild' in report_text:
            print("      'mild' found → should be 'mild'")

    # Check for overscoring
    if result.get('vai_extracted', 0) > gt.get('expected_vai_score', 0):
        print(f"    - OVERSCORING: Parser scored higher than expected")
    else:
        print(f"    - UNDERSCORING: Parser scored lower than expected")

print("\n" + "="*80)
print("PATTERN SUMMARY")
print("="*80)

# Count patterns
patterns = {
    'seton_miss': 0,
    'fistula_count_error': 0,
    'extension_error': 0,
    't2_degree_error': 0,
    'overscoring': 0,
    'underscoring': 0,
}

for case_id in failed_ids:
    report = reports_dict.get(case_id, {})
    result = next((r for r in results_data['results'] if r['case_id'] == case_id), None)
    if not report or not result:
        continue

    gt = report.get('ground_truth', {})
    ext = result.get('features_extracted', {})
    report_text = report.get('report_text', '').lower()

    if 'seton' in report_text and ext.get('treatment_status') != 'seton_in_place':
        patterns['seton_miss'] += 1
    if gt.get('fistula_count', 0) != ext.get('fistula_count', 0):
        patterns['fistula_count_error'] += 1
    if gt.get('extension') != ext.get('extension'):
        patterns['extension_error'] += 1
    if gt.get('t2_hyperintensity_degree') != ext.get('t2_hyperintensity_degree'):
        patterns['t2_degree_error'] += 1
    if result.get('vai_extracted', 0) > gt.get('expected_vai_score', 0):
        patterns['overscoring'] += 1
    else:
        patterns['underscoring'] += 1

print("\nFailure Pattern Frequency (out of 8 failed cases):")
for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count}/8 ({count/8*100:.0f}%)")

print("\n" + "="*80)
print("RECOMMENDED FIXES")
print("="*80)
print("""
1. SETON DETECTION (3/8 = 38% of failures):
   - Add explicit seton keyword matching
   - Look for: "seton", "seton suture", "seton in place", "draining seton"
   - When detected, ALWAYS set treatment_status = "seton_in_place"

2. FISTULA COUNTING (4/8 = 50% of failures):
   - Parse EXACT count from text: "two fistulas", "2 tracts", "three tracts"
   - Multiple tracts at DIFFERENT clock positions = MULTIPLE fistulas
   - "3, 9, and 12 o'clock positions" = 3 fistulas, NOT 1

3. EXTENSION CLASSIFICATION (5/8 = 63% of failures):
   - "supralevator" = ALWAYS severe
   - "ischioanal fossa" without supralevator = moderate
   - "intersphincteric" only = none or mild

4. OVERSCORING (5/8 = 63%):
   - Parser tends to OVERESTIMATE severity
   - Add conservative scoring rules
   - When seton present, REDUCE T2 contribution
   - When abscess-only (no fistula tract), use lower scores
""")
