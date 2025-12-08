#!/usr/bin/env python3
"""
Coverage Matrix Generator for MRI Report Parser Validation

Generates coverage statistics and visualization for the test case suite.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

# Define the coverage matrix structure
CASE_TYPES = [
    "Simple Intersphincteric",
    "Transsphincteric",
    "Complex/Branching",
    "With Abscess",
    "Healed/Fibrotic",
    "Post-Surgical",
    "Pediatric",
    "Ambiguous/Equivocal",
    "Horseshoe",
    "Extrasphincteric",
    "Normal/No Fistula"
]

SEVERITY_LEVELS = ["Remission", "Mild", "Moderate", "Severe"]

# Clinically impossible combinations
IMPOSSIBLE_CELLS = {
    ("With Abscess", "Remission"): "Abscess indicates active disease",
    ("Normal/No Fistula", "Remission"): "N/A - not fistula cases",
    ("Normal/No Fistula", "Mild"): "N/A - not fistula cases",
    ("Normal/No Fistula", "Moderate"): "N/A - not fistula cases",
    ("Normal/No Fistula", "Severe"): "N/A - not fistula cases"
}

def load_test_cases(filepath):
    """Load test cases from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('test_cases', [])


def categorize_case(case):
    """Extract case_type and severity from a test case"""
    case_type = case.get('case_type', 'Unknown')
    severity = case.get('severity', 'Unknown')
    return case_type, severity


def generate_coverage_matrix(test_cases):
    """Generate coverage matrix from test cases"""
    matrix = defaultdict(lambda: defaultdict(list))

    for case in test_cases:
        case_type, severity = categorize_case(case)
        if case_type != 'Unknown' and severity != 'Unknown' and severity != 'N/A':
            matrix[case_type][severity].append(case.get('id', 'unknown'))

    return matrix


def calculate_coverage_stats(matrix):
    """Calculate coverage statistics"""
    total_possible = 0
    filled_cells = 0
    coverage_details = []

    for case_type in CASE_TYPES:
        for severity in SEVERITY_LEVELS:
            if (case_type, severity) in IMPOSSIBLE_CELLS:
                continue  # Skip impossible combinations

            total_possible += 1
            count = len(matrix.get(case_type, {}).get(severity, []))

            if count >= 3:
                status = "FULL"
                filled_cells += 1
            elif count > 0:
                status = "PARTIAL"
                filled_cells += 1  # Count partial as filled for percentage
            else:
                status = "EMPTY"

            coverage_details.append({
                'case_type': case_type,
                'severity': severity,
                'count': count,
                'status': status,
                'case_ids': matrix.get(case_type, {}).get(severity, [])
            })

    return {
        'total_possible': total_possible,
        'filled_cells': filled_cells,
        'coverage_percentage': (filled_cells / total_possible * 100) if total_possible > 0 else 0,
        'details': coverage_details
    }


def print_ascii_matrix(matrix, stats):
    """Print ASCII representation of coverage matrix"""
    print("\n" + "=" * 80)
    print("PARSER VALIDATION COVERAGE MATRIX")
    print("=" * 80 + "\n")

    # Header row
    header = f"{'Case Type':<25} | {'Remission':^10} | {'Mild':^10} | {'Moderate':^10} | {'Severe':^10}"
    print(header)
    print("-" * len(header))

    for case_type in CASE_TYPES:
        row = f"{case_type:<25} |"
        for severity in SEVERITY_LEVELS:
            if (case_type, severity) in IMPOSSIBLE_CELLS:
                cell = "  N/A  "
            else:
                count = len(matrix.get(case_type, {}).get(severity, []))
                if count >= 3:
                    cell = f"  [{count}]  "  # Brackets indicate full coverage
                elif count > 0:
                    cell = f"   {count}   "  # Plain number for partial
                else:
                    cell = "   -   "  # Dash for empty
            row += f" {cell:^9}|"
        print(row)

    print("-" * len(header))
    print(f"\nLEGEND: [N] = Full coverage (3+ cases), N = Partial, - = Empty, N/A = Clinically impossible")
    print(f"\nCOVERAGE: {stats['filled_cells']}/{stats['total_possible']} cells = {stats['coverage_percentage']:.1f}%")
    print("=" * 80)


def print_gap_analysis(stats):
    """Print analysis of coverage gaps"""
    print("\nGAP ANALYSIS:")
    print("-" * 40)

    empty = [d for d in stats['details'] if d['status'] == 'EMPTY']
    partial = [d for d in stats['details'] if d['status'] == 'PARTIAL' and d['count'] < 3]

    if empty:
        print("\nEMPTY CELLS (need cases):")
        for gap in empty:
            print(f"  - {gap['case_type']} x {gap['severity']}")
    else:
        print("\nNo empty cells!")

    if partial:
        print("\nPARTIAL CELLS (have cases but < 3):")
        for gap in partial:
            print(f"  - {gap['case_type']} x {gap['severity']}: {gap['count']} case(s)")


def print_source_breakdown(test_cases):
    """Print breakdown of cases by source"""
    sources = defaultdict(int)
    for case in test_cases:
        source = case.get('source', 'unknown')
        sources[source] += 1

    print("\nCASES BY SOURCE:")
    print("-" * 40)
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    print(f"  TOTAL: {len(test_cases)}")


def main():
    """Main function"""
    # Find the test cases file
    script_dir = Path(__file__).parent
    test_file = script_dir / "mega_test_cases.json"

    if not test_file.exists():
        print(f"Error: {test_file} not found")
        sys.exit(1)

    print(f"Loading test cases from: {test_file}")
    test_cases = load_test_cases(test_file)

    print(f"Loaded {len(test_cases)} test cases")

    # Generate coverage matrix
    matrix = generate_coverage_matrix(test_cases)

    # Calculate statistics
    stats = calculate_coverage_stats(matrix)

    # Print results
    print_ascii_matrix(matrix, stats)
    print_gap_analysis(stats)
    print_source_breakdown(test_cases)

    # Check if target met
    print("\n" + "=" * 80)
    if stats['coverage_percentage'] >= 80:
        print(f"TARGET MET! Coverage is {stats['coverage_percentage']:.1f}% (>= 80%)")
    else:
        print(f"TARGET NOT MET. Coverage is {stats['coverage_percentage']:.1f}% (< 80%)")
        print(f"Need to fill {int(0.8 * stats['total_possible']) - stats['filled_cells']} more cells")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    main()
