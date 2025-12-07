#!/usr/bin/env python3
"""
Study 6: Adversarial Testing
=============================
Analyze adversarial test cases and document parser robustness.

Test cases include:
- Typos/OCR errors
- Abbreviations only
- Contradictions
- Incomplete reports
- Noise/artifacts

Output:
- /data/validation_results/adversarial_results.json
- /data/validation_results/adversarial_cases.json (copy with results)
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"
PARSER_DIR = DATA_DIR / "parser_tests"

# Load adversarial test cases
with open(PARSER_DIR / "adversarial_cases.json") as f:
    adversarial_data = json.load(f)

test_cases = adversarial_data['test_cases']
print(f"Loaded {len(test_cases)} adversarial test cases")

# Define adversarial categories
ADVERSARIAL_CATEGORIES = {
    'OCR_ERRORS': 'Typos/OCR Artifacts',
    'EXTREMELY_LONG': 'Verbose Reports',
    'MEDICAL_LATIN': 'Latin Terminology',
    'CONTRADICTORY': 'Contradictions',
    'MINIMAL_TEXT': 'Minimal/Terse',
    'NUMBERS_HEAVY': 'Numbers Heavy',
    'ALL_CAPS': 'Legacy Formatting',
    'NEGATIVE_HEAVY': 'Many Negatives',
    'GARBLED_SECTIONS': 'Corrupted Data',
    'DIFFERENTIAL_HEAVY': 'Differential Diagnosis'
}

print("\n" + "="*70)
print("ADVERSARIAL TEST CASE ANALYSIS")
print("="*70)

# Analyze each test case
results = []
for case in test_cases:
    case_id = case['id']
    name = case['name']
    category = ADVERSARIAL_CATEGORIES.get(name, 'Other')
    report_text = case['report_text']
    ground_truth = case['ground_truth']

    # Analyze report characteristics
    char_count = len(report_text)
    word_count = len(report_text.split())
    has_numbers = any(c.isdigit() for c in report_text)
    has_special_chars = any(c in '@#$%^&*' for c in report_text)
    is_all_caps = report_text.isupper()
    has_negatives = 'no ' in report_text.lower() or 'not ' in report_text.lower()
    has_conditionals = any(word in report_text.lower() for word in ['may', 'could', 'possible', 'versus', 'either'])

    # Expected difficulty based on characteristics
    difficulty_factors = []
    if has_special_chars:
        difficulty_factors.append('special_chars')
    if is_all_caps:
        difficulty_factors.append('all_caps')
    if word_count < 20:
        difficulty_factors.append('very_short')
    if word_count > 200:
        difficulty_factors.append('very_long')
    if has_conditionals:
        difficulty_factors.append('uncertain_language')
    if report_text.count('no ') > 5:
        difficulty_factors.append('many_negatives')

    result = {
        'id': case_id,
        'name': name,
        'category': category,
        'what_it_tests': case['what_it_tests'],
        'report_characteristics': {
            'char_count': char_count,
            'word_count': word_count,
            'has_numbers': has_numbers,
            'has_special_chars': has_special_chars,
            'is_all_caps': is_all_caps,
            'has_negatives': has_negatives,
            'has_conditionals': has_conditionals
        },
        'difficulty_factors': difficulty_factors,
        'ground_truth': {
            'expected_vai': ground_truth['expected_vai_score'],
            'expected_magnifi': ground_truth['expected_magnifi_score'],
            'fistula_count': ground_truth['fistula_count'],
            'notes': ground_truth.get('notes', '')
        },
        'report_text_preview': report_text[:200] + '...' if len(report_text) > 200 else report_text
    }

    results.append(result)

    print(f"\n{case_id}: {name}")
    print(f"  Category: {category}")
    print(f"  Words: {word_count}, Chars: {char_count}")
    print(f"  Expected: VAI={ground_truth['expected_vai_score']}, MAGNIFI={ground_truth['expected_magnifi_score']}")
    print(f"  Difficulty factors: {difficulty_factors or 'none'}")

# Summary statistics
print("\n" + "="*70)
print("ADVERSARIAL TEST SUITE SUMMARY")
print("="*70)

category_counts = Counter(r['category'] for r in results)
print("\nCategories covered:")
for cat, count in sorted(category_counts.items()):
    print(f"  - {cat}: {count} case(s)")

# Expected scoring ranges
vai_scores = [r['ground_truth']['expected_vai'] for r in results]
magnifi_scores = [r['ground_truth']['expected_magnifi'] for r in results]

print(f"\nExpected score ranges:")
print(f"  VAI: {min(vai_scores)} - {max(vai_scores)} (mean: {np.mean(vai_scores):.1f})")
print(f"  MAGNIFI: {min(magnifi_scores)} - {max(magnifi_scores)} (mean: {np.mean(magnifi_scores):.1f})")

# Document expected failure modes
failure_modes = {
    'OCR_ERRORS': {
        'expected_issue': 'Character substitution errors (1→l, O→0)',
        'mitigation': 'Fuzzy matching, context-aware interpretation',
        'risk_level': 'Medium'
    },
    'EXTREMELY_LONG': {
        'expected_issue': 'Key findings buried in verbose text',
        'mitigation': 'Focus on FINDINGS and IMPRESSION sections',
        'risk_level': 'Low'
    },
    'MEDICAL_LATIN': {
        'expected_issue': 'Unrecognized Latin medical terms',
        'mitigation': 'Medical terminology mapping',
        'risk_level': 'Medium'
    },
    'CONTRADICTORY': {
        'expected_issue': 'Conflicting statements in same report',
        'mitigation': 'Flag low confidence, extract definite findings',
        'risk_level': 'High'
    },
    'MINIMAL_TEXT': {
        'expected_issue': 'Insufficient information for accurate scoring',
        'mitigation': 'Use abbreviation expansion, flag uncertainty',
        'risk_level': 'Medium'
    },
    'NUMBERS_HEAVY': {
        'expected_issue': 'Confusion between technical and clinical numbers',
        'mitigation': 'Context-aware number extraction',
        'risk_level': 'Low'
    },
    'ALL_CAPS': {
        'expected_issue': 'Parsing issues with uppercase text',
        'mitigation': 'Case normalization in preprocessing',
        'risk_level': 'Low'
    },
    'NEGATIVE_HEAVY': {
        'expected_issue': 'Missing positive findings among many negatives',
        'mitigation': 'Explicit positive finding extraction',
        'risk_level': 'Medium'
    },
    'GARBLED_SECTIONS': {
        'expected_issue': 'Incomplete or corrupted data',
        'mitigation': 'Skip corrupted sections, flag missing data',
        'risk_level': 'High'
    },
    'DIFFERENTIAL_HEAVY': {
        'expected_issue': 'Uncertain language, no definitive findings',
        'mitigation': 'Extract probable findings, flag low confidence',
        'risk_level': 'High'
    }
}

print("\n" + "="*70)
print("EXPECTED FAILURE MODES BY CATEGORY")
print("="*70)
for name, mode in failure_modes.items():
    print(f"\n{ADVERSARIAL_CATEGORIES.get(name, name)}:")
    print(f"  Issue: {mode['expected_issue']}")
    print(f"  Mitigation: {mode['mitigation']}")
    print(f"  Risk Level: {mode['risk_level']}")

# Define pass/fail criteria
print("\n" + "="*70)
print("PASS/FAIL CRITERIA")
print("="*70)
print("A test case PASSES if:")
print("  - VAI error ≤ 3 points AND MAGNIFI error ≤ 3 points")
print("  - OR parser correctly flags low confidence (<50%) for problematic cases")
print("\nA test case FAILS if:")
print("  - VAI error > 3 points OR MAGNIFI error > 3 points")
print("  - AND parser reported high confidence (>70%)")

# Save results
output_data = {
    'method': 'Adversarial Testing',
    'description': 'Stress-testing the parser with edge cases, errors, and unusual inputs',
    'n_test_cases': len(test_cases),
    'categories_tested': list(ADVERSARIAL_CATEGORIES.values()),
    'test_case_results': results,
    'failure_modes': failure_modes,
    'pass_criteria': {
        'vai_tolerance': 3,
        'magnifi_tolerance': 3,
        'or_low_confidence': True,
        'low_confidence_threshold': 0.5
    },
    'expected_score_ranges': {
        'vai_min': int(min(vai_scores)),
        'vai_max': int(max(vai_scores)),
        'vai_mean': float(np.mean(vai_scores)),
        'magnifi_min': int(min(magnifi_scores)),
        'magnifi_max': int(max(magnifi_scores)),
        'magnifi_mean': float(np.mean(magnifi_scores))
    },
    'risk_assessment': {
        'high_risk_categories': [k for k, v in failure_modes.items() if v['risk_level'] == 'High'],
        'medium_risk_categories': [k for k, v in failure_modes.items() if v['risk_level'] == 'Medium'],
        'low_risk_categories': [k for k, v in failure_modes.items() if v['risk_level'] == 'Low']
    }
}

output_path = OUTPUT_DIR / "adversarial_results.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Adversarial Test Suite Analysis', fontsize=14, fontweight='bold')

# Plot 1: Category distribution
ax1 = axes[0, 0]
categories = list(ADVERSARIAL_CATEGORIES.values())
counts = [sum(1 for r in results if r['category'] == cat) for cat in categories]
colors = ['#e74c3c' if failure_modes[name]['risk_level'] == 'High'
          else '#f39c12' if failure_modes[name]['risk_level'] == 'Medium'
          else '#2ecc71' for name in ADVERSARIAL_CATEGORIES.keys()]

ax1.barh(range(len(categories)), counts, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(categories)))
ax1.set_yticklabels(categories, fontsize=9)
ax1.set_xlabel('Number of Test Cases', fontsize=11)
ax1.set_title('Test Cases by Category (color = risk level)', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='High Risk'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Medium Risk'),
    Patch(facecolor='#2ecc71', edgecolor='black', label='Low Risk')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Plot 2: Expected score distribution
ax2 = axes[0, 1]
ax2.scatter(vai_scores, magnifi_scores, s=100, c='#3498db', alpha=0.7,
            edgecolors='black', linewidth=1)
ax2.plot([0, 25], [0, 25], 'k--', linewidth=1, alpha=0.5, label='y=x')

# Label each point
for i, r in enumerate(results):
    ax2.annotate(r['name'][:3], (vai_scores[i], magnifi_scores[i]),
                 fontsize=7, ha='center', va='bottom')

ax2.set_xlabel('Expected VAI Score', fontsize=11)
ax2.set_ylabel('Expected MAGNIFI Score', fontsize=11)
ax2.set_title('Expected Score Distribution', fontsize=12)
ax2.set_xlim(-1, 25)
ax2.set_ylim(-1, 25)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Plot 3: Report length distribution
ax3 = axes[1, 0]
word_counts = [r['report_characteristics']['word_count'] for r in results]
names = [r['name'][:10] for r in results]

ax3.barh(range(len(names)), word_counts, color='#9b59b6', edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(names)))
ax3.set_yticklabels(names, fontsize=8)
ax3.set_xlabel('Word Count', fontsize=11)
ax3.set_title('Report Length by Test Case', fontsize=12)
ax3.axvline(x=50, color='green', linestyle='--', linewidth=1.5, label='Typical (50 words)')
ax3.legend(fontsize=9)
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Difficulty factors
ax4 = axes[1, 1]
all_factors = []
for r in results:
    all_factors.extend(r['difficulty_factors'])

factor_counts = Counter(all_factors)
if factor_counts:
    factors = list(factor_counts.keys())
    counts = list(factor_counts.values())
    ax4.barh(factors, counts, color='#1abc9c', edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Number of Test Cases', fontsize=11)
    ax4.set_title('Difficulty Factors Distribution', fontsize=12)
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No difficulty factors identified', ha='center', va='center',
             transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Difficulty Factors Distribution', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "adversarial_analysis.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'adversarial_analysis.png'}")
plt.close()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nAdversarial Test Suite: {len(test_cases)} cases across {len(ADVERSARIAL_CATEGORIES)} categories")
print(f"\nRisk Distribution:")
print(f"  High Risk: {len(output_data['risk_assessment']['high_risk_categories'])} categories")
print(f"  Medium Risk: {len(output_data['risk_assessment']['medium_risk_categories'])} categories")
print(f"  Low Risk: {len(output_data['risk_assessment']['low_risk_categories'])} categories")
print(f"\nHigh-risk categories requiring special attention:")
for cat in output_data['risk_assessment']['high_risk_categories']:
    print(f"  - {ADVERSARIAL_CATEGORIES.get(cat, cat)}")
