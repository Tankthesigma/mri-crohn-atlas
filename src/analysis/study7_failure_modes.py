#!/usr/bin/env python3
"""
Study 7: Failure Mode Taxonomy
===============================
Analyze all errors from studies 1, 5, 6 and categorize:
- Type A: Missing info in report
- Type B: Ambiguous language
- Type C: Unusual anatomy
- Type D: Parser error (hallucination)
- Type E: Formula limitation
- Type F: Ground truth error

Output:
- /data/validation_results/failure_taxonomy.json
- /data/validation_results/failure_piechart.png
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

# Define failure mode categories
FAILURE_MODES = {
    'A': {
        'name': 'Missing Information',
        'description': 'Report lacks key features needed for accurate scoring',
        'examples': ['No T2 signal mentioned', 'Extension not described', 'Fibrosis state unclear'],
        'color': '#e74c3c'
    },
    'B': {
        'name': 'Ambiguous Language',
        'description': 'Report uses vague or interpretable language',
        'examples': ['"may represent"', '"possible fistula"', '"cannot exclude"'],
        'color': '#f39c12'
    },
    'C': {
        'name': 'Unusual Anatomy',
        'description': 'Atypical anatomical presentation not well modeled',
        'examples': ['Horseshoe extension', 'Multiple separate fistulas', 'Extrasphincteric'],
        'color': '#9b59b6'
    },
    'D': {
        'name': 'Parser Error',
        'description': 'Parser misinterprets or hallucinates findings',
        'examples': ['Misread "no fistula"', 'Wrong count', 'Incorrect T2 grading'],
        'color': '#e67e22'
    },
    'E': {
        'name': 'Formula Limitation',
        'description': 'Crosswalk formula systematically mis-predicts',
        'examples': ['Healed edge cases', 'Very severe disease', 'Fibrosis term applicability'],
        'color': '#3498db'
    },
    'F': {
        'name': 'Ground Truth Error',
        'description': 'Expected scores may be incorrect or debatable',
        'examples': ['Radiologist disagreement', 'Scoring ambiguity', 'Study methodology'],
        'color': '#2ecc71'
    }
}

# Load error data from different sources
print("Loading error data from validation studies...")

# 1. Load LOSO results
with open(OUTPUT_DIR / "loso_results.json") as f:
    loso_data = json.load(f)

# 2. Load calibration results
with open(OUTPUT_DIR / "calibration_results.json") as f:
    calibration_data = json.load(f)

# 3. Load parser validation data
with open(DATA_DIR / "parser_tests" / "validation_results.json") as f:
    parser_data = json.load(f)

# 4. Load subgroup results
with open(OUTPUT_DIR / "subgroup_results.json") as f:
    subgroup_data = json.load(f)

# Collect all errors
errors = []

# From LOSO - formula prediction errors
print("\nAnalyzing LOSO errors...")
for study_result in loso_data['per_study_results']:
    for pred in study_result['predictions']:
        if abs(pred['error']) > 0.5:  # Consider errors > 0.5 as notable
            error_entry = {
                'source': 'LOSO',
                'study': study_result['study'],
                'vai': pred['vai'],
                'fibrosis': pred['fibrosis'],
                'error': pred['error'],
                'abs_error': abs(pred['error']),
                'data_source': pred['source']
            }

            # Categorize error
            if pred['vai'] <= 2:
                error_entry['failure_mode'] = 'E'
                error_entry['explanation'] = 'Healed/remission edge case - formula limitation'
            elif abs(pred['error']) > 2:
                error_entry['failure_mode'] = 'E'
                error_entry['explanation'] = 'Large residual - systematic formula limitation'
            else:
                error_entry['failure_mode'] = 'F'
                error_entry['explanation'] = 'Within expected variance - possible ground truth variation'

            errors.append(error_entry)

# From parser validation - parsing errors
print("Analyzing parser errors...")
for result in parser_data['results']:
    vai_error = result['vai_error']
    magnifi_error = result['magnifi_error']

    if vai_error > 2 or magnifi_error > 2:
        error_entry = {
            'source': 'Parser',
            'case_id': result['case_id'],
            'difficulty': result['difficulty'],
            'vai_error': vai_error,
            'magnifi_error': magnifi_error,
            'total_error': vai_error + magnifi_error
        }

        # Categorize based on difficulty and error type
        if result['difficulty'] == 'ambiguous':
            error_entry['failure_mode'] = 'B'
            error_entry['explanation'] = 'Ambiguous case - expected uncertainty'
        elif result['difficulty'] == 'edge_case':
            error_entry['failure_mode'] = 'A'
            error_entry['explanation'] = 'Edge case - likely missing information'
        elif result['difficulty'] == 'complex':
            error_entry['failure_mode'] = 'C'
            error_entry['explanation'] = 'Complex anatomy - unusual presentation'
        else:
            error_entry['failure_mode'] = 'D'
            error_entry['explanation'] = 'Standard case with error - parser issue'

        errors.append(error_entry)

# From calibration - cases with errors
print("Analyzing calibration errors...")
for case in calibration_data['raw_cases']:
    total_error = case['vai_error'] + case['magnifi_error']
    if total_error > 3:
        error_entry = {
            'source': 'Calibration',
            'case_id': case['id'],
            'name': case['name'],
            'confidence': case['confidence'],
            'vai_error': case['vai_error'],
            'magnifi_error': case['magnifi_error'],
            'total_error': total_error
        }

        # Categorize based on confidence
        if case['confidence'] < 0.6:
            error_entry['failure_mode'] = 'B'
            error_entry['explanation'] = 'Low confidence - parser detected ambiguity'
        else:
            error_entry['failure_mode'] = 'D'
            error_entry['explanation'] = 'High confidence but wrong - parser error'

        errors.append(error_entry)

# Count failure modes
print("\n" + "="*70)
print("FAILURE MODE TAXONOMY")
print("="*70)

mode_counts = Counter(e['failure_mode'] for e in errors)
total_errors = len(errors)

print(f"\nTotal errors analyzed: {total_errors}")
print("\nDistribution by failure mode:")
for mode, info in FAILURE_MODES.items():
    count = mode_counts.get(mode, 0)
    pct = (count / total_errors * 100) if total_errors > 0 else 0
    print(f"\n  Type {mode}: {info['name']}")
    print(f"    Count: {count} ({pct:.1f}%)")
    print(f"    Description: {info['description']}")

# Analyze severity distribution
print("\n" + "="*70)
print("ERROR SEVERITY ANALYSIS")
print("="*70)

# For LOSO errors
loso_errors = [e for e in errors if e['source'] == 'LOSO']
if loso_errors:
    abs_errors = [e['abs_error'] for e in loso_errors]
    print(f"\nLOSO Formula Errors:")
    print(f"  Count: {len(loso_errors)}")
    print(f"  Mean: {np.mean(abs_errors):.3f}")
    print(f"  Median: {np.median(abs_errors):.3f}")
    print(f"  Max: {np.max(abs_errors):.3f}")

# For parser errors
parser_errors = [e for e in errors if e['source'] in ['Parser', 'Calibration']]
if parser_errors:
    total_errs = [e['total_error'] for e in parser_errors if 'total_error' in e]
    if total_errs:
        print(f"\nParser Errors:")
        print(f"  Count: {len(parser_errors)}")
        print(f"  Mean total error: {np.mean(total_errs):.3f}")
        print(f"  Max total error: {np.max(total_errs):.3f}")

# Examples by failure mode
print("\n" + "="*70)
print("EXAMPLES BY FAILURE MODE")
print("="*70)

for mode in FAILURE_MODES.keys():
    mode_errors = [e for e in errors if e['failure_mode'] == mode][:3]  # Top 3
    if mode_errors:
        print(f"\nType {mode} ({FAILURE_MODES[mode]['name']}):")
        for e in mode_errors:
            if e['source'] == 'LOSO':
                print(f"  - VAI={e['vai']}, Error={e['error']:.2f}: {e['explanation']}")
            else:
                print(f"  - {e.get('name', e.get('case_id', 'unknown'))}: {e['explanation']}")

# Recommendations
recommendations = {
    'A': 'Enhance parser to flag when key features are missing and request clarification',
    'B': 'Implement uncertainty quantification and report confidence intervals',
    'C': 'Expand training data to include more unusual anatomical presentations',
    'D': 'Improve parser accuracy through additional training examples',
    'E': 'Consider formula refinement for edge cases (especially healed state)',
    'F': 'Establish gold standard with multiple radiologist consensus'
}

print("\n" + "="*70)
print("RECOMMENDATIONS BY FAILURE MODE")
print("="*70)
for mode, rec in recommendations.items():
    print(f"\nType {mode}: {rec}")

# Save results
output_data = {
    'method': 'Failure Mode Taxonomy',
    'description': 'Categorizing errors by root cause',
    'failure_modes': {k: {**v, 'recommendation': recommendations[k]} for k, v in FAILURE_MODES.items()},
    'total_errors_analyzed': total_errors,
    'distribution': {
        mode: {
            'count': mode_counts.get(mode, 0),
            'percentage': (mode_counts.get(mode, 0) / total_errors * 100) if total_errors > 0 else 0
        }
        for mode in FAILURE_MODES.keys()
    },
    'all_errors': errors,
    'severity_statistics': {
        'loso': {
            'count': len(loso_errors),
            'mean_abs_error': float(np.mean([e['abs_error'] for e in loso_errors])) if loso_errors else None,
            'max_abs_error': float(np.max([e['abs_error'] for e in loso_errors])) if loso_errors else None
        },
        'parser': {
            'count': len(parser_errors),
            'mean_total_error': float(np.mean([e['total_error'] for e in parser_errors if 'total_error' in e])) if parser_errors else None
        }
    }
}

output_path = OUTPUT_DIR / "failure_taxonomy.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Failure Mode Taxonomy Analysis', fontsize=14, fontweight='bold')

# Plot 1: Pie chart of failure modes
ax1 = axes[0, 0]
labels = [f"Type {k}: {v['name']}" for k, v in FAILURE_MODES.items() if mode_counts.get(k, 0) > 0]
sizes = [mode_counts.get(k, 0) for k in FAILURE_MODES.keys() if mode_counts.get(k, 0) > 0]
colors_pie = [FAILURE_MODES[k]['color'] for k in FAILURE_MODES.keys() if mode_counts.get(k, 0) > 0]

if sizes:
    wedges, texts, autotexts = ax1.pie(sizes, labels=None, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=[0.02]*len(sizes))
    ax1.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
ax1.set_title('Failure Mode Distribution', fontsize=12)

# Plot 2: Bar chart comparison
ax2 = axes[0, 1]
modes = list(FAILURE_MODES.keys())
counts = [mode_counts.get(m, 0) for m in modes]
colors_bar = [FAILURE_MODES[m]['color'] for m in modes]

bars = ax2.bar(modes, counts, color=colors_bar, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Failure Mode Type', fontsize=11)
ax2.set_ylabel('Number of Errors', fontsize=11)
ax2.set_title('Error Count by Failure Mode', fontsize=12)

# Add labels
for bar, count in zip(bars, counts):
    if count > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

ax2.grid(axis='y', alpha=0.3)

# Plot 3: Error severity distribution (LOSO)
ax3 = axes[1, 0]
if loso_errors:
    abs_errors = [e['abs_error'] for e in loso_errors]
    ax3.hist(abs_errors, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(abs_errors), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(abs_errors):.2f}')
    ax3.axvline(x=1.0, color='green', linestyle=':', linewidth=2,
                label='Threshold: 1.0')
ax3.set_xlabel('Absolute Error (MAGNIFI points)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Distribution of LOSO Prediction Errors', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Failure modes by source
ax4 = axes[1, 1]
sources = ['LOSO', 'Parser', 'Calibration']
source_mode_counts = {}
for source in sources:
    source_errors = [e for e in errors if e['source'] == source]
    source_mode_counts[source] = Counter(e['failure_mode'] for e in source_errors)

x = np.arange(len(modes))
width = 0.25

for i, source in enumerate(sources):
    counts = [source_mode_counts[source].get(m, 0) for m in modes]
    ax4.bar(x + i*width, counts, width, label=source, alpha=0.8)

ax4.set_xlabel('Failure Mode Type', fontsize=11)
ax4.set_ylabel('Number of Errors', fontsize=11)
ax4.set_title('Failure Modes by Data Source', fontsize=12)
ax4.set_xticks(x + width)
ax4.set_xticklabels(modes)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "failure_piechart.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'failure_piechart.png'}")
plt.close()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTotal errors analyzed: {total_errors}")
print(f"\nTop failure modes:")
for mode, count in mode_counts.most_common(3):
    pct = count / total_errors * 100
    print(f"  - Type {mode} ({FAILURE_MODES[mode]['name']}): {count} ({pct:.1f}%)")

print("\nKey findings:")
if mode_counts.get('E', 0) > mode_counts.get('D', 0):
    print("  - Formula limitations are more common than parser errors")
if mode_counts.get('A', 0) + mode_counts.get('B', 0) > total_errors * 0.3:
    print("  - Input quality (missing/ambiguous info) is a significant factor")
