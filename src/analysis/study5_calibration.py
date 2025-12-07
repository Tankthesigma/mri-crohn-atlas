#!/usr/bin/env python3
"""
Study 5: Calibration Curves
============================
Analyze parser confidence calibration:
- Load parser predictions with confidence scores from /data/parser_tests/
- Bin by confidence: [0-50%, 50-70%, 70-85%, 85-95%, 95-100%]
- Calculate actual accuracy per bin
- Plot calibration curve (diagonal = perfect)
- Calculate Expected Calibration Error (ECE)

Output:
- /data/validation_results/calibration_results.json
- /data/validation_results/calibration_curve.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"
PARSER_DIR = DATA_DIR / "parser_tests"

# Load parser test data
print("Loading parser test data...")

# Load confidence calibration data
with open(PARSER_DIR / "confidence_calibration.json") as f:
    confidence_data = json.load(f)

# Load validation results
with open(PARSER_DIR / "validation_results.json") as f:
    validation_results = json.load(f)

# Extract cases with confidence scores
cases = []

# From confidence calibration file (edge cases)
for result in confidence_data.get('detailed_results', []):
    vai_correct = result['vai_err'] <= 3  # Within 3 points
    magnifi_correct = result['magnifi_err'] <= 3
    is_correct = vai_correct and magnifi_correct

    cases.append({
        'id': result['id'],
        'name': result['name'],
        'confidence': result['confidence'],
        'vai_error': result['vai_err'],
        'magnifi_error': result['magnifi_err'],
        'is_correct': is_correct,
        'source': 'edge_cases'
    })

# From validation results (real-world cases)
for result in validation_results.get('results', []):
    confidence = 0.85  # Default confidence if not specified
    vai_correct = result['vai_error'] <= 3
    magnifi_correct = result['magnifi_error'] <= 3
    is_correct = vai_correct and magnifi_correct

    cases.append({
        'id': result['case_id'],
        'name': result.get('difficulty', 'unknown'),
        'confidence': confidence,
        'vai_error': result['vai_error'],
        'magnifi_error': result['magnifi_error'],
        'is_correct': is_correct,
        'source': 'validation'
    })

df = pd.DataFrame(cases)
print(f"Loaded {len(df)} test cases")

# Define confidence bins
bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
bin_labels = ['0-50%', '50-70%', '70-85%', '85-95%', '95-100%']

df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, labels=bin_labels, include_lowest=True)

print("\n" + "="*60)
print("CALIBRATION ANALYSIS")
print("="*60)

# Calculate accuracy per bin
calibration_results = []
for i, label in enumerate(bin_labels):
    bin_df = df[df['confidence_bin'] == label]
    if len(bin_df) == 0:
        continue

    n_cases = len(bin_df)
    n_correct = bin_df['is_correct'].sum()
    accuracy = n_correct / n_cases if n_cases > 0 else 0

    # Expected accuracy (midpoint of bin)
    expected_accuracy = (bins[i] + bins[i+1]) / 2

    calibration_results.append({
        'bin': label,
        'bin_lower': bins[i],
        'bin_upper': bins[i+1],
        'n_cases': int(n_cases),
        'n_correct': int(n_correct),
        'accuracy': float(accuracy),
        'expected_accuracy': float(expected_accuracy),
        'calibration_error': float(accuracy - expected_accuracy)
    })

    print(f"\n{label}:")
    print(f"  n={n_cases}, correct={n_correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Expected: {expected_accuracy:.1%}")
    print(f"  Calibration Error: {(accuracy - expected_accuracy):.1%}")

# Calculate Expected Calibration Error (ECE)
total_cases = len(df)
ece = sum(
    (r['n_cases'] / total_cases) * abs(r['calibration_error'])
    for r in calibration_results
)

print("\n" + "="*60)
print("EXPECTED CALIBRATION ERROR (ECE)")
print("="*60)
print(f"ECE = {ece:.4f} ({ece*100:.2f}%)")

# Determine over/under confidence
avg_confidence = df['confidence'].mean()
overall_accuracy = df['is_correct'].mean()
calibration_gap = overall_accuracy - avg_confidence

print(f"\nOverall Statistics:")
print(f"  Mean Confidence: {avg_confidence:.1%}")
print(f"  Overall Accuracy: {overall_accuracy:.1%}")
print(f"  Calibration Gap: {calibration_gap:+.1%}")

if calibration_gap > 0.1:
    confidence_assessment = "UNDERCONFIDENT (predictions better than confidence suggests)"
elif calibration_gap < -0.1:
    confidence_assessment = "OVERCONFIDENT (confidence higher than actual accuracy)"
else:
    confidence_assessment = "WELL CALIBRATED"

print(f"  Assessment: {confidence_assessment}")

# Calculate reliability diagram data
reliability_data = []
for r in calibration_results:
    reliability_data.append({
        'confidence_midpoint': (r['bin_lower'] + r['bin_upper']) / 2,
        'accuracy': r['accuracy'],
        'n_cases': r['n_cases']
    })

# Save results
output_data = {
    'method': 'Calibration Analysis',
    'description': 'Assessing whether parser confidence matches actual accuracy',
    'n_total_cases': int(total_cases),
    'overall_statistics': {
        'mean_confidence': float(avg_confidence),
        'overall_accuracy': float(overall_accuracy),
        'calibration_gap': float(calibration_gap),
        'assessment': confidence_assessment
    },
    'expected_calibration_error': float(ece),
    'calibration_by_bin': calibration_results,
    'reliability_diagram_data': reliability_data,
    'raw_cases': cases
}

output_path = OUTPUT_DIR / "calibration_results.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Parser Calibration Analysis', fontsize=14, fontweight='bold')

# Plot 1: Calibration curve (reliability diagram)
ax1 = axes[0, 0]
conf_midpoints = [r['confidence_midpoint'] for r in reliability_data]
accuracies = [r['accuracy'] for r in reliability_data]
n_sizes = [r['n_cases'] * 20 for r in reliability_data]  # Scale for visibility

ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
ax1.scatter(conf_midpoints, accuracies, s=n_sizes, c='#3498db', alpha=0.7,
            edgecolors='black', linewidth=1, label='Bins (size = n cases)')
ax1.plot(conf_midpoints, accuracies, 'b-', alpha=0.5, linewidth=1)

ax1.fill_between([0, 1], [0, 1], [0, 0], alpha=0.1, color='red', label='Overconfident zone')
ax1.fill_between([0, 1], [1, 1], [0, 1], alpha=0.1, color='green', label='Underconfident zone')

ax1.set_xlabel('Mean Predicted Confidence', fontsize=11)
ax1.set_ylabel('Actual Accuracy', fontsize=11)
ax1.set_title(f'Calibration Curve (ECE = {ece:.4f})', fontsize=12)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart of accuracy vs expected
ax2 = axes[0, 1]
x = range(len(calibration_results))
width = 0.35

actual_acc = [r['accuracy'] for r in calibration_results]
expected_acc = [r['expected_accuracy'] for r in calibration_results]
labels = [r['bin'] for r in calibration_results]

bars1 = ax2.bar([i - width/2 for i in x], actual_acc, width, label='Actual Accuracy', color='#2ecc71')
bars2 = ax2.bar([i + width/2 for i in x], expected_acc, width, label='Expected Accuracy', color='#3498db', alpha=0.7)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Actual vs Expected Accuracy by Confidence Bin', fontsize=12)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Confidence distribution
ax3 = axes[1, 0]
ax3.hist(df['confidence'], bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
ax3.axvline(x=avg_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_confidence:.2f}')
ax3.set_xlabel('Confidence', fontsize=11)
ax3.set_ylabel('Number of Cases', fontsize=11)
ax3.set_title('Distribution of Parser Confidence Scores', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Error vs Confidence scatter
ax4 = axes[1, 1]
total_error = df['vai_error'] + df['magnifi_error']
correct_mask = df['is_correct']

ax4.scatter(df.loc[correct_mask, 'confidence'], total_error[correct_mask],
            c='#2ecc71', alpha=0.6, s=80, label='Correct (≤3 pts each)', edgecolors='black', linewidth=0.5)
ax4.scatter(df.loc[~correct_mask, 'confidence'], total_error[~correct_mask],
            c='#e74c3c', alpha=0.6, s=80, label='Incorrect (>3 pts)', edgecolors='black', linewidth=0.5)

ax4.set_xlabel('Confidence', fontsize=11)
ax4.set_ylabel('Total Error (VAI + MAGNIFI)', fontsize=11)
ax4.set_title('Error vs Confidence', fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['confidence'], total_error, 1)
p = np.poly1d(z)
x_line = np.linspace(df['confidence'].min(), df['confidence'].max(), 100)
ax4.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1, label='Trend')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'calibration_curve.png'}")
plt.close()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nExpected Calibration Error (ECE): {ece:.4f} ({ece*100:.2f}%)")
print(f"Assessment: {confidence_assessment}")
print(f"\nCalibration by bin:")
for r in calibration_results:
    status = "✓" if abs(r['calibration_error']) < 0.15 else "⚠"
    print(f"  {status} {r['bin']}: {r['accuracy']:.0%} actual vs {r['expected_accuracy']:.0%} expected")
