#!/usr/bin/env python3
"""
Generate publication-quality validation plots for ISEF presentation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for severity levels
SEVERITY_COLORS = {
    'remission': '#2ecc71',  # Green
    'mild': '#3498db',       # Blue
    'moderate': '#f39c12',   # Orange
    'severe': '#e74c3c',     # Red
}

def load_data():
    """Load validation results."""
    base_path = Path(__file__).parent

    with open(base_path / 'full_validation_results.json', 'r') as f:
        results = json.load(f)

    with open(base_path / 'validation_metrics.json', 'r') as f:
        metrics = json.load(f)

    return results, metrics

def plot_vai_scatter(results, metrics, save_path):
    """Create VAI correlation scatter plot with identity line."""
    fig, ax = plt.subplots(figsize=(8, 7))

    expected = [r['expected_vai'] for r in results]
    predicted = [r['predicted_vai'] for r in results]
    severities = [r['severity'] for r in results]

    # Plot points colored by severity
    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [e for e, m in zip(expected, mask) if m]
        y = [p for p, m in zip(predicted, mask) if m]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=80, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=sev.capitalize())

    # Identity line
    ax.plot([0, 22], [0, 22], 'k--', alpha=0.5, linewidth=1.5, label='Perfect Agreement')

    # Regression line
    z = np.polyfit(expected, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 22, 100)
    ax.plot(x_line, p(x_line), 'b-', alpha=0.8, linewidth=2,
            label=f'Regression (R²={metrics["vai_r2"]:.3f})')

    ax.set_xlabel('Expected VAI Score')
    ax.set_ylabel('Predicted VAI Score')
    ax.set_title('Van Assche Index: Parser vs Ground Truth\n(n=68, ICC=0.981)')
    ax.set_xlim(-1, 23)
    ax.set_ylim(-1, 23)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    # Add metrics annotation
    textstr = f'ICC = {metrics["vai_icc"]:.3f}\nMAE = {metrics["vai_mae"]:.2f}\nr = {metrics["vai_pearson_r"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_magnifi_scatter(results, metrics, save_path):
    """Create MAGNIFI-CD correlation scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 7))

    expected = [r['expected_magnifi'] for r in results]
    predicted = [r['predicted_magnifi'] for r in results]
    severities = [r['severity'] for r in results]

    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [e for e, m in zip(expected, mask) if m]
        y = [p for p, m in zip(predicted, mask) if m]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=80, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=sev.capitalize())

    ax.plot([0, 25], [0, 25], 'k--', alpha=0.5, linewidth=1.5, label='Perfect Agreement')

    z = np.polyfit(expected, predicted, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 25, 100)
    ax.plot(x_line, p(x_line), 'b-', alpha=0.8, linewidth=2,
            label=f'Regression (R²={metrics["magnifi_r2"]:.3f})')

    ax.set_xlabel('Expected MAGNIFI-CD Score')
    ax.set_ylabel('Predicted MAGNIFI-CD Score')
    ax.set_title('MAGNIFI-CD: Parser vs Ground Truth\n(n=68, ICC=0.987)')
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 26)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    textstr = f'ICC = {metrics["magnifi_icc"]:.3f}\nMAE = {metrics["magnifi_mae"]:.2f}\nr = {metrics["magnifi_pearson_r"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_bland_altman_vai(results, metrics, save_path):
    """Create Bland-Altman plot for VAI."""
    fig, ax = plt.subplots(figsize=(9, 6))

    expected = np.array([r['expected_vai'] for r in results])
    predicted = np.array([r['predicted_vai'] for r in results])
    severities = [r['severity'] for r in results]

    means = (expected + predicted) / 2
    diffs = predicted - expected

    ba = metrics['vai_bland_altman']
    mean_diff = ba['mean_diff']
    loa_upper = ba['loa_upper']
    loa_lower = ba['loa_lower']

    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [m for m, mk in zip(means, mask) if mk]
        y = [d for d, mk in zip(diffs, mask) if mk]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=80, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=sev.capitalize())

    ax.axhline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f'Mean Bias ({mean_diff:.2f})')
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD ({loa_upper:.2f})')
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD ({loa_lower:.2f})')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.fill_between([0, 22], loa_lower, loa_upper, alpha=0.1, color='red')

    ax.set_xlabel('Mean of Expected and Predicted VAI')
    ax.set_ylabel('Difference (Predicted - Expected)')
    ax.set_title('Bland-Altman Plot: VAI Agreement Analysis\n(95% Limits of Agreement)')
    ax.set_xlim(-1, 22)
    ax.set_ylim(-6, 6)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_bland_altman_magnifi(results, metrics, save_path):
    """Create Bland-Altman plot for MAGNIFI-CD."""
    fig, ax = plt.subplots(figsize=(9, 6))

    expected = np.array([r['expected_magnifi'] for r in results])
    predicted = np.array([r['predicted_magnifi'] for r in results])
    severities = [r['severity'] for r in results]

    means = (expected + predicted) / 2
    diffs = predicted - expected

    ba = metrics['magnifi_bland_altman']
    mean_diff = ba['mean_diff']
    loa_upper = ba['loa_upper']
    loa_lower = ba['loa_lower']

    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [m for m, mk in zip(means, mask) if mk]
        y = [d for d, mk in zip(diffs, mask) if mk]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=80, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=sev.capitalize())

    ax.axhline(mean_diff, color='blue', linestyle='-', linewidth=2, label=f'Mean Bias ({mean_diff:.2f})')
    ax.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5, label=f'+1.96 SD ({loa_upper:.2f})')
    ax.axhline(loa_lower, color='red', linestyle='--', linewidth=1.5, label=f'-1.96 SD ({loa_lower:.2f})')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.fill_between([0, 25], loa_lower, loa_upper, alpha=0.1, color='red')

    ax.set_xlabel('Mean of Expected and Predicted MAGNIFI-CD')
    ax.set_ylabel('Difference (Predicted - Expected)')
    ax.set_title('Bland-Altman Plot: MAGNIFI-CD Agreement Analysis\n(95% Limits of Agreement)')
    ax.set_xlim(-1, 25)
    ax.set_ylim(-6, 6)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_error_distribution(results, save_path):
    """Create error distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vai_errors = [r['predicted_vai'] - r['expected_vai'] for r in results]
    magnifi_errors = [r['predicted_magnifi'] - r['expected_magnifi'] for r in results]

    # VAI errors
    ax = axes[0]
    bins = np.arange(-5.5, 6.5, 1)
    ax.hist(vai_errors, bins=bins, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(vai_errors), color='orange', linestyle='-', linewidth=2,
               label=f'Mean Bias ({np.mean(vai_errors):.2f})')
    ax.set_xlabel('Prediction Error (Predicted - Expected)')
    ax.set_ylabel('Frequency')
    ax.set_title('VAI Error Distribution')
    ax.legend()
    ax.set_xlim(-6, 6)

    # MAGNIFI errors
    ax = axes[1]
    ax.hist(magnifi_errors, bins=bins, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(magnifi_errors), color='orange', linestyle='-', linewidth=2,
               label=f'Mean Bias ({np.mean(magnifi_errors):.2f})')
    ax.set_xlabel('Prediction Error (Predicted - Expected)')
    ax.set_ylabel('Frequency')
    ax.set_title('MAGNIFI-CD Error Distribution')
    ax.legend()
    ax.set_xlim(-6, 6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_accuracy_by_severity(results, save_path):
    """Create bar chart of accuracy by severity level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    severities = ['remission', 'mild', 'moderate', 'severe']
    vai_acc = []
    magnifi_acc = []
    counts = []

    for sev in severities:
        sev_results = [r for r in results if r['severity'] == sev]
        n = len(sev_results)
        counts.append(n)

        vai_within_2 = sum(1 for r in sev_results if abs(r['predicted_vai'] - r['expected_vai']) <= 2)
        magnifi_within_3 = sum(1 for r in sev_results if abs(r['predicted_magnifi'] - r['expected_magnifi']) <= 3)

        vai_acc.append(100 * vai_within_2 / n if n > 0 else 0)
        magnifi_acc.append(100 * magnifi_within_3 / n if n > 0 else 0)

    x = np.arange(len(severities))
    width = 0.35

    bars1 = ax.bar(x - width/2, vai_acc, width, label='VAI (±2 pts)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, magnifi_acc, width, label='MAGNIFI-CD (±3 pts)', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Disease Severity')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Parser Accuracy by Severity Level')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s.capitalize()}\n(n={c})' for s, c in zip(severities, counts)])
    ax.set_ylim(0, 110)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.axhline(90, color='red', linestyle='--', alpha=0.5, label='90% Target')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_icc_comparison(metrics, save_path):
    """Create ICC comparison bar chart vs radiologist benchmarks."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['VAI\n(Parser)', 'VAI\n(Radiologists)', 'MAGNIFI-CD\n(Parser)', 'MAGNIFI-CD\n(Radiologists)']
    values = [
        metrics['vai_icc'],
        0.68,  # Published radiologist ICC for VAI
        metrics['magnifi_icc'],
        0.87   # Published radiologist ICC for MAGNIFI
    ]
    colors = ['#3498db', '#bdc3c7', '#2ecc71', '#bdc3c7']

    bars = ax.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_ylabel('Intraclass Correlation Coefficient (ICC)')
    ax.set_title('Parser vs Radiologist Agreement\n(ICC Comparison)')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.90, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.90)')
    ax.axhline(0.75, color='orange', linestyle='--', alpha=0.7, label='Good (>0.75)')
    ax.legend(loc='lower right')

    # Add improvement annotations
    vai_improvement = ((metrics['vai_icc'] - 0.68) / 0.68) * 100
    magnifi_improvement = ((metrics['magnifi_icc'] - 0.87) / 0.87) * 100

    ax.annotate(f'+{vai_improvement:.1f}%', xy=(0.5, 0.85), fontsize=12, color='green',
                fontweight='bold', ha='center')
    ax.annotate(f'+{magnifi_improvement:.1f}%', xy=(2.5, 0.95), fontsize=12, color='green',
                fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_combined_summary(results, metrics, save_path):
    """Create a combined 2x2 summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: VAI scatter
    ax = axes[0, 0]
    expected = [r['expected_vai'] for r in results]
    predicted = [r['predicted_vai'] for r in results]
    severities = [r['severity'] for r in results]

    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [e for e, m in zip(expected, mask) if m]
        y = [p for p, m in zip(predicted, mask) if m]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=60, alpha=0.7, label=sev.capitalize())

    ax.plot([0, 22], [0, 22], 'k--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Expected VAI')
    ax.set_ylabel('Predicted VAI')
    ax.set_title(f'VAI Correlation (ICC={metrics["vai_icc"]:.3f})')
    ax.set_xlim(-1, 23)
    ax.set_ylim(-1, 23)
    ax.legend(loc='lower right', fontsize=8)

    # Top-right: MAGNIFI scatter
    ax = axes[0, 1]
    expected = [r['expected_magnifi'] for r in results]
    predicted = [r['predicted_magnifi'] for r in results]

    for sev in ['remission', 'mild', 'moderate', 'severe']:
        mask = [s == sev for s in severities]
        x = [e for e, m in zip(expected, mask) if m]
        y = [p for p, m in zip(predicted, mask) if m]
        ax.scatter(x, y, c=SEVERITY_COLORS[sev], s=60, alpha=0.7, label=sev.capitalize())

    ax.plot([0, 25], [0, 25], 'k--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Expected MAGNIFI-CD')
    ax.set_ylabel('Predicted MAGNIFI-CD')
    ax.set_title(f'MAGNIFI-CD Correlation (ICC={metrics["magnifi_icc"]:.3f})')
    ax.set_xlim(-1, 26)
    ax.set_ylim(-1, 26)
    ax.legend(loc='lower right', fontsize=8)

    # Bottom-left: Bland-Altman VAI
    ax = axes[1, 0]
    expected = np.array([r['expected_vai'] for r in results])
    predicted = np.array([r['predicted_vai'] for r in results])
    means = (expected + predicted) / 2
    diffs = predicted - expected
    ba = metrics['vai_bland_altman']

    ax.scatter(means, diffs, c='#3498db', s=50, alpha=0.6)
    ax.axhline(ba['mean_diff'], color='blue', linestyle='-', linewidth=2)
    ax.axhline(ba['loa_upper'], color='red', linestyle='--', linewidth=1.5)
    ax.axhline(ba['loa_lower'], color='red', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Mean VAI Score')
    ax.set_ylabel('Difference')
    ax.set_title(f'VAI Bland-Altman (Bias={ba["mean_diff"]:.2f})')
    ax.set_ylim(-6, 6)

    # Bottom-right: ICC comparison
    ax = axes[1, 1]
    categories = ['VAI\n(Ours)', 'VAI\n(Rad.)', 'MAG\n(Ours)', 'MAG\n(Rad.)']
    values = [metrics['vai_icc'], 0.68, metrics['magnifi_icc'], 0.87]
    colors = ['#3498db', '#bdc3c7', '#2ecc71', '#bdc3c7']
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('ICC')
    ax.set_title('Parser vs Radiologist Agreement')
    ax.set_ylim(0, 1.1)
    ax.axhline(0.90, color='green', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val + 0.02),
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    print("="*60)
    print("MRI-Crohn Atlas Parser Validation Plots")
    print("="*60)

    results, metrics = load_data()
    base_path = Path(__file__).parent / 'plots'
    base_path.mkdir(exist_ok=True)

    print(f"\nGenerating plots for {len(results)} cases...")

    plot_vai_scatter(results, metrics, base_path / 'vai_correlation.png')
    plot_magnifi_scatter(results, metrics, base_path / 'magnifi_correlation.png')
    plot_bland_altman_vai(results, metrics, base_path / 'vai_bland_altman.png')
    plot_bland_altman_magnifi(results, metrics, base_path / 'magnifi_bland_altman.png')
    plot_error_distribution(results, base_path / 'error_distribution.png')
    plot_accuracy_by_severity(results, base_path / 'accuracy_by_severity.png')
    plot_icc_comparison(metrics, base_path / 'icc_comparison.png')
    plot_combined_summary(results, metrics, base_path / 'validation_summary.png')

    print("\n" + "="*60)
    print("All plots saved to:", base_path)
    print("="*60)

if __name__ == '__main__':
    main()
