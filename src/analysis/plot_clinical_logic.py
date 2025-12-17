#!/usr/bin/env python3
"""
CLINICAL LOGIC VISUALIZATION
=============================

Generate a high-resolution bar chart showing the model's clinical sensitivity
for the poster "Results" section.

Output: plots/final/clinical_validation_chart.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "plots" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from clinical sensitivity verification
TREATMENTS = [
    'Biologic\nMonotherapy',
    'Seton Drainage\n(Surgical)',
    'Combination\nTherapy',
    'Stem Cell\nTherapy'
]

PROBABILITIES = [31.5, 95.0, 95.8, 96.1]

# Color scheme
COLORS = [
    '#DC2626',  # Red - Biologic (low efficacy in refractory)
    '#16A34A',  # Green - Seton (high efficacy)
    '#15803D',  # Darker Green - Combination (high efficacy)
    '#2563EB',  # Blue - Stem Cell (emerging/high)
]

# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def create_clinical_validation_chart():
    """Create poster-quality clinical validation bar chart."""

    # Set up figure with poster-friendly sizing
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    # Create bars
    x = np.arange(len(TREATMENTS))
    bars = ax.bar(x, PROBABILITIES, color=COLORS, edgecolor='black', linewidth=2, width=0.7)

    # Add value labels on top of each bar
    for i, (bar, prob) in enumerate(zip(bars, PROBABILITIES)):
        height = bar.get_height()
        ax.annotate(f'{prob:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=18, fontweight='bold',
                    color='black')

    # Add synergy arrow from Biologic to Combination
    arrow_start_x = 0  # Biologic bar
    arrow_end_x = 2    # Combination bar
    arrow_start_y = PROBABILITIES[0] + 5
    arrow_end_y = PROBABILITIES[2] - 10

    # Draw curved arrow
    ax.annotate('',
                xy=(arrow_end_x, arrow_end_y),
                xytext=(arrow_start_x, arrow_start_y + 15),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#7C3AED',
                    lw=3,
                    connectionstyle='arc3,rad=0.3',
                    mutation_scale=20
                ))

    # Add synergy label
    ax.text(1.0, 65, '+64%\nSynergy\nBoost',
            fontsize=16, fontweight='bold',
            color='#7C3AED',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E8FF', edgecolor='#7C3AED', linewidth=2))

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(TREATMENTS, fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted Remission Probability (%)', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 110)

    # Title
    ax.set_title('Clinical Sensitivity Verification\nP.A.R.S.E.C. Model Predictions for Severe pfCD (VAI=14)',
                 fontsize=22, fontweight='bold', pad=20)

    # Add horizontal reference lines
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')

    # Add legend for color coding
    legend_elements = [
        mpatches.Patch(facecolor='#DC2626', edgecolor='black', label='Low Efficacy (Refractory)'),
        mpatches.Patch(facecolor='#16A34A', edgecolor='black', label='High Efficacy (Proven)'),
        mpatches.Patch(facecolor='#2563EB', edgecolor='black', label='Emerging Therapy'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.95)

    # Add training data context box
    context_text = (
        "Training Data Ground Truth:\n"
        "• Biologic: 6.9% effective (n=29)\n"
        "• Surgical: 85.7% effective (n=14)\n"
        "• Combination: 54.5% effective (n=11)\n"
        "• Stem Cell: 81.8% effective (n=11)"
    )
    ax.text(0.98, 0.02, context_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95))

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "clinical_validation_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Chart saved: {output_path}")

    # Also save a PDF version for print
    pdf_path = OUTPUT_DIR / "clinical_validation_chart.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ PDF saved: {pdf_path}")

    plt.close()

    return output_path


def create_simple_version():
    """Create a simpler, cleaner version for smaller poster sections."""

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    x = np.arange(len(TREATMENTS))
    bars = ax.bar(x, PROBABILITIES, color=COLORS, edgecolor='black', linewidth=1.5, width=0.65)

    # Value labels
    for bar, prob in zip(bars, PROBABILITIES):
        height = bar.get_height()
        ax.annotate(f'{prob:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

    # Synergy annotation (simpler)
    ax.annotate('+64%',
                xy=(2, 95.8),
                xytext=(0.3, 55),
                fontsize=14, fontweight='bold', color='#7C3AED',
                arrowprops=dict(arrowstyle='->', color='#7C3AED', lw=2))

    ax.set_xticks(x)
    ax.set_xticklabels(['Biologic\nOnly', 'Seton\nDrainage', 'Combination', 'Stem Cell'],
                       fontsize=14, fontweight='bold')
    ax.set_ylabel('Remission Probability (%)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_title('Model Clinical Sensitivity', fontsize=20, fontweight='bold')

    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    simple_path = OUTPUT_DIR / "clinical_validation_simple.png"
    plt.savefig(simple_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Simple version saved: {simple_path}")

    plt.close()

    return simple_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CLINICAL LOGIC VISUALIZATION")
    print("=" * 60)
    print()

    print("Creating main poster chart...")
    main_path = create_clinical_validation_chart()

    print("\nCreating simplified version...")
    simple_path = create_simple_version()

    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  1. {main_path}")
    print(f"  2. {main_path.with_suffix('.pdf')}")
    print(f"  3. {simple_path}")
