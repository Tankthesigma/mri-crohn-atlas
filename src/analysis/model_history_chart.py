#!/usr/bin/env python3
"""
MODEL HISTORY CHART
===================

Comprehensive visualization of ALL model versions from V1 to PARSEC.
Shows the complete journey of model development with AUC progression.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# EXTRACT ALL MODEL RESULTS
# =============================================================================

def extract_auc_from_report(report_path):
    """Extract AUC from various report formats."""
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)

        # Try different keys where AUC might be stored
        auc = None
        std = None

        # Check common locations
        if 'final_model' in data:
            auc = data['final_model'].get('auc')
        elif 'stability' in data:
            auc = data['stability'].get('mean_auc')
            std = data['stability'].get('std_auc')
        elif 'auc' in data:
            auc = data['auc']
        elif 'mean_auc' in data:
            auc = data['mean_auc']
            std = data.get('std_auc')
        elif 'results' in data:
            if 'auc' in data['results']:
                auc = data['results']['auc']
            elif 'mean_auc' in data['results']:
                auc = data['results']['mean_auc']
        elif 'metrics' in data:
            auc = data['metrics'].get('auc') or data['metrics'].get('mean_auc')
        elif 'cv_auc_mean' in data:
            auc = data['cv_auc_mean']
        elif 'test_auc' in data:
            auc = data['test_auc']

        # Try to find std
        if std is None:
            std = data.get('std_auc') or data.get('cv_auc_std')
            if 'stability' in data:
                std = data['stability'].get('std_auc')

        return auc, std
    except Exception as e:
        return None, None


def get_all_models():
    """Get all model results from reports."""
    models = []

    # Define model mapping with categories
    model_info = {
        # Early versions
        'training_report.json': ('V1', 'Early', 'First model'),
        'training_report_v2.json': ('V2', 'Early', 'Second iteration'),
        'classifier_report.json': ('V3', 'Early', 'Classifier'),
        'binary_classifier_report.json': ('V4', 'Early', 'Binary classifier'),
        'ensemble_v5_report.json': ('V5', 'Early', 'First ensemble'),
        'v6_report.json': ('V6', 'Early', 'V6'),
        'v7_report.json': ('V7', 'Early', 'V7'),
        'v8_report.json': ('V8', 'Early', 'V8'),
        'v9_report.json': ('V9', 'Early', 'V9'),
        'v10_report.json': ('V10', 'Early', 'V10'),

        # Middle versions
        'v11_report.json': ('V11', 'Middle', 'V11'),
        'v12_report.json': ('V12', 'Middle', 'V12'),
        'v13_report.json': ('V13', 'Middle', 'V13'),
        'v14_report.json': ('V14', 'Middle', 'V14'),
        'v15_pressure_report.json': ('V15', 'Middle', 'Pressure test'),
        'v16_report.json': ('V16', 'Middle', 'V16'),
        'v17_report.json': ('V17', 'Middle', 'V17'),
        'v18_report.json': ('V18', 'Middle', 'V18'),
        'v19_report.json': ('V19', 'Middle', 'V19'),
        'v20_report.json': ('V20', 'Middle', 'V20'),

        # Refinement phase
        'v21_report.json': ('V21', 'Refine', 'V21'),
        'v22_report.json': ('V22', 'Refine', 'V22'),
        'v23_report.json': ('V23', 'Refine', 'V23'),
        'v24_report.json': ('V24', 'Refine', 'V24'),
        'v25_report.json': ('V25', 'Refine', 'V25'),
        'v26_report.json': ('V26', 'Refine', 'V26'),
        'v27_report.json': ('V27', 'Refine', 'V27'),
        'v28_report.json': ('V28', 'Refine', 'V28'),
        'v29_report.json': ('V29', 'Refine', 'V29'),
        'v30_report.json': ('V30', 'Refine', 'V30'),

        # Advanced versions
        'v31_rct_report.json': ('V31-RCT', 'Advanced', 'RCT filtered'),
        'v31_biologic_report.json': ('V31-Bio', 'Advanced', 'Biologic filtered'),
        'v32_transfer_report.json': ('V32', 'Advanced', 'Transfer learning'),
        'v33_enhanced_report.json': ('V33', 'Advanced', 'Enhanced'),
        'v34_calibrated_report.json': ('V34', 'Advanced', 'Calibrated'),
        'v34_optuna_report.json': ('V34-Opt', 'Advanced', 'Optuna tuned'),
        'v35_moonshot_report.json': ('V35', 'Advanced', 'Moonshot'),
        'v35_multi_gbdt_report.json': ('V35-GBDT', 'Advanced', 'Multi-GBDT'),
        'v36_quad_stack_report.json': ('V36', 'Advanced', 'Quad stack'),
        'v37_hybrid_features_report.json': ('V37', 'Advanced', 'Hybrid features'),
        'v38_sensitivity_report.json': ('V38', 'Advanced', 'Sensitivity'),
        'v38_torture_report.json': ('V38-T', 'Advanced', 'Torture test'),
        'v39_stabilized_report.json': ('V39', 'Stable', 'Stabilized'),
        'v40_dictator_report.json': ('V40', 'Stable', 'Dictator'),
        'v41_bagged_report.json': ('V41', 'Stable', 'Bagged ensemble'),
        'v42_pure_report.json': ('V42', 'Stable', 'Pure TabPFN'),
        'v43_expanded_report.json': ('V43', 'Expanded', 'Expanded data'),

        # PARSEC family
        'parsec_report.json': ('PARSEC', 'PARSEC', 'Original (leaky)'),
        'parsec_control_report.json': ('PARSEC-Ctrl', 'PARSEC', 'Control (leaky)'),
        'parsec_leakfree_report.json': ('PARSEC-LF', 'PARSEC', 'Leak-free'),
    }

    for filename, (name, category, description) in model_info.items():
        report_path = MODELS_DIR / filename
        if report_path.exists():
            auc, std = extract_auc_from_report(report_path)
            if auc is not None:
                models.append({
                    'name': name,
                    'category': category,
                    'description': description,
                    'auc': auc,
                    'std': std if std else 0,
                    'filename': filename
                })

    # Add torture test result
    models.append({
        'name': 'Torture-30',
        'category': 'Final',
        'description': '30-run torture test',
        'auc': 0.8978,
        'std': 0.0065,
        'filename': 'torture_test'
    })

    return models


def create_history_chart(models):
    """Create comprehensive model history visualization."""

    # Filter to models with valid AUC
    valid_models = [m for m in models if m['auc'] is not None and m['auc'] > 0.5]

    # Sort by approximate version order
    version_order = {
        'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10,
        'V11': 11, 'V12': 12, 'V13': 13, 'V14': 14, 'V15': 15, 'V16': 16, 'V17': 17, 'V18': 18, 'V19': 19, 'V20': 20,
        'V21': 21, 'V22': 22, 'V23': 23, 'V24': 24, 'V25': 25, 'V26': 26, 'V27': 27, 'V28': 28, 'V29': 29, 'V30': 30,
        'V31-RCT': 31, 'V31-Bio': 31.5, 'V32': 32, 'V33': 33, 'V34': 34, 'V34-Opt': 34.5,
        'V35': 35, 'V35-GBDT': 35.5, 'V36': 36, 'V37': 37, 'V38': 38, 'V38-T': 38.5,
        'V39': 39, 'V40': 40, 'V41': 41, 'V42': 42, 'V43': 43,
        'PARSEC': 44, 'PARSEC-Ctrl': 45, 'PARSEC-LF': 46, 'Torture-30': 47
    }

    valid_models.sort(key=lambda x: version_order.get(x['name'], 100))

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Category colors
    category_colors = {
        'Early': '#94A3B8',      # Gray
        'Middle': '#60A5FA',     # Blue
        'Refine': '#34D399',     # Green
        'Advanced': '#FBBF24',   # Yellow
        'Stable': '#F97316',     # Orange
        'Expanded': '#A78BFA',   # Purple
        'PARSEC': '#EF4444',    # Red
        'Final': '#10B981',      # Emerald
    }

    # Plot bars
    x = np.arange(len(valid_models))
    colors = [category_colors.get(m['category'], '#666666') for m in valid_models]

    bars = ax.bar(x, [m['auc'] for m in valid_models], color=colors, edgecolor='white', linewidth=0.5)

    # Add error bars where available
    for i, m in enumerate(valid_models):
        if m['std'] > 0:
            ax.errorbar(i, m['auc'], yerr=m['std'], color='black', capsize=3, capthick=1, linewidth=1)

    # Add reference lines
    ax.axhline(y=0.90, color='#10B981', linestyle='--', linewidth=2, label='Target (0.90)')
    ax.axhline(y=0.8795, color='#F59E0B', linestyle=':', linewidth=2, label='V41 Baseline (0.8795)')
    ax.axhline(y=0.80, color='#EF4444', linestyle=':', linewidth=1, alpha=0.5, label='Acceptable (0.80)')

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels([m['name'] for m in valid_models], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax.set_title('Complete Model Evolution: V1 → PARSEC Leak-Free\nMRI-Crohn Atlas Classification Performance',
                 fontsize=14, fontweight='bold', pad=20)

    # Set y-axis limits
    ax.set_ylim(0.5, 1.0)

    # Add value labels on bars
    for i, (bar, m) in enumerate(zip(bars, valid_models)):
        height = bar.get_height()
        if height > 0.85:
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=6, rotation=90)

    # Create legend for categories
    legend_patches = [mpatches.Patch(color=color, label=cat)
                      for cat, color in category_colors.items()]
    legend_patches.extend([
        plt.Line2D([0], [0], color='#10B981', linestyle='--', linewidth=2, label='Target (0.90)'),
        plt.Line2D([0], [0], color='#F59E0B', linestyle=':', linewidth=2, label='V41 Baseline'),
    ])

    ax.legend(handles=legend_patches, loc='lower right', ncol=2, fontsize=8)

    # Add grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # Highlight best models
    best_idx = np.argmax([m['auc'] for m in valid_models])
    bars[best_idx].set_edgecolor('#000000')
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()

    # Save
    chart_path = PLOTS_DIR / "model_history_complete.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Chart saved: {chart_path}")

    return valid_models


def create_summary_table(models):
    """Create summary table of all models."""

    # Sort by AUC descending
    sorted_models = sorted(models, key=lambda x: x['auc'] if x['auc'] else 0, reverse=True)

    print("\n" + "=" * 80)
    print("COMPLETE MODEL HISTORY - RANKED BY AUC")
    print("=" * 80)
    print(f"{'Rank':<5} {'Model':<15} {'Category':<12} {'AUC':<10} {'Std':<10} {'Description':<25}")
    print("-" * 80)

    for i, m in enumerate(sorted_models[:30], 1):  # Top 30
        std_str = f"±{m['std']:.4f}" if m['std'] > 0 else "N/A"
        print(f"{i:<5} {m['name']:<15} {m['category']:<12} {m['auc']:.4f}    {std_str:<10} {m['description']:<25}")

    print("-" * 80)

    # Summary stats
    aucs = [m['auc'] for m in sorted_models if m['auc']]
    print(f"\nTotal Models: {len(sorted_models)}")
    print(f"Best AUC: {max(aucs):.4f} ({sorted_models[0]['name']})")
    print(f"Worst AUC: {min(aucs):.4f}")
    print(f"Models above 0.90: {sum(1 for a in aucs if a >= 0.90)}")
    print(f"Models above 0.85: {sum(1 for a in aucs if a >= 0.85)}")

    return sorted_models


def create_evolution_chart(models):
    """Create a line chart showing AUC evolution over time."""

    # Filter and sort chronologically
    version_order = {
        'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5, 'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10,
        'V11': 11, 'V12': 12, 'V13': 13, 'V14': 14, 'V15': 15, 'V16': 16, 'V17': 17, 'V18': 18, 'V19': 19, 'V20': 20,
        'V21': 21, 'V22': 22, 'V23': 23, 'V24': 24, 'V25': 25, 'V26': 26, 'V27': 27, 'V28': 28, 'V29': 29, 'V30': 30,
        'V31-RCT': 31, 'V31-Bio': 31.5, 'V32': 32, 'V33': 33, 'V34': 34, 'V34-Opt': 34.5,
        'V35': 35, 'V35-GBDT': 35.5, 'V36': 36, 'V37': 37, 'V38': 38, 'V38-T': 38.5,
        'V39': 39, 'V40': 40, 'V41': 41, 'V42': 42, 'V43': 43,
        'PARSEC': 44, 'PARSEC-Ctrl': 45, 'PARSEC-LF': 46, 'Torture-30': 47
    }

    valid_models = [m for m in models if m['auc'] is not None and m['auc'] > 0.5]
    valid_models.sort(key=lambda x: version_order.get(x['name'], 100))

    # Create figure with dual axis
    fig, ax1 = plt.subplots(figsize=(16, 8))

    x = np.arange(len(valid_models))
    aucs = [m['auc'] for m in valid_models]

    # Plot line with markers
    line = ax1.plot(x, aucs, 'b-', linewidth=2, marker='o', markersize=6, label='AUC')

    # Fill area under curve
    ax1.fill_between(x, 0.5, aucs, alpha=0.2, color='blue')

    # Add error bands where available
    upper = []
    lower = []
    for m in valid_models:
        if m['std'] > 0:
            upper.append(m['auc'] + m['std'])
            lower.append(m['auc'] - m['std'])
        else:
            upper.append(m['auc'])
            lower.append(m['auc'])

    ax1.fill_between(x, lower, upper, alpha=0.3, color='blue')

    # Reference lines
    ax1.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Target (0.90)')
    ax1.axhline(y=0.8795, color='orange', linestyle=':', linewidth=2, label='V41 Baseline')

    # Mark best model
    best_idx = np.argmax(aucs)
    ax1.scatter([best_idx], [aucs[best_idx]], color='red', s=200, zorder=5, marker='*', label=f'Best: {valid_models[best_idx]["name"]}')

    # Mark final model
    ax1.scatter([len(valid_models)-1], [aucs[-1]], color='green', s=150, zorder=5, marker='D', label=f'Final: {valid_models[-1]["name"]}')

    # Customize
    ax1.set_xticks(x[::2])  # Every other label
    ax1.set_xticklabels([valid_models[i]['name'] for i in range(0, len(valid_models), 2)], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Evolution Over Time\nFrom V1 to PARSEC Leak-Free (Torture Validated)',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0.6, 1.0)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    chart_path = PLOTS_DIR / "model_evolution_line.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Evolution chart saved: {chart_path}")


def main():
    print("\n" + "=" * 80)
    print("MODEL HISTORY VISUALIZATION")
    print("=" * 80)

    # Get all models
    print("\n  Extracting model results...")
    models = get_all_models()
    print(f"  Found {len(models)} model reports")

    # Create bar chart
    print("\n  Creating history bar chart...")
    valid_models = create_history_chart(models)

    # Create evolution line chart
    print("\n  Creating evolution line chart...")
    create_evolution_chart(valid_models)

    # Print summary table
    create_summary_table(valid_models)

    # Save CSV
    df = pd.DataFrame(valid_models)
    csv_path = BASE_DIR / "data" / "model_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Model history saved: {csv_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
