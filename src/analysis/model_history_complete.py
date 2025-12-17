#!/usr/bin/env python3
"""
COMPLETE MODEL HISTORY - ALL VERSIONS
Extracts AUC from every model report ever created
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
DATA_DIR = PROJECT_ROOT / "data"

def extract_auc_from_report(report_path):
    """Extract AUC from any report format"""
    try:
        with open(report_path) as f:
            data = json.load(f)

        # Try different key locations
        auc = None
        std = None
        name = report_path.stem.replace('_report', '')

        # Format 1: metrics.test_auc (v6-v13)
        if 'metrics' in data:
            auc = data['metrics'].get('test_auc') or data['metrics'].get('cv_auc')
            std = data['metrics'].get('cv_auc_std')

        # Format 2: results.mean_auc (torture tests)
        if 'results' in data and auc is None:
            auc = data['results'].get('mean_auc')
            std = data['results'].get('std_auc')

        # Format 3: final_model.auc (parsec)
        if 'final_model' in data and auc is None:
            auc = data['final_model'].get('auc')

        # Format 4: stability.mean_auc
        if 'stability' in data:
            if auc is None:
                auc = data['stability'].get('mean_auc')
            std = std or data['stability'].get('std_auc')

        # Format 5: direct auc key
        if auc is None:
            auc = data.get('auc') or data.get('cv_auc') or data.get('mean_auc')

        # Format 6: ensemble_auc
        if auc is None:
            auc = data.get('ensemble_auc')

        # Format 7: For reports with comparison data
        if auc is None and 'comparison' in data:
            # Get self-reported AUC from comparison
            comp = data['comparison']
            for key in comp:
                if name.replace('v', '') in key:
                    auc = comp[key]
                    break

        # Get description/model type
        desc = data.get('model_type', data.get('version', name))

        return {
            'name': name,
            'auc': auc,
            'std': std,
            'desc': desc[:50] if desc else name,
            'path': str(report_path)
        }
    except Exception as e:
        return None

def main():
    print("=" * 80)
    print("COMPLETE MODEL HISTORY - EXTRACTING ALL VERSIONS")
    print("=" * 80)

    # Find all report files
    report_files = list(MODELS_DIR.glob("*report*.json"))
    print(f"\nFound {len(report_files)} report files")

    # Extract AUC from each
    models = []
    for rf in sorted(report_files):
        result = extract_auc_from_report(rf)
        if result and result['auc'] is not None:
            models.append(result)
            print(f"  ✓ {result['name']}: AUC = {result['auc']:.4f}")
        else:
            print(f"  ✗ {rf.name}: No AUC found")

    # Also check for torture test results
    torture_csv = DATA_DIR / "parsec_torture_results.csv"
    if torture_csv.exists():
        df = pd.read_csv(torture_csv)
        models.append({
            'name': 'Torture-30',
            'auc': df['auc'].mean(),
            'std': df['auc'].std(),
            'desc': '30-run torture validated',
            'path': str(torture_csv)
        })
        print(f"  ✓ Torture-30: AUC = {df['auc'].mean():.4f}")

    # Sort by version number (if possible) then by AUC
    def sort_key(m):
        name = m['name'].lower()
        # Extract version number
        if name.startswith('v') and name[1:].split('_')[0].isdigit():
            return (0, int(name[1:].split('_')[0]), -m['auc'])
        elif 'parsec' in name:
            return (1, 0, -m['auc'])
        elif 'torture' in name:
            return (2, 0, -m['auc'])
        else:
            return (3, 0, -m['auc'])

    models_sorted = sorted(models, key=sort_key)

    # Create DataFrame
    df = pd.DataFrame(models_sorted)

    print(f"\n{'=' * 80}")
    print("ALL MODELS WITH VALID AUC")
    print("=" * 80)
    print(f"{'#':<4} {'Model':<25} {'AUC':>8} {'Std':>10} {'Description':<40}")
    print("-" * 90)

    for i, row in df.iterrows():
        std_str = f"±{row['std']:.4f}" if pd.notna(row['std']) else "N/A"
        print(f"{i+1:<4} {row['name']:<25} {row['auc']:>8.4f} {std_str:>10} {row['desc'][:40]:<40}")

    # Create comprehensive bar chart
    print("\n\nCreating comprehensive visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(20, 14))

    # === CHART 1: All models bar chart ===
    ax1 = axes[0]

    # Color code by category
    colors = []
    for name in df['name']:
        name_lower = name.lower()
        if 'parsec' in name_lower:
            colors.append('#E74C3C')  # Red for PARSEC
        elif 'torture' in name_lower:
            colors.append('#27AE60')  # Green for validated
        elif any(x in name_lower for x in ['ensemble', 'stack', 'hybrid']):
            colors.append('#9B59B6')  # Purple for ensemble
        elif name_lower.startswith('v'):
            # Version number coloring
            try:
                ver = int(name_lower[1:].split('_')[0])
                if ver <= 15:
                    colors.append('#3498DB')  # Blue - early
                elif ver <= 30:
                    colors.append('#F39C12')  # Orange - mid
                else:
                    colors.append('#1ABC9C')  # Teal - late
            except:
                colors.append('#95A5A6')  # Gray
        else:
            colors.append('#95A5A6')  # Gray - other

    x = np.arange(len(df))
    bars = ax1.bar(x, df['auc'], color=colors, edgecolor='black', linewidth=0.5)

    # Add error bars where available
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(row['std']):
            ax1.errorbar(i, row['auc'], yerr=row['std'], fmt='none', color='black', capsize=3)

    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax1.text(i, row['auc'] + 0.01, f"{row['auc']:.3f}", ha='center', va='bottom',
                fontsize=7, rotation=90)

    # Reference lines
    ax1.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Target (0.90)')
    ax1.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5, label='Baseline (0.85)')

    ax1.set_xlabel('Model Version', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Complete Model History: V6 → PARSEC Leak-Free\nMRI-Crohn Atlas Treatment Classifier Evolution',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=8)
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # === CHART 2: Evolution line chart (versions only) ===
    ax2 = axes[1]

    # Filter to just version numbers for clean evolution
    version_models = df[df['name'].str.match(r'^v\d+', case=False, na=False)].copy()

    # Sort by version number
    def get_version_num(name):
        try:
            return int(name.lower()[1:].split('_')[0])
        except:
            return 999

    version_models['version_num'] = version_models['name'].apply(get_version_num)
    version_models = version_models.sort_values('version_num')

    # Plot
    x2 = range(len(version_models))
    ax2.plot(x2, version_models['auc'], 'o-', linewidth=2, markersize=8, color='#3498DB')
    ax2.fill_between(x2, version_models['auc'], alpha=0.3, color='#3498DB')

    # Mark special points
    max_idx = version_models['auc'].idxmax()
    max_row = version_models.loc[max_idx]
    ax2.scatter([list(version_models.index).index(max_idx)], [max_row['auc']],
                color='gold', s=200, zorder=5, marker='*', edgecolor='black', label=f"Best: {max_row['name']}")

    # Reference lines
    ax2.axhline(y=0.90, color='green', linestyle='--', linewidth=2, label='Target (0.90)')
    ax2.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5, label='Baseline (0.85)')

    ax2.set_xlabel('Model Version', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Version Evolution (V6 → V43): Performance Over Time', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(version_models['name'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0.65, 1.0)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / "model_history_ALL.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Chart saved: {output_path}")

    # Save CSV
    csv_path = DATA_DIR / "model_history_complete.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Data saved: {csv_path}")

    # Print summary stats
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total models extracted: {len(df)}")
    print(f"Best AUC: {df['auc'].max():.4f} ({df.loc[df['auc'].idxmax(), 'name']})")
    print(f"Worst AUC: {df['auc'].min():.4f} ({df.loc[df['auc'].idxmin(), 'name']})")
    print(f"Mean AUC: {df['auc'].mean():.4f}")
    print(f"Models above 0.90: {len(df[df['auc'] >= 0.90])}")
    print(f"Models above 0.85: {len(df[df['auc'] >= 0.85])}")

    # Best per category
    print(f"\n{'=' * 80}")
    print("BEST BY CATEGORY")
    print("=" * 80)

    versions = df[df['name'].str.match(r'^v\d+', case=False, na=False)]
    if len(versions) > 0:
        best_v = versions.loc[versions['auc'].idxmax()]
        print(f"Best Version Model: {best_v['name']} (AUC: {best_v['auc']:.4f})")

    parsecs = df[df['name'].str.contains('parsec', case=False, na=False)]
    if len(parsecs) > 0:
        best_c = parsecs.loc[parsecs['auc'].idxmax()]
        print(f"Best PARSEC Model: {best_c['name']} (AUC: {best_c['auc']:.4f})")

    print(f"\n{'=' * 80}")
    print("DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
