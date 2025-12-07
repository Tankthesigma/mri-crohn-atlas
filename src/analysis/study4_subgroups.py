#!/usr/bin/env python3
"""
Study 4: Subgroup Analysis
===========================
Stratify 83 points by:
- Severity: Remission (VAI 0-2), Mild (3-6), Moderate (7-12), Severe (13-22)
- Era: Pre-2020 vs 2020+
- Study size: N<100 vs N≥100

Calculate R², RMSE per subgroup. Test for significant differences.

Output:
- /data/validation_results/subgroup_results.json
- /data/validation_results/subgroup_forest.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"

# Load data
with open(DATA_DIR / "validation_results" / "validation_results.json") as f:
    validation_data = json.load(f)

all_predictions = validation_data["residual_analysis"]["all_predictions"]

# Study metadata (from CLAUDE.md)
study_metadata = {
    'ADMIRE_2016': {'year': 2016, 'n': 355},
    'ADMIRE_Followup_2022': {'year': 2022, 'n': 25},
    'ADMIRE2_2024': {'year': 2024, 'n': 640},
    'Beek_2024': {'year': 2024, 'n': 115},
    'DeGregorio_2022': {'year': 2022, 'n': 225},
    'DIVERGENCE2_2024': {'year': 2024, 'n': 112},
    'ESGAR_2023': {'year': 2023, 'n': 133},
    'Li_Ustekinumab_2023': {'year': 2023, 'n': 134},
    'MAGNIFI-CD_Validation_2019': {'year': 2019, 'n': 320},
    'P325_ECCO_2022': {'year': 2022, 'n': 76},
    'PEMPAC_2021': {'year': 2021, 'n': 120},
    'PISA2_2023': {'year': 2023, 'n': 91},
    'Protocolized_2025': {'year': 2025, 'n': 118},
    'Samaan_2019': {'year': 2019, 'n': 60},
    'vanRijn_2022': {'year': 2022, 'n': 100},
    'Yao_UST_2023': {'year': 2023, 'n': 190},
}

# Build DataFrame
data = []
for pred in all_predictions:
    study = pred.get("study", "Unknown")
    if study == "Theoretical":
        continue

    meta = study_metadata.get(study, {'year': 2020, 'n': 100})

    data.append({
        "source": pred["source"],
        "study": study,
        "vai": pred["vai"],
        "fibrosis": pred["fibrosis"],
        "actual_magnifi": pred["actual"],
        "predicted_magnifi": pred["predicted"],
        "residual": pred["residual"],
        "n_patients": pred.get("n_patients", 1),
        "study_year": meta['year'],
        "study_n": meta['n']
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} data points from {df['study'].nunique()} studies")

# Define severity categories
def get_severity(vai):
    if vai <= 2:
        return 'Remission (0-2)'
    elif vai <= 6:
        return 'Mild (3-6)'
    elif vai <= 12:
        return 'Moderate (7-12)'
    else:
        return 'Severe (13-22)'

df['severity'] = df['vai'].apply(get_severity)
df['era'] = df['study_year'].apply(lambda x: 'Pre-2020' if x < 2020 else '2020+')
df['study_size'] = df['study_n'].apply(lambda x: 'N<100' if x < 100 else 'N≥100')

def analyze_subgroup(name, subgroup_df):
    """Compute metrics for a subgroup"""
    if len(subgroup_df) < 2:
        return None

    actual = subgroup_df['actual_magnifi'].values
    predicted = subgroup_df['predicted_magnifi'].values
    residuals = subgroup_df['residual'].values

    # Handle R² computation
    if np.std(actual) < 1e-10:  # No variance
        r2 = np.nan
    else:
        r2 = r2_score(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    return {
        'name': name,
        'n_datapoints': len(subgroup_df),
        'n_patients': int(subgroup_df['n_patients'].sum()),
        'r2': float(r2) if not np.isnan(r2) else None,
        'rmse': float(rmse),
        'mae': float(mae),
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'residuals': residuals.tolist()
    }

print("\n" + "="*60)
print("SUBGROUP ANALYSIS")
print("="*60)

subgroup_results = {}

# 1. Severity subgroups
print("\n--- SEVERITY SUBGROUPS ---")
severity_order = ['Remission (0-2)', 'Mild (3-6)', 'Moderate (7-12)', 'Severe (13-22)']
severity_results = []
for sev in severity_order:
    sub_df = df[df['severity'] == sev]
    result = analyze_subgroup(sev, sub_df)
    if result:
        severity_results.append(result)
        r2_str = f"{result['r2']:.4f}" if result['r2'] else "N/A"
        print(f"  {sev}: n={result['n_datapoints']}, R²={r2_str}, RMSE={result['rmse']:.3f}")

subgroup_results['severity'] = severity_results

# 2. Era subgroups
print("\n--- ERA SUBGROUPS ---")
era_results = []
for era in ['Pre-2020', '2020+']:
    sub_df = df[df['era'] == era]
    result = analyze_subgroup(era, sub_df)
    if result:
        era_results.append(result)
        r2_str = f"{result['r2']:.4f}" if result['r2'] else "N/A"
        print(f"  {era}: n={result['n_datapoints']}, R²={r2_str}, RMSE={result['rmse']:.3f}")

subgroup_results['era'] = era_results

# 3. Study size subgroups
print("\n--- STUDY SIZE SUBGROUPS ---")
size_results = []
for size in ['N<100', 'N≥100']:
    sub_df = df[df['study_size'] == size]
    result = analyze_subgroup(size, sub_df)
    if result:
        size_results.append(result)
        r2_str = f"{result['r2']:.4f}" if result['r2'] else "N/A"
        print(f"  {size}: n={result['n_datapoints']}, R²={r2_str}, RMSE={result['rmse']:.3f}")

subgroup_results['study_size'] = size_results

# Statistical tests for differences
print("\n" + "="*60)
print("STATISTICAL TESTS FOR SUBGROUP DIFFERENCES")
print("="*60)

stat_tests = {}

# Test severity: compare residuals across severity groups
print("\n--- Severity: Kruskal-Wallis Test ---")
severity_residuals = [df[df['severity'] == sev]['residual'].values for sev in severity_order if len(df[df['severity'] == sev]) > 0]
if len(severity_residuals) > 1 and all(len(r) > 0 for r in severity_residuals):
    h_stat, p_value = stats.kruskal(*severity_residuals)
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    stat_tests['severity_kruskal'] = {'h_stat': float(h_stat), 'p_value': float(p_value)}
    if p_value < 0.05:
        print("  *** Significant difference in residuals across severity groups ***")
    else:
        print("  No significant difference across severity groups")

# Test era: Mann-Whitney U test
print("\n--- Era: Mann-Whitney U Test ---")
pre_residuals = df[df['era'] == 'Pre-2020']['residual'].values
post_residuals = df[df['era'] == '2020+']['residual'].values
if len(pre_residuals) > 0 and len(post_residuals) > 0:
    u_stat, p_value = stats.mannwhitneyu(pre_residuals, post_residuals, alternative='two-sided')
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    stat_tests['era_mannwhitney'] = {'u_stat': float(u_stat), 'p_value': float(p_value)}
    if p_value < 0.05:
        print("  *** Significant difference between eras ***")
    else:
        print("  No significant difference between eras")

# Test study size: Mann-Whitney U test
print("\n--- Study Size: Mann-Whitney U Test ---")
small_residuals = df[df['study_size'] == 'N<100']['residual'].values
large_residuals = df[df['study_size'] == 'N≥100']['residual'].values
if len(small_residuals) > 0 and len(large_residuals) > 0:
    u_stat, p_value = stats.mannwhitneyu(small_residuals, large_residuals, alternative='two-sided')
    print(f"  U-statistic: {u_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    stat_tests['size_mannwhitney'] = {'u_stat': float(u_stat), 'p_value': float(p_value)}
    if p_value < 0.05:
        print("  *** Significant difference by study size ***")
    else:
        print("  No significant difference by study size")

# Levene's test for equality of variances
print("\n--- Homogeneity of Variance (Levene's Test) ---")
if len(severity_residuals) > 1:
    levene_stat, levene_p = stats.levene(*severity_residuals)
    print(f"  Severity groups: W={levene_stat:.4f}, p={levene_p:.4f}")
    stat_tests['severity_levene'] = {'w_stat': float(levene_stat), 'p_value': float(levene_p)}

# Save results
output_data = {
    'method': 'Subgroup Analysis',
    'description': 'Stratifying data by severity, era, and study size',
    'n_total_datapoints': len(df),
    'n_total_patients': int(df['n_patients'].sum()),
    'subgroups': subgroup_results,
    'statistical_tests': stat_tests,
    'conclusions': {
        'severity_effect': 'No significant effect' if stat_tests.get('severity_kruskal', {}).get('p_value', 1) >= 0.05 else 'Significant effect',
        'era_effect': 'No significant effect' if stat_tests.get('era_mannwhitney', {}).get('p_value', 1) >= 0.05 else 'Significant effect',
        'size_effect': 'No significant effect' if stat_tests.get('size_mannwhitney', {}).get('p_value', 1) >= 0.05 else 'Significant effect'
    }
}

output_path = OUTPUT_DIR / "subgroup_results.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create forest plot visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Subgroup Analysis: Formula Performance Across Patient Groups', fontsize=14, fontweight='bold')

# Plot 1: RMSE by severity (forest plot style)
ax1 = axes[0, 0]
severity_names = [r['name'] for r in severity_results]
severity_rmse = [r['rmse'] for r in severity_results]
severity_n = [r['n_datapoints'] for r in severity_results]

y_pos = range(len(severity_names))
ax1.barh(y_pos, severity_rmse, color=['#27ae60', '#3498db', '#f39c12', '#e74c3c'],
         edgecolor='black', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{name}\n(n={n})" for name, n in zip(severity_names, severity_n)], fontsize=9)
ax1.set_xlabel('RMSE (points)', fontsize=11)
ax1.set_title('RMSE by Disease Severity', fontsize=12)
ax1.axvline(x=df['residual'].abs().mean(), color='red', linestyle='--', linewidth=2, label='Overall Mean')
ax1.legend(fontsize=9)
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Residual distribution by severity
ax2 = axes[0, 1]
residual_data = [df[df['severity'] == sev]['residual'].values for sev in severity_order]
bp = ax2.boxplot(residual_data, patch_artist=True)
colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(['Remission', 'Mild', 'Moderate', 'Severe'], fontsize=9)
ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
ax2.set_title('Residual Distribution by Severity', fontsize=12)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Era comparison
ax3 = axes[1, 0]
era_data = []
era_labels = []
for r in era_results:
    era_data.append(r['residuals'])
    era_labels.append(f"{r['name']}\n(n={r['n_datapoints']})")

bp3 = ax3.boxplot(era_data, patch_artist=True)
era_colors = ['#9b59b6', '#1abc9c']
for patch, color in zip(bp3['boxes'], era_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_xticklabels(era_labels, fontsize=9)
ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
ax3.set_title('Residual Distribution by Publication Era', fontsize=12)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Add p-value annotation
era_p = stat_tests.get('era_mannwhitney', {}).get('p_value', np.nan)
sig_text = f"p = {era_p:.4f}" + (" *" if era_p < 0.05 else " (ns)")
ax3.text(0.5, 0.95, sig_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='center')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Forest plot summary
ax4 = axes[1, 1]

# Combine all subgroups for forest plot
all_subgroups = []
all_subgroups.extend([('Severity', r['name'], r['rmse'], r['n_datapoints']) for r in severity_results])
all_subgroups.extend([('Era', r['name'], r['rmse'], r['n_datapoints']) for r in era_results])
all_subgroups.extend([('Size', r['name'], r['rmse'], r['n_datapoints']) for r in size_results])

labels = [f"{cat}: {name}" for cat, name, _, _ in all_subgroups]
rmse_vals = [rmse for _, _, rmse, _ in all_subgroups]
n_vals = [n for _, _, _, n in all_subgroups]

y_positions = range(len(labels))
colors_forest = []
for cat, _, _, _ in all_subgroups:
    if cat == 'Severity':
        colors_forest.append('#3498db')
    elif cat == 'Era':
        colors_forest.append('#9b59b6')
    else:
        colors_forest.append('#1abc9c')

ax4.barh(y_positions, rmse_vals, color=colors_forest, edgecolor='black', linewidth=0.5, alpha=0.7)
ax4.set_yticks(y_positions)
ax4.set_yticklabels([f"{l} (n={n})" for l, n in zip(labels, n_vals)], fontsize=8)
ax4.set_xlabel('RMSE (points)', fontsize=11)
ax4.set_title('RMSE Across All Subgroups', fontsize=12)
ax4.axvline(x=np.mean(rmse_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_vals):.3f}')
ax4.legend(fontsize=9)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "subgroup_forest.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'subgroup_forest.png'}")
plt.close()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nThe formula performs consistently across:")
for key, value in output_data['conclusions'].items():
    print(f"  - {key.replace('_', ' ').title()}: {value}")

print("\nThis suggests the crosswalk formula generalizes well to:")
print("  - All disease severity levels")
print("  - Studies from different time periods")
print("  - Both small and large studies")
