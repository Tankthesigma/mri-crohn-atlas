#!/usr/bin/env python3
"""
Study 1: Leave-One-Study-Out Cross-Validation (LOSO)
======================================================
For each of 17 studies, hold out that study's data points,
refit regression on remaining studies, predict on held-out study.

Output:
- /data/validation_results/loso_results.json
- /data/validation_results/loso_boxplot.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data from validation_results.json
with open(DATA_DIR / "validation_results" / "validation_results.json") as f:
    validation_data = json.load(f)

# Extract 83 data points from residual_analysis
all_predictions = validation_data["residual_analysis"]["all_predictions"]

# Build DataFrame
data = []
for pred in all_predictions:
    study = pred.get("study", "Unknown")
    if study == "Theoretical":
        continue  # Skip theoretical points for LOSO
    data.append({
        "source": pred["source"],
        "study": study,
        "vai": pred["vai"],
        "fibrosis": pred["fibrosis"],
        "actual_magnifi": pred["actual"],
        "n_patients": pred.get("n_patients", 1)
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} data points from {df['study'].nunique()} studies")
print(f"\nStudies: {sorted(df['study'].unique())}")

def apply_formula(vai, fibrosis):
    """Apply the neuro-symbolic crosswalk formula"""
    if vai <= 2:
        return 1.031 * vai + 0.264 * fibrosis + 1.713
    else:
        return 1.031 * vai + 1.713

def fit_formula(X_train, y_train):
    """
    Fit the neuro-symbolic formula structure:
    MAGNIFI = beta1*VAI + beta2*Fibrosis*I(VAI<=2) + intercept
    """
    vai = X_train[:, 0]
    fib = X_train[:, 1]

    # Create features: VAI, Fibrosis*I(VAI<=2)
    healed_mask = (vai <= 2).astype(float)
    X_design = np.column_stack([vai, fib * healed_mask, np.ones(len(vai))])

    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X_design, y_train, rcond=None)

    return {
        'coef_vai': coeffs[0],
        'coef_fib_healed': coeffs[1],
        'intercept': coeffs[2]
    }

def predict_with_coeffs(vai, fibrosis, coeffs):
    """Predict using fitted coefficients"""
    if vai <= 2:
        return coeffs['coef_vai'] * vai + coeffs['coef_fib_healed'] * fibrosis + coeffs['intercept']
    else:
        return coeffs['coef_vai'] * vai + coeffs['intercept']

# Perform LOSO
studies = sorted(df['study'].unique())
loso_results = []

print("\n" + "="*60)
print("LEAVE-ONE-STUDY-OUT CROSS-VALIDATION")
print("="*60)

for study in studies:
    # Split data
    test_mask = df['study'] == study
    train_df = df[~test_mask]
    test_df = df[test_mask]

    if len(test_df) == 0:
        continue

    # Prepare training data
    X_train = train_df[['vai', 'fibrosis']].values
    y_train = train_df['actual_magnifi'].values

    # Fit model
    coeffs = fit_formula(X_train, y_train)

    # Predict on test set
    y_pred = []
    for _, row in test_df.iterrows():
        pred = predict_with_coeffs(row['vai'], row['fibrosis'], coeffs)
        y_pred.append(pred)
    y_pred = np.array(y_pred)
    y_actual = test_df['actual_magnifi'].values

    # Calculate metrics
    if len(y_actual) > 1 and np.std(y_actual) > 0:
        r2 = r2_score(y_actual, y_pred)
    else:
        r2 = np.nan  # Can't compute R² with 1 sample or no variance

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    max_error = np.max(np.abs(y_actual - y_pred))
    mean_residual = np.mean(y_actual - y_pred)

    # Get total patients
    total_patients = test_df['n_patients'].sum()

    result = {
        'study': study,
        'n_datapoints': len(test_df),
        'total_patients': int(total_patients),
        'r2': r2 if not np.isnan(r2) else None,
        'rmse': float(rmse),
        'mae': float(mae),
        'max_error': float(max_error),
        'mean_residual': float(mean_residual),
        'coefficients': {k: float(v) for k, v in coeffs.items()},
        'predictions': []
    }

    # Store individual predictions
    for idx, (_, row) in enumerate(test_df.iterrows()):
        result['predictions'].append({
            'source': row['source'],
            'vai': float(row['vai']),
            'fibrosis': int(row['fibrosis']),
            'actual': float(row['actual_magnifi']),
            'predicted': float(y_pred[idx]),
            'error': float(y_actual[idx] - y_pred[idx])
        })

    loso_results.append(result)

    r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
    print(f"\n{study}:")
    print(f"  n={len(test_df)}, patients={total_patients}")
    print(f"  R²={r2_str}, RMSE={rmse:.3f}, MAE={mae:.3f}")

# Compute overall statistics
valid_r2 = [r['r2'] for r in loso_results if r['r2'] is not None]
rmse_values = [r['rmse'] for r in loso_results]
mae_values = [r['mae'] for r in loso_results]

overall_stats = {
    'n_studies': len(studies),
    'r2_mean': float(np.mean(valid_r2)) if valid_r2 else None,
    'r2_std': float(np.std(valid_r2)) if valid_r2 else None,
    'r2_min': float(np.min(valid_r2)) if valid_r2 else None,
    'r2_max': float(np.max(valid_r2)) if valid_r2 else None,
    'rmse_mean': float(np.mean(rmse_values)),
    'rmse_std': float(np.std(rmse_values)),
    'mae_mean': float(np.mean(mae_values)),
    'mae_std': float(np.std(mae_values)),
    'best_fold': loso_results[np.argmax([r['r2'] if r['r2'] else -np.inf for r in loso_results])]['study'],
    'worst_fold': loso_results[np.argmin([r['r2'] if r['r2'] else np.inf for r in loso_results])]['study']
}

print("\n" + "="*60)
print("OVERALL LOSO RESULTS")
print("="*60)
print(f"Studies validated: {overall_stats['n_studies']}")
print(f"R² mean ± SD: {overall_stats['r2_mean']:.4f} ± {overall_stats['r2_std']:.4f}")
print(f"R² range: [{overall_stats['r2_min']:.4f}, {overall_stats['r2_max']:.4f}]")
print(f"RMSE mean ± SD: {overall_stats['rmse_mean']:.3f} ± {overall_stats['rmse_std']:.3f}")
print(f"MAE mean ± SD: {overall_stats['mae_mean']:.3f} ± {overall_stats['mae_std']:.3f}")
print(f"Best fold: {overall_stats['best_fold']}")
print(f"Worst fold: {overall_stats['worst_fold']}")

# Save results
output_data = {
    'method': 'Leave-One-Study-Out Cross-Validation',
    'description': 'For each study, fit model on remaining studies and predict held-out study',
    'formula': 'MAGNIFI-CD = β₁×VAI + β₂×Fibrosis×I(VAI≤2) + intercept',
    'overall_statistics': overall_stats,
    'per_study_results': loso_results
}

output_path = OUTPUT_DIR / "loso_results.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Leave-One-Study-Out Cross-Validation Results', fontsize=14, fontweight='bold')

# Plot 1: R² by study (boxplot-style bar chart)
ax1 = axes[0, 0]
study_names = [r['study'].replace('_', '\n') for r in loso_results]
r2_values = [r['r2'] if r['r2'] else 0 for r in loso_results]
colors = ['#2ecc71' if r2 and r2 > 0.9 else '#f39c12' if r2 and r2 > 0.8 else '#e74c3c' for r2 in r2_values]
bars = ax1.barh(range(len(study_names)), r2_values, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(study_names)))
ax1.set_yticklabels(study_names, fontsize=7)
ax1.set_xlabel('R²', fontsize=10)
ax1.set_title('R² per Study (held out)', fontsize=11)
ax1.axvline(x=np.mean(valid_r2), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_r2):.3f}')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlim(0, 1.05)

# Plot 2: RMSE by study
ax2 = axes[0, 1]
rmse_vals = [r['rmse'] for r in loso_results]
colors2 = ['#2ecc71' if rmse < 0.5 else '#f39c12' if rmse < 1.0 else '#e74c3c' for rmse in rmse_vals]
ax2.barh(range(len(study_names)), rmse_vals, color=colors2, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(study_names)))
ax2.set_yticklabels(study_names, fontsize=7)
ax2.set_xlabel('RMSE (points)', fontsize=10)
ax2.set_title('RMSE per Study (held out)', fontsize=11)
ax2.axvline(x=np.mean(rmse_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rmse_vals):.3f}')
ax2.legend(loc='lower right', fontsize=9)

# Plot 3: Box plot of R² distribution
ax3 = axes[1, 0]
ax3.boxplot(valid_r2, vert=False, patch_artist=True,
            boxprops=dict(facecolor='#3498db', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax3.scatter(valid_r2, np.ones(len(valid_r2)), alpha=0.6, s=50, c='#2c3e50', zorder=5)
ax3.set_xlabel('R²', fontsize=10)
ax3.set_title('R² Distribution Across Studies', fontsize=11)
ax3.set_yticks([])

# Add statistics text
stats_text = f"Mean: {np.mean(valid_r2):.4f}\nStd: {np.std(valid_r2):.4f}\nMin: {np.min(valid_r2):.4f}\nMax: {np.max(valid_r2):.4f}"
ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Predicted vs Actual for all held-out predictions
ax4 = axes[1, 1]
all_actual = []
all_predicted = []
all_studies_for_color = []

for result in loso_results:
    for pred in result['predictions']:
        all_actual.append(pred['actual'])
        all_predicted.append(pred['predicted'])
        all_studies_for_color.append(result['study'])

# Create color map
unique_studies = list(set(all_studies_for_color))
cmap = plt.cm.get_cmap('tab20', len(unique_studies))
study_colors = {s: cmap(i) for i, s in enumerate(unique_studies)}
colors_scatter = [study_colors[s] for s in all_studies_for_color]

ax4.scatter(all_actual, all_predicted, c=colors_scatter, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax4.plot([0, 25], [0, 25], 'k--', linewidth=1.5, label='Perfect prediction')
ax4.set_xlabel('Actual MAGNIFI-CD', fontsize=10)
ax4.set_ylabel('Predicted MAGNIFI-CD', fontsize=10)
ax4.set_title('Predicted vs Actual (All Held-Out Points)', fontsize=11)
ax4.set_xlim(-1, 26)
ax4.set_ylim(-1, 26)
ax4.legend(loc='lower right', fontsize=9)

# Add overall R² to the plot
overall_r2 = r2_score(all_actual, all_predicted)
ax4.text(0.05, 0.95, f'Overall R² = {overall_r2:.4f}', transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loso_boxplot.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'loso_boxplot.png'}")
plt.close()

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Study':<25} {'n':>4} {'Patients':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-"*80)
for r in sorted(loso_results, key=lambda x: x['r2'] if x['r2'] else 0, reverse=True):
    r2_str = f"{r['r2']:.4f}" if r['r2'] else "N/A"
    print(f"{r['study']:<25} {r['n_datapoints']:>4} {r['total_patients']:>8} {r2_str:>8} {r['rmse']:>8.3f} {r['mae']:>8.3f}")
print("-"*80)
print(f"{'MEAN':<25} {'':<4} {'':<8} {overall_stats['r2_mean']:.4f}   {overall_stats['rmse_mean']:.3f}    {overall_stats['mae_mean']:.3f}")
