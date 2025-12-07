#!/usr/bin/env python3
"""
Study 2: Ablation Study
========================
Test formula variants to show the full model is optimal.

Variants tested:
- Full: MAGNIFI = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713
- No fibrosis: MAGNIFI = β×VAI + c
- No coefficients: MAGNIFI = VAI + c
- Fibrosis always: MAGNIFI = β₁×VAI + β₂×Fibrosis + c
- Different thresholds: VAI≤1, VAI≤3, VAI≤4

Output:
- /data/validation_results/ablation_results.json
- /data/validation_results/ablation_comparison.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"

# Load data
with open(DATA_DIR / "validation_results" / "validation_results.json") as f:
    validation_data = json.load(f)

all_predictions = validation_data["residual_analysis"]["all_predictions"]

# Build DataFrame (exclude theoretical)
data = []
for pred in all_predictions:
    study = pred.get("study", "Unknown")
    if study == "Theoretical":
        continue
    data.append({
        "vai": pred["vai"],
        "fibrosis": pred["fibrosis"],
        "actual_magnifi": pred["actual"],
        "n_patients": pred.get("n_patients", 1)
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} data points")

# Extract arrays
vai = df['vai'].values
fibrosis = df['fibrosis'].values
actual = df['actual_magnifi'].values
n = len(df)

def compute_aic_bic(y_true, y_pred, n_params, n_samples):
    """Compute AIC and BIC"""
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    mse = sse / n_samples

    # Log-likelihood (assuming Gaussian errors)
    log_likelihood = -n_samples/2 * (np.log(2*np.pi) + np.log(mse) + 1)

    # AIC and BIC
    aic = 2*n_params - 2*log_likelihood
    bic = n_params*np.log(n_samples) - 2*log_likelihood

    return aic, bic

def fit_and_evaluate(name, design_matrix, n_params, description):
    """Fit a model and compute metrics"""
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(design_matrix, actual, rcond=None)

    # Predict
    y_pred = design_matrix @ coeffs

    # Metrics
    r2 = r2_score(actual, y_pred)
    rmse = np.sqrt(mean_squared_error(actual, y_pred))
    mae = mean_absolute_error(actual, y_pred)
    aic, bic = compute_aic_bic(actual, y_pred, n_params, n)

    return {
        'name': name,
        'description': description,
        'coefficients': coeffs.tolist(),
        'n_params': n_params,
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'aic': float(aic),
        'bic': float(bic),
        'predictions': y_pred.tolist()
    }

# Define models
models = []

# Model 1: Full model (with threshold VAI≤2)
print("\n" + "="*60)
print("ABLATION STUDY: TESTING FORMULA VARIANTS")
print("="*60)

# Full model: MAGNIFI = β₁×VAI + β₂×Fibrosis×I(VAI≤2) + intercept
healed_mask_2 = (vai <= 2).astype(float)
X_full = np.column_stack([vai, fibrosis * healed_mask_2, np.ones(n)])
models.append(fit_and_evaluate(
    'Full Model (VAI≤2)',
    X_full, 3,
    'MAGNIFI = β₁×VAI + β₂×Fibrosis×I(VAI≤2) + c'
))

# Model 2: No fibrosis term
X_no_fib = np.column_stack([vai, np.ones(n)])
models.append(fit_and_evaluate(
    'No Fibrosis',
    X_no_fib, 2,
    'MAGNIFI = β×VAI + c'
))

# Model 3: No coefficients (simple offset)
X_simple = np.column_stack([vai, np.ones(n)])
# Force coefficient to be 1
X_simple_forced = np.column_stack([np.ones(n)])  # Just intercept
simple_residual = actual - vai
intercept_only = np.mean(simple_residual)
y_pred_simple = vai + intercept_only
r2_simple = r2_score(actual, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(actual, y_pred_simple))
mae_simple = mean_absolute_error(actual, y_pred_simple)
aic_simple, bic_simple = compute_aic_bic(actual, y_pred_simple, 1, n)
models.append({
    'name': 'Identity + Offset',
    'description': 'MAGNIFI = VAI + c (forced β=1)',
    'coefficients': [1.0, intercept_only],
    'n_params': 1,
    'r2': float(r2_simple),
    'rmse': float(rmse_simple),
    'mae': float(mae_simple),
    'aic': float(aic_simple),
    'bic': float(bic_simple),
    'predictions': y_pred_simple.tolist()
})

# Model 4: Fibrosis always (no indicator function)
X_fib_always = np.column_stack([vai, fibrosis, np.ones(n)])
models.append(fit_and_evaluate(
    'Fibrosis Always',
    X_fib_always, 3,
    'MAGNIFI = β₁×VAI + β₂×Fibrosis + c'
))

# Model 5-7: Different thresholds
for threshold in [1, 3, 4]:
    healed_mask = (vai <= threshold).astype(float)
    X_thresh = np.column_stack([vai, fibrosis * healed_mask, np.ones(n)])
    models.append(fit_and_evaluate(
        f'Threshold VAI≤{threshold}',
        X_thresh, 3,
        f'MAGNIFI = β₁×VAI + β₂×Fibrosis×I(VAI≤{threshold}) + c'
    ))

# Model 8: Quadratic term
X_quad = np.column_stack([vai, vai**2, np.ones(n)])
models.append(fit_and_evaluate(
    'Quadratic',
    X_quad, 3,
    'MAGNIFI = β₁×VAI + β₂×VAI² + c'
))

# Model 9: Full model with quadratic
X_full_quad = np.column_stack([vai, vai**2, fibrosis * healed_mask_2, np.ones(n)])
models.append(fit_and_evaluate(
    'Full + Quadratic',
    X_full_quad, 4,
    'MAGNIFI = β₁×VAI + β₂×VAI² + β₃×Fibrosis×I(VAI≤2) + c'
))

# Print results table
print(f"\n{'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'AIC':>10} {'BIC':>10} {'Params':>6}")
print("-"*85)
for m in sorted(models, key=lambda x: x['bic']):
    print(f"{m['name']:<25} {m['r2']:>8.4f} {m['rmse']:>8.3f} {m['mae']:>8.3f} {m['aic']:>10.2f} {m['bic']:>10.2f} {m['n_params']:>6}")

# Find best model by different criteria
best_r2 = max(models, key=lambda x: x['r2'])
best_aic = min(models, key=lambda x: x['aic'])
best_bic = min(models, key=lambda x: x['bic'])
best_rmse = min(models, key=lambda x: x['rmse'])

print("\n" + "="*60)
print("BEST MODELS BY CRITERION")
print("="*60)
print(f"Best R²:   {best_r2['name']} (R² = {best_r2['r2']:.4f})")
print(f"Best AIC:  {best_aic['name']} (AIC = {best_aic['aic']:.2f})")
print(f"Best BIC:  {best_bic['name']} (BIC = {best_bic['bic']:.2f})")
print(f"Best RMSE: {best_rmse['name']} (RMSE = {best_rmse['rmse']:.3f})")

# Analysis: is the full model optimal?
full_model = models[0]
print("\n" + "="*60)
print("ANALYSIS: IS THE FULL MODEL OPTIMAL?")
print("="*60)

# Compare to simpler models
no_fib = models[1]
delta_r2 = full_model['r2'] - no_fib['r2']
delta_bic = no_fib['bic'] - full_model['bic']  # Positive means full model is better
print(f"\nFull Model vs No Fibrosis:")
print(f"  ΔR² = +{delta_r2:.4f} (fibrosis term adds explanatory power)")
print(f"  ΔBIC = {delta_bic:.2f} ({'favors full model' if delta_bic > 0 else 'favors simpler model'})")

# Compare to fibrosis always
fib_always = models[3]
delta_r2_fa = full_model['r2'] - fib_always['r2']
delta_bic_fa = fib_always['bic'] - full_model['bic']
print(f"\nFull Model vs Fibrosis Always:")
print(f"  ΔR² = {delta_r2_fa:+.4f}")
print(f"  ΔBIC = {delta_bic_fa:.2f} ({'indicator function is beneficial' if delta_bic_fa > 0 else 'always applying fibrosis is better'})")

# Threshold comparison
threshold_models = [m for m in models if 'Threshold' in m['name']]
print(f"\nThreshold Comparison:")
for tm in threshold_models:
    print(f"  {tm['name']}: R²={tm['r2']:.4f}, BIC={tm['bic']:.2f}")

# Save results
output_data = {
    'method': 'Ablation Study',
    'description': 'Comparing formula variants to validate the full model',
    'full_formula': 'MAGNIFI-CD = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713',
    'n_datapoints': n,
    'models': models,
    'best_by_criterion': {
        'r2': best_r2['name'],
        'aic': best_aic['name'],
        'bic': best_bic['name'],
        'rmse': best_rmse['name']
    },
    'conclusion': {
        'optimal_model': 'Full Model (VAI≤2)' if best_bic['name'] == 'Full Model (VAI≤2)' else best_bic['name'],
        'fibrosis_term_value': f"Adds {delta_r2:.4f} to R²",
        'indicator_function_value': f"BIC difference of {delta_bic_fa:.2f} vs always applying fibrosis"
    }
}

output_path = OUTPUT_DIR / "ablation_results.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Ablation Study: Formula Variant Comparison', fontsize=14, fontweight='bold')

# Plot 1: R² comparison
ax1 = axes[0, 0]
model_names = [m['name'] for m in sorted(models, key=lambda x: x['r2'], reverse=True)]
r2_values = [m['r2'] for m in sorted(models, key=lambda x: x['r2'], reverse=True)]
colors = ['#27ae60' if 'Full' in name and 'Quadratic' not in name else '#3498db' for name in model_names]
bars = ax1.barh(range(len(model_names)), r2_values, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(model_names)))
ax1.set_yticklabels(model_names, fontsize=9)
ax1.set_xlabel('R²', fontsize=10)
ax1.set_title('R² by Model (higher is better)', fontsize=11)
ax1.set_xlim(0.9, 1.0)
ax1.axvline(x=full_model['r2'], color='green', linestyle='--', alpha=0.7, linewidth=2, label='Full Model')
ax1.legend(fontsize=9)

# Plot 2: BIC comparison (lower is better)
ax2 = axes[0, 1]
model_names_bic = [m['name'] for m in sorted(models, key=lambda x: x['bic'])]
bic_values = [m['bic'] for m in sorted(models, key=lambda x: x['bic'])]
colors2 = ['#27ae60' if 'Full' in name and 'Quadratic' not in name else '#3498db' for name in model_names_bic]
ax2.barh(range(len(model_names_bic)), bic_values, color=colors2, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(model_names_bic)))
ax2.set_yticklabels(model_names_bic, fontsize=9)
ax2.set_xlabel('BIC', fontsize=10)
ax2.set_title('BIC by Model (lower is better)', fontsize=11)
ax2.axvline(x=full_model['bic'], color='green', linestyle='--', alpha=0.7, linewidth=2, label='Full Model')
ax2.legend(fontsize=9)

# Plot 3: RMSE comparison
ax3 = axes[1, 0]
model_names_rmse = [m['name'] for m in sorted(models, key=lambda x: x['rmse'])]
rmse_values = [m['rmse'] for m in sorted(models, key=lambda x: x['rmse'])]
colors3 = ['#27ae60' if 'Full' in name and 'Quadratic' not in name else '#3498db' for name in model_names_rmse]
ax3.barh(range(len(model_names_rmse)), rmse_values, color=colors3, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(model_names_rmse)))
ax3.set_yticklabels(model_names_rmse, fontsize=9)
ax3.set_xlabel('RMSE (points)', fontsize=10)
ax3.set_title('RMSE by Model (lower is better)', fontsize=11)
ax3.axvline(x=full_model['rmse'], color='green', linestyle='--', alpha=0.7, linewidth=2, label='Full Model')
ax3.legend(fontsize=9)

# Plot 4: Residual comparison for key models
ax4 = axes[1, 1]

# Get predictions for key models
full_preds = np.array(models[0]['predictions'])
no_fib_preds = np.array(models[1]['predictions'])
simple_preds = np.array(models[2]['predictions'])

residuals_full = actual - full_preds
residuals_no_fib = actual - no_fib_preds
residuals_simple = actual - simple_preds

positions = [1, 2, 3]
bp = ax4.boxplot([residuals_full, residuals_no_fib, residuals_simple],
                  positions=positions, patch_artist=True, widths=0.6)

colors4 = ['#27ae60', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors4):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_xticks(positions)
ax4.set_xticklabels(['Full Model', 'No Fibrosis', 'Identity + Offset'], fontsize=9)
ax4.set_ylabel('Residual (Actual - Predicted)', fontsize=10)
ax4.set_title('Residual Distribution by Model', fontsize=11)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_comparison.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'ablation_comparison.png'}")
plt.close()

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if best_bic['name'] == 'Full Model (VAI≤2)':
    print("The FULL MODEL with threshold VAI≤2 is OPTIMAL by BIC.")
    print("This validates the formula: MAGNIFI = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713")
else:
    print(f"The optimal model by BIC is: {best_bic['name']}")
    print(f"Consider if this model should replace the full model.")
