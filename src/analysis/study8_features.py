#!/usr/bin/env python3
"""
Study 8: Feature Importance Analysis
=====================================
For crosswalk:
- Partial R² for each term (VAI, Fibrosis, interaction)
- Sensitivity: ∂MAGNIFI/∂VAI, ∂MAGNIFI/∂Fibrosis

For parser:
- Correlate missing features with prediction error
- Rank features by importance

Output:
- /data/validation_results/feature_importance.json
- /data/validation_results/feature_ranking.png
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "validation_results"

# Load data
with open(DATA_DIR / "validation_results" / "validation_results.json") as f:
    validation_data = json.load(f)

# Load parser validation data
with open(DATA_DIR / "parser_tests" / "validation_results.json") as f:
    parser_data = json.load(f)

# Build crosswalk DataFrame
all_predictions = validation_data["residual_analysis"]["all_predictions"]
data = []
for pred in all_predictions:
    study = pred.get("study", "Unknown")
    if study == "Theoretical":
        continue
    data.append({
        "vai": pred["vai"],
        "fibrosis": pred["fibrosis"],
        "actual_magnifi": pred["actual"],
        "predicted_magnifi": pred["predicted"],
        "residual": pred["residual"]
    })

df = pd.DataFrame(data)
print(f"Loaded {len(df)} crosswalk data points")

# Extract arrays
vai = df['vai'].values
fibrosis = df['fibrosis'].values
actual = df['actual_magnifi'].values
n = len(df)

print("\n" + "="*70)
print("FEATURE IMPORTANCE: CROSSWALK FORMULA")
print("="*70)

# Formula: MAGNIFI = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713

# 1. Partial R² analysis - what each term contributes
print("\n--- Partial R² Analysis ---")

# Baseline: intercept only
y_mean = np.mean(actual)
ss_total = np.sum((actual - y_mean)**2)

# Model 1: VAI only
vai_model = LinearRegression().fit(vai.reshape(-1, 1), actual)
vai_pred = vai_model.predict(vai.reshape(-1, 1))
ss_vai = np.sum((actual - vai_pred)**2)
r2_vai = 1 - ss_vai / ss_total

# Model 2: VAI + Fibrosis (no interaction)
X_vai_fib = np.column_stack([vai, fibrosis])
vai_fib_model = LinearRegression().fit(X_vai_fib, actual)
vai_fib_pred = vai_fib_model.predict(X_vai_fib)
ss_vai_fib = np.sum((actual - vai_fib_pred)**2)
r2_vai_fib = 1 - ss_vai_fib / ss_total

# Model 3: Full model (VAI + Fibrosis×I(VAI≤2))
healed_mask = (vai <= 2).astype(float)
X_full = np.column_stack([vai, fibrosis * healed_mask])
full_model = LinearRegression().fit(X_full, actual)
full_pred = full_model.predict(X_full)
ss_full = np.sum((actual - full_pred)**2)
r2_full = 1 - ss_full / ss_total

print(f"VAI only: R² = {r2_vai:.4f}")
print(f"VAI + Fibrosis: R² = {r2_vai_fib:.4f}")
print(f"Full model: R² = {r2_full:.4f}")

# Partial R² contributions
partial_vai = r2_vai
partial_fibrosis = r2_vai_fib - r2_vai
partial_interaction = r2_full - r2_vai_fib

print(f"\nPartial R² contributions:")
print(f"  VAI: {partial_vai:.4f} ({partial_vai/r2_full*100:.1f}% of explained variance)")
print(f"  Fibrosis: {partial_fibrosis:.4f} ({partial_fibrosis/r2_full*100:.1f}%)")
print(f"  Interaction term: {partial_interaction:.4f} ({partial_interaction/r2_full*100:.1f}%)")

# 2. Sensitivity analysis (partial derivatives)
print("\n--- Sensitivity Analysis (Partial Derivatives) ---")

# ∂MAGNIFI/∂VAI = 1.031 (constant)
# ∂MAGNIFI/∂Fibrosis = 0.264 if VAI≤2, else 0

print("\nFrom formula: MAGNIFI = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713")
print(f"  ∂MAGNIFI/∂VAI = 1.031 (for all VAI)")
print(f"  ∂MAGNIFI/∂Fibrosis = 0.264 (if VAI≤2), 0 (if VAI>2)")

# Calculate empirical sensitivities
# Sensitivity at different VAI levels
vai_levels = [0, 2, 5, 10, 15, 20]
print(f"\nEmpirical sensitivity by VAI level:")
for vai_level in vai_levels:
    # Find nearby points
    nearby = df[(df['vai'] >= vai_level - 1) & (df['vai'] <= vai_level + 1)]
    if len(nearby) > 2:
        local_model = LinearRegression().fit(nearby[['vai']].values, nearby['actual_magnifi'].values)
        sensitivity = local_model.coef_[0]
        print(f"  VAI={vai_level}: ∂MAGNIFI/∂VAI ≈ {sensitivity:.3f}")

# 3. Parser feature importance
print("\n" + "="*70)
print("FEATURE IMPORTANCE: PARSER")
print("="*70)

# Analyze which extracted features correlate with errors
parser_features = ['fistula_count', 'fistula_type', 't2_hyperintensity', 'extension',
                   'collections_abscesses', 'rectal_wall_involvement', 'inflammatory_mass',
                   'predominant_feature']

feature_importance = {}
print("\n--- Feature Accuracy vs Error Correlation ---")

for feature in parser_features:
    correct_count = 0
    incorrect_count = 0
    errors_when_correct = []
    errors_when_incorrect = []

    for result in parser_data['results']:
        total_error = result['vai_error'] + result['magnifi_error']

        if feature in result['features_correct']:
            if result['features_correct'][feature]:
                correct_count += 1
                errors_when_correct.append(total_error)
            else:
                incorrect_count += 1
                errors_when_incorrect.append(total_error)

    accuracy = correct_count / (correct_count + incorrect_count) if (correct_count + incorrect_count) > 0 else 0
    mean_error_correct = np.mean(errors_when_correct) if errors_when_correct else 0
    mean_error_incorrect = np.mean(errors_when_incorrect) if errors_when_incorrect else 0

    # Importance = accuracy × error_impact
    error_impact = mean_error_incorrect - mean_error_correct if errors_when_incorrect else 0
    importance = accuracy * (1 + abs(error_impact) / 10)  # Normalize

    feature_importance[feature] = {
        'accuracy': float(accuracy),
        'n_correct': correct_count,
        'n_incorrect': incorrect_count,
        'mean_error_when_correct': float(mean_error_correct),
        'mean_error_when_incorrect': float(mean_error_incorrect),
        'error_impact': float(error_impact),
        'importance_score': float(importance)
    }

    print(f"\n{feature}:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Error when correct: {mean_error_correct:.2f}")
    print(f"  Error when incorrect: {mean_error_incorrect:.2f}")
    print(f"  Importance score: {importance:.3f}")

# Rank features by importance
ranked_features = sorted(feature_importance.items(), key=lambda x: x[1]['importance_score'], reverse=True)

print("\n--- Feature Ranking by Importance ---")
for i, (feat, info) in enumerate(ranked_features, 1):
    print(f"{i}. {feat}: {info['importance_score']:.3f} (accuracy: {info['accuracy']:.1%})")

# Minimum viable report specification
print("\n" + "="*70)
print("MINIMUM VIABLE REPORT SPECIFICATION")
print("="*70)

# Features with >90% accuracy are essential
essential_features = [f for f, info in feature_importance.items() if info['accuracy'] >= 0.9]
important_features = [f for f, info in feature_importance.items() if 0.8 <= info['accuracy'] < 0.9]
optional_features = [f for f, info in feature_importance.items() if info['accuracy'] < 0.8]

print("\nFor accurate scoring, reports MUST include:")
for feat in essential_features:
    print(f"  ✓ {feat}")

print("\nHighly recommended:")
for feat in important_features:
    print(f"  ○ {feat}")

print("\nOptional (may be inferred):")
for feat in optional_features:
    print(f"  · {feat}")

# Save results
output_data = {
    'method': 'Feature Importance Analysis',
    'description': 'Analyzing which features contribute most to predictions',
    'crosswalk_analysis': {
        'formula': 'MAGNIFI-CD = 1.031×VAI + 0.264×Fibrosis×I(VAI≤2) + 1.713',
        'partial_r2': {
            'vai_only': float(r2_vai),
            'vai_plus_fibrosis': float(r2_vai_fib),
            'full_model': float(r2_full)
        },
        'partial_r2_contributions': {
            'vai': float(partial_vai),
            'fibrosis': float(partial_fibrosis),
            'interaction': float(partial_interaction)
        },
        'contribution_percentages': {
            'vai': float(partial_vai/r2_full*100),
            'fibrosis': float(partial_fibrosis/r2_full*100),
            'interaction': float(partial_interaction/r2_full*100)
        },
        'sensitivities': {
            'd_magnifi_d_vai': 1.031,
            'd_magnifi_d_fibrosis_healed': 0.264,
            'd_magnifi_d_fibrosis_active': 0
        }
    },
    'parser_analysis': {
        'feature_importance': feature_importance,
        'ranked_features': [{'feature': f, **info} for f, info in ranked_features]
    },
    'minimum_viable_report': {
        'essential': essential_features,
        'recommended': important_features,
        'optional': optional_features
    }
}

output_path = OUTPUT_DIR / "feature_importance.json"
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Plot 1: Partial R² contributions (Crosswalk)
ax1 = axes[0, 0]
terms = ['VAI', 'Fibrosis', 'Interaction\n(Fibrosis×I(VAI≤2))']
contributions = [partial_vai, partial_fibrosis, partial_interaction]
colors = ['#3498db', '#2ecc71', '#9b59b6']

bars = ax1.bar(terms, contributions, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Partial R²', fontsize=11)
ax1.set_title('Crosswalk: Partial R² Contribution by Term', fontsize=12)
ax1.axhline(y=r2_full, color='red', linestyle='--', linewidth=2, label=f'Total R² = {r2_full:.4f}')

# Add value labels
for bar, val in zip(bars, contributions):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}\n({val/r2_full*100:.1f}%)', ha='center', va='bottom', fontsize=9)

ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Sensitivity visualization
ax2 = axes[0, 1]
vai_range = np.linspace(0, 22, 100)
magnifi_fib0 = 1.031 * vai_range + 1.713  # Fibrosis = 0
magnifi_fib6 = np.where(vai_range <= 2,
                         1.031 * vai_range + 0.264 * 6 + 1.713,
                         1.031 * vai_range + 1.713)  # Fibrosis = 6

ax2.plot(vai_range, magnifi_fib0, 'b-', linewidth=2, label='Fibrosis = 0')
ax2.plot(vai_range, magnifi_fib6, 'r-', linewidth=2, label='Fibrosis = 6')
ax2.fill_between(vai_range, magnifi_fib0, magnifi_fib6, alpha=0.2, color='purple')
ax2.axvline(x=2, color='gray', linestyle='--', linewidth=1, label='Threshold (VAI=2)')

ax2.set_xlabel('VAI Score', fontsize=11)
ax2.set_ylabel('MAGNIFI-CD Score', fontsize=11)
ax2.set_title('Formula Sensitivity: Effect of Fibrosis', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Annotate the fibrosis effect zone
ax2.annotate('Fibrosis effect\nzone', xy=(1, 3.5), fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Plot 3: Parser feature accuracy
ax3 = axes[1, 0]
features_sorted = [f[0] for f in ranked_features]
accuracies = [f[1]['accuracy'] for f in ranked_features]
colors_acc = ['#2ecc71' if a >= 0.9 else '#f39c12' if a >= 0.8 else '#e74c3c' for a in accuracies]

ax3.barh(range(len(features_sorted)), accuracies, color=colors_acc, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(features_sorted)))
ax3.set_yticklabels([f.replace('_', '\n') for f in features_sorted], fontsize=8)
ax3.set_xlabel('Extraction Accuracy', fontsize=11)
ax3.set_title('Parser: Feature Extraction Accuracy', fontsize=12)
ax3.axvline(x=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='90% threshold')
ax3.axvline(x=0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
ax3.set_xlim(0.5, 1.05)
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Feature importance scores
ax4 = axes[1, 1]
importance_scores = [f[1]['importance_score'] for f in ranked_features]

ax4.barh(range(len(features_sorted)), importance_scores, color='#3498db', edgecolor='black', linewidth=0.5)
ax4.set_yticks(range(len(features_sorted)))
ax4.set_yticklabels([f.replace('_', '\n') for f in features_sorted], fontsize=8)
ax4.set_xlabel('Importance Score', fontsize=11)
ax4.set_title('Parser: Feature Importance Ranking', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

# Add ranking numbers
for i, (feat, info) in enumerate(ranked_features):
    ax4.text(0.02, i, f'#{i+1}', ha='left', va='center', fontsize=8,
             fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_ranking.png", dpi=150, bbox_inches='tight')
print(f"Figure saved to: {OUTPUT_DIR / 'feature_ranking.png'}")
plt.close()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nCrosswalk Formula:")
print(f"  - VAI contributes {partial_vai/r2_full*100:.1f}% of explained variance")
print(f"  - Fibrosis term contributes only {(partial_fibrosis+partial_interaction)/r2_full*100:.1f}%")
print(f"  - The formula is almost entirely driven by VAI")

print("\nParser Features:")
print(f"  - Most reliable: {', '.join(essential_features[:3])}")
print(f"  - Least reliable: {', '.join([f[0] for f in ranked_features if f[1]['accuracy'] < 0.9][-2:])}")

print("\nClinical Implication:")
print("  VAI is the dominant predictor - prioritize accurate VAI measurement")
print("  Fibrosis only matters for healed/remission cases (VAI≤2)")
