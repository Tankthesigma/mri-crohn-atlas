#!/usr/bin/env python3
"""
Cross-Conformal Prediction for MRI-Crohn Parser Calibration.

Uses MAPIE for conformal prediction to generate calibrated confidence intervals
for VAI and MAGNIFI score predictions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from mapie.regression import CrossConformalRegressor

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input paths
GOLD_CASES_PATH = PROJECT_ROOT / "data" / "calibration" / "gold_cases.json"
VALIDATION_RESULTS_PATH = PROJECT_ROOT / "data" / "calibration" / "parser_validation_results.json"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration"
RESULTS_PATH = OUTPUT_DIR / "conformal_results.json"
CALIBRATION_PLOT_PATH = OUTPUT_DIR / "calibration_curve.png"
REPORT_PATH = OUTPUT_DIR / "calibration_report.md"

# Conformal prediction parameters
CONFIDENCE_LEVEL = 0.90  # 90% confidence intervals
CV_FOLDS = 5


def load_validation_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Load validation results with predictions and ground truth.

    Returns:
        predicted_vai, true_vai, predicted_magnifi, true_magnifi, cases
    """
    with open(VALIDATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle wrapped format (new parser_validation_results.json)
    if isinstance(data, dict) and "cases" in data:
        results = data["cases"]
    else:
        results = data

    predicted_vai = []
    true_vai = []
    predicted_magnifi = []
    true_magnifi = []
    valid_cases = []

    for case in results:
        # Skip cases with errors or missing predictions
        if case.get("error") or case.get("predicted_vai") is None:
            continue

        pred_vai = case.get("predicted_vai")
        exp_vai = case.get("expected_vai")
        pred_magnifi = case.get("predicted_magnifi")
        exp_magnifi = case.get("expected_magnifi")

        if all(v is not None for v in [pred_vai, exp_vai, pred_magnifi, exp_magnifi]):
            predicted_vai.append(pred_vai)
            true_vai.append(exp_vai)
            predicted_magnifi.append(pred_magnifi)
            true_magnifi.append(exp_magnifi)
            valid_cases.append(case)

    return (
        np.array(predicted_vai),
        np.array(true_vai),
        np.array(predicted_magnifi),
        np.array(true_magnifi),
        valid_cases
    )


def run_conformal_prediction(
    predictions: np.ndarray,
    true_values: np.ndarray,
    score_name: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run cross-conformal prediction using MAPIE.

    Uses a simple identity model (predictions as features) with MAPIE's
    cross-conformal approach to generate calibrated intervals.

    Returns:
        lower_bounds, upper_bounds, metrics
    """
    # Reshape for sklearn
    X = predictions.reshape(-1, 1)
    y = true_values

    # Use Ridge regression as base model (identity-like with regularization)
    base_model = Ridge(alpha=0.1)

    # MAPIE with cross-conformal
    mapie = CrossConformalRegressor(
        estimator=base_model,
        cv=CV_FOLDS,
        confidence_level=CONFIDENCE_LEVEL,
        method="plus"
    )

    mapie.fit_conformalize(X, y)

    # Get predictions with confidence intervals
    y_pred, y_pis = mapie.predict_interval(X)

    lower_bounds = y_pis[:, 0]
    upper_bounds = y_pis[:, 1]

    # Calculate metrics
    interval_widths = upper_bounds - lower_bounds
    coverage = np.mean((true_values >= lower_bounds) & (true_values <= upper_bounds))

    metrics = {
        "score_name": score_name,
        "n_samples": len(predictions),
        "mean_interval_width": float(np.mean(interval_widths)),
        "median_interval_width": float(np.median(interval_widths)),
        "std_interval_width": float(np.std(interval_widths)),
        "min_interval_width": float(np.min(interval_widths)),
        "max_interval_width": float(np.max(interval_widths)),
        "coverage_rate": float(coverage),
        "target_coverage": CONFIDENCE_LEVEL,
        "n_flagged_for_review": int(np.sum(interval_widths > 2)),
        "pct_flagged_for_review": float(np.mean(interval_widths > 2) * 100),
    }

    return lower_bounds, upper_bounds, metrics


def create_calibration_plot(
    vai_predictions: np.ndarray,
    vai_true: np.ndarray,
    vai_lower: np.ndarray,
    vai_upper: np.ndarray,
    magnifi_predictions: np.ndarray,
    magnifi_true: np.ndarray,
    magnifi_lower: np.ndarray,
    magnifi_upper: np.ndarray,
    vai_metrics: Dict,
    magnifi_metrics: Dict
):
    """Create calibration visualization."""
    # Flatten arrays if needed
    vai_lower = vai_lower.flatten()
    vai_upper = vai_upper.flatten()
    magnifi_lower = magnifi_lower.flatten()
    magnifi_upper = magnifi_upper.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: VAI Prediction vs True with intervals
    ax1 = axes[0, 0]
    vai_covered = (vai_true >= vai_lower) & (vai_true <= vai_upper)

    # Plot points with intervals (show interval from lower to upper)
    for i in range(len(vai_predictions)):
        color = '#10B981' if vai_covered[i] else '#EF4444'
        ax1.plot([vai_predictions[i], vai_predictions[i]], [vai_lower[i], vai_upper[i]],
                 color=color, alpha=0.3, linewidth=2)
        ax1.scatter(vai_predictions[i], vai_true[i], color=color, alpha=0.7, s=40)

    # Create legend manually
    ax1.scatter([], [], color='#10B981', label=f'Covered ({np.sum(vai_covered)})')
    ax1.scatter([], [], color='#EF4444', label=f'Not Covered ({np.sum(~vai_covered)})')
    ax1.plot([0, 22], [0, 22], 'k--', alpha=0.5, label='Perfect')
    ax1.set_xlabel('Predicted VAI')
    ax1.set_ylabel('True VAI')
    ax1.set_title(f'VAI: Coverage={vai_metrics["coverage_rate"]:.1%}, Width={vai_metrics["mean_interval_width"]:.2f}')
    ax1.legend()
    ax1.set_xlim(-1, 23)
    ax1.set_ylim(-1, 23)
    ax1.grid(True, alpha=0.3)

    # Plot 2: MAGNIFI Prediction vs True with intervals
    ax2 = axes[0, 1]
    magnifi_covered = (magnifi_true >= magnifi_lower) & (magnifi_true <= magnifi_upper)

    # Plot points with intervals
    for i in range(len(magnifi_predictions)):
        color = '#10B981' if magnifi_covered[i] else '#EF4444'
        ax2.plot([magnifi_predictions[i], magnifi_predictions[i]], [magnifi_lower[i], magnifi_upper[i]],
                 color=color, alpha=0.3, linewidth=2)
        ax2.scatter(magnifi_predictions[i], magnifi_true[i], color=color, alpha=0.7, s=40)

    # Create legend manually
    ax2.scatter([], [], color='#10B981', label=f'Covered ({np.sum(magnifi_covered)})')
    ax2.scatter([], [], color='#EF4444', label=f'Not Covered ({np.sum(~magnifi_covered)})')
    ax2.plot([0, 25], [0, 25], 'k--', alpha=0.5, label='Perfect')
    ax2.set_xlabel('Predicted MAGNIFI-CD')
    ax2.set_ylabel('True MAGNIFI-CD')
    ax2.set_title(f'MAGNIFI: Coverage={magnifi_metrics["coverage_rate"]:.1%}, Width={magnifi_metrics["mean_interval_width"]:.2f}')
    ax2.legend()
    ax2.set_xlim(-1, 26)
    ax2.set_ylim(-1, 26)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Interval width distribution
    ax3 = axes[1, 0]
    vai_widths = vai_upper - vai_lower
    magnifi_widths = magnifi_upper - magnifi_lower
    ax3.hist(vai_widths, bins=15, alpha=0.6, label=f'VAI (μ={np.mean(vai_widths):.2f})', color='#0066CC')
    ax3.hist(magnifi_widths, bins=15, alpha=0.6, label=f'MAGNIFI (μ={np.mean(magnifi_widths):.2f})', color='#F59E0B')
    ax3.axvline(x=2, color='#EF4444', linestyle='--', label='Review threshold (>2)')
    ax3.set_xlabel('Interval Width (points)')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Confidence Interval Widths')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Calibration curve (binned)
    ax4 = axes[1, 1]

    # Calculate empirical coverage at different confidence levels
    confidence_levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # For VAI
    X_vai = vai_predictions.reshape(-1, 1)
    vai_empirical = []
    for conf in confidence_levels:
        base_model_vai = Ridge(alpha=0.1)
        mapie_vai = CrossConformalRegressor(
            estimator=base_model_vai, cv=CV_FOLDS, confidence_level=conf, method="plus"
        )
        mapie_vai.fit_conformalize(X_vai, vai_true)
        _, y_pis = mapie_vai.predict_interval(X_vai)
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
        coverage = np.mean((vai_true >= lower) & (vai_true <= upper))
        vai_empirical.append(coverage)

    # For MAGNIFI
    X_magnifi = magnifi_predictions.reshape(-1, 1)
    magnifi_empirical = []
    for conf in confidence_levels:
        base_model_magnifi = Ridge(alpha=0.1)
        mapie_magnifi = CrossConformalRegressor(
            estimator=base_model_magnifi, cv=CV_FOLDS, confidence_level=conf, method="plus"
        )
        mapie_magnifi.fit_conformalize(X_magnifi, magnifi_true)
        _, y_pis = mapie_magnifi.predict_interval(X_magnifi)
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
        coverage = np.mean((magnifi_true >= lower) & (magnifi_true <= upper))
        magnifi_empirical.append(coverage)

    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax4.plot(confidence_levels, vai_empirical, 'o-', color='#0066CC', label='VAI', linewidth=2, markersize=8)
    ax4.plot(confidence_levels, magnifi_empirical, 's-', color='#F59E0B', label='MAGNIFI-CD', linewidth=2, markersize=8)
    ax4.set_xlabel('Target Confidence Level')
    ax4.set_ylabel('Empirical Coverage')
    ax4.set_title('Calibration Curve: Expected vs Actual Coverage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.45, 1.0)
    ax4.set_ylim(0.45, 1.0)

    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {CALIBRATION_PLOT_PATH}")


def generate_report(
    vai_metrics: Dict,
    magnifi_metrics: Dict,
    cases_with_intervals: List[Dict]
) -> str:
    """Generate markdown calibration report."""

    # Count cases flagged for review
    vai_flagged = [c for c in cases_with_intervals if c["vai_interval_width"] > 2]
    magnifi_flagged = [c for c in cases_with_intervals if c["magnifi_interval_width"] > 2]

    report = f"""# Cross-Conformal Prediction Calibration Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method:** MAPIE Jackknife+ with {CV_FOLDS}-fold cross-validation
**Confidence Level:** {CONFIDENCE_LEVEL:.0%}
**Total Cases:** {vai_metrics['n_samples']}

---

## Summary Metrics

| Metric | VAI | MAGNIFI-CD |
|--------|-----|------------|
| **Mean Interval Width** | {vai_metrics['mean_interval_width']:.2f} pts | {magnifi_metrics['mean_interval_width']:.2f} pts |
| **Median Interval Width** | {vai_metrics['median_interval_width']:.2f} pts | {magnifi_metrics['median_interval_width']:.2f} pts |
| **Coverage Rate** | {vai_metrics['coverage_rate']:.1%} | {magnifi_metrics['coverage_rate']:.1%} |
| **Target Coverage** | {vai_metrics['target_coverage']:.0%} | {magnifi_metrics['target_coverage']:.0%} |
| **Flagged for Review (width > 2)** | {vai_metrics['n_flagged_for_review']} ({vai_metrics['pct_flagged_for_review']:.1f}%) | {magnifi_metrics['n_flagged_for_review']} ({magnifi_metrics['pct_flagged_for_review']:.1f}%) |

---

## Interpretation

### Coverage Analysis
- **VAI Coverage:** {vai_metrics['coverage_rate']:.1%} vs target {vai_metrics['target_coverage']:.0%} → {"✅ Well-calibrated" if abs(vai_metrics['coverage_rate'] - vai_metrics['target_coverage']) < 0.05 else "⚠️ Needs adjustment"}
- **MAGNIFI Coverage:** {magnifi_metrics['coverage_rate']:.1%} vs target {magnifi_metrics['target_coverage']:.0%} → {"✅ Well-calibrated" if abs(magnifi_metrics['coverage_rate'] - magnifi_metrics['target_coverage']) < 0.05 else "⚠️ Needs adjustment"}

### Interval Width Analysis
- Average VAI interval spans **{vai_metrics['mean_interval_width']:.1f} points** (range: {vai_metrics['min_interval_width']:.1f} - {vai_metrics['max_interval_width']:.1f})
- Average MAGNIFI interval spans **{magnifi_metrics['mean_interval_width']:.1f} points** (range: {magnifi_metrics['min_interval_width']:.1f} - {magnifi_metrics['max_interval_width']:.1f})

### Clinical Decision Support
Cases with interval width > 2 points should be flagged for human radiologist review:
- **{vai_metrics['n_flagged_for_review']} cases** ({vai_metrics['pct_flagged_for_review']:.1f}%) flagged based on VAI uncertainty
- **{magnifi_metrics['n_flagged_for_review']} cases** ({magnifi_metrics['pct_flagged_for_review']:.1f}%) flagged based on MAGNIFI uncertainty

---

## Cases Flagged for Human Review

### High VAI Uncertainty (interval width > 2)
"""

    if vai_flagged:
        report += "| Case ID | Predicted | Interval | True | Width |\n"
        report += "|---------|-----------|----------|------|-------|\n"
        for case in sorted(vai_flagged, key=lambda x: -x["vai_interval_width"])[:10]:
            report += f"| {case['case_id']} | {case['predicted_vai']} | [{case['vai_lower']:.1f}, {case['vai_upper']:.1f}] | {case['expected_vai']} | {case['vai_interval_width']:.2f} |\n"
        if len(vai_flagged) > 10:
            report += f"\n*...and {len(vai_flagged) - 10} more cases*\n"
    else:
        report += "*No cases flagged*\n"

    report += """
### High MAGNIFI Uncertainty (interval width > 2)
"""

    if magnifi_flagged:
        report += "| Case ID | Predicted | Interval | True | Width |\n"
        report += "|---------|-----------|----------|------|-------|\n"
        for case in sorted(magnifi_flagged, key=lambda x: -x["magnifi_interval_width"])[:10]:
            report += f"| {case['case_id']} | {case['predicted_magnifi']} | [{case['magnifi_lower']:.1f}, {case['magnifi_upper']:.1f}] | {case['expected_magnifi']} | {case['magnifi_interval_width']:.2f} |\n"
        if len(magnifi_flagged) > 10:
            report += f"\n*...and {len(magnifi_flagged) - 10} more cases*\n"
    else:
        report += "*No cases flagged*\n"

    report += f"""
---

## Technical Details

### Conformal Prediction Method
- **Algorithm:** Jackknife+ (MAPIE implementation)
- **Base Model:** Ridge Regression (α=0.1)
- **Cross-Validation:** {CV_FOLDS}-fold
- **Coverage Guarantee:** Asymptotically valid at {CONFIDENCE_LEVEL:.0%} confidence

### Output Files
- `conformal_results.json` - All cases with predictions and intervals
- `calibration_curve.png` - Calibration visualization
- `calibration_report.md` - This report

---

*Generated by MRI-Crohn Atlas Conformal Prediction Pipeline*
"""

    return report


def main():
    print("=" * 60)
    print("CROSS-CONFORMAL PREDICTION FOR MRI-CROHN PARSER")
    print("=" * 60)
    print()

    # Load data
    print("Loading validation data...")
    pred_vai, true_vai, pred_magnifi, true_magnifi, cases = load_validation_data()
    print(f"  Loaded {len(cases)} cases with valid predictions")
    print()

    # Run conformal prediction for VAI
    print("Running conformal prediction for VAI...")
    vai_lower, vai_upper, vai_metrics = run_conformal_prediction(pred_vai, true_vai, "VAI")
    print(f"  Coverage: {vai_metrics['coverage_rate']:.1%}")
    print(f"  Mean interval width: {vai_metrics['mean_interval_width']:.2f} points")
    print()

    # Run conformal prediction for MAGNIFI
    print("Running conformal prediction for MAGNIFI-CD...")
    magnifi_lower, magnifi_upper, magnifi_metrics = run_conformal_prediction(pred_magnifi, true_magnifi, "MAGNIFI-CD")
    print(f"  Coverage: {magnifi_metrics['coverage_rate']:.1%}")
    print(f"  Mean interval width: {magnifi_metrics['mean_interval_width']:.2f} points")
    print()

    # Build results with intervals
    print("Building results...")
    # Flatten bounds arrays
    vai_lower_flat = vai_lower.flatten()
    vai_upper_flat = vai_upper.flatten()
    magnifi_lower_flat = magnifi_lower.flatten()
    magnifi_upper_flat = magnifi_upper.flatten()

    cases_with_intervals = []
    for i, case in enumerate(cases):
        vai_lo = float(vai_lower_flat[i])
        vai_hi = float(vai_upper_flat[i])
        mag_lo = float(magnifi_lower_flat[i])
        mag_hi = float(magnifi_upper_flat[i])

        case_result = {
            "case_id": case["case_id"],
            "source": case.get("source"),
            "case_type": case.get("case_type"),
            "severity": case.get("severity"),
            "predicted_vai": int(pred_vai[i]),
            "expected_vai": int(true_vai[i]),
            "vai_lower": vai_lo,
            "vai_upper": vai_hi,
            "vai_interval_width": vai_hi - vai_lo,
            "vai_covered": bool(true_vai[i] >= vai_lo and true_vai[i] <= vai_hi),
            "predicted_magnifi": int(pred_magnifi[i]),
            "expected_magnifi": int(true_magnifi[i]),
            "magnifi_lower": mag_lo,
            "magnifi_upper": mag_hi,
            "magnifi_interval_width": mag_hi - mag_lo,
            "magnifi_covered": bool(true_magnifi[i] >= mag_lo and true_magnifi[i] <= mag_hi),
            "flagged_for_review": bool(vai_hi - vai_lo > 2 or mag_hi - mag_lo > 2)
        }
        cases_with_intervals.append(case_result)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results_output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "method": "MAPIE Jackknife+",
            "cv_folds": CV_FOLDS,
            "confidence_level": CONFIDENCE_LEVEL,
            "n_cases": len(cases)
        },
        "vai_metrics": vai_metrics,
        "magnifi_metrics": magnifi_metrics,
        "cases": cases_with_intervals
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_output, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")

    # Create calibration plot
    print("Creating calibration plot...")
    create_calibration_plot(
        pred_vai, true_vai, vai_lower, vai_upper,
        pred_magnifi, true_magnifi, magnifi_lower, magnifi_upper,
        vai_metrics, magnifi_metrics
    )

    # Generate and save report
    print("Generating report...")
    report = generate_report(vai_metrics, magnifi_metrics, cases_with_intervals)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report to {REPORT_PATH}")

    # Print summary
    print()
    print("=" * 60)
    print("CONFORMAL PREDICTION RESULTS")
    print("=" * 60)
    print()
    print(f"Mean interval width VAI:      {vai_metrics['mean_interval_width']:.2f} points")
    print(f"Mean interval width MAGNIFI:  {magnifi_metrics['mean_interval_width']:.2f} points")
    print(f"Coverage rate VAI:            {vai_metrics['coverage_rate']:.1%} (target: {CONFIDENCE_LEVEL:.0%})")
    print(f"Coverage rate MAGNIFI:        {magnifi_metrics['coverage_rate']:.1%} (target: {CONFIDENCE_LEVEL:.0%})")
    print(f"Flagged for review (VAI):     {vai_metrics['n_flagged_for_review']} cases ({vai_metrics['pct_flagged_for_review']:.1f}%)")
    print(f"Flagged for review (MAGNIFI): {magnifi_metrics['n_flagged_for_review']} cases ({magnifi_metrics['pct_flagged_for_review']:.1f}%)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
