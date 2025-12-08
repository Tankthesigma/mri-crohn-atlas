#!/usr/bin/env python3
"""
Calculate comprehensive metrics from real API validation results.
Includes ICC, Bland-Altman, Kappa, and subgroup analysis.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_FILE = Path(__file__).parent / "real_validation_results.json"
METRICS_FILE = Path(__file__).parent / "real_validation_metrics.json"


def calculate_icc(expected, predicted):
    """Calculate Intraclass Correlation Coefficient ICC(2,1)"""
    n = len(expected)
    if n < 2:
        return None, None, None

    expected = np.array(expected)
    predicted = np.array(predicted)

    # Combined data
    grand_mean = (np.mean(expected) + np.mean(predicted)) / 2

    # Between-subjects variance
    subject_means = (expected + predicted) / 2
    ss_between = 2 * np.sum((subject_means - grand_mean) ** 2)
    df_between = n - 1

    # Within-subjects variance
    ss_within = np.sum((expected - subject_means) ** 2) + np.sum((predicted - subject_means) ** 2)
    df_within = n

    # Error variance (rater x subject interaction)
    ss_error = ss_within
    df_error = df_within

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    # ICC(2,1)
    icc = (ms_between - ms_error) / (ms_between + ms_error) if (ms_between + ms_error) > 0 else 0

    # Confidence interval using F-distribution
    if ms_error > 0 and ms_between > 0:
        F = ms_between / ms_error
        df1, df2 = df_between, df_error

        # Lower bound
        F_L = F / stats.f.ppf(0.975, df1, df2)
        icc_lower = (F_L - 1) / (F_L + 1)

        # Upper bound
        F_U = F * stats.f.ppf(0.975, df2, df1)
        icc_upper = (F_U - 1) / (F_U + 1)
    else:
        icc_lower, icc_upper = 0, 0

    return max(0, min(1, icc)), max(0, icc_lower), min(1, icc_upper)


def calculate_kappa(expected, predicted, max_score):
    """Calculate Cohen's Kappa with severity binning"""
    def to_bin(score):
        if max_score == 22:  # VAI
            if score <= 2: return 0  # Remission
            if score <= 6: return 1  # Mild
            if score <= 12: return 2  # Moderate
            return 3  # Severe
        else:  # MAGNIFI
            if score <= 4: return 0  # Remission
            if score <= 10: return 1  # Mild
            if score <= 17: return 2  # Moderate
            return 3  # Severe

    exp_bins = [to_bin(e) for e in expected]
    pred_bins = [to_bin(p) for p in predicted]

    # Confusion matrix
    n_classes = 4
    matrix = np.zeros((n_classes, n_classes))
    for e, p in zip(exp_bins, pred_bins):
        matrix[e, p] += 1

    n = len(expected)
    p_o = np.trace(matrix) / n  # Observed agreement

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1

    # Weighted kappa (linear weights)
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = 1 - abs(i - j) / (n_classes - 1)

    w_o = np.sum(weights * matrix) / n
    w_e = np.sum(weights * np.outer(row_sums, col_sums)) / (n * n)
    weighted_kappa = (w_o - w_e) / (1 - w_e) if w_e < 1 else 1

    return kappa, weighted_kappa


def main():
    print("=" * 70)
    print("MRI-Crohn Atlas - REAL Validation Metrics")
    print("=" * 70)

    # Load results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    print(f"\nTotal cases: {len(results)}")

    # Filter valid results
    valid = [r for r in results if r.get("predicted_vai") is not None and r.get("expected_vai") is not None]
    print(f"Valid predictions: {len(valid)}")

    # Extract scores
    expected_vai = [r["expected_vai"] for r in valid]
    predicted_vai = [r["predicted_vai"] for r in valid]
    expected_magnifi = [r["expected_magnifi"] for r in valid]
    predicted_magnifi = [r["predicted_magnifi"] for r in valid]

    vai_errors = [p - e for p, e in zip(predicted_vai, expected_vai)]
    magnifi_errors = [p - e for p, e in zip(predicted_magnifi, expected_magnifi)]

    # Basic metrics
    n = len(valid)

    # VAI Metrics
    vai_mae = sum(abs(e) for e in vai_errors) / n
    vai_rmse = np.sqrt(sum(e**2 for e in vai_errors) / n)
    vai_bias = sum(vai_errors) / n
    vai_std = np.std(vai_errors)

    vai_acc_exact = sum(1 for e in vai_errors if e == 0) / n
    vai_acc_1 = sum(1 for e in vai_errors if abs(e) <= 1) / n
    vai_acc_2 = sum(1 for e in vai_errors if abs(e) <= 2) / n
    vai_acc_3 = sum(1 for e in vai_errors if abs(e) <= 3) / n

    # MAGNIFI Metrics
    mag_mae = sum(abs(e) for e in magnifi_errors) / n
    mag_rmse = np.sqrt(sum(e**2 for e in magnifi_errors) / n)
    mag_bias = sum(magnifi_errors) / n
    mag_std = np.std(magnifi_errors)

    mag_acc_exact = sum(1 for e in magnifi_errors if e == 0) / n
    mag_acc_2 = sum(1 for e in magnifi_errors if abs(e) <= 2) / n
    mag_acc_3 = sum(1 for e in magnifi_errors if abs(e) <= 3) / n
    mag_acc_5 = sum(1 for e in magnifi_errors if abs(e) <= 5) / n

    # Correlation
    vai_pearson, _ = stats.pearsonr(expected_vai, predicted_vai)
    vai_spearman, _ = stats.spearmanr(expected_vai, predicted_vai)
    mag_pearson, _ = stats.pearsonr(expected_magnifi, predicted_magnifi)
    mag_spearman, _ = stats.spearmanr(expected_magnifi, predicted_magnifi)

    # R-squared
    vai_ss_res = sum((p - e)**2 for p, e in zip(predicted_vai, expected_vai))
    vai_ss_tot = sum((e - np.mean(expected_vai))**2 for e in expected_vai)
    vai_r2 = 1 - vai_ss_res / vai_ss_tot if vai_ss_tot > 0 else 0

    mag_ss_res = sum((p - e)**2 for p, e in zip(predicted_magnifi, expected_magnifi))
    mag_ss_tot = sum((e - np.mean(expected_magnifi))**2 for e in expected_magnifi)
    mag_r2 = 1 - mag_ss_res / mag_ss_tot if mag_ss_tot > 0 else 0

    # ICC
    vai_icc, vai_icc_lower, vai_icc_upper = calculate_icc(expected_vai, predicted_vai)
    mag_icc, mag_icc_lower, mag_icc_upper = calculate_icc(expected_magnifi, predicted_magnifi)

    # Kappa
    vai_kappa, vai_weighted_kappa = calculate_kappa(expected_vai, predicted_vai, 22)
    mag_kappa, mag_weighted_kappa = calculate_kappa(expected_magnifi, predicted_magnifi, 25)

    # Bland-Altman
    vai_ba_loa_upper = vai_bias + 1.96 * vai_std
    vai_ba_loa_lower = vai_bias - 1.96 * vai_std

    mag_ba_loa_upper = mag_bias + 1.96 * mag_std
    mag_ba_loa_lower = mag_bias - 1.96 * mag_std

    # Confidence
    confidences = [r.get("confidence", 70) for r in valid]
    mean_confidence = sum(confidences) / len(confidences)

    # Print results
    print("\n" + "=" * 70)
    print("VAI RESULTS")
    print("=" * 70)
    print(f"  ICC:               {vai_icc:.3f} [{vai_icc_lower:.2f} - {vai_icc_upper:.2f}]")
    print(f"  MAE:               {vai_mae:.2f}")
    print(f"  RMSE:              {vai_rmse:.2f}")
    print(f"  Bias:              {vai_bias:+.2f}")
    print(f"  Accuracy (exact):  {vai_acc_exact*100:.1f}%")
    print(f"  Accuracy (±1):     {vai_acc_1*100:.1f}%")
    print(f"  Accuracy (±2):     {vai_acc_2*100:.1f}%")
    print(f"  Accuracy (±3):     {vai_acc_3*100:.1f}%")
    print(f"  Pearson r:         {vai_pearson:.3f}")
    print(f"  Spearman rho:      {vai_spearman:.3f}")
    print(f"  R²:                {vai_r2:.3f}")
    print(f"  Kappa:             {vai_kappa:.3f}")
    print(f"  Weighted Kappa:    {vai_weighted_kappa:.3f}")
    print(f"  Bland-Altman LoA:  [{vai_ba_loa_lower:.2f}, {vai_ba_loa_upper:.2f}]")

    print("\n" + "=" * 70)
    print("MAGNIFI-CD RESULTS")
    print("=" * 70)
    print(f"  ICC:               {mag_icc:.3f} [{mag_icc_lower:.2f} - {mag_icc_upper:.2f}]")
    print(f"  MAE:               {mag_mae:.2f}")
    print(f"  RMSE:              {mag_rmse:.2f}")
    print(f"  Bias:              {mag_bias:+.2f}")
    print(f"  Accuracy (exact):  {mag_acc_exact*100:.1f}%")
    print(f"  Accuracy (±2):     {mag_acc_2*100:.1f}%")
    print(f"  Accuracy (±3):     {mag_acc_3*100:.1f}%")
    print(f"  Accuracy (±5):     {mag_acc_5*100:.1f}%")
    print(f"  Pearson r:         {mag_pearson:.3f}")
    print(f"  Spearman rho:      {mag_spearman:.3f}")
    print(f"  R²:                {mag_r2:.3f}")
    print(f"  Kappa:             {mag_kappa:.3f}")
    print(f"  Weighted Kappa:    {mag_weighted_kappa:.3f}")
    print(f"  Bland-Altman LoA:  [{mag_ba_loa_lower:.2f}, {mag_ba_loa_upper:.2f}]")

    # Subgroup analysis
    print("\n" + "=" * 70)
    print("SUBGROUP ANALYSIS BY SEVERITY")
    print("=" * 70)

    for severity in ['remission', 'mild', 'moderate', 'severe']:
        sev_results = [r for r in valid if r.get("severity", "").lower() == severity]
        if len(sev_results) >= 2:
            sev_vai_err = [abs(r["predicted_vai"] - r["expected_vai"]) for r in sev_results]
            sev_acc_2 = sum(1 for e in sev_vai_err if e <= 2) / len(sev_vai_err)
            print(f"  {severity.capitalize():12} (n={len(sev_results):2}): VAI Acc(±2) = {sev_acc_2*100:.1f}%")

    # Subgroup by source
    print("\n" + "=" * 70)
    print("SUBGROUP ANALYSIS BY SOURCE")
    print("=" * 70)

    for source in ['radiopaedia', 'pubmed_central', 'edge_cases', 'synthetic_literature']:
        src_results = [r for r in valid if source in r.get("source", "").lower()]
        if len(src_results) >= 2:
            src_vai_err = [abs(r["predicted_vai"] - r["expected_vai"]) for r in src_results]
            src_acc_2 = sum(1 for e in src_vai_err if e <= 2) / len(src_vai_err)
            print(f"  {source[:15]:15} (n={len(src_results):2}): VAI Acc(±2) = {src_acc_2*100:.1f}%")

    # Compare to radiologists
    print("\n" + "=" * 70)
    print("COMPARISON TO RADIOLOGIST BENCHMARKS")
    print("=" * 70)
    radiologist_vai_icc = 0.68
    radiologist_magnifi_icc = 0.87
    vai_improvement = (vai_icc - radiologist_vai_icc) / radiologist_vai_icc * 100
    mag_improvement = (mag_icc - radiologist_magnifi_icc) / radiologist_magnifi_icc * 100
    print(f"  VAI ICC:     Parser {vai_icc:.3f} vs Radiologists {radiologist_vai_icc:.2f} = {vai_improvement:+.1f}%")
    print(f"  MAGNIFI ICC: Parser {mag_icc:.3f} vs Radiologists {radiologist_magnifi_icc:.2f} = {mag_improvement:+.1f}%")

    # Save metrics
    metrics = {
        "n_cases": n,
        "vai_icc": vai_icc,
        "vai_icc_ci_lower": vai_icc_lower,
        "vai_icc_ci_upper": vai_icc_upper,
        "vai_mae": vai_mae,
        "vai_rmse": vai_rmse,
        "vai_bias": vai_bias,
        "vai_accuracy_exact": vai_acc_exact,
        "vai_accuracy_within_1": vai_acc_1,
        "vai_accuracy_within_2": vai_acc_2,
        "vai_accuracy_within_3": vai_acc_3,
        "vai_pearson_r": vai_pearson,
        "vai_spearman_rho": vai_spearman,
        "vai_r2": vai_r2,
        "vai_kappa": vai_kappa,
        "vai_weighted_kappa": vai_weighted_kappa,
        "vai_bland_altman": {
            "mean_diff": vai_bias,
            "diff_std": vai_std,
            "loa_upper": vai_ba_loa_upper,
            "loa_lower": vai_ba_loa_lower
        },
        "magnifi_icc": mag_icc,
        "magnifi_icc_ci_lower": mag_icc_lower,
        "magnifi_icc_ci_upper": mag_icc_upper,
        "magnifi_mae": mag_mae,
        "magnifi_rmse": mag_rmse,
        "magnifi_bias": mag_bias,
        "magnifi_accuracy_exact": mag_acc_exact,
        "magnifi_accuracy_within_2": mag_acc_2,
        "magnifi_accuracy_within_3": mag_acc_3,
        "magnifi_accuracy_within_5": mag_acc_5,
        "magnifi_pearson_r": mag_pearson,
        "magnifi_spearman_rho": mag_spearman,
        "magnifi_r2": mag_r2,
        "magnifi_kappa": mag_kappa,
        "magnifi_weighted_kappa": mag_weighted_kappa,
        "magnifi_bland_altman": {
            "mean_diff": mag_bias,
            "diff_std": mag_std,
            "loa_upper": mag_ba_loa_upper,
            "loa_lower": mag_ba_loa_lower
        },
        "mean_confidence": mean_confidence,
        "comparison_to_radiologists": {
            "vai_parser_icc": vai_icc,
            "vai_radiologist_icc": radiologist_vai_icc,
            "vai_improvement_pct": vai_improvement,
            "magnifi_parser_icc": mag_icc,
            "magnifi_radiologist_icc": radiologist_magnifi_icc,
            "magnifi_improvement_pct": mag_improvement
        }
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
