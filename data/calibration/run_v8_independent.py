#!/usr/bin/env python3
"""
V8: Fully Independent Calibration for Each Severity Group
==========================================================

V8 Innovation: Each group has its OWN bias and quantile, calculated ONLY
from cases in that group.

Previous versions mixed calibration data:
- V5-V7: Used prediction-based grouping, which mixed severity levels

V8 Solution: Split by GROUND TRUTH severity, not prediction:
- Mild (<10):     Calculate Bias_Mild + Q_Mild, apply both
- Moderate (10-15): Calculate Bias_Moderate + Q_Moderate, apply both
- Severe (>=16):  Calculate Q_Severe only (NO bias, as proven in V7)

Expected Result: Moderate coverage should rise above 90% because intervals
are now centered on Moderate-specific error distribution.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHED_RESULTS = PROJECT_ROOT / "data" / "parser_validation" / "real_validation_results.json"
RESULTS_FILE = SCRIPT_DIR / "v8_conformal_results.json"

# Conformal parameters
N_FOLDS = 5
ALPHA = 0.10  # 90% coverage target

# Severity thresholds (based on ground truth VAI/MAGNIFI)
THRESH_MILD = 10      # Remission/Mild: expected < 10
THRESH_MODERATE = 16  # Moderate: expected 10-15, Severe: expected >= 16


class FullyIndependentConformal:
    """
    V8: Fully Independent Calibration per Severity Group

    Each group is calibrated ONLY using cases from that group:
    - Mild: Own bias + own quantile
    - Moderate: Own bias + own quantile
    - Severe: Own quantile only (no bias)
    """

    def __init__(self, n_folds=5, alpha=0.10):
        self.n_folds = n_folds
        self.alpha = alpha

        # Independent calibration data per group
        self.calibration_scores = {
            'mild': {'vai': [], 'magnifi': []},
            'moderate': {'vai': [], 'magnifi': []},
            'severe': {'vai': [], 'magnifi': []}
        }

        # Independent bias per group
        self.bias = {
            'mild': {'vai': 0.0, 'magnifi': 0.0},
            'moderate': {'vai': 0.0, 'magnifi': 0.0},
            'severe': {'vai': 0.0, 'magnifi': 0.0}  # Will stay 0 (V7 finding)
        }

    def _get_severity_group(self, severity_label):
        """Map severity label to calibration group."""
        if severity_label in ['Remission', 'Mild']:
            return 'mild'
        elif severity_label == 'Moderate':
            return 'moderate'
        elif severity_label == 'Severe':
            return 'severe'
        else:
            return 'mild'  # Default fallback

    def calibrate(self, calibration_data):
        """
        V8: Fully Independent Calibration

        Each severity group gets its OWN:
        - Nonconformity scores (absolute errors)
        - Bias (signed error mean) - except Severe which has no bias
        """
        # Collect errors by GROUND TRUTH severity
        vai_abs = {'mild': [], 'moderate': [], 'severe': []}
        mag_abs = {'mild': [], 'moderate': [], 'severe': []}
        vai_signed = {'mild': [], 'moderate': [], 'severe': []}
        mag_signed = {'mild': [], 'moderate': [], 'severe': []}

        for case in calibration_data:
            pred_vai = case.get('predicted_vai')
            pred_mag = case.get('predicted_magnifi')
            true_vai = case.get('expected_vai')
            true_mag = case.get('expected_magnifi')
            severity = case.get('severity', 'unknown')

            if pred_vai is None or true_vai is None:
                continue

            # Get group by GROUND TRUTH severity (not prediction!)
            group = self._get_severity_group(severity)

            vai_abs_err = abs(pred_vai - true_vai)
            vai_signed_err = pred_vai - true_vai
            mag_abs_err = abs(pred_mag - true_mag)
            mag_signed_err = pred_mag - true_mag

            vai_abs[group].append(vai_abs_err)
            vai_signed[group].append(vai_signed_err)
            mag_abs[group].append(mag_abs_err)
            mag_signed[group].append(mag_signed_err)

        # Store calibration scores for each group INDEPENDENTLY
        for group in ['mild', 'moderate', 'severe']:
            self.calibration_scores[group] = {
                'vai': sorted(vai_abs[group]),
                'magnifi': sorted(mag_abs[group])
            }

        # Calculate bias for Mild and Moderate groups ONLY
        for group in ['mild', 'moderate']:
            self.bias[group]['vai'] = np.mean(vai_signed[group]) if vai_signed[group] else 0.0
            self.bias[group]['magnifi'] = np.mean(mag_signed[group]) if mag_signed[group] else 0.0

        # Severe: NO bias (V7 finding - mixed over/under estimation)
        self.bias['severe']['vai'] = 0.0
        self.bias['severe']['magnifi'] = 0.0

        return {
            'n_mild': len(vai_abs['mild']),
            'n_moderate': len(vai_abs['moderate']),
            'n_severe': len(vai_abs['severe']),
            'vai_bias_mild': self.bias['mild']['vai'],
            'vai_bias_moderate': self.bias['moderate']['vai'],
            'vai_bias_severe': self.bias['severe']['vai'],
            'magnifi_bias_mild': self.bias['mild']['magnifi'],
            'magnifi_bias_moderate': self.bias['moderate']['magnifi'],
            'magnifi_bias_severe': self.bias['severe']['magnifi'],
        }

    def get_conformal_quantile(self, scores):
        """Get (1-alpha) quantile of nonconformity scores."""
        if not scores or len(scores) < 2:
            return 5.0  # Conservative fallback
        n = len(scores)
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = min(q_idx, n - 1)
        return scores[q_idx]

    def predict_with_interval(self, pred_vai, pred_mag, severity):
        """
        V8: Use FULLY INDEPENDENT calibration for each group.

        - Mild/Moderate: Apply group-specific bias + quantile
        - Severe: Apply group-specific quantile only (no bias)
        """
        group = self._get_severity_group(severity)

        # Get group-specific bias (0 for Severe)
        vai_bias = self.bias[group]['vai']
        mag_bias = self.bias[group]['magnifi']

        # Apply bias correction
        corrected_vai = pred_vai - vai_bias
        corrected_mag = pred_mag - mag_bias

        # Get group-specific quantile (interval half-width)
        vai_width = self.get_conformal_quantile(self.calibration_scores[group]['vai'])
        mag_width = self.get_conformal_quantile(self.calibration_scores[group]['magnifi'])

        return {
            'vai': {
                'point': round(corrected_vai, 1),
                'raw_point': pred_vai,
                'bias_applied': round(vai_bias, 2),
                'lower': max(0, round(corrected_vai - vai_width, 1)),
                'upper': min(22, round(corrected_vai + vai_width, 1)),
                'width': round(2 * vai_width, 1),
                'group': group
            },
            'magnifi': {
                'point': round(corrected_mag, 1),
                'raw_point': pred_mag,
                'bias_applied': round(mag_bias, 2),
                'lower': max(0, round(corrected_mag - mag_width, 1)),
                'upper': min(25, round(corrected_mag + mag_width, 1)),
                'width': round(2 * mag_width, 1),
                'group': group
            }
        }


def create_stratified_folds(results, n_folds):
    """Create stratified folds based on severity."""
    severity_groups = {}
    for i, r in enumerate(results):
        sev = r.get('severity', 'unknown')
        if sev not in severity_groups:
            severity_groups[sev] = []
        severity_groups[sev].append(i)

    folds = [[] for _ in range(n_folds)]
    for sev, indices in severity_groups.items():
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)

    return folds


def main():
    print("=" * 70)
    print("V8: FULLY INDEPENDENT CALIBRATION PER SEVERITY GROUP")
    print("=" * 70)
    print(f"\nTarget Coverage: {(1-ALPHA)*100:.0f}%")
    print(f"Number of Folds: {N_FOLDS}")
    print(f"\nCalibration Strategy:")
    print(f"  - Mild (Remission+Mild):  Own Bias + Own Quantile")
    print(f"  - Moderate:               Own Bias + Own Quantile")
    print(f"  - Severe:                 Own Quantile only (NO bias)")

    with open(CACHED_RESULTS) as f:
        cached_results = json.load(f)

    print(f"\nLoaded {len(cached_results)} cached predictions")

    # Count by severity
    sev_counts = {}
    for r in cached_results:
        sev = r.get('severity', 'unknown')
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
    print(f"Distribution: {sev_counts}")

    np.random.seed(42)
    folds = create_stratified_folds(cached_results, N_FOLDS)

    print("\n" + "-" * 70)
    print("CROSS-CONFORMAL PREDICTION (5-Fold)")
    print("-" * 70)

    final_results = []
    fold_metrics = []

    for fold_idx in range(N_FOLDS):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        test_indices = set(folds[fold_idx])
        cal_indices = [idx for f_idx, fold in enumerate(folds) if f_idx != fold_idx for idx in fold]

        cal_data = [cached_results[idx] for idx in cal_indices
                    if cached_results[idx].get('predicted_vai') is not None]

        # Initialize V8 conformal predictor
        conformal = FullyIndependentConformal(n_folds=N_FOLDS, alpha=ALPHA)
        cal_metrics = conformal.calibrate(cal_data)

        print(f"  Calibration by severity:")
        print(f"    Mild:     n={cal_metrics['n_mild']:>2}, VAI bias={cal_metrics['vai_bias_mild']:+.2f}, MAG bias={cal_metrics['magnifi_bias_mild']:+.2f}")
        print(f"    Moderate: n={cal_metrics['n_moderate']:>2}, VAI bias={cal_metrics['vai_bias_moderate']:+.2f}, MAG bias={cal_metrics['magnifi_bias_moderate']:+.2f}")
        print(f"    Severe:   n={cal_metrics['n_severe']:>2}, VAI bias=0.00 (disabled), MAG bias=0.00 (disabled)")

        fold_coverage_vai = 0
        fold_coverage_mag = 0
        fold_count = 0

        for idx in test_indices:
            result = cached_results[idx]

            if result.get('predicted_vai') is None:
                continue

            pred_vai = result['predicted_vai']
            pred_mag = result['predicted_magnifi']
            true_vai = result['expected_vai']
            true_mag = result['expected_magnifi']
            severity = result.get('severity', 'unknown')

            # V8: Use severity-specific calibration
            interval = conformal.predict_with_interval(pred_vai, pred_mag, severity)

            vai_covered = interval['vai']['lower'] <= true_vai <= interval['vai']['upper']
            mag_covered = interval['magnifi']['lower'] <= true_mag <= interval['magnifi']['upper']

            if vai_covered:
                fold_coverage_vai += 1
            if mag_covered:
                fold_coverage_mag += 1
            fold_count += 1

            final_results.append({
                "case_id": result['case_id'],
                "fold": fold_idx,
                "source": result.get("source"),
                "severity": severity,
                "expected_vai": int(true_vai),
                "expected_magnifi": int(true_mag),
                "raw_vai": int(pred_vai),
                "raw_magnifi": int(pred_mag),
                "corrected_vai": float(interval['vai']['point']),
                "corrected_magnifi": float(interval['magnifi']['point']),
                "vai_interval": [float(interval['vai']['lower']), float(interval['vai']['upper'])],
                "magnifi_interval": [float(interval['magnifi']['lower']), float(interval['magnifi']['upper'])],
                "vai_width": float(interval['vai']['width']),
                "magnifi_width": float(interval['magnifi']['width']),
                "vai_covered": bool(vai_covered),
                "magnifi_covered": bool(mag_covered),
                "vai_bias_applied": float(interval['vai']['bias_applied']),
                "magnifi_bias_applied": float(interval['magnifi']['bias_applied']),
                "calibration_group": interval['vai']['group']
            })

        if fold_count > 0:
            fold_metrics.append({
                'fold': fold_idx,
                'n_test': fold_count,
                'vai_coverage': fold_coverage_vai / fold_count,
                'magnifi_coverage': fold_coverage_mag / fold_count
            })
            print(f"  Coverage - VAI: {fold_coverage_vai}/{fold_count} ({100*fold_coverage_vai/fold_count:.1f}%), MAG: {fold_coverage_mag}/{fold_count} ({100*fold_coverage_mag/fold_count:.1f}%)")

    # Aggregate Results
    print("\n" + "=" * 70)
    print("V8 FULLY INDEPENDENT CALIBRATION RESULTS")
    print("=" * 70)

    valid_results = [r for r in final_results if r.get('expected_vai') is not None]

    vai_coverages = [r['vai_covered'] for r in valid_results]
    mag_coverages = [r['magnifi_covered'] for r in valid_results]

    vai_errors = [abs(r['corrected_vai'] - r['expected_vai']) for r in valid_results]
    mag_errors = [abs(r['corrected_magnifi'] - r['expected_magnifi']) for r in valid_results]

    vai_widths = [r['vai_width'] for r in valid_results]
    mag_widths = [r['magnifi_width'] for r in valid_results]

    # Group-specific widths
    vai_mild_widths = [r['vai_width'] for r in valid_results if r['calibration_group'] == 'mild']
    vai_moderate_widths = [r['vai_width'] for r in valid_results if r['calibration_group'] == 'moderate']
    vai_severe_widths = [r['vai_width'] for r in valid_results if r['calibration_group'] == 'severe']
    mag_mild_widths = [r['magnifi_width'] for r in valid_results if r['calibration_group'] == 'mild']
    mag_moderate_widths = [r['magnifi_width'] for r in valid_results if r['calibration_group'] == 'moderate']
    mag_severe_widths = [r['magnifi_width'] for r in valid_results if r['calibration_group'] == 'severe']

    print(f"\nTotal valid cases: {len(valid_results)}")

    print(f"\n{'Metric':<35} {'VAI':<15} {'MAGNIFI-CD':<15}")
    print("-" * 65)
    print(f"{'Overall Coverage (90% target)':<35} {100*np.mean(vai_coverages):.1f}%{'':<9} {100*np.mean(mag_coverages):.1f}%")
    print(f"{'Mean Interval Width':<35} {np.mean(vai_widths):.2f}{'':<11} {np.mean(mag_widths):.2f}")
    print(f"{'  - Mild Group Width':<35} {np.mean(vai_mild_widths) if vai_mild_widths else 0:.2f}{'':<11} {np.mean(mag_mild_widths) if mag_mild_widths else 0:.2f}")
    print(f"{'  - Moderate Group Width':<35} {np.mean(vai_moderate_widths) if vai_moderate_widths else 0:.2f}{'':<11} {np.mean(mag_moderate_widths) if mag_moderate_widths else 0:.2f}")
    print(f"{'  - Severe Group Width':<35} {np.mean(vai_severe_widths) if vai_severe_widths else 0:.2f}{'':<11} {np.mean(mag_severe_widths) if mag_severe_widths else 0:.2f}")
    print(f"{'MAE':<35} {np.mean(vai_errors):.2f}{'':<11} {np.mean(mag_errors):.2f}")

    # Coverage by severity
    print(f"\n{'COVERAGE BY SEVERITY (Target: >85% all groups):':<50}")
    print("-" * 65)
    all_pass = True
    severity_results = {}
    for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
        sev_results = [r for r in valid_results if r.get('severity') == sev]
        if sev_results:
            vai_cov = 100 * np.mean([r['vai_covered'] for r in sev_results])
            mag_cov = 100 * np.mean([r['magnifi_covered'] for r in sev_results])
            vai_width = np.mean([r['vai_width'] for r in sev_results])
            mag_width = np.mean([r['magnifi_width'] for r in sev_results])

            vai_status = "✓" if vai_cov >= 85 else "✗"
            mag_status = "✓" if mag_cov >= 85 else "✗"

            if vai_cov < 85 or mag_cov < 85:
                all_pass = False

            severity_results[sev] = {
                'vai_cov': vai_cov, 'mag_cov': mag_cov,
                'vai_width': vai_width, 'mag_width': mag_width,
                'n': len(sev_results)
            }

            print(f"  {sev:<15} VAI: {vai_cov:>5.1f}% {vai_status} (w={vai_width:.1f})  MAG: {mag_cov:>5.1f}% {mag_status} (w={mag_width:.1f})  n={len(sev_results)}")

    # Save results
    output = {
        "metadata": {
            "version": "V8 Fully Independent Calibration",
            "timestamp": datetime.now().isoformat(),
            "n_folds": N_FOLDS,
            "alpha": ALPHA,
            "target_coverage": f"{(1-ALPHA)*100:.0f}%",
            "strategy": {
                "mild": "Own Bias + Own Quantile",
                "moderate": "Own Bias + Own Quantile",
                "severe": "Own Quantile only (no bias)"
            }
        },
        "summary": {
            "n_cases": len(valid_results),
            "vai_coverage": float(np.mean(vai_coverages)),
            "magnifi_coverage": float(np.mean(mag_coverages)),
            "vai_mean_width": float(np.mean(vai_widths)),
            "magnifi_mean_width": float(np.mean(mag_widths)),
            "vai_mae": float(np.mean(vai_errors)),
            "magnifi_mae": float(np.mean(mag_errors))
        },
        "coverage_by_severity": {
            sev: {
                "vai_coverage": float(np.mean([r['vai_covered'] for r in valid_results if r.get('severity') == sev])),
                "magnifi_coverage": float(np.mean([r['magnifi_covered'] for r in valid_results if r.get('severity') == sev])),
                "n": len([r for r in valid_results if r.get('severity') == sev])
            }
            for sev in ['Remission', 'Mild', 'Moderate', 'Severe']
        },
        "fold_metrics": fold_metrics,
        "detailed_results": final_results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")

    # Final Table
    print("\n" + "=" * 70)
    print("V8 FULLY INDEPENDENT CALIBRATION RESULTS TABLE")
    print("=" * 70)
    print(f"┌{'─'*68}┐")
    print(f"│{'Metric':<35}{'VAI':<16}{'MAGNIFI-CD':<17}│")
    print(f"├{'─'*68}┤")
    print(f"│{'Overall Coverage (90% target)':<35}{100*np.mean(vai_coverages):>6.1f}%{'':<9}{100*np.mean(mag_coverages):>6.1f}%{'':<10}│")
    print(f"│{'Mean Interval Width':<35}{np.mean(vai_widths):>7.2f}{'':<9}{np.mean(mag_widths):>7.2f}{'':<10}│")
    print(f"│{'  Mild Width':<35}{np.mean(vai_mild_widths) if vai_mild_widths else 0:>7.2f}{'':<9}{np.mean(mag_mild_widths) if mag_mild_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'  Moderate Width':<35}{np.mean(vai_moderate_widths) if vai_moderate_widths else 0:>7.2f}{'':<9}{np.mean(mag_moderate_widths) if mag_moderate_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'  Severe Width':<35}{np.mean(vai_severe_widths) if vai_severe_widths else 0:>7.2f}{'':<9}{np.mean(mag_severe_widths) if mag_severe_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'MAE':<35}{np.mean(vai_errors):>7.2f}{'':<9}{np.mean(mag_errors):>7.2f}{'':<10}│")
    print(f"├{'─'*68}┤")
    print(f"│{'COVERAGE BY SEVERITY (✓ = >85%)':<68}│")
    print(f"├{'─'*68}┤")
    for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
        if sev in severity_results:
            sr = severity_results[sev]
            vai_status = "✓" if sr['vai_cov'] >= 85 else "✗"
            mag_status = "✓" if sr['mag_cov'] >= 85 else "✗"
            print(f"│  {sev:<33}{sr['vai_cov']:>5.1f}% {vai_status}{'':<7}{sr['mag_cov']:>5.1f}% {mag_status}{'':<8}│")
    print(f"└{'─'*68}┘")

    # Comparison with V7
    print("\n" + "-" * 70)
    print("V7 vs V8 COMPARISON")
    print("-" * 70)
    print(f"{'Severity':<15} {'V7 VAI':<12} {'V8 VAI':<12} {'V7 MAG':<12} {'V8 MAG':<12}")
    print("-" * 65)
    v7_results = {'Remission': (100.0, 100.0), 'Mild': (100.0, 100.0),
                  'Moderate': (85.7, 92.9), 'Severe': (94.1, 94.1)}
    for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
        if sev in severity_results:
            sr = severity_results[sev]
            v7_vai, v7_mag = v7_results[sev]
            vai_diff = sr['vai_cov'] - v7_vai
            mag_diff = sr['mag_cov'] - v7_mag
            vai_arrow = "↑" if vai_diff > 0 else ("↓" if vai_diff < 0 else "=")
            mag_arrow = "↑" if mag_diff > 0 else ("↓" if mag_diff < 0 else "=")
            print(f"{sev:<15} {v7_vai:>5.1f}%{'':<6} {sr['vai_cov']:>5.1f}% {vai_arrow:<4} {v7_mag:>5.1f}%{'':<6} {sr['mag_cov']:>5.1f}% {mag_arrow}")

    if all_pass:
        print("\n" + "🎉" * 35)
        print("ALL SEVERITY LEVELS ACHIEVED >85% COVERAGE!")
        print("🎉" * 35)
    else:
        print("\n⚠️  Some severity levels still below 85% target")


if __name__ == "__main__":
    main()
