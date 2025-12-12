#!/usr/bin/env python3
"""
V5 Granular Group-Balanced Cross-Conformal Prediction (Using Cached Results)
=============================================================================

Uses cached API predictions from real_validation_results.json to run V5
3-tier conformal prediction without needing API calls.

V5: 3 Calibration Groups based on Predicted Score:
  - Group 1 (Low): pred < 10  → Remission/Mild
  - Group 2 (Mid): pred 10-15 → Moderate
  - Group 3 (High): pred >= 16 → Severe
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHED_RESULTS = PROJECT_ROOT / "data" / "parser_validation" / "real_validation_results.json"
CASES_FILE = PROJECT_ROOT / "data" / "parser_validation" / "mega_test_cases.json"
RESULTS_FILE = SCRIPT_DIR / "v5_conformal_results.json"

# Conformal parameters
N_FOLDS = 5
ALPHA = 0.10  # 90% coverage target

# V5: 3-tier thresholds
THRESH_LOW = 10   # Group 1: pred < 10 (Remission/Mild)
THRESH_HIGH = 16  # Group 3: pred >= 16 (Severe), Group 2: 10-15 (Moderate)


class GranularGroupBalancedConformal:
    """
    V5 Granular Group-Balanced Cross-Conformal Prediction with Bias Correction

    3 Calibration Groups:
      - Low: pred < 10 (Remission/Mild)
      - Mid: pred 10-15 (Moderate)
      - High: pred >= 16 (Severe)
    """

    def __init__(self, n_folds=5, alpha=0.10, thresh_low=10, thresh_high=16):
        self.n_folds = n_folds
        self.alpha = alpha
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

        # Calibration scores per group
        self.calibration_scores = {
            'low': {'vai': [], 'magnifi': []},
            'mid': {'vai': [], 'magnifi': []},
            'high': {'vai': [], 'magnifi': []}
        }

        # Bias per group
        self.bias = {
            'low': {'vai': 0.0, 'magnifi': 0.0},
            'mid': {'vai': 0.0, 'magnifi': 0.0},
            'high': {'vai': 0.0, 'magnifi': 0.0}
        }

    def _get_group(self, score):
        """Determine which of 3 groups a score belongs to."""
        if score < self.thresh_low:
            return 'low'
        elif score >= self.thresh_high:
            return 'high'
        else:
            return 'mid'

    def calibrate(self, predictions, ground_truths):
        """Calculate nonconformity scores and bias, split by 3 groups."""
        vai_scores = {'low': [], 'mid': [], 'high': []}
        mag_scores = {'low': [], 'mid': [], 'high': []}
        vai_errors = {'low': [], 'mid': [], 'high': []}
        mag_errors = {'low': [], 'mid': [], 'high': []}

        for pred, truth in zip(predictions, ground_truths):
            pred_vai = pred.get('predicted_vai', 0)
            pred_mag = pred.get('predicted_magnifi', 0)
            true_vai = truth.get('expected_vai', 0)
            true_mag = truth.get('expected_magnifi', 0)

            if pred_vai is None or true_vai is None:
                continue

            vai_abs_err = abs(pred_vai - true_vai)
            vai_signed_err = pred_vai - true_vai
            mag_abs_err = abs(pred_mag - true_mag)
            mag_signed_err = pred_mag - true_mag

            # Assign to group based on predicted score
            vai_group = self._get_group(pred_vai)
            mag_group = self._get_group(pred_mag)

            vai_scores[vai_group].append(vai_abs_err)
            vai_errors[vai_group].append(vai_signed_err)
            mag_scores[mag_group].append(mag_abs_err)
            mag_errors[mag_group].append(mag_signed_err)

        # Store calibration scores and bias per group
        for group in ['low', 'mid', 'high']:
            self.calibration_scores[group] = {
                'vai': sorted(vai_scores[group]),
                'magnifi': sorted(mag_scores[group])
            }
            self.bias[group]['vai'] = np.mean(vai_errors[group]) if vai_errors[group] else 0.0
            self.bias[group]['magnifi'] = np.mean(mag_errors[group]) if mag_errors[group] else 0.0

        return {
            'n_vai_low': len(vai_scores['low']),
            'n_vai_mid': len(vai_scores['mid']),
            'n_vai_high': len(vai_scores['high']),
            'n_magnifi_low': len(mag_scores['low']),
            'n_magnifi_mid': len(mag_scores['mid']),
            'n_magnifi_high': len(mag_scores['high']),
            'vai_bias_low': self.bias['low']['vai'],
            'vai_bias_mid': self.bias['mid']['vai'],
            'vai_bias_high': self.bias['high']['vai'],
            'magnifi_bias_low': self.bias['low']['magnifi'],
            'magnifi_bias_mid': self.bias['mid']['magnifi'],
            'magnifi_bias_high': self.bias['high']['magnifi'],
        }

    def get_conformal_quantile(self, scores):
        """Get (1-alpha) quantile of nonconformity scores."""
        if not scores or len(scores) < 2:
            return 5.0  # conservative fallback

        n = len(scores)
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = min(q_idx, n - 1)

        return scores[q_idx]

    def predict_with_interval(self, pred_vai, pred_mag):
        """Generate prediction interval with 3-tier calibration + bias correction."""
        vai_group = self._get_group(pred_vai)
        mag_group = self._get_group(pred_mag)

        vai_bias = self.bias[vai_group]['vai']
        mag_bias = self.bias[mag_group]['magnifi']

        corrected_vai = pred_vai - vai_bias
        corrected_mag = pred_mag - mag_bias

        vai_width = self.get_conformal_quantile(self.calibration_scores[vai_group]['vai'])
        mag_width = self.get_conformal_quantile(self.calibration_scores[mag_group]['magnifi'])

        return {
            'vai': {
                'point': round(corrected_vai, 1),
                'raw_point': pred_vai,
                'bias_applied': round(vai_bias, 2),
                'lower': max(0, round(corrected_vai - vai_width, 1)),
                'upper': min(22, round(corrected_vai + vai_width, 1)),
                'width': round(2 * vai_width, 1),
                'group': vai_group
            },
            'magnifi': {
                'point': round(corrected_mag, 1),
                'raw_point': pred_mag,
                'bias_applied': round(mag_bias, 2),
                'lower': max(0, round(corrected_mag - mag_width, 1)),
                'upper': min(25, round(corrected_mag + mag_width, 1)),
                'width': round(2 * mag_width, 1),
                'group': mag_group
            }
        }


def create_folds(results, n_folds):
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
    print("V5 GRANULAR GROUP-BALANCED CROSS-CONFORMAL PREDICTION")
    print("(Using Cached API Results)")
    print("=" * 70)
    print(f"\nTarget Coverage: {(1-ALPHA)*100:.0f}%")
    print(f"Number of Folds: {N_FOLDS}")
    print(f"3-Tier Thresholds: Low (<{THRESH_LOW}), Mid ({THRESH_LOW}-{THRESH_HIGH-1}), High (>={THRESH_HIGH})")

    # Load cached results
    with open(CACHED_RESULTS) as f:
        cached_results = json.load(f)

    print(f"\nLoaded {len(cached_results)} cached predictions")

    # Create lookup by case_id
    results_by_id = {r['case_id']: r for r in cached_results}

    # Create stratified folds
    np.random.seed(42)
    folds = create_folds(cached_results, N_FOLDS)
    print(f"Fold sizes: {[len(f) for f in folds]}")

    # Run cross-conformal prediction
    print("\n" + "-" * 70)
    print("CROSS-CONFORMAL PREDICTION (5-Fold)")
    print("-" * 70)

    final_results = []
    fold_metrics = []

    for fold_idx in range(N_FOLDS):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        test_indices = set(folds[fold_idx])
        cal_indices = [idx for f_idx, fold in enumerate(folds) if f_idx != fold_idx for idx in fold]

        # Gather calibration data
        cal_data = [cached_results[idx] for idx in cal_indices if cached_results[idx].get('predicted_vai') is not None]

        # Initialize conformal predictor
        conformal = GranularGroupBalancedConformal(
            n_folds=N_FOLDS, alpha=ALPHA,
            thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH
        )
        cal_metrics = conformal.calibrate(cal_data, cal_data)

        print(f"  Calibration: {len(cal_data)} cases")
        print(f"    VAI  - Low: {cal_metrics['n_vai_low']} (bias {cal_metrics['vai_bias_low']:+.2f}), Mid: {cal_metrics['n_vai_mid']} (bias {cal_metrics['vai_bias_mid']:+.2f}), High: {cal_metrics['n_vai_high']} (bias {cal_metrics['vai_bias_high']:+.2f})")
        print(f"    MAG  - Low: {cal_metrics['n_magnifi_low']} (bias {cal_metrics['magnifi_bias_low']:+.2f}), Mid: {cal_metrics['n_magnifi_mid']} (bias {cal_metrics['magnifi_bias_mid']:+.2f}), High: {cal_metrics['n_magnifi_high']} (bias {cal_metrics['magnifi_bias_high']:+.2f})")

        # Predict on test fold
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

            # Get interval
            interval = conformal.predict_with_interval(pred_vai, pred_mag)

            # Check coverage
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
                "severity": result.get("severity"),
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
                "vai_group": interval['vai']['group'],
                "magnifi_group": interval['magnifi']['group']
            })

        if fold_count > 0:
            fold_metrics.append({
                'fold': fold_idx,
                'n_test': fold_count,
                'vai_coverage': fold_coverage_vai / fold_count,
                'magnifi_coverage': fold_coverage_mag / fold_count
            })
            print(f"  Test coverage - VAI: {fold_coverage_vai}/{fold_count} ({100*fold_coverage_vai/fold_count:.1f}%), MAG: {fold_coverage_mag}/{fold_count} ({100*fold_coverage_mag/fold_count:.1f}%)")

    # Aggregate results
    print("\n" + "=" * 70)
    print("V5 GRANULAR GROUP-BALANCED CONFORMAL PREDICTION RESULTS")
    print("=" * 70)

    valid_results = [r for r in final_results if r.get('expected_vai') is not None]

    vai_coverages = [r['vai_covered'] for r in valid_results]
    mag_coverages = [r['magnifi_covered'] for r in valid_results]

    vai_errors = [abs(r['corrected_vai'] - r['expected_vai']) for r in valid_results]
    mag_errors = [abs(r['corrected_magnifi'] - r['expected_magnifi']) for r in valid_results]

    vai_widths = [r['vai_width'] for r in valid_results]
    mag_widths = [r['magnifi_width'] for r in valid_results]

    # Group-specific widths
    vai_low_widths = [r['vai_width'] for r in valid_results if r['vai_group'] == 'low']
    vai_mid_widths = [r['vai_width'] for r in valid_results if r['vai_group'] == 'mid']
    vai_high_widths = [r['vai_width'] for r in valid_results if r['vai_group'] == 'high']
    mag_low_widths = [r['magnifi_width'] for r in valid_results if r['magnifi_group'] == 'low']
    mag_mid_widths = [r['magnifi_width'] for r in valid_results if r['magnifi_group'] == 'mid']
    mag_high_widths = [r['magnifi_width'] for r in valid_results if r['magnifi_group'] == 'high']

    print(f"\nTotal valid cases: {len(valid_results)}")

    print(f"\n{'Metric':<35} {'VAI':<15} {'MAGNIFI-CD':<15}")
    print("-" * 65)
    print(f"{'Coverage (target 90%)':<35} {100*np.mean(vai_coverages):.1f}%{'':<9} {100*np.mean(mag_coverages):.1f}%")
    print(f"{'Mean Interval Width':<35} {np.mean(vai_widths):.2f}{'':<11} {np.mean(mag_widths):.2f}")
    print(f"{'  - Low (<{0}) Width'.format(THRESH_LOW):<35} {np.mean(vai_low_widths) if vai_low_widths else 0:.2f}{'':<11} {np.mean(mag_low_widths) if mag_low_widths else 0:.2f}")
    print(f"{'  - Mid ({0}-{1}) Width'.format(THRESH_LOW, THRESH_HIGH-1):<35} {np.mean(vai_mid_widths) if vai_mid_widths else 0:.2f}{'':<11} {np.mean(mag_mid_widths) if mag_mid_widths else 0:.2f}")
    print(f"{'  - High (>={0}) Width'.format(THRESH_HIGH):<35} {np.mean(vai_high_widths) if vai_high_widths else 0:.2f}{'':<11} {np.mean(mag_high_widths) if mag_high_widths else 0:.2f}")
    print(f"{'Corrected MAE':<35} {np.mean(vai_errors):.2f}{'':<11} {np.mean(mag_errors):.2f}")

    # Coverage by severity
    print(f"\n{'COVERAGE BY SEVERITY (Target: >85% all groups):':<50}")
    print("-" * 65)
    for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
        sev_results = [r for r in valid_results if r.get('severity') == sev]
        if sev_results:
            vai_cov = 100 * np.mean([r['vai_covered'] for r in sev_results])
            mag_cov = 100 * np.mean([r['magnifi_covered'] for r in sev_results])
            vai_width = np.mean([r['vai_width'] for r in sev_results])
            mag_width = np.mean([r['magnifi_width'] for r in sev_results])

            vai_status = "✓" if vai_cov >= 85 else "✗"
            mag_status = "✓" if mag_cov >= 85 else "✗"

            print(f"  {sev:<15} VAI: {vai_cov:>5.1f}% {vai_status} (w={vai_width:.1f})  MAG: {mag_cov:>5.1f}% {mag_status} (w={mag_width:.1f})  n={len(sev_results)}")

    # Coverage by group
    print(f"\n{'COVERAGE BY PREDICTION GROUP (3-Tier):':<50}")
    print("-" * 65)
    group_labels = {
        'low': f'<{THRESH_LOW}',
        'mid': f'{THRESH_LOW}-{THRESH_HIGH-1}',
        'high': f'>={THRESH_HIGH}'
    }
    for group in ['low', 'mid', 'high']:
        vai_group_results = [r for r in valid_results if r['vai_group'] == group]
        mag_group_results = [r for r in valid_results if r['magnifi_group'] == group]

        if vai_group_results:
            vai_cov = 100 * np.mean([r['vai_covered'] for r in vai_group_results])
            vai_status = "✓" if vai_cov >= 85 else "✗"
            print(f"  VAI {group.upper():>4} ({group_labels[group]:>7}): {vai_cov:>5.1f}% {vai_status} (n={len(vai_group_results)})")

        if mag_group_results:
            mag_cov = 100 * np.mean([r['magnifi_covered'] for r in mag_group_results])
            mag_status = "✓" if mag_cov >= 85 else "✗"
            print(f"  MAG {group.upper():>4} ({group_labels[group]:>7}): {mag_cov:>5.1f}% {mag_status} (n={len(mag_group_results)})")

    # Save results
    output = {
        "metadata": {
            "version": "V5 Granular Group-Balanced Cross-Conformal",
            "timestamp": datetime.now().isoformat(),
            "n_folds": N_FOLDS,
            "alpha": ALPHA,
            "target_coverage": f"{(1-ALPHA)*100:.0f}%",
            "thresh_low": THRESH_LOW,
            "thresh_high": THRESH_HIGH,
            "groups": f"Low (<{THRESH_LOW}), Mid ({THRESH_LOW}-{THRESH_HIGH-1}), High (>={THRESH_HIGH})"
        },
        "summary": {
            "n_cases": len(valid_results),
            "vai_coverage": float(np.mean(vai_coverages)),
            "magnifi_coverage": float(np.mean(mag_coverages)),
            "vai_mean_width": float(np.mean(vai_widths)),
            "magnifi_mean_width": float(np.mean(mag_widths)),
            "vai_low_width": float(np.mean(vai_low_widths)) if vai_low_widths else None,
            "vai_mid_width": float(np.mean(vai_mid_widths)) if vai_mid_widths else None,
            "vai_high_width": float(np.mean(vai_high_widths)) if vai_high_widths else None,
            "magnifi_low_width": float(np.mean(mag_low_widths)) if mag_low_widths else None,
            "magnifi_mid_width": float(np.mean(mag_mid_widths)) if mag_mid_widths else None,
            "magnifi_high_width": float(np.mean(mag_high_widths)) if mag_high_widths else None,
            "vai_corrected_mae": float(np.mean(vai_errors)),
            "magnifi_corrected_mae": float(np.mean(mag_errors))
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

    # Final summary table
    print("\n" + "=" * 70)
    print("V5 GRANULAR GROUP-BALANCED CONFORMAL PREDICTION RESULTS TABLE")
    print("=" * 70)
    print(f"┌{'─'*68}┐")
    print(f"│{'Metric':<35}{'VAI':<16}{'MAGNIFI-CD':<17}│")
    print(f"├{'─'*68}┤")
    print(f"│{'Overall Coverage (90% target)':<35}{100*np.mean(vai_coverages):>6.1f}%{'':<9}{100*np.mean(mag_coverages):>6.1f}%{'':<10}│")
    print(f"│{'Mean Interval Width':<35}{np.mean(vai_widths):>7.2f}{'':<9}{np.mean(mag_widths):>7.2f}{'':<10}│")
    print(f"│{'  Low (<{0}) Width'.format(THRESH_LOW):<35}{np.mean(vai_low_widths) if vai_low_widths else 0:>7.2f}{'':<9}{np.mean(mag_low_widths) if mag_low_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'  Mid ({0}-{1}) Width'.format(THRESH_LOW, THRESH_HIGH-1):<35}{np.mean(vai_mid_widths) if vai_mid_widths else 0:>7.2f}{'':<9}{np.mean(mag_mid_widths) if mag_mid_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'  High (>={0}) Width'.format(THRESH_HIGH):<35}{np.mean(vai_high_widths) if vai_high_widths else 0:>7.2f}{'':<9}{np.mean(mag_high_widths) if mag_high_widths else 0:>7.2f}{'':<10}│")
    print(f"│{'MAE':<35}{np.mean(vai_errors):>7.2f}{'':<9}{np.mean(mag_errors):>7.2f}{'':<10}│")
    print(f"├{'─'*68}┤")
    print(f"│{'COVERAGE BY SEVERITY (✓ = >85%)':<68}│")
    print(f"├{'─'*68}┤")
    for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
        sev_results = [r for r in valid_results if r.get('severity') == sev]
        if sev_results:
            vai_cov = 100 * np.mean([r['vai_covered'] for r in sev_results])
            mag_cov = 100 * np.mean([r['magnifi_covered'] for r in sev_results])
            vai_status = "✓" if vai_cov >= 85 else "✗"
            mag_status = "✓" if mag_cov >= 85 else "✗"
            print(f"│  {sev:<33}{vai_cov:>5.1f}% {vai_status}{'':<7}{mag_cov:>5.1f}% {mag_status}{'':<8}│")
    print(f"└{'─'*68}┘")


if __name__ == "__main__":
    main()
