#!/usr/bin/env python3
"""
V3 Cross-Conformal Prediction with Bias Correction
===================================================

MRI-Crohn Atlas ISEF 2026 Project
Parser Validation with Distribution-Free Uncertainty Quantification

This script implements:
1. K-fold cross-conformal prediction for calibrated prediction intervals
2. BIAS CORRECTION: Subtracts mean calibration error before interval generation
3. Full validation metrics (MAE, ICC, coverage, interval width)

V2 Results showed: MAE 1.30, Coverage 25%, Bias +0.78
V3 Goal: Apply bias correction to achieve >80% coverage

Author: Tanmay Vasudeva
"""

import json
import time
import os
import sys
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ============== CONFIGURATION ==============
# API key from environment variable (set by fix_and_reset.py or manually)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
MODEL = "deepseek/deepseek-chat"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CASES_FILE = PROJECT_ROOT / "data" / "parser_validation" / "mega_test_cases.json"
PROGRESS_FILE = SCRIPT_DIR / "parser_validation_progress.json"
RESULTS_FILE = SCRIPT_DIR / "v3_conformal_results.json"

# Cross-conformal parameters
N_FOLDS = 5
ALPHA = 0.10  # 90% coverage target (1 - alpha)
RATE_LIMIT_DELAY = 1.5  # seconds between API calls

# ============== EXTRACTION PROMPT ==============
EXTRACTION_PROMPT = """You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. For EACH feature you identify, include the EXACT quote from the report that supports it.

REPORT TEXT:
{report_text}

Return a JSON object with this EXACT structure:
{
    "features": {
        "fistula_count": {
            "value": <number or null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "fistula_type": {
            "value": <"intersphincteric"|"transsphincteric"|"suprasphincteric"|"extrasphincteric"|"complex"|"simple"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "t2_hyperintensity": {
            "value": <true|false|null>,
            "degree": <"mild"|"moderate"|"marked"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "extension": {
            "value": <"none"|"mild"|"moderate"|"severe"|null>,
            "description": "<description of extension pattern>",
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "collections_abscesses": {
            "value": <true|false|null>,
            "size": <"small"|"large"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "rectal_wall_involvement": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "inflammatory_mass": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "predominant_feature": {
            "value": <"inflammatory"|"fibrotic"|"mixed"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "sphincter_involvement": {
            "internal": <true|false|null>,
            "external": <true|false|null>,
            "evidence": "<exact quote from report>"
        },
        "internal_opening": {
            "location": "<clock position or description>",
            "evidence": "<exact quote from report>"
        },
        "branching": {
            "value": <true|false|null>,
            "evidence": "<exact quote from report>"
        }
    },
    "overall_assessment": {
        "activity": <"active"|"healing"|"healed"|"chronic">,
        "severity": <"mild"|"moderate"|"severe">,
        "confidence": <"high"|"medium"|"low">
    },
    "clinical_notes": "<any additional relevant observations>"
}

IMPORTANT:
- Use null if a feature is not mentioned or cannot be determined
- The "evidence" field must contain the EXACT text from the report
- Be conservative - only report features explicitly stated
- Return ONLY valid JSON, no markdown or explanations"""


# ============== SCORING FUNCTIONS ==============
def calculate_vai(features: Dict) -> int:
    """Calculate VAI score from extracted features - matches parser.js exactly"""
    score = 0

    # Fistula count (0-2)
    count = features.get('fistula_count', {}).get('value')
    if count is not None:
        if count == 0:
            pass
        elif count == 1:
            score += 1
        else:
            score += 2

    # Fistula location/type (0-2)
    ftype = features.get('fistula_type', {}).get('value')
    if ftype:
        if ftype in ['simple', 'intersphincteric']:
            score += 1
        elif ftype in ['transsphincteric', 'suprasphincteric', 'extrasphincteric', 'complex']:
            score += 2

    # Extension (0-4)
    extension = features.get('extension', {}).get('value')
    if extension:
        ext_scores = {'none': 0, 'mild': 2, 'moderate': 3, 'severe': 4}
        score += ext_scores.get(extension, 0)

    # T2 hyperintensity (0-8) - most important for activity
    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        t2_scores = {'mild': 4, 'moderate': 6, 'marked': 8}
        score += t2_scores.get(degree, 6)

    # Collections (0-4)
    if features.get('collections_abscesses', {}).get('value') is True:
        score += 4

    # Rectal wall involvement (0-2)
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 2

    return min(score, 22)


def calculate_magnifi(features: Dict) -> int:
    """Calculate MAGNIFI-CD score from extracted features - matches parser.js exactly"""
    score = 0

    # Fistula count (0-3)
    count = features.get('fistula_count', {}).get('value')
    if count is not None:
        if count == 0:
            pass
        elif count == 1:
            score += 1
        elif count == 2:
            score += 2
        else:
            score += 3

    # Fistula activity based on T2/enhancement (0-6)
    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        act_scores = {'mild': 2, 'moderate': 4, 'marked': 6}
        score += act_scores.get(degree, 4)

    # Collections (0-4)
    collections = features.get('collections_abscesses', {})
    if collections.get('value') is True:
        size = collections.get('size', 'small')
        score += 4 if size == 'large' else 2

    # Inflammatory mass (0-3)
    if features.get('inflammatory_mass', {}).get('value') is True:
        score += 3

    # Rectal wall/proctitis (0-4)
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 4

    # Extension component (0-3)
    extension = features.get('extension', {}).get('value')
    if extension:
        ext_scores = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        score += ext_scores.get(extension, 0)

    # Predominant feature adjustment (0-2)
    pf = features.get('predominant_feature', {}).get('value')
    if pf == 'inflammatory':
        score += 2
    elif pf == 'mixed':
        score += 1

    return min(score, 25)


# ============== API FUNCTIONS ==============
def parse_json_response(content: str) -> Dict:
    """Extract JSON from API response, handling markdown code blocks"""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    return json.loads(content)


def call_api(report_text: str) -> Dict:
    """Call DeepSeek API and return parsed features"""
    prompt = EXTRACTION_PROMPT.replace('{report_text}', report_text.strip())

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mri-crohn-atlas.vercel.app",
                "X-Title": "MRI-Crohn Atlas V3 Conformal Validation"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500
            },
            timeout=60
        )

        if response.status_code != 200:
            return {"error": f"API error {response.status_code}: {response.text[:200]}"}

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        return {"success": True, "data": parsed, "raw": content}

    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ============== CROSS-CONFORMAL PREDICTION ==============
class BiasCorrectCrossConformal:
    """
    Cross-Conformal Prediction with Bias Correction

    Key insight from V2: Positive bias (+0.78) causes low coverage (25%).
    V3 fix: Subtract mean error (bias) from predictions before computing intervals.
    """

    def __init__(self, n_folds: int = 5, alpha: float = 0.10):
        self.n_folds = n_folds
        self.alpha = alpha  # 1 - alpha = coverage level (e.g., 0.10 -> 90%)
        self.calibration_scores = []
        self.bias_vai = 0.0  # Will be calculated from calibration set
        self.bias_magnifi = 0.0

    def calibrate(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """
        Calculate nonconformity scores and bias from calibration set.

        Nonconformity score = |prediction - truth| (absolute residual)
        Bias = mean(prediction - truth) (signed, for correction)
        """
        vai_scores = []
        mag_scores = []
        vai_errors = []  # signed errors for bias
        mag_errors = []

        for pred, truth in zip(predictions, ground_truths):
            pred_vai = pred.get('vai_score', 0)
            pred_mag = pred.get('magnifi_score', 0)
            true_vai = truth.get('expected_vai_score', 0)
            true_mag = truth.get('expected_magnifi_score', 0)

            if pred_vai is not None and true_vai is not None:
                vai_scores.append(abs(pred_vai - true_vai))
                vai_errors.append(pred_vai - true_vai)  # signed

            if pred_mag is not None and true_mag is not None:
                mag_scores.append(abs(pred_mag - true_mag))
                mag_errors.append(pred_mag - true_mag)  # signed

        self.calibration_scores = {
            'vai': sorted(vai_scores),
            'magnifi': sorted(mag_scores)
        }

        # Calculate bias (mean signed error)
        self.bias_vai = np.mean(vai_errors) if vai_errors else 0.0
        self.bias_magnifi = np.mean(mag_errors) if mag_errors else 0.0

        return {
            'n_calibration': len(vai_scores),
            'vai_bias': self.bias_vai,
            'magnifi_bias': self.bias_magnifi,
            'vai_mae': np.mean(vai_scores) if vai_scores else 0,
            'magnifi_mae': np.mean(mag_scores) if mag_scores else 0
        }

    def get_conformal_quantile(self, scores: List[float]) -> float:
        """
        Get the (1-alpha) quantile of nonconformity scores.
        This gives us the interval half-width for coverage guarantee.
        """
        if not scores:
            return 3.0  # default fallback

        n = len(scores)
        # Conformal prediction quantile: ceil((n+1)*(1-alpha))/n
        q_idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_idx = min(q_idx, n - 1)  # don't exceed array bounds

        return scores[q_idx]

    def predict_with_interval(self, pred_vai: float, pred_mag: float) -> Dict:
        """
        Generate prediction interval with BIAS CORRECTION.

        V3 Innovation:
        1. Correct the point prediction: corrected = raw - bias
        2. Generate symmetric interval around corrected prediction
        """
        # Step 1: Apply bias correction
        corrected_vai = pred_vai - self.bias_vai
        corrected_mag = pred_mag - self.bias_magnifi

        # Step 2: Get conformal interval widths
        vai_width = self.get_conformal_quantile(self.calibration_scores.get('vai', []))
        mag_width = self.get_conformal_quantile(self.calibration_scores.get('magnifi', []))

        # Step 3: Generate intervals around CORRECTED predictions
        return {
            'vai': {
                'point': round(corrected_vai, 1),
                'raw_point': pred_vai,  # original uncorrected
                'bias_applied': round(self.bias_vai, 2),
                'lower': max(0, round(corrected_vai - vai_width, 1)),
                'upper': min(22, round(corrected_vai + vai_width, 1)),
                'width': round(2 * vai_width, 1)
            },
            'magnifi': {
                'point': round(corrected_mag, 1),
                'raw_point': pred_mag,
                'bias_applied': round(self.bias_magnifi, 2),
                'lower': max(0, round(corrected_mag - mag_width, 1)),
                'upper': min(25, round(corrected_mag + mag_width, 1)),
                'width': round(2 * mag_width, 1)
            }
        }


def create_folds(cases: List[Dict], n_folds: int) -> List[List[int]]:
    """Create stratified folds based on severity"""
    # Group by severity
    severity_groups = {}
    for i, case in enumerate(cases):
        sev = case.get('severity', 'unknown')
        if sev not in severity_groups:
            severity_groups[sev] = []
        severity_groups[sev].append(i)

    # Distribute each severity group across folds
    folds = [[] for _ in range(n_folds)]
    for sev, indices in severity_groups.items():
        np.random.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)

    return folds


# ============== PROGRESS MANAGEMENT ==============
def load_progress() -> Dict:
    """Load existing progress or return empty state"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": [], "fold_results": {}}


def save_progress(progress: Dict):
    """Save progress after each case"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ============== MAIN VALIDATION ==============
def run_cross_conformal_validation():
    """
    Run K-fold cross-conformal prediction with bias correction.

    For each fold:
    1. Use other folds as calibration set
    2. Calculate bias from calibration errors
    3. Predict on test fold with bias correction
    4. Generate calibrated prediction intervals
    """
    print("=" * 70)
    print("V3 CROSS-CONFORMAL PREDICTION WITH BIAS CORRECTION")
    print("=" * 70)
    print(f"\nTarget Coverage: {(1-ALPHA)*100:.0f}%")
    print(f"Number of Folds: {N_FOLDS}")
    print(f"API: {MODEL}")

    # Load test cases
    if not CASES_FILE.exists():
        print(f"\nERROR: Test cases file not found: {CASES_FILE}")
        return

    with open(CASES_FILE) as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    print(f"\nTotal test cases: {len(cases)}")

    # Load progress
    progress = load_progress()
    completed_ids = set(progress.get("completed", []))
    print(f"Already completed: {len(completed_ids)}")

    # Create stratified folds
    np.random.seed(42)  # for reproducibility
    folds = create_folds(cases, N_FOLDS)
    print(f"Fold sizes: {[len(f) for f in folds]}")

    # Phase 1: Collect all predictions (if not already done)
    print("\n" + "-" * 70)
    print("PHASE 1: Collecting Predictions")
    print("-" * 70)

    all_results = {r['case_id']: r for r in progress.get("results", [])}

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case_{i}")

        if case_id in completed_ids:
            continue

        print(f"\n[{len(completed_ids) + 1}/{len(cases)}] {case_id}")

        ground_truth = case.get("ground_truth", {})
        expected_vai = ground_truth.get("expected_vai_score")
        expected_magnifi = ground_truth.get("expected_magnifi_score")

        # Call API
        result = call_api(case.get("report_text", ""))

        if result.get("error"):
            print(f"    ERROR: {result['error']}")
            case_result = {
                "case_id": case_id,
                "error": result['error'],
                "expected_vai": expected_vai,
                "expected_magnifi": expected_magnifi
            }
        else:
            features = result.get("data", {}).get("features", {})
            pred_vai = calculate_vai(features)
            pred_mag = calculate_magnifi(features)

            case_result = {
                "case_id": case_id,
                "source": case.get("source", "unknown"),
                "case_type": case.get("case_type", "unknown"),
                "severity": case.get("severity", "unknown"),
                "expected_vai": expected_vai,
                "expected_magnifi": expected_magnifi,
                "predicted_vai": pred_vai,
                "predicted_magnifi": pred_mag,
                "vai_error": pred_vai - expected_vai if expected_vai is not None else None,
                "magnifi_error": pred_mag - expected_magnifi if expected_magnifi is not None else None
            }

            print(f"    VAI: {expected_vai} -> {pred_vai} (err: {case_result['vai_error']})")
            print(f"    MAG: {expected_magnifi} -> {pred_mag} (err: {case_result['magnifi_error']})")

        all_results[case_id] = case_result
        completed_ids.add(case_id)

        # Save progress
        progress["completed"] = list(completed_ids)
        progress["results"] = list(all_results.values())
        save_progress(progress)

        time.sleep(RATE_LIMIT_DELAY)

    # Phase 2: Cross-Conformal with Bias Correction
    print("\n" + "-" * 70)
    print("PHASE 2: Cross-Conformal Prediction with Bias Correction")
    print("-" * 70)

    final_results = []
    fold_metrics = []

    for fold_idx in range(N_FOLDS):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        test_indices = set(folds[fold_idx])
        cal_indices = [idx for f_idx, fold in enumerate(folds) if f_idx != fold_idx for idx in fold]

        # Gather calibration data
        cal_predictions = []
        cal_truths = []

        for idx in cal_indices:
            case = cases[idx]
            case_id = case.get("id")
            result = all_results.get(case_id, {})

            if result.get("predicted_vai") is not None:
                cal_predictions.append({
                    'vai_score': result['predicted_vai'],
                    'magnifi_score': result['predicted_magnifi']
                })
                cal_truths.append(case.get("ground_truth", {}))

        # Initialize conformal predictor and calibrate
        conformal = BiasCorrectCrossConformal(n_folds=N_FOLDS, alpha=ALPHA)
        cal_metrics = conformal.calibrate(cal_predictions, cal_truths)

        print(f"  Calibration set size: {cal_metrics['n_calibration']}")
        print(f"  VAI Bias: {cal_metrics['vai_bias']:+.2f}")
        print(f"  MAGNIFI Bias: {cal_metrics['magnifi_bias']:+.2f}")

        # Predict on test fold with bias correction
        fold_coverage_vai = 0
        fold_coverage_mag = 0
        fold_count = 0

        for idx in test_indices:
            case = cases[idx]
            case_id = case.get("id")
            result = all_results.get(case_id, {})
            ground_truth = case.get("ground_truth", {})

            if result.get("predicted_vai") is None:
                continue

            # Get bias-corrected prediction interval
            interval = conformal.predict_with_interval(
                result['predicted_vai'],
                result['predicted_magnifi']
            )

            # Check coverage
            true_vai = ground_truth.get("expected_vai_score", 0)
            true_mag = ground_truth.get("expected_magnifi_score", 0)

            vai_covered = interval['vai']['lower'] <= true_vai <= interval['vai']['upper']
            mag_covered = interval['magnifi']['lower'] <= true_mag <= interval['magnifi']['upper']

            if vai_covered:
                fold_coverage_vai += 1
            if mag_covered:
                fold_coverage_mag += 1
            fold_count += 1

            final_results.append({
                "case_id": case_id,
                "fold": fold_idx,
                "source": case.get("source"),
                "severity": case.get("severity"),
                "expected_vai": int(true_vai) if true_vai is not None else None,
                "expected_magnifi": int(true_mag) if true_mag is not None else None,
                "raw_vai": int(result['predicted_vai']),
                "raw_magnifi": int(result['predicted_magnifi']),
                "corrected_vai": float(interval['vai']['point']),
                "corrected_magnifi": float(interval['magnifi']['point']),
                "vai_interval": [float(interval['vai']['lower']), float(interval['vai']['upper'])],
                "magnifi_interval": [float(interval['magnifi']['lower']), float(interval['magnifi']['upper'])],
                "vai_width": float(interval['vai']['width']),
                "magnifi_width": float(interval['magnifi']['width']),
                "vai_covered": bool(vai_covered),
                "magnifi_covered": bool(mag_covered),
                "vai_bias_applied": float(interval['vai']['bias_applied']),
                "magnifi_bias_applied": float(interval['magnifi']['bias_applied'])
            })

        if fold_count > 0:
            fold_metrics.append({
                'fold': fold_idx,
                'n_test': fold_count,
                'vai_coverage': fold_coverage_vai / fold_count,
                'magnifi_coverage': fold_coverage_mag / fold_count,
                'vai_bias': cal_metrics['vai_bias'],
                'magnifi_bias': cal_metrics['magnifi_bias']
            })
            print(f"  Test set coverage - VAI: {fold_coverage_vai}/{fold_count} ({100*fold_coverage_vai/fold_count:.1f}%)")
            print(f"  Test set coverage - MAG: {fold_coverage_mag}/{fold_count} ({100*fold_coverage_mag/fold_count:.1f}%)")

    # Phase 3: Aggregate Results
    print("\n" + "=" * 70)
    print("V3 CROSS-CONFORMAL PREDICTION RESULTS")
    print("=" * 70)

    valid_results = [r for r in final_results if r.get('expected_vai') is not None]

    if valid_results:
        # Calculate aggregate metrics
        vai_coverages = [r['vai_covered'] for r in valid_results]
        mag_coverages = [r['magnifi_covered'] for r in valid_results]

        vai_errors = [abs(r['corrected_vai'] - r['expected_vai']) for r in valid_results]
        mag_errors = [abs(r['corrected_magnifi'] - r['expected_magnifi']) for r in valid_results]

        raw_vai_errors = [abs(r['raw_vai'] - r['expected_vai']) for r in valid_results]
        raw_mag_errors = [abs(r['raw_magnifi'] - r['expected_magnifi']) for r in valid_results]

        vai_widths = [r['vai_width'] for r in valid_results]
        mag_widths = [r['magnifi_width'] for r in valid_results]

        # Bias statistics
        vai_biases = [r['vai_bias_applied'] for r in valid_results]
        mag_biases = [r['magnifi_bias_applied'] for r in valid_results]

        print(f"\nTotal valid cases: {len(valid_results)}")

        print(f"\n{'Metric':<30} {'VAI':<15} {'MAGNIFI-CD':<15}")
        print("-" * 60)
        print(f"{'Coverage (target 90%)':<30} {100*np.mean(vai_coverages):.1f}%{'':<9} {100*np.mean(mag_coverages):.1f}%")
        print(f"{'Mean Interval Width':<30} {np.mean(vai_widths):.2f}{'':<11} {np.mean(mag_widths):.2f}")
        print(f"{'Corrected MAE':<30} {np.mean(vai_errors):.2f}{'':<11} {np.mean(mag_errors):.2f}")
        print(f"{'Raw MAE (before correction)':<30} {np.mean(raw_vai_errors):.2f}{'':<11} {np.mean(raw_mag_errors):.2f}")
        print(f"{'Mean Bias Applied':<30} {np.mean(vai_biases):+.2f}{'':<10} {np.mean(mag_biases):+.2f}")

        # Coverage by severity
        print(f"\n{'Coverage by Severity:':<30}")
        print("-" * 60)
        for sev in ['Remission', 'Mild', 'Moderate', 'Severe']:
            sev_results = [r for r in valid_results if r.get('severity') == sev]
            if sev_results:
                vai_cov = 100 * np.mean([r['vai_covered'] for r in sev_results])
                mag_cov = 100 * np.mean([r['magnifi_covered'] for r in sev_results])
                print(f"  {sev:<26} {vai_cov:.1f}%{'':<9} {mag_cov:.1f}%  (n={len(sev_results)})")

        # Save results
        output = {
            "metadata": {
                "version": "V3 Bias-Corrected Cross-Conformal",
                "timestamp": datetime.now().isoformat(),
                "n_folds": N_FOLDS,
                "alpha": ALPHA,
                "target_coverage": f"{(1-ALPHA)*100:.0f}%",
                "model": MODEL
            },
            "summary": {
                "n_cases": len(valid_results),
                "vai_coverage": float(np.mean(vai_coverages)),
                "magnifi_coverage": float(np.mean(mag_coverages)),
                "vai_mean_width": float(np.mean(vai_widths)),
                "magnifi_mean_width": float(np.mean(mag_widths)),
                "vai_corrected_mae": float(np.mean(vai_errors)),
                "magnifi_corrected_mae": float(np.mean(mag_errors)),
                "vai_raw_mae": float(np.mean(raw_vai_errors)),
                "magnifi_raw_mae": float(np.mean(raw_mag_errors)),
                "mean_vai_bias": float(np.mean(vai_biases)),
                "mean_magnifi_bias": float(np.mean(mag_biases))
            },
            "fold_metrics": fold_metrics,
            "detailed_results": final_results
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {RESULTS_FILE}")

        # Final summary box
        print("\n" + "=" * 70)
        print("V3 CROSS-CONFORMAL PREDICTION RESULTS TABLE")
        print("=" * 70)
        print(f"┌{'─'*68}┐")
        print(f"│{'Metric':<35}{'VAI':<16}{'MAGNIFI-CD':<17}│")
        print(f"├{'─'*68}┤")
        print(f"│{'Coverage (90% target)':<35}{100*np.mean(vai_coverages):>6.1f}%{'':<9}{100*np.mean(mag_coverages):>6.1f}%{'':<10}│")
        print(f"│{'Mean Interval Width':<35}{np.mean(vai_widths):>7.2f}{'':<9}{np.mean(mag_widths):>7.2f}{'':<10}│")
        print(f"│{'Corrected MAE':<35}{np.mean(vai_errors):>7.2f}{'':<9}{np.mean(mag_errors):>7.2f}{'':<10}│")
        print(f"│{'Mean Bias Applied':<35}{np.mean(vai_biases):>+7.2f}{'':<9}{np.mean(mag_biases):>+7.2f}{'':<10}│")
        print(f"│{'Cases Evaluated':<35}{len(valid_results):>7}{'':<9}{len(valid_results):>7}{'':<10}│")
        print(f"└{'─'*68}┘")

        return output
    else:
        print("\nNo valid results to analyze")
        return None


if __name__ == "__main__":
    run_cross_conformal_validation()
