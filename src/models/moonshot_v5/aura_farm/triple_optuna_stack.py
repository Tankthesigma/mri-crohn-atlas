#!/usr/bin/env python3
"""
AURA FARM Phase 5: Triple Optuna Stack (FINAL BOSS)
=====================================================

Stack all three Optuna-tuned GBDTs:
- LightGBM Optuna v2
- CatBoost Optuna
- XGBoost Optuna

This is the final ensemble. If this doesn't hit 0.72, nothing will.

Usage:
    python triple_optuna_stack.py

Output:
    models/triple_optuna_final.npz

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# =============================================================================
# CONFIG
# =============================================================================

OOF_FILES = {
    # Optuna-tuned models (if available)
    'lightgbm_optuna_v2': 'models/lightgbm_optuna_v2_oof.npz',
    'catboost_optuna': 'models/catboost_optuna_oof.npz',
    'xgboost_optuna': 'models/xgboost_optuna_oof.npz',
    # Fallbacks to original models
    'lightgbm_optuna': 'models/lightgbm_optuna_oof.npz',
    'catboost': 'models/catboost_oof.npz',
    'xgboost': 'models/xgboost_oof.npz',
}

SEED = 42
N_FOLDS = 5
TARGET_AUC = 0.72

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prob_to_logit(p, eps=1e-5):
    """Convert probabilities to logits."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def load_best_models():
    """Load best available OOF files."""
    print("=" * 60)
    print("AURA FARM Phase 5: Triple Optuna Stack")
    print("=" * 60)

    oof_data = {}
    loaded = []

    # Priority order: Optuna v2 > Optuna > Original
    priority = [
        ('lightgbm_optuna_v2', 'lightgbm_optuna', 'lightgbm'),
        ('catboost_optuna', 'catboost'),
        ('xgboost_optuna', 'xgboost'),
    ]

    for group in priority:
        for name in group:
            path = OOF_FILES.get(name)
            if path and os.path.exists(path):
                oof_data[name] = np.load(path, allow_pickle=True)
                loaded.append(name)
                print(f"  Loaded: {name}")
                break  # Only load first available from each group

    print(f"\n  Total models: {len(oof_data)}")

    if len(oof_data) < 2:
        print("ERROR: Need at least 2 models")
        return None

    return oof_data


def validate_and_align(oof_data):
    """Validate alignment and compute base AUCs."""
    ref_name = list(oof_data.keys())[0]
    ref = oof_data[ref_name]
    y = ref['y']
    patient_ids = ref['patient_id']
    study_ids = ref['study_id']

    print("\n--- Base Model Performance ---")
    base_aucs = {}

    for name, data in oof_data.items():
        if not np.array_equal(data['y'], y):
            print(f"  WARNING: {name} labels don't match!")
            continue
        if not np.array_equal(data['patient_id'], patient_ids):
            print(f"  WARNING: {name} patient_ids don't match!")
            continue
        if not np.array_equal(data['study_id'], study_ids):
            print(f"  WARNING: {name} study_ids don't match!")
            continue

        oof = data['oof']
        if not np.all(np.isfinite(oof)):
            print(f"  WARNING: {name} has non-finite predictions!")
            continue
        if np.any(oof < 0) or np.any(oof > 1):
            print(f"  WARNING: {name} OOF not in [0,1] range!")
            continue

        auc = roc_auc_score(y, oof)
        base_aucs[name] = auc
        diff = auc - TARGET_AUC
        symbol = '+' if diff >= 0 else ''
        print(f"  {name}: {auc:.4f} ({symbol}{diff:.4f} vs 0.72)")

        if auc > 0.78:
            print(f"    FLAG: Suspiciously high AUC - check for leakage!")

    return y, patient_ids, study_ids, base_aucs


def scipy_optimize_weights(X_stack, y, study_ids, model_names):
    """Scipy weight optimization with CV."""
    print("\n--- Scipy Weight Optimization ---")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_models = X_stack.shape[1]

    all_weights = []
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_stack, y, groups=study_ids)):
        X_train, X_val = X_stack[train_idx], X_stack[val_idx]
        y_train = y[train_idx]

        def objective(weights):
            preds = np.dot(X_train, weights)
            preds = np.clip(preds, 1e-5, 1 - 1e-5)
            return brier_score_loss(y_train, preds)

        x0 = np.ones(n_models) / n_models
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )

        # FIX: Check optimizer success before using weights
        if not result.success or not np.all(np.isfinite(result.x)):
            weights = np.ones(n_models) / n_models  # Fallback to equal weights
        else:
            weights = result.x / np.sum(result.x)
        all_weights.append(weights)
        oof_preds[val_idx] = np.dot(X_val, weights)

    mean_weights = np.mean(all_weights, axis=0)
    overall_auc = roc_auc_score(y, oof_preds)

    print("\n  Optimized Weights:")
    for name, w in zip(model_names, mean_weights):
        print(f"    {name}: {w:.4f}")
    print(f"\n  Scipy Optimized AUC: {overall_auc:.4f}")

    return overall_auc, oof_preds, mean_weights


def calibrated_stack(X_stack, y, study_ids):
    """Calibrated stacking with isotonic regression."""
    print("\n--- Calibrated Stack (Isotonic) ---")

    # Average predictions first
    avg_pred = np.mean(X_stack, axis=1)

    # Calibrate with isotonic regression using CV
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    calibrated_preds = np.zeros(len(y))

    for train_idx, val_idx in sgkf.split(X_stack, y, groups=study_ids):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(avg_pred[train_idx], y[train_idx])
        calibrated_preds[val_idx] = iso.predict(avg_pred[val_idx])

    auc = roc_auc_score(y, calibrated_preds)
    print(f"  Calibrated AUC: {auc:.4f}")

    return auc, calibrated_preds


def rank_average(X_stack, y):
    """Rank-based averaging (more robust to scale differences)."""
    from scipy.stats import rankdata

    print("\n--- Rank Average ---")

    # Convert each column to ranks
    X_ranked = np.column_stack([rankdata(X_stack[:, i]) for i in range(X_stack.shape[1])])
    # Normalize to [0, 1]
    X_ranked = X_ranked / len(y)
    # Average ranks
    rank_avg = np.mean(X_ranked, axis=1)

    auc = roc_auc_score(y, rank_avg)
    print(f"  Rank Average AUC: {auc:.4f}")

    return auc, rank_avg


def geometric_mean(X_stack, y):
    """Geometric mean (good for probabilities)."""
    print("\n--- Geometric Mean ---")

    # Clip to avoid log(0)
    X_clipped = np.clip(X_stack, 1e-5, 1 - 1e-5)
    geo_mean = np.exp(np.mean(np.log(X_clipped), axis=1))

    auc = roc_auc_score(y, geo_mean)
    print(f"  Geometric Mean AUC: {auc:.4f}")

    return auc, geo_mean


def power_mean(X_stack, y, power=2):
    """Power mean (power=2 emphasizes higher predictions)."""
    print(f"\n--- Power Mean (p={power}) ---")

    pm = np.mean(X_stack ** power, axis=1) ** (1/power)

    auc = roc_auc_score(y, pm)
    print(f"  Power Mean AUC: {auc:.4f}")

    return auc, pm


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load best available models
    oof_data = load_best_models()
    if oof_data is None:
        return

    # Validate and get base AUCs
    y, patient_ids, study_ids, base_aucs = validate_and_align(oof_data)

    model_names = list(base_aucs.keys())
    X_stack = np.column_stack([oof_data[name]['oof'] for name in model_names])

    print(f"\nStack shape: {X_stack.shape} ({len(model_names)} models)")

    # Compute correlations
    print("\n--- Model Diversity ---")
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i+1:]:
            corr, _ = pearsonr(oof_data[m1]['oof'], oof_data[m2]['oof'])
            flag = " [HIGH]" if corr > 0.90 else ""
            print(f"  {m1} vs {m2}: {corr:.3f}{flag}")

    # Try all stacking methods
    results = {}

    # 1. Simple average
    avg_pred = np.mean(X_stack, axis=1)
    avg_auc = roc_auc_score(y, avg_pred)
    results['Simple Average'] = (avg_auc, avg_pred)
    print(f"\n  Simple Average AUC: {avg_auc:.4f}")

    # 2. Scipy optimized
    scipy_auc, scipy_pred, scipy_weights = scipy_optimize_weights(
        X_stack, y, study_ids, model_names
    )
    results['Scipy Optimized'] = (scipy_auc, scipy_pred)

    # 3. Calibrated stack
    cal_auc, cal_pred = calibrated_stack(X_stack, y, study_ids)
    results['Calibrated (Isotonic)'] = (cal_auc, cal_pred)

    # 4. Rank average
    rank_auc, rank_pred = rank_average(X_stack, y)
    results['Rank Average'] = (rank_auc, rank_pred)

    # 5. Geometric mean
    geo_auc, geo_pred = geometric_mean(X_stack, y)
    results['Geometric Mean'] = (geo_auc, geo_pred)

    # 6. Power mean
    pow_auc, pow_pred = power_mean(X_stack, y, power=2)
    results['Power Mean (p=2)'] = (pow_auc, pow_pred)

    # Find best
    best_method = max(results, key=lambda x: results[x][0])
    best_auc, best_pred = results[best_method]

    # Final summary
    print("\n" + "=" * 60)
    print("TRIPLE OPTUNA STACK FINAL RESULTS")
    print("=" * 60)

    print(f"\nModels in ensemble: {len(model_names)}")
    for name in model_names:
        print(f"  - {name}: {base_aucs[name]:.4f}")

    print("\nStacking Methods (sorted by AUC):")
    for method, (auc, _) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        diff = auc - TARGET_AUC
        symbol = '+' if diff >= 0 else ''
        star = ' <- BEST' if method == best_method else ''
        print(f"  {method}: {auc:.4f} ({symbol}{diff:.4f}){star}")

    # Compare to single best
    single_best_auc = max(base_aucs.values())
    single_best_name = max(base_aucs, key=base_aucs.get)

    print(f"\n--- Final Comparison ---")
    print(f"  Single Best ({single_best_name}): {single_best_auc:.4f}")
    print(f"  Best Ensemble ({best_method}): {best_auc:.4f}")

    if best_auc > single_best_auc:
        print(f"\n  ENSEMBLE WINS by {best_auc - single_best_auc:.4f}!")
    else:
        print(f"\n  Single model wins. Difference: {single_best_auc - best_auc:.4f}")

    if best_auc >= TARGET_AUC:
        print("\n" + "=" * 60)
        print("  MOONSHOT ACHIEVED! 0.72+ AUC!")
        print("=" * 60)
    elif best_auc >= 0.71:
        print(f"\n  SO CLOSE! {best_auc:.4f} - just {TARGET_AUC - best_auc:.4f} away!")
    else:
        print(f"\n  Final AUC: {best_auc:.4f}")
        print(f"  Gap to 0.72: {TARGET_AUC - best_auc:.4f}")

    # Save best predictions
    os.makedirs('models', exist_ok=True)

    final_preds = best_pred if best_auc > single_best_auc else oof_data[single_best_name]['oof']
    final_auc = best_auc if best_auc > single_best_auc else single_best_auc
    final_method = best_method if best_auc > single_best_auc else f"Single: {single_best_name}"

    np.savez(
        'models/triple_optuna_final.npz',
        patient_id=patient_ids,
        study_id=study_ids,
        y=y,
        oof=final_preds,
        method=final_method,
        auc=final_auc,
    )
    print(f"\nSaved final predictions to models/triple_optuna_final.npz")

    # Save report
    report = {
        'models': model_names,
        'base_aucs': {k: float(v) for k, v in base_aucs.items()},
        'stacking_aucs': {k: float(v[0]) for k, v in results.items()},
        'best_method': best_method,
        'best_auc': float(best_auc),
        'single_best_model': single_best_name,
        'single_best_auc': float(single_best_auc),
        'final_method': final_method,
        'final_auc': float(final_auc),
        'target_auc': TARGET_AUC,
        'moonshot_achieved': best_auc >= TARGET_AUC,
        'scipy_weights': {name: float(w) for name, w in zip(model_names, scipy_weights)},
    }
    with open('models/triple_optuna_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to models/triple_optuna_report.json")


if __name__ == '__main__':
    main()
