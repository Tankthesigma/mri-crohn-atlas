#!/usr/bin/env python3
"""
PARSEC Study-Level Training Script
===================================

Trains GBDT ensemble on study-level data (1,112 studies) instead of
patient-level data (88,773 expanded patients).

WHY STUDY-LEVEL:
- Patient expansion creates ICC=1.0 for all features (artifacts)
- Patient expansion creates duplicate rows (cosine similarity = 1.0)
- Study-level data is the TRUE unit of observation

Features (15 leak-free):
    Treatment: cat_Biologic, cat_Surgical, cat_Combination, cat_Stem_Cell, cat_Antibiotic, cat_Other
    Clinical: followup_weeks, is_refractory, is_rct, combo_therapy
    Fistula: fistula_complexity_Simple, fistula_complexity_Mixed, fistula_complexity_Complex
    History: previous_biologic_failure, is_seton

Target: success_rate_percent > 50 (binary classification)
Weighting: sqrt(n_total) - balances large studies without dominating

Output:
    models/parsec_study_level_oof.npz - OOF predictions
    models/parsec_study_level_report.json - Training metrics

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import minimize

# GBDT imports
import lightgbm as lgb
import xgboost as xgb

# Try catboost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed. Using LightGBM + XGBoost only.")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'v30_ultimate_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SEED = 42
N_FOLDS = 5
N_BAGS = 10  # Bagging for stability

# 15 leak-free features
FEATURE_COLS = [
    # Treatment categories (one-hot, mutually exclusive)
    'cat_Biologic', 'cat_Surgical', 'cat_Combination', 'cat_Stem_Cell',
    'cat_Antibiotic', 'cat_Other',
    # Clinical features
    'followup_weeks', 'is_refractory', 'is_rct', 'combo_therapy',
    # Fistula complexity (one-hot from categorical)
    'fistula_complexity_Simple', 'fistula_complexity_Mixed', 'fistula_complexity_Complex',
    # Patient history
    'previous_biologic_failure', 'is_seton',
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prepare_data():
    """Load v30 study-level data and prepare features."""
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    # Filter to rows with valid success_rate
    df = df[df['success_rate_percent'].notna()].copy()
    print(f"  After removing NaN success_rate: {df.shape}")

    # Create binary target
    df['target'] = (df['success_rate_percent'] > 50).astype(int)

    # Create fistula_complexity one-hot columns
    df['fistula_complexity_Simple'] = (df['fistula_complexity'] == 'Simple').astype(int)
    df['fistula_complexity_Mixed'] = (df['fistula_complexity'] == 'Mixed').astype(int)
    df['fistula_complexity_Complex'] = (df['fistula_complexity'] == 'Complex').astype(int)

    # Create is_seton feature
    df['is_seton'] = df['intervention_name'].str.lower().str.contains('seton', na=False).astype(int)

    # Count setons
    n_seton = df['is_seton'].sum()
    print(f"  Seton studies: {n_seton} ({100*n_seton/len(df):.1f}%)")

    # Fill missing values
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  WARNING: {col} not found, creating with zeros")
            df[col] = 0
        if df[col].isna().any():
            median_val = df[col].median() if df[col].dtype in ['float64', 'int64'] else 0
            df[col] = df[col].fillna(median_val)

    # Create sample weights: sqrt(n_total)
    df['sample_weight'] = np.sqrt(df['n_total'].clip(lower=1))

    # Normalize weights to mean=1
    df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()

    # Create study_id if not present
    if 'study_id' not in df.columns and 'source_file' in df.columns:
        df['study_id'] = pd.factorize(df['source_file'])[0]
    elif 'study_id' not in df.columns:
        df['study_id'] = np.arange(len(df))

    print(f"\nTarget distribution:")
    print(f"  Class 0 (fail): {(df['target']==0).sum()} ({100*(df['target']==0).mean():.1f}%)")
    print(f"  Class 1 (success): {(df['target']==1).sum()} ({100*(df['target']==1).mean():.1f}%)")

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val, w_train=None, w_val=None, seed=42):
    """Train LightGBM with Optuna-tuned params."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'verbosity': -1,
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val] if w_val is not None else None,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model


def train_xgboost(X_train, y_train, X_val, y_val, w_train=None, w_val=None, seed=42):
    """Train XGBoost."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'verbosity': 0,
        'early_stopping_rounds': 50,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val] if w_val is not None else None,
        verbose=False
    )

    return model


def train_catboost(X_train, y_train, X_val, y_val, w_train=None, w_val=None, seed=42):
    """Train CatBoost."""
    if not HAS_CATBOOST:
        return None

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_seed': seed,
        'verbose': False,
    }

    model = cb.CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    return model


def train_all_models_bagged(X, y, train_idx, val_idx, weights, n_bags=10):
    """Train all GBDT models with bagging for stability."""
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = weights[train_idx] if weights is not None else None
    w_val = weights[val_idx] if weights is not None else None

    preds = {
        'lightgbm': [],
        'xgboost': [],
    }
    if HAS_CATBOOST:
        preds['catboost'] = []

    for bag in range(n_bags):
        bag_seed = SEED + bag * 100

        # Bootstrap sample
        np.random.seed(bag_seed)
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[boot_idx]
        y_boot = y_train[boot_idx]
        w_boot = w_train[boot_idx] if w_train is not None else None

        # Train LightGBM
        lgb_model = train_lightgbm(X_boot, y_boot, X_val, y_val, w_boot, w_val, bag_seed)
        preds['lightgbm'].append(lgb_model.predict_proba(X_val)[:, 1])

        # Train XGBoost
        xgb_model = train_xgboost(X_boot, y_boot, X_val, y_val, w_boot, w_val, bag_seed)
        preds['xgboost'].append(xgb_model.predict_proba(X_val)[:, 1])

        # Train CatBoost
        if HAS_CATBOOST:
            cb_model = train_catboost(X_boot, y_boot, X_val, y_val, w_boot, w_val, bag_seed)
            if cb_model is not None:
                preds['catboost'].append(cb_model.predict_proba(X_val)[:, 1])

    # Average bags
    bagged_preds = {}
    for name, pred_list in preds.items():
        if pred_list:
            bagged_preds[name] = np.mean(pred_list, axis=0)

    return bagged_preds


def optimize_weights(preds_dict, y_val):
    """Optimize ensemble weights using Brier score."""
    models = list(preds_dict.keys())
    n_models = len(models)

    X_stack = np.column_stack([preds_dict[m] for m in models])

    def objective(weights):
        blended = np.dot(X_stack, weights)
        blended = np.clip(blended, 1e-7, 1 - 1e-7)
        return brier_score_loss(y_val, blended)

    x0 = np.ones(n_models) / n_models
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=[(0, 1)] * n_models,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )

    return {m: w for m, w in zip(models, result.x / result.x.sum())}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PARSEC Study-Level Training")
    print("=" * 70)
    print(f"\nWhy study-level:")
    print("  - Patient expansion creates ICC=1.0 artifacts")
    print("  - Patient expansion creates duplicate rows")
    print("  - Study is the TRUE unit of observation\n")

    # Load data
    df = load_and_prepare_data()

    # Prepare arrays
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df['target'].values
    weights = df['sample_weight'].values
    study_ids = df['study_id'].values

    print(f"\nFeatures: {X.shape[1]}")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {i+1:2d}. {col}: mean={X[:, i].mean():.3f}, std={X[:, i].std():.3f}")

    # Cross-validation
    print(f"\n{'='*70}")
    print(f"5-Fold Stratified Cross-Validation with {N_BAGS} bags per model")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = {
        'lightgbm': np.zeros(len(y)),
        'xgboost': np.zeros(len(y)),
    }
    if HAS_CATBOOST:
        oof_preds['catboost'] = np.zeros(len(y))

    fold_aucs = {name: [] for name in oof_preds.keys()}
    all_weights = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Train all models with bagging
        fold_preds = train_all_models_bagged(X, y, train_idx, val_idx, weights, N_BAGS)

        # Compute individual AUCs and store OOF
        for name, pred in fold_preds.items():
            oof_preds[name][val_idx] = pred
            auc = roc_auc_score(y[val_idx], pred)
            fold_aucs[name].append(auc)
            print(f"  {name}: AUC = {auc:.4f}")

        # Optimize weights on this fold
        fold_weights = optimize_weights(fold_preds, y[val_idx])
        all_weights.append(fold_weights)
        print(f"  Optimal weights: {fold_weights}")

    # Compute final OOF AUCs
    print(f"\n{'='*70}")
    print("FINAL OOF RESULTS")
    print(f"{'='*70}")

    print("\nIndividual Model AUCs:")
    base_aucs = {}
    for name, preds in oof_preds.items():
        if np.any(preds != 0):
            auc = roc_auc_score(y, preds)
            base_aucs[name] = auc
            print(f"  {name}: {auc:.4f} (folds: {np.mean(fold_aucs[name]):.4f} ± {np.std(fold_aucs[name]):.4f})")

    # Average weights across folds
    avg_weights = {}
    for name in oof_preds.keys():
        avg_weights[name] = np.mean([w.get(name, 0) for w in all_weights])

    # Normalize
    total = sum(avg_weights.values())
    avg_weights = {k: v/total for k, v in avg_weights.items()}

    print(f"\nEnsemble weights (averaged across folds):")
    for name, w in avg_weights.items():
        print(f"  {name}: {w:.4f}")

    # Create ensemble OOF
    ensemble_oof = np.zeros(len(y))
    for name, w in avg_weights.items():
        ensemble_oof += w * oof_preds[name]

    ensemble_auc = roc_auc_score(y, ensemble_oof)
    ensemble_brier = brier_score_loss(y, ensemble_oof)

    print(f"\nEnsemble OOF AUC: {ensemble_auc:.4f}")
    print(f"Ensemble Brier: {ensemble_brier:.4f}")

    # Compare to best single
    best_single = max(base_aucs.values())
    best_single_name = max(base_aucs, key=base_aucs.get)

    if ensemble_auc > best_single:
        print(f"\n✓ Ensemble beats best single ({best_single_name}) by {ensemble_auc - best_single:.4f}")
    else:
        print(f"\n✗ Best single ({best_single_name}) beats ensemble by {best_single - ensemble_auc:.4f}")

    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save OOF predictions
    oof_path = os.path.join(MODELS_DIR, 'parsec_study_level_oof.npz')
    np.savez(
        oof_path,
        study_id=study_ids,
        y=y,
        oof=ensemble_oof,
        feature_names=FEATURE_COLS,
        n_studies=len(y),
        **{f'oof_{name}': preds for name, preds in oof_preds.items()},
    )
    print(f"\nSaved OOF to: {oof_path}")

    # Save report
    report = {
        'data': {
            'n_studies': int(len(y)),
            'n_features': len(FEATURE_COLS),
            'feature_names': FEATURE_COLS,
            'class_balance': {
                'fail': int((y == 0).sum()),
                'success': int((y == 1).sum()),
            },
        },
        'cv': {
            'n_folds': N_FOLDS,
            'n_bags': N_BAGS,
        },
        'base_model_aucs': {k: float(v) for k, v in base_aucs.items()},
        'ensemble': {
            'weights': {k: float(v) for k, v in avg_weights.items()},
            'auc': float(ensemble_auc),
            'brier': float(ensemble_brier),
        },
        'best_single': {
            'model': best_single_name,
            'auc': float(best_single),
        },
    }

    report_path = os.path.join(MODELS_DIR, 'parsec_study_level_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to: {report_path}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext: Run torture_test_parsec.py with study-level data")

    return ensemble_auc


if __name__ == '__main__':
    main()
