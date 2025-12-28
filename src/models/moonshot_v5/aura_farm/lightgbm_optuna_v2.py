#!/usr/bin/env python3
"""
AURA FARM Phase 4: LightGBM Optuna V2 (NUCLEAR)
================================================

Re-tune LightGBM with:
- 500 trials (more than before)
- Wider hyperparameter search
- New v33 features

Usage:
    python lightgbm_optuna_v2.py

Output:
    models/lightgbm_optuna_v2_oof.npz

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================

# Use v33 if available (with new features), else fall back to v32
DATA_PATH_V33 = 'data/v33_aura_features.csv'
DATA_PATH_V32 = 'data/v32_with_interactions.csv'

# Extended feature list (includes new v33 features if available)
FEATURES_V33 = [
    # Original leak-free features
    'cat_Biologic', 'cat_Surgical', 'cat_Combination', 'cat_Stem_Cell',
    'cat_Antibiotic', 'cat_Other', 'followup_weeks',
    'is_refractory', 'is_rct', 'combo_therapy',
    'fistula_complexity_Simple', 'fistula_complexity_Mixed',
    'fistula_complexity_Complex', 'previous_biologic_failure',
    'is_seton', 'refractory_x_complex', 'bio_failure_x_refractory',
    # New v33 features
    'bio_x_surgical', 'bio_x_combo', 'surgical_x_combo', 'stem_x_bio',
    'seton_x_bio', 'seton_x_surgical',
    'complex_x_refractory', 'complex_x_bio_failure', 'refractory_x_bio_failure',
    'rct_x_refractory', 'rct_x_complex', 'rct_x_bio_failure',
    'followup_short', 'followup_medium', 'followup_long',
    'short_x_complex', 'long_x_bio', 'long_x_surgical',
    'complexity_score', 'complexity_x_refractory',
    'treatment_intensity', 'risk_score', 'risk_x_intensity',
    'complex_refractory_bio', 'complex_refractory_surgical',
    'followup_per_intensity', 'risk_adjusted_followup',
    'followup_squared', 'complexity_squared',
]

SEED = 42
N_FOLDS = 5
N_TRIALS = 500  # NUCLEAR: 500 trials
TIMEOUT = 7200  # 2 hours max

# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def create_objective(X, y, study_ids, sample_weights):
    """Create Optuna objective with proper CV."""

    def objective(trial):
        # FIX: Make num_leaves conditional on max_depth to avoid degenerate search
        max_depth = trial.suggest_int('max_depth', 3, 15)
        max_leaves = min(255, 2 ** max_depth - 1)
        num_leaves = trial.suggest_int('num_leaves', 15, max(16, max_leaves))

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 10, log=True),
            'path_smooth': trial.suggest_float('path_smooth', 0, 10),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'objective': 'binary',
            'metric': 'auc',
            'random_state': SEED,
            'n_jobs': -1,
            'verbose': -1,
        }

        # num_leaves already constrained above, no need to check again

        # Use GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                params['device'] = 'gpu'
        except:
            pass

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=study_ids)):
            model = lgb.LGBMClassifier(**params)

            # NO early stopping - train on full iterations
            model.fit(
                X[train_idx], y[train_idx],
                sample_weight=sample_weights[train_idx],
            )

            val_pred = model.predict_proba(X[val_idx])[:, 1]
            fold_auc = roc_auc_score(y[val_idx], val_pred)
            fold_aucs.append(fold_auc)

            # FIX: Enable pruning to speed up search
            trial.report(float(np.mean(fold_aucs)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_aucs)

    return objective


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AURA FARM Phase 4: LightGBM Optuna V2 (NUCLEAR)")
    print("=" * 60)
    print(f"\nTrials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT}s")

    # Load data (prefer v33 with new features)
    if os.path.exists(DATA_PATH_V33):
        data_path = DATA_PATH_V33
        features = FEATURES_V33
        print(f"\nUsing v33 data with extended features")
    else:
        data_path = DATA_PATH_V32
        features = FEATURES_V33[:17]  # Original features only
        print(f"\nUsing v32 data (run feature_engineering_v2.py first for more features)")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='latin-1')
    print(f"  Patients: {len(df):,}")
    print(f"  Studies: {df['study_id'].nunique():,}")

    # Prepare features
    available_features = [f for f in features if f in df.columns]
    print(f"  Using {len(available_features)} features")

    X = df[available_features].values
    y = df['outcome'].values
    study_ids = df['study_id'].values

    if 'patient_id' not in df.columns:
        raise ValueError("CRITICAL: patient_id column missing!")
    patient_ids = df['patient_id'].values

    # GEMINI FIX: Root-N Weighting (not 1/N which is too aggressive)
    # Small studies get a boost, but don't dominate large studies
    study_counts = df['study_id'].value_counts()

    def get_weight(n):
        n_safe = max(n, 5)  # Clip to prevent explosion for N=1 or N=2
        return 1.0 / np.sqrt(n_safe)

    sample_weights = df['study_id'].map(lambda s: get_weight(study_counts[s])).values
    sample_weights = sample_weights / sample_weights.mean()
    print(f"  Weight stats: Min={sample_weights.min():.4f}, Max={sample_weights.max():.4f}")

    print(f"  Outcome distribution: {y.mean():.1%} positive")

    # Run Optuna
    print("\n--- Running Optuna (this will take a while) ---")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )

    objective = create_objective(X, y, study_ids, sample_weights)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )

    print(f"\n  Best trial: {study.best_trial.number}")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Train final model with best params and get OOF predictions
    print("\n--- Training Final Model with Best Params ---")

    best_params = study.best_params.copy()
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['random_state'] = SEED
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1

    # Ensure num_leaves constraint
    if best_params['num_leaves'] > 2 ** best_params['max_depth']:
        best_params['num_leaves'] = 2 ** best_params['max_depth'] - 1

    # GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            best_params['device'] = 'gpu'
    except:
        pass

    oof_preds = np.zeros(len(y))
    fold_aucs = []

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=study_ids)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}")

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sample_weights[train_idx],
        )

        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        fold_auc = roc_auc_score(y[val_idx], oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_brier = brier_score_loss(y, oof_preds)

    print("\n" + "=" * 60)
    print("LIGHTGBM OPTUNA V2 NUCLEAR RESULTS")
    print("=" * 60)
    print(f"\nOverall OOF AUC: {overall_auc:.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"Brier Score: {overall_brier:.4f}")
    print(f"\nvs LightGBM Optuna v1 (0.7004): {'+' if overall_auc > 0.7004 else ''}{overall_auc - 0.7004:.4f}")
    print(f"vs Current Best (0.7009): {'+' if overall_auc > 0.7009 else ''}{overall_auc - 0.7009:.4f}")

    # Save OOF predictions
    os.makedirs('models', exist_ok=True)
    np.savez(
        'models/lightgbm_optuna_v2_oof.npz',
        patient_id=patient_ids,
        study_id=study_ids,
        y=y,
        oof=oof_preds,
    )
    print(f"\nSaved OOF predictions to models/lightgbm_optuna_v2_oof.npz")

    # Save report
    report = {
        'best_trial': study.best_trial.number,
        'best_auc': float(study.best_value),
        'final_auc': float(overall_auc),
        'fold_aucs': [float(a) for a in fold_aucs],
        'brier_score': float(overall_brier),
        'best_params': {k: v for k, v in study.best_params.items() if not callable(v)},
        'n_trials': len(study.trials),
        'n_features': len(available_features),
        'features_used': available_features,
    }
    with open('models/lightgbm_optuna_v2_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to models/lightgbm_optuna_v2_report.json")


if __name__ == '__main__':
    main()
