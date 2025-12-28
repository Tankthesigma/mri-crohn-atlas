#!/usr/bin/env python3
"""
AURA FARM Phase 1: CatBoost Optuna (NUCLEAR)
=============================================

500 Optuna trials with wide hyperparameter search.
No early stopping. Study-balanced weights. Leak-proof.

Usage:
    python catboost_optuna_nuclear.py

Output:
    models/catboost_optuna_oof.npz

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = 'data/v32_with_interactions.csv'

# Same features as your working scripts
LEAK_FREE_FEATURES = [
    'cat_Biologic', 'cat_Surgical', 'cat_Combination', 'cat_Stem_Cell',
    'cat_Antibiotic', 'cat_Other', 'followup_weeks',
    'is_refractory', 'is_rct', 'combo_therapy',
    'fistula_complexity_Simple', 'fistula_complexity_Mixed',
    'fistula_complexity_Complex', 'previous_biologic_failure',
    'is_seton',
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
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 100, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'random_seed': SEED,
            'verbose': False,
            'thread_count': -1,
        }

        # Use GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                params['task_type'] = 'GPU'
                params['devices'] = '0'
        except:
            pass

        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs = []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=study_ids)):
            model = CatBoostClassifier(**params)

            # NO early stopping - train on full iterations
            model.fit(
                X[train_idx], y[train_idx],
                sample_weight=sample_weights[train_idx],
                verbose=False,
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
    print("AURA FARM Phase 1: CatBoost Optuna (NUCLEAR)")
    print("=" * 60)
    print(f"\nTrials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT}s")

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    print(f"  Patients: {len(df):,}")
    print(f"  Studies: {df['study_id'].nunique():,}")

    # Prepare features
    available_features = [f for f in LEAK_FREE_FEATURES if f in df.columns]
    print(f"  Using {len(available_features)} leak-free features")

    X = df[available_features].values
    y = df['outcome'].values
    study_ids = df['study_id'].values

    if 'patient_id' not in df.columns:
        raise ValueError("CRITICAL: patient_id column missing!")
    patient_ids = df['patient_id'].values

    # GEMINI FIX: Root-N Weighting (not 1/N which is too aggressive)
    study_counts = df['study_id'].value_counts()

    def get_weight(n):
        n_safe = max(n, 5)
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
    best_params['random_seed'] = SEED
    best_params['verbose'] = False
    best_params['thread_count'] = -1

    # GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            best_params['task_type'] = 'GPU'
            best_params['devices'] = '0'
    except:
        pass

    oof_preds = np.zeros(len(y))
    fold_aucs = []

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=study_ids)):
        print(f"\nFold {fold+1}/{N_FOLDS}")
        print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}")

        model = CatBoostClassifier(**best_params)
        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sample_weights[train_idx],
            verbose=100,
        )

        oof_preds[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        fold_auc = roc_auc_score(y[val_idx], oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_brier = brier_score_loss(y, oof_preds)

    print("\n" + "=" * 60)
    print("CATBOOST OPTUNA NUCLEAR RESULTS")
    print("=" * 60)
    print(f"\nOverall OOF AUC: {overall_auc:.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"Brier Score: {overall_brier:.4f}")
    print(f"\nvs Original CatBoost (0.6911): {'+' if overall_auc > 0.6911 else ''}{overall_auc - 0.6911:.4f}")
    print(f"vs LightGBM Optuna (0.7004): {'+' if overall_auc > 0.7004 else ''}{overall_auc - 0.7004:.4f}")

    # Save OOF predictions
    os.makedirs('models', exist_ok=True)
    np.savez(
        'models/catboost_optuna_oof.npz',
        patient_id=patient_ids,
        study_id=study_ids,
        y=y,
        oof=oof_preds,
    )
    print(f"\nSaved OOF predictions to models/catboost_optuna_oof.npz")

    # Save report
    report = {
        'best_trial': study.best_trial.number,
        'best_auc': float(study.best_value),
        'final_auc': float(overall_auc),
        'fold_aucs': [float(a) for a in fold_aucs],
        'brier_score': float(overall_brier),
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    with open('models/catboost_optuna_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to models/catboost_optuna_report.json")


if __name__ == '__main__':
    main()
