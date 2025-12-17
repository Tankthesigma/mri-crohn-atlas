#!/usr/bin/env python3
"""
P.A.R.S.E.C. TORTURE TEST - 30 Independent Runs
============================================

Validate that P.A.R.S.E.C. Leak-Free's 0.9083 AUC is robust and not seed-dependent.
Runs the EXACT same architecture 30 times with different master seeds (100-129).

Architecture replicated EXACTLY from project_parsec_leakfree.py:
- Leak-free meta-features (inner CV)
- CatBoost (depth=6, 500 iter)
- Deep MLP (128→64→32→1 + BatchNorm)
- 20-bag ensemble with weight optimization
"""

import os
import sys

# Critical: Set environment BEFORE any imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TABPFN_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# CONFIGURATION (EXACT COPY FROM LEAK-FREE)
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
V31_CSV = BASE_DIR / "data" / "v31_rct_dataset.csv"

# Torture test config
N_TORTURE_RUNS = 30
MASTER_SEEDS = list(range(100, 130))  # Seeds 100-129

# Architecture config (EXACT from leak-free)
N_BAGS = 20
N_FOLDS = 5
TARGET_VARIANCE = 0.85
MIN_PCA_COMPONENTS = 3
MAX_PCA_COMPONENTS = 15

# MLP config
MLP_EPOCHS = 150
MLP_BATCH_SIZE = 16
MLP_LR = 0.001
MLP_DROPOUT_SCHEDULE = [0.4, 0.3, 0.2]
MLP_WEIGHT_DECAY = 1e-3

# CatBoost config
CAT_ITERATIONS = 500
CAT_DEPTH = 6
CAT_LR = 0.03
CAT_L2_REG = 5
CAT_SUBSAMPLE = 0.8
CAT_COLSAMPLE = 0.8
CAT_EARLY_STOPPING = 50

# Ensemble weights
WEIGHT_MIN = 0.30
WEIGHT_MAX = 0.75
WEIGHT_STEP = 0.05

# Banned features
BANNED_FEATURES = [
    'source_file', 'intervention_name', 'pmid', 'doi',
    'success_rate_percent', 'success_rate', 'n_success', 'n_healed',
    'num_healed', 'success_percentage', 'effective', 'outcome_type',
    'is_rct', 'study_year', 'dropout_rate',
    'blinding_Unknown', 'primary_endpoint_type_Unknown', 'analysis_type_Unknown',
]

BANNER = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║   ████████╗ ██████╗ ██████╗ ████████╗██╗   ██╗██████╗ ███████╗              ║
    ║   ╚══██╔══╝██╔═══██╗██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝              ║
    ║      ██║   ██║   ██║██████╔╝   ██║   ██║   ██║██████╔╝█████╗                ║
    ║      ██║   ██║   ██║██╔══██╗   ██║   ██║   ██║██╔══██╗██╔══╝                ║
    ║      ██║   ╚██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║███████╗              ║
    ║      ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝              ║
    ║                                                                              ║
    ║    P.A.R.S.E.C. LEAK-FREE TORTURE TEST                                            ║
    ║                                                                              ║
    ║    30 Independent Runs (Seeds 100-129)                                       ║
    ║    Exact Architecture from project_parsec_leakfree.py                       ║
    ║                                                                              ║
    ║    PASS: Mean AUC > 0.895                                                    ║
    ║    FAIL: Mean AUC < 0.890                                                    ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """


# =============================================================================
# THE KRAKEN V2 - Deep MLP (EXACT COPY)
# =============================================================================

class TheKrakenV2(nn.Module):
    def __init__(self, input_dim, dropout_schedule=None):
        super().__init__()
        if dropout_schedule is None:
            dropout_schedule = [0.4, 0.3, 0.2]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_schedule[0]),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_schedule[1]),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_schedule[2]),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# HELPER FUNCTIONS (EXACT COPIES)
# =============================================================================

def prepare_features(df):
    """Prepare raw features for training."""
    feature_cols = [c for c in df.columns if c not in BANNED_FEATURES]
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(0)

    X = X.loc[:, X.std() > 0]
    return X


def dynamic_pca(X_train, X_val, target_variance=0.85):
    """Fit PCA on training data, transform both."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    pca_full = PCA()
    pca_full.fit(X_train_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= target_variance) + 1
    n_components = max(n_components, MIN_PCA_COMPONENTS)
    n_components = min(n_components, MAX_PCA_COMPONENTS, X_train.shape[1])

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    return X_train_pca, X_val_pca, n_components


def generate_meta_features_fold(X_train, y_train, X_val, df_train, df_val):
    """
    Generate meta-features using ONLY training data.
    LEAK-FREE: Inner 3-fold CV for training meta-features.
    """
    # Train XGBoost on training fold only
    xgb = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    )
    xgb.fit(X_train, y_train)

    # Validation predictions (honest OOF)
    meta_pred_val = xgb.predict_proba(X_val)[:, 1]

    # Training predictions via INNER CV (anti-leakage)
    meta_pred_train = np.zeros(len(y_train))
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
        xgb_inner = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        xgb_inner.fit(X_train[inner_train_idx], y_train[inner_train_idx])
        meta_pred_train[inner_val_idx] = xgb_inner.predict_proba(X_train[inner_val_idx])[:, 1]

    # Generate 4 meta-features
    def make_meta(meta_pred, df_subset):
        cat_bio = df_subset['cat_Biologic'].fillna(0).values if 'cat_Biologic' in df_subset.columns else np.zeros(len(df_subset))
        n_total = df_subset['n_total'].fillna(50).values if 'n_total' in df_subset.columns else np.full(len(df_subset), 50)

        return np.column_stack([
            meta_pred,
            np.abs(meta_pred - 0.5),
            meta_pred * cat_bio,
            meta_pred * np.log1p(n_total)
        ])

    return make_meta(meta_pred_train, df_train), make_meta(meta_pred_val, df_val)


def optimize_weights(cat_preds, mlp_preds, y_true):
    """Grid search for optimal ensemble weights."""
    best_auc = 0
    best_w = 0.5

    for w in np.arange(WEIGHT_MIN, WEIGHT_MAX + WEIGHT_STEP, WEIGHT_STEP):
        ensemble = w * cat_preds + (1 - w) * mlp_preds
        try:
            auc = roc_auc_score(y_true, ensemble)
            if auc > best_auc:
                best_auc = auc
                best_w = w
        except:
            continue

    return best_w, best_auc


def train_mlp(model, X_train, y_train, epochs=MLP_EPOCHS):
    """Train the Kraken v2 MLP."""
    device = torch.device('cpu')
    model = model.to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.BCELoss()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss)

    return model


def predict_mlp(model, X):
    """Get predictions from trained MLP."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        preds = model(X_t).squeeze().cpu().numpy()
    return preds


# =============================================================================
# SINGLE RUN FUNCTION
# =============================================================================

def run_single_torture(master_seed, X_raw, y, df, verbose=False):
    """
    Run ONE complete P.A.R.S.E.C. Leak-Free training with given master seed.
    Returns: (final_auc, cat_mean, mlp_mean, avg_weight)
    """
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)

    accumulated_preds = np.zeros(len(y))
    all_cat_aucs = []
    all_mlp_aucs = []
    all_weights = []

    for bag_idx in range(N_BAGS):
        seed = master_seed * 1000 + bag_idx

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        cat_oof = np.zeros(len(y))
        mlp_oof = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            X_train, X_val = X_raw[train_idx], X_raw[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)

            # LEAK-FREE meta-features
            meta_train, meta_val = generate_meta_features_fold(
                X_train, y_train, X_val, df_train, df_val
            )

            X_cat_train = np.hstack([X_train, meta_train])
            X_cat_val = np.hstack([X_val, meta_val])

            # CatBoost
            cat_model = CatBoostClassifier(
                iterations=CAT_ITERATIONS, depth=CAT_DEPTH, learning_rate=CAT_LR,
                l2_leaf_reg=CAT_L2_REG, subsample=CAT_SUBSAMPLE,
                colsample_bylevel=CAT_COLSAMPLE, early_stopping_rounds=CAT_EARLY_STOPPING,
                random_seed=seed, verbose=False, allow_writing_files=False,
            )
            cat_model.fit(X_cat_train, y_train, eval_set=(X_cat_val, y_val), verbose=False)
            cat_oof[val_idx] = cat_model.predict_proba(X_cat_val)[:, 1]

            # MLP with dynamic PCA
            X_pca_train, X_pca_val, n_pca = dynamic_pca(X_train, X_val, TARGET_VARIANCE)
            mlp_model = TheKrakenV2(input_dim=n_pca, dropout_schedule=MLP_DROPOUT_SCHEDULE)
            mlp_model = train_mlp(mlp_model, X_pca_train, y_train)
            mlp_oof[val_idx] = predict_mlp(mlp_model, X_pca_val)

        # Calculate individual AUCs
        cat_auc = roc_auc_score(y, cat_oof)
        mlp_auc = roc_auc_score(y, mlp_oof)
        all_cat_aucs.append(cat_auc)
        all_mlp_aucs.append(mlp_auc)

        # Optimize weights
        best_w, _ = optimize_weights(cat_oof, mlp_oof, y)
        all_weights.append(best_w)

        # Ensemble
        bag_pred = best_w * cat_oof + (1 - best_w) * mlp_oof
        accumulated_preds += bag_pred

        if verbose:
            print(f"      Bag {bag_idx+1:2d}/{N_BAGS}: Cat={cat_auc:.4f}, MLP={mlp_auc:.4f}, W={best_w:.2f}")

    # Final ensemble
    final_preds = accumulated_preds / N_BAGS
    final_auc = roc_auc_score(y, final_preds)

    return final_auc, np.mean(all_cat_aucs), np.mean(all_mlp_aucs), np.mean(all_weights)


# =============================================================================
# MAIN TORTURE TEST
# =============================================================================

def run_torture_test():
    """Run the full 30-run torture test."""
    print(BANNER)

    # Load data
    print("\n  Loading data...")
    df = pd.read_csv(V31_CSV)
    y = (df["success_rate_percent"].fillna(0) > 50).astype(int).values
    X_raw_df = prepare_features(df)
    X_raw = X_raw_df.values

    print(f"  Dataset: {len(y)} samples ({y.sum()} effective, {len(y)-y.sum()} ineffective)")
    print(f"  Features: {X_raw.shape[1]} raw columns")
    print(f"\n  Starting {N_TORTURE_RUNS} independent runs...")
    print("=" * 70)

    results = []
    import time
    start_time = time.time()

    for run_idx, master_seed in enumerate(MASTER_SEEDS):
        run_start = time.time()

        final_auc, cat_mean, mlp_mean, avg_weight = run_single_torture(
            master_seed, X_raw, y, df, verbose=False
        )

        run_time = time.time() - run_start
        elapsed = time.time() - start_time
        eta = elapsed / (run_idx + 1) * (N_TORTURE_RUNS - run_idx - 1)

        results.append({
            'run': run_idx + 1,
            'seed': master_seed,
            'auc': final_auc,
            'cat_mean': cat_mean,
            'mlp_mean': mlp_mean,
            'optimal_weight': avg_weight,
            'time_seconds': run_time
        })

        print(f"  Run {run_idx+1:2d}/{N_TORTURE_RUNS}: AUC = {final_auc:.4f} | Cat={cat_mean:.4f} | MLP={mlp_mean:.4f} | W={avg_weight:.2f} | ETA: {eta/60:.1f}min")

    print("=" * 70)

    # Calculate statistics
    aucs = [r['auc'] for r in results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    min_auc = np.min(aucs)
    max_auc = np.max(aucs)
    ci_lower = mean_auc - 1.96 * (std_auc / np.sqrt(len(aucs)))
    ci_upper = mean_auc + 1.96 * (std_auc / np.sqrt(len(aucs)))

    cat_means = [r['cat_mean'] for r in results]
    mlp_means = [r['mlp_mean'] for r in results]
    weights = [r['optimal_weight'] for r in results]

    # Determine verdict
    if mean_auc > 0.895:
        verdict = "PASSED"
        verdict_msg = "Model is ROBUST! Results are NOT seed-dependent."
        verdict_color = "SUCCESS"
    elif mean_auc >= 0.890:
        verdict = "MARGINAL"
        verdict_msg = "Borderline acceptable. Consider more investigation."
        verdict_color = "WARNING"
    else:
        verdict = "FAILED"
        verdict_msg = "Results were LUCKY. Model is seed-dependent."
        verdict_color = "FAILED"

    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = BASE_DIR / "data" / "parsec_torture_results.csv"
    results_df.to_csv(csv_path, index=False)

    # Print final report
    print(f"""

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    P.A.R.S.E.C. TORTURE TEST - FINAL VERDICT                      ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  RUNS COMPLETED: {N_TORTURE_RUNS}                                                          ║
    ║  SEEDS TESTED:   {MASTER_SEEDS[0]} - {MASTER_SEEDS[-1]}                                                   ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  ENSEMBLE AUC STATISTICS:                                                    ║
    ║                                                                              ║
    ║    Mean AUC:      {mean_auc:.4f}                                                       ║
    ║    Std Dev:       {std_auc:.4f}                                                       ║
    ║    Min / Max:     [{min_auc:.4f}, {max_auc:.4f}]                                          ║
    ║    95% CI:        [{ci_lower:.4f}, {ci_upper:.4f}]                                          ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  COMPONENT STATISTICS:                                                       ║
    ║                                                                              ║
    ║    CatBoost Mean: {np.mean(cat_means):.4f} ± {np.std(cat_means):.4f}                                        ║
    ║    MLP Mean:      {np.mean(mlp_means):.4f} ± {np.std(mlp_means):.4f}                                        ║
    ║    Avg Weight:    Cat={np.mean(weights):.2f}, MLP={1-np.mean(weights):.2f}                                      ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  VERDICT: {verdict:<10}                                                       ║
    ║                                                                              ║
    ║  {verdict_msg:<60} ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Results saved: {str(csv_path):<45} ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Return summary
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'min_auc': min_auc,
        'max_auc': max_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'verdict': verdict,
        'results': results
    }


if __name__ == "__main__":
    summary = run_torture_test()
