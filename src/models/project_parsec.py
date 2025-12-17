#!/usr/bin/env python3
"""
PROJECT P.A.R.S.E.C. - The Deep Hybrid Beast
========================================

Architecture:
- 50% CatBoost (trained on 39 raw features for diversity)
- 50% Deep MLP "The Kraken" (trained on 5 PCA components for pure signal)

Features:
- PCA preprocessing to extract top 5 principal components
- Deep MLP with dropout (0.3) and L2 regularization
- Bagged ensemble (20 seeds) for stability
- Bootstrap CI for confidence intervals

Target:
- AUC > 0.90 (The Dream)
- Std Dev < 0.005 (Stable)

Author: Tanmay (ISEF 2026)
"""

import os
import sys

# --- CRITICAL: MAC OS DEADLOCK & OFFLINE FIXES ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = "/Users/tanmaydagoat/Desktop/Antigrav crohns trial"
sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------

import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# PyTorch for Deep MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path("/Users/tanmaydagoat/Desktop/Antigrav crohns trial")
V43_CSV = BASE_DIR / "data/v43_expanded_dataset.csv"
OUTPUT_REPORT = BASE_DIR / "models/parsec_report.json"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

# PCA Config
N_PCA_COMPONENTS = 5

# Deep MLP "The Kraken" Config
MLP_HIDDEN_1 = 32
MLP_HIDDEN_2 = 16
MLP_DROPOUT = 0.3
MLP_L2_DECAY = 1e-4
MLP_EPOCHS = 100
MLP_BATCH_SIZE = 16
MLP_LR = 0.001

# Ensemble Config
N_BAGS = 20
CATBOOST_WEIGHT = 0.5
MLP_WEIGHT = 0.5

# Stability Config
N_STABILITY_TESTS = 10

# Features to ban (leakage)
BANNED_FEATURES = [
    'num_healed', 'n_healed', 'n_success',
    'success_percentage', 'success_class', 'success_rate', 'success_rate_percent',
    'oracle_vibe_score', 'adjusted_vibe', 'vibe_score',
    'confidence_score', 'stringency_score', 'stringency_x_vibe', 'stringency_rating',
    'dropout_rate', 'dropout_percentage',
    'effective', 'outcome_type',
]


def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║     ██████╗██╗  ██╗██╗███╗   ███╗███████╗██████╗  █████╗                     ║
    ║    ██╔════╝██║  ██║██║████╗ ████║██╔════╝██╔══██╗██╔══██╗                    ║
    ║    ██║     ███████║██║██╔████╔██║█████╗  ██████╔╝███████║                    ║
    ║    ██║     ██╔══██║██║██║╚██╔╝██║██╔══╝  ██╔══██╗██╔══██║                    ║
    ║    ╚██████╗██║  ██║██║██║ ╚═╝ ██║███████╗██║  ██║██║  ██║                    ║
    ║     ╚═════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                    ║
    ║                                                                              ║
    ║    PROJECT P.A.R.S.E.C. - The Deep Hybrid Beast                                   ║
    ║                                                                              ║
    ║    50% CatBoost (39 raw features) + 50% Deep MLP (5 PCA components)          ║
    ║    Bagged Ensemble (20 seeds) for maximum stability                          ║
    ║                                                                              ║
    ║    Target: AUC > 0.90 | Std < 0.005                                          ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# THE KRAKEN - Deep MLP Model
# =============================================================================

class TheKraken(nn.Module):
    """
    Deep MLP for binary classification on PCA features.

    Architecture:
        Input (5 PCA) -> Dense(32, ReLU) -> Dropout(0.3)
        -> Dense(16, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)
    """

    def __init__(self, input_dim=5, hidden1=32, hidden2=16, dropout=0.3):
        super(TheKraken, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_kraken(X_train, y_train, X_val, seed=42):
    """Train The Kraken (Deep MLP) on PCA features."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)

    # Create dataset and loader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)

    # Initialize model
    model = TheKraken(
        input_dim=X_train.shape[1],
        hidden1=MLP_HIDDEN_1,
        hidden2=MLP_HIDDEN_2,
        dropout=MLP_DROPOUT
    )

    # Loss and optimizer with L2 regularization (weight decay)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_L2_DECAY)

    # Training loop
    model.train()
    for epoch in range(MLP_EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).numpy().flatten()

    return model, val_preds


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data():
    """Load V43 expanded dataset."""
    if not V43_CSV.exists():
        raise FileNotFoundError(f"V43 dataset not found: {V43_CSV}")

    df = pd.read_csv(V43_CSV)
    print(f"  Loaded {len(df)} samples from V43 expanded dataset")
    return df


def prepare_features(df):
    """Prepare features for training."""

    # Target: success > 50%
    if 'success_rate_percent' in df.columns:
        target = (df['success_rate_percent'].fillna(50) > 50).astype(int)
    else:
        target = (df['success_percentage'].fillna(50) > 50).astype(int)

    # Metadata columns to exclude
    metadata_cols = [
        'source_file', 'intervention_name', 'intervention_category',
        'study_group', 'sample_weight', 'data_source', 'pmc_id',
    ]

    # Build feature matrix
    feature_cols = []
    X_list = []

    for col in df.columns:
        # Skip banned and metadata
        if col.lower() in [b.lower() for b in BANNED_FEATURES]:
            continue
        if col in metadata_cols:
            continue
        if col.startswith('Unnamed'):
            continue
        if col == 'effective':  # Skip target
            continue

        try:
            col_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            if np.std(col_data) > 0:  # Only include features with variance
                X_list.append(col_data)
                feature_cols.append(col)
        except:
            pass

    X = np.column_stack(X_list).astype(np.float32)
    y = target.values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(y)} (Effective: {sum(y)}, Ineffective: {len(y) - sum(y)})")

    return X, y, feature_cols


def apply_pca(X_train, X_val, n_components=5, scaler=None, pca=None):
    """Apply PCA to extract top N principal components."""

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    if pca is None:
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
    else:
        X_train_pca = pca.transform(X_train_scaled)

    X_val_pca = pca.transform(X_val_scaled)

    return X_train_pca, X_val_pca, scaler, pca


# =============================================================================
# CATBOOST COMPONENT
# =============================================================================

def train_catboost(X_train, y_train, X_val, seed=42):
    """Train CatBoost on raw features."""

    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:, 1]

    return model, val_preds


# =============================================================================
# P.A.R.S.E.C. ENSEMBLE
# =============================================================================

def train_parsec_bag(X_raw, y, seed=42):
    """
    Train one bag of the P.A.R.S.E.C. ensemble.

    Returns out-of-fold predictions for both components.
    """
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    oof_catboost = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # CatBoost on raw features
        _, cat_preds = train_catboost(X_train_raw, y_train, X_val_raw, seed=seed+fold)
        oof_catboost[val_idx] = cat_preds

        # PCA + MLP on PCA features
        X_train_pca, X_val_pca, _, _ = apply_pca(X_train_raw, X_val_raw, n_components=N_PCA_COMPONENTS)
        _, mlp_preds = train_kraken(X_train_pca, y_train, X_val_pca, seed=seed+fold)
        oof_mlp[val_idx] = mlp_preds

    # Ensemble: 50% CatBoost + 50% MLP
    oof_ensemble = CATBOOST_WEIGHT * oof_catboost + MLP_WEIGHT * oof_mlp

    return oof_ensemble, oof_catboost, oof_mlp


def run_stability_test(X, y, n_bags=20, test_seed=0):
    """Run one stability test with N_BAGS seeds."""

    accumulated_preds = np.zeros(len(y))
    cat_aucs = []
    mlp_aucs = []

    for bag_idx in range(n_bags):
        seed = test_seed * 1000 + bag_idx

        oof_ensemble, oof_cat, oof_mlp = train_parsec_bag(X, y, seed=seed)
        accumulated_preds += oof_ensemble

        cat_auc = roc_auc_score(y, oof_cat)
        mlp_auc = roc_auc_score(y, oof_mlp)
        cat_aucs.append(cat_auc)
        mlp_aucs.append(mlp_auc)

        print(f"    Bag {bag_idx+1:2d}/{n_bags}: CatBoost={cat_auc:.4f}, MLP={mlp_auc:.4f}")

    # Average predictions across bags
    final_preds = accumulated_preds / n_bags
    final_auc = roc_auc_score(y, final_preds)

    return final_auc, final_preds, np.mean(cat_aucs), np.mean(mlp_aucs)


def calculate_bootstrap_ci(y_true, y_pred, n_iterations=1000):
    """Calculate 95% Bootstrap Confidence Interval."""

    bootstrap_aucs = []
    n = len(y_true)

    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        try:
            auc = roc_auc_score(y_true[idx], y_pred[idx])
            bootstrap_aucs.append(auc)
        except:
            pass

    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return ci_lower, ci_upper, bootstrap_aucs


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_banner()

    # Check dependencies
    if not HAS_CATBOOST:
        print("ERROR: CatBoost required! Run: pip install catboost")
        sys.exit(1)
    if not HAS_TORCH:
        print("ERROR: PyTorch required! Run: pip install torch")
        sys.exit(1)

    print(f"\n  Configuration:")
    print(f"    PCA Components: {N_PCA_COMPONENTS}")
    print(f"    MLP Architecture: {N_PCA_COMPONENTS} -> {MLP_HIDDEN_1} -> {MLP_HIDDEN_2} -> 1")
    print(f"    Dropout: {MLP_DROPOUT}, L2 Decay: {MLP_L2_DECAY}")
    print(f"    Ensemble: {int(CATBOOST_WEIGHT*100)}% CatBoost + {int(MLP_WEIGHT*100)}% MLP")
    print(f"    Bags: {N_BAGS}, Stability Tests: {N_STABILITY_TESTS}")

    # Step 1: Load Data
    print("\n" + "=" * 70)
    print("STEP 1: LOAD V43 EXPANDED DATA")
    print("=" * 70)

    df = load_data()
    X, y, feature_cols = prepare_features(df)

    # Step 2: PCA Analysis
    print("\n" + "=" * 70)
    print("STEP 2: PCA ANALYSIS")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(N_PCA_COMPONENTS, X.shape[1]))
    pca.fit(X_scaled)

    print(f"  Top {N_PCA_COMPONENTS} PCA Components:")
    cumulative_var = 0
    for i, var in enumerate(pca.explained_variance_ratio_[:N_PCA_COMPONENTS]):
        cumulative_var += var
        print(f"    PC{i+1}: {var*100:.1f}% variance (cumulative: {cumulative_var*100:.1f}%)")

    # Step 3: Stability Tests
    print("\n" + "=" * 70)
    print("STEP 3: P.A.R.S.E.C. STABILITY TESTS")
    print("=" * 70)

    all_aucs = []
    all_cat_aucs = []
    all_mlp_aucs = []

    for test_idx in range(N_STABILITY_TESTS):
        print(f"\n  Stability Test {test_idx+1}/{N_STABILITY_TESTS}:")

        test_auc, test_preds, cat_mean, mlp_mean = run_stability_test(
            X, y, n_bags=N_BAGS, test_seed=test_idx
        )

        all_aucs.append(test_auc)
        all_cat_aucs.append(cat_mean)
        all_mlp_aucs.append(mlp_mean)

        elapsed_pct = (test_idx + 1) / N_STABILITY_TESTS
        eta_min = (1 - elapsed_pct) / elapsed_pct * (test_idx + 1) * 1.3 if elapsed_pct > 0 else 0

        print(f"    => Ensemble AUC: {test_auc:.4f} | CatBoost: {cat_mean:.4f} | MLP: {mlp_mean:.4f} | ETA: {eta_min:.1f}min")

    # Step 4: Final Model
    print("\n" + "=" * 70)
    print("STEP 4: FINAL P.A.R.S.E.C. MODEL")
    print("=" * 70)

    print("  Training final ensemble...")
    final_auc, final_preds, final_cat, final_mlp = run_stability_test(
        X, y, n_bags=N_BAGS, test_seed=999
    )

    final_brier = brier_score_loss(y, final_preds)

    print(f"\n  Final P.A.R.S.E.C. AUC: {final_auc:.4f}")
    print(f"  Final Brier Score: {final_brier:.4f}")

    # Step 5: Bootstrap CI
    print("\n" + "=" * 70)
    print("STEP 5: BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 70)

    print("  Calculating Bootstrap 95% CI (n=1000)...")
    ci_lower, ci_upper, bootstrap_aucs = calculate_bootstrap_ci(y, final_preds, n_iterations=1000)
    ci_width = ci_upper - ci_lower

    print(f"  Bootstrap Mean: {np.mean(bootstrap_aucs):.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI Width: {ci_width:.4f}")

    # Results Summary
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_cat = np.mean(all_cat_aucs)
    mean_mlp = np.mean(all_mlp_aucs)

    # Determine verdict
    if mean_auc >= 0.90 and std_auc < 0.005:
        verdict = "TARGET ACHIEVED!"
        status = "SUCCESS"
    elif mean_auc >= 0.88 and std_auc < 0.01:
        verdict = "GOOD AND STABLE"
        status = "GOOD"
    elif std_auc < 0.01:
        verdict = "STABLE BUT LOW AUC"
        status = "WARNING"
    else:
        verdict = "NEEDS IMPROVEMENT"
        status = "FAIL"

    # Save report
    report = {
        "project": "P.A.R.S.E.C.",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": str(V43_CSV),
            "n_samples": len(y),
            "n_features_raw": len(feature_cols),
            "n_pca_components": N_PCA_COMPONENTS,
            "mlp_architecture": f"{N_PCA_COMPONENTS}->{MLP_HIDDEN_1}->{MLP_HIDDEN_2}->1",
            "mlp_dropout": MLP_DROPOUT,
            "mlp_l2_decay": MLP_L2_DECAY,
            "ensemble_weights": {"catboost": CATBOOST_WEIGHT, "mlp": MLP_WEIGHT},
            "n_bags": N_BAGS,
            "n_stability_tests": N_STABILITY_TESTS,
        },
        "pca": {
            "variance_explained": pca.explained_variance_ratio_[:N_PCA_COMPONENTS].tolist(),
            "cumulative_variance": float(sum(pca.explained_variance_ratio_[:N_PCA_COMPONENTS])),
        },
        "final_model": {
            "auc": float(final_auc),
            "brier": float(final_brier),
            "bootstrap_ci_lower": float(ci_lower),
            "bootstrap_ci_upper": float(ci_upper),
            "bootstrap_ci_width": float(ci_width),
        },
        "stability": {
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
            "min_auc": float(min(all_aucs)),
            "max_auc": float(max(all_aucs)),
            "all_aucs": [float(a) for a in all_aucs],
        },
        "components": {
            "catboost_mean_auc": float(mean_cat),
            "mlp_mean_auc": float(mean_mlp),
        },
        "verdict": verdict,
        "status": status,
    }

    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {OUTPUT_REPORT}")

    # Final Display
    print(f"""

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    PROJECT P.A.R.S.E.C. - FINAL RESULTS                           ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  Configuration:                                                              ║
    ║    Dataset:     V43 Expanded (N={len(y)})                                       ║
    ║    Raw Features: {len(feature_cols)} -> PCA: {N_PCA_COMPONENTS} components                                ║
    ║    MLP:         {N_PCA_COMPONENTS} -> {MLP_HIDDEN_1} -> {MLP_HIDDEN_2} -> 1 (Dropout={MLP_DROPOUT}, L2={MLP_L2_DECAY})                  ║
    ║    Ensemble:    {int(CATBOOST_WEIGHT*100)}% CatBoost + {int(MLP_WEIGHT*100)}% Deep MLP                               ║
    ║    Bags:        {N_BAGS} seeds averaged                                          ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  FINAL MODEL:                                                                ║
    ║                                                                              ║
    ║    Ensemble AUC:    {final_auc:.4f}                                                    ║
    ║    Bootstrap 95%:   [{ci_lower:.4f}, {ci_upper:.4f}] (width: {ci_width:.4f})                     ║
    ║    Brier Score:     {final_brier:.4f}                                                    ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  STABILITY ({N_STABILITY_TESTS} runs):                                                        ║
    ║                                                                              ║
    ║    Mean AUC:        {mean_auc:.4f} +/- {std_auc:.4f}                                       ║
    ║    Range:           [{min(all_aucs):.4f}, {max(all_aucs):.4f}]                                       ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  COMPONENT BREAKDOWN:                                                        ║
    ║                                                                              ║
    ║    CatBoost (raw):  {mean_cat:.4f} avg AUC                                          ║
    ║    Deep MLP (PCA):  {mean_mlp:.4f} avg AUC                                          ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  VERDICT: {verdict:30s}                               ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Target check
    if mean_auc >= 0.90:
        print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║    TARGET ACHIEVED! AUC >= 0.90!                                             ║
    ║                                                                              ║
    ║    Project P.A.R.S.E.C. has awakened!                                             ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"""
    Next steps to reach 0.90:
    1. Tune MLP hyperparameters (more layers, different dropout)
    2. Try different PCA component counts (3, 7, 10)
    3. Adjust ensemble weights
    4. Feature engineering on raw features
        """)


if __name__ == "__main__":
    main()
