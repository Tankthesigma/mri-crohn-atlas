#!/usr/bin/env python3
"""
PARSEC CHIMERA API - Real ML Model Predictions
===============================================

Flask API that loads the EXACT trained Chimera model (AUC 0.908)
and serves predictions. NO FAKE MATH - REAL MODEL.

Usage:
    python api.py
    # Then visit http://localhost:5001/health

Endpoints:
    GET  /health  - Check if model is loaded
    POST /predict - Get prediction for treatment scenario
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# =============================================================================
# MLP CLASS (must match training definition EXACTLY for pickle loading)
# =============================================================================

class TheKrakenV2(nn.Module):
    """Deep MLP with BatchNorm - matches training architecture exactly."""
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

app = Flask(__name__)
CORS(app)  # Allow browser requests from any origin

# =============================================================================
# LOAD THE REAL CHIMERA MODEL
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "parsec_chimera_final.pkl"

print(f"Loading PARSEC Chimera model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as f:
        MODEL = pickle.load(f)
    print(f"  ✓ Model version: {MODEL.get('version', 'unknown')}")
    print(f"  ✓ AUC: {MODEL.get('auc', 'N/A'):.4f}")
    print(f"  ✓ Weights: CatBoost={MODEL['weights']['catboost']:.3f}, MLP={MODEL['weights']['mlp']:.3f}")
    MODEL_LOADED = True
except Exception as e:
    print(f"  ✗ ERROR loading model: {e}")
    MODEL = None
    MODEL_LOADED = False

# The 19 features the model expects (in exact order)
FEATURES = [
    'cat_Biologic', 'cat_Surgical', 'cat_Combination', 'cat_Stem_Cell',
    'cat_Antibiotic', 'cat_Other', 'n_total', 'followup_weeks',
    'is_refractory', 'is_rct', 'combo_therapy', 'oracle_vibe_score',
    'fistula_complexity_Simple', 'fistula_complexity_Mixed',
    'fistula_complexity_Complex', 'previous_biologic_failure',
    'stringency_score', 'confidence_score',
    'is_seton'  # Distinguishes palliative setons from curative surgeries
]


def predict_chimera(features_dict):
    """
    Run the full Chimera ensemble prediction.

    Pipeline:
    1. Extract core features
    2. Generate meta-features via XGBoost
    3. CatBoost prediction (raw + meta features)
    4. MLP prediction (scaled + PCA features)
    5. Weighted ensemble
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # Extract features in correct order
    X = np.array([[features_dict.get(f, 0) for f in FEATURES]], dtype=float)

    # 1. Generate meta-features using XGBoost
    xgb_pred = MODEL['xgb_meta'].predict_proba(X)[:, 1]

    cat_bio = features_dict.get('cat_Biologic', 0)
    n_total = features_dict.get('n_total', 50)

    meta_features = np.column_stack([
        xgb_pred,
        np.abs(xgb_pred - 0.5),
        xgb_pred * cat_bio,
        xgb_pred * np.log1p(n_total)
    ])

    # 2. CatBoost prediction (raw + meta)
    X_cat = np.hstack([X, meta_features])
    cat_prob = MODEL['catboost'].predict_proba(X_cat)[:, 1]

    # 3. MLP prediction (scaled + PCA)
    X_scaled = MODEL['scaler'].transform(X)
    X_pca = MODEL['pca'].transform(X_scaled)

    # Run MLP in eval mode
    mlp = MODEL['mlp']
    mlp.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pca)
        mlp_prob = mlp(X_tensor).squeeze().cpu().numpy()

    # Handle scalar vs array
    if isinstance(mlp_prob, np.ndarray) and mlp_prob.ndim == 0:
        mlp_prob = float(mlp_prob)

    # 4. Weighted ensemble
    w = MODEL['weights']
    ensemble_prob = w['catboost'] * cat_prob[0] + w['mlp'] * mlp_prob

    return {
        'probability': float(ensemble_prob),
        'catboost_prob': float(cat_prob[0]),
        'mlp_prob': float(mlp_prob),
        'xgb_meta_prob': float(xgb_pred[0])
    }


def map_clinical_to_features(clinical_input):
    """
    Map user-friendly clinical inputs to the 18 model features.

    Clinical inputs:
    - treatments: list of selected treatments
    - complexity: 'simple' or 'complex'
    - prior_failure: 'yes' or 'no'
    - inflammation: 'yes' or 'no'
    - collections: 'yes' or 'no'
    - fistula_type: 'intersphincteric', 'transsphincteric', etc.
    """
    features = {f: 0 for f in FEATURES}

    # Get treatments
    treatments = clinical_input.get('treatments', [])

    # Map treatment categories
    biologics = ['infliximab', 'adalimumab', 'vedolizumab', 'ustekinumab']
    surgicals = ['seton', 'fistulotomy', 'lift', 'flap', 'proctectomy']

    has_biologic = any(t in biologics for t in treatments)
    has_surgical = any(t in surgicals for t in treatments)
    has_stem_cell = 'stem_cell' in treatments
    has_antibiotic = 'antibiotics' in treatments
    has_immunomod = 'immunomod' in treatments

    # Detect seton specifically (palliative, not curative)
    has_seton = 'seton' in treatments
    # Seton ALONE is palliative - but when combined with biologic, the biologic does the healing
    seton_alone = has_seton and not has_biologic and not has_stem_cell

    features['cat_Biologic'] = 1 if has_biologic else 0
    features['cat_Surgical'] = 1 if has_surgical else 0
    features['cat_Stem_Cell'] = 1 if has_stem_cell else 0
    features['cat_Antibiotic'] = 1 if has_antibiotic else 0
    features['cat_Other'] = 1 if has_immunomod else 0
    features['is_seton'] = 1 if seton_alone else 0  # Only penalize seton-alone cases

    # Combination therapy
    if has_biologic and has_surgical:
        features['cat_Combination'] = 1
        features['combo_therapy'] = 1

    # Study defaults (simulating a typical RCT patient)
    features['n_total'] = 100
    features['followup_weeks'] = 52
    features['is_rct'] = 1
    features['stringency_score'] = 1.0
    features['confidence_score'] = 90

    # Patient factors
    prior_failure = clinical_input.get('prior_failure', 'no')
    if prior_failure == 'yes':
        features['is_refractory'] = 1
        features['previous_biologic_failure'] = 1

    # Inflammation affects vibe score
    inflammation = clinical_input.get('inflammation', 'yes')
    features['oracle_vibe_score'] = 70 if inflammation == 'yes' else 40

    # Complexity
    complexity = clinical_input.get('complexity', 'complex')
    if complexity == 'simple':
        features['fistula_complexity_Simple'] = 1
    elif complexity == 'mixed':
        features['fistula_complexity_Mixed'] = 1
    else:
        features['fistula_complexity_Complex'] = 1

    # Fistula type can override complexity
    fistula_type = clinical_input.get('fistula_type', '')
    if fistula_type in ['extrasphincteric', 'horseshoe']:
        features['fistula_complexity_Simple'] = 0
        features['fistula_complexity_Mixed'] = 0
        features['fistula_complexity_Complex'] = 1
    elif fistula_type == 'intersphincteric':
        features['fistula_complexity_Simple'] = 1
        features['fistula_complexity_Mixed'] = 0
        features['fistula_complexity_Complex'] = 0

    # Collections reduce confidence
    if clinical_input.get('collections', 'no') == 'yes':
        features['confidence_score'] = 70

    return features


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict treatment effectiveness using the REAL Chimera model.

    POST body (clinical format):
    {
        "treatments": ["infliximab", "seton"],
        "complexity": "complex",
        "prior_failure": "yes",
        "inflammation": "yes",
        "collections": "no",
        "fistula_type": "transsphincteric"
    }

    OR raw features format (18 features directly).
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded', 'offline': True}), 500

    try:
        data = request.json
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Check if this is clinical format or raw features
        if 'treatments' in data:
            # Clinical format - map to features
            features_dict = map_clinical_to_features(data)
        else:
            # Raw features format
            features_dict = data

        result = predict_chimera(features_dict)
        result['model'] = 'PARSEC_CHIMERA'
        result['auc'] = MODEL.get('auc', 0.908)
        result['version'] = MODEL.get('version', 'v1')

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok' if MODEL_LOADED else 'error',
        'model_loaded': MODEL_LOADED,
        'model_version': MODEL.get('version', 'unknown') if MODEL else None,
        'auc': MODEL.get('auc', 0) if MODEL else None,
        'weights': MODEL.get('weights', {}) if MODEL else None,
        'features': FEATURES
    })


@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple scenarios."""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded', 'offline': True}), 500

    try:
        data = request.json
        scenarios = data.get('scenarios', [])

        results = []
        for scenario in scenarios:
            if 'treatments' in scenario:
                features_dict = map_clinical_to_features(scenario)
            else:
                features_dict = scenario
            result = predict_chimera(features_dict)
            results.append(result)

        return jsonify({
            'predictions': results,
            'count': len(results),
            'model': 'PARSEC_CHIMERA'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("⚡ PARSEC CHIMERA API - REAL ML MODEL")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Status: {'✓ LOADED' if MODEL_LOADED else '✗ FAILED'}")
    if MODEL_LOADED:
        print(f"  AUC: {MODEL.get('auc', 'N/A'):.4f}")
        print(f"  Weights: CatBoost {MODEL['weights']['catboost']:.1%}, MLP {MODEL['weights']['mlp']:.1%}")
    print("=" * 60)
    print("\n  Starting server on http://localhost:5001")
    print("  Endpoints:")
    print("    GET  /health  - Check status")
    print("    POST /predict - Single prediction")
    print("    POST /batch   - Batch predictions")
    print("\n  Press Ctrl+C to stop\n")

    app.run(port=5001, debug=False)
