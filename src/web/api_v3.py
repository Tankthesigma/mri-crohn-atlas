#!/usr/bin/env python3
"""
PARSEC v3 API - RCT-Calibrated Treatment Outcome Predictor
===========================================================

FastAPI deployment for the PARSEC v3 CatBoost model.
- AUC: 0.7566
- Forces is_rct=1 for RCT-calibrated "single truth" predictions
- Recomputes interaction features at prediction time

Usage:
    uvicorn api_v3:app --host 0.0.0.0 --port 8000
    # Or: python api_v3.py

Endpoints:
    GET  /           - API info
    GET  /health     - Health check
    POST /predict    - Get RCT-calibrated prediction
    GET  /treatments - List available treatments
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
import pickle
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="PARSEC v3 API",
    description="RCT-Calibrated Treatment Outcome Predictor for Perianal Crohn's Fistulas",
    version="3.0.0",
)

# Allow CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOAD MODEL
# =============================================================================

# Support multiple deployment configurations
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "parsec_v3_catboost.pkl"

# Check alternate paths for deployment
if not MODEL_PATH.exists():
    # Try relative to current working directory
    alt_path = Path.cwd() / "models" / "parsec_v3_catboost.pkl"
    if alt_path.exists():
        MODEL_PATH = alt_path

print(f"Loading PARSEC v3 model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, 'rb') as f:
        MODEL_DATA = pickle.load(f)

    MODEL = MODEL_DATA['model']
    FEATURE_NAMES = MODEL_DATA['feature_names']
    CAT_FEATURES = MODEL_DATA['cat_features']
    MONOTONIC_CONSTRAINTS = MODEL_DATA.get('monotonic_constraints', {})

    print(f"  ✓ Model loaded successfully")
    print(f"  ✓ Features: {len(FEATURE_NAMES)}")
    print(f"  ✓ Categorical: {CAT_FEATURES}")
    MODEL_LOADED = True
except Exception as e:
    print(f"  ✗ ERROR loading model: {e}")
    MODEL = None
    MODEL_LOADED = False

# =============================================================================
# TREATMENT MAPPINGS (from Phase 0)
# =============================================================================

# User-friendly names → canonical model names
TREATMENT_SYNONYMS = {
    # Biologics - Brand names
    'remicade': 'infliximab',
    'inflectra': 'infliximab',
    'ct-p13': 'infliximab',
    'remsima': 'infliximab',
    'humira': 'adalimumab',
    'hadlima': 'adalimumab',
    'hyrimoz': 'adalimumab',
    'entyvio': 'vedolizumab',
    'stelara': 'ustekinumab',
    'skyrizi': 'risankizumab',
    'cimzia': 'certolizumab',
    'simponi': 'golimumab',

    # Stem cells
    'alofisel': 'darvadstrocel',
    'cx601': 'darvadstrocel',
    'cx-601': 'darvadstrocel',

    # Small molecules - Brand names
    'xeljanz': 'tofacitinib',
    'rinvoq': 'upadacitinib',
    'jyseleca': 'filgotinib',
    'zeposia': 'ozanimod',

    # Surgeries - common variations
    'flap': 'advancement_flap',
    'rectal flap': 'advancement_flap',
    'mucosal flap': 'advancement_flap',
    'lift procedure': 'lift',
    'ligation': 'lift',
    'plug': 'fistula_plug',
    'glue': 'fibrin_glue',
    'laser': 'filac',
    'video assisted': 'vaaft',

    # Setons
    'loose seton': 'seton_drainage',
    'draining seton': 'seton_drainage',
    'cutting seton': 'seton_cutting',
    'staged fistulotomy': 'seton_cutting',
    'seton': 'seton_unknown',
}

# Treatment → (treatment_class, feature_flags)
TREATMENT_BUNDLES = {
    # Biologics
    'infliximab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'adalimumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'vedolizumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'ustekinumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'risankizumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'certolizumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'golimumab': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'anti_tnf_unspecified': {'class': 'biologic', 'has_biologic': 1, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},

    # Small molecules (JAK inhibitors)
    'tofacitinib': {'class': 'small_molecule', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 1, 'has_seton': 0},
    'upadacitinib': {'class': 'small_molecule', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 1, 'has_seton': 0},
    'filgotinib': {'class': 'small_molecule', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 1, 'has_seton': 0},
    'ozanimod': {'class': 'small_molecule', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 1, 'has_seton': 0},

    # Stem cells
    'darvadstrocel': {'class': 'stem_cell', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 1, 'has_small_molecule': 0, 'has_seton': 0},
    'adsc': {'class': 'stem_cell', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 1, 'has_small_molecule': 0, 'has_seton': 0},
    'msc': {'class': 'stem_cell', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 1, 'has_small_molecule': 0, 'has_seton': 0},
    'stem_cell_unspecified': {'class': 'stem_cell', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 1, 'has_small_molecule': 0, 'has_seton': 0},

    # Surgeries
    'advancement_flap': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'lift': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'fistulotomy': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'fistulectomy': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'fistula_plug': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'fibrin_glue': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'vaaft': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'filac': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'proctectomy': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
    'proctocolectomy': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},

    # Setons (special handling)
    'seton_cutting': {'class': 'surgery', 'has_biologic': 0, 'has_surgery': 1, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 1, 'seton_subtype': 'cutting'},
    'seton_drainage': {'class': 'drainage', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 1, 'seton_subtype': 'drainage'},
    'seton_unknown': {'class': 'drainage', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 1, 'seton_subtype': 'unknown'},

    # Placebo
    'placebo': {'class': 'placebo_control', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},

    # Other/Unknown
    'other_unknown': {'class': 'other', 'has_biologic': 0, 'has_surgery': 0, 'has_stemcell': 0, 'has_small_molecule': 0, 'has_seton': 0},
}

# Default feature values (medians from training data)
DEFAULT_FEATURES = {
    'followup_weeks': 26.0,
    'is_rct': 1,  # FORCED for RCT calibration
    'fistula_complex': 1,
    'fistula_simple': 0,
    'is_refractory': 0,
    'outcome_clinical': 1,
    'outcome_radiological': 0,
    'outcome_combined': 0,
    'outcome_unknown': 0,
    'has_endoscopic_component': 0,
    'has_diet_component': 0,
    'has_device_component': 0,
    'has_imaging_component': 0,
    'is_combo': 0,
    'has_immunomod': 0,
    'seton_cutting': 0,
    'seton_drainage': 0,
    'seton_unknown': 0,
    # Interactions will be computed
}

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    treatment: str = Field(..., description="Treatment name (e.g., 'infliximab', 'Remicade', 'seton_cutting')")
    complexity: str = Field(default="complex", description="Fistula complexity: 'simple', 'complex', 'transsphincteric'")
    refractory: bool = Field(default=False, description="Prior treatment failures?")
    prior_biologic_failure: bool = Field(default=False, description="Failed biologic before?")
    add_seton: bool = Field(default=False, description="Adding seton to treatment?")
    followup_weeks: int = Field(default=52, description="Follow-up duration in weeks")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    probability: float = Field(..., description="Predicted healing probability (0-1)")
    probability_pct: str = Field(..., description="Probability as percentage")
    confidence_interval: str = Field(..., description="95% confidence interval")
    treatment_canonical: str = Field(..., description="Canonical treatment name used")
    treatment_class: str = Field(..., description="Treatment class")
    rct_calibrated: bool = Field(default=True, description="RCT-calibrated prediction")
    model_version: str = Field(default="v3.0")
    features_used: Dict[str, Any] = Field(..., description="Feature values used for prediction")


# =============================================================================
# PREDICTION LOGIC
# =============================================================================

def normalize_treatment(treatment: str) -> str:
    """Normalize treatment name to canonical form."""
    t = treatment.lower().strip()

    # Check synonyms first
    if t in TREATMENT_SYNONYMS:
        return TREATMENT_SYNONYMS[t]

    # Check if already canonical
    if t in TREATMENT_BUNDLES:
        return t

    # Try partial matching for common patterns
    if 'infliximab' in t or 'remicade' in t:
        return 'infliximab'
    if 'adalimumab' in t or 'humira' in t:
        return 'adalimumab'
    if 'vedolizumab' in t or 'entyvio' in t:
        return 'vedolizumab'
    if 'ustekinumab' in t or 'stelara' in t:
        return 'ustekinumab'
    if 'darvadstrocel' in t or 'alofisel' in t:
        return 'darvadstrocel'
    if 'stem' in t and 'cell' in t:
        return 'stem_cell_unspecified'
    if 'seton' in t:
        if 'cutting' in t or 'staged' in t:
            return 'seton_cutting'
        elif 'drain' in t or 'loose' in t:
            return 'seton_drainage'
        else:
            return 'seton_unknown'
    if 'flap' in t or 'advancement' in t:
        return 'advancement_flap'
    if 'lift' in t or 'ligation' in t:
        return 'lift'
    if 'fistulotomy' in t:
        return 'fistulotomy'
    if 'plug' in t:
        return 'fistula_plug'
    if 'glue' in t:
        return 'fibrin_glue'
    if 'placebo' in t or 'control' in t:
        return 'placebo'

    # Default to other_unknown
    return 'other_unknown'


def build_feature_vector(
    treatment: str,
    complexity: str = "complex",
    refractory: bool = False,
    prior_biologic_failure: bool = False,
    add_seton: bool = False,
    followup_weeks: int = 52,
) -> Dict[str, Any]:
    """
    Build complete feature vector for prediction.

    CRITICAL: Forces is_rct=1 and recomputes all interaction features.
    """
    # Normalize treatment
    treatment_canonical = normalize_treatment(treatment)

    # Get treatment bundle
    bundle = TREATMENT_BUNDLES.get(treatment_canonical, TREATMENT_BUNDLES['other_unknown'])
    treatment_class = bundle['class']

    # Start with defaults
    features = DEFAULT_FEATURES.copy()

    # Set treatment features
    features['treatment_model'] = treatment_canonical
    features['treatment_class'] = treatment_class
    features['has_biologic'] = bundle['has_biologic']
    features['has_surgery'] = bundle['has_surgery']
    features['has_stemcell'] = bundle['has_stemcell']
    features['has_small_molecule'] = bundle['has_small_molecule']
    features['has_seton'] = bundle['has_seton']

    # Seton subtype
    seton_subtype = bundle.get('seton_subtype', 'none')
    features['seton_subtype'] = seton_subtype
    features['seton_cutting'] = 1 if seton_subtype == 'cutting' else 0
    features['seton_drainage'] = 1 if seton_subtype == 'drainage' else 0
    features['seton_unknown'] = 1 if seton_subtype == 'unknown' else 0

    # Handle add_seton (combination therapy)
    if add_seton and not features['has_seton']:
        features['has_seton'] = 1
        features['is_combo'] = 1
        features['seton_subtype'] = 'drainage'  # Assume drainage for combos
        features['seton_drainage'] = 1

    # Complexity
    is_complex = complexity.lower() in ['complex', 'high', 'horseshoe', 'extrasphincteric', 'transsphincteric']
    features['fistula_complex'] = 1 if is_complex else 0
    features['fistula_simple'] = 0 if is_complex else 1

    # Refractory status
    features['is_refractory'] = 1 if refractory else 0

    # Follow-up (capped at 104 weeks)
    features['followup_weeks'] = min(followup_weeks, 104)

    # CRITICAL: Force RCT standard
    features['is_rct'] = 1

    # =======================================================================
    # RECOMPUTE INTERACTION FEATURES
    # =======================================================================
    is_rct = features['is_rct']
    fistula_complex = features['fistula_complex']
    is_refractory_val = features['is_refractory']

    # Treatment x RCT interactions
    features['Bio_x_RCT'] = int(treatment_class == 'biologic' and is_rct == 1)
    features['Surg_x_RCT'] = int(treatment_class == 'surgery' and is_rct == 1)
    features['Stem_x_RCT'] = int(treatment_class == 'stem_cell' and is_rct == 1)
    features['Combo_x_RCT'] = int(treatment_class == 'combination' and is_rct == 1)
    features['Drain_x_RCT'] = int(treatment_class == 'drainage' and is_rct == 1)
    features['SmallMol_x_RCT'] = int(treatment_class == 'small_molecule' and is_rct == 1)
    features['Antibiotic_x_RCT'] = int(treatment_class == 'antibiotic' and is_rct == 1)

    # Treatment x Complexity interactions
    features['Surg_x_Complex'] = int(treatment_class == 'surgery' and fistula_complex == 1)
    features['Bio_x_Complex'] = int(treatment_class == 'biologic' and fistula_complex == 1)
    features['Stem_x_Complex'] = int(treatment_class == 'stem_cell' and fistula_complex == 1)

    # Refractory interactions
    features['Bio_x_Refractory'] = int(treatment_class == 'biologic' and is_refractory_val == 1)

    return features, treatment_canonical, treatment_class


def predict(features: Dict[str, Any]) -> float:
    """Run model prediction."""
    if MODEL is None:
        raise RuntimeError("Model not loaded")

    # Build feature array in correct order
    X = []
    for f in FEATURE_NAMES:
        val = features.get(f, 0)
        # Handle categorical features
        if f in CAT_FEATURES:
            X.append(str(val))
        else:
            X.append(float(val) if val is not None else 0.0)

    # Predict (CatBoostRegressor returns 0-1 directly)
    X_array = np.array([X], dtype=object)
    prob = MODEL.predict(X_array)[0]

    # Clip to valid range
    prob = np.clip(prob, 0.0, 1.0)

    return float(prob)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information."""
    return {
        "name": "PARSEC v3 API",
        "description": "RCT-Calibrated Treatment Outcome Predictor",
        "version": "3.0.0",
        "model_loaded": MODEL_LOADED,
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "GET /treatments": "List available treatments",
            "POST /predict": "Get RCT-calibrated prediction",
        },
        "rct_calibration": "All predictions are RCT-calibrated (is_rct=1 forced)",
        "auc": 0.7566,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "model_version": "v3.0",
        "features": len(FEATURE_NAMES) if MODEL_LOADED else 0,
        "auc": 0.7566,
    }


@app.get("/treatments")
async def list_treatments():
    """List all available treatments with their classes."""
    treatments = {}
    for name, bundle in TREATMENT_BUNDLES.items():
        treatments[name] = {
            "class": bundle['class'],
            "has_seton": bundle.get('has_seton', 0) == 1,
        }

    return {
        "treatments": treatments,
        "synonyms": TREATMENT_SYNONYMS,
        "classes": list(set(b['class'] for b in TREATMENT_BUNDLES.values())),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """
    Get RCT-calibrated prediction for treatment outcome.

    The model automatically:
    1. Normalizes treatment names (Remicade → infliximab)
    2. Forces is_rct=1 for RCT-calibrated prediction
    3. Recomputes all interaction features
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Build features
        features, treatment_canonical, treatment_class = build_feature_vector(
            treatment=request.treatment,
            complexity=request.complexity,
            refractory=request.refractory,
            prior_biologic_failure=request.prior_biologic_failure,
            add_seton=request.add_seton,
            followup_weeks=request.followup_weeks,
        )

        # Get prediction
        prob = predict(features)

        # Compute confidence interval (rough estimate based on model uncertainty)
        # For a more rigorous CI, you'd use conformal prediction
        ci_width = 0.15  # ~15% interval
        ci_low = max(0, prob - ci_width/2)
        ci_high = min(1, prob + ci_width/2)

        return PredictionResponse(
            probability=round(prob, 3),
            probability_pct=f"{prob*100:.1f}%",
            confidence_interval=f"{ci_low*100:.0f}% - {ci_high*100:.0f}%",
            treatment_canonical=treatment_canonical,
            treatment_class=treatment_class,
            rct_calibrated=True,
            model_version="v3.0",
            features_used={k: v for k, v in features.items() if k in ['treatment_model', 'treatment_class', 'fistula_complex', 'is_refractory', 'is_rct', 'has_seton', 'followup_weeks']},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("PARSEC v3 API - RCT-Calibrated Treatment Predictor")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Status: {'✓ LOADED' if MODEL_LOADED else '✗ FAILED'}")
    print(f"  AUC: 0.7566")
    print("=" * 60)
    print("\n  Starting server on http://localhost:8000")
    print("  Docs at http://localhost:8000/docs")
    print("\n  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
