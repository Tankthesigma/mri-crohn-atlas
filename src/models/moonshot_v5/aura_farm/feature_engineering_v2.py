#!/usr/bin/env python3
"""
AURA FARM Phase 3: Feature Engineering V2
==========================================

Create new leak-free interaction features to boost signal.

Usage:
    python feature_engineering_v2.py

Output:
    data/v33_aura_features.csv

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = 'data/v32_with_interactions.csv'
OUTPUT_PATH = 'data/v33_aura_features.csv'

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """Create new leak-free interaction and polynomial features."""

    # FIX: Validate required columns exist
    REQUIRED_COLS = [
        'cat_Biologic', 'cat_Surgical', 'cat_Combination', 'cat_Stem_Cell',
        'is_seton', 'is_refractory', 'is_rct', 'previous_biologic_failure',
        'fistula_complexity_Simple', 'fistula_complexity_Mixed',
        'fistula_complexity_Complex', 'followup_weeks', 'combo_therapy'
    ]
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Engineering new features...")

    # 1. Treatment Combinations (leak-free)
    df['bio_x_surgical'] = df['cat_Biologic'] * df['cat_Surgical']
    df['bio_x_combo'] = df['cat_Biologic'] * df['cat_Combination']
    df['surgical_x_combo'] = df['cat_Surgical'] * df['cat_Combination']
    df['stem_x_bio'] = df['cat_Stem_Cell'] * df['cat_Biologic']
    df['seton_x_bio'] = df['is_seton'] * df['cat_Biologic']
    df['seton_x_surgical'] = df['is_seton'] * df['cat_Surgical']

    # 2. Severity Interactions
    df['complex_x_refractory'] = df['fistula_complexity_Complex'] * df['is_refractory']
    df['complex_x_bio_failure'] = df['fistula_complexity_Complex'] * df['previous_biologic_failure']
    df['refractory_x_bio_failure'] = df['is_refractory'] * df['previous_biologic_failure']

    # 3. Study Quality × Patient Factors
    df['rct_x_refractory'] = df['is_rct'] * df['is_refractory']
    df['rct_x_complex'] = df['is_rct'] * df['fistula_complexity_Complex']
    df['rct_x_bio_failure'] = df['is_rct'] * df['previous_biologic_failure']

    # 4. Followup Interactions (binned to avoid continuous × continuous issues)
    df['followup_short'] = (df['followup_weeks'] < 26).astype(int)
    df['followup_medium'] = ((df['followup_weeks'] >= 26) & (df['followup_weeks'] < 52)).astype(int)
    df['followup_long'] = (df['followup_weeks'] >= 52).astype(int)

    df['short_x_complex'] = df['followup_short'] * df['fistula_complexity_Complex']
    df['long_x_bio'] = df['followup_long'] * df['cat_Biologic']
    df['long_x_surgical'] = df['followup_long'] * df['cat_Surgical']

    # 5. Complexity Score (ordinal encoding)
    df['complexity_score'] = (
        1 * df['fistula_complexity_Simple'] +
        2 * df['fistula_complexity_Mixed'] +
        3 * df['fistula_complexity_Complex']
    )
    df['complexity_x_refractory'] = df['complexity_score'] * df['is_refractory']

    # 6. Treatment Intensity Score
    df['treatment_intensity'] = (
        df['cat_Biologic'] +
        df['cat_Surgical'] +
        df['cat_Combination'] +
        df['cat_Stem_Cell'] +
        df['combo_therapy']
    )

    # 7. Risk Score (patient severity)
    df['risk_score'] = (
        df['is_refractory'] +
        df['previous_biologic_failure'] +
        df['fistula_complexity_Complex'] +
        df['fistula_complexity_Mixed'] * 0.5
    )
    df['risk_x_intensity'] = df['risk_score'] * df['treatment_intensity']

    # 8. Triple Interactions (high-order)
    df['complex_refractory_bio'] = (
        df['fistula_complexity_Complex'] *
        df['is_refractory'] *
        df['cat_Biologic']
    )
    df['complex_refractory_surgical'] = (
        df['fistula_complexity_Complex'] *
        df['is_refractory'] *
        df['cat_Surgical']
    )

    # 9. Ratio Features (with safety for division)
    df['followup_per_intensity'] = df['followup_weeks'] / (df['treatment_intensity'] + 1)
    df['risk_adjusted_followup'] = df['followup_weeks'] / (df['risk_score'] + 1)

    # 10. Polynomial Features (squared terms for key features)
    df['followup_squared'] = df['followup_weeks'] ** 2
    df['complexity_squared'] = df['complexity_score'] ** 2

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AURA FARM Phase 3: Feature Engineering V2")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, encoding='latin-1')
    print(f"  Original features: {len(df.columns)}")
    print(f"  Patients: {len(df):,}")

    # Engineer features
    df = engineer_features(df)

    # Count new features
    new_features = [
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

    print(f"\n  New features created: {len(new_features)}")
    print(f"  Total features: {len(df.columns)}")

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    # Print feature list for use in other scripts
    print("\n--- NEW LEAK-FREE FEATURES ---")
    print("Add these to your FEATURES list:")
    print()
    for f in new_features:
        print(f"    '{f}',")


if __name__ == '__main__':
    main()
