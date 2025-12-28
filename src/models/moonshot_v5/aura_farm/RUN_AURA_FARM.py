#!/usr/bin/env python3
"""
AURA FARM: Master Runner
=========================

Executes the complete AURA FARM pipeline:

Phase 1: CatBoost Optuna (500 trials) - ~1-2 hours
Phase 2: XGBoost Optuna (500 trials) - ~1-2 hours
Phase 3: Feature Engineering v2 - ~1 minute
Phase 4: LightGBM Optuna v2 (500 trials, new features) - ~1-2 hours
Phase 5: Triple Optuna Stack - ~5 minutes

Total estimated time: 4-7 hours (depends on GPU availability)

Usage:
    python RUN_AURA_FARM.py           # Run all phases
    python RUN_AURA_FARM.py --phase 3 # Run specific phase

Author: Tanmay + Claude Code
Date: December 2025
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================

PHASES = [
    ('Phase 1: CatBoost Optuna', 'catboost_optuna_nuclear.py'),
    ('Phase 2: XGBoost Optuna', 'xgboost_optuna_nuclear.py'),
    ('Phase 3: Feature Engineering', 'feature_engineering_v2.py'),
    ('Phase 4: LightGBM Optuna v2', 'lightgbm_optuna_v2.py'),
    ('Phase 5: Triple Optuna Stack', 'triple_optuna_stack.py'),
]

# =============================================================================
# MAIN
# =============================================================================

def run_phase(phase_idx, script_name, phase_name):
    """Run a single phase."""
    print("\n" + "=" * 70)
    print(f"  {phase_name}")
    print(f"  Script: {script_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False

    # Run the script from project root (so data/v32_with_interactions.csv works)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_root,
    )

    if result.returncode != 0:
        print(f"\nERROR: {phase_name} failed with code {result.returncode}")
        return False

    print(f"\n{phase_name} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description='AURA FARM Master Runner')
    parser.add_argument('--phase', type=int, help='Run specific phase (1-5)')
    parser.add_argument('--start-from', type=int, default=1, help='Start from phase N')
    args = parser.parse_args()

    print("=" * 70)
    print("  AURA FARM: Complete Hyperparameter Optimization Pipeline")
    print("=" * 70)
    print()
    print("Phases:")
    for i, (name, script) in enumerate(PHASES, 1):
        print(f"  {i}. {name}")
    print()
    print("Estimated total time: 4-7 hours")
    print("=" * 70)

    # Determine which phases to run
    if args.phase:
        phases_to_run = [(args.phase - 1, *PHASES[args.phase - 1])]
    else:
        phases_to_run = [(i, name, script) for i, (name, script) in enumerate(PHASES) if i + 1 >= args.start_from]

    start_time = datetime.now()
    print(f"\nStarting at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run phases
    for phase_idx, phase_name, script_name in phases_to_run:
        success = run_phase(phase_idx, script_name, phase_name)
        if not success:
            print("\nPipeline halted due to error.")
            sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("  AURA FARM COMPLETE")
    print("=" * 70)
    print(f"\nStarted:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    print("Results saved to models/")
    print("  - catboost_optuna_oof.npz")
    print("  - xgboost_optuna_oof.npz")
    print("  - lightgbm_optuna_v2_oof.npz")
    print("  - triple_optuna_final.npz")
    print()
    print("Check models/triple_optuna_report.json for final AUC")


if __name__ == '__main__':
    main()
