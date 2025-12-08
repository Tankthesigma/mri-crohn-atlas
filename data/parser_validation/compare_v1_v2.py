#!/usr/bin/env python3
"""
Compare V1 vs V2 parser validation results.
"""

import json
from pathlib import Path
import numpy as np

V1_FILE = Path(__file__).parent / "real_validation_results.json"
V2_FILE = Path(__file__).parent / "real_validation_results_v2.json"


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def calculate_metrics(results):
    valid = [r for r in results if r.get("predicted_vai") is not None and r.get("expected_vai") is not None]

    vai_errors = [r["predicted_vai"] - r["expected_vai"] for r in valid]
    mag_errors = [r["predicted_magnifi"] - r["expected_magnifi"] for r in valid if r.get("predicted_magnifi") is not None]

    vai_abs = [abs(e) for e in vai_errors]
    mag_abs = [abs(e) for e in mag_errors]

    metrics = {
        "n": len(valid),
        "vai_mae": sum(vai_abs) / len(vai_abs),
        "vai_rmse": np.sqrt(sum(e**2 for e in vai_errors) / len(vai_errors)),
        "vai_bias": sum(vai_errors) / len(vai_errors),
        "vai_acc_exact": sum(1 for e in vai_abs if e == 0) / len(vai_abs),
        "vai_acc_1": sum(1 for e in vai_abs if e <= 1) / len(vai_abs),
        "vai_acc_2": sum(1 for e in vai_abs if e <= 2) / len(vai_abs),
        "vai_acc_3": sum(1 for e in vai_abs if e <= 3) / len(vai_abs),
        "mag_mae": sum(mag_abs) / len(mag_abs) if mag_abs else 0,
        "mag_acc_3": sum(1 for e in mag_abs if e <= 3) / len(mag_abs) if mag_abs else 0,
        "mag_acc_5": sum(1 for e in mag_abs if e <= 5) / len(mag_abs) if mag_abs else 0,
    }

    # Subgroup by source
    for source in ['radiopaedia', 'edge_cases', 'synthetic_literature', 'pubmed_central']:
        src_results = [r for r in valid if source in r.get("source", "").lower()]
        if src_results:
            src_vai_err = [abs(r["predicted_vai"] - r["expected_vai"]) for r in src_results]
            metrics[f"{source}_acc_2"] = sum(1 for e in src_vai_err if e <= 2) / len(src_vai_err)
            metrics[f"{source}_n"] = len(src_results)

    # Subgroup by severity
    for sev in ['remission', 'mild', 'moderate', 'severe']:
        sev_results = [r for r in valid if r.get("severity", "").lower() == sev]
        if sev_results:
            sev_vai_err = [abs(r["predicted_vai"] - r["expected_vai"]) for r in sev_results]
            metrics[f"{sev}_acc_2"] = sum(1 for e in sev_vai_err if e <= 2) / len(sev_vai_err)
            metrics[f"{sev}_n"] = len(sev_results)

    return metrics


def main():
    print("=" * 80)
    print("V1 vs V2 PARSER COMPARISON")
    print("=" * 80)

    v1 = load_results(V1_FILE)
    v2 = load_results(V2_FILE)

    m1 = calculate_metrics(v1)
    m2 = calculate_metrics(v2)

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"{'Metric':<30} {'V1':<15} {'V2':<15} {'Change':<15}")
    print("-" * 75)

    for key in ['vai_mae', 'vai_rmse', 'vai_bias', 'vai_acc_exact', 'vai_acc_1', 'vai_acc_2', 'vai_acc_3']:
        v1_val = m1[key]
        v2_val = m2[key]
        if 'acc' in key:
            change = (v2_val - v1_val) * 100
            print(f"{key:<30} {v1_val*100:.1f}%{'':<10} {v2_val*100:.1f}%{'':<10} {change:+.1f}%")
        else:
            change = v2_val - v1_val
            print(f"{key:<30} {v1_val:.2f}{'':<12} {v2_val:.2f}{'':<12} {change:+.2f}")

    print("\n" + "=" * 80)
    print("MAGNIFI METRICS")
    print("=" * 80)
    for key in ['mag_mae', 'mag_acc_3', 'mag_acc_5']:
        v1_val = m1[key]
        v2_val = m2[key]
        if 'acc' in key:
            change = (v2_val - v1_val) * 100
            print(f"{key:<30} {v1_val*100:.1f}%{'':<10} {v2_val*100:.1f}%{'':<10} {change:+.1f}%")
        else:
            change = v2_val - v1_val
            print(f"{key:<30} {v1_val:.2f}{'':<12} {v2_val:.2f}{'':<12} {change:+.2f}")

    print("\n" + "=" * 80)
    print("BY SOURCE (VAI Accuracy ±2)")
    print("=" * 80)
    for source in ['radiopaedia', 'edge_cases', 'synthetic_literature', 'pubmed_central']:
        key_acc = f"{source}_acc_2"
        key_n = f"{source}_n"
        if key_acc in m1 and key_acc in m2:
            n = m1.get(key_n, 0)
            change = (m2[key_acc] - m1[key_acc]) * 100
            print(f"{source:<25} (n={n:2d}) V1: {m1[key_acc]*100:.1f}%  V2: {m2[key_acc]*100:.1f}%  Change: {change:+.1f}%")

    print("\n" + "=" * 80)
    print("BY SEVERITY (VAI Accuracy ±2)")
    print("=" * 80)
    for sev in ['remission', 'mild', 'moderate', 'severe']:
        key_acc = f"{sev}_acc_2"
        key_n = f"{sev}_n"
        if key_acc in m1 and key_acc in m2:
            n = m1.get(key_n, 0)
            change = (m2[key_acc] - m1[key_acc]) * 100
            print(f"{sev:<15} (n={n:2d}) V1: {m1[key_acc]*100:.1f}%  V2: {m2[key_acc]*100:.1f}%  Change: {change:+.1f}%")

    print("\n" + "=" * 80)
    print("CASE-BY-CASE CHANGES")
    print("=" * 80)

    # Find cases where V2 is better
    better = []
    worse = []

    for r1, r2 in zip(v1, v2):
        if r1["case_id"] == r2["case_id"]:
            e1 = abs(r1.get("vai_error", 0) or 0)
            e2 = abs(r2.get("vai_error", 0) or 0)
            if e2 < e1:
                better.append((r1["case_id"], e1, e2, r1.get("case_type")))
            elif e2 > e1:
                worse.append((r1["case_id"], e1, e2, r1.get("case_type")))

    print(f"\nCases where V2 improved: {len(better)}")
    for case_id, e1, e2, ctype in sorted(better, key=lambda x: x[1] - x[2], reverse=True)[:10]:
        print(f"  {case_id:<35} {e1} -> {e2} ({ctype})")

    print(f"\nCases where V2 regressed: {len(worse)}")
    for case_id, e1, e2, ctype in sorted(worse, key=lambda x: x[2] - x[1], reverse=True)[:10]:
        print(f"  {case_id:<35} {e1} -> {e2} ({ctype})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    vai_improvement = (m2["vai_acc_2"] - m1["vai_acc_2"]) * 100
    print(f"VAI Accuracy (±2): {m1['vai_acc_2']*100:.1f}% -> {m2['vai_acc_2']*100:.1f}% ({vai_improvement:+.1f}%)")
    print(f"VAI MAE: {m1['vai_mae']:.2f} -> {m2['vai_mae']:.2f} ({m2['vai_mae'] - m1['vai_mae']:+.2f})")

    if vai_improvement > 0:
        print("\nV2 is BETTER overall")
    else:
        print("\nV2 is WORSE overall")


if __name__ == "__main__":
    main()
