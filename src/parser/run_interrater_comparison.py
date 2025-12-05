#!/usr/bin/env python3
"""
Inter-Rater Comparison Analysis for MRI Report Parser
Compares parser performance against published expert reliability data

Part of MRI-Crohn Atlas ISEF 2026 Project
Phase 3: Compare parser accuracy to human inter-rater reliability
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Published inter-rater reliability data from literature
PUBLISHED_RELIABILITY = {
    "van_assche_original": {
        "source": "Hindryckx et al. 2017 (APT)",
        "n_raters": 4,
        "n_cases": 50,
        "icc_inter": 0.68,
        "icc_inter_ci": (0.56, 0.77),
        "icc_intra": 0.81,
        "notes": "Four experienced radiologists, 3 scoring occasions"
    },
    "van_assche_modified": {
        "source": "De Gregorio et al. 2020",
        "n_raters": 2,
        "n_cases": 53,
        "icc_inter": 0.67,
        "icc_inter_ci": (0.55, 0.75),
        "icc_intra": 0.81,
        "notes": "Modified 7-item version"
    },
    "magnifi_cd_original": {
        "source": "Spinelli et al. 2019 (Gastroenterology)",
        "n_raters": 4,
        "n_cases": 160,
        "icc_inter": 0.74,
        "icc_inter_ci": (0.63, 0.80),
        "icc_intra": 0.85,
        "notes": "ADMIRE-CD validation cohort"
    },
    "magnifi_cd_external": {
        "source": "Beek et al. 2024 (Eur Radiol)",
        "n_raters": 2,
        "n_cases": 65,
        "icc_inter": 0.87,
        "icc_inter_ci": (0.80, 0.92),
        "icc_intra": 0.88,
        "notes": "External validation, complex pfCD"
    },
    "esgar_2023": {
        "source": "ESGAR 2023 Conference",
        "n_raters": 3,
        "n_cases": 67,
        "icc_inter": 0.88,
        "icc_inter_ci": (0.78, 0.92),
        "icc_intra": None,
        "notes": "Multi-center European study"
    }
}

# Expert variance estimates (based on published ICC and score ranges)
# For VAI (0-22): typical SD between raters ~2-3 points
# For MAGNIFI-CD (0-25): typical SD between raters ~2-4 points
EXPERT_VARIANCE = {
    "vai": {
        "mean_inter_rater_sd": 2.5,
        "range": (1.5, 3.5),
        "max_score": 22
    },
    "magnifi": {
        "mean_inter_rater_sd": 3.0,
        "range": (2.0, 4.0),
        "max_score": 25
    }
}


def load_all_validation_results():
    """Load results from all validation runs"""
    project_root = Path(__file__).parent.parent.parent

    results = {
        "edge_cases": None,
        "expanded": None,
        "original": None
    }

    # Edge case results
    edge_path = project_root / "data" / "parser_tests" / "edge_case_results.json"
    if edge_path.exists():
        with open(edge_path, 'r', encoding='utf-8') as f:
            results["edge_cases"] = json.load(f)

    # Expanded validation results
    expanded_path = project_root / "data" / "parser_tests" / "expanded_validation_results.json"
    if expanded_path.exists():
        with open(expanded_path, 'r', encoding='utf-8') as f:
            results["expanded"] = json.load(f)

    # Original validation results
    original_path = project_root / "data" / "parser_tests" / "validation_results.json"
    if original_path.exists():
        with open(original_path, 'r', encoding='utf-8') as f:
            results["original"] = json.load(f)

    return results


def calculate_parser_icc(results_data):
    """
    Calculate ICC-like metric for parser vs ground truth

    Uses ICC(3,1) approximation: single measurement, fixed raters
    """
    vai_pairs = []
    magnifi_pairs = []

    for result_set in results_data.values():
        if result_set is None:
            continue

        for r in result_set.get("results", []):
            if "vai_extracted" in r and "vai_expected" in r:
                vai_pairs.append((r["vai_extracted"], r["vai_expected"]))
            if "magnifi_extracted" in r and "magnifi_expected" in r:
                magnifi_pairs.append((r["magnifi_extracted"], r["magnifi_expected"]))

    def compute_icc(pairs):
        """Compute ICC(3,1) for parser vs reference"""
        if len(pairs) < 3:
            return None, None

        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])

        # Create data matrix for ICC
        data = np.column_stack([x, y])

        n = len(pairs)
        k = 2  # Two "raters": parser and ground truth

        # Calculate variance components
        grand_mean = np.mean(data)
        row_means = np.mean(data, axis=1)
        col_means = np.mean(data, axis=0)

        # Between-subjects variance (MS_S)
        ss_subjects = k * np.sum((row_means - grand_mean) ** 2)
        ms_subjects = ss_subjects / (n - 1)

        # Residual variance (MS_E)
        ss_error = np.sum((data - row_means.reshape(-1, 1) - col_means + grand_mean) ** 2)
        ms_error = ss_error / ((n - 1) * (k - 1))

        # ICC(3,1) = (MS_S - MS_E) / (MS_S + (k-1)*MS_E)
        icc = (ms_subjects - ms_error) / (ms_subjects + (k - 1) * ms_error)

        # Pearson correlation as additional metric
        corr, _ = stats.pearsonr(x, y)

        return max(0, icc), corr

    vai_icc, vai_corr = compute_icc(vai_pairs)
    magnifi_icc, magnifi_corr = compute_icc(magnifi_pairs)

    return {
        "vai": {"icc": vai_icc, "pearson_r": vai_corr, "n_pairs": len(vai_pairs)},
        "magnifi": {"icc": magnifi_icc, "pearson_r": magnifi_corr, "n_pairs": len(magnifi_pairs)}
    }


def calculate_error_metrics(results_data):
    """Calculate detailed error metrics"""
    vai_errors = []
    magnifi_errors = []

    for result_set in results_data.values():
        if result_set is None:
            continue

        for r in result_set.get("results", []):
            if "vai_error" in r:
                vai_errors.append(r["vai_error"])
            if "magnifi_error" in r:
                magnifi_errors.append(r["magnifi_error"])

    def compute_metrics(errors):
        if not errors:
            return None
        arr = np.array(errors)
        return {
            "mae": float(np.mean(arr)),
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "within_2": float(np.mean(arr <= 2)),
            "within_3": float(np.mean(arr <= 3)),
            "within_5": float(np.mean(arr <= 5)),
            "n": len(errors)
        }

    return {
        "vai": compute_metrics(vai_errors),
        "magnifi": compute_metrics(magnifi_errors)
    }


def compare_to_experts():
    """Compare parser performance to published expert reliability"""
    results_data = load_all_validation_results()

    parser_icc = calculate_parser_icc(results_data)
    error_metrics = calculate_error_metrics(results_data)

    print("\n" + "=" * 70)
    print("INTER-RATER COMPARISON ANALYSIS (Phase 3)")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("PUBLISHED EXPERT INTER-RATER RELIABILITY")
    print("-" * 50)

    for name, data in PUBLISHED_RELIABILITY.items():
        print(f"\n{name.upper()}")
        print(f"  Source: {data['source']}")
        print(f"  N={data['n_cases']} cases, {data['n_raters']} raters")
        ci_low, ci_high = data['icc_inter_ci']
        print(f"  Inter-rater ICC: {data['icc_inter']:.2f} (95% CI: {ci_low:.2f}-{ci_high:.2f})")
        if data['icc_intra']:
            print(f"  Intra-rater ICC: {data['icc_intra']:.2f}")

    print("\n" + "-" * 50)
    print("PARSER PERFORMANCE vs GROUND TRUTH")
    print("-" * 50)

    print(f"\n  VAI:")
    if parser_icc["vai"]["icc"] is not None:
        print(f"    ICC (parser vs GT): {parser_icc['vai']['icc']:.2f}")
        print(f"    Pearson r: {parser_icc['vai']['pearson_r']:.2f}")
        print(f"    N pairs: {parser_icc['vai']['n_pairs']}")

    if error_metrics["vai"]:
        print(f"    MAE: {error_metrics['vai']['mae']:.2f}")
        print(f"    RMSE: {error_metrics['vai']['rmse']:.2f}")
        print(f"    Within 2 pts: {error_metrics['vai']['within_2']*100:.1f}%")
        print(f"    Within 3 pts: {error_metrics['vai']['within_3']*100:.1f}%")

    print(f"\n  MAGNIFI-CD:")
    if parser_icc["magnifi"]["icc"] is not None:
        print(f"    ICC (parser vs GT): {parser_icc['magnifi']['icc']:.2f}")
        print(f"    Pearson r: {parser_icc['magnifi']['pearson_r']:.2f}")
        print(f"    N pairs: {parser_icc['magnifi']['n_pairs']}")

    if error_metrics["magnifi"]:
        print(f"    MAE: {error_metrics['magnifi']['mae']:.2f}")
        print(f"    RMSE: {error_metrics['magnifi']['rmse']:.2f}")
        print(f"    Within 2 pts: {error_metrics['magnifi']['within_2']*100:.1f}%")
        print(f"    Within 3 pts: {error_metrics['magnifi']['within_3']*100:.1f}%")

    # Interpretation
    print("\n" + "-" * 50)
    print("INTERPRETATION")
    print("-" * 50)

    vai_icc = parser_icc["vai"]["icc"] or 0
    magnifi_icc = parser_icc["magnifi"]["icc"] or 0

    # Compare to expert ICC ranges
    expert_vai_icc = PUBLISHED_RELIABILITY["van_assche_original"]["icc_inter"]
    expert_magnifi_icc = PUBLISHED_RELIABILITY["magnifi_cd_external"]["icc_inter"]

    print(f"\n  VAI Performance:")
    if vai_icc >= expert_vai_icc:
        print(f"    Parser ICC ({vai_icc:.2f}) >= Expert ICC ({expert_vai_icc:.2f})")
        print(f"    STATUS: MEETS OR EXCEEDS EXPERT RELIABILITY")
    elif vai_icc >= expert_vai_icc - 0.10:
        print(f"    Parser ICC ({vai_icc:.2f}) within 0.10 of Expert ICC ({expert_vai_icc:.2f})")
        print(f"    STATUS: APPROACHING EXPERT RELIABILITY")
    else:
        print(f"    Parser ICC ({vai_icc:.2f}) < Expert ICC ({expert_vai_icc:.2f})")
        print(f"    STATUS: BELOW EXPERT RELIABILITY (gap: {expert_vai_icc - vai_icc:.2f})")

    print(f"\n  MAGNIFI-CD Performance:")
    if magnifi_icc >= expert_magnifi_icc:
        print(f"    Parser ICC ({magnifi_icc:.2f}) >= Expert ICC ({expert_magnifi_icc:.2f})")
        print(f"    STATUS: MEETS OR EXCEEDS EXPERT RELIABILITY")
    elif magnifi_icc >= expert_magnifi_icc - 0.10:
        print(f"    Parser ICC ({magnifi_icc:.2f}) within 0.10 of Expert ICC ({expert_magnifi_icc:.2f})")
        print(f"    STATUS: APPROACHING EXPERT RELIABILITY")
    else:
        print(f"    Parser ICC ({magnifi_icc:.2f}) < Expert ICC ({expert_magnifi_icc:.2f})")
        print(f"    STATUS: BELOW EXPERT RELIABILITY (gap: {expert_magnifi_icc - magnifi_icc:.2f})")

    # Context about expected human disagreement
    print("\n" + "-" * 50)
    print("EXPECTED HUMAN VARIATION (from literature)")
    print("-" * 50)
    print(f"\n  For VAI (range 0-22):")
    print(f"    Typical inter-rater SD: {EXPERT_VARIANCE['vai']['mean_inter_rater_sd']} points")
    print(f"    Major disagreement sources: fistula count, extension, rectal wall")
    print(f"\n  For MAGNIFI-CD (range 0-25):")
    print(f"    Typical inter-rater SD: {EXPERT_VARIANCE['magnifi']['mean_inter_rater_sd']} points")
    print(f"    Major disagreement sources: fistula complexity, dominant feature")

    # Save results
    output_path = Path(__file__).parent.parent.parent / "data" / "parser_tests" / "interrater_comparison.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "parser_metrics": {
            "vai": {
                "icc": parser_icc["vai"]["icc"],
                "pearson_r": parser_icc["vai"]["pearson_r"],
                "mae": error_metrics["vai"]["mae"] if error_metrics["vai"] else None,
                "rmse": error_metrics["vai"]["rmse"] if error_metrics["vai"] else None,
                "within_3_pct": error_metrics["vai"]["within_3"] if error_metrics["vai"] else None,
                "n_cases": parser_icc["vai"]["n_pairs"]
            },
            "magnifi": {
                "icc": parser_icc["magnifi"]["icc"],
                "pearson_r": parser_icc["magnifi"]["pearson_r"],
                "mae": error_metrics["magnifi"]["mae"] if error_metrics["magnifi"] else None,
                "rmse": error_metrics["magnifi"]["rmse"] if error_metrics["magnifi"] else None,
                "within_3_pct": error_metrics["magnifi"]["within_3"] if error_metrics["magnifi"] else None,
                "n_cases": parser_icc["magnifi"]["n_pairs"]
            }
        },
        "published_expert_reliability": PUBLISHED_RELIABILITY,
        "comparison": {
            "vai_vs_expert": {
                "parser_icc": parser_icc["vai"]["icc"],
                "expert_icc": expert_vai_icc,
                "difference": vai_icc - expert_vai_icc if vai_icc else None,
                "status": "meets_or_exceeds" if vai_icc >= expert_vai_icc else (
                    "approaching" if vai_icc >= expert_vai_icc - 0.10 else "below"
                )
            },
            "magnifi_vs_expert": {
                "parser_icc": parser_icc["magnifi"]["icc"],
                "expert_icc": expert_magnifi_icc,
                "difference": magnifi_icc - expert_magnifi_icc if magnifi_icc else None,
                "status": "meets_or_exceeds" if magnifi_icc >= expert_magnifi_icc else (
                    "approaching" if magnifi_icc >= expert_magnifi_icc - 0.10 else "below"
                )
            }
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    compare_to_experts()
