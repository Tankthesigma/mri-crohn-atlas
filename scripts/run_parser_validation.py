#!/usr/bin/env python3
"""
Run REAL parser validation on gold cases using the same logic as parser.html.

Replicates the exact API call and scoring functions from the web parser.
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths
GOLD_CASES_PATH = PROJECT_ROOT / "data" / "calibration" / "gold_cases.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "calibration"
RESULTS_PATH = OUTPUT_DIR / "parser_validation_results.json"
PROGRESS_PATH = OUTPUT_DIR / "parser_validation_progress.json"

# API Configuration
OPENROUTER_API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

# Rate limiting
DELAY_BETWEEN_CALLS = 1.0  # seconds


def get_parser_prompt(report_text: str) -> str:
    """Generate the exact prompt used by parser.html."""
    return f"""You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. AVOID OVERESTIMATING severity.

REPORT TEXT:
{report_text}

CRITICAL RULES:

=== AMBIGUITY HANDLING (PRIORITIZE SENSITIVITY) ===
If a finding is described as 'possible', 'suspected', 'cannot be excluded', 'equivocal', 'likely', 'may represent', or 'probable', you MUST treat it as PRESENT and extract the corresponding feature. Do NOT return null for ambiguous findings. Prioritize Sensitivity (catching disease) over Specificity.

=== TREATMENT AWARENESS ===
- "seton in place" / "seton placement" = ACTIVELY MANAGED disease (treatment_status = "seton_in_place")
- When seton present: T2 hyperintensity is EXPECTED, use "mild" unless explicitly "marked"

=== NEGATIVE FINDINGS ARE IMPORTANT ===
- "no abscess" / "no collection" = collections_abscesses = false
- "no wall thickening" = rectal_wall_involvement = false
- List these in negative_findings array

=== FISTULA COUNTING ===
- Primary tract + secondary tract from SAME origin = 1 fistula with branching, NOT 2 separate fistulas
- Only count as 2+ if they have SEPARATE internal openings at different clock positions

=== T2 HYPERINTENSITY ===
- Mark as TRUE if described: "hyperintense", "T2 bright", "high signal on T2", "increased signal", "inflammation"
- If "possible inflammation" or "may show hyperintensity" → treat as TRUE with degree "mild"
- If fistula is present and active, assume at least mild T2 hyperintensity unless explicitly stated otherwise

=== EXTENSION ===
- "moderate" = ischioanal fossa, typical transsphincteric (3-4cm tracts)
- "severe" = ONLY for supralevator, horseshoe pattern, extends to thigh/buttock

Return a JSON object with this EXACT structure:
{{
    "features": {{
        "fistula_count": {{ "value": <number or null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "fistula_type": {{ "value": <"intersphincteric"|"transsphincteric"|"suprasphincteric"|"extrasphincteric"|"complex"|"simple"|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "t2_hyperintensity": {{ "value": <true|false|null>, "degree": <"mild"|"moderate"|"marked"|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "extension": {{ "value": <"none"|"mild"|"moderate"|"severe"|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "collections_abscesses": {{ "value": <true|false|null>, "size": <"small"|"large"|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "rectal_wall_involvement": {{ "value": <true|false|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "inflammatory_mass": {{ "value": <true|false|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }},
        "predominant_feature": {{ "value": <"inflammatory"|"fibrotic"|"mixed"|null>, "confidence": <"high"|"medium"|"low">, "evidence": "<exact quote>" }}
    }},
    "treatment_status": <"seton_in_place"|"post_surgical"|"on_medication"|"no_treatment"|"unknown">,
    "negative_findings": [<list of explicit negative statements>],
    "overall_assessment": {{ "activity": <"active"|"healing"|"healed"|"chronic">, "severity": <"mild"|"moderate"|"severe"> }}
}}

IMPORTANT: Use null if a feature is not mentioned. Return ONLY valid JSON."""


def calculate_vai(features: Dict, treatment_status: str) -> int:
    """Calculate VAI score - exact copy of parser.html logic."""
    score = 0
    has_seton = treatment_status == "seton_in_place"

    # Fistula count
    count = features.get("fistula_count", {}).get("value")
    if count == 1:
        score += 1
    elif count is not None and count >= 2:
        score += 2

    # Fistula type
    ftype = features.get("fistula_type", {}).get("value")
    if ftype in ["simple", "intersphincteric"]:
        score += 1
    elif ftype in ["transsphincteric", "suprasphincteric", "extrasphincteric", "complex"]:
        score += 2

    # Extension
    ext = features.get("extension", {}).get("value")
    if ext == "mild":
        score += 2
    elif ext == "moderate":
        score += 3
    elif ext == "severe":
        score += 4

    # T2 hyperintensity
    t2 = features.get("t2_hyperintensity", {})
    if t2.get("value") is True:
        degree = t2.get("degree")
        t2_score = 6  # default moderate
        if degree == "mild":
            t2_score = 4
        elif degree == "moderate":
            t2_score = 6
        elif degree == "marked":
            t2_score = 8

        if has_seton:
            t2_score = max(2, t2_score - 2)
        score += t2_score

    # Collections/abscesses
    if features.get("collections_abscesses", {}).get("value") is True:
        score += 4

    # Rectal wall involvement
    if features.get("rectal_wall_involvement", {}).get("value") is True:
        score += 2

    return min(score, 22)


def calculate_magnifi(features: Dict, treatment_status: str) -> int:
    """Calculate MAGNIFI-CD score - exact copy of parser.html logic."""
    score = 0
    has_seton = treatment_status == "seton_in_place"

    # Fistula count
    count = features.get("fistula_count", {}).get("value")
    if count == 1:
        score += 1
    elif count == 2:
        score += 3
    elif count is not None and count >= 3:
        score += 5

    # T2 hyperintensity
    t2 = features.get("t2_hyperintensity", {})
    if t2.get("value") is True:
        degree = t2.get("degree")
        t2_score = 4  # default moderate
        if degree == "mild":
            t2_score = 2
        elif degree == "moderate":
            t2_score = 4
        elif degree == "marked":
            t2_score = 6

        if has_seton:
            t2_score = max(1, t2_score - 2)
        score += t2_score

    # Extension
    ext = features.get("extension", {}).get("value")
    if ext == "mild":
        score += 1
    elif ext == "moderate":
        score += 2
    elif ext == "severe":
        score += 3

    # Collections/abscesses
    collections = features.get("collections_abscesses", {})
    if collections.get("value") is True:
        if collections.get("size") == "large":
            score += 4
        else:
            score += 2

    # Inflammatory mass
    if features.get("inflammatory_mass", {}).get("value") is True:
        score += 4

    # Rectal wall involvement
    if features.get("rectal_wall_involvement", {}).get("value") is True:
        score += 2

    # Predominant feature
    pred = features.get("predominant_feature", {}).get("value")
    if pred == "inflammatory":
        if has_seton and not features.get("collections_abscesses", {}).get("value"):
            score += 2
        else:
            score += 4
    elif pred == "mixed":
        score += 2

    return min(score, 25)


def call_parser_api(report_text: str) -> Optional[Dict]:
    """Call the OpenRouter API with the parser prompt."""
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mri-crohn-atlas.vercel.app",
                "X-Title": "MRI-Crohn Atlas Parser Validation"
            },
            json={
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": get_parser_prompt(report_text)
                }],
                "temperature": 0.1,
                "max_tokens": 1500
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"    API error: {response.status_code} - {response.text[:200]}")
            return None

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            print(f"    No JSON found in response")
            return None

        return json.loads(json_match.group())

    except Exception as e:
        print(f"    API exception: {e}")
        return None


def load_progress() -> Dict:
    """Load progress from previous run."""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": {}, "api_calls": 0}


def save_progress(progress: Dict):
    """Save progress for resume capability."""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    print("=" * 60)
    print("PARSER VALIDATION ON GOLD CASES")
    print("=" * 60)
    print()

    # Load gold cases
    print(f"Loading {GOLD_CASES_PATH}...")
    with open(GOLD_CASES_PATH, "r") as f:
        gold_cases = json.load(f)
    print(f"  Loaded {len(gold_cases)} gold cases")

    # Filter to cases with ground truth
    cases_with_gt = [c for c in gold_cases if c.get("scored_vai") is not None]
    print(f"  {len(cases_with_gt)} cases have ground truth scores")
    print()

    # Load progress
    progress = load_progress()
    completed = progress.get("completed", {})
    api_calls = progress.get("api_calls", 0)

    print(f"Resuming from {len(completed)} completed cases, {api_calls} API calls made")
    print()

    # Process each case
    results = []
    for i, case in enumerate(cases_with_gt):
        case_id = case.get("case_id", f"case_{i}")
        findings = case.get("findings_text", "")

        print(f"[{i+1}/{len(cases_with_gt)}] Processing {case_id}...")

        # Check if already completed
        if case_id in completed:
            print(f"    Already completed, loading from cache")
            results.append(completed[case_id])
            continue

        # Skip if no findings text
        if not findings or len(findings) < 50:
            print(f"    Skipping: findings too short ({len(findings) if findings else 0} chars)")
            continue

        # Call API
        parsed = call_parser_api(findings)
        api_calls += 1

        if parsed is None:
            print(f"    Failed to parse, skipping")
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        # Calculate scores
        features = parsed.get("features", {})
        treatment_status = parsed.get("treatment_status", "unknown")

        predicted_vai = calculate_vai(features, treatment_status)
        predicted_magnifi = calculate_magnifi(features, treatment_status)

        expected_vai = case.get("scored_vai")
        expected_magnifi = case.get("scored_magnifi")

        # Build result
        result = {
            "case_id": case_id,
            "source": case.get("source"),
            "predicted_vai": predicted_vai,
            "expected_vai": expected_vai,
            "vai_error": predicted_vai - expected_vai if expected_vai is not None else None,
            "predicted_magnifi": predicted_magnifi,
            "expected_magnifi": expected_magnifi,
            "magnifi_error": predicted_magnifi - expected_magnifi if expected_magnifi is not None else None,
            "treatment_status": treatment_status,
            "features": features,
            "parsed_response": parsed
        }

        results.append(result)
        completed[case_id] = result

        # Save progress
        progress["completed"] = completed
        progress["api_calls"] = api_calls
        save_progress(progress)

        vai_err = result["vai_error"]
        mag_err = result["magnifi_error"]
        print(f"    VAI: {predicted_vai} (expected {expected_vai}, error {vai_err:+d})")
        print(f"    MAGNIFI: {predicted_magnifi} (expected {expected_magnifi}, error {mag_err:+d})")

        time.sleep(DELAY_BETWEEN_CALLS)

    # Calculate metrics
    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print()

    vai_errors = [r["vai_error"] for r in results if r["vai_error"] is not None]
    magnifi_errors = [r["magnifi_error"] for r in results if r["magnifi_error"] is not None]

    if vai_errors:
        vai_mae = np.mean(np.abs(vai_errors))
        vai_rmse = np.sqrt(np.mean(np.array(vai_errors) ** 2))
        vai_bias = np.mean(vai_errors)

        print(f"VAI Metrics (n={len(vai_errors)}):")
        print(f"  MAE:  {vai_mae:.2f} points")
        print(f"  RMSE: {vai_rmse:.2f} points")
        print(f"  Bias: {vai_bias:+.2f} points")
        print()

    if magnifi_errors:
        magnifi_mae = np.mean(np.abs(magnifi_errors))
        magnifi_rmse = np.sqrt(np.mean(np.array(magnifi_errors) ** 2))
        magnifi_bias = np.mean(magnifi_errors)

        print(f"MAGNIFI-CD Metrics (n={len(magnifi_errors)}):")
        print(f"  MAE:  {magnifi_mae:.2f} points")
        print(f"  RMSE: {magnifi_rmse:.2f} points")
        print(f"  Bias: {magnifi_bias:+.2f} points")
        print()

    print(f"Total API calls made: {api_calls}")

    # Save final results
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "n_cases": len(results),
            "api_calls": api_calls,
            "model": MODEL
        },
        "metrics": {
            "vai": {
                "mae": float(vai_mae) if vai_errors else None,
                "rmse": float(vai_rmse) if vai_errors else None,
                "bias": float(vai_bias) if vai_errors else None,
                "n": len(vai_errors)
            },
            "magnifi": {
                "mae": float(magnifi_mae) if magnifi_errors else None,
                "rmse": float(magnifi_rmse) if magnifi_errors else None,
                "bias": float(magnifi_bias) if magnifi_errors else None,
                "n": len(magnifi_errors)
            }
        },
        "cases": results
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
