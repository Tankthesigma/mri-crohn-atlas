#!/usr/bin/env python3
"""
Run REAL API validation on all 68 test cases using DeepSeek V3.2.
Saves progress after each case for resume capability.
"""

import json
import time
import requests
import re
from pathlib import Path

# API Configuration
OPENROUTER_API_KEY = "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"
MODEL = "deepseek/deepseek-chat"  # Using deepseek-chat as in the web parser
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Files
CASES_FILE = Path(__file__).parent / "mega_test_cases.json"
PROGRESS_FILE = Path(__file__).parent / "real_validation_progress.json"
RESULTS_FILE = Path(__file__).parent / "real_validation_results.json"

# Use the EXACT same prompt as the web parser (from parser.js)
EXTRACTION_PROMPT = """You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. For EACH feature you identify, include the EXACT quote from the report that supports it.

REPORT TEXT:
{report_text}

Return a JSON object with this EXACT structure:
{
    "features": {
        "fistula_count": {
            "value": <number or null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "fistula_type": {
            "value": <"intersphincteric"|"transsphincteric"|"suprasphincteric"|"extrasphincteric"|"complex"|"simple"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "t2_hyperintensity": {
            "value": <true|false|null>,
            "degree": <"mild"|"moderate"|"marked"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "extension": {
            "value": <"none"|"mild"|"moderate"|"severe"|null>,
            "description": "<description of extension pattern>",
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "collections_abscesses": {
            "value": <true|false|null>,
            "size": <"small"|"large"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "rectal_wall_involvement": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "inflammatory_mass": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "predominant_feature": {
            "value": <"inflammatory"|"fibrotic"|"mixed"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "sphincter_involvement": {
            "internal": <true|false|null>,
            "external": <true|false|null>,
            "evidence": "<exact quote from report>"
        },
        "internal_opening": {
            "location": "<clock position or description>",
            "evidence": "<exact quote from report>"
        },
        "branching": {
            "value": <true|false|null>,
            "evidence": "<exact quote from report>"
        }
    },
    "overall_assessment": {
        "activity": <"active"|"healing"|"healed"|"chronic">,
        "severity": <"mild"|"moderate"|"severe">,
        "confidence": <"high"|"medium"|"low">
    },
    "clinical_notes": "<any additional relevant observations>"
}

IMPORTANT:
- Use null if a feature is not mentioned or cannot be determined
- The "evidence" field must contain the EXACT text from the report
- Be conservative - only report features explicitly stated
- Return ONLY valid JSON, no markdown or explanations"""


def calculate_vai(features):
    """Calculate VAI score from extracted features - matches parser.js exactly"""
    score = 0

    # Fistula count (0-2)
    count = features.get('fistula_count', {}).get('value')
    if count is not None:
        if count == 0:
            pass
        elif count == 1:
            score += 1
        else:
            score += 2

    # Fistula location/type (0-2)
    ftype = features.get('fistula_type', {}).get('value')
    if ftype:
        if ftype in ['simple', 'intersphincteric']:
            score += 1
        elif ftype in ['transsphincteric', 'suprasphincteric', 'extrasphincteric', 'complex']:
            score += 2

    # Extension (0-4)
    extension = features.get('extension', {}).get('value')
    if extension:
        ext_scores = {'none': 0, 'mild': 2, 'moderate': 3, 'severe': 4}
        score += ext_scores.get(extension, 0)

    # T2 hyperintensity (0-8) - most important for activity
    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        t2_scores = {'mild': 4, 'moderate': 6, 'marked': 8}
        score += t2_scores.get(degree, 6)

    # Collections (0-4)
    if features.get('collections_abscesses', {}).get('value') is True:
        score += 4

    # Rectal wall involvement (0-2)
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 2

    return min(score, 22)


def calculate_magnifi(features):
    """Calculate MAGNIFI-CD score from extracted features - matches parser.js exactly"""
    score = 0

    # Fistula count (0-3)
    count = features.get('fistula_count', {}).get('value')
    if count is not None:
        if count == 0:
            pass
        elif count == 1:
            score += 1
        elif count == 2:
            score += 2
        else:
            score += 3

    # Fistula activity based on T2/enhancement (0-6)
    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        act_scores = {'mild': 2, 'moderate': 4, 'marked': 6}
        score += act_scores.get(degree, 4)

    # Collections (0-4)
    collections = features.get('collections_abscesses', {})
    if collections.get('value') is True:
        size = collections.get('size', 'small')
        score += 4 if size == 'large' else 2

    # Inflammatory mass (0-3)
    if features.get('inflammatory_mass', {}).get('value') is True:
        score += 3

    # Rectal wall/proctitis (0-4)
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 4

    # Extension component (0-3)
    extension = features.get('extension', {}).get('value')
    if extension:
        ext_scores = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        score += ext_scores.get(extension, 0)

    # Predominant feature adjustment (0-2)
    pf = features.get('predominant_feature', {}).get('value')
    if pf == 'inflammatory':
        score += 2
    elif pf == 'mixed':
        score += 1

    return min(score, 25)


def load_progress():
    """Load existing progress or return empty state"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_progress(progress):
    """Save progress after each case"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def parse_json_response(content):
    """Extract JSON from API response, handling markdown code blocks"""
    # Try to extract JSON from code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    # Clean up common issues
    content = content.strip()

    # Try to parse
    return json.loads(content)


def call_api(report_text):
    """Call DeepSeek API and return parsed features"""
    prompt = EXTRACTION_PROMPT.replace('{report_text}', report_text.strip())

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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500
            },
            timeout=60
        )

        if response.status_code != 200:
            return {"error": f"API error {response.status_code}: {response.text[:200]}"}

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Parse the JSON response
        parsed = parse_json_response(content)
        return {"success": True, "data": parsed, "raw": content}

    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": content if 'content' in dir() else None}
    except Exception as e:
        return {"error": str(e)}


def process_case(case):
    """Process a single test case"""
    case_id = case.get("id", "unknown")

    # Get report text
    report_text = case.get("report_text", "")
    if not report_text:
        return {"error": "no report text"}

    # Call API
    result = call_api(report_text)

    if result.get("error"):
        return result

    # Extract features and calculate scores
    parsed_data = result.get("data", {})
    features = parsed_data.get("features", {})

    vai_score = calculate_vai(features)
    magnifi_score = calculate_magnifi(features)

    # Get confidence from overall assessment
    confidence = 70  # default
    overall = parsed_data.get("overall_assessment", {})
    if overall.get("confidence") == "high":
        confidence = 90
    elif overall.get("confidence") == "medium":
        confidence = 70
    elif overall.get("confidence") == "low":
        confidence = 50

    return {
        "success": True,
        "vai_score": vai_score,
        "magnifi_score": magnifi_score,
        "confidence": confidence,
        "features": features,
        "overall_assessment": overall
    }


def main():
    print("=" * 70)
    print("MRI-Crohn Atlas Parser - REAL API Validation")
    print("=" * 70)

    # Load test cases
    with open(CASES_FILE) as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    print(f"\nTotal test cases: {len(cases)}")

    # Load existing progress
    progress = load_progress()
    completed_ids = set(progress["completed"])
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(cases) - len(completed_ids)}")
    print("-" * 70)

    # Process each case
    for i, case in enumerate(cases):
        case_id = case.get("id", f"case_{i}")

        # Skip if already done
        if case_id in completed_ids:
            continue

        print(f"\n[{len(completed_ids) + 1}/{len(cases)}] Processing: {case_id}")
        print(f"    Type: {case.get('case_type', 'unknown')} | Severity: {case.get('severity', 'unknown')}")

        # Get expected values
        ground_truth = case.get("ground_truth", {})
        expected_vai = ground_truth.get("expected_vai_score")
        expected_magnifi = ground_truth.get("expected_magnifi_score")

        # Process case
        result = process_case(case)

        # Build result record
        case_result = {
            "case_id": case_id,
            "source": case.get("source", "unknown"),
            "case_type": case.get("case_type", "unknown"),
            "severity": case.get("severity", "unknown"),
            "expected_vai": expected_vai,
            "expected_magnifi": expected_magnifi,
            "predicted_vai": result.get("vai_score"),
            "predicted_magnifi": result.get("magnifi_score"),
            "confidence": result.get("confidence"),
            "error": result.get("error")
        }

        # Calculate errors
        if case_result["predicted_vai"] is not None and expected_vai is not None:
            case_result["vai_error"] = case_result["predicted_vai"] - expected_vai
        if case_result["predicted_magnifi"] is not None and expected_magnifi is not None:
            case_result["magnifi_error"] = case_result["predicted_magnifi"] - expected_magnifi

        # Print result
        if result.get("error"):
            print(f"    ERROR: {result['error']}")
        else:
            vai_err = case_result.get("vai_error", "N/A")
            mag_err = case_result.get("magnifi_error", "N/A")
            print(f"    VAI: {expected_vai} -> {case_result['predicted_vai']} (error: {vai_err})")
            print(f"    MAGNIFI: {expected_magnifi} -> {case_result['predicted_magnifi']} (error: {mag_err})")

        # Update progress
        progress["results"].append(case_result)
        progress["completed"].append(case_id)
        completed_ids.add(case_id)

        # Save after EVERY case
        save_progress(progress)

        # Rate limiting
        time.sleep(1.5)

    # Save final results
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)

    # Calculate summary stats
    valid_results = [r for r in progress["results"] if r.get("predicted_vai") is not None]

    if valid_results:
        vai_errors = [abs(r["vai_error"]) for r in valid_results if r.get("vai_error") is not None]
        mag_errors = [abs(r["magnifi_error"]) for r in valid_results if r.get("magnifi_error") is not None]

        print(f"\nSuccessful cases: {len(valid_results)}/{len(progress['results'])}")

        if vai_errors:
            vai_mae = sum(vai_errors) / len(vai_errors)
            vai_acc_2 = sum(1 for e in vai_errors if e <= 2) / len(vai_errors) * 100
            vai_acc_3 = sum(1 for e in vai_errors if e <= 3) / len(vai_errors) * 100
            print(f"\nVAI Results:")
            print(f"  MAE: {vai_mae:.2f}")
            print(f"  Accuracy (±2): {vai_acc_2:.1f}%")
            print(f"  Accuracy (±3): {vai_acc_3:.1f}%")

        if mag_errors:
            mag_mae = sum(mag_errors) / len(mag_errors)
            mag_acc_3 = sum(1 for e in mag_errors if e <= 3) / len(mag_errors) * 100
            mag_acc_5 = sum(1 for e in mag_errors if e <= 5) / len(mag_errors) * 100
            print(f"\nMAGNIFI Results:")
            print(f"  MAE: {mag_mae:.2f}")
            print(f"  Accuracy (±3): {mag_acc_3:.1f}%")
            print(f"  Accuracy (±5): {mag_acc_5:.1f}%")

    # Save final results file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(progress["results"], f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
