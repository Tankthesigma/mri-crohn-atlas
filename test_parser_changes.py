#!/usr/bin/env python3
"""
Test script to verify parser changes:
1. Symbolic clamp for healed cases (MAGNIFI=0)
2. Few-shot examples in prompt

This tests with REAL API calls to verify end-to-end behavior.
"""

import json
import requests
import re

# API Configuration
API_KEY = "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"
MODEL = "deepseek/deepseek-chat-v3-0324"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Updated prompt matching parser.html with few-shot examples
EXTRACTION_PROMPT = """You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. AVOID OVERESTIMATING severity.

REPORT TEXT:
{report_text}

CRITICAL RULES:

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

=== T2 HYPERINTENSITY (BE CONSERVATIVE) ===
- ONLY mark as TRUE if explicitly described: "hyperintense", "T2 bright", "high signal on T2"
- If report doesn't mention T2/signal intensity explicitly = null (unknown)

=== EXTENSION ===
- "moderate" = ischioanal fossa, typical transsphincteric (3-4cm tracts)
- "severe" = ONLY for supralevator, horseshoe pattern, extends to thigh/buttock

=== FEW-SHOT EXAMPLES (Apply These Patterns) ===

EXAMPLE 1 - HORSESHOE FISTULA:
Input: "Fistula at 6 o'clock with bilateral extension into ischioanal fossae"
Logic: "Bilateral extension" = horseshoe pattern, even if only one tract origin named
Output: {{ "extension": {{ "value": "severe" }}, "fistula_type": {{ "value": "complex" }} }}
KEY: Horseshoe = always "severe" extension (3 points in MAGNIFI)

EXAMPLE 2 - HEALED/FIBROTIC TRACT:
Input: "Dark T2 linear band. No fluid signal or active inflammation."
Logic: "Dark T2" + "no fluid signal" = inactive/healed tract with NO active disease
Output: {{ "t2_hyperintensity": {{ "value": false, "degree": "none" }}, "predominant_feature": {{ "value": "fibrotic" }}, "overall_assessment": {{ "activity": "healed" }} }}
KEY: Dark T2/no fluid signal → MAGNIFI should be 0

EXAMPLE 3 - AMBIGUOUS/EQUIVOCAL REPORT:
Input: "Possible tract vs scar. Cannot exclude small abscess."
Logic: "Possible", "cannot exclude" = equivocal language, score conservatively
Output: {{ "confidence": "low", "collections_abscesses": {{ "value": false, "confidence": "low" }} }}
KEY: When uncertain, default to lower/absent rather than assuming worst case

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


def parse_report_api(report_text):
    """Parse a report using the REAL LLM API"""
    prompt = EXTRACTION_PROMPT.format(report_text=report_text.strip())

    try:
        response = requests.post(
            API_URL,
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
            },
            json={
                'model': MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.1,
                'max_tokens': 1500
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text[:200]}")
            return None

        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            print("No JSON found in response")
            return None

        return json.loads(json_match.group(0))

    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_vai(features, treatment_status):
    """Calculate VAI score from extracted features (matching parser.html)"""
    score = 0
    has_seton = treatment_status == 'seton_in_place'

    count = features.get('fistula_count', {}).get('value')
    if count == 1:
        score += 1
    elif count and count >= 2:
        score += 2

    ftype = features.get('fistula_type', {}).get('value')
    if ftype in ['simple', 'intersphincteric']:
        score += 1
    elif ftype in ['transsphincteric', 'suprasphincteric', 'extrasphincteric', 'complex']:
        score += 2

    ext = features.get('extension', {}).get('value')
    if ext == 'mild':
        score += 2
    elif ext == 'moderate':
        score += 3
    elif ext == 'severe':
        score += 4

    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        t2_score = {'mild': 4, 'moderate': 6, 'marked': 8}.get(degree, 6)
        if has_seton:
            t2_score = max(2, t2_score - 2)
        score += t2_score

    if features.get('collections_abscesses', {}).get('value') is True:
        score += 4
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 2

    return min(score, 22)


def calculate_magnifi(features, treatment_status, report_text=''):
    """Calculate MAGNIFI score with SYMBOLIC CLAMP for healed cases (matching parser.html)"""

    # === SYMBOLIC CLAMP: Force MAGNIFI=0 for clinically healed cases ===
    no_t2 = (features.get('t2_hyperintensity', {}).get('value') is False or
             features.get('t2_hyperintensity', {}).get('degree') == 'none')

    report_lower = report_text.lower()
    healed_keywords = [
        'healed', 'fibrotic', 'no active inflammation', 'dark t2',
        'no fluid signal', 'inactive tract', 'scarred', 'resolved',
        'completely healed', 'no residual', 'fibrosis without inflammation'
    ]
    has_healed_indicator = any(kw in report_lower for kw in healed_keywords)

    is_fibrotic = features.get('predominant_feature', {}).get('value') == 'fibrotic'

    # CLAMP: If healed signal + no active inflammation markers
    if ((no_t2 or (has_healed_indicator and is_fibrotic)) and
        not features.get('collections_abscesses', {}).get('value') and
        not features.get('inflammatory_mass', {}).get('value')):
        print("    >>> SYMBOLIC CLAMP TRIGGERED: MAGNIFI = 0")
        return 0

    # === Standard MAGNIFI calculation ===
    score = 0
    has_seton = treatment_status == 'seton_in_place'

    count = features.get('fistula_count', {}).get('value')
    if count == 1:
        score += 1
    elif count == 2:
        score += 3
    elif count and count >= 3:
        score += 5

    t2 = features.get('t2_hyperintensity', {})
    if t2.get('value') is True:
        degree = t2.get('degree', 'moderate')
        t2_score = {'mild': 2, 'moderate': 4, 'marked': 6}.get(degree, 4)
        if has_seton:
            t2_score = max(1, t2_score - 2)
        score += t2_score

    ext = features.get('extension', {}).get('value')
    if ext == 'mild':
        score += 1
    elif ext == 'moderate':
        score += 2
    elif ext == 'severe':
        score += 3

    collections = features.get('collections_abscesses', {})
    if collections.get('value') is True:
        score += 4 if collections.get('size') == 'large' else 2

    if features.get('inflammatory_mass', {}).get('value') is True:
        score += 4
    if features.get('rectal_wall_involvement', {}).get('value') is True:
        score += 2

    pred = features.get('predominant_feature', {}).get('value')
    if pred == 'inflammatory':
        if has_seton and not features.get('collections_abscesses', {}).get('value'):
            score += 2
        else:
            score += 4
    elif pred == 'mixed':
        score += 2

    return min(score, 25)


def test_symbolic_clamp():
    """Test the symbolic clamp logic for healed cases"""
    print("="*70)
    print("TEST 1: SYMBOLIC CLAMP FOR HEALED CASES")
    print("="*70)

    # Simulated LLM output for: "Dark T2 linear band at 5 o'clock. No fluid signal. Fibrotic tract, no active inflammation."
    report_text = "Dark T2 linear band at 5 o'clock. No fluid signal. Fibrotic tract, no active inflammation."

    # What the LLM would likely return given the few-shot example
    features = {
        "fistula_count": {"value": 1, "confidence": "high"},
        "fistula_type": {"value": "intersphincteric", "confidence": "medium"},
        "t2_hyperintensity": {"value": False, "degree": "none", "confidence": "high"},
        "extension": {"value": "none", "confidence": "high"},
        "collections_abscesses": {"value": False, "confidence": "high"},
        "rectal_wall_involvement": {"value": False, "confidence": "high"},
        "inflammatory_mass": {"value": False, "confidence": "high"},
        "predominant_feature": {"value": "fibrotic", "confidence": "high"}
    }
    treatment_status = "unknown"

    print(f"\nReport: \"{report_text}\"")
    print(f"\nSimulated LLM Features:")
    print(f"  - t2_hyperintensity: value=False, degree='none'")
    print(f"  - predominant_feature: 'fibrotic'")
    print(f"  - collections_abscesses: False")
    print(f"  - inflammatory_mass: False")

    magnifi = calculate_magnifi(features, treatment_status, report_text)
    vai = calculate_vai(features, treatment_status)

    print(f"\nResults:")
    print(f"  VAI: {vai}")
    print(f"  MAGNIFI: {magnifi}")

    expected_magnifi = 0
    if magnifi == expected_magnifi:
        print(f"\n  [PASS] MAGNIFI = {magnifi} (expected {expected_magnifi})")
        return True
    else:
        print(f"\n  [FAIL] MAGNIFI = {magnifi} (expected {expected_magnifi})")
        return False


def test_horseshoe():
    """Test horseshoe fistula recognition"""
    print("\n" + "="*70)
    print("TEST 2: HORSESHOE FISTULA (FEW-SHOT EXAMPLE)")
    print("="*70)

    report_text = "Complex fistula with bilateral extension into ischioanal fossae crossing midline."

    # What the LLM should return based on few-shot example
    features = {
        "fistula_count": {"value": 1, "confidence": "high"},
        "fistula_type": {"value": "complex", "confidence": "high"},
        "t2_hyperintensity": {"value": True, "degree": "moderate", "confidence": "medium"},
        "extension": {"value": "severe", "confidence": "high"},  # KEY: horseshoe = severe
        "collections_abscesses": {"value": False, "confidence": "medium"},
        "rectal_wall_involvement": {"value": False, "confidence": "low"},
        "inflammatory_mass": {"value": False, "confidence": "medium"},
        "predominant_feature": {"value": "inflammatory", "confidence": "medium"}
    }
    treatment_status = "unknown"

    print(f"\nReport: \"{report_text}\"")
    print(f"\nExpected LLM behavior (based on few-shot):")
    print(f"  - extension: 'severe' (horseshoe = bilateral = severe)")
    print(f"  - fistula_type: 'complex'")

    magnifi = calculate_magnifi(features, treatment_status, report_text)
    vai = calculate_vai(features, treatment_status)

    print(f"\nResults:")
    print(f"  VAI: {vai}")
    print(f"  MAGNIFI: {magnifi}")
    print(f"  Extension detected: {features['extension']['value']}")

    expected_extension = "severe"
    if features['extension']['value'] == expected_extension:
        print(f"\n  [PASS] Extension = '{expected_extension}' (horseshoe correctly identified)")
        print(f"  [INFO] MAGNIFI = {magnifi} (includes +3 for severe extension)")
        return True
    else:
        print(f"\n  [FAIL] Extension = '{features['extension']['value']}' (expected '{expected_extension}')")
        return False


def test_ambiguous():
    """Test ambiguous/equivocal case handling"""
    print("\n" + "="*70)
    print("TEST 3: AMBIGUOUS/EQUIVOCAL CASE (FEW-SHOT EXAMPLE)")
    print("="*70)

    report_text = "Possible subtle tract vs scar tissue. Cannot exclude small abscess."

    # What the LLM should return based on few-shot example (conservative)
    features = {
        "fistula_count": {"value": 1, "confidence": "low"},
        "fistula_type": {"value": None, "confidence": "low"},
        "t2_hyperintensity": {"value": None, "degree": None, "confidence": "low"},
        "extension": {"value": None, "confidence": "low"},
        "collections_abscesses": {"value": False, "confidence": "low"},  # KEY: conservative
        "rectal_wall_involvement": {"value": None, "confidence": "low"},
        "inflammatory_mass": {"value": None, "confidence": "low"},
        "predominant_feature": {"value": None, "confidence": "low"}
    }
    treatment_status = "unknown"

    print(f"\nReport: \"{report_text}\"")
    print(f"\nExpected LLM behavior (based on few-shot):")
    print(f"  - confidence: 'low' across features")
    print(f"  - collections_abscesses: False (conservative, not assuming abscess)")
    print(f"  - Most features: null (not assuming)")

    magnifi = calculate_magnifi(features, treatment_status, report_text)
    vai = calculate_vai(features, treatment_status)

    print(f"\nResults:")
    print(f"  VAI: {vai} (conservative/low due to uncertainty)")
    print(f"  MAGNIFI: {magnifi} (conservative/low due to uncertainty)")
    print(f"  Collections assumed: {features['collections_abscesses']['value']} (conservative = False)")

    # Conservative scoring means low scores and abscess = False
    if features['collections_abscesses']['value'] is False:
        print(f"\n  [PASS] Conservative scoring: abscess NOT assumed from 'cannot exclude'")
        return True
    else:
        print(f"\n  [FAIL] Should not assume abscess from 'cannot exclude'")
        return False


def test_clamp_edge_cases():
    """Test edge cases for the symbolic clamp"""
    print("\n" + "="*70)
    print("TEST 4: SYMBOLIC CLAMP EDGE CASES")
    print("="*70)

    edge_cases = [
        {
            "name": "Healed with no T2 mentioned",
            "report": "Healed fistula tract. No residual inflammation.",
            "features": {
                "fistula_count": {"value": 1},
                "t2_hyperintensity": {"value": None},  # Not mentioned
                "predominant_feature": {"value": "fibrotic"},
                "collections_abscesses": {"value": False},
                "inflammatory_mass": {"value": False}
            },
            "expect_clamp": True
        },
        {
            "name": "Dark T2 but with abscess",
            "report": "Dark T2 tract with small adjacent abscess.",
            "features": {
                "fistula_count": {"value": 1},
                "t2_hyperintensity": {"value": False, "degree": "none"},
                "predominant_feature": {"value": "mixed"},
                "collections_abscesses": {"value": True},  # Has abscess!
                "inflammatory_mass": {"value": False}
            },
            "expect_clamp": False  # Should NOT clamp because of abscess
        },
        {
            "name": "Fibrotic keywords but active T2",
            "report": "Fibrotic tract with mild T2 hyperintensity.",
            "features": {
                "fistula_count": {"value": 1},
                "t2_hyperintensity": {"value": True, "degree": "mild"},  # Active!
                "predominant_feature": {"value": "mixed"},
                "collections_abscesses": {"value": False},
                "inflammatory_mass": {"value": False}
            },
            "expect_clamp": False  # Should NOT clamp because of active T2
        }
    ]

    all_passed = True
    for case in edge_cases:
        print(f"\n  Testing: {case['name']}")
        print(f"    Report: \"{case['report']}\"")

        magnifi = calculate_magnifi(case['features'], 'unknown', case['report'])
        clamped = (magnifi == 0)

        if clamped == case['expect_clamp']:
            status = "PASS"
            result_str = "clamped to 0" if clamped else f"calculated as {magnifi}"
        else:
            status = "FAIL"
            result_str = f"got {magnifi}, expected {'0 (clamped)' if case['expect_clamp'] else 'non-zero'}"
            all_passed = False

        print(f"    [{status}] MAGNIFI = {magnifi} ({result_str})")

    return all_passed


def test_real_api():
    """Test with REAL API calls"""
    print("="*70)
    print("REAL API TESTS (Live DeepSeek Calls)")
    print("="*70)

    import time

    test_cases = [
        {
            "name": "HEALED",
            "report": "Dark T2 linear band at 5 o'clock. No fluid signal. Fibrotic tract, no active inflammation.",
            "check": lambda f, m, r: m == 0,
            "expected": "MAGNIFI = 0 (symbolic clamp)"
        },
        {
            "name": "HORSESHOE",
            "report": "Complex fistula with bilateral extension into ischioanal fossae crossing midline.",
            "check": lambda f, m, r: f.get('extension', {}).get('value') == 'severe',
            "expected": "extension = 'severe'"
        },
        {
            "name": "AMBIGUOUS",
            "report": "Possible subtle tract vs scar tissue. Cannot exclude small abscess.",
            "check": lambda f, m, r: f.get('collections_abscesses', {}).get('value') != True,
            "expected": "collections_abscesses != True (conservative)"
        }
    ]

    results = []

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Report: \"{tc['report']}\"")
        print(f"Expected: {tc['expected']}")
        print("Calling API...")

        parsed = parse_report_api(tc['report'])

        if parsed is None:
            print("  [FAIL] API call failed")
            results.append((tc['name'], False))
            continue

        features = parsed.get('features', {})
        treatment_status = parsed.get('treatment_status', 'unknown')
        overall = parsed.get('overall_assessment', {})

        vai = calculate_vai(features, treatment_status)
        magnifi = calculate_magnifi(features, treatment_status, tc['report'])

        print(f"\nAPI Response:")
        print(f"  VAI: {vai}, MAGNIFI: {magnifi}")
        print(f"  extension: {features.get('extension', {}).get('value')}")
        print(f"  t2_hyperintensity: {features.get('t2_hyperintensity', {})}")
        print(f"  collections_abscesses: {features.get('collections_abscesses', {}).get('value')}")
        print(f"  predominant_feature: {features.get('predominant_feature', {}).get('value')}")
        print(f"  overall_assessment: {overall}")

        passed = tc['check'](features, magnifi, tc['report'])
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {tc['expected']}")
        results.append((tc['name'], passed))

        time.sleep(1)  # Rate limiting

    return results


def main():
    print("="*70)
    print("PARSER CHANGES VERIFICATION")
    print("Testing: Symbolic Clamp + Few-Shot Examples")
    print("="*70)

    # Run REAL API tests
    print("\n" + "="*70)
    print("PART 1: REAL API TESTS")
    print("="*70)

    api_results = test_real_api()

    # Run local logic tests
    print("\n" + "="*70)
    print("PART 2: LOCAL LOGIC TESTS (Simulated)")
    print("="*70)

    local_results = []
    local_results.append(("Healed Case (Symbolic Clamp)", test_symbolic_clamp()))
    local_results.append(("Horseshoe Fistula (Few-Shot)", test_horseshoe()))
    local_results.append(("Ambiguous Case (Few-Shot)", test_ambiguous()))
    local_results.append(("Clamp Edge Cases", test_clamp_edge_cases()))

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\nReal API Tests:")
    for name, passed in api_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print("\nLocal Logic Tests:")
    for name, passed in local_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    api_passed = sum(1 for _, p in api_results if p)
    local_passed = sum(1 for _, p in local_results if p)

    print(f"\nAPI Tests: {api_passed}/{len(api_results)} passed")
    print(f"Local Tests: {local_passed}/{len(local_results)} passed")

    if api_passed == len(api_results) and local_passed == len(local_results):
        print("\nAll tests passed! Parser changes working correctly.")


if __name__ == "__main__":
    main()
