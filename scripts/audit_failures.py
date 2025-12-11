#!/usr/bin/env python3
"""
Audit High-Error Cases with LLM Judge

For cases where |predicted_vai - true_vai| >= 4, ask an LLM to adjudicate:
- Is the ground truth label wrong?
- Is the parser wrong?
- Is the report genuinely ambiguous?

This helps determine whether to fix the prompt or fix the dataset.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths
CLEAN_CASES_PATH = PROJECT_ROOT / "data" / "training" / "clean_cases.json"
GOLD_CASES_PATH = PROJECT_ROOT / "data" / "calibration" / "gold_cases.json"
VALIDATION_RESULTS_PATH = PROJECT_ROOT / "data" / "calibration" / "parser_validation_progress.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "calibration" / "failure_audit_report.json"

# API Configuration
OPENROUTER_API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v3.2"

# Error threshold
ERROR_THRESHOLD = 4
DELAY_BETWEEN_CALLS = 1.5


def get_judge_prompt(findings_text: str, true_vai: int, pred_vai: int,
                     true_magnifi: int, pred_magnifi: int) -> str:
    """Generate the LLM judge prompt."""
    return f"""You are an expert radiologist reviewing a scoring disagreement for a perianal fistula MRI report.

REPORT TEXT:
{findings_text}

SCORING DISAGREEMENT:
- Ground Truth VAI: {true_vai} | Parser Predicted VAI: {pred_vai} | Difference: {pred_vai - true_vai:+d}
- Ground Truth MAGNIFI: {true_magnifi} | Parser Predicted MAGNIFI: {pred_magnifi} | Difference: {pred_magnifi - true_magnifi:+d}

VAI SCORING RULES (0-22 scale):
- Fistula count: 1=1pt, ≥2=2pts
- Fistula type: simple/intersphincteric=1pt, complex/transsphincteric=2pts
- T2 hyperintensity: mild=4pts, moderate=6pts, marked=8pts
- Extension: mild=2pts, moderate=3pts, severe=4pts
- Collections/abscesses: 4pts if present
- Rectal wall involvement: 2pts if present

YOUR TASK:
Based STRICTLY on the text provided, analyze:

1. What features are EXPLICITLY mentioned in the report?
2. What features are IMPLIED but not explicitly stated?
3. What features are ABSENT or NEGATED?
4. Is there hedging language ("possible", "may represent", "cannot exclude")?

Then determine:
- Does the text support the GROUND TRUTH score of VAI={true_vai}?
- Does the text support the PARSER PREDICTED score of VAI={pred_vai}?
- Which score is more accurate given what the text actually says?

Return a JSON object with this EXACT structure:
{{
    "explicit_features": {{
        "fistula_count": <number or null>,
        "fistula_type": <type or null>,
        "t2_hyperintensity": <true/false/null>,
        "t2_degree": <mild/moderate/marked or null>,
        "extension": <none/mild/moderate/severe or null>,
        "abscess": <true/false/null>,
        "rectal_wall": <true/false/null>
    }},
    "hedging_language": <true/false>,
    "hedging_examples": [<list of hedging phrases found>],
    "key_sentences": [<2-3 most scoring-relevant sentences>],
    "calculated_vai_from_text": <your calculation based on explicit features>,
    "verdict": <"LABEL_WRONG" | "PARSER_WRONG" | "AMBIGUOUS">,
    "verdict_confidence": <"high" | "medium" | "low">,
    "reasoning": "<1-2 sentence explanation>"
}}

VERDICT GUIDELINES:
- "LABEL_WRONG": The ground truth score is NOT supported by the text. The parser is correct.
- "PARSER_WRONG": The text CLEARLY supports the ground truth. The parser missed something.
- "AMBIGUOUS": The text is genuinely unclear/equivocal. Neither score is obviously correct.

IMPORTANT: Use null for features not mentioned. Be conservative - only mark features as present if explicitly stated."""


def call_judge_api(prompt: str) -> Optional[Dict]:
    """Call the OpenRouter API with the judge prompt."""
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mri-crohn-atlas.vercel.app",
                "X-Title": "MRI-Crohn Atlas Failure Audit"
            },
            json={
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt
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


def load_data() -> tuple:
    """Load validation results and case texts."""
    # Load validation results (predictions)
    with open(VALIDATION_RESULTS_PATH, "r") as f:
        validation_data = json.load(f)

    completed = validation_data.get("completed", {})

    # Load gold cases (for findings_text)
    with open(GOLD_CASES_PATH, "r") as f:
        gold_cases = json.load(f)

    # Build lookup by case_id
    gold_lookup = {c.get("case_id"): c for c in gold_cases}

    return completed, gold_lookup


def find_major_disagreements(completed: Dict, gold_lookup: Dict) -> List[Dict]:
    """Find cases with |VAI error| >= threshold."""
    disagreements = []

    for case_id, result in completed.items():
        vai_error = result.get("vai_error")
        if vai_error is not None and abs(vai_error) >= ERROR_THRESHOLD:
            gold_case = gold_lookup.get(case_id, {})
            findings_text = gold_case.get("findings_text", "")

            if not findings_text:
                continue

            disagreements.append({
                "case_id": case_id,
                "findings_text": findings_text,
                "source": result.get("source"),
                "predicted_vai": result.get("predicted_vai"),
                "expected_vai": result.get("expected_vai"),
                "vai_error": vai_error,
                "predicted_magnifi": result.get("predicted_magnifi"),
                "expected_magnifi": result.get("expected_magnifi"),
                "magnifi_error": result.get("magnifi_error"),
                "features_extracted": result.get("features", {})
            })

    # Sort by absolute error descending
    disagreements.sort(key=lambda x: abs(x["vai_error"]), reverse=True)
    return disagreements


def print_summary_table(audits: List[Dict]):
    """Print a summary table of verdicts."""
    print()
    print("=" * 90)
    print("FAILURE AUDIT SUMMARY")
    print("=" * 90)
    print()

    # Count verdicts
    verdicts = {"LABEL_WRONG": 0, "PARSER_WRONG": 0, "AMBIGUOUS": 0, "ERROR": 0}

    print(f"{'Case ID':<25} | {'True':<4} | {'Pred':<4} | {'Err':<4} | {'Verdict':<12} | {'Confidence':<10}")
    print("-" * 90)

    for audit in audits:
        case_id = audit["case_id"]
        true_vai = audit["expected_vai"]
        pred_vai = audit["predicted_vai"]
        err = audit["vai_error"]

        judge = audit.get("judge_analysis", {})
        verdict = judge.get("verdict", "ERROR")
        confidence = judge.get("verdict_confidence", "N/A")

        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        print(f"{case_id:<25} | {true_vai:<4} | {pred_vai:<4} | {err:+4d} | {verdict:<12} | {confidence:<10}")

    print("-" * 90)
    print()
    print("VERDICT COUNTS:")
    print(f"  LABEL_WRONG:  {verdicts['LABEL_WRONG']:>2} cases - Ground truth is incorrect, parser is right")
    print(f"  PARSER_WRONG: {verdicts['PARSER_WRONG']:>2} cases - Parser missed something, fix prompt")
    print(f"  AMBIGUOUS:    {verdicts['AMBIGUOUS']:>2} cases - Report is genuinely unclear")
    print(f"  ERROR:        {verdicts['ERROR']:>2} cases - Judge API failed")
    print()

    total = sum(verdicts.values())
    if total > 0:
        label_pct = verdicts["LABEL_WRONG"] / total * 100
        parser_pct = verdicts["PARSER_WRONG"] / total * 100
        ambig_pct = verdicts["AMBIGUOUS"] / total * 100

        print("RECOMMENDATION:")
        if label_pct > parser_pct and label_pct > ambig_pct:
            print(f"  >> FIX DATASET: {label_pct:.0f}% of failures are mislabeled ground truth")
        elif parser_pct > label_pct and parser_pct > ambig_pct:
            print(f"  >> FIX PROMPT: {parser_pct:.0f}% of failures are parser extraction errors")
        else:
            print(f"  >> MIXED: Consider both dataset cleanup AND prompt improvement")


def main():
    print("=" * 70)
    print("FAILURE AUDIT: LLM Judge Analysis")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    completed, gold_lookup = load_data()
    print(f"  {len(completed)} validation results loaded")
    print(f"  {len(gold_lookup)} gold cases loaded")
    print()

    # Find major disagreements
    print(f"Finding cases with |VAI error| >= {ERROR_THRESHOLD}...")
    disagreements = find_major_disagreements(completed, gold_lookup)
    print(f"  Found {len(disagreements)} major disagreements")
    print()

    if not disagreements:
        print("No major disagreements found. Exiting.")
        return

    # Audit each disagreement
    audits = []
    for i, case in enumerate(disagreements):
        case_id = case["case_id"]
        print(f"[{i+1}/{len(disagreements)}] Auditing {case_id}...")
        print(f"    VAI: {case['predicted_vai']} predicted vs {case['expected_vai']} true (error {case['vai_error']:+d})")

        # Call judge API
        prompt = get_judge_prompt(
            case["findings_text"],
            case["expected_vai"],
            case["predicted_vai"],
            case["expected_magnifi"],
            case["predicted_magnifi"]
        )

        judge_result = call_judge_api(prompt)

        if judge_result:
            verdict = judge_result.get("verdict", "UNKNOWN")
            confidence = judge_result.get("verdict_confidence", "N/A")
            reasoning = judge_result.get("reasoning", "")
            print(f"    Verdict: {verdict} ({confidence})")
            print(f"    Reasoning: {reasoning[:80]}...")
        else:
            judge_result = {"verdict": "ERROR", "reasoning": "API call failed"}
            print(f"    Verdict: ERROR (API failed)")

        audit = {
            **case,
            "judge_analysis": judge_result
        }
        audits.append(audit)

        time.sleep(DELAY_BETWEEN_CALLS)

    # Print summary
    print_summary_table(audits)

    # Save detailed results
    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "error_threshold": ERROR_THRESHOLD,
            "total_cases_audited": len(audits),
            "model": MODEL
        },
        "summary": {
            "label_wrong": sum(1 for a in audits if a.get("judge_analysis", {}).get("verdict") == "LABEL_WRONG"),
            "parser_wrong": sum(1 for a in audits if a.get("judge_analysis", {}).get("verdict") == "PARSER_WRONG"),
            "ambiguous": sum(1 for a in audits if a.get("judge_analysis", {}).get("verdict") == "AMBIGUOUS"),
            "error": sum(1 for a in audits if a.get("judge_analysis", {}).get("verdict") == "ERROR")
        },
        "audits": audits
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
