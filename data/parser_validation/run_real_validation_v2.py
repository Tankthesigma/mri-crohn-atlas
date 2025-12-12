#!/usr/bin/env python3
"""
Run REAL API validation with IMPROVED V2 prompt.
"""

import json
import time
import requests
from pathlib import Path

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-chat"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Files
CASES_FILE = Path(__file__).parent / "mega_test_cases.json"
PROGRESS_FILE = Path(__file__).parent / "real_validation_progress_v2.json"
RESULTS_FILE = Path(__file__).parent / "real_validation_results_v2.json"

# Load V2 prompt
PROMPT_FILE = Path(__file__).parent.parent.parent / "src/web/parser_prompt_v2.txt"
with open(PROMPT_FILE) as f:
    V2_PROMPT_TEMPLATE = f.read()


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def parse_json_response(content):
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())


def call_api(report_text):
    """Call DeepSeek API with V2 prompt"""
    prompt = V2_PROMPT_TEMPLATE + f"\n\n=== MRI REPORT TO ANALYZE ===\n\n{report_text.strip()}"

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mri-crohn-atlas.vercel.app",
                "X-Title": "MRI-Crohn Atlas Parser V2"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 800
            },
            timeout=60
        )

        if response.status_code != 200:
            return {"error": f"API error {response.status_code}"}

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        return {"success": True, "data": parsed}

    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("MRI-Crohn Atlas Parser V2 - REAL API Validation")
    print("=" * 70)

    with open(CASES_FILE) as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    print(f"\nTotal test cases: {len(cases)}")

    progress = load_progress()
    completed_ids = set(progress["completed"])
    print(f"Already completed: {len(completed_ids)}")
    print("-" * 70)

    for i, case in enumerate(cases):
        case_id = case.get("id", f"case_{i}")

        if case_id in completed_ids:
            continue

        print(f"\n[{len(completed_ids) + 1}/{len(cases)}] Processing: {case_id}")

        ground_truth = case.get("ground_truth", {})
        expected_vai = ground_truth.get("expected_vai_score")
        expected_magnifi = ground_truth.get("expected_magnifi_score")

        report_text = case.get("report_text", "")
        if not report_text:
            result = {"error": "no report text"}
        else:
            result = call_api(report_text)

        parsed_data = result.get("data", {})

        case_result = {
            "case_id": case_id,
            "source": case.get("source", "unknown"),
            "case_type": case.get("case_type", "unknown"),
            "severity": case.get("severity", "unknown"),
            "expected_vai": expected_vai,
            "expected_magnifi": expected_magnifi,
            "predicted_vai": parsed_data.get("vai_score"),
            "predicted_magnifi": parsed_data.get("magnifi_score"),
            "confidence": parsed_data.get("confidence"),
            "vai_breakdown": parsed_data.get("vai_breakdown"),
            "reasoning": parsed_data.get("reasoning"),
            "flags": parsed_data.get("flags"),
            "error": result.get("error")
        }

        if case_result["predicted_vai"] is not None and expected_vai is not None:
            case_result["vai_error"] = case_result["predicted_vai"] - expected_vai
        if case_result["predicted_magnifi"] is not None and expected_magnifi is not None:
            case_result["magnifi_error"] = case_result["predicted_magnifi"] - expected_magnifi

        if result.get("error"):
            print(f"    ERROR: {result['error']}")
        else:
            vai_err = case_result.get("vai_error", "N/A")
            mag_err = case_result.get("magnifi_error", "N/A")
            print(f"    VAI: {expected_vai} -> {case_result['predicted_vai']} (error: {vai_err})")
            print(f"    MAGNIFI: {expected_magnifi} -> {case_result['predicted_magnifi']} (error: {mag_err})")
            if parsed_data.get("flags"):
                print(f"    Flags: {parsed_data['flags']}")

        progress["results"].append(case_result)
        progress["completed"].append(case_id)
        completed_ids.add(case_id)

        save_progress(progress)
        time.sleep(1.5)

    # Calculate summary
    print("\n" + "=" * 70)
    print("V2 VALIDATION COMPLETE!")
    print("=" * 70)

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
            print(f"\nMAGNIFI Results:")
            print(f"  MAE: {mag_mae:.2f}")
            print(f"  Accuracy (±3): {mag_acc_3:.1f}%")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(progress["results"], f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
