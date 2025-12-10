"""
Multi-Model Benchmark for MRI Report Parsing
=============================================
Compares our fine-tuned CrohnBridge model against major LLMs.

Models tested:
1. sigmsisgam/crohnbridge-parser-v1 (fine-tuned Qwen 0.5B)
2. DeepSeek V3
3. GPT-4o
4. Gemini 2.5 Pro
5. Grok 2

Usage:
    python run_benchmark.py
"""

import json
import time
import os
import sys
from datetime import datetime
import requests

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# Models to benchmark
MODELS = {
    "deepseek-v3": {
        "id": "deepseek/deepseek-chat",
        "provider": "openrouter",
        "cost_per_1k_input": 0.00014,
        "cost_per_1k_output": 0.00028,
    },
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "provider": "openrouter",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
    },
    "gemini-2.0-flash": {
        "id": "google/gemini-2.0-flash-001",
        "provider": "openrouter",
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.0004,
    },
    "claude-3.5-sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "qwen-2.5-72b": {
        "id": "qwen/qwen-2.5-72b-instruct",
        "provider": "openrouter",
        "cost_per_1k_input": 0.00035,
        "cost_per_1k_output": 0.0004,
    },
}

SYSTEM_PROMPT = """You are a medical AI that extracts Van Assche Index and MAGNIFI-CD scores from MRI radiology reports.

Analyze the MRI report and extract:
1. vai_score (0-22): Van Assche Index score
2. magnifi_score (0-25): MAGNIFI-CD score
3. extracted_features: Key clinical features found
4. confidence (0.0-1.0): Your confidence in the scores

Output ONLY valid JSON in this exact format:
{"vai_score": X, "magnifi_score": Y, "extracted_features": {"fistula_count": N, "fistula_type": "type", "t2_hyperintensity": "level", "has_abscess": bool}, "confidence": 0.XX}"""


def call_openrouter(model_id: str, user_message: str) -> tuple[str, float, int, int]:
    """Call OpenRouter API and return response, time, input tokens, output tokens."""
    start_time = time.time()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mri-crohn-atlas.vercel.app",
        "X-Title": "MRI-Crohn Atlas Benchmark"
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )

    elapsed = time.time() - start_time

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}", elapsed, 0, 0

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return content, elapsed, input_tokens, output_tokens


def parse_model_response(response: str) -> dict:
    """Parse JSON from model response."""
    import re

    try:
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 2:
                response = parts[1]

        # Find JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            parsed = json.loads(json_str)

            # Normalize field names
            vai = parsed.get("vai_score") or parsed.get("VAI") or parsed.get("vai") or parsed.get("van_assche_index")
            magnifi = parsed.get("magnifi_score") or parsed.get("MAGNIFI") or parsed.get("magnifi") or parsed.get("magnifi_cd")

            return {
                "vai_score": int(vai) if vai is not None else None,
                "magnifi_score": int(magnifi) if magnifi is not None else None,
                "extracted_features": parsed.get("extracted_features", {}),
                "confidence": parsed.get("confidence", 0.5)
            }
    except Exception as e:
        pass

    # Try regex fallback for vai_score
    try:
        vai_match = re.search(r'"vai_score"\s*:\s*(\d+)', response)
        magnifi_match = re.search(r'"magnifi_score"\s*:\s*(\d+)', response)
        if vai_match or magnifi_match:
            return {
                "vai_score": int(vai_match.group(1)) if vai_match else None,
                "magnifi_score": int(magnifi_match.group(1)) if magnifi_match else None,
            }
    except:
        pass

    return {"vai_score": None, "magnifi_score": None, "error": "Parse failed"}


def calculate_metrics(results: list) -> dict:
    """Calculate accuracy metrics from results."""
    vai_errors = []
    magnifi_errors = []
    vai_within_1 = 0
    vai_within_2 = 0
    magnifi_within_1 = 0
    magnifi_within_2 = 0
    valid_count = 0

    for r in results:
        if r.get("predicted_vai") is not None and r.get("expected_vai") is not None:
            vai_err = abs(r["predicted_vai"] - r["expected_vai"])
            vai_errors.append(vai_err)
            if vai_err <= 1:
                vai_within_1 += 1
            if vai_err <= 2:
                vai_within_2 += 1

        if r.get("predicted_magnifi") is not None and r.get("expected_magnifi") is not None:
            mag_err = abs(r["predicted_magnifi"] - r["expected_magnifi"])
            magnifi_errors.append(mag_err)
            if mag_err <= 1:
                magnifi_within_1 += 1
            if mag_err <= 2:
                magnifi_within_2 += 1
            valid_count += 1

    if valid_count == 0:
        return {"error": "No valid results"}

    return {
        "valid_cases": valid_count,
        "vai_mae": sum(vai_errors) / len(vai_errors) if vai_errors else None,
        "magnifi_mae": sum(magnifi_errors) / len(magnifi_errors) if magnifi_errors else None,
        "vai_accuracy_1pt": vai_within_1 / len(vai_errors) * 100 if vai_errors else 0,
        "vai_accuracy_2pt": vai_within_2 / len(vai_errors) * 100 if vai_errors else 0,
        "magnifi_accuracy_1pt": magnifi_within_1 / len(magnifi_errors) * 100 if magnifi_errors else 0,
        "magnifi_accuracy_2pt": magnifi_within_2 / len(magnifi_errors) * 100 if magnifi_errors else 0,
    }


def run_benchmark(test_cases: list, num_cases: int = 30) -> dict:
    """Run benchmark on all models."""

    # Select test cases (stratified by severity if possible)
    selected_cases = test_cases[:num_cases]

    print(f"\n{'='*60}")
    print(f"MRI REPORT PARSER BENCHMARK")
    print(f"{'='*60}")
    print(f"Test cases: {len(selected_cases)}")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    all_results = {}

    for model_name, model_config in MODELS.items():
        print(f"\n--- Testing: {model_name} ---")

        results = []
        total_time = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i, case in enumerate(selected_cases):
            report_text = case.get("report_text", "")
            expected_vai = case.get("ground_truth", {}).get("expected_vai_score")
            expected_magnifi = case.get("ground_truth", {}).get("expected_magnifi_score")

            print(f"  Case {i+1}/{len(selected_cases)}: {case.get('id', 'unknown')[:20]}...", end=" ", flush=True)

            # Call OpenRouter API
            response, elapsed, input_tok, output_tok = call_openrouter(model_config["id"], report_text)

            total_time += elapsed
            total_input_tokens += input_tok
            total_output_tokens += output_tok

            # Parse response
            parsed = parse_model_response(response)

            result = {
                "case_id": case.get("id"),
                "expected_vai": expected_vai,
                "expected_magnifi": expected_magnifi,
                "predicted_vai": parsed.get("vai_score"),
                "predicted_magnifi": parsed.get("magnifi_score"),
                "time_seconds": elapsed,
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "raw_response": response[:500],
            }
            results.append(result)

            vai_err = abs(parsed.get("vai_score", 0) - expected_vai) if parsed.get("vai_score") and expected_vai else "N/A"
            print(f"VAI err: {vai_err}, Time: {elapsed:.2f}s")

            # Rate limiting
            time.sleep(0.5)

        # Calculate metrics
        metrics = calculate_metrics(results)

        # Calculate cost
        cost = (total_input_tokens / 1000 * model_config["cost_per_1k_input"] +
                total_output_tokens / 1000 * model_config["cost_per_1k_output"])

        all_results[model_name] = {
            "model_id": model_config["id"],
            "results": results,
            "metrics": metrics,
            "total_time_seconds": total_time,
            "avg_time_per_case": total_time / len(results),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_usd": cost,
            "cost_per_case_usd": cost / len(results),
        }

        print(f"\n  Results for {model_name}:")
        print(f"    VAI MAE: {metrics.get('vai_mae', 'N/A'):.2f}" if metrics.get('vai_mae') else "    VAI MAE: N/A")
        print(f"    MAGNIFI MAE: {metrics.get('magnifi_mae', 'N/A'):.2f}" if metrics.get('magnifi_mae') else "    MAGNIFI MAE: N/A")
        print(f"    VAI Accuracy (±1): {metrics.get('vai_accuracy_1pt', 0):.1f}%")
        print(f"    Avg time: {total_time / len(results):.2f}s")
        print(f"    Cost: ${cost:.4f}")

    return all_results


def generate_report(results: dict, output_dir: str):
    """Generate benchmark report."""

    # Save raw results
    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown report
    report = []
    report.append("# MRI Report Parser - Multi-Model Benchmark\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"**Test Cases:** {len(list(results.values())[0]['results']) if results else 0}\n")
    report.append("\n## Summary Table\n")

    # Create comparison table
    report.append("| Model | VAI MAE | MAGNIFI MAE | VAI Acc (±1) | VAI Acc (±2) | Avg Time | Cost/Case |")
    report.append("|-------|---------|-------------|--------------|--------------|----------|-----------|")

    # Sort by VAI MAE (best first)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"].get("vai_mae", 999) if x[1]["metrics"].get("vai_mae") else 999
    )

    for model_name, data in sorted_models:
        m = data["metrics"]
        vai_mae = f"{m.get('vai_mae', 0):.2f}" if m.get('vai_mae') else "N/A"
        mag_mae = f"{m.get('magnifi_mae', 0):.2f}" if m.get('magnifi_mae') else "N/A"
        vai_acc1 = f"{m.get('vai_accuracy_1pt', 0):.1f}%"
        vai_acc2 = f"{m.get('vai_accuracy_2pt', 0):.1f}%"
        avg_time = f"{data['avg_time_per_case']:.2f}s"
        cost = f"${data['cost_per_case_usd']:.4f}"

        report.append(f"| {model_name} | {vai_mae} | {mag_mae} | {vai_acc1} | {vai_acc2} | {avg_time} | {cost} |")

    report.append("\n## Winner Analysis\n")

    # Determine winners
    if sorted_models:
        best_accuracy = sorted_models[0][0]
        fastest = min(results.items(), key=lambda x: x[1]["avg_time_per_case"])[0]
        cheapest = min(results.items(), key=lambda x: x[1]["cost_per_case_usd"])[0]

        report.append(f"- **Best Accuracy (VAI MAE):** {best_accuracy}")
        report.append(f"- **Fastest:** {fastest}")
        report.append(f"- **Most Cost-Effective:** {cheapest}")

    report.append("\n## Model Details\n")

    for model_name, data in results.items():
        report.append(f"\n### {model_name}\n")
        report.append(f"- Model ID: `{data['model_id']}`")
        report.append(f"- Valid cases: {data['metrics'].get('valid_cases', 0)}")
        report.append(f"- Total time: {data['total_time_seconds']:.1f}s")
        report.append(f"- Total cost: ${data['total_cost_usd']:.4f}")
        report.append(f"- Tokens used: {data['total_input_tokens']} in / {data['total_output_tokens']} out")

    report.append("\n## Methodology\n")
    report.append("- Each model received the same MRI reports with identical system prompts")
    report.append("- Temperature set to 0.1 for consistency")
    report.append("- Accuracy measured as percentage of scores within 1 or 2 points of ground truth")
    report.append("- MAE = Mean Absolute Error (lower is better)")
    report.append("- Cost based on OpenRouter pricing as of December 2024")

    # Save report
    with open(os.path.join(output_dir, "benchmark_report.md"), "w") as f:
        f.write("\n".join(report))

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/benchmark_results.json")
    print(f"Report saved to: {output_dir}/benchmark_report.md")


def main():
    # Load test cases
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    test_cases_path = os.path.join(project_root, "data", "parser_validation", "mega_test_cases.json")

    # Try relative path if absolute fails
    if not os.path.exists(test_cases_path):
        test_cases_path = "../parser_validation/mega_test_cases.json"

    with open(test_cases_path, "r") as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    print(f"Loaded {len(test_cases)} test cases")

    # Run benchmark with 30 cases
    results = run_benchmark(test_cases, num_cases=30)

    # Generate report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    generate_report(results, output_dir)


if __name__ == "__main__":
    main()
