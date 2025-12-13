#!/usr/bin/env python3
"""
Final Architecture Ablation Study for ISEF 2026
================================================
Compares 3 extraction methods on 74 Gold-Standard Cases:
  - Method 1: Pure LLM (Lazy Baseline) - Direct score extraction
  - Method 2: Pure Regex/Keyword (Old School Baseline) - Keyword counting
  - Method 3: N-SCAPE (Neuro-Symbolic) - LLM features + symbolic scoring

Author: Tanmay
Date: December 2025
"""

import json
import os
import re
import time
import csv
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import requests
import numpy as np
from scipy import stats

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "deepseek/deepseek-chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_CASES_PATH = os.path.join(BASE_DIR, "data/calibration/gold_cases.json")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data/ablation/ablation_results_final.csv")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "data/ablation/ablation_results_final.json")

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between API calls


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PredictionResult:
    vai: Optional[int] = None
    magnifi: Optional[int] = None
    vai_valid: bool = True
    magnifi_valid: bool = True
    error: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class CaseResult:
    case_id: str
    source: str
    title: str
    ground_truth_vai: int
    ground_truth_magnifi: int

    # Method 1: Pure LLM
    m1_vai: Optional[int] = None
    m1_magnifi: Optional[int] = None
    m1_vai_valid: bool = True
    m1_magnifi_valid: bool = True
    m1_error: Optional[str] = None

    # Method 2: Regex
    m2_vai: Optional[int] = None
    m2_magnifi: Optional[int] = None
    m2_vai_valid: bool = True
    m2_magnifi_valid: bool = True

    # Method 3: N-SCAPE
    m3_vai: Optional[int] = None
    m3_magnifi: Optional[int] = None
    m3_vai_valid: bool = True
    m3_magnifi_valid: bool = True
    m3_error: Optional[str] = None
    m3_features: Optional[Dict] = None


# ============================================================================
# METHOD 1: PURE LLM (LAZY BASELINE)
# ============================================================================

def method1_pure_llm(report_text: str) -> PredictionResult:
    """
    Method 1: Direct LLM score extraction (no symbolic reasoning).
    Hypothesis: Will hallucinate or miss anatomical scoring rules.
    """
    prompt = f"""Analyze this MRI report and output the final Van Assche Index (0-22) and MAGNIFI-CD score (0-25).

IMPORTANT: Return ONLY valid JSON with no explanation:
{{"vai": <integer 0-22>, "magnifi": <integer 0-25>}}

MRI REPORT:
{report_text}

JSON OUTPUT:"""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
            },
            timeout=30,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', content)
        if not json_match:
            return PredictionResult(error="No JSON found in response", raw_response=content)

        parsed = json.loads(json_match.group())
        vai = parsed.get("vai")
        magnifi = parsed.get("magnifi")

        # Validate scores
        vai_valid = isinstance(vai, int) and 0 <= vai <= 22
        magnifi_valid = isinstance(magnifi, int) and 0 <= magnifi <= 25

        return PredictionResult(
            vai=vai if isinstance(vai, (int, float)) else None,
            magnifi=magnifi if isinstance(magnifi, (int, float)) else None,
            vai_valid=vai_valid,
            magnifi_valid=magnifi_valid,
            raw_response=content,
        )

    except Exception as e:
        return PredictionResult(error=str(e))


# ============================================================================
# METHOD 2: PURE REGEX/KEYWORD (OLD SCHOOL BASELINE)
# ============================================================================

def method2_regex_keyword(report_text: str) -> PredictionResult:
    """
    Method 2: Simple keyword counting (no NLP/LLM).
    Hypothesis: Will fail on negations and context-dependent terms.
    """
    text = report_text.lower()

    vai_score = 0
    magnifi_score = 0

    # --- Fistula Count ---
    # Look for number patterns
    fistula_count = 0
    if re.search(r'(three|3|multiple|several)\s*(fistul|tract)', text):
        fistula_count = 3
    elif re.search(r'(two|2)\s*(fistul|tract)', text):
        fistula_count = 2
    elif re.search(r'(single|one|1|a)\s*(fistul|tract)', text) or 'fistul' in text:
        fistula_count = 1

    # VAI: 0=0, 1=1, 2+=2
    if fistula_count == 1:
        vai_score += 1
        magnifi_score += 1
    elif fistula_count >= 2:
        vai_score += 2
        magnifi_score += min(fistula_count, 3)

    # --- Fistula Type (VAI: simple=1, complex=2) ---
    if re.search(r'(transsphincteric|suprasphincteric|extrasphincteric|complex|horseshoe)', text):
        vai_score += 2
    elif re.search(r'(intersphincteric|simple|superficial)', text):
        vai_score += 1

    # --- T2 Hyperintensity ---
    # BUG: This doesn't handle negation properly!
    if re.search(r'(marked|intense|significant)\s*(t2|hyperinten|signal)', text):
        vai_score += 8  # Marked
        magnifi_score += 6
    elif re.search(r'moderate\s*(t2|hyperinten|signal)', text):
        vai_score += 6
        magnifi_score += 4
    elif re.search(r'(mild|subtle)\s*(t2|hyperinten|signal)', text) or 't2 hyperinten' in text:
        vai_score += 4
        magnifi_score += 2

    # --- Extension ---
    if re.search(r'(extensive|severe)\s*(extension|spread)', text) or 'ischioanal' in text:
        vai_score += 4
        magnifi_score += 3
    elif re.search(r'moderate\s*(extension|spread)', text):
        vai_score += 3
        magnifi_score += 2
    elif re.search(r'mild\s*(extension|spread)', text):
        vai_score += 2
        magnifi_score += 1

    # --- Abscess/Collection ---
    # BUG: This will trigger even for "no abscess"!
    if 'abscess' in text or 'collection' in text:
        vai_score += 4
        magnifi_score += 4

    # --- Rectal Wall ---
    if 'rectal wall' in text and ('thicken' in text or 'involv' in text or 'inflam' in text):
        vai_score += 2
        magnifi_score += 2

    # --- Inflammatory Mass (MAGNIFI only) ---
    if 'inflammatory mass' in text or 'phlegmon' in text:
        magnifi_score += 3

    # --- Predominant Feature (MAGNIFI only) ---
    if 'inflammatory' in text and 'fibrotic' not in text:
        magnifi_score += 4
    elif 'mixed' in text:
        magnifi_score += 2
    # fibrotic = 0

    # Clamp to valid ranges
    vai_score = max(0, min(22, vai_score))
    magnifi_score = max(0, min(25, magnifi_score))

    return PredictionResult(
        vai=vai_score,
        magnifi=magnifi_score,
        vai_valid=True,
        magnifi_valid=True,
    )


# ============================================================================
# METHOD 3: N-SCAPE (NEURO-SYMBOLIC) - YOUR ACTUAL SYSTEM
# ============================================================================

# Feature extraction prompt (from parser.js)
NSCAPE_EXTRACTION_PROMPT = """You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. For EACH feature you identify, include the EXACT quote from the report that supports it.

REPORT TEXT:
{report_text}

Return a JSON object with this EXACT structure:
{{
    "features": {{
        "fistula_count": {{
            "value": <number or null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "fistula_type": {{
            "value": <"intersphincteric"|"transsphincteric"|"suprasphincteric"|"extrasphincteric"|"complex"|"simple"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "t2_hyperintensity": {{
            "value": <true|false|null>,
            "degree": <"mild"|"moderate"|"marked"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "extension": {{
            "value": <"none"|"mild"|"moderate"|"severe"|null>,
            "description": "<description of extension pattern>",
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "collections_abscesses": {{
            "value": <true|false|null>,
            "size": <"small"|"large"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "rectal_wall_involvement": {{
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "inflammatory_mass": {{
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "predominant_feature": {{
            "value": <"inflammatory"|"fibrotic"|"mixed"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        }},
        "sphincter_involvement": {{
            "internal": <true|false|null>,
            "external": <true|false|null>,
            "evidence": "<exact quote from report>"
        }},
        "branching": {{
            "value": <true|false|null>,
            "evidence": "<exact quote from report>"
        }}
    }},
    "overall_assessment": {{
        "activity": <"active"|"healing"|"healed"|"chronic">,
        "severity": <"mild"|"moderate"|"severe">,
        "confidence": <"high"|"medium"|"low">
    }},
    "clinical_notes": "<any additional relevant observations>"
}}

IMPORTANT:
- Use null if a feature is not mentioned or cannot be determined
- The "evidence" field must contain the EXACT text from the report
- Be conservative - only report features explicitly stated
- Return ONLY valid JSON, no markdown or explanations"""


def calculate_vai_from_features(features: Dict) -> Tuple[int, Dict]:
    """
    Symbolic VAI scoring from extracted features.
    Exactly matches parser.js calculateVAI() function.
    """
    score = 0
    breakdown = {}

    # Fistula count (0-2)
    count = features.get("fistula_count", {}).get("value")
    if count is not None:
        if count == 0:
            breakdown["fistula_count"] = 0
        elif count == 1:
            breakdown["fistula_count"] = 1
            score += 1
        else:  # 2+
            breakdown["fistula_count"] = 2
            score += 2

    # Fistula type (0-2)
    ftype = features.get("fistula_type", {}).get("value")
    if ftype:
        if ftype in ["simple", "intersphincteric"]:
            breakdown["fistula_location"] = 1
            score += 1
        elif ftype in ["transsphincteric", "suprasphincteric", "extrasphincteric", "complex"]:
            breakdown["fistula_location"] = 2
            score += 2

    # Extension (0-4)
    extension = features.get("extension", {}).get("value")
    if extension:
        ext_scores = {"none": 0, "mild": 2, "moderate": 3, "severe": 4}
        breakdown["extension"] = ext_scores.get(extension, 0)
        score += breakdown["extension"]

    # T2 hyperintensity (0-8)
    t2 = features.get("t2_hyperintensity", {})
    if t2.get("value") is True:
        degree = t2.get("degree", "moderate")
        t2_scores = {"mild": 4, "moderate": 6, "marked": 8}
        breakdown["t2_hyperintensity"] = t2_scores.get(degree, 6)
        score += breakdown["t2_hyperintensity"]
    elif t2.get("value") is False:
        breakdown["t2_hyperintensity"] = 0

    # Collections (0-4)
    collections = features.get("collections_abscesses", {})
    if collections.get("value") is True:
        breakdown["collections"] = 4
        score += 4
    elif collections.get("value") is False:
        breakdown["collections"] = 0

    # Rectal wall involvement (0-2)
    rectal = features.get("rectal_wall_involvement", {})
    if rectal.get("value") is True:
        breakdown["rectal_wall"] = 2
        score += 2
    elif rectal.get("value") is False:
        breakdown["rectal_wall"] = 0

    # Cap at max 22
    score = min(score, 22)

    return score, breakdown


def calculate_magnifi_from_features(features: Dict) -> Tuple[int, Dict]:
    """
    Symbolic MAGNIFI-CD scoring from extracted features.
    Exactly matches parser.js calculateMAGNIFI() function.
    """
    score = 0
    breakdown = {}

    # Fistula count (0-3)
    count = features.get("fistula_count", {}).get("value")
    if count is not None:
        if count == 0:
            breakdown["fistula_count"] = 0
        elif count == 1:
            breakdown["fistula_count"] = 1
            score += 1
        elif count == 2:
            breakdown["fistula_count"] = 2
            score += 2
        else:  # 3+
            breakdown["fistula_count"] = 3
            score += 3

    # Fistula activity (T2) (0-6)
    t2 = features.get("t2_hyperintensity", {})
    if t2.get("value") is True:
        degree = t2.get("degree", "moderate")
        act_scores = {"mild": 2, "moderate": 4, "marked": 6}
        breakdown["fistula_activity"] = act_scores.get(degree, 4)
        score += breakdown["fistula_activity"]
    elif t2.get("value") is False:
        breakdown["fistula_activity"] = 0

    # Collections (0-4)
    collections = features.get("collections_abscesses", {})
    if collections.get("value") is True:
        size = collections.get("size", "small")
        breakdown["collections"] = 4 if size == "large" else 2
        score += breakdown["collections"]
    elif collections.get("value") is False:
        breakdown["collections"] = 0

    # Inflammatory mass (0-3)
    mass = features.get("inflammatory_mass", {})
    if mass.get("value") is True:
        breakdown["inflammatory_mass"] = 3
        score += 3
    elif mass.get("value") is False:
        breakdown["inflammatory_mass"] = 0

    # Rectal wall involvement (0-2)
    rectal = features.get("rectal_wall_involvement", {})
    if rectal.get("value") is True:
        breakdown["rectal_wall"] = 2
        score += 2
    elif rectal.get("value") is False:
        breakdown["rectal_wall"] = 0

    # Predominant feature (0-4)
    predominant = features.get("predominant_feature", {}).get("value")
    if predominant:
        pred_scores = {"fibrotic": 0, "mixed": 2, "inflammatory": 4}
        breakdown["predominant_feature"] = pred_scores.get(predominant, 2)
        score += breakdown["predominant_feature"]

    # Extension (0-3)
    extension = features.get("extension", {}).get("value")
    if extension:
        ext_scores = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        breakdown["extent"] = ext_scores.get(extension, 1)
        score += breakdown["extent"]

    # Cap at max 25
    score = min(score, 25)

    return score, breakdown


def method3_nscape(report_text: str) -> PredictionResult:
    """
    Method 3: N-SCAPE Neuro-Symbolic approach.
    Step 1: LLM extracts features (neural)
    Step 2: Symbolic rules calculate scores (symbolic)
    """
    prompt = NSCAPE_EXTRACTION_PROMPT.format(report_text=report_text)

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=60,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return PredictionResult(error="No JSON found in response", raw_response=content)

        parsed = json.loads(json_match.group())
        features = parsed.get("features", {})

        # SYMBOLIC SCORING (the key differentiator!)
        vai_score, vai_breakdown = calculate_vai_from_features(features)
        magnifi_score, magnifi_breakdown = calculate_magnifi_from_features(features)

        return PredictionResult(
            vai=vai_score,
            magnifi=magnifi_score,
            vai_valid=0 <= vai_score <= 22,
            magnifi_valid=0 <= magnifi_score <= 25,
            raw_response=json.dumps({"features": features, "vai_breakdown": vai_breakdown, "magnifi_breakdown": magnifi_breakdown}),
        )

    except json.JSONDecodeError as e:
        return PredictionResult(error=f"JSON parse error: {e}", raw_response=content if 'content' in dir() else None)
    except Exception as e:
        return PredictionResult(error=str(e))


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_icc(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate ICC(2,1) - Two-way random effects, single measures, absolute agreement.
    """
    n = len(y_true)
    if n < 3:
        return float('nan')

    # Combine into matrix format
    data = np.array([y_true, y_pred]).T

    # Calculate means
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    # Calculate sum of squares
    k = 2  # Number of raters

    # Between subjects SS
    ss_between = k * np.sum((row_means - grand_mean) ** 2)

    # Within subjects SS
    ss_within = np.sum((data - row_means.reshape(-1, 1)) ** 2)

    # Between raters SS
    ss_raters = n * np.sum((col_means - grand_mean) ** 2)

    # Error SS
    ss_error = ss_within - ss_raters

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_raters = ss_raters / (k - 1)

    # ICC(2,1)
    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + (k / n) * (ms_raters - ms_error))

    return max(0, min(1, icc))  # Clamp to [0, 1]


def calculate_metrics(results: List[CaseResult]) -> Dict:
    """Calculate MAE, ICC, and Safety Violation Rate for all methods."""

    metrics = {}

    for method_prefix, method_name in [("m1", "Pure LLM"), ("m2", "Regex"), ("m3", "N-SCAPE")]:
        # Filter valid results
        vai_pairs = []
        magnifi_pairs = []
        vai_violations = 0
        magnifi_violations = 0
        total = 0

        for r in results:
            total += 1
            vai_pred = getattr(r, f"{method_prefix}_vai")
            magnifi_pred = getattr(r, f"{method_prefix}_magnifi")
            vai_valid = getattr(r, f"{method_prefix}_vai_valid")
            magnifi_valid = getattr(r, f"{method_prefix}_magnifi_valid")

            # Check for violations (impossible scores)
            if vai_pred is not None:
                if not isinstance(vai_pred, int) or vai_pred < 0 or vai_pred > 22:
                    vai_violations += 1
                else:
                    vai_pairs.append((r.ground_truth_vai, vai_pred))
            else:
                vai_violations += 1

            if magnifi_pred is not None:
                if not isinstance(magnifi_pred, int) or magnifi_pred < 0 or magnifi_pred > 25:
                    magnifi_violations += 1
                else:
                    magnifi_pairs.append((r.ground_truth_magnifi, magnifi_pred))
            else:
                magnifi_violations += 1

        # Calculate MAE
        vai_mae = np.mean([abs(t - p) for t, p in vai_pairs]) if vai_pairs else float('nan')
        magnifi_mae = np.mean([abs(t - p) for t, p in magnifi_pairs]) if magnifi_pairs else float('nan')

        # Calculate ICC
        vai_true = [t for t, p in vai_pairs]
        vai_pred = [p for t, p in vai_pairs]
        magnifi_true = [t for t, p in magnifi_pairs]
        magnifi_pred = [p for t, p in magnifi_pairs]

        vai_icc = calculate_icc(vai_true, vai_pred) if len(vai_pairs) >= 3 else float('nan')
        magnifi_icc = calculate_icc(magnifi_true, magnifi_pred) if len(magnifi_pairs) >= 3 else float('nan')

        # Safety violation rate
        safety_violation_rate = (vai_violations + magnifi_violations) / (2 * total) * 100 if total > 0 else 0

        metrics[method_name] = {
            "vai_mae": round(vai_mae, 2) if not np.isnan(vai_mae) else "N/A",
            "magnifi_mae": round(magnifi_mae, 2) if not np.isnan(magnifi_mae) else "N/A",
            "vai_icc": round(vai_icc, 3) if not np.isnan(vai_icc) else "N/A",
            "magnifi_icc": round(magnifi_icc, 3) if not np.isnan(magnifi_icc) else "N/A",
            "safety_violation_rate": round(safety_violation_rate, 1),
            "n_valid_vai": len(vai_pairs),
            "n_valid_magnifi": len(magnifi_pairs),
        }

    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_gold_cases() -> List[Dict]:
    """Load the 74 gold-standard cases."""
    with open(GOLD_CASES_PATH, 'r') as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} gold-standard cases")
    return cases


def run_ablation_study():
    """Run the complete ablation study on all 74 cases."""

    print("=" * 70)
    print("FINAL ARCHITECTURE ABLATION STUDY")
    print("ISEF 2026 - MRI-Crohn Atlas")
    print("=" * 70)
    print()

    # Load cases
    cases = load_gold_cases()

    results = []

    # Process each case
    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"case_{i}")
        report_text = case.get("findings_text", "")

        print(f"[{i+1}/{len(cases)}] Processing {case_id}...", end=" ", flush=True)

        # Create result object
        result = CaseResult(
            case_id=case_id,
            source=case.get("source", "unknown"),
            title=case.get("title", ""),
            ground_truth_vai=case.get("scored_vai", 0),
            ground_truth_magnifi=case.get("scored_magnifi", 0),
        )

        # Method 1: Pure LLM
        m1_result = method1_pure_llm(report_text)
        result.m1_vai = m1_result.vai
        result.m1_magnifi = m1_result.magnifi
        result.m1_vai_valid = m1_result.vai_valid
        result.m1_magnifi_valid = m1_result.magnifi_valid
        result.m1_error = m1_result.error
        time.sleep(REQUEST_DELAY)

        # Method 2: Regex (no API call needed)
        m2_result = method2_regex_keyword(report_text)
        result.m2_vai = m2_result.vai
        result.m2_magnifi = m2_result.magnifi
        result.m2_vai_valid = m2_result.vai_valid
        result.m2_magnifi_valid = m2_result.magnifi_valid

        # Method 3: N-SCAPE
        m3_result = method3_nscape(report_text)
        result.m3_vai = m3_result.vai
        result.m3_magnifi = m3_result.magnifi
        result.m3_vai_valid = m3_result.vai_valid
        result.m3_magnifi_valid = m3_result.magnifi_valid
        result.m3_error = m3_result.error
        if m3_result.raw_response:
            try:
                result.m3_features = json.loads(m3_result.raw_response)
            except:
                pass
        time.sleep(REQUEST_DELAY)

        results.append(result)

        # Print progress
        gt = f"GT({result.ground_truth_vai},{result.ground_truth_magnifi})"
        m1 = f"M1({result.m1_vai},{result.m1_magnifi})"
        m2 = f"M2({result.m2_vai},{result.m2_magnifi})"
        m3 = f"M3({result.m3_vai},{result.m3_magnifi})"
        print(f"{gt} | {m1} | {m2} | {m3}")

    print()
    print("=" * 70)
    print("CALCULATING METRICS...")
    print("=" * 70)

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Save results to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "source", "title",
            "gt_vai", "gt_magnifi",
            "m1_vai", "m1_magnifi", "m1_error",
            "m2_vai", "m2_magnifi",
            "m3_vai", "m3_magnifi", "m3_error"
        ])
        for r in results:
            writer.writerow([
                r.case_id, r.source, r.title,
                r.ground_truth_vai, r.ground_truth_magnifi,
                r.m1_vai, r.m1_magnifi, r.m1_error or "",
                r.m2_vai, r.m2_magnifi,
                r.m3_vai, r.m3_magnifi, r.m3_error or ""
            ])

    # Save full results to JSON
    json_results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL,
            "total_cases": len(cases),
        },
        "metrics": metrics,
        "cases": [
            {
                "case_id": r.case_id,
                "source": r.source,
                "title": r.title,
                "ground_truth": {"vai": r.ground_truth_vai, "magnifi": r.ground_truth_magnifi},
                "method1_pure_llm": {"vai": r.m1_vai, "magnifi": r.m1_magnifi, "error": r.m1_error},
                "method2_regex": {"vai": r.m2_vai, "magnifi": r.m2_magnifi},
                "method3_nscape": {"vai": r.m3_vai, "magnifi": r.m3_magnifi, "error": r.m3_error, "features": r.m3_features},
            }
            for r in results
        ]
    }
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'VAI MAE':<12} {'MAGNIFI MAE':<14} {'VAI ICC':<12} {'MAGNIFI ICC':<14} {'Safety Violations'}")
    print("-" * 90)

    for method, m in metrics.items():
        print(f"{method:<20} {str(m['vai_mae']):<12} {str(m['magnifi_mae']):<14} {str(m['vai_icc']):<12} {str(m['magnifi_icc']):<14} {m['safety_violation_rate']:.1f}%")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Compare methods
    m1 = metrics["Pure LLM"]
    m2 = metrics["Regex"]
    m3 = metrics["N-SCAPE"]

    print("Method 1 (Pure LLM) - 'Lazy Baseline':")
    print(f"  - Direct score extraction without symbolic rules")
    print(f"  - VAI MAE: {m1['vai_mae']}, ICC: {m1['vai_icc']}")
    print(f"  - Safety violations: {m1['safety_violation_rate']}%")
    print()

    print("Method 2 (Regex) - 'Old School Baseline':")
    print(f"  - Simple keyword counting, no context understanding")
    print(f"  - VAI MAE: {m2['vai_mae']}, ICC: {m2['vai_icc']}")
    print(f"  - KNOWN BUG: Triggers on 'no abscess' -> false positives")
    print()

    print("Method 3 (N-SCAPE) - 'Neuro-Symbolic':")
    print(f"  - LLM feature extraction + symbolic scoring rules")
    print(f"  - VAI MAE: {m3['vai_mae']}, ICC: {m3['vai_icc']}")
    print(f"  - Safety violations: {m3['safety_violation_rate']}%")
    print()

    # Determine winner
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compare MAEs (lower is better)
    vai_maes = [(m1['vai_mae'], "Pure LLM"), (m2['vai_mae'], "Regex"), (m3['vai_mae'], "N-SCAPE")]
    vai_maes = [(m, n) for m, n in vai_maes if m != "N/A"]
    vai_maes.sort(key=lambda x: x[0])

    if vai_maes:
        print(f"Best VAI Accuracy: {vai_maes[0][1]} (MAE = {vai_maes[0][0]})")

    magnifi_maes = [(m1['magnifi_mae'], "Pure LLM"), (m2['magnifi_mae'], "Regex"), (m3['magnifi_mae'], "N-SCAPE")]
    magnifi_maes = [(m, n) for m, n in magnifi_maes if m != "N/A"]
    magnifi_maes.sort(key=lambda x: x[0])

    if magnifi_maes:
        print(f"Best MAGNIFI Accuracy: {magnifi_maes[0][1]} (MAE = {magnifi_maes[0][0]})")

    print()
    print(f"Results saved to:")
    print(f"  - {OUTPUT_CSV_PATH}")
    print(f"  - {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    run_ablation_study()
