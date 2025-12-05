"""
New Papers Crosswalk Data Extractor
====================================
Extracts VAI/MAGNIFI-CD crosswalk data from the newly added papers.
Focuses on: paired scores, correlations, patient-level data, sample sizes, fibrosis scores.
"""

import os
import json
import asyncio
import logging
import aiohttp
import pymupdf4llm
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Configuration
PAPERS_DIR = Path(__file__).parent.parent.parent / "data" / "papers"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

# New papers to extract (from tuff papers folder)
NEW_PAPERS = [
    "Development_and_Validation_of_a_Pediatric_MRI-Based_Perianal_Crohn_Disease_PEMPAC_Index-A_Report_from_the_ImageKids_Study.pdf",
    "Efficacy_and_Safety_of_Upadacitinib_for_Perianal_Fistulizing_Crohns_Disease-_A_Post_Hoc_Analysis_of_3_Phase_3_Trials.pdf",
    "Efficacy_and_safety_of_darvadstrocel_treatment_in_patients_with_complex_perianal_fistulas_ADMIRE-CD_II.pdf",
    "Evaluating_the_effectiveness_of_infliximab_on_perianal_fistulizing_Crohns_disease_by_magnetic_resonance_imaging.pdf",
    "Expanded_allogeneic_adipose-derived_mesenchymal_stem_cells_Cx601_for_complex_perianal_fi_stulas_in_Crohns_disease-_a_phase_3_randomised_double-blind_controlled_trial.pdf",
    "Fistulizing_Perianal_Crohns_Disease-_Contrast-enhanced_Magnetic_Resonance_Imaging_Assessment_at_1_Year_on_Maintenance_Anti-TNF-alpha_Therapy.pdf",
    "Follow-up_Study_to_Evaluate_the_Long-term_Safety_and_Efficacy_of_Darvadstrocel_Mesenchymal_Stem_Cell_Treatment_in_Patients_With_Perianal_Fistulizing_Crohns_Disease-_ADMIRE-CD_Phase_3_Randomized_Controlled_Trial.pdf",
    "Higher_Anti-tumor_Necrosis_Factor-a_Levels_Correlate_With_Improved_Radiologic_Outcomes_in_Crohns_Perianal_Fistulas.pdf",
    "MRI_predictors_of_treatment_response_for_perianal_fistulizing_Crohn_disease_in_children_and_young_adults.pdf",
    "Meima-van_Praag_PISA-II.pdf",
    "Mesenchymal_stem_cell_therapy_for_therapy_refractory_complex_Crohns_perianal_fistulas-_a_case_series.pdf",
    "Multiple_-_Efficacy_and_Safety_of_Filgotinib_for_the_Treatment_of_Perianal_Fistulising_Crohns_Disease_DIVERGENCE_2_.pdf",
    "Multiple_-_Magnetic_resonance_imaging_may_predict_deep_remission_in_patients_with_perianal_fistulizing_Crohns_disease.pdf",
    "Multiple_-_TOpCLASS_Expert_Consensus_Classification_of_Perianal_Fistulising_Crohns_Disease-_A_Real-world_Application.pdf",
    "Perianal_Imaging_in_Crohn_Disease.pdf",
    "Ustekinumab_Promotes_Radiological_Fistula_Healing_in_Perianal_Fistulizing_Crohns_Disease.pdf",
]

PRIMARY_MODEL = "deepseek/deepseek-v3.2"
MAX_TOKENS = 100000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CrosswalkDataPoint(BaseModel):
    """Individual data point for crosswalk."""
    study: str
    patient_state: str = Field(..., description="active, healed, responder, non-responder, baseline, followup")
    vai_score: Optional[float] = None
    magnificd_score: Optional[float] = None
    fibrosis_score: Optional[float] = None
    n_patients: Optional[int] = None
    notes: Optional[str] = None


class PaperExtraction(BaseModel):
    """Structured extraction from a paper."""
    filename: str
    paper_title: Optional[str] = None
    year: Optional[int] = None
    study_type: Optional[str] = None
    total_patients: Optional[int] = None

    # Key scoring systems mentioned
    has_vai: bool = False
    has_magnificd: bool = False
    has_mvai: bool = False
    has_pempac: bool = False
    has_fibrosis: bool = False

    # Score summaries (medians/means with context)
    vai_scores: List[Dict[str, Any]] = Field(default_factory=list)
    magnificd_scores: List[Dict[str, Any]] = Field(default_factory=list)
    fibrosis_scores: List[Dict[str, Any]] = Field(default_factory=list)

    # Correlations between any scoring systems
    correlations: List[Dict[str, Any]] = Field(default_factory=list)

    # Crosswalk data points (usable for regression)
    crosswalk_datapoints: List[Dict[str, Any]] = Field(default_factory=list)

    # Clinical thresholds
    thresholds: List[Dict[str, Any]] = Field(default_factory=list)

    # Key quotes
    key_findings: List[str] = Field(default_factory=list)

    extraction_confidence: float = 0.5


def get_extraction_prompt() -> str:
    return """You are extracting MRI scoring data for perianal Crohn's disease from a research paper.

SCORING SYSTEMS TO LOOK FOR:
1. **Van Assche Index (VAI)** - Range 0-22, measures fistula severity
2. **MAGNIFI-CD** - Range 0-25, modern activity index
3. **Modified VAI (mVAI)** - Simplified version of VAI
4. **PEMPAC** - Pediatric index, correlates with adult VAI (r=0.93)
5. **Fibrosis Score** - Range 0-6, measures healing/scarring

EXTRACT THE FOLLOWING:

1. **PAPER METADATA**
   - Title, year, study type (RCT, cohort, case series, etc.)
   - Total patient count with perianal fistulas

2. **SCORING SYSTEMS USED**
   - Which indices are mentioned/used
   - Set has_vai, has_magnificd, has_mvai, has_pempac, has_fibrosis accordingly

3. **SCORE VALUES** (vai_scores, magnificd_scores, fibrosis_scores)
   For EACH score reported, extract:
   - value (mean or median)
   - iqr or sd (if available)
   - n_patients
   - context: "baseline", "followup", "responders", "non-responders", "healed", "active", "week12", "week24", etc.
   - treatment: what treatment the patients received

   Example: {"value": 12, "iqr": "8-16", "n": 30, "context": "baseline", "treatment": "infliximab"}

4. **CORRELATIONS** (critical for crosswalk)
   - Correlation coefficients (r, rho, ICC) between ANY scoring systems
   - AUROC values for predicting outcomes
   - p-values and confidence intervals

   Example: {"system_a": "VAI", "system_b": "MAGNIFI-CD", "correlation": 0.85, "type": "Pearson r", "p": "<0.001", "n": 65}

5. **CROSSWALK DATA POINTS** (most important!)
   If the SAME patients have scores on multiple systems, extract paired data:
   - patient_state: "active", "healed", "responder", etc.
   - vai_score: mean/median VAI
   - magnificd_score: mean/median MAGNIFI-CD
   - fibrosis_score: if available
   - n_patients: how many patients in this group

   Example: {"patient_state": "active", "vai_score": 12, "magnificd_score": 14, "fibrosis_score": 2, "n_patients": 30}

6. **CLINICAL THRESHOLDS**
   - What scores define "healed" vs "active"
   - Response vs remission cutoffs
   - AUROC for clinical outcomes

   Example: {"index": "MAGNIFI-CD", "threshold": 6, "meaning": "radiological remission", "sensitivity": 0.87, "specificity": 0.91}

7. **KEY FINDINGS**
   - Any statements comparing VAI and MAGNIFI-CD
   - Insights about when to use each index
   - Fibrosis as predictor of outcomes

OUTPUT FORMAT:
Return a JSON object matching this schema:
{
  "filename": "...",
  "paper_title": "...",
  "year": 2024,
  "study_type": "RCT",
  "total_patients": 212,
  "has_vai": true,
  "has_magnificd": true,
  "has_mvai": false,
  "has_pempac": false,
  "has_fibrosis": true,
  "vai_scores": [...],
  "magnificd_scores": [...],
  "fibrosis_scores": [...],
  "correlations": [...],
  "crosswalk_datapoints": [...],
  "thresholds": [...],
  "key_findings": [...],
  "extraction_confidence": 0.85
}

IMPORTANT:
- Extract ALL numerical values even if uncertain about context
- For figures/tables you can't see clearly, estimate based on text descriptions
- If no VAI/MAGNIFI-CD data found, still return the structure with empty arrays
- Set extraction_confidence based on how much relevant data was found (0-1)
"""


async def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using pymupdf4llm."""
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        # Truncate if too long
        if len(md_text) > 150000:
            md_text = md_text[:150000] + "\n\n[TRUNCATED - Document too long]"
        return md_text
    except Exception as e:
        logging.error(f"Error extracting {pdf_path.name}: {e}")
        return None


async def call_llm(text: str, filename: str) -> Optional[Dict]:
    """Call LLM API for extraction."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = get_extraction_prompt()

    payload = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Extract crosswalk data from this paper:\n\nFilename: {filename}\n\n{text}"}
        ],
        "temperature": 0.1,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"}
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=180) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logging.error(f"API error for {filename}: {error}")
                    return None

                data = await resp.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON response
                result = json.loads(content)
                result["filename"] = filename
                return result

    except Exception as e:
        logging.error(f"Error calling LLM for {filename}: {e}")
        return None


async def process_paper(filename: str) -> Optional[Dict]:
    """Process a single paper."""
    pdf_path = PAPERS_DIR / filename

    if not pdf_path.exists():
        logging.warning(f"File not found: {filename}")
        return None

    logging.info(f"📄 Processing: {filename}")

    # Extract text
    text = await extract_text_from_pdf(pdf_path)
    if not text:
        return None

    # Call LLM
    result = await call_llm(text, filename)
    return result


async def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("NEW PAPERS CROSSWALK DATA EXTRACTION")
    print("=" * 60)
    print(f"\n📚 Processing {len(NEW_PAPERS)} new papers...")

    results = []

    for i, filename in enumerate(NEW_PAPERS, 1):
        print(f"\n[{i}/{len(NEW_PAPERS)}] {filename[:50]}...")
        result = await process_paper(filename)

        if result:
            results.append(result)

            # Print quick summary
            n = result.get("total_patients", "?")
            vai = "✓" if result.get("has_vai") else "✗"
            mag = "✓" if result.get("has_magnificd") else "✗"
            fib = "✓" if result.get("has_fibrosis") else "✗"
            xw = len(result.get("crosswalk_datapoints", []))

            print(f"   → n={n}, VAI:{vai} MAGNIFI-CD:{mag} Fibrosis:{fib}, Crosswalk points: {xw}")
        else:
            print(f"   → FAILED to extract")

        # Small delay to avoid rate limiting
        await asyncio.sleep(1)

    # Save results
    output_file = OUTPUT_DIR / "new_papers_extraction.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully extracted: {len(results)}/{len(NEW_PAPERS)}")

    # Calculate totals
    total_patients = sum(r.get("total_patients", 0) or 0 for r in results)
    vai_papers = sum(1 for r in results if r.get("has_vai"))
    mag_papers = sum(1 for r in results if r.get("has_magnificd"))
    total_crosswalk = sum(len(r.get("crosswalk_datapoints", [])) for r in results)
    total_correlations = sum(len(r.get("correlations", [])) for r in results)

    print(f"👥 Total patients: {total_patients}")
    print(f"📊 Papers with VAI: {vai_papers}")
    print(f"📊 Papers with MAGNIFI-CD: {mag_papers}")
    print(f"🔗 Crosswalk data points: {total_crosswalk}")
    print(f"📈 Correlations found: {total_correlations}")
    print(f"\n📁 Output: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
