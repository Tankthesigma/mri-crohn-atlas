"""
Deep Crosswalk Extractor for VAI ↔ MAGNIFI-CD Mapping
=====================================================

Specialized extraction script for the 3 key papers that contain both
Van Assche Index and MAGNIFI-CD data. Uses an enhanced prompt focused
on finding paired scores, correlation data, and component mappings.

Usage:
    python deep_crosswalk_extractor.py
"""

import os
import re
import json
import asyncio
import logging
import aiohttp
import tiktoken
import pymupdf4llm
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError

# ============ CONFIGURATION ============
PAPERS_DIR = Path(__file__).parent.parent.parent / "data" / "papers"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Run: export OPENROUTER_API_KEY='your-key'")

# Target papers with both VAI and MAGNIFI-CD
TARGET_PAPERS = [
    # Newly downloaded open access papers with crosswalk data
    "van_Rijn_2022_Fibrosis_MAGNIFI-CD.pdf",              # n=50, MAGNIFI-CD + Fibrosis, AUC=0.95
    "Beek_2024_External_Validation_MAGNIFI-CD.pdf",       # n=65, External validation, ICC=0.87
    "Radiological_Outcomes_Systematic_Review_2020.pdf",   # Systematic review, meta-analysis
    "MRI_Role_Review_2018.pdf",                           # Comprehensive review of scoring systems
]

PRIMARY_MODEL = "deepseek/deepseek-v3.2"
MAX_TOKENS = 100000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============ ENHANCED SCHEMA ============

class PairedScore(BaseModel):
    """A single patient's or cohort's paired VAI and MAGNIFI-CD scores."""
    patient_id: Optional[str] = Field(None, description="Patient ID if available")
    vai_score: Optional[float] = Field(None, description="Van Assche Index score")
    vai_inflammatory: Optional[float] = Field(None, description="VAI inflammatory subscore if reported")
    magnificd_score: Optional[float] = Field(None, description="MAGNIFI-CD score")
    fibrosis_score: Optional[float] = Field(None, description="Fibrosis score (0-3 or 0-6)")
    timepoint: Optional[str] = Field(None, description="baseline, week12, healed, etc.")
    notes: Optional[str] = Field(None)


class ComponentMapping(BaseModel):
    """Mapping between individual components of VAI and MAGNIFI-CD."""
    vai_component: Optional[str] = Field(None, description="VAI component name")
    magnificd_component: Optional[str] = Field(None, description="Corresponding MAGNIFI-CD component")
    relationship: Optional[str] = Field(None, description="How they relate (same, similar, different)")
    notes: Optional[str] = Field(None)


class ScatterplotData(BaseModel):
    """Data from scatterplots showing VAI vs MAGNIFI-CD."""
    figure_reference: str = Field(..., description="Figure number/reference")
    x_axis: str = Field(..., description="What's on x-axis")
    y_axis: str = Field(..., description="What's on y-axis")
    n_datapoints: Optional[int] = Field(None)
    correlation_shown: Optional[float] = Field(None)
    regression_line: Optional[str] = Field(None, description="Regression equation if shown")
    estimated_points: List[dict] = Field(default_factory=list, description="Approximate (x,y) values from plot")


class TableData(BaseModel):
    """Data from tables showing scoring system results."""
    table_reference: str = Field(..., description="Table number/reference")
    columns: List[str] = Field(default_factory=list)
    has_vai: bool = False
    has_magnificd: bool = False
    has_fibrosis: bool = False
    row_summaries: List[str] = Field(default_factory=list, description="Summary of each row's data")


class CorrelationData(BaseModel):
    """Correlation between any scoring systems."""
    system_a: str
    system_b: str
    correlation_coefficient: Optional[float] = Field(None, ge=-1, le=1)
    correlation_type: Optional[str] = Field(None, description="pearson, spearman, ICC, kappa")
    p_value: Optional[float] = Field(None)
    confidence_interval: Optional[str] = Field(None)
    n_patients: Optional[int] = Field(None)
    context: Optional[str] = Field(None, description="What was being compared")


class DeepCrosswalkExtraction(BaseModel):
    """Deep extraction schema focused on VAI ↔ MAGNIFI-CD crosswalk data."""

    filename: str
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Paper context
    paper_title: Optional[str] = None
    publication_year: Optional[int] = None
    study_type: Optional[str] = None
    n_patients: Optional[int] = None

    # KEY: Paired scores on same patients
    has_paired_vai_magnificd_scores: bool = False
    paired_scores: List[PairedScore] = Field(default_factory=list)

    # Group-level score summaries
    vai_summary: Optional[dict] = Field(None, description="mean, median, IQR, range for VAI")
    magnificd_summary: Optional[dict] = Field(None, description="mean, median, IQR, range for MAGNIFI-CD")
    fibrosis_summary: Optional[dict] = Field(None, description="mean, median, IQR, range for Fibrosis Score")

    # Correlation data
    correlations: List[CorrelationData] = Field(default_factory=list)
    vai_magnificd_correlation: Optional[float] = Field(None, description="Direct VAI-MAGNIFI-CD correlation if reported")

    # Visual data sources
    scatterplots: List[ScatterplotData] = Field(default_factory=list)
    tables_with_scores: List[TableData] = Field(default_factory=list)

    # Component-level analysis
    component_mappings: List[ComponentMapping] = Field(default_factory=list)

    # Index definitions found
    vai_components_described: List[str] = Field(default_factory=list)
    magnificd_components_described: List[str] = Field(default_factory=list)

    # Fibrosis data (key for our model)
    fibrosis_definition: Optional[str] = None
    fibrosis_scoring_method: Optional[str] = None

    # Any explicit conversion or comparison statements
    conversion_statements: List[str] = Field(default_factory=list, description="Direct quotes about VAI vs MAGNIFI-CD")

    # Extraction quality
    extraction_confidence: float = Field(0.5, ge=0, le=1)
    key_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    raw_evidence: List[str] = Field(default_factory=list, description="Key supporting quotes (max 15)")


def get_deep_extraction_prompt() -> str:
    """Enhanced prompt for deep crosswalk extraction."""
    schema_json = DeepCrosswalkExtraction.model_json_schema()

    return f"""You are a specialized medical data extraction system focused on finding crosswalk data between MRI scoring systems for perianal Crohn's disease.

YOUR CRITICAL MISSION:
Find ANY data that could help map between Van Assche Index (VAI) and MAGNIFI-CD scoring systems.

SEARCH FOR THESE SPECIFIC DATA TYPES:

1. **PAIRED SCORES** (Most valuable!)
   - Individual patient scores on BOTH VAI and MAGNIFI-CD
   - Cohort means/medians for both systems at same timepoint
   - Pre/post treatment scores for both indices
   - Extract ALL numerical values you find

2. **SCATTERPLOTS AND FIGURES**
   - Any figure showing VAI vs MAGNIFI-CD
   - Describe axis labels, approximate data points
   - Note any regression lines or correlation values shown
   - Even if you can't read exact values, describe what the plot shows

3. **TABLES**
   - Tables showing scores for multiple indices
   - Columns containing both VAI and MAGNIFI-CD data
   - Patient-level or group-level summaries
   - Extract actual numbers if possible

4. **CORRELATIONS**
   - Any correlation coefficient between scoring systems
   - Include correlations with clinical outcomes that might help (e.g., both correlated with healing)
   - Spearman, Pearson, ICC, kappa values
   - P-values and confidence intervals

5. **COMPONENT MAPPINGS**
   - How individual VAI components relate to MAGNIFI-CD components
   - Both measure: fistula number, T2 signal, abscesses, rectal wall involvement
   - Any discussion of which components are equivalent

6. **FIBROSIS DATA**
   - Fibrosis score definitions and ranges
   - How fibrosis relates to either index
   - This is KEY for our conversion model

7. **CONVERSION STATEMENTS**
   - Any explicit comparison of the two indices
   - Statements about when to use one vs the other
   - Discussion of advantages/disadvantages
   - Any mention of why no direct conversion exists

8. **PATIENT-LEVEL DATA** (Critical!)
   - Individual patient data tables (not just group means)
   - Patient ID with corresponding scores
   - Any supplementary tables or appendices mentioned
   - Sample sizes for EACH comparison made

9. **SUPPLEMENTARY DATA**
   - Links to online supplementary materials
   - References to appendices with raw data
   - DOIs or URLs for additional data
   - Any mention of data availability statements

10. **INDEX DEVELOPMENT/VALIDATION**
    - How the scoring system was developed
    - Inter-rater reliability (ICC, kappa) for EACH component
    - Validation cohort sizes and demographics
    - Score distributions (means, SDs, ranges, percentiles)

EXTRACTION RULES:
- Extract EVERY relevant number, even if uncertain
- For figures you cannot read exactly, estimate ranges
- Quote key sentences verbatim in raw_evidence
- Set has_paired_vai_magnificd_scores=true only if same patients were scored on both
- List ALL correlations found, even between subscores or with clinical outcomes
- Be exhaustive - we need every piece of relevant data

SCHEMA:
{json.dumps(schema_json, indent=2)}

Return ONLY valid JSON matching this schema."""


def parse_pdf(file_path: Path) -> str:
    """Convert PDF to markdown."""
    try:
        md_text = pymupdf4llm.to_markdown(str(file_path))

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(md_text)

        if len(tokens) > MAX_TOKENS:
            logging.warning(f"{file_path.name}: Truncating from {len(tokens)} to {MAX_TOKENS} tokens")
            md_text = enc.decode(tokens[:MAX_TOKENS])

        return md_text
    except Exception as e:
        logging.error(f"Failed to parse {file_path.name}: {e}")
        return ""


async def extract_crosswalk_data(markdown: str, filename: str, session: aiohttp.ClientSession) -> Optional[DeepCrosswalkExtraction]:
    """Extract crosswalk data using enhanced prompt."""

    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY not set!")
        return None

    system_prompt = get_deep_extraction_prompt()

    payload = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Perform DEEP extraction on this paper, searching exhaustively for VAI ↔ MAGNIFI-CD crosswalk data.

Filename: {filename}

PAPER CONTENT:
{markdown}

Remember: Extract EVERY piece of data that could help map between Van Assche Index and MAGNIFI-CD, including:
- Paired scores, correlations, scatterplots, tables
- Component mappings, fibrosis data
- Any quotes comparing the two indices

Return comprehensive JSON."""}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 8000
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status != 200:
                error = await response.text()
                logging.error(f"API error for {filename}: {error[:200]}")
                return None

            result = await response.json()
            content = result['choices'][0]['message']['content']

            # Handle code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Try to fix common JSON issues
            # Sometimes LLM adds trailing content after the JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to find the JSON object boundaries
                start = content.find('{')
                if start >= 0:
                    # Find matching closing brace
                    depth = 0
                    end = start
                    for i, c in enumerate(content[start:], start):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    content = content[start:end]
                    data = json.loads(content)
                else:
                    raise

            data['filename'] = filename

            # Log token usage
            usage = result.get('usage', {})
            logging.info(f"{filename}: {usage.get('prompt_tokens', 0)} in, {usage.get('completion_tokens', 0)} out")

            return DeepCrosswalkExtraction(**data)

    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error for {filename}: {e}")
        # Log the problematic content for debugging
        logging.debug(f"Content was: {content[:500]}...")
        return None
    except ValidationError as e:
        logging.error(f"Validation error for {filename}: {e}")
        return None
    except Exception as e:
        logging.error(f"Extraction failed for {filename}: {e}")
        return None


async def process_key_papers():
    """Process the 3 key papers with deep extraction."""

    results = []

    async with aiohttp.ClientSession() as session:
        for paper_name in TARGET_PAPERS:
            paper_path = PAPERS_DIR / paper_name

            if not paper_path.exists():
                logging.warning(f"Paper not found: {paper_path}")
                continue

            logging.info(f"\n{'='*60}")
            logging.info(f"Processing: {paper_name}")
            logging.info(f"{'='*60}")

            # Parse PDF
            markdown = parse_pdf(paper_path)
            if not markdown:
                logging.error(f"Failed to parse: {paper_name}")
                continue

            logging.info(f"Parsed {len(markdown)} characters")

            # Extract with enhanced prompt
            result = await extract_crosswalk_data(markdown, paper_name, session)

            if result:
                results.append(result)
                logging.info(f"✅ Extracted: {paper_name}")

                # Log key findings
                if result.has_paired_vai_magnificd_scores:
                    logging.info(f"   🎯 HAS PAIRED SCORES!")
                if result.vai_magnificd_correlation:
                    logging.info(f"   📊 VAI-MAGNIFI-CD correlation: {result.vai_magnificd_correlation}")
                if result.correlations:
                    logging.info(f"   📈 Found {len(result.correlations)} correlations")
                if result.scatterplots:
                    logging.info(f"   📉 Found {len(result.scatterplots)} scatterplots")
                if result.key_findings:
                    for finding in result.key_findings[:3]:
                        logging.info(f"   → {finding}")
            else:
                logging.error(f"❌ Failed: {paper_name}")

    return results


def save_results(results: List[DeepCrosswalkExtraction]):
    """Save extraction results."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save detailed JSON
    json_path = OUTPUT_DIR / "deep_crosswalk_extraction.json"
    with open(json_path, 'w') as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)
    logging.info(f"Saved detailed results to {json_path}")

    # Generate summary report
    report_path = OUTPUT_DIR / "crosswalk_summary.md"
    with open(report_path, 'w') as f:
        f.write("# Deep Crosswalk Extraction Summary\n\n")
        f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for r in results:
            f.write(f"## {r.filename}\n\n")

            if r.paper_title:
                f.write(f"**Title:** {r.paper_title}\n\n")

            f.write(f"**Has paired VAI/MAGNIFI-CD scores:** {'YES' if r.has_paired_vai_magnificd_scores else 'No'}\n\n")

            if r.vai_summary:
                f.write(f"**VAI Summary:** {r.vai_summary}\n\n")
            if r.magnificd_summary:
                f.write(f"**MAGNIFI-CD Summary:** {r.magnificd_summary}\n\n")
            if r.fibrosis_summary:
                f.write(f"**Fibrosis Summary:** {r.fibrosis_summary}\n\n")

            if r.correlations:
                f.write("### Correlations Found\n\n")
                for corr in r.correlations:
                    f.write(f"- **{corr.system_a} ↔ {corr.system_b}**: r={corr.correlation_coefficient}, type={corr.correlation_type}\n")
                f.write("\n")

            if r.paired_scores:
                f.write("### Paired Scores\n\n")
                for ps in r.paired_scores:
                    f.write(f"- VAI={ps.vai_score}, MAGNIFI-CD={ps.magnificd_score}, Fibrosis={ps.fibrosis_score} ({ps.timepoint})\n")
                f.write("\n")

            if r.tables_with_scores:
                f.write("### Tables with Scoring Data\n\n")
                for t in r.tables_with_scores:
                    f.write(f"- {t.table_reference}: VAI={t.has_vai}, MAGNIFI-CD={t.has_magnificd}\n")
                f.write("\n")

            if r.key_findings:
                f.write("### Key Findings\n\n")
                for finding in r.key_findings:
                    f.write(f"- {finding}\n")
                f.write("\n")

            if r.conversion_statements:
                f.write("### Conversion Statements\n\n")
                for stmt in r.conversion_statements:
                    f.write(f"> {stmt}\n\n")

            if r.raw_evidence:
                f.write("### Supporting Evidence\n\n")
                for ev in r.raw_evidence[:5]:
                    f.write(f"> {ev}\n\n")

            f.write("---\n\n")

    logging.info(f"Saved summary to {report_path}")


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("DEEP CROSSWALK EXTRACTOR")
    print("VAI ↔ MAGNIFI-CD Mapping Data Extraction")
    print("="*60 + "\n")

    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set")
        print("   Run: export OPENROUTER_API_KEY='your-key'")
        return

    print(f"Target papers ({len(TARGET_PAPERS)}):")
    for p in TARGET_PAPERS:
        exists = "✓" if (PAPERS_DIR / p).exists() else "✗ NOT FOUND"
        print(f"  {exists} {p}")
    print()

    results = asyncio.run(process_key_papers())

    if results:
        save_results(results)

        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"✅ Processed: {len(results)}/{len(TARGET_PAPERS)} papers")

        # Summary stats
        has_paired = sum(1 for r in results if r.has_paired_vai_magnificd_scores)
        total_corr = sum(len(r.correlations) for r in results)
        total_pairs = sum(len(r.paired_scores) for r in results)

        print(f"🎯 Papers with paired scores: {has_paired}")
        print(f"📊 Total correlations found: {total_corr}")
        print(f"📋 Total paired score entries: {total_pairs}")
        print(f"\n📁 Output: data/deep_crosswalk_extraction.json")
        print(f"📁 Summary: data/crosswalk_summary.md")
    else:
        print("❌ No successful extractions")


if __name__ == "__main__":
    main()
