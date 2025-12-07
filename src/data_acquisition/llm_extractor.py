"""
MRI-Crohn Atlas Clinical PDF Data Extractor
============================================

Extracts structured clinical research data from PDF documents for building
neuro-symbolic AI crosswalk between Van Assche Index and MAGNIFI-CD scoring systems.

Usage:
    python llm_extractor.py --input ./data/papers --output data/extraction_results.csv
    python llm_extractor.py --test  # Process only first 3 files

Dependencies:
    pip install pymupdf4llm openai pydantic pandas tqdm tiktoken aiohttp
"""

import os
import re
import json
import asyncio
import logging
import argparse
import aiohttp
import tiktoken
import pandas as pd
import pymupdf4llm
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError
from tqdm.asyncio import tqdm_asyncio

# ============ USER CONFIGURATION ============
INPUT_FOLDER = "./data/papers"
OUTPUT_CSV = "data/extraction_results.csv"
FAILED_LOG = "data/failed_log.txt"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Run: export OPENROUTER_API_KEY='your-key'")

# ============ PIPELINE SETTINGS ============
TEST_MODE = False
MAX_CONCURRENT_REQUESTS = 5
MAX_TOKENS_THRESHOLD = 80000  # V3.2 has 131K context, be generous
REQUEST_TIMEOUT = 180
MAX_RETRIES = 2
COST_LIMIT = float('inf')  # Unlimited - cost tracking only for logging

# ============ MODEL CONFIGURATION ============
# DeepSeek V3.2: $0.28/M input, $0.42/M output - production version, best for extraction
# (V3.2-Exp is experimental, V3.2-Speciale doesn't support tool calling)
PRIMARY_MODEL = "deepseek/deepseek-v3.2"
FALLBACK_MODEL = "deepseek/deepseek-v3.2-exp"  # Exp as fallback

# ============ COST TRACKING ============
PRICING = {
    "deepseek/deepseek-v3.2": {"input": 0.28, "output": 0.42},
    "deepseek/deepseek-v3.2-exp": {"input": 0.28, "output": 0.41},
    "deepseek/deepseek-r1": {"input": 2.00, "output": 8.00},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "default": {"input": 1.00, "output": 3.00}
}

class CostTracker:
    def __init__(self, limit: float):
        self.limit = limit
        self.current_cost = 0.0
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.paper_count = 0

    async def add_cost(self, model: str, input_tokens: int, output_tokens: int):
        price = PRICING.get(model, PRICING["default"])
        cost = (input_tokens / 1_000_000 * price["input"]) + (output_tokens / 1_000_000 * price["output"])
        
        async with self.lock:
            self.current_cost += cost
            self.paper_count += 1
            
            # Log every 10 papers
            if self.paper_count % 10 == 0:
                logging.info(f"💰 Cost so far: ${self.current_cost:.4f} ({self.paper_count} papers)")
            
            if self.current_cost >= self.limit:
                logging.warning(f"$$$ COST LIMIT REACHED: ${self.current_cost:.4f} / ${self.limit:.2f} $$$")
                self.stop_event.set()

    def check_limit(self):
        if self.stop_event.is_set():
            raise Exception("Cost limit exceeded")

tracker = CostTracker(COST_LIMIT)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/extractor.log"),
        logging.StreamHandler()
    ]
)

# ============ SCHEMA DEFINITION ============
# Expanded schema for MRI scoring crosswalk research

class ScoringSystemData(BaseModel):
    """Data about a specific MRI scoring system used in the paper."""
    system_name: str = Field(..., description="Name: 'Van Assche', 'MAGNIFI-CD', 'mVAI', 'PEMPAC', 'Fibrosis Score', or other")
    mean_score: Optional[float] = Field(None, description="Mean/median score reported")
    score_range: Optional[str] = Field(None, description="Range reported, e.g., '4-18' or 'IQR 5-12'")
    n_patients_scored: Optional[int] = Field(None, description="Number of patients scored with this system")
    reliability_icc: Optional[float] = Field(None, ge=0, le=1, description="Inter-rater ICC if reported")


class CrosswalkData(BaseModel):
    """Correlation/mapping data between scoring systems."""
    system_a: str = Field(..., description="First scoring system")
    system_b: str = Field(..., description="Second scoring system")
    correlation_r: Optional[float] = Field(None, ge=-1, le=1, description="Pearson/Spearman correlation coefficient")
    correlation_type: Optional[str] = Field(None, description="'pearson', 'spearman', 'ICC', etc.")
    p_value: Optional[float] = Field(None, ge=0, le=1)
    mapping_formula: Optional[str] = Field(None, description="Any explicit conversion formula mentioned")
    notes: Optional[str] = Field(None, max_length=300)


class ClinicalData(BaseModel):
    """Schema for extracted clinical research variables - MRI-Crohn Atlas focused."""
    
    # Metadata
    filename: str = Field(..., description="Source PDF filename")
    publication_year: Optional[int] = Field(None, ge=1990, le=2030)
    study_type: Optional[str] = Field(None, description="'prospective', 'retrospective', 'RCT', 'meta-analysis', 'review', etc.")
    
    # Patient Population
    n_patients: Optional[int] = Field(None, ge=0, description="Total patient count")
    patient_population: Optional[str] = Field(None, description="'adult', 'pediatric', 'mixed'")
    mean_age: Optional[float] = Field(None, ge=0, le=100)
    percent_male: Optional[float] = Field(None, ge=0, le=100)
    disease_phenotype: Optional[str] = Field(None, description="'perianal fistulizing', 'luminal', 'stricturing', etc.")
    
    # MRI Scoring Systems Used
    scoring_systems_used: List[str] = Field(default_factory=list, description="List of MRI scoring systems used")
    scoring_system_details: List[ScoringSystemData] = Field(default_factory=list)
    
    # Crosswalk/Correlation Data (KEY FOR PROJECT)
    crosswalk_data: List[CrosswalkData] = Field(default_factory=list, description="Correlations between scoring systems")
    has_van_assche_magnificd_comparison: bool = Field(False, description="True if paper compares these two systems")
    
    # MRI Technical Parameters
    mri_field_strength: Optional[str] = Field(None, description="'1.5T', '3T', etc.")
    mri_sequences_used: List[str] = Field(default_factory=list, description="T2, DWI, contrast, etc.")
    
    # Clinical Outcomes
    fibrosis_reported: Optional[bool] = Field(None)
    percent_fibrosis: Optional[float] = Field(None, ge=0, le=100)
    percent_abscess: Optional[float] = Field(None, ge=0, le=100)
    percent_healing: Optional[float] = Field(None, ge=0, le=100)
    clinical_remission_definition: Optional[str] = Field(None, max_length=200)
    
    # Treatment Context
    treatment_studied: Optional[str] = Field(None, description="e.g., 'infliximab', 'stem cell', 'seton', etc.")
    followup_duration_months: Optional[float] = Field(None, ge=0)
    
    # Quality Metrics
    extraction_confidence: float = Field(..., ge=0, le=1, description="LLM confidence in extraction")
    data_richness_score: int = Field(..., ge=1, le=5, description="1=minimal data, 5=comprehensive scoring data")
    evidence_snippet: Optional[str] = Field(None, max_length=500, description="Key quote supporting extraction")
    extraction_notes: Optional[str] = Field(None, max_length=300, description="Any issues or notable findings")


# ============ PDF PARSING ============

def parse_pdf(file_path: Path) -> str:
    """Convert PDF to Markdown with intelligent truncation for long documents."""
    try:
        md_text = pymupdf4llm.to_markdown(str(file_path))
        
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(md_text)
        
        if len(tokens) > MAX_TOKENS_THRESHOLD:
            logging.info(f"{file_path.name}: {len(tokens)} tokens, filtering...")
            
            # Remove references section (usually not needed)
            md_text = re.sub(r'(?i)(references|bibliography|acknowledgements?|supplementary).*$', '', md_text, flags=re.DOTALL)
            
            tokens = enc.encode(md_text)
            if len(tokens) > MAX_TOKENS_THRESHOLD:
                decoded_text = enc.decode(tokens[:MAX_TOKENS_THRESHOLD])
                md_text = decoded_text + "\n\n[TRUNCATED]"
                logging.warning(f"{file_path.name}: Truncated to {MAX_TOKENS_THRESHOLD} tokens.")

        return md_text

    except Exception as e:
        logging.error(f"Failed to parse PDF {file_path.name}: {e}")
        with open(FAILED_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {file_path.name} | PDFReadError | {str(e)}\n")
        return ""


# ============ LLM EXTRACTION ============

def get_extraction_prompt() -> str:
    """Return the system prompt for clinical data extraction."""
    schema_json = ClinicalData.model_json_schema()
    
    return f"""You are a clinical data extraction system specialized in MRI-based Crohn's disease research.
Your task is to extract structured data from research papers, with special focus on:
1. MRI scoring systems (Van Assche Index, MAGNIFI-CD, mVAI, PEMPAC, Fibrosis Score)
2. Correlations/comparisons BETWEEN different scoring systems (this is critical!)
3. Patient populations and clinical outcomes

EXTRACTION RULES:
1. Output ONLY valid JSON matching the exact schema below
2. If a value cannot be found with confidence, return null for that field
3. NEVER invent or hallucinate data - only extract what's explicitly stated
4. For scoring_systems_used, list ALL MRI scoring indices mentioned
5. For crosswalk_data, extract ANY correlations between different scoring systems
6. Set has_van_assche_magnificd_comparison=true if paper compares these two systems in any way
7. data_richness_score: 1=no scoring data, 2=mentions scores, 3=reports means, 4=reports correlations, 5=detailed crosswalk
8. extraction_confidence: 1.0=explicit values, 0.7=clear but inferred, 0.4=uncertain, 0.2=very limited data

SCHEMA:
{json.dumps(schema_json, indent=2)}

Return ONLY the JSON object, no other text."""


async def extract_variables(markdown_text: str, filename: str, session: aiohttp.ClientSession) -> Optional[ClinicalData]:
    """Send markdown to LLM and extract structured clinical data."""
    if tracker.stop_event.is_set():
        return None

    system_prompt = get_extraction_prompt()
    
    payload = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract clinical data from this paper.\n\nFilename: {filename}\n\nContent:\n{markdown_text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1  # Low temp for consistent extraction
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/mri-crohn-atlas",
        "X-Title": "MRI-Crohn Atlas Extractor",
        "Content-Type": "application/json"
    }
    
    base_url = "https://openrouter.ai/api/v1/chat/completions"

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with session.post(base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error {response.status}: {error_text[:200]}")
                
                result_json = await response.json()
                
                # Track costs
                usage = result_json.get("usage", {})
                await tracker.add_cost(payload["model"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

                content = result_json['choices'][0]['message']['content']
                
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                data = json.loads(content)
                data['filename'] = filename
                
                return ClinicalData(**data)

        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error for {filename}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
            continue
            
        except ValidationError as e:
            logging.warning(f"Validation error for {filename}: {e}")
            # Try to salvage partial data
            try:
                data['extraction_confidence'] = 0.3
                data['extraction_notes'] = f"Validation issues: {str(e)[:100]}"
                return ClinicalData(**data)
            except:
                pass
            
        except Exception as e:
            logging.warning(f"Attempt {attempt+1}/{MAX_RETRIES+1} failed for {filename}: {str(e)[:100]}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
            else:
                # Try fallback model
                if payload["model"] == PRIMARY_MODEL:
                    logging.info(f"Switching to fallback model for {filename}")
                    payload["model"] = FALLBACK_MODEL
                    try:
                        async with session.post(base_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as fb_response:
                            if fb_response.status == 200:
                                fb_json = await fb_response.json()
                                usage = fb_json.get("usage", {})
                                await tracker.add_cost(FALLBACK_MODEL, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                                fb_content = fb_json['choices'][0]['message']['content']
                                if "```json" in fb_content:
                                    fb_content = fb_content.split("```json")[1].split("```")[0].strip()
                                data = json.loads(fb_content)
                                data['filename'] = filename
                                return ClinicalData(**data)
                    except Exception as fb_e:
                        logging.error(f"Fallback failed for {filename}: {fb_e}")
                
                with open(FAILED_LOG, "a") as f:
                    f.write(f"{datetime.now().isoformat()} | {filename} | ExtractionError | {str(e)[:200]}\n")
                return None
    
    return None


# ============ BATCH PROCESSING ============

async def process_single_file(file_path: Path, semaphore: asyncio.Semaphore, session: aiohttp.ClientSession) -> Optional[ClinicalData]:
    """Process a single PDF file with concurrency control."""
    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            markdown = await loop.run_in_executor(None, parse_pdf, file_path)
            
            if not markdown:
                raise ValueError("Empty markdown output")
            
            result = await extract_variables(markdown, file_path.name, session)
            return result
            
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            with open(FAILED_LOG, "a") as f:
                f.write(f"{datetime.now().isoformat()} | {file_path.name} | ProcessError | {str(e)}\n")
            return None


async def process_batch(input_folder: Path) -> List[ClinicalData]:
    """Main batch processing function."""
    pdf_files = list(input_folder.glob("**/*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_folder}")
        return []

    if TEST_MODE:
        pdf_files = pdf_files[:3]
        logging.info(f"⚠️ TEST MODE: Processing only {len(pdf_files)} files")
    
    logging.info(f"Found {len(pdf_files)} PDFs to process.")
    logging.info(f"Using model: {PRIMARY_MODEL}")
    logging.info(f"Cost limit: ${COST_LIMIT:.2f}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_single_file(f, semaphore, session) for f in pdf_files]
        results = await tqdm_asyncio.gather(*tasks, desc="Extracting data")
    
    return [r for r in results if r is not None]


# ============ POST-PROCESSING ============

def flatten_results(results: List[ClinicalData]) -> pd.DataFrame:
    """Flatten nested data for CSV export."""
    rows = []
    for r in results:
        row = r.model_dump()
        
        # Flatten scoring_systems_used to comma-separated string
        row['scoring_systems_used'] = ', '.join(row.get('scoring_systems_used', []))
        row['mri_sequences_used'] = ', '.join(row.get('mri_sequences_used', []))
        
        # Count crosswalk entries
        crosswalk = row.pop('crosswalk_data', [])
        row['n_crosswalk_comparisons'] = len(crosswalk)
        
        # Extract key crosswalk if Van Assche <-> MAGNIFI-CD exists
        for cw in crosswalk:
            systems = {cw.get('system_a', '').lower(), cw.get('system_b', '').lower()}
            if 'van assche' in ' '.join(systems) and 'magnifi' in ' '.join(systems):
                row['va_magnificd_correlation'] = cw.get('correlation_r')
                row['va_magnificd_formula'] = cw.get('mapping_formula')
                break
        
        # Flatten scoring_system_details
        details = row.pop('scoring_system_details', [])
        row['n_scoring_systems_detailed'] = len(details)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============ ENTRY POINT ============

def main():
    """Entry point with argument parsing."""
    global TEST_MODE, INPUT_FOLDER, OUTPUT_CSV, COST_LIMIT
    
    parser = argparse.ArgumentParser(description="MRI-Crohn Atlas Clinical PDF Data Extractor")
    parser.add_argument("--input", "-i", default=INPUT_FOLDER, help="Input folder path")
    parser.add_argument("--output", "-o", default=OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--test", action="store_true", help="Enable test mode (3 files only)")
    parser.add_argument("--cost-limit", type=float, default=COST_LIMIT, help="Max spend in USD")
    args = parser.parse_args()
    
    if args.test:
        TEST_MODE = True
    INPUT_FOLDER = args.input
    OUTPUT_CSV = args.output
    COST_LIMIT = args.cost_limit
    
    if not OPENROUTER_API_KEY:
        print("❌ CRITICAL: OPENROUTER_API_KEY not set.")
        print("   export OPENROUTER_API_KEY='your-key-here'")
        return
    
    global tracker
    tracker = CostTracker(COST_LIMIT)
    
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"Creating input directory: {input_path}")
        input_path.mkdir(parents=True, exist_ok=True)
        print("Please place PDF files in this directory and run again.")
        return

    # Ensure data dir exists for logs
    Path("data").mkdir(exist_ok=True)
    
    print(f"🚀 MRI-Crohn Atlas Extractor")
    print(f"   Model: {PRIMARY_MODEL}")
    print(f"   Input: {input_path}")
    print(f"   Cost limit: ${COST_LIMIT:.2f}")
    print()

    try:
        results = asyncio.run(process_batch(input_path))
        
        if results:
            df = flatten_results(results)
            df["extraction_timestamp"] = pd.Timestamp.now()
            
            output_path = Path(OUTPUT_CSV)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(OUTPUT_CSV, index=False)
            
            # Summary stats
            print()
            print("=" * 60)
            print("EXTRACTION COMPLETE")
            print("=" * 60)
            print(f"✅ Extracted: {len(results)} papers")
            print(f"💰 Total cost: ${tracker.current_cost:.4f}")
            print(f"📁 Output: {OUTPUT_CSV}")
            print()
            
            # Key metrics
            has_crosswalk = sum(1 for r in results if r.has_van_assche_magnificd_comparison)
            high_richness = sum(1 for r in results if r.data_richness_score >= 4)
            print(f"📊 Papers with Van Assche ↔ MAGNIFI-CD data: {has_crosswalk}")
            print(f"📊 High data richness (score ≥4): {high_richness}")
            
            # Save detailed JSON too
            json_output = OUTPUT_CSV.replace('.csv', '.json')
            with open(json_output, 'w') as f:
                json.dump([r.model_dump() for r in results], f, indent=2, default=str)
            print(f"📁 Detailed JSON: {json_output}")
            
        else:
            print("❌ No successful extractions")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline stopped by user.")
        print(f"💰 Cost before stop: ${tracker.current_cost:.4f}")


if __name__ == "__main__":
    main()