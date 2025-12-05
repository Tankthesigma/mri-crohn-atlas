"""
Feature Extractor using DeepSeek V3.2 via OpenRouter
Extracts structured MRI features from free-text radiology findings
"""

import json
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract structured MRI features using LLM"""

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Extraction prompt template
    EXTRACTION_PROMPT = """You are an expert radiologist analyzing MRI findings for perianal fistulas in Crohn's disease.

Extract structured features from the following radiology report text. Be precise and conservative - only report features that are explicitly mentioned or clearly implied.

REPORT TEXT:
{report_text}

Extract the following features and return ONLY a valid JSON object:

{{
    "fistula_count": <integer or null if not mentioned>,
    "fistula_type": <"intersphincteric" | "transsphincteric" | "suprasphincteric" | "extrasphincteric" | "superficial" | "complex" | "multiple_types" | null>,
    "t2_hyperintensity": <true | false | null>,
    "t2_hyperintensity_degree": <"mild" | "moderate" | "marked" | null>,
    "extension_pattern": <description of fistula extension or null>,
    "collections_abscesses": <true | false | null>,
    "collection_count": <integer or null>,
    "collection_location": <description or null>,
    "rectal_wall_involvement": <true | false | null>,
    "internal_opening_location": <clock position like "6 o'clock" or description or null>,
    "external_opening": <true | false | null>,
    "sphincter_involvement": {{
        "internal_sphincter": <true | false | null>,
        "external_sphincter": <true | false | null>,
        "puborectalis": <true | false | null>
    }},
    "branching": <true | false | null>,
    "horseshoe_extension": <true | false | null>,
    "enhancement_pattern": <"rim" | "homogeneous" | "heterogeneous" | null>,
    "fibrosis_present": <true | false | null>,
    "activity_assessment": <"active" | "healing" | "healed" | "chronic" | null>,
    "severity_estimate": <"mild" | "moderate" | "severe" | null>,
    "estimated_vai_range": <"0-4" | "5-10" | "11-16" | "17-22" | null>,
    "additional_findings": <string description of other relevant findings or null>,
    "confidence": <"high" | "medium" | "low">
}}

Return ONLY the JSON object, no explanations or markdown."""

    def __init__(self, api_key: str, delay: float = 1.0):
        """Initialize with OpenRouter API key"""
        self.api_key = api_key
        self.delay = delay
        self.session = requests.Session()

    def extract_features(self, report_text: str) -> Optional[Dict]:
        """Extract features from a single report"""
        if not report_text or len(report_text) < 50:
            return None

        # Truncate very long texts
        if len(report_text) > 3000:
            report_text = report_text[:3000] + "..."

        prompt = self.EXTRACTION_PROMPT.format(report_text=report_text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mri-crohn-atlas.local",
            "X-Title": "MRI-Crohn Atlas Feature Extractor",
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
        }

        time.sleep(self.delay)

        try:
            response = self.session.post(
                self.OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                features = json.loads(json_match.group())
                return features
            else:
                logger.warning("No JSON found in LLM response")
                return None

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return None

    def process_reports(self, reports: List[Dict]) -> List[Dict]:
        """Process a list of reports and add extracted features"""
        processed = []

        for i, report in enumerate(reports):
            logger.info(f"Processing report {i+1}/{len(reports)}: {report.get('title', 'Unknown')[:50]}")

            findings_text = report.get("findings_text", "")
            features = self.extract_features(findings_text)

            if features:
                report["extracted_features"] = features
                processed.append(report)
                logger.info(f"  -> Extracted features (confidence: {features.get('confidence', 'unknown')})")
            else:
                # Keep report but mark as failed extraction
                report["extracted_features"] = None
                report["extraction_error"] = True
                processed.append(report)
                logger.warning(f"  -> Failed to extract features")

        return processed


def load_scraped_reports(data_dir: Path) -> List[Dict]:
    """Load all scraped reports from JSON files"""
    reports = []

    radiopaedia_path = data_dir / "radiopaedia_cases.json"
    if radiopaedia_path.exists():
        with open(radiopaedia_path, 'r', encoding='utf-8') as f:
            radiopaedia_reports = json.load(f)
            reports.extend(radiopaedia_reports)
            logger.info(f"Loaded {len(radiopaedia_reports)} Radiopaedia cases")

    pubmed_path = data_dir / "pubmed_cases.json"
    if pubmed_path.exists():
        with open(pubmed_path, 'r', encoding='utf-8') as f:
            pubmed_reports = json.load(f)
            reports.extend(pubmed_reports)
            logger.info(f"Loaded {len(pubmed_reports)} PubMed cases")

    return reports


def main():
    """Main function to run feature extraction"""
    # API key
    api_key = "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"

    data_dir = Path(__file__).parent.parent.parent / "data" / "real_reports"

    # Load scraped reports
    reports = load_scraped_reports(data_dir)

    if not reports:
        logger.error("No reports found to process. Run scrapers first.")
        return

    logger.info(f"Processing {len(reports)} total reports")

    # Extract features
    extractor = FeatureExtractor(api_key=api_key, delay=1.5)
    processed_reports = extractor.process_reports(reports)

    # Save combined results
    output_path = data_dir / "collected_reports.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_reports, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(processed_reports)} processed reports to {output_path}")

    # Print summary
    successful = sum(1 for r in processed_reports if r.get("extracted_features"))
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"Total reports: {len(processed_reports)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {len(processed_reports) - successful}")
    print(f"\nOutput saved to: {output_path}")

    # Show sample features
    if successful > 0:
        print(f"\nSample extracted features:")
        for report in processed_reports[:3]:
            if report.get("extracted_features"):
                features = report["extracted_features"]
                print(f"\n  {report['title'][:50]}...")
                print(f"    Fistula type: {features.get('fistula_type')}")
                print(f"    T2 hyperintensity: {features.get('t2_hyperintensity')}")
                print(f"    Severity: {features.get('severity_estimate')}")
                print(f"    Estimated VAI: {features.get('estimated_vai_range')}")


if __name__ == "__main__":
    main()
