#!/usr/bin/env python3
"""
Master Script for MRI Report Collection
Runs all scrapers and feature extraction in sequence
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from radiopaedia_scraper import RadiopaediaScraper
from pubmed_scraper import PubMedScraper
from extract_features import FeatureExtractor, load_scraped_reports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print a nice banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         MRI-Crohn Atlas - Report Collection System           ║
    ║                                                              ║
    ║   Collecting real MRI report language for parser validation  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_radiopaedia_scraper(output_dir: Path, max_cases: int = 15) -> int:
    """Run Radiopaedia scraper"""
    print("\n" + "="*60)
    print("STEP 1: Scraping Radiopaedia.org")
    print("="*60)

    scraper = RadiopaediaScraper(delay=2.0)
    cases = scraper.scrape_all(max_cases=max_cases)

    output_path = output_dir / "radiopaedia_cases.json"
    scraper.save_results(output_path)

    print(f"\n✓ Collected {len(cases)} cases from Radiopaedia")
    return len(cases)


def run_pubmed_scraper(output_dir: Path, max_articles: int = 15) -> int:
    """Run PubMed Central scraper"""
    print("\n" + "="*60)
    print("STEP 2: Scraping PubMed Central")
    print("="*60)

    scraper = PubMedScraper(delay=1.0)
    reports = scraper.scrape_all(max_articles=max_articles)

    output_path = output_dir / "pubmed_cases.json"
    scraper.save_results(output_path)

    print(f"\n✓ Collected {len(reports)} case reports from PubMed Central")
    return len(reports)


def run_feature_extraction(output_dir: Path, api_key: str) -> int:
    """Run LLM feature extraction"""
    print("\n" + "="*60)
    print("STEP 3: Extracting Features with DeepSeek V3.2")
    print("="*60)

    # Load all scraped reports
    reports = load_scraped_reports(output_dir)

    if not reports:
        print("No reports found to process!")
        return 0

    print(f"Processing {len(reports)} reports...")

    extractor = FeatureExtractor(api_key=api_key, delay=1.5)
    processed = extractor.process_reports(reports)

    # Save combined results
    output_path = output_dir / "collected_reports.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in processed if r.get("extracted_features"))
    print(f"\n✓ Extracted features from {successful}/{len(processed)} reports")

    return successful


def generate_summary(output_dir: Path):
    """Generate a summary report"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)

    collected_path = output_dir / "collected_reports.json"
    if not collected_path.exists():
        print("No collected reports found!")
        return

    with open(collected_path, 'r', encoding='utf-8') as f:
        reports = json.load(f)

    # Statistics
    total = len(reports)
    by_source = {}
    by_fistula_type = {}
    by_severity = {}
    by_activity = {}

    for report in reports:
        # By source
        source = report.get("source", "unknown")
        by_source[source] = by_source.get(source, 0) + 1

        # By extracted features
        features = report.get("extracted_features", {})
        if features:
            ftype = features.get("fistula_type", "unknown")
            by_fistula_type[ftype] = by_fistula_type.get(ftype, 0) + 1

            severity = features.get("severity_estimate", "unknown")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            activity = features.get("activity_assessment", "unknown")
            by_activity[activity] = by_activity.get(activity, 0) + 1

    print(f"\nTotal Reports Collected: {total}")
    print(f"\nBy Source:")
    for source, count in by_source.items():
        print(f"  - {source}: {count}")

    print(f"\nBy Fistula Type:")
    for ftype, count in sorted(by_fistula_type.items(), key=lambda x: -x[1]):
        if ftype and ftype != "null":
            print(f"  - {ftype}: {count}")

    print(f"\nBy Severity:")
    for severity, count in sorted(by_severity.items(), key=lambda x: -x[1]):
        if severity and severity != "null":
            print(f"  - {severity}: {count}")

    print(f"\nBy Activity Status:")
    for activity, count in sorted(by_activity.items(), key=lambda x: -x[1]):
        if activity and activity != "null":
            print(f"  - {activity}: {count}")

    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_reports": total,
        "by_source": by_source,
        "by_fistula_type": by_fistula_type,
        "by_severity": by_severity,
        "by_activity": by_activity,
    }

    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full data saved to: {collected_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Collect real MRI report language for parser validation"
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping and only run feature extraction"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip feature extraction"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=15,
        help="Maximum cases to collect per source (default: 15)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="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",
        help="OpenRouter API key for feature extraction"
    )

    args = parser.parse_args()

    print_banner()

    # Setup output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "real_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    radiopaedia_count = 0
    pubmed_count = 0
    extracted_count = 0

    # Step 1 & 2: Scraping
    if not args.skip_scrape:
        try:
            radiopaedia_count = run_radiopaedia_scraper(output_dir, args.max_cases)
        except Exception as e:
            logger.error(f"Radiopaedia scraper failed: {e}")
            print(f"⚠ Radiopaedia scraper failed: {e}")

        try:
            pubmed_count = run_pubmed_scraper(output_dir, args.max_cases)
        except Exception as e:
            logger.error(f"PubMed scraper failed: {e}")
            print(f"⚠ PubMed scraper failed: {e}")
    else:
        print("\n⏭ Skipping scraping step")

    # Step 3: Feature Extraction
    if not args.skip_extract:
        try:
            extracted_count = run_feature_extraction(output_dir, args.api_key)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            print(f"⚠ Feature extraction failed: {e}")
    else:
        print("\n⏭ Skipping feature extraction step")

    # Generate summary
    generate_summary(output_dir)

    # Final message
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"""
Results saved to: {output_dir}

Files created:
  - radiopaedia_cases.json    ({radiopaedia_count} cases)
  - pubmed_cases.json         ({pubmed_count} cases)
  - collected_reports.json    (combined with features)
  - collection_summary.json   (statistics)

Next steps:
  1. Review collected_reports.json
  2. Use reports to test MRI Report Parser
  3. Validate extracted features against manual review
    """)


if __name__ == "__main__":
    main()
