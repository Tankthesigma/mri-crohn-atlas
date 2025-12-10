#!/usr/bin/env python3
"""
PubMed-focused scraper using E-utilities API
This approach is rate-limit friendly and designed for programmatic access
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
import requests
from xml.etree import ElementTree as ET

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"

# NCBI E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Rate limiting: NCBI allows 3 requests per second without API key
RATE_LIMIT_DELAY = 0.4  # 400ms between requests

def search_pubmed(query, retmax=100):
    """Search PubMed for articles matching query"""
    search_url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y"
    }

    try:
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = data.get('esearchresult', {})
        count = int(result.get('count', 0))
        id_list = result.get('idlist', [])

        print(f"   Found {count} total results, retrieved {len(id_list)} IDs")
        return id_list

    except Exception as e:
        print(f"   Error searching: {e}")
        return []

def fetch_article_details(pmcid):
    """Fetch article details from PMC"""
    time.sleep(RATE_LIMIT_DELAY)

    fetch_url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml"
    }

    try:
        response = requests.get(fetch_url, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"   Error fetching PMC{pmcid}: {e}")
        return None

def parse_article_xml(xml_content, pmcid):
    """Parse PMC XML to extract case data"""
    try:
        root = ET.fromstring(xml_content)

        # Get title
        title_elem = root.find('.//article-title')
        title = title_elem.text if title_elem is not None else ""

        # Get abstract
        abstract_parts = root.findall('.//abstract//p')
        abstract = " ".join([p.text for p in abstract_parts if p.text])

        # Get body text - look for case presentation or findings sections
        body_text = ""
        body = root.find('.//body')
        if body is not None:
            # Look for relevant sections
            sections = body.findall('.//sec')
            for sec in sections:
                title_elem = sec.find('title')
                if title_elem is not None:
                    sec_title = title_elem.text.lower() if title_elem.text else ""
                    if any(kw in sec_title for kw in ['case', 'finding', 'imaging', 'mri', 'result', 'report']):
                        # Get all text from this section
                        for p in sec.findall('.//p'):
                            if p.text:
                                body_text += p.text + " "

        # If no specific sections found, get all body paragraphs
        if not body_text and body is not None:
            for p in body.findall('.//p'):
                if p.text:
                    body_text += p.text + " "

        # Combine text for analysis
        full_text = f"{title} {abstract} {body_text}".lower()

        # Check if relevant
        relevance_keywords = [
            'perianal fistula', 'anal fistula', 'fistula-in-ano', 'fistula in ano',
            'perianal crohn', 'crohn.*perianal', 'intersphincteric', 'transsphincteric',
            'ischioanal', 'ischiorectal', 'horseshoe fistula', 'anal sphincter'
        ]

        if not any(re.search(kw, full_text) for kw in relevance_keywords):
            return None

        # Exclude non-relevant types
        exclude_keywords = [
            'review article', 'systematic review', 'meta-analysis', 'guidelines',
            'vesicovaginal', 'rectovaginal', 'enterocutaneous', 'tracheo'
        ]

        if any(kw in full_text for kw in exclude_keywords):
            return None

        # Determine fistula type
        fistula_type = None
        if 'intersphincteric' in full_text:
            fistula_type = 'intersphincteric'
        elif 'transsphincteric' in full_text or 'trans-sphincteric' in full_text:
            fistula_type = 'transsphincteric'
        elif 'extrasphincteric' in full_text:
            fistula_type = 'extrasphincteric'
        elif 'suprasphincteric' in full_text:
            fistula_type = 'suprasphincteric'
        elif 'horseshoe' in full_text:
            fistula_type = 'horseshoe'
        elif 'complex' in full_text or 'multiple' in full_text:
            fistula_type = 'complex'

        # Estimate severity
        severity = None
        if 'severe' in full_text or 'extensive' in full_text:
            severity = 'severe'
        elif 'moderate' in full_text:
            severity = 'moderate'
        elif 'mild' in full_text or 'minimal' in full_text:
            severity = 'mild'
        elif 'healed' in full_text or 'remission' in full_text:
            severity = 'remission'

        # Check for MRI findings
        has_mri = bool(re.search(r't2|t1|hyperintense|hypointense|enhancement|diffusion|dwi|mri', full_text))

        # Quality assessment
        quality = 3
        if has_mri:
            quality += 1
        if len(body_text) > 500:
            quality += 1
        if fistula_type:
            quality += 0.5
        quality = min(5, int(quality))

        findings_text = body_text if body_text else abstract

        if len(findings_text) < 100:
            return None

        return {
            "case_id": f"pmc_{pmcid}",
            "source": "pubmed_central",
            "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
            "title": title,
            "clinical_history": "",
            "findings_text": findings_text[:3000],  # Limit to 3000 chars
            "diagnosis": "",
            "fistula_type": fistula_type,
            "severity_indicators": severity,
            "treatment_mentioned": None,
            "has_abscess": 'abscess' in full_text,
            "pediatric": bool(re.search(r'pediatric|paediatric|child|infant', full_text)),
            "scored_vai": None,
            "scored_magnifi": None,
            "scraped_date": datetime.now().strftime("%Y-%m-%d"),
            "quality_score": quality,
            "has_mri_findings": has_mri
        }

    except Exception as e:
        print(f"   Error parsing PMC{pmcid}: {e}")
        return None

def load_existing_cases():
    """Load existing master cases"""
    master_path = TRAINING_DIR / "master_cases.json"
    if master_path.exists():
        with open(master_path, 'r') as f:
            data = json.load(f)
            return data.get('cases', [])
    return []

def get_existing_pmcids(cases):
    """Get set of existing PMC IDs"""
    pmcids = set()
    for case in cases:
        url = case.get('source_url', '')
        match = re.search(r'PMC(\d+)', url)
        if match:
            pmcids.add(match.group(1))
        case_id = case.get('case_id', '')
        if case_id.startswith('pmc_'):
            pmcids.add(case_id.replace('pmc_', ''))
    return pmcids

def save_cases(new_cases, existing_cases):
    """Save combined cases"""
    all_cases = existing_cases + new_cases

    master_path = TRAINING_DIR / "master_cases.json"
    with open(master_path, 'w') as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_cases": len(all_cases),
                "description": "Master deduplicated case file for fine-tuning"
            },
            "cases": all_cases
        }, f, indent=2)

    print(f"\n💾 Saved {len(all_cases)} total cases")

def main():
    print("=" * 60)
    print("PubMed Central Scraper for MRI-Crohn Atlas")
    print("=" * 60)

    # Load existing
    existing_cases = load_existing_cases()
    existing_pmcids = get_existing_pmcids(existing_cases)
    print(f"\n📂 Starting with {len(existing_cases)} existing cases")
    print(f"   Already have {len(existing_pmcids)} PMC articles")

    # Search queries
    queries = [
        "perianal fistula MRI case report",
        "anal fistula magnetic resonance imaging",
        "perianal Crohn disease MRI",
        "Van Assche index",
        "fistula-in-ano MRI",
        "transsphincteric fistula MRI",
        "intersphincteric fistula imaging",
        "horseshoe fistula MRI",
        "perianal abscess MRI",
        "complex anal fistula imaging",
        "ischioanal abscess MRI",
        "anorectal fistula magnetic resonance"
    ]

    all_new_cases = []
    all_pmcids = set()

    for query in queries:
        print(f"\n🔍 Searching: {query}")
        pmcids = search_pubmed(query, retmax=50)

        for pmcid in pmcids:
            if pmcid in existing_pmcids or pmcid in all_pmcids:
                continue

            all_pmcids.add(pmcid)

            print(f"   Fetching PMC{pmcid}...", end=" ")
            xml_content = fetch_article_details(pmcid)

            if xml_content:
                case_data = parse_article_xml(xml_content, pmcid)
                if case_data:
                    all_new_cases.append(case_data)
                    print(f"✓ {case_data.get('title', 'Unknown')[:40]}...")
                else:
                    print("(not relevant)")
            else:
                print("(fetch failed)")

        # Progress save every 10 new cases
        if len(all_new_cases) >= 10 and len(all_new_cases) % 10 == 0:
            print(f"\n   === Progress: {len(all_new_cases)} new cases ===")
            save_cases(all_new_cases, existing_cases)

    # Final save
    save_cases(all_new_cases, existing_cases)

    # Report
    print("\n" + "=" * 60)
    print("PUBMED SCRAPE COMPLETE")
    print("=" * 60)
    print(f"Total new cases: {len(all_new_cases)}")
    print(f"Total cases now: {len(existing_cases) + len(all_new_cases)}")

    # Type breakdown
    type_counts = {}
    for case in all_new_cases:
        ft = case.get('fistula_type', 'unknown') or 'unknown'
        type_counts[ft] = type_counts.get(ft, 0) + 1

    print(f"\nNew cases by type:")
    for ft, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {ft}: {count}")

if __name__ == "__main__":
    main()
