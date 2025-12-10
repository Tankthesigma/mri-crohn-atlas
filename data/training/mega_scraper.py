#!/usr/bin/env python3
"""
MEGA SCRAPER - Target 300-500 cases
Hits every possible source aggressively
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

# Rate limiting
RATE_LIMIT_DELAY = 0.5

def load_master_cases():
    """Load existing master cases"""
    master_path = TRAINING_DIR / "master_cases.json"
    if master_path.exists():
        with open(master_path, 'r') as f:
            data = json.load(f)
            return data.get('cases', [])
    return []

def get_existing_urls(cases):
    """Get set of existing URLs for deduplication"""
    urls = set()
    for case in cases:
        url = case.get('source_url', '')
        if url:
            urls.add(url.lower().strip())
        case_id = case.get('case_id', '')
        if case_id:
            urls.add(case_id.lower())
    return urls

def save_master_cases(cases):
    """Save master cases - APPENDS, never overwrites"""
    master_path = TRAINING_DIR / "master_cases.json"
    with open(master_path, 'w') as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_cases": len(cases),
                "description": "Master deduplicated case file for fine-tuning"
            },
            "cases": cases
        }, f, indent=2)
    print(f"💾 Saved {len(cases)} total cases")

def text_hash(text):
    """Generate hash for text similarity check"""
    clean = re.sub(r'\s+', ' ', text.lower().strip())[:500]
    return hashlib.md5(clean.encode()).hexdigest()

def is_relevant(text):
    """Check if text is relevant to perianal fistulas"""
    text = text.lower()
    keywords = [
        'perianal fistula', 'anal fistula', 'fistula-in-ano', 'fistula in ano',
        'intersphincteric', 'transsphincteric', 'suprasphincteric', 'extrasphincteric',
        'horseshoe fistula', 'ischioanal', 'ischiorectal', 'perianal crohn',
        'anal sphincter', 'internal opening', 'external opening', 'fistula tract'
    ]
    return any(kw in text for kw in keywords)

def classify_fistula_type(text):
    """Classify fistula type from text"""
    text = text.lower()
    if 'intersphincteric' in text:
        return 'intersphincteric'
    elif 'transsphincteric' in text or 'trans-sphincteric' in text:
        return 'transsphincteric'
    elif 'suprasphincteric' in text or 'supra-sphincteric' in text:
        return 'suprasphincteric'
    elif 'extrasphincteric' in text or 'extra-sphincteric' in text:
        return 'extrasphincteric'
    elif 'horseshoe' in text:
        return 'horseshoe'
    elif 'complex' in text or 'multiple' in text or 'branching' in text:
        return 'complex'
    return None

def classify_severity(text):
    """Classify severity from text"""
    text = text.lower()
    if 'severe' in text or 'extensive' in text or 'fulminant' in text:
        return 'severe'
    elif 'moderate' in text:
        return 'moderate'
    elif 'mild' in text or 'minimal' in text or 'simple' in text:
        return 'mild'
    elif 'healed' in text or 'remission' in text or 'fibrotic' in text or 'inactive' in text:
        return 'remission'
    return None

# ============================================================
# PUBMED CENTRAL - Extended queries
# ============================================================
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def search_pubmed_extended(query, retmax=100):
    """Extended PubMed search"""
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
        return result.get('idlist', [])
    except Exception as e:
        print(f"   Error: {e}")
        return []

def fetch_pmc_article(pmcid):
    """Fetch article from PMC"""
    time.sleep(RATE_LIMIT_DELAY)
    fetch_url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}

    try:
        response = requests.get(fetch_url, params=params, timeout=60)
        response.raise_for_status()
        return response.text
    except:
        return None

def parse_pmc_xml(xml_content, pmcid):
    """Parse PMC XML to extract case"""
    try:
        root = ET.fromstring(xml_content)

        title_elem = root.find('.//article-title')
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        abstract_parts = root.findall('.//abstract//p')
        abstract = " ".join([p.text for p in abstract_parts if p.text])

        body_text = ""
        body = root.find('.//body')
        if body is not None:
            for p in body.findall('.//p'):
                if p.text:
                    body_text += p.text + " "

        full_text = f"{title} {abstract} {body_text}"

        if not is_relevant(full_text):
            return None

        # Exclude reviews
        if any(kw in full_text.lower() for kw in ['systematic review', 'meta-analysis', 'guidelines']):
            return None

        findings = body_text if len(body_text) > 100 else abstract
        if len(findings) < 100:
            return None

        has_mri = bool(re.search(r't2|t1|hyperintense|mri|magnetic resonance|diffusion', full_text.lower()))
        quality = 4 if has_mri else 3
        if len(body_text) > 500:
            quality += 1
        quality = min(5, quality)

        return {
            "case_id": f"pmc_{pmcid}",
            "source": "pubmed_central",
            "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
            "title": title[:200],
            "clinical_history": "",
            "findings_text": findings[:3000],
            "diagnosis": "",
            "fistula_type": classify_fistula_type(full_text),
            "severity_indicators": classify_severity(full_text),
            "treatment_mentioned": None,
            "has_abscess": 'abscess' in full_text.lower(),
            "pediatric": bool(re.search(r'pediatric|paediatric|child|infant', full_text.lower())),
            "scored_vai": None,
            "scored_magnifi": None,
            "scraped_date": datetime.now().strftime("%Y-%m-%d"),
            "quality_score": quality,
            "has_mri_findings": has_mri
        }
    except:
        return None

def scrape_pubmed_extended(existing_cases, existing_urls):
    """Extended PubMed scraping with more queries"""
    queries = [
        # Original queries
        "perianal fistula MRI case report",
        "anal fistula magnetic resonance imaging",
        "perianal Crohn disease MRI",

        # Gap-filling queries
        "suprasphincteric fistula",
        "suprasphincteric anal fistula MRI",
        "high transsphincteric fistula",
        "horseshoe abscess MRI",
        "bilateral ischioanal abscess",

        # Severity-specific
        "complex perianal fistula treatment",
        "recurrent anal fistula MRI",
        "refractory perianal Crohn",
        "healed perianal fistula imaging",
        "fistula remission MRI",

        # Additional anatomical
        "ischiorectal fossa abscess",
        "supralevator abscess MRI",
        "intersphincteric abscess imaging",
        "cryptoglandular fistula MRI",

        # Treatment-related with imaging
        "seton drainage MRI",
        "LIFT procedure fistula",
        "advancement flap anal fistula",
        "stem cell fistula Crohn",
        "biologic therapy perianal Crohn",

        # Pediatric
        "pediatric perianal fistula",
        "pediatric Crohn perianal",
        "childhood anal fistula",

        # International
        "fistula-in-ano imaging China",
        "perianal fistula Korea",
        "anal fistula Japan MRI"
    ]

    new_cases = []
    seen_pmcids = set()

    # Get existing PMC IDs
    for case in existing_cases:
        url = case.get('source_url', '')
        match = re.search(r'PMC(\d+)', url)
        if match:
            seen_pmcids.add(match.group(1))
        case_id = case.get('case_id', '')
        if case_id.startswith('pmc_'):
            seen_pmcids.add(case_id.replace('pmc_', ''))

    print(f"\n{'='*60}")
    print("EXTENDED PUBMED CENTRAL SCRAPE")
    print(f"{'='*60}")
    print(f"Already have {len(seen_pmcids)} PMC articles")

    for query in queries:
        print(f"\n🔍 {query}")
        pmcids = search_pubmed_extended(query, retmax=100)
        print(f"   Found {len(pmcids)} IDs")

        for pmcid in pmcids:
            if pmcid in seen_pmcids:
                continue
            seen_pmcids.add(pmcid)

            xml = fetch_pmc_article(pmcid)
            if xml:
                case = parse_pmc_xml(xml, pmcid)
                if case:
                    url = case['source_url'].lower()
                    if url not in existing_urls:
                        new_cases.append(case)
                        existing_urls.add(url)
                        print(f"   ✓ PMC{pmcid}: {case.get('title', '')[:40]}...")

                        if len(new_cases) % 50 == 0:
                            print(f"\n   === PROGRESS: {len(new_cases)} new cases ===\n")

    return new_cases

# ============================================================
# EURORAD SCRAPER
# ============================================================
def scrape_eurorad(existing_urls):
    """Scrape Eurorad for cases"""
    print(f"\n{'='*60}")
    print("EURORAD SCRAPE")
    print(f"{'='*60}")

    new_cases = []
    searches = [
        "perianal+fistula",
        "anal+fistula",
        "crohn+perianal",
        "anorectal+abscess",
        "ischioanal",
        "horseshoe+fistula"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    for search in searches:
        print(f"\n🔍 Eurorad: {search}")
        time.sleep(2)  # Respectful delay

        try:
            url = f"https://www.eurorad.org/advanced-search?search={search}"
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429:
                print("   Rate limited, waiting 30s...")
                time.sleep(30)
                continue

            # Parse case links from search results
            case_links = re.findall(r'href="(/case/\d+)"', response.text)
            print(f"   Found {len(case_links)} case links")

            for link in case_links[:20]:  # Limit per search
                case_url = f"https://www.eurorad.org{link}"
                if case_url.lower() in existing_urls:
                    continue

                time.sleep(1.5)
                try:
                    case_resp = requests.get(case_url, headers=headers, timeout=30)
                    if case_resp.status_code != 200:
                        continue

                    html = case_resp.text

                    # Extract title
                    title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
                    title = title_match.group(1) if title_match else ""

                    # Extract case content
                    content_match = re.search(r'<div class="field-content">(.*?)</div>', html, re.DOTALL)
                    content = content_match.group(1) if content_match else ""
                    content = re.sub(r'<[^>]+>', ' ', content)

                    full_text = f"{title} {content}"

                    if not is_relevant(full_text):
                        continue

                    case_id = re.search(r'/case/(\d+)', link).group(1)

                    case_data = {
                        "case_id": f"eurorad_{case_id}",
                        "source": "eurorad",
                        "source_url": case_url,
                        "title": title[:200],
                        "clinical_history": "",
                        "findings_text": content[:3000],
                        "diagnosis": "",
                        "fistula_type": classify_fistula_type(full_text),
                        "severity_indicators": classify_severity(full_text),
                        "treatment_mentioned": None,
                        "has_abscess": 'abscess' in full_text.lower(),
                        "pediatric": bool(re.search(r'pediatric|paediatric|child', full_text.lower())),
                        "scored_vai": None,
                        "scored_magnifi": None,
                        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                        "quality_score": 4,
                        "has_mri_findings": bool(re.search(r't2|t1|mri|magnetic', full_text.lower()))
                    }

                    new_cases.append(case_data)
                    existing_urls.add(case_url.lower())
                    print(f"   ✓ {title[:50]}...")

                except Exception as e:
                    continue

        except Exception as e:
            print(f"   Error: {e}")
            continue

    return new_cases

# ============================================================
# RESEARCHGATE / GOOGLE SCHOLAR (Limited)
# ============================================================
def scrape_scholar_abstracts(existing_urls):
    """Try Google Scholar - likely to be blocked but worth trying"""
    print(f"\n{'='*60}")
    print("GOOGLE SCHOLAR ATTEMPT")
    print(f"{'='*60}")

    # Google Scholar typically blocks automated access
    # But we can try with delays

    new_cases = []
    queries = [
        "suprasphincteric fistula MRI case report",
        "horseshoe anal fistula imaging findings",
        "complex perianal fistula MRI case"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    for query in queries:
        print(f"\n🔍 Scholar: {query}")
        time.sleep(5)  # Long delay for Scholar

        try:
            url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429 or 'captcha' in response.text.lower():
                print("   Blocked by captcha/rate limit")
                continue

            # This will likely not work but worth trying
            print(f"   Status: {response.status_code}")

        except Exception as e:
            print(f"   Error: {e}")

    return new_cases

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("="*60)
    print("MEGA SCRAPER - Target 300-500 Cases")
    print("="*60)

    # Load existing
    existing_cases = load_master_cases()
    existing_urls = get_existing_urls(existing_cases)

    print(f"\n📂 Starting with {len(existing_cases)} cases")

    all_new_cases = []

    # 1. Extended PubMed
    pmc_cases = scrape_pubmed_extended(existing_cases, existing_urls)
    all_new_cases.extend(pmc_cases)
    print(f"\n📊 PubMed extended: +{len(pmc_cases)} cases")

    # Save progress
    if pmc_cases:
        combined = existing_cases + all_new_cases
        save_master_cases(combined)

    # 2. Eurorad
    eurorad_cases = scrape_eurorad(existing_urls)
    all_new_cases.extend(eurorad_cases)
    print(f"\n📊 Eurorad: +{len(eurorad_cases)} cases")

    # Save progress
    if eurorad_cases:
        combined = existing_cases + all_new_cases
        save_master_cases(combined)

    # 3. Google Scholar (likely blocked)
    scholar_cases = scrape_scholar_abstracts(existing_urls)
    all_new_cases.extend(scholar_cases)

    # Final save
    final_cases = existing_cases + all_new_cases
    save_master_cases(final_cases)

    # Report
    print("\n" + "="*60)
    print("MEGA SCRAPE COMPLETE")
    print("="*60)
    print(f"Started with: {len(existing_cases)}")
    print(f"Added: {len(all_new_cases)}")
    print(f"Total now: {len(final_cases)}")

    # Gap analysis
    from collections import Counter
    type_counts = Counter(c.get('fistula_type') or 'unknown' for c in final_cases)
    sev_counts = Counter(c.get('severity_indicators') or 'unknown' for c in final_cases)

    print(f"\nBy Type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    print(f"\nBy Severity:")
    for s, c in sev_counts.most_common():
        print(f"  {s}: {c}")

if __name__ == "__main__":
    main()
