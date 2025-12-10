#!/usr/bin/env python3
"""
MEGA SCRAPER V2 - Target 700 cases
Proper rate limiting, multiple APIs, bypass blocks
"""

import json
import os
import re
import time
import hashlib
import random
from datetime import datetime
from pathlib import Path
import requests
from xml.etree import ElementTree as ET

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
TRAINING_DIR = BASE_DIR / "data" / "training"
MASTER_FILE = TRAINING_DIR / "master_cases.json"

# Rate limiting
BASE_DELAY = 3  # seconds between requests
MAX_RETRIES = 2

# User agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

def rate_limit_request(url, params=None, timeout=60, retry_count=0):
    """Make a rate-limited request with retries"""
    time.sleep(BASE_DELAY + random.uniform(0, 2))

    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=timeout)

        if response.status_code == 429:
            if retry_count < MAX_RETRIES:
                print(f"   Rate limited, waiting 30s...")
                time.sleep(30)
                return rate_limit_request(url, params, timeout, retry_count + 1)
            return None

        if response.status_code == 403:
            if retry_count < MAX_RETRIES:
                print(f"   403 Forbidden, waiting 15s and retrying...")
                time.sleep(15)
                return rate_limit_request(url, params, timeout, retry_count + 1)
            return None

        response.raise_for_status()
        return response
    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(10)
            return rate_limit_request(url, params, timeout, retry_count + 1)
        return None

def load_master_cases():
    if MASTER_FILE.exists():
        with open(MASTER_FILE, 'r') as f:
            data = json.load(f)
            return data.get('cases', [])
    return []

def get_existing_urls(cases):
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
    with open(MASTER_FILE, 'w') as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_cases": len(cases)
            },
            "cases": cases
        }, f, indent=2)

def is_relevant(text):
    text = text.lower()
    keywords = [
        'perianal fistula', 'anal fistula', 'fistula-in-ano', 'fistula in ano',
        'intersphincteric', 'transsphincteric', 'suprasphincteric', 'extrasphincteric',
        'horseshoe fistula', 'ischioanal', 'ischiorectal', 'perianal crohn',
        'anal sphincter', 'internal opening', 'external opening', 'fistula tract',
        'perianal abscess', 'cryptoglandular'
    ]
    return any(kw in text for kw in keywords)

def classify_fistula_type(text):
    text = text.lower()
    if 'suprasphincteric' in text or 'supra-sphincteric' in text:
        return 'suprasphincteric'
    elif 'extrasphincteric' in text or 'extra-sphincteric' in text:
        return 'extrasphincteric'
    elif 'horseshoe' in text:
        return 'horseshoe'
    elif 'transsphincteric' in text or 'trans-sphincteric' in text:
        return 'transsphincteric'
    elif 'intersphincteric' in text or 'inter-sphincteric' in text:
        return 'intersphincteric'
    elif 'complex' in text or 'multiple' in text or 'branching' in text:
        return 'complex'
    return None

def classify_severity(text):
    text = text.lower()
    if 'severe' in text or 'extensive' in text or 'fulminant' in text or 'large abscess' in text:
        return 'severe'
    elif 'moderate' in text or 'grade 3' in text or 'grade 4' in text:
        return 'moderate'
    elif 'mild' in text or 'minimal' in text or 'simple' in text or 'grade 1' in text or 'grade 2' in text:
        return 'mild'
    elif 'healed' in text or 'remission' in text or 'fibrotic' in text or 'inactive' in text:
        return 'remission'
    return None

# ============================================================
# EUROPE PMC API (separate from US PMC)
# ============================================================
def scrape_europe_pmc(existing_urls, cases, progress_callback):
    """Scrape Europe PMC for additional cases"""
    print(f"\n{'='*60}")
    print("EUROPE PMC SCRAPE")
    print(f"{'='*60}")

    queries = [
        "perianal fistula MRI case",
        "anal fistula magnetic resonance",
        "transsphincteric fistula imaging",
        "suprasphincteric fistula",
        "horseshoe perianal abscess",
        "Crohn perianal MRI",
        "fistula-in-ano imaging",
        "anorectal sepsis MRI",
        "intersphincteric fistula case",
        "ischioanal abscess MRI"
    ]

    new_cases = []

    for query in queries:
        print(f"\n🔍 Europe PMC: {query}")

        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"{query} AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": 100,
            "resultType": "core"
        }

        response = rate_limit_request(url, params)
        if not response:
            continue

        try:
            data = response.json()
            results = data.get('resultList', {}).get('result', [])
            print(f"   Found {len(results)} results")

            for result in results:
                pmcid = result.get('pmcid', '')
                pmid = result.get('pmid', '')
                title = result.get('title', '')
                abstract = result.get('abstractText', '')

                full_text = f"{title} {abstract}"

                if not is_relevant(full_text):
                    continue

                # Skip reviews
                if any(kw in full_text.lower() for kw in ['systematic review', 'meta-analysis', 'guidelines']):
                    continue

                case_url = f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                if case_url.lower() in existing_urls:
                    continue

                has_mri = bool(re.search(r't2|t1|hyperintense|mri|magnetic resonance', full_text.lower()))

                case = {
                    "case_id": f"epmc_{pmcid or pmid}",
                    "source": "europe_pmc",
                    "source_url": case_url,
                    "title": title[:200],
                    "clinical_history": "",
                    "findings_text": abstract[:3000] if abstract else "",
                    "diagnosis": "",
                    "fistula_type": classify_fistula_type(full_text),
                    "severity_indicators": classify_severity(full_text),
                    "has_abscess": 'abscess' in full_text.lower(),
                    "pediatric": bool(re.search(r'pediatric|paediatric|child', full_text.lower())),
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                    "quality_score": 4 if has_mri else 3,
                    "has_mri_findings": has_mri
                }

                new_cases.append(case)
                existing_urls.add(case_url.lower())
                cases.append(case)

                progress_callback(len(cases), case['title'][:40])

        except Exception as e:
            print(f"   Error: {e}")
            continue

    return new_cases

# ============================================================
# OPENALEX API (completely free, no limits)
# ============================================================
def scrape_openalex(existing_urls, cases, progress_callback):
    """Scrape OpenAlex for cases"""
    print(f"\n{'='*60}")
    print("OPENALEX SCRAPE")
    print(f"{'='*60}")

    queries = [
        "perianal fistula MRI",
        "anal fistula imaging case report",
        "transsphincteric fistula",
        "suprasphincteric fistula MRI",
        "horseshoe abscess perianal",
        "Crohn disease perianal fistula",
        "intersphincteric fistula",
        "extrasphincteric fistula",
        "anorectal abscess MRI",
        "Van Assche index",
        "MAGNIFI-CD fistula"
    ]

    new_cases = []

    for query in queries:
        print(f"\n🔍 OpenAlex: {query}")

        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "filter": "is_oa:true,type:article",
            "per-page": 100,
            "mailto": "research@example.com"  # Polite pool
        }

        response = rate_limit_request(url, params)
        if not response:
            continue

        try:
            data = response.json()
            results = data.get('results', [])
            print(f"   Found {len(results)} results")

            for result in results:
                title = result.get('title', '') or ''

                # Get abstract
                abstract_obj = result.get('abstract_inverted_index', {})
                abstract = ""
                if abstract_obj:
                    # Reconstruct abstract from inverted index
                    words = {}
                    for word, positions in abstract_obj.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = ' '.join([words[i] for i in sorted(words.keys())])

                full_text = f"{title} {abstract}"

                if not is_relevant(full_text):
                    continue

                # Skip reviews
                if any(kw in full_text.lower() for kw in ['systematic review', 'meta-analysis', 'guideline']):
                    continue

                doi = result.get('doi', '')
                work_id = result.get('id', '').split('/')[-1]
                case_url = doi if doi else f"https://openalex.org/works/{work_id}"

                if case_url.lower() in existing_urls:
                    continue

                has_mri = bool(re.search(r't2|t1|hyperintense|mri|magnetic resonance', full_text.lower()))

                case = {
                    "case_id": f"openalex_{work_id}",
                    "source": "openalex",
                    "source_url": case_url,
                    "title": title[:200],
                    "clinical_history": "",
                    "findings_text": abstract[:3000] if abstract else "",
                    "diagnosis": "",
                    "fistula_type": classify_fistula_type(full_text),
                    "severity_indicators": classify_severity(full_text),
                    "has_abscess": 'abscess' in full_text.lower(),
                    "pediatric": bool(re.search(r'pediatric|paediatric|child', full_text.lower())),
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                    "quality_score": 4 if has_mri else 3,
                    "has_mri_findings": has_mri
                }

                new_cases.append(case)
                existing_urls.add(case_url.lower())
                cases.append(case)

                progress_callback(len(cases), case['title'][:40])

        except Exception as e:
            print(f"   Error: {e}")
            continue

    return new_cases

# ============================================================
# SEMANTIC SCHOLAR API
# ============================================================
def scrape_semantic_scholar(existing_urls, cases, progress_callback):
    """Scrape Semantic Scholar"""
    print(f"\n{'='*60}")
    print("SEMANTIC SCHOLAR SCRAPE")
    print(f"{'='*60}")

    queries = [
        "perianal fistula MRI case report",
        "anal fistula magnetic resonance imaging",
        "transsphincteric fistula",
        "suprasphincteric anal fistula",
        "horseshoe perianal abscess",
        "Crohn's perianal fistula imaging",
        "intersphincteric fistula MRI",
        "anorectal sepsis imaging"
    ]

    new_cases = []

    for query in queries:
        print(f"\n🔍 Semantic Scholar: {query}")

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 100,
            "fields": "title,abstract,url,openAccessPdf,publicationDate"
        }

        response = rate_limit_request(url, params)
        if not response:
            continue

        try:
            data = response.json()
            results = data.get('data', [])
            print(f"   Found {len(results)} results")

            for result in results:
                title = result.get('title', '') or ''
                abstract = result.get('abstract', '') or ''

                full_text = f"{title} {abstract}"

                if not is_relevant(full_text):
                    continue

                if any(kw in full_text.lower() for kw in ['systematic review', 'meta-analysis']):
                    continue

                paper_id = result.get('paperId', '')
                case_url = result.get('url', '') or f"https://www.semanticscholar.org/paper/{paper_id}"

                if case_url.lower() in existing_urls:
                    continue

                has_mri = bool(re.search(r't2|t1|hyperintense|mri|magnetic resonance', full_text.lower()))

                case = {
                    "case_id": f"s2_{paper_id[:20]}",
                    "source": "semantic_scholar",
                    "source_url": case_url,
                    "title": title[:200],
                    "clinical_history": "",
                    "findings_text": abstract[:3000] if abstract else "",
                    "diagnosis": "",
                    "fistula_type": classify_fistula_type(full_text),
                    "severity_indicators": classify_severity(full_text),
                    "has_abscess": 'abscess' in full_text.lower(),
                    "pediatric": bool(re.search(r'pediatric|paediatric|child', full_text.lower())),
                    "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                    "quality_score": 4 if has_mri else 3,
                    "has_mri_findings": has_mri
                }

                new_cases.append(case)
                existing_urls.add(case_url.lower())
                cases.append(case)

                progress_callback(len(cases), case['title'][:40])

        except Exception as e:
            print(f"   Error: {e}")
            continue

    return new_cases

# ============================================================
# EXTENDED PUBMED (more queries)
# ============================================================
def scrape_pubmed_extended_v2(existing_urls, cases, progress_callback):
    """Extended PubMed with more targeted queries"""
    print(f"\n{'='*60}")
    print("EXTENDED PUBMED V2")
    print(f"{'='*60}")

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    queries = [
        # Target suprasphincteric specifically
        "suprasphincteric fistula",
        "suprasphincteric anal fistula MRI",
        "high anal fistula MRI",
        "supralevator fistula",
        "Parks type IV fistula",

        # Target moderate severity
        "moderate perianal Crohn MRI",
        "grade 3 perianal fistula",
        "grade 4 perianal fistula",

        # More transsphincteric
        "high transsphincteric fistula",
        "low transsphincteric fistula",
        "trans sphincteric anal fistula MRI",

        # Specific MRI features
        "T2 hyperintense fistula tract",
        "perianal MRI gadolinium fistula",
        "diffusion weighted imaging anal fistula",

        # Treatment-related
        "fistula plug MRI",
        "VAAFT fistula MRI",
        "FiLaC fistula laser",

        # International
        "perianal fistula MRI India",
        "perianal fistula imaging China",
        "anal fistula MRI Japan",

        # Complications
        "rectovaginal fistula MRI",
        "rectourethral fistula MRI",
        "pouch fistula MRI"
    ]

    new_cases = []
    seen_pmcids = set()

    # Get existing PMC IDs
    for case in cases:
        url = case.get('source_url', '')
        match = re.search(r'PMC(\d+)', url)
        if match:
            seen_pmcids.add(match.group(1))
        case_id = case.get('case_id', '')
        if 'pmc_' in case_id:
            seen_pmcids.add(case_id.replace('pmc_', ''))

    for query in queries:
        print(f"\n🔍 PubMed: {query}")

        # Search
        search_url = f"{EUTILS_BASE}/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"{query} AND open access[filter]",
            "retmax": 100,
            "retmode": "json"
        }

        response = rate_limit_request(search_url, params)
        if not response:
            continue

        try:
            data = response.json()
            pmcids = data.get('esearchresult', {}).get('idlist', [])
            print(f"   Found {len(pmcids)} IDs")

            for pmcid in pmcids:
                if pmcid in seen_pmcids:
                    continue
                seen_pmcids.add(pmcid)

                # Fetch article
                time.sleep(0.5)
                fetch_url = f"{EUTILS_BASE}/efetch.fcgi"
                fetch_params = {"db": "pmc", "id": pmcid, "retmode": "xml"}

                fetch_response = rate_limit_request(fetch_url, fetch_params)
                if not fetch_response:
                    continue

                try:
                    root = ET.fromstring(fetch_response.text)

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
                        continue

                    if any(kw in full_text.lower() for kw in ['systematic review', 'meta-analysis', 'guidelines']):
                        continue

                    case_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"

                    if case_url.lower() in existing_urls:
                        continue

                    findings = body_text if len(body_text) > 100 else abstract
                    if len(findings) < 50:
                        continue

                    has_mri = bool(re.search(r't2|t1|hyperintense|mri|magnetic resonance|diffusion', full_text.lower()))
                    quality = 4 if has_mri else 3
                    if len(body_text) > 500:
                        quality = min(5, quality + 1)

                    case = {
                        "case_id": f"pmc_{pmcid}",
                        "source": "pubmed_central",
                        "source_url": case_url,
                        "title": title[:200],
                        "clinical_history": "",
                        "findings_text": findings[:3000],
                        "diagnosis": "",
                        "fistula_type": classify_fistula_type(full_text),
                        "severity_indicators": classify_severity(full_text),
                        "has_abscess": 'abscess' in full_text.lower(),
                        "pediatric": bool(re.search(r'pediatric|paediatric|child|infant', full_text.lower())),
                        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
                        "quality_score": quality,
                        "has_mri_findings": has_mri
                    }

                    new_cases.append(case)
                    existing_urls.add(case_url.lower())
                    cases.append(case)

                    progress_callback(len(cases), title[:40])

                except Exception as e:
                    continue

        except Exception as e:
            print(f"   Error: {e}")
            continue

    return new_cases

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("MEGA SCRAPER V2 - TARGET 700 CASES")
    print("="*60)

    cases = load_master_cases()
    existing_urls = get_existing_urls(cases)

    start_count = len(cases)
    print(f"\n📂 Starting with {start_count} cases")

    last_milestone = (start_count // 25) * 25

    def progress_callback(total, title):
        nonlocal last_milestone
        if total >= last_milestone + 25:
            last_milestone = (total // 25) * 25
            print(f"\n   === MILESTONE: {total} cases ===")
            save_master_cases(cases)

    # Run all scrapers
    print("\n" + "="*60)

    # 1. Extended PubMed
    pmc_cases = scrape_pubmed_extended_v2(existing_urls, cases, progress_callback)
    print(f"\n📊 Extended PubMed: +{len(pmc_cases)} new cases")
    save_master_cases(cases)

    # 2. Europe PMC
    epmc_cases = scrape_europe_pmc(existing_urls, cases, progress_callback)
    print(f"\n📊 Europe PMC: +{len(epmc_cases)} new cases")
    save_master_cases(cases)

    # 3. OpenAlex
    oa_cases = scrape_openalex(existing_urls, cases, progress_callback)
    print(f"\n📊 OpenAlex: +{len(oa_cases)} new cases")
    save_master_cases(cases)

    # 4. Semantic Scholar
    s2_cases = scrape_semantic_scholar(existing_urls, cases, progress_callback)
    print(f"\n📊 Semantic Scholar: +{len(s2_cases)} new cases")
    save_master_cases(cases)

    # Final save
    save_master_cases(cases)

    # Report
    print("\n" + "="*60)
    print("MEGA SCRAPE V2 COMPLETE")
    print("="*60)
    print(f"Started with: {start_count}")
    print(f"Total now: {len(cases)}")
    print(f"Added: {len(cases) - start_count}")

    # Gap analysis
    from collections import Counter
    type_counts = Counter(c.get('fistula_type') or 'unknown' for c in cases)
    sev_counts = Counter(c.get('severity_indicators') or 'unknown' for c in cases)

    print(f"\nBy Type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    print(f"\nBy Severity:")
    for s, c in sev_counts.most_common():
        print(f"  {s}: {c}")

if __name__ == "__main__":
    main()
