#!/usr/bin/env python3
"""
Nuclear Scraper for MRI-Crohn Atlas
Target: Scrape all available perianal fistula MRI cases
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
import requests
from bs4 import BeautifulSoup

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"

# Headers to avoid bot detection
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def load_master_cases():
    """Load existing master cases"""
    master_path = TRAINING_DIR / "master_cases.json"
    if master_path.exists():
        with open(master_path, 'r') as f:
            data = json.load(f)
            return data.get('cases', [])
    return []

def get_existing_urls(cases):
    """Get set of existing URLs to avoid duplicates"""
    urls = set()
    for case in cases:
        url = case.get('source_url', '')
        if url:
            # Normalize URL
            url = re.sub(r'\?.*$', '', url)  # Remove query params
            url = url.rstrip('/')
            urls.add(url.lower())
    return urls

def fetch_page(url, delay=2):
    """Fetch a page with rate limiting"""
    time.sleep(delay)
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"   Error fetching {url}: {e}")
        return None

def scrape_radiopaedia_search(search_term, existing_urls, max_pages=5):
    """Search Radiopaedia for cases"""
    cases = []
    base_url = "https://radiopaedia.org"

    for page in range(1, max_pages + 1):
        search_url = f"{base_url}/search?q={quote(search_term)}&scope=cases&page={page}"
        print(f"   Searching: {search_term} (page {page})")

        html = fetch_page(search_url)
        if not html:
            break

        soup = BeautifulSoup(html, 'html.parser')

        # Find case links
        case_links = soup.select('a.search-result-case-title, a[href*="/cases/"]')

        if not case_links:
            print(f"   No more results for '{search_term}'")
            break

        for link in case_links:
            href = link.get('href', '')
            if '/cases/' not in href or '/edit' in href:
                continue

            case_url = urljoin(base_url, href)
            case_url = re.sub(r'\?.*$', '', case_url)

            if case_url.lower() in existing_urls:
                continue

            # Fetch case page
            case_data = scrape_radiopaedia_case(case_url)
            if case_data:
                cases.append(case_data)
                existing_urls.add(case_url.lower())
                print(f"      ✓ Found: {case_data.get('title', 'Unknown')[:50]}")

                # Progress check
                if len(cases) >= 10:
                    print(f"   === Progress: {len(cases)} new cases from '{search_term}' ===")

    return cases

def scrape_radiopaedia_case(url):
    """Scrape a single Radiopaedia case"""
    html = fetch_page(url, delay=1)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Check if it's actually about perianal fistula
    text_content = soup.get_text().lower()
    if not any(kw in text_content for kw in ['perianal', 'anal fistula', 'fistula-in-ano', 'intersphincteric', 'transsphincteric', 'anal sphincter']):
        return None

    # Extract title
    title_elem = soup.select_one('h1.case-title, h1')
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Skip non-perianal fistulas
    skip_titles = ['vesico', 'enterocutaneous', 'tracheo', 'bronchopleural', 'arteriovenous']
    if any(skip in title.lower() for skip in skip_titles):
        return None

    # Extract findings
    findings_section = soup.select_one('div.case-section-findings, [data-section="findings"]')
    findings = ""
    if findings_section:
        findings = findings_section.get_text(strip=True)

    # Try to get from case body if no findings section
    if not findings:
        body = soup.select_one('div.case-body, div.case-content')
        if body:
            # Look for text near "findings" or "imaging"
            all_text = body.get_text(separator='\n')
            findings = all_text

    # Extract discussion
    discussion_section = soup.select_one('div.case-section-discussion, [data-section="discussion"]')
    discussion = discussion_section.get_text(strip=True) if discussion_section else ""

    # Extract presentation/history
    presentation_section = soup.select_one('div.case-section-presentation, [data-section="presentation"]')
    presentation = presentation_section.get_text(strip=True) if presentation_section else ""

    if not findings or len(findings) < 50:
        return None

    # Determine fistula type
    fistula_type = None
    combined = (findings + " " + discussion).lower()
    if 'intersphincteric' in combined:
        fistula_type = 'intersphincteric'
    elif 'transsphincteric' in combined or 'trans-sphincteric' in combined:
        fistula_type = 'transsphincteric'
    elif 'extrasphincteric' in combined:
        fistula_type = 'extrasphincteric'
    elif 'suprasphincteric' in combined:
        fistula_type = 'suprasphincteric'
    elif 'horseshoe' in combined:
        fistula_type = 'horseshoe'
    elif 'complex' in combined or 'multiple' in combined:
        fistula_type = 'complex'

    # Check for abscess
    has_abscess = bool(re.search(r'abscess|collection', combined))

    # Estimate severity
    severity = None
    if 'severe' in combined or 'extensive' in combined:
        severity = 'severe'
    elif 'moderate' in combined:
        severity = 'moderate'
    elif 'mild' in combined or 'minimal' in combined or 'simple' in combined:
        severity = 'mild'
    elif 'healed' in combined or 'fibrotic' in combined:
        severity = 'remission'

    # Check for MRI specific terms
    has_mri = bool(re.search(r't2|t1|hyperintense|hypointense|enhancement|diffusion|dwi|mri|mr\s', combined))

    return {
        "case_id": f"rp_{hashlib.md5(url.encode()).hexdigest()[:8]}",
        "source": "radiopaedia",
        "source_url": url,
        "title": title,
        "clinical_history": presentation,
        "findings_text": findings,
        "diagnosis": "",
        "fistula_type": fistula_type,
        "severity_indicators": severity,
        "treatment_mentioned": None,
        "has_abscess": has_abscess,
        "pediatric": bool(re.search(r'pediatric|paediatric|child|year.?old', combined)),
        "scored_vai": None,
        "scored_magnifi": None,
        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
        "quality_score": 4 if has_mri else 3,
        "has_mri_findings": has_mri
    }

def scrape_radiopaedia_article_cases(article_url, existing_urls):
    """Get case links from a Radiopaedia article page"""
    cases = []
    html = fetch_page(article_url)
    if not html:
        return cases

    soup = BeautifulSoup(html, 'html.parser')

    # Find related cases section
    case_links = soup.select('a[href*="/cases/"]')

    for link in case_links:
        href = link.get('href', '')
        if '/edit' in href or not '/cases/' in href:
            continue

        case_url = urljoin("https://radiopaedia.org", href)
        case_url = re.sub(r'\?.*$', '', case_url)

        if case_url.lower() in existing_urls:
            continue

        case_data = scrape_radiopaedia_case(case_url)
        if case_data:
            cases.append(case_data)
            existing_urls.add(case_url.lower())
            print(f"   ✓ Found from article: {case_data.get('title', 'Unknown')[:50]}")

    return cases

def scrape_pubmed_search(search_term, existing_urls, max_results=50):
    """Search PubMed for case reports"""
    cases = []

    # Use NCBI E-utilities
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = f"{base_url}/esearch.fcgi?db=pmc&term={quote(search_term)}+AND+open+access[filter]&retmax={max_results}&retmode=json"

    print(f"   Searching PubMed: {search_term}")

    try:
        response = requests.get(search_url, headers=HEADERS, timeout=30)
        data = response.json()

        id_list = data.get('esearchresult', {}).get('idlist', [])

        if not id_list:
            print(f"   No PMC results for '{search_term}'")
            return cases

        print(f"   Found {len(id_list)} PMC articles")

        # Fetch each article
        for pmcid in id_list[:20]:  # Limit to 20 per search
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"

            if pmc_url.lower() in existing_urls:
                continue

            time.sleep(1)

            # Try to fetch article content
            article_data = scrape_pmc_article(pmcid)
            if article_data:
                cases.append(article_data)
                existing_urls.add(pmc_url.lower())
                print(f"   ✓ Found: PMC{pmcid}")

    except Exception as e:
        print(f"   Error searching PubMed: {e}")

    return cases

def scrape_pmc_article(pmcid):
    """Scrape a PMC article for case data"""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
    html = fetch_page(url, delay=1)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Get title
    title_elem = soup.select_one('h1.content-title')
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Check if relevant
    full_text = soup.get_text().lower()
    if not any(kw in full_text for kw in ['perianal fistula', 'anal fistula', 'fistula-in-ano', 'crohn.*perianal', 'perianal crohn']):
        return None

    # Skip review articles
    if any(kw in title.lower() for kw in ['review', 'meta-analysis', 'systematic', 'guidelines']):
        return None

    # Extract body text
    body = soup.select_one('div.jig-ncbiinpagenav')
    if not body:
        body = soup.select_one('article')

    text = body.get_text(separator='\n', strip=True) if body else full_text

    # Look for case presentation or findings sections
    case_text = ""
    sections = soup.select('div.tsec')
    for section in sections:
        header = section.select_one('h2, h3')
        if header:
            header_text = header.get_text().lower()
            if any(kw in header_text for kw in ['case', 'finding', 'imaging', 'mri', 'result']):
                case_text += section.get_text(separator=' ', strip=True) + "\n"

    if not case_text:
        case_text = text[:2000]  # First 2000 chars

    if len(case_text) < 100:
        return None

    # Determine type
    fistula_type = None
    if 'intersphincteric' in case_text.lower():
        fistula_type = 'intersphincteric'
    elif 'transsphincteric' in case_text.lower():
        fistula_type = 'transsphincteric'
    elif 'extrasphincteric' in case_text.lower():
        fistula_type = 'extrasphincteric'
    elif 'horseshoe' in case_text.lower():
        fistula_type = 'horseshoe'
    elif 'complex' in case_text.lower():
        fistula_type = 'complex'

    return {
        "case_id": f"pmc_{pmcid}",
        "source": "pubmed_central",
        "source_url": url,
        "title": title,
        "clinical_history": "",
        "findings_text": case_text,
        "diagnosis": "",
        "fistula_type": fistula_type,
        "severity_indicators": None,
        "treatment_mentioned": None,
        "has_abscess": 'abscess' in case_text.lower(),
        "pediatric": bool(re.search(r'pediatric|paediatric|child', case_text.lower())),
        "scored_vai": None,
        "scored_magnifi": None,
        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
        "quality_score": 3,
        "has_mri_findings": bool(re.search(r't2|t1|mri|hyperintense', case_text.lower()))
    }

def scrape_eurorad_search(search_term, existing_urls, max_pages=3):
    """Search Eurorad for cases"""
    cases = []
    base_url = "https://www.eurorad.org"

    for page in range(1, max_pages + 1):
        search_url = f"{base_url}/advanced-search?search={quote(search_term)}&page={page}"
        print(f"   Searching Eurorad: {search_term} (page {page})")

        html = fetch_page(search_url)
        if not html:
            break

        soup = BeautifulSoup(html, 'html.parser')

        # Find case links
        case_links = soup.select('a[href*="/case/"]')

        if not case_links:
            break

        for link in case_links:
            href = link.get('href', '')
            case_url = urljoin(base_url, href)

            if case_url.lower() in existing_urls:
                continue

            case_data = scrape_eurorad_case(case_url)
            if case_data:
                cases.append(case_data)
                existing_urls.add(case_url.lower())
                print(f"   ✓ Found: {case_data.get('title', 'Unknown')[:50]}")

    return cases

def scrape_eurorad_case(url):
    """Scrape a Eurorad case"""
    html = fetch_page(url, delay=1)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    text_content = soup.get_text().lower()
    if not any(kw in text_content for kw in ['perianal', 'anal fistula', 'fistula-in-ano', 'intersphincteric', 'transsphincteric']):
        return None

    title_elem = soup.select_one('h1')
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Find imaging findings
    findings_section = soup.select_one('[id*="imaging"], .imaging-findings')
    findings = findings_section.get_text(strip=True) if findings_section else ""

    if not findings:
        body = soup.select_one('.case-content, article')
        findings = body.get_text(strip=True) if body else ""

    if len(findings) < 50:
        return None

    combined = findings.lower()
    fistula_type = None
    if 'intersphincteric' in combined:
        fistula_type = 'intersphincteric'
    elif 'transsphincteric' in combined:
        fistula_type = 'transsphincteric'
    elif 'extrasphincteric' in combined:
        fistula_type = 'extrasphincteric'
    elif 'horseshoe' in combined:
        fistula_type = 'horseshoe'

    return {
        "case_id": f"er_{hashlib.md5(url.encode()).hexdigest()[:8]}",
        "source": "eurorad",
        "source_url": url,
        "title": title,
        "clinical_history": "",
        "findings_text": findings,
        "diagnosis": "",
        "fistula_type": fistula_type,
        "severity_indicators": None,
        "treatment_mentioned": None,
        "has_abscess": 'abscess' in combined,
        "pediatric": bool(re.search(r'pediatric|paediatric|child', combined)),
        "scored_vai": None,
        "scored_magnifi": None,
        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
        "quality_score": 4,
        "has_mri_findings": bool(re.search(r't2|t1|mri|hyperintense', combined))
    }

def save_scraped_cases(new_cases, existing_cases):
    """Save combined cases to master file"""
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

    print(f"\n💾 Saved {len(all_cases)} total cases to {master_path}")

def update_scrape_log(source, count, search_terms):
    """Update scrape log"""
    log_path = TRAINING_DIR / "scrape_log.md"

    entry = f"""
## {source} - {datetime.now().strftime("%Y-%m-%d %H:%M")}
- Cases found: {count}
- Search terms: {', '.join(search_terms)}
"""

    mode = 'a' if log_path.exists() else 'w'
    with open(log_path, mode) as f:
        if mode == 'w':
            f.write("# Scrape Log\n\n")
        f.write(entry)

def main():
    print("=" * 60)
    print("NUCLEAR SCRAPE - MRI-Crohn Atlas")
    print("Target: 200+ unique real MRI cases")
    print("=" * 60)

    # Load existing cases
    existing_cases = load_master_cases()
    existing_urls = get_existing_urls(existing_cases)
    print(f"\n📂 Starting with {len(existing_cases)} existing cases")

    all_new_cases = []

    # ===========================================
    # TIER 1: RADIOPAEDIA (High Yield)
    # ===========================================
    print("\n" + "=" * 60)
    print("TIER 1: RADIOPAEDIA")
    print("=" * 60)

    radiopaedia_search_terms = [
        "perianal fistula",
        "anal fistula",
        "fistula in ano",
        "perianal abscess",
        "perianal crohn",
        "intersphincteric fistula",
        "transsphincteric fistula",
        "horseshoe fistula",
        "anorectal fistula",
        "extrasphincteric fistula",
        "suprasphincteric fistula",
        "complex anal fistula",
        "ischioanal abscess",
        "ischiorectal abscess"
    ]

    rp_cases = []
    for term in radiopaedia_search_terms:
        new_cases = scrape_radiopaedia_search(term, existing_urls, max_pages=3)
        rp_cases.extend(new_cases)
        print(f"   Subtotal from '{term}': {len(new_cases)} cases")

        # Save progress every 10 cases
        if len(rp_cases) >= 10:
            save_scraped_cases(rp_cases + all_new_cases, existing_cases)
            print(f"   === Progress saved: {len(rp_cases)} new Radiopaedia cases ===")

    all_new_cases.extend(rp_cases)
    update_scrape_log("Radiopaedia", len(rp_cases), radiopaedia_search_terms)

    # Also check article pages
    article_urls = [
        "https://radiopaedia.org/articles/perianal-fistula",
        "https://radiopaedia.org/articles/crohn-disease",
        "https://radiopaedia.org/articles/fistula-in-ano",
        "https://radiopaedia.org/articles/anorectal-abscess"
    ]

    for article_url in article_urls:
        print(f"\n   Checking article: {article_url}")
        article_cases = scrape_radiopaedia_article_cases(article_url, existing_urls)
        all_new_cases.extend(article_cases)

    print(f"\n✓ RADIOPAEDIA COMPLETE: {len(rp_cases)} new cases")

    # ===========================================
    # TIER 2: PUBMED CENTRAL
    # ===========================================
    print("\n" + "=" * 60)
    print("TIER 2: PUBMED CENTRAL")
    print("=" * 60)

    pubmed_search_terms = [
        "perianal fistula MRI case report",
        "anal fistula magnetic resonance",
        "perianal Crohn's disease MRI",
        "Van Assche index case",
        "MAGNIFI-CD case",
        "complex anal fistula case",
        "horseshoe fistula MRI",
        "transsphincteric fistula imaging"
    ]

    pmc_cases = []
    for term in pubmed_search_terms:
        new_cases = scrape_pubmed_search(term, existing_urls, max_results=30)
        pmc_cases.extend(new_cases)

    all_new_cases.extend(pmc_cases)
    update_scrape_log("PubMed Central", len(pmc_cases), pubmed_search_terms)
    print(f"\n✓ PUBMED COMPLETE: {len(pmc_cases)} new cases")

    # ===========================================
    # TIER 3: EURORAD
    # ===========================================
    print("\n" + "=" * 60)
    print("TIER 3: EURORAD")
    print("=" * 60)

    eurorad_search_terms = [
        "perianal fistula",
        "anal fistula",
        "perianal abscess",
        "Crohn perianal"
    ]

    er_cases = []
    for term in eurorad_search_terms:
        new_cases = scrape_eurorad_search(term, existing_urls, max_pages=2)
        er_cases.extend(new_cases)

    all_new_cases.extend(er_cases)
    update_scrape_log("Eurorad", len(er_cases), eurorad_search_terms)
    print(f"\n✓ EURORAD COMPLETE: {len(er_cases)} new cases")

    # ===========================================
    # FINAL SAVE
    # ===========================================
    save_scraped_cases(all_new_cases, existing_cases)

    # Final stats
    total_cases = len(existing_cases) + len(all_new_cases)
    print("\n" + "=" * 60)
    print("=== FINAL SCRAPE REPORT ===")
    print("=" * 60)
    print(f"Total unique cases: {total_cases}")
    print(f"Target: 200")
    print(f"\nBy source:")
    print(f"  - Radiopaedia (new): {len(rp_cases)}")
    print(f"  - PubMed Central (new): {len(pmc_cases)}")
    print(f"  - Eurorad (new): {len(er_cases)}")
    print(f"  - Previous: {len(existing_cases)}")

    if total_cases < 200:
        print(f"\n⚠️ Still need {200 - total_cases} more cases")
    else:
        print(f"\n✅ TARGET REACHED!")

if __name__ == "__main__":
    main()
