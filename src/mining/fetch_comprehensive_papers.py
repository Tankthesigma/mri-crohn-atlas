#!/usr/bin/env python3
"""
Comprehensive Paper Mining Script for ISEF 2026 - V2 BROAD SEARCH
==================================================================
Uses SIMPLE boolean queries to maximize hits, then filters locally.

Strategy: Cast a wide net, filter with code.

Author: Tanmay
Date: December 2025
"""

import os
import sys
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import threading

import requests
from Bio import Entrez

# ============================================================================
# CONFIGURATION
# ============================================================================

Entrez.email = "johs03047@gmail.com"
Entrez.api_key = "e6dbad0565fd2136d119de63f233dbd92108"

# Rate limiting: 10 requests/second with API key
RATE_LIMIT = 10
REQUEST_INTERVAL = 1.0 / RATE_LIMIT

# Search configuration - BROAD APPROACH
RETMAX = 500  # Get 500 raw hits per term, filter locally

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "papers"
PDF_DIR = DATA_DIR / "raw_pdfs"
LOG_FILE = DATA_DIR / "download_log.json"
METADATA_FILE = DATA_DIR / "paper_metadata.json"
PROGRESS_FILE = DATA_DIR / "download_progress.json"

# Setup logging
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "fetch.log")
    ]
)
logger = logging.getLogger(__name__)

# Rate limiter
_last_request_time = 0
_rate_lock = threading.Lock()


# ============================================================================
# SIMPLE BROAD SEARCH TERMS WITH CLUSTER ASSIGNMENT
# ============================================================================

SEARCH_TERMS = [
    # Cluster A: Biologics
    {"query": '("Infliximab" AND "Fistula")', "cluster": "Cluster_A_Biologics"},
    {"query": '("Adalimumab" AND "Fistula")', "cluster": "Cluster_A_Biologics"},
    {"query": '("Ustekinumab" AND "Fistula")', "cluster": "Cluster_B_New_Guard"},
    {"query": '("Vedolizumab" AND "Fistula")', "cluster": "Cluster_B_New_Guard"},

    # Cluster C: Surgical
    {"query": '("Seton" AND "Fistula")', "cluster": "Cluster_C_Surgical"},
    {"query": '("Fistulotomy" AND "Fistula")', "cluster": "Cluster_C_Surgical"},
    {"query": '("LIFT" AND "Fistula")', "cluster": "Cluster_D_Reconstruction"},
    {"query": '("VAAFT" AND "Fistula")', "cluster": "Cluster_D_Reconstruction"},

    # Cluster E: Regenerative
    {"query": '("Stem Cells" AND "Fistula")', "cluster": "Cluster_E_Regenerative"},
    {"query": '("Darvadstrocel" AND "Fistula")', "cluster": "Cluster_E_Regenerative"},
    {"query": '("Fibrin Glue" AND "Fistula")', "cluster": "Cluster_E_Regenerative"},

    # Additional broad terms -> Cluster A (general Crohn's)
    {"query": '("Perianal" AND "Crohn")', "cluster": "Cluster_A_Biologics"},
    {"query": '("Anal Fistula" AND "Treatment")', "cluster": "Cluster_C_Surgical"},
    {"query": '("Fistula-in-ano" AND "MRI")', "cluster": "Cluster_C_Surgical"},
    {"query": '("Advancement Flap" AND "Fistula")', "cluster": "Cluster_D_Reconstruction"},
]


# ============================================================================
# RELEVANCE FILTER - KEEP THIS
# ============================================================================

LOCATION_KEYWORDS = {
    "perianal", "peri-anal", "peri anal",
    "anal", "rectovaginal", "anorectal",
    "ano-rectal", "recto-vaginal",
    "anoperineal", "ano-perineal",
    "perineal", "cryptoglandular",
    "fistula-in-ano", "fistula in ano"
}

PATHOLOGY_KEYWORDS = {
    "fistula", "fistulae", "fistulizing", "fistulising",
    "crohn", "crohn's", "crohns",
    "ibd", "inflammatory bowel",
    "abscess", "abscesses",
    "pfcd"
}


def is_relevant(text: str) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Strict relevance filter.
    Returns True ONLY if text contains BOTH:
    - Location keyword (anal, perianal, etc.)
    - Pathology keyword (fistula, Crohn, etc.)
    """
    if not text:
        return False, {"location": [], "pathology": []}

    text_lower = text.lower()

    location_matches = []
    for kw in LOCATION_KEYWORDS:
        if kw in text_lower:
            location_matches.append(kw)

    pathology_matches = []
    for kw in PATHOLOGY_KEYWORDS:
        if kw in text_lower:
            pathology_matches.append(kw)

    is_rel = len(location_matches) > 0 and len(pathology_matches) > 0

    return is_rel, {"location": location_matches, "pathology": pathology_matches}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PRISMALog:
    start_time: str = ""
    end_time: str = ""
    search_terms: Dict[str, Dict] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=lambda: {
        "total_found": 0,
        "unique_papers": 0,
        "relevant": 0,
        "downloaded": 0,
        "skipped_irrelevant": 0,
        "skipped_no_pdf": 0,
        "skipped_error": 0,
        "duplicates_removed": 0
    })
    errors: List[Dict] = field(default_factory=list)


# ============================================================================
# RATE LIMITER
# ============================================================================

def rate_limited_request():
    """Ensures we don't exceed 10 requests/second"""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()


# ============================================================================
# PMC FUNCTIONS - SEARCH PMC DIRECTLY
# ============================================================================

def search_pmc(query: str, retmax: int = RETMAX) -> Tuple[List[str], int]:
    """
    Search PMC directly (not PubMed) for free full text papers.
    Returns (list of PMCIDs, total count).
    """
    rate_limited_request()

    try:
        handle = Entrez.esearch(
            db="pmc",  # Search PMC directly - this ensures free full text
            term=query,
            retmax=retmax,
            sort="relevance"
        )
        record = Entrez.read(handle)
        handle.close()

        pmcids = record.get("IdList", [])
        total_count = int(record.get("Count", 0))

        logger.info(f"Query: {query[:60]}...")
        logger.info(f"  Found: {total_count} total, returning {len(pmcids)}")

        return pmcids, total_count

    except Exception as e:
        logger.error(f"Search error: {e}")
        return [], 0


def fetch_pmc_details(pmcids: List[str]) -> List[Dict]:
    """
    Fetch metadata for PMC articles.
    """
    if not pmcids:
        return []

    all_papers = []
    batch_size = 50

    for i in range(0, len(pmcids), batch_size):
        batch = pmcids[i:i + batch_size]
        rate_limited_request()

        try:
            handle = Entrez.efetch(
                db="pmc",
                id=",".join(batch),
                rettype="xml",
                retmode="xml"
            )
            content = handle.read()
            handle.close()

            # Parse the XML manually for PMC format
            papers = parse_pmc_xml(content, batch)
            all_papers.extend(papers)

            logger.info(f"  Fetched batch {i//batch_size + 1}: {len(papers)} papers")

        except Exception as e:
            logger.error(f"Fetch error for batch {i//batch_size + 1}: {e}")
            # Try individual fetches as fallback
            for pmcid in batch:
                paper = fetch_single_pmc(pmcid)
                if paper:
                    all_papers.append(paper)

    return all_papers


def fetch_single_pmc(pmcid: str) -> Optional[Dict]:
    """Fetch single PMC article metadata"""
    rate_limited_request()

    try:
        handle = Entrez.esummary(db="pmc", id=pmcid)
        record = Entrez.read(handle)
        handle.close()

        if record:
            doc = record[0]
            return {
                "pmcid": f"PMC{pmcid}",
                "title": doc.get("Title", ""),
                "authors": doc.get("AuthorList", []),
                "first_author": doc.get("AuthorList", ["Unknown"])[0].split()[0] if doc.get("AuthorList") else "Unknown",
                "year": str(doc.get("PubDate", ""))[:4],
                "journal": doc.get("Source", ""),
                "abstract": "",  # Summary doesn't include abstract
                "doi": doc.get("DOI", "")
            }
    except Exception as e:
        logger.warning(f"Single fetch failed for {pmcid}: {e}")

    return None


def parse_pmc_xml(xml_content: bytes, pmcids: List[str]) -> List[Dict]:
    """Parse PMC XML response to extract paper metadata"""
    papers = []

    try:
        content = xml_content.decode('utf-8', errors='ignore')

        # Split by article
        articles = re.split(r'<article[^>]*>', content)

        for i, article in enumerate(articles[1:], 1):  # Skip first empty split
            try:
                paper = {}

                # PMC ID
                pmcid_match = re.search(r'<article-id pub-id-type="pmc">(\d+)</article-id>', article)
                paper["pmcid"] = f"PMC{pmcid_match.group(1)}" if pmcid_match else f"PMC{pmcids[i-1] if i-1 < len(pmcids) else ''}"

                # Title
                title_match = re.search(r'<article-title[^>]*>(.+?)</article-title>', article, re.DOTALL)
                paper["title"] = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ""

                # Authors
                authors = []
                author_matches = re.findall(r'<surname>([^<]+)</surname>\s*<given-names>([^<]+)</given-names>', article)
                for surname, given in author_matches:
                    authors.append(f"{surname} {given}")
                paper["authors"] = authors
                paper["first_author"] = authors[0].split()[0] if authors else "Unknown"

                # Year
                year_match = re.search(r'<year>(\d{4})</year>', article)
                paper["year"] = year_match.group(1) if year_match else "Unknown"

                # Journal
                journal_match = re.search(r'<journal-title[^>]*>([^<]+)</journal-title>', article)
                paper["journal"] = journal_match.group(1).strip() if journal_match else ""

                # Abstract
                abstract_match = re.search(r'<abstract[^>]*>(.+?)</abstract>', article, re.DOTALL)
                if abstract_match:
                    paper["abstract"] = re.sub(r'<[^>]+>', ' ', abstract_match.group(1)).strip()
                    paper["abstract"] = re.sub(r'\s+', ' ', paper["abstract"])
                else:
                    paper["abstract"] = ""

                # DOI
                doi_match = re.search(r'<article-id pub-id-type="doi">([^<]+)</article-id>', article)
                paper["doi"] = doi_match.group(1) if doi_match else ""

                # Keywords
                keywords = re.findall(r'<kwd>([^<]+)</kwd>', article)
                paper["keywords"] = keywords

                if paper["pmcid"] and (paper["title"] or paper["abstract"]):
                    papers.append(paper)

            except Exception as e:
                logger.warning(f"Parse error for article {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"XML parse error: {e}")

    return papers


def get_pmc_pdf_url(pmcid: str) -> Optional[str]:
    """Get PDF download URL from PMC"""
    # Remove "PMC" prefix if present
    pmc_num = pmcid.replace("PMC", "")

    rate_limited_request()

    try:
        # Try OA service first
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmc_num}"
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            content = response.text

            # Look for PDF link
            pdf_match = re.search(r'href="([^"]+\.pdf)"', content)
            if pdf_match:
                pdf_url = pdf_match.group(1)
                if pdf_url.startswith("ftp://"):
                    # Convert FTP to HTTPS
                    pdf_url = pdf_url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
                return pdf_url

            # Look for any download link
            link_match = re.search(r'<link[^>]+href="([^"]+)"[^>]+format="pdf"', content)
            if link_match:
                return link_match.group(1)

        # Fallback: direct PMC PDF URL
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/"

    except Exception as e:
        logger.warning(f"Error getting PDF URL for {pmcid}: {e}")
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_num}/pdf/"


def download_pdf(url: str, save_path: Path) -> Tuple[bool, Optional[str]]:
    """Download PDF from URL"""
    rate_limited_request()

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()

            # Check if it's a PDF
            if "pdf" in content_type or url.endswith(".pdf") or response.content[:4] == b'%PDF':
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Verify file
                if save_path.stat().st_size > 5000:  # At least 5KB
                    return True, None
                else:
                    save_path.unlink(missing_ok=True)
                    return False, "File too small"
            else:
                return False, f"Not PDF: {content_type[:50]}"
        else:
            return False, f"HTTP {response.status_code}"

    except requests.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:100]


# ============================================================================
# PROGRESS SAVING
# ============================================================================

def save_progress(prisma_log: PRISMALog, all_metadata: List[Dict], seen_pmcids: Set[str]):
    """Save progress for resumability"""
    progress = {
        "prisma_log": asdict(prisma_log),
        "seen_pmcids": list(seen_pmcids),
        "paper_count": len(all_metadata)
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)


def load_progress() -> Tuple[Optional[PRISMALog], Set[str], List[Dict]]:
    """Load previous progress if exists"""
    if not PROGRESS_FILE.exists():
        return None, set(), []

    try:
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)

        # Reconstruct PRISMALog
        log_data = progress.get("prisma_log", {})
        prisma_log = PRISMALog(
            start_time=log_data.get("start_time", ""),
            end_time=log_data.get("end_time", ""),
            search_terms=log_data.get("search_terms", {}),
            totals=log_data.get("totals", PRISMALog().totals),
            errors=log_data.get("errors", [])
        )

        seen_pmcids = set(progress.get("seen_pmcids", []))

        # Load metadata
        metadata = []
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)

        logger.info(f"Resumed from progress: {len(seen_pmcids)} papers seen")
        return prisma_log, seen_pmcids, metadata

    except Exception as e:
        logger.warning(f"Could not load progress: {e}")
        return None, set(), []


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def sanitize_filename(name: str) -> str:
    """Create safe filename"""
    name = re.sub(r'[<>:"/\\|?*\']', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:40]


def process_search_term(term_info: Dict, prisma_log: PRISMALog, seen_pmcids: Set[str],
                        all_metadata: List[Dict]) -> int:
    """Process a single search term"""

    query = term_info["query"]
    cluster = term_info["cluster"]
    term_name = query.replace('"', '').replace('(', '').replace(')', '')[:40]

    if term_name in prisma_log.search_terms:
        logger.info(f"Skipping already processed: {term_name}")
        return 0

    term_log = {
        "query": query,
        "cluster": cluster,
        "found": 0,
        "new": 0,
        "relevant": 0,
        "downloaded": 0,
        "skipped_irrelevant": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0
    }

    # Search PMC
    pmcids, total_count = search_pmc(query)
    term_log["found"] = total_count
    prisma_log.totals["total_found"] += len(pmcids)

    if not pmcids:
        prisma_log.search_terms[term_name] = term_log
        return 0

    # Remove duplicates
    new_pmcids = [p for p in pmcids if f"PMC{p}" not in seen_pmcids]
    term_log["skipped_duplicate"] = len(pmcids) - len(new_pmcids)
    term_log["new"] = len(new_pmcids)
    prisma_log.totals["duplicates_removed"] += term_log["skipped_duplicate"]

    logger.info(f"  New papers: {len(new_pmcids)} (skipped {term_log['skipped_duplicate']} duplicates)")

    if not new_pmcids:
        prisma_log.search_terms[term_name] = term_log
        return 0

    # Fetch metadata
    papers = fetch_pmc_details(new_pmcids)

    downloaded = 0

    for paper in papers:
        pmcid = paper.get("pmcid", "")
        if not pmcid:
            continue

        seen_pmcids.add(pmcid)
        prisma_log.totals["unique_papers"] += 1

        # Combine text for relevance check
        text_to_check = " ".join([
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("keywords", []))
        ])

        # RELEVANCE FILTER
        is_rel, matches = is_relevant(text_to_check)

        if not is_rel:
            term_log["skipped_irrelevant"] += 1
            prisma_log.totals["skipped_irrelevant"] += 1
            continue

        term_log["relevant"] += 1
        prisma_log.totals["relevant"] += 1

        # Generate filename - SAVE TO CLUSTER SUBFOLDER
        safe_author = sanitize_filename(paper.get("first_author", "Unknown"))
        year = paper.get("year", "Unknown")
        safe_query = sanitize_filename(term_name.split("AND")[0])
        filename = f"{safe_author}_{year}_{safe_query}_{pmcid}.pdf"
        cluster_dir = PDF_DIR / cluster
        cluster_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = cluster_dir / filename

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 5000:
            paper["pdf_path"] = str(pdf_path)
            paper["status"] = "exists"
            paper["cluster"] = cluster
            paper["relevance_matches"] = matches
            paper["search_term"] = query
            all_metadata.append(paper)
            term_log["downloaded"] += 1
            downloaded += 1
            continue

        # Get PDF URL and download
        pdf_url = get_pmc_pdf_url(pmcid)

        if pdf_url:
            success, error = download_pdf(pdf_url, pdf_path)

            if success:
                paper["pdf_path"] = str(pdf_path)
                paper["status"] = "downloaded"
                paper["cluster"] = cluster
                paper["relevance_matches"] = matches
                paper["search_term"] = query
                all_metadata.append(paper)
                term_log["downloaded"] += 1
                prisma_log.totals["downloaded"] += 1
                downloaded += 1
                logger.info(f"    DOWNLOADED: {filename[:60]}")
            else:
                paper["status"] = "error"
                paper["error"] = error
                paper["cluster"] = cluster
                paper["search_term"] = query
                all_metadata.append(paper)
                term_log["skipped_error"] += 1
                prisma_log.totals["skipped_error"] += 1
                prisma_log.errors.append({
                    "pmcid": pmcid,
                    "title": paper.get("title", "")[:80],
                    "error": error
                })
        else:
            term_log["skipped_error"] += 1
            prisma_log.totals["skipped_no_pdf"] += 1

    prisma_log.search_terms[term_name] = term_log

    # Save progress after each term
    save_progress(prisma_log, all_metadata, seen_pmcids)

    return downloaded


def main():
    """Main execution"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       COMPREHENSIVE PAPER MINING v2 - BROAD SEARCH STRATEGY          ║
║                        ISEF 2026 Research                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Strategy: Wide net + strict local filter                            ║
║  Search Terms: 15 broad queries                                      ║
║  RetMax: 500 per term                                                ║
║  Filter: Location AND Pathology keywords                             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Try to resume from progress
    prisma_log, seen_pmcids, all_metadata = load_progress()

    if prisma_log is None:
        prisma_log = PRISMALog(start_time=datetime.now().isoformat())
        seen_pmcids = set()
        all_metadata = []

    # Create output directory
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    # Process each search term
    for i, term_info in enumerate(SEARCH_TERMS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SEARCH_TERMS)}] {term_info['query']} -> {term_info['cluster']}")
        print(f"{'='*70}")

        try:
            downloaded = process_search_term(term_info, prisma_log, seen_pmcids, all_metadata)
            total_downloaded += downloaded
            print(f"  -> Downloaded {downloaded} papers to {term_info['cluster']}/")
        except Exception as e:
            logger.error(f"Error processing {term_info['query']}: {e}")
            continue

    # Finalize
    prisma_log.end_time = datetime.now().isoformat()

    # Save final log
    with open(LOG_FILE, "w") as f:
        json.dump(asdict(prisma_log), f, indent=2)

    # Save final metadata
    with open(METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - PRISMA FLOW")
    print("="*70)
    print(f"\n{'Metric':<45} {'Count':>10}")
    print("-"*57)
    print(f"{'Total records found in PMC':<45} {prisma_log.totals['total_found']:>10}")
    print(f"{'Unique papers (after dedup)':<45} {prisma_log.totals['unique_papers']:>10}")
    print(f"{'Passed relevance filter':<45} {prisma_log.totals['relevant']:>10}")
    print(f"{'Successfully downloaded':<45} {prisma_log.totals['downloaded']:>10}")
    print("-"*57)
    print(f"{'Skipped - irrelevant':<45} {prisma_log.totals['skipped_irrelevant']:>10}")
    print(f"{'Skipped - duplicates':<45} {prisma_log.totals['duplicates_removed']:>10}")
    print(f"{'Skipped - download error':<45} {prisma_log.totals['skipped_error']:>10}")
    print("-"*57)

    print("\n\nBY SEARCH TERM:")
    print("-"*70)
    for term_name, term_data in prisma_log.search_terms.items():
        if term_data.get("downloaded", 0) > 0:
            print(f"  {term_name[:40]}: {term_data['downloaded']} PDFs "
                  f"(found {term_data['found']}, relevant {term_data['relevant']})")

    print("\n" + "="*70)
    print(f"Output directory: {PDF_DIR}")
    print(f"PRISMA log: {LOG_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print("="*70 + "\n")

    return prisma_log


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user - progress saved")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
