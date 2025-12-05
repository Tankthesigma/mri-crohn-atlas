#!/usr/bin/env python3
"""
MRI-Crohn Atlas Paper Downloader
Downloads 442 open access papers from various sources.

Usage:
    python src/data_acquisition/download_papers.py

Requirements:
    pip install requests tqdm

Output:
    ./data/papers/         - Downloaded PDFs
    ./data/papers/failed.json - Papers that failed to download
"""

import json
import os
import re
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # from tqdm import tqdm
    # HAS_TQDM = True
    HAS_TQDM = False
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")

print("Script started...", flush=True)

# Configuration
INPUT_FILE = "data/candidate_papers.json"
OUTPUT_DIR = Path("data/papers")
MAX_WORKERS = 3  # Parallel downloads (be nice to servers)
TIMEOUT = 60  # seconds

def sanitize_filename(title, max_length=100):
    """Create a safe filename from paper title"""
    safe = re.sub(r'[<>:"/\\|?*\n\r\t]', '', title)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe[:max_length].strip('_.')
    return safe or "untitled"

def get_pdf_url(paper):
    """
    Convert OA landing page URLs to direct PDF URLs.
    """
    oa_url = paper.get('oa_url', '')
    
    # PMC - NCBI
    if 'ncbi.nlm.nih.gov/pmc/articles/PMC' in oa_url:
        pmcid = re.search(r'PMC(\d+)', oa_url)
        if pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid.group(1)}/pdf/"
    
    # EuropePMC
    if 'europepmc.org' in oa_url:
        pmcid = re.search(r'PMC(\d+)', oa_url)
        if pmcid:
            return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmcid.group(1)}&blobtype=pdf"
    
    # arXiv
    if 'arxiv.org/abs/' in oa_url:
        arxiv_id = oa_url.split('/abs/')[-1].split('v')[0]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # bioRxiv / medRxiv
    if 'biorxiv.org' in oa_url or 'medrxiv.org' in oa_url:
        # Add .full.pdf to content URLs
        if '/content/' in oa_url:
            base = oa_url.split('?')[0].rstrip('/')
            if not base.endswith('.pdf'):
                return base + '.full.pdf'
    
    # Already a PDF
    if oa_url.endswith('.pdf'):
        return oa_url
    
    return oa_url

def download_paper(paper, output_dir):
    """Download a single paper. Returns (success, message, paper_info)"""
    title = paper.get('title', 'Unknown')
    doi = paper.get('doi', '')
    oa_url = paper.get('oa_url', '')
    
    if not oa_url:
        return False, "No OA URL", paper
    
    safe_title = sanitize_filename(title)
    filename = f"{safe_title}.pdf"
    filepath = output_dir / filename
    
    # Skip if exists
    if filepath.exists() and filepath.stat().st_size > 1000:
        return True, "Already exists", paper
    
    pdf_url = get_pdf_url(paper)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,*/*',
    }
    
    try:
        response = requests.get(pdf_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        
        if response.status_code == 200:
            content = response.content
            
            # Check if it's actually a PDF
            if content[:4] == b'%PDF' or b'%PDF' in content[:1024]:
                with open(filepath, 'wb') as f:
                    f.write(content)
                return True, "Downloaded", paper
            else:
                # Might be HTML landing page
                return False, "Not a PDF (got HTML)", paper
        else:
            return False, f"HTTP {response.status_code}", paper
            
    except requests.Timeout:
        return False, "Timeout", paper
    except Exception as e:
        return False, str(e)[:100], paper

def main():
    # Load papers
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        print("Make sure candidate_papers.json is in the data directory.")
        return
    
    print(f"Loading papers from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    papers = data['papers']
    oa_papers = [p for p in papers if p.get('is_oa') and p.get('oa_url')]
    
    print(f"Found {len(oa_papers)} open access papers")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Stats
    success_count = 0
    skip_count = 0
    failed_papers = []
    
    # Download with progress
    if HAS_TQDM:
        pbar = tqdm(total=len(oa_papers), desc="Downloading", unit="paper")
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_paper, p, OUTPUT_DIR): p for p in oa_papers}
        
        for future in as_completed(futures):
            success, message, paper = future.result()
            
            if success:
                if "exists" in message.lower():
                    skip_count += 1
                else:
                    success_count += 1
            else:
                failed_papers.append({
                    'title': paper.get('title', 'Unknown'),
                    'doi': paper.get('doi', ''),
                    'oa_url': paper.get('oa_url', ''),
                    'pdf_url': get_pdf_url(paper),
                    'error': message
                })
            
            if HAS_TQDM:
                pbar.update(1)
                pbar.set_postfix({'OK': success_count, 'Skip': skip_count, 'Fail': len(failed_papers)})
            else:
                total = success_count + skip_count + len(failed_papers)
                print(f"[{total}/{len(oa_papers)}] {message}: {paper.get('title', 'Unknown')[:50]}...")
    
    if HAS_TQDM:
        pbar.close()
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successfully downloaded: {success_count}")
    print(f"Already existed: {skip_count}")
    print(f"Failed: {len(failed_papers)}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Save failed list
    if failed_papers:
        failed_file = OUTPUT_DIR / "failed_downloads.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_papers, f, indent=2)
        print(f"\nFailed papers saved to: {failed_file}")
        print("You can try these manually or use Sci-Hub for paywalled ones.")
    
    # Quick stats by error type
    if failed_papers:
        error_types = {}
        for p in failed_papers:
            err = p['error']
            error_types[err] = error_types.get(err, 0) + 1
        print("\nFailure breakdown:")
        for err, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
            print(f"  {err}: {count}")

if __name__ == "__main__":
    main()
