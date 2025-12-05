"""
PubMed Central Scraper for Perianal Fistula MRI Case Reports
Extracts imaging findings sections from open-access papers
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PubMedScraper:
    """Scraper for PubMed Central open-access case reports"""

    # NCBI E-utilities API endpoints
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PMC_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"

    SEARCH_QUERIES = [
        '"perianal fistula" MRI "case report"',
        '"Crohn disease" "perianal" MRI findings',
        '"anal fistula" MRI imaging findings',
        '"fistula in ano" magnetic resonance',
        '"perianal abscess" MRI "case report"',
        '"transsphincteric fistula" MRI',
        '"intersphincteric fistula" imaging',
        'perianal Crohn MRI radiological',
    ]

    HEADERS = {
        "User-Agent": "MRI-Crohn-Atlas/1.0 (ISEF Research Project; tanmay@student.edu)",
        "Accept": "application/xml,text/xml,text/html",
    }

    def __init__(self, delay: float = 1.0, email: str = "tanmay@student.edu"):
        """Initialize scraper with rate limiting"""
        self.delay = delay
        self.email = email
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.collected_reports: List[Dict] = []
        self.seen_pmcids: set = set()

    def _respectful_request(self, url: str, params: dict = None) -> Optional[requests.Response]:
        """Make a request with rate limiting and error handling"""
        time.sleep(self.delay)
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def search_pmc(self, query: str, max_results: int = 20) -> List[str]:
        """Search PMC and return PMCIDs"""
        logger.info(f"Searching PMC: {query}")

        params = {
            "db": "pmc",
            "term": f"{query} AND open access[filter]",
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
            "tool": "mri-crohn-atlas",
        }

        response = self._respectful_request(self.ESEARCH_URL, params)
        if not response:
            return []

        try:
            data = response.json()
            pmcids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmcids)} articles for '{query}'")
            return pmcids
        except json.JSONDecodeError:
            logger.error("Failed to parse search response")
            return []

    def fetch_article_xml(self, pmcid: str) -> Optional[str]:
        """Fetch full article XML from PMC"""
        params = {
            "db": "pmc",
            "id": pmcid,
            "retmode": "xml",
            "email": self.email,
            "tool": "mri-crohn-atlas",
        }

        response = self._respectful_request(self.EFETCH_URL, params)
        if response:
            return response.text
        return None

    def extract_findings_from_xml(self, xml_text: str, pmcid: str) -> Optional[Dict]:
        """Extract imaging findings from article XML"""
        try:
            # Parse XML
            root = ET.fromstring(xml_text)

            # Get article title
            title_elem = root.find('.//article-title')
            title = title_elem.text if title_elem is not None and title_elem.text else "Unknown Title"

            # Get abstract
            abstract_elem = root.find('.//abstract')
            abstract = ""
            if abstract_elem is not None:
                abstract = ' '.join(abstract_elem.itertext())

            # Get body sections
            body = root.find('.//body')
            body_text = ""
            findings_text = ""
            imaging_text = ""

            if body is not None:
                # Look for imaging/findings/results sections
                sections = body.findall('.//sec')

                for sec in sections:
                    # Get section title
                    sec_title_elem = sec.find('title')
                    sec_title = sec_title_elem.text.lower() if sec_title_elem is not None and sec_title_elem.text else ""

                    sec_content = ' '.join(sec.itertext())

                    # Check if this is a relevant section
                    if any(kw in sec_title for kw in ['imaging', 'finding', 'result', 'radiolog', 'mri', 'magnetic']):
                        imaging_text += sec_content + " "
                    elif any(kw in sec_title for kw in ['case', 'presentation', 'report']):
                        findings_text += sec_content + " "

                # Also get general body text
                body_text = ' '.join(body.itertext())

            # Combine texts prioritizing imaging sections
            combined = imaging_text or findings_text or body_text or abstract

            # Check relevance
            relevance_keywords = ['fistula', 'mri', 'tract', 't2', 'hyperintens', 'sphincter', 'perianal', 'crohn']
            is_relevant = sum(1 for kw in relevance_keywords if kw in combined.lower()) >= 3

            if not is_relevant or len(combined) < 200:
                logger.info(f"Skipping irrelevant article: {title[:50]}")
                return None

            # Clean up text
            combined = re.sub(r'\s+', ' ', combined).strip()

            return {
                "source": "pubmed_central",
                "url": f"{self.PMC_BASE}/PMC{pmcid}/",
                "pmcid": f"PMC{pmcid}",
                "title": title,
                "findings_text": combined[:4000],  # Limit length
                "abstract": abstract[:1000] if abstract else None,
                "sections": {
                    "imaging": imaging_text[:1500] if imaging_text else None,
                    "case_presentation": findings_text[:1500] if findings_text else None,
                }
            }

        except ET.ParseError as e:
            logger.error(f"XML parse error for PMC{pmcid}: {e}")
            return None

    def scrape_pmc_html(self, pmcid: str) -> Optional[Dict]:
        """Fallback: Scrape HTML version of PMC article"""
        url = f"{self.PMC_BASE}/PMC{pmcid}/"
        logger.info(f"Scraping HTML: {url}")

        response = self._respectful_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get title
        title_elem = soup.find('h1', class_='content-title') or soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"

        # Find all paragraphs in article body
        article_body = soup.find('div', class_='jig-ncbiinpagenav') or soup.find('article') or soup.find('div', class_='article')

        text_content = ""
        if article_body:
            # Look for specific sections
            sections = article_body.find_all(['section', 'div'], class_=re.compile(r'sec', re.I))

            for sec in sections:
                heading = sec.find(['h2', 'h3', 'h4'])
                heading_text = heading.get_text(strip=True).lower() if heading else ""

                if any(kw in heading_text for kw in ['imaging', 'finding', 'result', 'radiolog', 'case', 'mri']):
                    paragraphs = sec.find_all('p')
                    text_content += ' '.join([p.get_text(strip=True) for p in paragraphs]) + " "

            # If no specific sections found, get all paragraphs
            if not text_content:
                paragraphs = article_body.find_all('p')
                text_content = ' '.join([p.get_text(strip=True) for p in paragraphs[:20]])

        # Check relevance
        relevance_keywords = ['fistula', 'mri', 'tract', 't2', 'sphincter', 'perianal']
        is_relevant = sum(1 for kw in relevance_keywords if kw in text_content.lower()) >= 2

        if not is_relevant or len(text_content) < 200:
            return None

        # Clean up
        text_content = re.sub(r'\s+', ' ', text_content).strip()

        return {
            "source": "pubmed_central",
            "url": url,
            "pmcid": f"PMC{pmcid}",
            "title": title,
            "findings_text": text_content[:4000],
        }

    def scrape_all(self, max_articles: int = 20) -> List[Dict]:
        """Run all searches and collect case reports"""
        logger.info("Starting PubMed Central scrape...")

        all_pmcids = []

        # Search with each query
        for query in self.SEARCH_QUERIES:
            pmcids = self.search_pmc(query, max_results=10)
            for pmcid in pmcids:
                if pmcid not in self.seen_pmcids:
                    all_pmcids.append(pmcid)
                    self.seen_pmcids.add(pmcid)

            if len(all_pmcids) >= max_articles * 2:
                break

        logger.info(f"Total unique PMCIDs: {len(all_pmcids)}")

        # Extract data from each article
        for pmcid in all_pmcids:
            if len(self.collected_reports) >= max_articles:
                break

            # Try XML first, then HTML fallback
            xml_text = self.fetch_article_xml(pmcid)
            if xml_text:
                report_data = self.extract_findings_from_xml(xml_text, pmcid)
                if report_data:
                    self.collected_reports.append(report_data)
                    logger.info(f"Collected (XML): {report_data['title'][:50]}")
                    continue

            # HTML fallback
            report_data = self.scrape_pmc_html(pmcid)
            if report_data:
                self.collected_reports.append(report_data)
                logger.info(f"Collected (HTML): {report_data['title'][:50]}")

        logger.info(f"Total reports collected: {len(self.collected_reports)}")
        return self.collected_reports

    def save_results(self, output_path: str):
        """Save collected reports to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_reports, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.collected_reports)} reports to {output_path}")


def main():
    """Main function to run the scraper"""
    scraper = PubMedScraper(delay=1.0)
    reports = scraper.scrape_all(max_articles=15)

    output_path = Path(__file__).parent.parent.parent / "data" / "real_reports" / "pubmed_cases.json"
    scraper.save_results(output_path)

    print(f"\nCollected {len(reports)} case reports from PubMed Central")
    for report in reports[:5]:
        print(f"  - {report['title'][:60]}...")


if __name__ == "__main__":
    main()
