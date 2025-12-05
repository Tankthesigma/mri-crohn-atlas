"""
Radiopaedia Scraper for Perianal Fistula MRI Cases
Collects real radiologist language from case studies
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadiopaediaScraper:
    """Scraper for Radiopaedia.org perianal fistula cases"""

    BASE_URL = "https://radiopaedia.org"
    SEARCH_URL = "https://radiopaedia.org/search"

    SEARCH_QUERIES = [
        "perianal fistula",
        "Crohn's disease fistula",
        "pelvic MRI fistula",
        "anal fistula MRI",
        "fistula in ano",
        "perianal abscess fistula",
        "transsphincteric fistula",
        "intersphincteric fistula",
    ]

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self, delay: float = 2.0):
        """Initialize scraper with rate limiting delay"""
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.collected_cases: List[Dict] = []
        self.seen_urls: set = set()

    def _respectful_request(self, url: str) -> Optional[requests.Response]:
        """Make a request with rate limiting and error handling"""
        time.sleep(self.delay)
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def search_cases(self, query: str, max_results: int = 10) -> List[str]:
        """Search for cases and return case URLs"""
        case_urls = []

        # Radiopaedia search with scope=cases
        search_url = f"{self.SEARCH_URL}?utf8=%E2%9C%93&q={requests.utils.quote(query)}&scope=cases&lang=us"
        logger.info(f"Searching: {query}")

        response = self._respectful_request(search_url)
        if not response:
            return case_urls

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find case links
        case_links = soup.find_all('a', href=re.compile(r'/cases/[^/]+$'))

        for link in case_links[:max_results]:
            href = link.get('href', '')
            if href and '/cases/' in href:
                full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                if full_url not in self.seen_urls:
                    case_urls.append(full_url)
                    self.seen_urls.add(full_url)

        logger.info(f"Found {len(case_urls)} new cases for '{query}'")
        return case_urls

    def extract_case_data(self, case_url: str) -> Optional[Dict]:
        """Extract findings and discussion from a case page"""
        logger.info(f"Extracting: {case_url}")

        response = self._respectful_request(case_url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Get title
        title_elem = soup.find('h1', class_='case-title') or soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else "Unknown Case"

        # Extract various sections
        findings_text = ""

        # Look for findings section
        findings_section = soup.find('div', {'id': 'case-findings'}) or \
                          soup.find('section', {'id': 'findings'}) or \
                          soup.find('div', class_=re.compile(r'findings', re.I))

        if findings_section:
            findings_text = findings_section.get_text(separator=' ', strip=True)

        # Look for case discussion
        discussion_section = soup.find('div', {'id': 'case-discussion'}) or \
                            soup.find('section', {'id': 'discussion'}) or \
                            soup.find('div', class_=re.compile(r'discussion', re.I))

        discussion_text = ""
        if discussion_section:
            discussion_text = discussion_section.get_text(separator=' ', strip=True)

        # Look for patient presentation
        presentation_section = soup.find('div', {'id': 'case-patient-presentation'}) or \
                              soup.find('section', {'id': 'patient-presentation'})

        presentation_text = ""
        if presentation_section:
            presentation_text = presentation_section.get_text(separator=' ', strip=True)

        # Also try to get the main body content
        body_content = soup.find('div', class_='case-body') or \
                      soup.find('article') or \
                      soup.find('div', class_='content')

        body_text = ""
        if body_content:
            # Extract paragraphs
            paragraphs = body_content.find_all('p')
            body_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Combine all text for analysis
        combined_text = ' '.join(filter(None, [findings_text, discussion_text, presentation_text, body_text]))

        # Check if this is relevant (contains MRI/fistula keywords)
        relevance_keywords = ['fistula', 'mri', 'tract', 't2', 'hyperintens', 'sphincter', 'perianal', 'abscess']
        is_relevant = any(kw in combined_text.lower() for kw in relevance_keywords)

        if not is_relevant or len(combined_text) < 100:
            logger.info(f"Skipping irrelevant or short case: {title}")
            return None

        # Clean up text
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()

        return {
            "source": "radiopaedia",
            "url": case_url,
            "title": title,
            "findings_text": combined_text[:3000],  # Limit length
            "sections": {
                "findings": findings_text[:1000] if findings_text else None,
                "discussion": discussion_text[:1000] if discussion_text else None,
                "presentation": presentation_text[:500] if presentation_text else None,
            }
        }

    def scrape_all(self, max_cases: int = 30) -> List[Dict]:
        """Run all searches and collect cases"""
        logger.info("Starting Radiopaedia scrape...")

        all_case_urls = []

        # Search with each query
        for query in self.SEARCH_QUERIES:
            urls = self.search_cases(query, max_results=8)
            all_case_urls.extend(urls)

            if len(all_case_urls) >= max_cases * 2:  # Get extra to account for filtering
                break

        # Deduplicate
        all_case_urls = list(dict.fromkeys(all_case_urls))
        logger.info(f"Total unique case URLs: {len(all_case_urls)}")

        # Extract data from each case
        for url in all_case_urls:
            if len(self.collected_cases) >= max_cases:
                break

            case_data = self.extract_case_data(url)
            if case_data:
                self.collected_cases.append(case_data)
                logger.info(f"Collected: {case_data['title']}")

        logger.info(f"Total cases collected: {len(self.collected_cases)}")
        return self.collected_cases

    def save_results(self, output_path: str):
        """Save collected cases to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_cases, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.collected_cases)} cases to {output_path}")


def main():
    """Main function to run the scraper"""
    scraper = RadiopaediaScraper(delay=2.0)
    cases = scraper.scrape_all(max_cases=15)

    output_path = Path(__file__).parent.parent.parent / "data" / "real_reports" / "radiopaedia_cases.json"
    scraper.save_results(output_path)

    print(f"\nCollected {len(cases)} cases from Radiopaedia")
    for case in cases[:5]:
        print(f"  - {case['title']}")


if __name__ == "__main__":
    main()
