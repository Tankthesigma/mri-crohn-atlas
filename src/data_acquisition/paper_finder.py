import requests
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from Bio import Entrez
import json
import time
import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================================
# CONFIGURATION - Update these before running!
# ============================================================================
YOUR_EMAIL = "your_email@example.com"  # Required for PubMed, CrossRef, Unpaywall

Entrez.email = YOUR_EMAIL


@dataclass
class RateLimiter:
    """Simple rate limiter to avoid API bans."""
    calls_per_window: int
    window_seconds: float
    timestamps: List[float] = field(default_factory=list)
    
    def wait_if_needed(self):
        """Block until we're allowed to make another call."""
        now = time.time()
        # Remove timestamps outside the window
        self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]
        
        if len(self.timestamps) >= self.calls_per_window:
            # Need to wait until oldest timestamp expires
            sleep_time = self.window_seconds - (now - self.timestamps[0]) + 0.1
            if sleep_time > 0:
                print(f"  ⏳ Rate limit: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        self.timestamps.append(time.time())


class PaperFinder:
    def __init__(self, email: str = YOUR_EMAIL):
        self.email = email
        self.found_papers = []
        
        # Rate limiters for different APIs
        self.semantic_scholar_limiter = RateLimiter(calls_per_window=95, window_seconds=300)  # 95/5min (safe margin)
        self.crossref_limiter = RateLimiter(calls_per_window=45, window_seconds=1)  # polite pool
        self.unpaywall_limiter = RateLimiter(calls_per_window=95, window_seconds=1)  # 100k/day but be nice
        
        # Session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"MRI-CrohnAtlas/1.0 (mailto:{email})",
        })

    def calculate_relevance_score(self, paper: Dict) -> int:
        """
        Score the paper based on how likely it is to contain the data we need
        for the Universal Translator (Van Assche <-> MAGNIFI-CD mapping).
        """
        score = 0
        text = (paper.get('title', '') + " " + paper.get('abstract', '')).lower()
        
        # ===================
        # JACKPOT CONDITIONS (direct comparison studies)
        # ===================
        if "magnifi-cd" in text and "van assche" in text:
            score += 15  # Direct comparison - exactly what we need
        if ("correlation" in text or "concordance" in text) and ("magnifi" in text or "van assche" in text):
            score += 10  # Correlation study between indices
        
        # ===================
        # HIGH VALUE - Validation & Reliability
        # ===================
        if "inter-rater" in text or "interrater" in text or "inter-observer" in text or "interobserver" in text:
            score += 7  # Reliability data = scoring criteria details
        if "intraclass correlation" in text or "icc" in text:
            score += 5
        if "kappa" in text and ("agreement" in text or "reliability" in text):
            score += 5
        if "validation" in text and ("mri" in text or "index" in text):
            score += 5
        if "responsiveness" in text or "sensitivity to change" in text:
            score += 4  # Tracks how scores change with treatment
            
        # ===================
        # MEDIUM VALUE - Clinical outcome data
        # ===================
        if "fistula healing" in text and ("predict" in text or "outcome" in text):
            score += 6  # Outcome prediction = translation anchor points
        if "treatment response" in text and "mri" in text:
            score += 4
        if "fibrosis" in text and "mri" in text:
            score += 4
        if "activity" in text and "fibrosis" in text:
            score += 3  # Activity vs fibrosis distinction
            
        # ===================
        # DATA AVAILABILITY SIGNALS
        # ===================
        if "supplementary" in text or "appendix" in text or "supplement" in text:
            score += 4  # Likely has detailed scoring data
        if re.search(r'n\s*=\s*\d{2,}', text):  # n=XX where XX is 2+ digits
            score += 3  # Decent sample size
        if "cohort" in text or "retrospective" in text or "prospective" in text:
            score += 2
        if "individual patient" in text or "patient-level" in text:
            score += 3  # Granular data
            
        # ===================
        # INDEX-SPECIFIC TERMS
        # ===================
        if "magnifi-cd" in text:
            score += 4
        if "van assche" in text:
            score += 4
        if "modified van assche" in text:
            score += 5  # Variant we need to handle
        if "pcdai" in text or "perianal disease activity" in text:
            score += 2
            
        # ===================
        # TECHNICAL/IMAGING FEATURES (for feature mapping)
        # ===================
        if "t2 hyperintensity" in text or "t2-weighted" in text:
            score += 2
        if "diffusion" in text and ("restriction" in text or "weighted" in text):
            score += 2
        if "enhancement" in text and ("gadolinium" in text or "contrast" in text):
            score += 2
        if "fistula tract" in text and ("length" in text or "complexity" in text or "extension" in text):
            score += 3
            
        return score

    def check_open_access(self, doi: str) -> Optional[Dict]:
        """Check Unpaywall for open access availability."""
        if not doi:
            return None
            
        self.unpaywall_limiter.wait_if_needed()
        
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('is_oa'):
                    best_loc = data.get('best_oa_location', {})
                    return {
                        "is_oa": True,
                        "oa_url": best_loc.get('url_for_pdf') or best_loc.get('url'),
                        "oa_type": data.get('oa_status'),  # gold, green, hybrid, bronze
                        "host_type": best_loc.get('host_type'),  # publisher, repository
                    }
                return {"is_oa": False}
            return None
        except Exception as e:
            print(f"  Unpaywall error for {doi}: {e}")
            return None

    def check_pmc(self, pmid: str) -> Optional[str]:
        """Check if paper is available in PubMed Central (free full text)."""
        if not pmid:
            return None
        
        try:
            # Use NCBI elink to find PMC ID
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(handle)
            handle.close()
            
            for linkset in record:
                for linksetdb in linkset.get('LinkSetDb', []):
                    if linksetdb.get('DbTo') == 'pmc':
                        links = linksetdb.get('Link', [])
                        if links:
                            pmc_id = links[0]['Id']
                            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            return None
        except Exception as e:
            return None

    def search_core(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search CORE.ac.uk - aggregates 200M+ OA papers from repositories worldwide.
        Note: CORE API requires free registration for API key.
        Without API key, we can still construct search URLs for manual use.
        """
        print(f"🔍 CORE.ac.uk: '{query}'...")
        
        # CORE API (if you have a key, add it to __init__)
        if hasattr(self, 'core_api_key') and self.core_api_key:
            url = "https://api.core.ac.uk/v3/search/works"
            headers = {"Authorization": f"Bearer {self.core_api_key}"}
            params = {"q": query, "limit": max_results}
            
            try:
                response = self.session.get(url, headers=headers, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    for item in data.get('results', []):
                        results.append({
                            "source": "CORE",
                            "title": item.get('title'),
                            "doi": item.get('doi'),
                            "abstract": item.get('abstract') or "",
                            "url": item.get('downloadUrl') or item.get('sourceFulltextUrls', [None])[0],
                            "is_oa": True,
                            "oa_url": item.get('downloadUrl'),
                            "year": item.get('yearPublished')
                        })
                    return results
            except Exception as e:
                print(f"  ❌ CORE API error: {e}")
        
        # Without API key, return empty (user can search manually)
        return []

    def search_europe_pmc(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search Europe PMC - includes PMC content plus European repositories.
        Free API, no key needed!
        """
        print(f"🔍 Europe PMC: '{query}'...")
        
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('resultList', {}).get('result', []):
                    # Check if full text is available
                    is_oa = item.get('isOpenAccess') == 'Y'
                    pmc_id = item.get('pmcid')
                    
                    oa_url = None
                    if pmc_id:
                        oa_url = f"https://europepmc.org/articles/{pmc_id}"
                    
                    results.append({
                        "source": "Europe PMC",
                        "title": item.get('title'),
                        "doi": item.get('doi'),
                        "pmid": item.get('pmid'),
                        "pmcid": pmc_id,
                        "abstract": item.get('abstractText') or "",
                        "url": f"https://europepmc.org/article/MED/{item.get('pmid')}" if item.get('pmid') else None,
                        "is_oa": is_oa,
                        "oa_url": oa_url,
                        "year": item.get('pubYear')
                    })
                return results
            return []
        except Exception as e:
            print(f"  ❌ Europe PMC error: {e}")
            return []

    def search_biomedical_preprints(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Search bioRxiv and medRxiv preprints via their API.
        These are always free/OA!
        """
        print(f"🔍 bioRxiv/medRxiv: '{query}'...")
        
        results = []
        
        # bioRxiv/medRxiv use the same API structure
        for server in ['biorxiv', 'medrxiv']:
            url = f"https://api.biorxiv.org/details/{server}/2020-01-01/2025-12-31"
            
            try:
                # Their API is date-based, so we fetch recent and filter
                # This is a simplified approach - full implementation would page through
                response = self.session.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    query_lower = query.lower()
                    
                    for item in data.get('collection', [])[:100]:  # Check first 100
                        title = item.get('title', '').lower()
                        abstract = item.get('abstract', '').lower()
                        
                        # Simple keyword match
                        if any(word in title or word in abstract for word in query_lower.split()):
                            doi = item.get('doi')
                            results.append({
                                "source": f"{server.capitalize()} Preprint",
                                "title": item.get('title'),
                                "doi": doi,
                                "abstract": item.get('abstract') or "",
                                "url": f"https://doi.org/{doi}" if doi else None,
                                "is_oa": True,
                                "oa_url": f"https://www.{server}.org/content/{doi}.full.pdf" if doi else None,
                                "year": item.get('date', '')[:4]
                            })
                        
                        if len(results) >= max_results:
                            break
            except Exception as e:
                print(f"  ❌ {server} error: {e}")
        
        return results[:max_results]

    def generate_author_email_template(self, paper: Dict) -> str:
        """
        Generate a polite email template to request a paper from the author.
        This works surprisingly often! Authors love knowing people read their work.
        """
        title = paper.get('title', 'your paper')
        doi = paper.get('doi', '')
        
        template = f"""Subject: PDF Request: {title[:50]}...

Dear Dr. [Author Name],

I am a high school student working on a research project about MRI scoring systems for Crohn's disease (specifically investigating the relationship between Van Assche Index and MAGNIFI-CD).

I came across your paper "{title}" and believe it would be very valuable for my research. Unfortunately, I don't have institutional access to the full text.

Would you be willing to share a PDF copy? I would greatly appreciate it.

Paper DOI: {doi}

Thank you for your time and for your contributions to this field.

Best regards,
[Your Name]

---
(Tip: Find author emails on ResearchGate, Google Scholar profiles, or university websites)
"""
        return template

    def find_author_contacts(self, paper: Dict) -> Dict:
        """
        Try to find author contact info via various sources.
        Returns dict with possible contact routes.
        """
        contacts = {
            "researchgate_search": None,
            "google_scholar_search": None,
            "orcid_search": None
        }
        
        title = paper.get('title', '')
        if title:
            # URL-encode for search links
            encoded_title = urllib.parse.quote(title[:100])
            contacts["researchgate_search"] = f"https://www.researchgate.net/search?q={encoded_title}"
            contacts["google_scholar_search"] = f"https://scholar.google.com/scholar?q={encoded_title}"
        
        doi = paper.get('doi', '')
        if doi:
            contacts["orcid_search"] = f"https://orcid.org/orcid-search/search?searchQuery={doi}"
        
        return contacts

    def search_pubmed(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search PubMed."""
        print(f"🔍 PubMed: '{query}'...")
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            if not id_list:
                return []

            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")
            papers = Entrez.read(handle)
            handle.close()

            results = []
            for article in papers['PubmedArticle']:
                medline = article['MedlineCitation']['Article']
                title = medline.get('ArticleTitle', 'No title')
                
                abstract_text = ""
                if 'Abstract' in medline and 'AbstractText' in medline['Abstract']:
                    abstract_list = medline['Abstract']['AbstractText']
                    abstract_text = " ".join(str(a) for a in abstract_list) if isinstance(abstract_list, list) else str(abstract_list)

                doi = ""
                if 'ELocationID' in medline:
                    for eloc in medline['ELocationID']:
                        if eloc.attributes.get('EIdType') == 'doi':
                            doi = str(eloc)
                
                # Also check ArticleIdList for DOI
                if not doi:
                    try:
                        for aid in article['PubmedData']['ArticleIdList']:
                            if aid.attributes.get('IdType') == 'doi':
                                doi = str(aid)
                                break
                    except (KeyError, TypeError):
                        pass

                pmid = str(article['MedlineCitation']['PMID'])
                
                results.append({
                    "source": "PubMed",
                    "title": title,
                    "doi": doi,
                    "pmid": pmid,
                    "abstract": abstract_text,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "pmc_url": None  # Will be populated if PMC available
                })
            return results
        except Exception as e:
            print(f"  ❌ PubMed error: {e}")
            return []

    def search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search Semantic Scholar with rate limiting."""
        print(f"🔍 Semantic Scholar: '{query}'...")
        
        self.semantic_scholar_limiter.wait_if_needed()
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query, 
            "limit": max_results, 
            "fields": "title,abstract,url,externalIds,year,citationCount"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for paper in data.get('data', []):
                    ext_ids = paper.get('externalIds') or {}
                    results.append({
                        "source": "Semantic Scholar",
                        "title": paper.get('title'),
                        "doi": ext_ids.get('DOI'),
                        "pmid": ext_ids.get('PubMed'),
                        "abstract": paper.get('abstract') or "",
                        "url": paper.get('url'),
                        "year": paper.get('year'),
                        "citation_count": paper.get('citationCount', 0)
                    })
                return results
            elif response.status_code == 429:
                print("  ⚠️ Semantic Scholar rate limit hit, backing off...")
                time.sleep(60)
                return []
            return []
        except Exception as e:
            print(f"  ❌ Semantic Scholar error: {e}")
            return []

    def search_crossref(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search CrossRef with polite pool headers."""
        print(f"🔍 CrossRef: '{query}'...")
        
        self.crossref_limiter.wait_if_needed()
        
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": max_results}
        headers = {
            "User-Agent": f"MRI-CrohnAtlas/1.0 (mailto:{self.email})",
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data['message']['items']:
                    title = item.get('title', ['No title'])[0] if item.get('title') else 'No title'
                    doi = item.get('DOI', '')
                    
                    # CrossRef abstracts often have XML junk - clean it
                    abstract = item.get('abstract', '')
                    if abstract:
                        abstract = re.sub(r'<[^>]+>', '', abstract)  # Strip XML tags
                    
                    results.append({
                        "source": "CrossRef",
                        "title": title,
                        "doi": doi,
                        "abstract": abstract,
                        "url": item.get('URL', ''),
                        "year": item.get('published', {}).get('date-parts', [[None]])[0][0]
                    })
                return results
            return []
        except Exception as e:
            print(f"  ❌ CrossRef error: {e}")
            return []

    def search_arxiv(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search ArXiv for AI/ML papers."""
        print(f"🔍 ArXiv: '{query}'...")
        base_url = 'http://export.arxiv.org/api/query?'
        formatted_query = urllib.parse.quote(query)
        url = f"{base_url}search_query=all:{formatted_query}&start=0&max_results={max_results}"
        
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                results = []
                for entry in root.findall('atom:entry', ns):
                    title_el = entry.find('atom:title', ns)
                    summary_el = entry.find('atom:summary', ns)
                    id_el = entry.find('atom:id', ns)
                    
                    if title_el is None or summary_el is None:
                        continue
                        
                    title = title_el.text.strip().replace('\n', ' ')
                    summary = summary_el.text.strip().replace('\n', ' ')
                    link = id_el.text.strip() if id_el is not None else ""
                    
                    doi = ""
                    for link_node in entry.findall('atom:link', ns):
                        if link_node.attrib.get('title') == 'doi':
                            doi = link_node.attrib.get('href', '').replace('http://dx.doi.org/', '')
                    
                    results.append({
                        "source": "ArXiv",
                        "title": title,
                        "doi": doi,
                        "abstract": summary,
                        "url": link,
                        "is_oa": True,  # ArXiv is always OA
                        "oa_url": link.replace('/abs/', '/pdf/') + '.pdf' if '/abs/' in link else None
                    })
                return results
        except Exception as e:
            print(f"  ❌ ArXiv error: {e}")
            return []

    def get_citations(self, doi: str) -> List[Dict]:
        """Get papers that cite the given DOI using Semantic Scholar."""
        if not doi:
            return []
        
        self.semantic_scholar_limiter.wait_if_needed()
        
        print(f"  🔗 Snowballing citations for: {doi[:40]}...")
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"
        params = {"fields": "title,abstract,url,externalIds,year", "limit": 50}
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('data', []):
                    paper = item.get('citingPaper')
                    if not paper:
                        continue
                    
                    ext_ids = paper.get('externalIds') or {}
                    results.append({
                        "source": "Semantic Scholar (Citation)",
                        "title": paper.get('title'),
                        "doi": ext_ids.get('DOI'),
                        "pmid": ext_ids.get('PubMed'),
                        "abstract": paper.get('abstract') or "",
                        "url": paper.get('url'),
                        "year": paper.get('year')
                    })
                return results
            return []
        except Exception as e:
            print(f"  ❌ Citation fetch error: {e}")
            return []

    def get_references(self, doi: str) -> List[Dict]:
        """Get papers that this DOI references (backward snowballing)."""
        if not doi:
            return []
        
        self.semantic_scholar_limiter.wait_if_needed()
        
        print(f"  🔗 Snowballing references for: {doi[:40]}...")
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/references"
        params = {"fields": "title,abstract,url,externalIds,year", "limit": 50}
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('data', []):
                    paper = item.get('citedPaper')
                    if not paper:
                        continue
                    
                    ext_ids = paper.get('externalIds') or {}
                    results.append({
                        "source": "Semantic Scholar (Reference)",
                        "title": paper.get('title'),
                        "doi": ext_ids.get('DOI'),
                        "pmid": ext_ids.get('PubMed'),
                        "abstract": paper.get('abstract') or "",
                        "url": paper.get('url'),
                        "year": paper.get('year')
                    })
                return results
            return []
        except Exception as e:
            print(f"  ❌ Reference fetch error: {e}")
            return []

    def find_relevant_papers(self, check_oa: bool = True):
        """
        Main search routine. Set check_oa=False to skip Unpaywall lookups (faster).
        """
        queries = [
            # ===================
            # Core Index Comparisons (highest priority)
            # ===================
            "MAGNIFI-CD Van Assche comparison",
            "MAGNIFI-CD validation Crohn",
            "Van Assche Index MRI Crohn validation",
            "modified Van Assche perianal fistula",
            "perianal fistula MRI scoring index comparison",
            
            # ===================
            # Reliability & Validation Studies
            # ===================
            "perianal Crohn MRI inter-rater reliability",
            "fistula MRI index interobserver agreement",
            "MAGNIFI-CD responsiveness treatment",
            "Van Assche score reproducibility",
            "MRI fistula score validation cohort",
            
            # ===================
            # Clinical Outcome Correlation
            # ===================
            "MRI fistula healing prediction Crohn",
            "perianal Crohn treatment response MRI",
            "fistula closure MRI biomarker",
            "Van Assche fistula healing outcome",
            "MAGNIFI-CD clinical response",
            
            # ===================
            # Fibrosis vs Activity (key for translation)
            # ===================
            "perianal fistula fibrosis MRI",
            "fistula activity fibrosis distinction MRI",
            "T2 hyperintensity fistula activity",
            "diffusion weighted imaging perianal Crohn",
            "contrast enhancement fistula healing",
            
            # ===================
            # Broader MRI Scoring
            # ===================
            "MRI perianal fistula activity index",
            "quantitative MRI perianal Crohn disease",
            "pelvic MRI Crohn fistula assessment",
            
            # ===================
            # AI/Radiomics (for your ML component)
            # ===================
            "deep learning MRI Crohn fistula",
            "radiomics perianal fistula",
            "machine learning MRI fibrosis prediction",
            "automated MRI Crohn assessment"
        ]
        
        all_results = []
        total_queries = len(queries)
        
        for i, q in enumerate(queries, 1):
            print(f"\n[{i}/{total_queries}] Searching: {q}")
            
            # Traditional sources
            all_results.extend(self.search_pubmed(q, max_results=25))
            all_results.extend(self.search_semantic_scholar(q, max_results=30))
            all_results.extend(self.search_crossref(q, max_results=15))
            all_results.extend(self.search_arxiv(q, max_results=10))
            
            # OA-focused sources (these often have free full text!)
            all_results.extend(self.search_europe_pmc(q, max_results=20))
            
            time.sleep(0.5)  # Small pause between query batches
        
        # Also search preprint servers for recent/cutting-edge work
        print("\n[Bonus] Searching preprint servers...")
        for q in queries[:5]:  # Top 5 queries only (preprint search is slower)
            all_results.extend(self.search_biomedical_preprints(q, max_results=10))

        # ===================
        # Deduplicate
        # ===================
        print("\n📊 Deduplicating results...")
        seen_titles = set()
        seen_dois = set()
        unique_results = []
        
        for r in all_results:
            if not r.get('title'):
                continue
                
            # Normalize title for comparison
            norm_title = re.sub(r'\W+', '', r['title'].lower())
            
            # Check both title and DOI for duplicates
            doi = r.get('doi', '').lower() if r.get('doi') else None
            
            if norm_title in seen_titles:
                continue
            if doi and doi in seen_dois:
                continue
                
            seen_titles.add(norm_title)
            if doi:
                seen_dois.add(doi)
            unique_results.append(r)

        # ===================
        # Score all papers
        # ===================
        print("📊 Scoring relevance...")
        for r in unique_results:
            r['relevance_score'] = self.calculate_relevance_score(r)
            
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # ===================
        # SNOWBALLING: Forward + Backward from top papers
        # ===================
        print("\n🔗 Snowballing from top 8 papers...")
        snowball_results = []
        top_papers = [p for p in unique_results[:8] if p.get('doi')]
        
        for paper in top_papers:
            # Forward: who cites this paper?
            citations = self.get_citations(paper['doi'])
            snowball_results.extend(citations)
            
            # Backward: what does this paper cite?
            references = self.get_references(paper['doi'])
            snowball_results.extend(references)
            
            time.sleep(0.3)
                
        # Merge snowball results
        for r in snowball_results:
            if not r.get('title'):
                continue
            norm_title = re.sub(r'\W+', '', r['title'].lower())
            doi = r.get('doi', '').lower() if r.get('doi') else None
            
            if norm_title in seen_titles:
                continue
            if doi and doi in seen_dois:
                continue
                
            seen_titles.add(norm_title)
            if doi:
                seen_dois.add(doi)
            r['relevance_score'] = self.calculate_relevance_score(r)
            unique_results.append(r)
            
        # Final sort
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # ===================
        # Check Open Access (optional, takes time)
        # ===================
        if check_oa:
            print("\n🔓 Checking open access availability (top 100 papers)...")
            for paper in unique_results[:100]:
                if paper.get('is_oa') is not None:  # Already know (e.g., ArXiv, Europe PMC)
                    continue
                
                # Try Unpaywall first
                if paper.get('doi'):
                    oa_info = self.check_open_access(paper['doi'])
                    if oa_info:
                        paper.update(oa_info)
                
                # If still no OA, check PMC directly
                if not paper.get('is_oa') and paper.get('pmid'):
                    pmc_url = self.check_pmc(paper['pmid'])
                    if pmc_url:
                        paper['is_oa'] = True
                        paper['oa_url'] = pmc_url
                        paper['oa_type'] = 'pmc'
                
                # Mark non-OA explicitly
                if paper.get('is_oa') is None:
                    paper['is_oa'] = False
        
        return unique_results

    def generate_download_guide(self, results: List[Dict], filename: str = "download_guide.md"):
        """
        Generate a markdown guide with instructions for getting each paper.
        Separates into: Direct Download, PMC Available, Request from Author, etc.
        """
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        direct_download = []
        pmc_available = []
        request_needed = []
        
        for p in results[:50]:  # Top 50 papers
            if p.get('is_oa') and p.get('oa_url'):
                direct_download.append(p)
            elif p.get('pmcid') or (p.get('oa_type') == 'pmc'):
                pmc_available.append(p)
            else:
                request_needed.append(p)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Paper Download Guide for MRI-Crohn Atlas\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            # Direct downloads
            f.write("## 🟢 Direct Download (Open Access)\n\n")
            f.write("These papers have free PDFs available:\n\n")
            for i, p in enumerate(direct_download, 1):
                f.write(f"### {i}. {p.get('title', 'Unknown')}\n")
                f.write(f"- **Relevance Score:** {p.get('relevance_score', 0)}\n")
                f.write(f"- **DOI:** {p.get('doi', 'N/A')}\n")
                f.write(f"- **Download:** [{p.get('oa_url', '')}]({p.get('oa_url', '')})\n")
                f.write(f"- **OA Type:** {p.get('oa_type', 'unknown')}\n\n")
            
            # PMC available
            if pmc_available:
                f.write("## 🟡 PubMed Central Available\n\n")
                f.write("Free full text via PMC:\n\n")
                for i, p in enumerate(pmc_available, 1):
                    pmcid = p.get('pmcid', '')
                    url = p.get('oa_url') or f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else p.get('url', '')
                    f.write(f"{i}. [{p.get('title', 'Unknown')[:80]}...]({url})\n")
                    f.write(f"   - DOI: {p.get('doi', 'N/A')}\n\n")
            
            # Need to request
            if request_needed:
                f.write("## 🔴 Request Needed (Paywalled)\n\n")
                f.write("Try these methods to get these papers:\n\n")
                f.write("1. **Check your library** - Many public/school libraries have database access\n")
                f.write("2. **Interlibrary Loan (ILL)** - Free through most libraries, takes 1-3 days\n")
                f.write("3. **Email the authors** - Works ~50% of the time! (template below)\n")
                f.write("4. **ResearchGate** - Authors often upload PDFs there\n")
                f.write("5. **Google Scholar** - Click 'All versions' to find free copies\n\n")
                
                for i, p in enumerate(request_needed[:20], 1):  # Top 20 paywalled
                    title = p.get('title', 'Unknown')
                    f.write(f"### {i}. {title[:80]}...\n")
                    f.write(f"- **Relevance Score:** {p.get('relevance_score', 0)}\n")
                    f.write(f"- **DOI:** {p.get('doi', 'N/A')}\n")
                    
                    contacts = self.find_author_contacts(p)
                    f.write(f"- **Find authors:** [ResearchGate]({contacts['researchgate_search']}) | ")
                    f.write(f"[Google Scholar]({contacts['google_scholar_search']})\n\n")
                
                # Email template
                f.write("---\n\n")
                f.write("## 📧 Email Template for Requesting Papers\n\n")
                f.write("```\n")
                f.write(self.generate_author_email_template(request_needed[0] if request_needed else {}))
                f.write("```\n")
        
        print(f"✅ Generated download guide: {filename}")

    def download_oa_papers(self, results: List[Dict], output_dir: str = "papers"):
        """
        Actually download the open access PDFs.
        Only downloads papers with direct PDF links.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded = 0
        failed = 0
        
        oa_papers = [p for p in results if p.get('is_oa') and p.get('oa_url')]
        print(f"\n📥 Attempting to download {len(oa_papers)} open access papers...")
        
        for paper in oa_papers:
            url = paper.get('oa_url', '')
            if not url:
                continue
            
            # Create safe filename from title
            title = paper.get('title', 'unknown')[:60]
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            doi_suffix = paper.get('doi', '').replace('/', '_')[:20] if paper.get('doi') else ''
            filename = f"{safe_title}_{doi_suffix}.pdf"
            filepath = os.path.join(output_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                print(f"  ⏭️  Already exists: {filename[:50]}...")
                continue
            
            try:
                print(f"  📥 Downloading: {title[:50]}...")
                response = self.session.get(url, timeout=30, allow_redirects=True)
                
                # Check if we got a PDF
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                    paper['local_path'] = filepath
                else:
                    # Might be HTML page with PDF link - save URL for manual download
                    print(f"    ⚠️  Not a direct PDF, save link: {url}")
                    failed += 1
                    
                time.sleep(1)  # Be nice to servers
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                failed += 1
        
        print(f"\n✅ Downloaded: {downloaded} | ❌ Failed: {failed}")
        return downloaded

    def save_results(self, results: List[Dict], filename: str = "found_papers.json"):
        """Save results to JSON, creating directories as needed."""
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Add metadata
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_papers": len(results),
                "oa_papers": sum(1 for r in results if r.get('is_oa')),
            },
            "papers": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(results)} papers to {filename}")
        
    def print_summary(self, results: List[Dict], top_n: int = 15):
        """Print a nice summary of results."""
        print(f"\n{'='*70}")
        print(f"FOUND {len(results)} UNIQUE PAPERS")
        print(f"{'='*70}")
        
        oa_count = sum(1 for r in results if r.get('is_oa'))
        high_relevance = sum(1 for r in results if r.get('relevance_score', 0) >= 10)
        
        print(f"📊 Open Access: {oa_count} papers")
        print(f"📊 High Relevance (score ≥10): {high_relevance} papers")
        print(f"\nTop {top_n} Most Relevant:\n")
        
        for i, p in enumerate(results[:top_n], 1):
            oa_marker = "🔓" if p.get('is_oa') else "🔒"
            score = p.get('relevance_score', 0)
            title = p.get('title', 'No title')[:70]
            
            print(f"{i:2}. [{score:2}] {oa_marker} {title}...")
            print(f"    Source: {p.get('source')} | DOI: {p.get('doi', 'N/A')}")
            if p.get('oa_url'):
                print(f"    📥 PDF: {p.get('oa_url')}")
            print()


if __name__ == "__main__":
    # ============================================
    # UPDATE YOUR EMAIL BEFORE RUNNING!
    # ============================================
    finder = PaperFinder(email=YOUR_EMAIL)
    
    print("🚀 Starting MRI-Crohn Atlas Paper Search")
    print("=" * 50)
    
    # Set check_oa=False for faster runs without OA lookups
    papers = finder.find_relevant_papers(check_oa=True)
    
    finder.print_summary(papers, top_n=15)
    
    # Save to data directory (will be created if it doesn't exist)
    finder.save_results(papers, "data/candidate_papers.json")
    
    # Generate download guide with instructions for each paper
    finder.generate_download_guide(papers, "data/download_guide.md")
    
    # Save OA-only list
    oa_papers = [p for p in papers if p.get('is_oa')]
    if oa_papers:
        finder.save_results(oa_papers, "data/open_access_papers.json")
        print(f"\n📥 Found {len(oa_papers)} open access papers")
    
    # Optional: Actually download the PDFs
    # Uncomment the line below to download all available OA papers
    # finder.download_oa_papers(papers, output_dir="data/papers")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Check data/download_guide.md for how to get each paper")
    print("2. Open access papers can be downloaded directly")
    print("3. For paywalled papers, try:")
    print("   - Your school/public library database access")
    print("   - Interlibrary loan (free, takes 1-3 days)")
    print("   - Email authors directly (template in download_guide.md)")
    print("   - ResearchGate / Google Scholar 'All versions'")
    print("=" * 50)