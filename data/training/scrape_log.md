# Scrape Log - MRI-Crohn Atlas Training Data Collection
Generated: 2025-12-09

## Session Summary

### Starting State
- **Initial unique cases:** 73 (after deduplication)
- **Raw cases found:** 165 across multiple files
- **Duplicates removed:** 92

### Scraping Attempts

#### 1. Radiopaedia (FAILED)
- **Method:** BeautifulSoup web scraping
- **Result:** 429 Too Many Requests
- **Reason:** Radiopaedia rate limits programmatic access
- **Recommendation:** Use WebFetch tool with delays, or manual collection

#### 2. PubMed Central (SUCCESS)
- **Method:** NCBI E-utilities API
- **Rate limiting:** 400ms between requests
- **Queries searched:** 12
  - perianal fistula MRI case report
  - anal fistula magnetic resonance imaging
  - perianal Crohn disease MRI
  - Van Assche index
  - fistula-in-ano MRI
  - transsphincteric fistula MRI
  - intersphincteric fistula imaging
  - horseshoe fistula MRI
  - perianal abscess MRI
  - complex anal fistula imaging
  - ischioanal abscess MRI
  - anorectal fistula magnetic resonance
- **Articles fetched:** ~600
- **Relevant cases extracted:** 105
- **Final total:** 178 cases

### Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total cases | 73 | 178 | +105 |
| PubMed Central | 5 | 110 | +105 |
| Progress to 200 | 36.5% | 89% | +52.5% |

### New Cases by Fistula Type
| Type | Count |
|------|-------|
| Intersphincteric | 34 |
| Complex | 29 |
| Unknown | 28 |
| Transsphincteric | 9 |
| Horseshoe | 3 |
| Suprasphincteric | 1 |
| Extrasphincteric | 1 |

### Sources Not Attempted
- Eurorad (requires manual collection)
- Direct institutional radiology reports (HIPAA considerations)
- Case Reports in Radiology journal

### Files Created
- `data/training/master_cases.json` - 178 deduplicated cases
- `data/training/gap_analysis.md` - Current gap analysis
- `data/training/pubmed_scraper.py` - PubMed E-utilities scraper
- `data/training/nuclear_scraper.py` - Radiopaedia scraper (blocked)
- `data/training/deduplicate_and_scrape.py` - Deduplication utility

### Next Steps to Reach 200
1. Target suprasphincteric cases specifically (only 1 case)
2. Add moderate severity cases (need 34 more)
3. Manual collection from Radiopaedia for specific gaps
4. Consider synthetic data generation for rare subtypes
