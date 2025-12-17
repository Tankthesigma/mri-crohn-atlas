# MRI-Crohn Atlas - ISEF 2026 Project

## ⛔ CRITICAL UI RULES - READ FIRST ⛔

### FORBIDDEN: Purple/Violet Design
NEVER use these colors ANYWHERE in the project:
- #8B5CF6 (purple)
- #7C3AED (purple)
- #A855F7 (purple)
- #9333EA (purple)
- #7E22CE (purple)
- Any color with "purple", "violet", "indigo" in the name
- Any gradient containing purple
- The old "glassmorphism" style with purple glows

The purple design is ARCHIVED in parser-old.html for reference only. NEVER restore it.

### REQUIRED: Clean Medical Design
ALWAYS use these colors:
- #FFFFFF - white background (primary)
- #F9FAFB - light gray background (secondary)
- #0066CC - primary blue accent
- #E5E7EB - border gray
- #111827 - text black
- #6B7280 - text gray
- #10B981 - success green
- #F59E0B - warning amber
- #EF4444 - error red

### Page Structure (DO NOT MIX)

**index.html (Main Page) - CROSSWALK ONLY:**
- Title: "MRI-Crohn Atlas: The First Validated Neuro-Symbolic Crosswalk Between Van Assche Index and MAGNIFI-CD for Perianal Fistulizing Crohn's Disease"
- Hero section with crosswalk formula
- Calculator tool
- 19 Crosswalk validation studies (R² = 0.96)
- Crosswalk methodology slideshow
- Source studies list
- NO PARSER CONTENT

**parser.html - PARSER ONLY:**
- Title: "AI-Powered MRI Report Parser: Automated VAI and MAGNIFI-CD Extraction with Publication-Grade Validation"
- Parser input tool
- Parser validation stats (68 cases, ICC 0.940)
- 6 interactive charts
- Subgroup analysis
- "Why 90% Isn't Achievable" section
- Failure analysis
- Coverage matrix
- Clinical implications
- Known limitations
- NO CROSSWALK CONTENT

### Reference Commits
- Clean medical design: fabdf2a (branch: pensive-wiles)
- If UI ever breaks, restore from: `git checkout fabdf2a -- src/web/index.html src/web/parser.html`

---

## Project Overview

**ISEF research project** building a **neuro-symbolic AI system** for Perianal Fistulizing Crohn's Disease (pfCD) that:

1. **Crosswalk Formula**: Creates the first validated mathematical conversion between two MRI scoring systems:
   - **Van Assche Index (VAI)** - established historical standard (0-22 scale)
   - **MAGNIFI-CD** - modern scoring system (0-25 scale)

2. **MRI Report Parser**: LLM-powered extraction of clinical features from radiology reports with automated VAI/MAGNIFI-CD scoring

3. **Latent Fibrosis Score**: Novel inference of fibrosis state (0-6) from clinical text to enable accurate conversion

### The Innovation
No published direct correlation between VAI and MAGNIFI-CD exists in the literature. This project fills that gap with the **first validated crosswalk formula** backed by **2,818 patients from 17 studies**.

---

## CURRENT STATUS (as of Dec 14, 2025)

### What Works:
- Crosswalk formula: `MAGNIFI-CD = 1.031 × VAI + 0.264 × Fibrosis × I(VAI≤2) + 1.713`
- R² = 0.96, 10-Fold CV R² = 0.94, RMSE = 0.96
- 83 data points from 17 studies (~2,800 patients)
- Parser accuracy: 91.2% within ±3 pts MAGNIFI (VAI MAE 1.65, MAGNIFI MAE 1.47)
- Parser ICC beats expert radiologists (+38% VAI, +10.5% MAGNIFI)
- **Parser Validation Coverage: 100% (68 test cases, 39/39 cells)**
- **Conformal Prediction (V8b): 97.1% coverage, ALL severities >85%**
- Live dashboard at mri-crohn-atlas.vercel.app
- Color-coded scatterplot by study with legend
- Residual plot and Bland-Altman plot
- Calculator with severity gauges
- Light/dark mode toggle

### Known Issues:
- API key is loaded from gitignored config.js (secure for production)
- For local dev: copy config.example.js to config.js and add your key
- For Vercel: add OPENROUTER_API_KEY environment variable in dashboard

---

## TECHNICAL ARCHITECTURE

### Stack:
- Frontend: HTML/CSS/JavaScript (vanilla)
- Parser LLM: DeepSeek via OpenRouter API
- Hosting: Vercel (auto-deploys from GitHub main branch)
- Data Processing: Python (pandas, numpy, PySR for symbolic regression)
- Charts: Plotly.js

### File Structure:
```
/src/web/
  ├── index.html      # Main dashboard with calculator, scatterplot, validation
  ├── parser.html     # MRI report parser with API integration
  └── serve.py        # Local HTTP server

/data/
  ├── symbolic_results/baseline_results.json
  ├── validation_results/validation_results.json
  └── parser_tests/*.json

/src/
  ├── parser/         # Python parser validation scripts
  ├── symbolic/       # PySR regression code
  └── validation/     # Cross-validation scripts
```

### Commands:
```bash
# Run locally (see API Key Setup below first!)
cd src/web && python3 -m http.server 8080
# Or use: python3 src/web/serve.py

# Deploy (auto on push)
git add . && git commit -m "message" && git push origin main
```

### API Key Setup

The parser requires an OpenRouter API key. **Never commit API keys to git.**

#### Local Development:
1. Copy `src/web/config.example.js` to `src/web/config.js`
2. Edit `config.js` and replace `'your-openrouter-api-key-here'` with your actual key
3. `config.js` is gitignored - it won't be committed

```bash
cd src/web
cp config.example.js config.js
# Edit config.js with your API key
```

#### Vercel Deployment:
1. Go to your Vercel project dashboard
2. Navigate to **Settings > Environment Variables**
3. Add a new variable:
   - Name: `OPENROUTER_API_KEY`
   - Value: Your OpenRouter API key (starts with `sk-or-`)
4. Redeploy the project

The build script (`scripts/build-config.js`) automatically generates `config.js` from the environment variable during deployment.

#### Get an API Key:
1. Go to https://openrouter.ai/keys
2. Create a new API key
3. The parser uses the `deepseek/deepseek-chat` model (very affordable)

---

## SCORING SYSTEMS REFERENCE

### Van Assche Index (VAI) - 0 to 22 points:
| Component | Points |
|-----------|--------|
| Fistula count (1/2/≥3) | 1/2/2 |
| Fistula type (simple/complex) | 1/2 |
| T2 hyperintensity (mild/mod/marked) | 4/6/8 |
| Extension | 0-4 |
| Collections/abscesses | 4 |
| Rectal wall involvement | 2 |

### MAGNIFI-CD - 0 to 25 points:
| Component | Points |
|-----------|--------|
| Fistula count (1/2/≥3) | 1/3/5 |
| T2 hyperintensity (mild/mod/marked) | 2/4/6 |
| Extension (none/mild/mod/severe) | 0/1/2/3 |
| Collections/abscesses | 4 |
| Rectal wall involvement | 2 |
| Inflammatory mass | 4 |
| Predominant feature (fibrotic/mixed/inflammatory) | 0/2/4 |

### Severity Thresholds:
| Level | VAI | MAGNIFI-CD |
|-------|-----|------------|
| Remission | 0-2 | 0-4 |
| Mild | 3-6 | 5-10 |
| Moderate | 7-12 | 11-17 |
| Severe | 13-22 | 18-25 |

---

## VALIDATION METRICS (Cite These)

### Crosswalk Formula Validation
| Metric | Value |
|--------|-------|
| R² | 0.96 |
| 10-Fold CV R² | 0.94 ± 0.05 |
| Leave-One-Study-Out R² | 0.94 ± 0.08 |
| RMSE | 0.96 points |

### Parser Validation (68 Cases, REAL API Results)
| Metric | VAI | MAGNIFI-CD |
|--------|-----|------------|
| **ICC** | **0.940** [0.91-0.96] | **0.961** [0.94-0.98] |
| MAE | 1.65 points | 1.47 points |
| RMSE | 2.43 | 2.07 |
| Accuracy (exact) | 30.9% | 26.5% |
| Accuracy (±2) | 79.4% | 83.8% |
| Accuracy (±3) | 85.3% | 91.2% |
| Pearson r | 0.940 | 0.964 |
| Weighted κ | 0.80 | 0.78 |
| Bias | +0.18 | -0.56 |

### Parser vs Radiologist Agreement
| Metric | Parser | Radiologists | Improvement |
|--------|--------|--------------|-------------|
| VAI ICC | 0.940 | 0.68 | **+38.3%** |
| MAGNIFI ICC | 0.961 | 0.87 | **+10.5%** |
| VAI Weighted κ | 0.80 | 0.61 | +31.1% |

### Architecture Ablation Study (74 Gold Cases)
| Method | VAI MAE | VAI ICC | Result |
|--------|---------|---------|--------|
| **N-SCAPE** | **1.43** | **0.954** | Winner |
| Pure LLM | 3.08 | 0.828 | Hallucinated |
| Regex | 4.81 | 0.597 | Negation failures |

---

## 17 SOURCE STUDIES

ADMIRE-CD II (2024, n=640), ADMIRE-CD (2016, n=355), MAGNIFI-CD (2019, n=320), De Gregorio (2022, n=225), Yao Ustekinumab (2023, n=190), Li Ustekinumab (2023, n=134), ESGAR (2023, n=133), PEMPAC (2021, n=120), Protocolized (2025, n=118), Beek (2024, n=115), DIVERGENCE 2 (2024, n=112), van Rijn (2022, n=100), PISA-II (2023, n=91), P325 ECCO (2022, n=76), Samaan (2019, n=60), ADMIRE 104wk (2022, n=25)

**Total: 2,818 patients**

---

## KNOWN LIMITATIONS

### Crosswalk Formula Limitations
1. **Healed case edge:** Formula predicts MAGNIFI ~3.3 for VAI=0 with Fibrosis=6, but some studies report MAGNIFI=0
2. **Complex multi-fistula with seton:** T2 reduction may be overestimated with 3+ fistulas

### Parser Validation Limitations
3. **Severity-dependent accuracy:** Remission 100%, Mild 80%, Moderate 64%, Severe 65%
4. **Dataset composition:** 42/68 cases (61.8%) are synthetic
5. **Horseshoe fistulas:** Consistently underscored by 5-7 points
6. **Ambiguous reports:** Parser defaults to 0 for equivocal language
7. **No external validation:** All cases from literature/Radiopaedia

---

## ISEF TIMELINE

| Date | Milestone |
|------|-----------|
| Jan 30, 2026 | Registration deadline (STEMWizard, $40 fee) |
| Feb 6, 2026 | East Texas Regional @ Kilgore College |
| March 27-28, 2026 | Texas State Science Fair (backup path) |
| May 2026 | ISEF (if qualified) |

---

## KEY TALKING POINTS FOR JUDGES

1. **Novel:** First published crosswalk between VAI and MAGNIFI-CD
2. **Impact:** Unlocks 20+ years of incompatible research for meta-analysis
3. **Rigorous:** R²=0.96, cross-validated, 68 test cases with 100% coverage
4. **AI beats experts:** Parser ICC 0.940 vs radiologists' 0.68 (**+38.3% improvement**)
5. **Publication-quality validation:** Bland-Altman, ICC with 95% CI, subgroup analysis
6. **Honest:** Limitations clearly documented
7. **Reproducible:** Open source with all data/methods

---

## CONTACT

- **Student:** Tanmay
- **GitHub:** Tankthesigma
- **Email:** vasudevatanmay@gmail.com
- **School:** Texas Virtual Academy at Hallsville
- **Repository:** https://github.com/Tankthesigma/mri-crohn-atlas

---

## CSS Architecture Notes

### Spacing Variables
Uses CSS custom properties: `--space-1` through `--space-12`
- Compact design uses space-3/space-4
- Chunky/spacious uses space-6/space-8

### Dark Mode
- Triggered via `[data-theme="dark"]` attribute on `<html>`
- Theme persisted in localStorage
- Charts must be destroyed and recreated on theme toggle

### Yellow Background Rule
**CRITICAL:** Yellow backgrounds (#FEF3C7, #FDE68A) ALWAYS need dark text (#78350F or #92400E) regardless of light/dark mode. Never use var(--gray-*) on yellow - it becomes invisible in dark mode.

### Chart Colors
```javascript
function getChartColors() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        text: isDark ? '#E5E7EB' : '#1F2937',
        grid: isDark ? '#374151' : '#E5E7EB'
    };
}
```

---

## Paper Mining Infrastructure

### PMC Scraper (fetch_comprehensive_papers.py)
```bash
python3 src/mining/fetch_comprehensive_papers.py
python3 src/mining/fetch_comprehensive_papers.py --retry-failed
```

### Europe PMC Scraper (fetch_europe_pmc.py)
```bash
python3 src/mining/fetch_europe_pmc.py
python3 src/mining/fetch_europe_pmc.py --resume
```

### Unpaywall Scraper (fetch_unpaywall.py)
```bash
python3 src/mining/fetch_unpaywall.py
python3 src/mining/fetch_unpaywall.py --failed-only
```

### Treatment Extractor (extract_treatments.py)
```bash
python3 src/mining/extract_treatments.py
python3 src/mining/extract_treatments.py --resume
python3 src/mining/extract_treatments.py --test
```

**Output:** `data/treatment_extraction_results/`

---

## Conformal Prediction (V8b Final)

- Overall coverage: **97.1%** (VAI and MAGNIFI)
- All severity levels >85%: Remission 100%, Mild 100%, Moderate 92.9%, Severe 94.1%
- Hybrid optimal calibration with group-specific bias correction
- Prediction intervals adapt to case difficulty

---

*See SESSION_LOG.md for detailed development history*

*Last Updated: December 14, 2025*
*Live Site: https://mri-crohn-atlas.vercel.app*
