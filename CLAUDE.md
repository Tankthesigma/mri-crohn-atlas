# MRI-Crohn Atlas - ISEF 2026 Project

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

## CURRENT STATUS (December 7, 2025)

### What's Working:
- **Crosswalk Formula**: `MAGNIFI-CD = 1.031 × VAI + 0.264 × Fibrosis × I(VAI≤2) + 1.713`
- **Validation**: R² = 0.96, 10-Fold CV R² = 0.94, RMSE = 0.96
- **Dataset**: 83 data points from 17 studies (~2,818 patients)
- **Parser Accuracy**: 15/15 real-world cases, 10/11 edge cases (1 flagged low confidence)
- **Parser ICC**: Beats expert radiologists (+37% VAI, +8% MAGNIFI)
- **Live Site**: https://mri-crohn-atlas.vercel.app

### Dashboard Features:
- Bidirectional calculator (VAI→MAGNIFI and MAGNIFI→VAI)
- Severity gauges with color-coded thresholds
- Interactive scatterplot with severity zone shading (Remission/Mild/Moderate/Severe)
- Color-coded data points by study (12 unique colors with clickable legend)
- User calculation result shown on scatterplot ("Show on Scatterplot" button)
- Residual plot (error vs predicted)
- Bland-Altman plot with limits of agreement
- 95% prediction intervals on all conversions
- Light/dark mode toggle (synced across pages via localStorage)

### Parser Features:
- Real-time MRI report parsing via DeepSeek API
- Automatic VAI and MAGNIFI-CD scoring
- Confidence percentage with calibration
- Challenging cases section highlighting AMBIGUOUS case (46% confidence)
- Stats displayed as "15/15 cases" format with 95% CI (not "100%")

### Completed Today (Dec 7, 2025):
- Changed "100%" displays to "15/15 cases" format with 95% confidence intervals
- Added "Challenging Cases" section highlighting where parser struggled
- Highlighted AMBIGUOUS case with "Low Confidence" badge
- Added severity zone background shading to scatterplot
- Added "Show on Scatterplot" button to display user calculation
- Added dark mode toggle to both index.html and parser.html
- Theme synced across pages via localStorage

---

## TECHNICAL ARCHITECTURE

### Stack:
- **Frontend**: HTML/CSS/JavaScript (vanilla, no frameworks)
- **Parser LLM**: DeepSeek via OpenRouter API (`deepseek/deepseek-chat`)
- **Hosting**: Vercel (auto-deploys from GitHub)
- **Data Processing**: Python (pandas, numpy, PySR for symbolic regression)
- **Charts**: Plotly.js

### File Structure:
```
/
├── CLAUDE.md                    # This file - project documentation
├── package.json                 # Node.js config for Vercel build
├── vercel.json                  # Vercel deployment config
├── .gitignore                   # Excludes config.js, dist/, data/papers/
│
├── scripts/
│   └── build-config.js          # Generates config.js from env vars for Vercel
│
├── src/web/
│   ├── index.html               # Main dashboard (calculator, scatterplot, validation)
│   ├── parser.html              # MRI report parser with API integration
│   ├── config.example.js        # Template for API key config
│   ├── config.js                # (gitignored) Actual API key
│   └── serve.py                 # Local HTTP server
│
├── data/
│   ├── symbolic_results/
│   │   └── baseline_results.json
│   ├── validation_results/
│   │   └── validation_results.json
│   └── parser_tests/            # Test cases for parser validation
│       └── *.json
│
└── src/
    ├── parser/                  # Python parser validation scripts
    ├── symbolic/                # PySR regression code
    └── validation/              # Cross-validation scripts
```

### Commands:
```bash
# Run locally (set up API key first!)
cd src/web && python3 -m http.server 8080

# Or use the serve script
python3 src/web/serve.py

# Deploy (auto on push to main)
git add . && git commit -m "message" && git push origin main
```

### API Key Setup

The parser requires an OpenRouter API key. **Never commit API keys to git.**

#### Local Development:
```bash
cd src/web
cp config.example.js config.js
# Edit config.js and add your API key
```

#### Vercel Deployment:
1. Go to Vercel project dashboard → Settings → Environment Variables
2. Add: `OPENROUTER_API_KEY` = your key (starts with `sk-or-`)
3. Redeploy

#### Get an API Key:
1. Go to https://openrouter.ai/keys
2. Create a new API key
3. Uses `deepseek/deepseek-chat` model (very affordable)

---

## VALIDATION METRICS

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.96 | Primary goodness-of-fit |
| 10-Fold CV R² | 0.94 ± 0.05 | Cross-validation |
| Leave-One-Study-Out R² | 0.94 ± 0.08 | Study-level validation |
| RMSE | 0.96 points | Root mean square error |
| Parser VAI MAE | 0.93 points | Mean absolute error |
| Parser MAGNIFI MAE | 0.73 points | Mean absolute error |
| Parser ICC (VAI) | 0.934 | Intraclass correlation |
| Parser ICC (MAGNIFI) | 0.940 | Intraclass correlation |
| Real-World Cases | 15/15 (95% CI: 78-100%) | Within ±3 points |
| Edge Cases | 10/11 (95% CI: 60-100%) | 1 flagged low confidence |
| Adversarial Cases | 10/10 | Stress test suite |

---

## 17 SOURCE STUDIES

| Study | Year | Patients |
|-------|------|----------|
| ADMIRE-CD II | 2024 | 640 |
| ADMIRE-CD | 2016 | 355 |
| MAGNIFI-CD | 2019 | 320 |
| De Gregorio | 2022 | 225 |
| Yao Ustekinumab | 2023 | 190 |
| Li Ustekinumab | 2023 | 134 |
| ESGAR | 2023 | 133 |
| PEMPAC | 2021 | 120 |
| Protocolized | 2025 | 118 |
| Beek | 2024 | 115 |
| DIVERGENCE 2 | 2024 | 112 |
| van Rijn | 2022 | 100 |
| PISA-II | 2023 | 91 |
| P325 ECCO | 2022 | 76 |
| Samaan | 2019 | 60 |
| ADMIRE 104wk | 2022 | 25 |
| Tozer (SWIFT) | 2020 | 4 |

**Total: 2,818 patients**

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
| Level | VAI | MAGNIFI-CD | Color |
|-------|-----|------------|-------|
| Remission | 0-2 | 0-4 | Green |
| Mild | 3-6 | 5-10 | Teal |
| Moderate | 7-12 | 11-17 | Amber |
| Severe | 13-22 | 18-25 | Red |

---

## KNOWN LIMITATIONS

1. **Healed case edge:** Formula predicts MAGNIFI ~3.3 for VAI=0 with Fibrosis=6, but some studies report MAGNIFI=0
2. **Complex multi-fistula with seton:** T2 reduction may be overestimated with 3+ fistulas
3. **Ambiguous reports:** Parser flags low confidence (<50%) for vague findings - this is a feature, not a bug
4. **Confidence calibration:** V4 is 24.8% underconfident (conservative for clinical use)
5. **Edge case error:** AMBIGUOUS case scored VAI 10 vs expected 8 (flagged at 46% confidence)

---

## NEXT PHASE: ADVANCED VALIDATION STUDIES

### 8 Validation Studies to Add:

1. **Prospective Validation (n=50+)**
   - Collect new MRI reports not used in training
   - Blinded comparison: Parser vs 2 radiologists
   - Calculate ICC with 95% CI

2. **Inter-rater Reliability Study**
   - 3 radiologists score same 30 reports
   - Compare parser ICC to human-human ICC
   - Prove AI consistency advantage

3. **Temporal Stability Analysis**
   - Track same patients over 3+ timepoints
   - Validate formula predicts treatment response trajectory

4. **External Dataset Validation**
   - Partner with academic medical center
   - Test on completely independent cohort
   - Geographic/demographic diversity

5. **Fibrosis Score Validation**
   - Ground truth fibrosis via histopathology (n=20+)
   - Correlate inferred fibrosis with tissue analysis

6. **Edge Case Expansion**
   - Collect 50+ challenging reports
   - Pediatric cases, post-surgical, rare presentations
   - Document failure modes

7. **Clinical Utility Study**
   - Survey gastroenterologists on usefulness
   - Time savings quantification
   - Decision impact assessment

8. **Sensitivity Analysis**
   - Systematic perturbation of input features
   - Identify which features drive largest errors
   - Publish feature importance rankings

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
3. **Rigorous:** R²=0.96, cross-validated, tested on real Radiopaedia cases
4. **AI beats experts:** Parser has higher ICC than radiologists (+37% for VAI)
5. **Honest:** Limitations clearly documented, low-confidence cases highlighted
6. **Reproducible:** Open source with all data/methods

---

## CONTACT

- **Student:** Tanmay
- **GitHub:** Tankthesigma
- **Email:** vasudevatanmay@gmail.com
- **School:** Texas Virtual Academy at Hallsville
- **Repository:** https://github.com/Tankthesigma/mri-crohn-atlas
- **Live Site:** https://mri-crohn-atlas.vercel.app

---

## SESSION LOG

### Dec 7, 2025 (Data Presentation & Dark Mode)
- Changed "100%" displays to "15/15 cases" format with 95% CI
- Added "Challenging Cases" section with AMBIGUOUS case highlighted
- Added severity zone shading to scatterplot (Remission/Mild/Moderate/Severe)
- Added "Show on Scatterplot" button for user calculations
- Added dark mode toggle to parser.html (synced with index.html via localStorage)
- Updated CLAUDE.md with comprehensive project state

### Dec 7, 2025 (API Security Fix)
- Removed hardcoded API keys from parser.js and parser.html
- Created config.example.js template for API key configuration
- Added config.js to .gitignore (never committed)
- Created vercel.json and scripts/build-config.js for Vercel deployment
- Parser now shows user-friendly error when API key not configured

### Dec 7, 2025 (Chart Improvements)
- Added color-coded scatterplot by study (12 unique colors with legend)
- Added residual plot showing error vs predicted values
- Added Bland-Altman plot with limits of agreement
- Restructured accuracy presentation to lead with MAE values
- Added error distribution bar chart
- Made validation charts responsive on mobile

### Dec 4, 2025 (V4 FINAL)
- Fixed confidence calibration (V3 was 22.7% overconfident, V4 is 24.8% underconfident)
- Ground truth audit for all edge cases
- Tiered seton handling for multi-fistula cases
- Input validation module added
- Adversarial test suite (10 cases) created
- Dashboard redesigned with professional medical journal aesthetic
- Calculator severity gauges added
- Light/dark mode toggle added

---

*Last Updated: December 7, 2025*
*Parser: 15/15 real-world, 10/11 edge cases (1 low conf)*
*Crosswalk: R² = 0.96, 2,818 patients*
*Live Site: https://mri-crohn-atlas.vercel.app*
