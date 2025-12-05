# Data Sourcing Workflow: MRI-Crohn Atlas

## 1. Target Literature
The primary source of data will be studies that compare the **Van Assche Index (VAI)** with **MAGNIFI-CD**, or longitudinal studies that track MRI changes in Crohn's patients.

### Key Paper
**Title**: Development and internal validation of the Magnetic Resonance Novel Index for Fistula Imaging in Crohn's Disease (MAGNIFI-CD)
**Relevance**: This is the seminal paper (n=160) that likely contains the most direct comparison data.
**Action**: Locate the **Supplementary Appendix** or **Source Data** file.

## 2. Search Strategy
Use the following queries on PubMed, Google Scholar, and ClinicalTrials.gov:
- `"MAGNIFI-CD" AND "Van Assche" correlation`
- `"MAGNIFI-CD" validation study supplementary data`
- `"MRI" AND "perianal Crohn's" AND "fibrosis" score`

## 3. Data Extraction Protocol
If raw CSVs are not available, we will extract data from **Tables** and **Scatterplots**.

### A. Tabular Data (Aggregate)
Look for "Baseline Characteristics" or "Correlation" tables.
- **Extract**: Mean/Median VAI, Mean/Median MAGNIFI-CD, Correlation Coefficients (r).
- **Use**: These provide "anchor points" for our symbolic regression model to ensure it behaves correctly on average.

### B. Individual Data Points (The Gold Mine)
Look for scatterplots comparing VAI (x-axis) vs MAGNIFI-CD (y-axis).
- **Tool**: Use [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) (or similar) to extract (x, y) coordinates for each dot.
- **Output**: A CSV file with `vai_score` and `magnifi_cd_score` columns.

### C. Textual Descriptions (For the Neuro-Agent)
Since we need text to infer "Latent Fibrosis", we need to find case studies or figure captions in these papers.
- **Search**: "Figure 1", "Case Study", "Representative MRI".
- **Extract**:
    - **Image Caption**: e.g., "T2-weighted image showing a *fibrotic* tract with low signal intensity..."
    - **Associated Scores**: If the caption mentions "VAI score of 12", link it to the text.
- **Synthetic Generation**: If real text is scarce, we will use the LLM to *generate* synthetic clinical descriptions corresponding to specific VAI/Fibrosis profiles to train the system (Simulated Data Augmentation).

## 4. Dataset Structure
Create a file `data/literature_data.csv` with the following columns:
- `source_paper_id`: DOI or Title
- `patient_id`: (if available, else sequential ID)
- `vai_score`: (0-20)
- `magnifi_cd_score`: (0-25ish)
- `clinical_text_snippet`: "Fibrotic tract visible..." (Real or Synthetic)
- `data_type`: "Real" or "Synthetic"
