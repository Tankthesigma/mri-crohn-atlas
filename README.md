# MRI-Crohn Atlas

## Project Overview
A Neuro-Symbolic AI system to harmonize disparate MRI indices for Perianal Fistulizing Crohn’s Disease. It converts historical **Van Assche Index (VAI)** scores into modern **MAGNIFI-CD** scores by inferring a missing "Latent Fibrosis Score" from clinical text.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Set your LLM API key (e.g., OpenAI, DeepSeek):
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Usage

### Neuro Component (Fibrosis Extraction)
Extract fibrosis scores from clinical text:
```python
from src.neuro.extractor import ClinicalReader

reader = ClinicalReader()
score = reader.extract_fibrosis_score("Thickened fibrotic tract...")
print(score)
```

### Symbolic Component (Formula Discovery)
Find the conversion formula (requires Julia installed for PySR):
```python
from src.symbolic.regression import SymbolicConverter

# Load your data
# X = ... (VAI + Fibrosis)
# y = ... (MAGNIFI-CD)

converter = SymbolicConverter()
converter.fit(X, y)
print(converter.get_best_equation())
```

## Data
Place your data in the `data/` directory.
- `data/literature_data.csv`: Extracted data from papers.
