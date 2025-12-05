import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neuro.extractor import ClinicalReader

def test_clinical_reader():
    print("Testing ClinicalReader with synthetic data...")
    
    # Mocking the client for demonstration if no API key is present
    # In a real scenario, we'd want to actually hit the API or use a proper mock library
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: No OPENAI_API_KEY found. Skipping live API test.")
        print("Please set OPENAI_API_KEY to run the actual inference.")
        return

    reader = ClinicalReader()
    
    test_cases = [
        {
            "text": "Active inflammation with high T2 signal, no signs of scarring.",
            "expected_range": [0, 1]
        },
        {
            "text": "Thickened fibrotic wall with low T2 signal intensity.",
            "expected_range": [2, 3]
        }
    ]

    for case in test_cases:
        print(f"\nInput: {case['text']}")
        result = reader.extract_fibrosis_score(case['text'])
        print(f"Result: {result}")
        
        score = result.get("fibrosis_score")
        if score is not None and score in case['expected_range']:
            print("✅ Score within expected range.")
        else:
            print(f"⚠️ Score {score} outside expected range {case['expected_range']} (or API error).")

if __name__ == "__main__":
    test_clinical_reader()
