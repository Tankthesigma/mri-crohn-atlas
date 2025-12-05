import pandas as pd

df = pd.read_csv('data/extraction_results.csv')
high_richness = df[df['data_richness_score'] >= 3].sort_values('data_richness_score', ascending=False)

print('Papers with data_richness_score >= 3:')
print('='*80)
for idx, row in high_richness.iterrows():
    print(f"\nFile: {row['filename']}")
    print(f"Richness: {row['data_richness_score']}, Confidence: {row['extraction_confidence']}")
    print(f"Study: {row['study_type']}, N={row['n_patients']}")
    print(f"Scoring systems: {row['scoring_systems_used']}")
    evidence = row['evidence_snippet']
    if pd.notna(evidence):
        print(f"Evidence: {evidence[:200]}...")
    else:
        print("Evidence: N/A")
