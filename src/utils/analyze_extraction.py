import json
import pandas as pd

def analyze_results():
    input_file = "data/extracted_abstract_data.json"
    try:
        with open(input_file, 'r') as f:
            papers = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total papers processed: {len(papers)}")
    
    # Filter for relevant papers
    candidates = []
    for p in papers:
        ext = p.get('extraction', {})
        if not isinstance(ext, dict): continue
        
        # Criteria: Relevant AND (Has Sample Size OR Mentions Both Indices)
        if ext.get('is_relevant'):
            indices = [i.lower() for i in ext.get('indices', [])]
            has_vai = any('van assche' in i or 'va' in i for i in indices)
            has_magnifi = any('magnifi' in i for i in indices)
            
            score = 0
            if has_vai and has_magnifi: score += 5
            if ext.get('sample_size'): score += 2
            if ext.get('correlations'): score += 3
            
            if score > 0:
                candidates.append({
                    "title": p.get('title'),
                    "score": score,
                    "sample_size": ext.get('sample_size'),
                    "indices": ext.get('indices'),
                    "correlations": ext.get('correlations'),
                    "url": p.get('url')
                })

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\nFound {len(candidates)} high-potential papers.")
    print("\n--- TOP 5 GOLDEN CANDIDATES ---")
    for i, c in enumerate(candidates[:5]):
        print(f"\n{i+1}. [Score: {c['score']}] {c['title']}")
        print(f"   N={c['sample_size']}")
        print(f"   Indices: {c['indices']}")
        print(f"   Correlations: {c['correlations']}")
        print(f"   URL: {c['url']}")

if __name__ == "__main__":
    analyze_results()
