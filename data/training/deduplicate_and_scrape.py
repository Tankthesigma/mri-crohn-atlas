#!/usr/bin/env python3
"""
Nuclear Scrape Script for MRI-Crohn Atlas
Target: 200+ unique real MRI cases for fine-tuning
"""

import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "training"

def normalize_url(url):
    """Normalize URL for comparison"""
    if not url:
        return ""
    # Remove lang parameter, trailing slash, protocol variations
    url = re.sub(r'\?lang=\w+', '', url)
    url = re.sub(r'https?://', '', url)
    url = url.rstrip('/')
    return url.lower()

def text_similarity(text1, text2):
    """Calculate similarity ratio between two texts"""
    if not text1 or not text2:
        return 0.0
    # Normalize texts
    t1 = re.sub(r'\s+', ' ', str(text1).lower().strip())
    t2 = re.sub(r'\s+', ' ', str(text2).lower().strip())
    return SequenceMatcher(None, t1, t2).ratio()

def extract_findings_text(case):
    """Extract the main findings text from a case"""
    # Try different possible field names
    for field in ['findings_text', 'report_text', 'text', 'findings']:
        if field in case and case[field]:
            return case[field]
    # Try nested sections
    if 'sections' in case and isinstance(case['sections'], dict):
        for key in ['findings', 'imaging', 'case_presentation']:
            if key in case['sections'] and case['sections'][key]:
                return case['sections'][key]
    return ""

def is_perianal_fistula_case(case):
    """Check if case is actually about perianal fistula (not vesicovaginal, etc.)"""
    text = extract_findings_text(case).lower()
    title = str(case.get('title', '')).lower()

    # Exclusion keywords (non-perianal fistulas)
    exclude_keywords = [
        'vesicovaginal', 'vesicouterine', 'ureterovaginal', 'rectovaginal',
        'tailgut cyst', 'anal neoplasm', 'anal cancer', 'squamous cell carcinoma',
        'colonoscopy', 'endoscopy findings', 'ohvira syndrome', 'il10rb',
        'metabolic dysfunction', 'masld', '3d modelling', '3d printing',
        'capsule endoscopy', 'cluster analysis', 'review article',
        'systematic review', 'meta-analysis'
    ]

    combined_text = text + ' ' + title
    for kw in exclude_keywords:
        if kw in combined_text:
            return False

    # Must have some perianal/anal fistula keywords
    include_keywords = [
        'perianal fistula', 'anal fistula', 'fistula-in-ano', 'fistula in ano',
        'intersphincteric', 'transsphincteric', 'extrasphincteric', 'suprasphincteric',
        'ischioanal', 'ischiorectal', 'perirectal', 'anal sphincter',
        'horseshoe fistula', 'seton', 'perianal abscess', 'perianal crohn'
    ]

    for kw in include_keywords:
        if kw in combined_text:
            return True

    return False

def has_mri_findings(case):
    """Check if case has actual MRI findings text (not just CT or ultrasound)"""
    text = extract_findings_text(case).lower()

    # MRI-specific terms
    mri_terms = [
        't2', 't1', 'hyperintense', 'hypointense', 'gadolinium', 'enhancement',
        'diffusion', 'dwi', 'adc', 'flair', 'stir', 'haste', 'vibe',
        'mri', 'mr ', 'magnetic resonance', 'fat sat', 'fat suppression',
        'signal intensity', 'sphincter complex'
    ]

    for term in mri_terms:
        if term in text:
            return True

    # Also accept if it describes fistula anatomy in detail
    anatomy_terms = ['internal sphincter', 'external sphincter', 'fistulous tract', 'fistula tract']
    count = sum(1 for term in anatomy_terms if term in text)
    return count >= 2

def load_existing_cases():
    """Load all existing case files"""
    all_cases = []
    files_loaded = {}

    # Define all possible case files
    case_files = [
        DATA_DIR / "real_reports" / "collected_reports.json",
        DATA_DIR / "real_reports" / "collected_reports_v2.json",
        DATA_DIR / "real_reports" / "pubmed_cases.json",
        DATA_DIR / "real_reports" / "radiopaedia_cases.json",
        DATA_DIR / "parser_validation" / "mega_test_cases.json",
        DATA_DIR / "parser_tests" / "test_cases.json",
        DATA_DIR / "parser_tests" / "edge_cases.json",
        DATA_DIR / "parser_tests" / "adversarial_cases.json",
    ]

    for filepath in case_files:
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Handle different file structures
                cases = []
                if isinstance(data, list):
                    cases = data
                elif isinstance(data, dict):
                    if 'test_cases' in data:
                        cases = data['test_cases']
                    elif 'cases' in data:
                        cases = data['cases']
                    else:
                        cases = [data]

                files_loaded[str(filepath.name)] = len(cases)
                all_cases.extend(cases)

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    return all_cases, files_loaded

def deduplicate_cases(cases):
    """Remove duplicate cases based on URL, ID, and text similarity"""
    unique_cases = []
    seen_urls = set()
    seen_ids = set()

    for case in cases:
        # Skip non-perianal fistula cases
        if not is_perianal_fistula_case(case):
            continue

        # Check URL
        url = normalize_url(case.get('url', '') or case.get('source_url', ''))
        if url and url in seen_urls:
            continue

        # Check case ID
        case_id = str(case.get('id', '') or case.get('case_id', ''))
        if case_id and case_id in seen_ids:
            continue

        # Check text similarity against existing cases
        text = extract_findings_text(case)
        is_duplicate = False

        for existing in unique_cases:
            existing_text = extract_findings_text(existing)
            if text_similarity(text, existing_text) > 0.90:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Add to unique cases
        if url:
            seen_urls.add(url)
        if case_id:
            seen_ids.add(case_id)
        unique_cases.append(case)

    return unique_cases

def standardize_case(case, index):
    """Standardize case format for master file"""
    text = extract_findings_text(case)
    url = case.get('url', '') or case.get('source_url', '')

    # Determine source
    source = case.get('source', 'unknown')
    if 'radiopaedia' in str(url).lower():
        source = 'radiopaedia'
    elif 'pubmed' in str(url).lower() or 'ncbi' in str(url).lower():
        source = 'pubmed_central'
    elif 'eurorad' in str(url).lower():
        source = 'eurorad'

    # Determine fistula type from text
    fistula_type = None
    text_lower = text.lower()
    if 'intersphincteric' in text_lower:
        fistula_type = 'intersphincteric'
    elif 'transsphincteric' in text_lower or 'trans-sphincteric' in text_lower:
        fistula_type = 'transsphincteric'
    elif 'extrasphincteric' in text_lower:
        fistula_type = 'extrasphincteric'
    elif 'suprasphincteric' in text_lower:
        fistula_type = 'suprasphincteric'
    elif 'horseshoe' in text_lower:
        fistula_type = 'horseshoe'
    elif 'complex' in text_lower or 'multiple' in text_lower:
        fistula_type = 'complex'

    # Check for abscess
    has_abscess = bool(re.search(r'abscess|collection|rim.?enhanc', text_lower))

    # Check for pediatric
    is_pediatric = bool(re.search(r'pediatric|paediatric|child|infant|adolescent|year.?old.*(boy|girl)|y/?o (male|female) child', text_lower))

    # Estimate severity from text
    severity = None
    if 'severe' in text_lower or 'extensive' in text_lower or 'marked' in text_lower:
        severity = 'severe'
    elif 'moderate' in text_lower:
        severity = 'moderate'
    elif 'mild' in text_lower or 'minimal' in text_lower:
        severity = 'mild'
    elif 'healed' in text_lower or 'fibrotic' in text_lower or 'remission' in text_lower:
        severity = 'remission'

    # Quality score based on detail level
    quality = 3  # Default moderate
    if has_mri_findings(case):
        quality += 1
    if len(text) > 500:
        quality += 1
    if fistula_type:
        quality += 0.5
    quality = min(5, int(quality))

    standardized = {
        "case_id": case.get('id', '') or case.get('case_id', '') or f"case_{index:04d}",
        "source": source,
        "source_url": url,
        "title": case.get('title', ''),
        "clinical_history": case.get('presentation', '') or case.get('clinical_history', ''),
        "findings_text": text,
        "diagnosis": case.get('diagnosis', ''),
        "fistula_type": fistula_type,
        "severity_indicators": severity,
        "treatment_mentioned": None,
        "has_abscess": has_abscess,
        "pediatric": is_pediatric,
        "scored_vai": case.get('ground_truth', {}).get('expected_vai_score'),
        "scored_magnifi": case.get('ground_truth', {}).get('expected_magnifi_score'),
        "scraped_date": datetime.now().strftime("%Y-%m-%d"),
        "quality_score": quality,
        "has_mri_findings": has_mri_findings(case)
    }

    return standardized

def analyze_coverage(cases):
    """Analyze coverage by severity and type"""
    severity_counts = {'remission': 0, 'mild': 0, 'moderate': 0, 'severe': 0, 'unknown': 0}
    type_counts = {
        'intersphincteric': 0,
        'transsphincteric': 0,
        'suprasphincteric': 0,
        'extrasphincteric': 0,
        'horseshoe': 0,
        'complex': 0,
        'unknown': 0
    }
    special_counts = {
        'with_abscess': 0,
        'post_surgical': 0,
        'pediatric': 0,
        'healed_fibrotic': 0,
        'has_mri_findings': 0
    }
    source_counts = {}
    quality_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for case in cases:
        # Severity
        sev = case.get('severity_indicators', 'unknown') or 'unknown'
        if sev in severity_counts:
            severity_counts[sev] += 1
        else:
            severity_counts['unknown'] += 1

        # Type
        ft = case.get('fistula_type', 'unknown') or 'unknown'
        if ft in type_counts:
            type_counts[ft] += 1
        else:
            type_counts['unknown'] += 1

        # Special cases
        if case.get('has_abscess'):
            special_counts['with_abscess'] += 1
        if case.get('pediatric'):
            special_counts['pediatric'] += 1
        if case.get('has_mri_findings'):
            special_counts['has_mri_findings'] += 1

        text_lower = case.get('findings_text', '').lower()
        if 'seton' in text_lower or 'post-op' in text_lower or 'surgical' in text_lower:
            special_counts['post_surgical'] += 1
        if 'fibrotic' in text_lower or 'healed' in text_lower or 'hypointense' in text_lower:
            special_counts['healed_fibrotic'] += 1

        # Source
        src = case.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1

        # Quality
        q = case.get('quality_score', 3)
        if q in quality_counts:
            quality_counts[q] += 1

    return {
        'severity': severity_counts,
        'fistula_type': type_counts,
        'special_cases': special_counts,
        'source': source_counts,
        'quality': quality_counts
    }

def generate_gap_analysis(coverage, target=200):
    """Generate gap analysis markdown"""
    gaps = []

    # Severity targets
    severity_targets = {
        'remission': 30,
        'mild': 40,
        'moderate': 50,
        'severe': 40
    }

    type_targets = {
        'intersphincteric': 30,
        'transsphincteric': 50,
        'suprasphincteric': 15,
        'extrasphincteric': 10,
        'horseshoe': 15,
        'complex': 30
    }

    special_targets = {
        'with_abscess': 25,
        'post_surgical': 20,
        'pediatric': 15,
        'healed_fibrotic': 20
    }

    md = f"""# Gap Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Summary
- **Total unique cases:** {sum(coverage['source'].values())}
- **Target:** {target}
- **Gap:** {max(0, target - sum(coverage['source'].values()))} cases needed

## By Severity
| Severity | Current | Target | Gap |
|----------|---------|--------|-----|
"""

    for sev, target_count in severity_targets.items():
        current = coverage['severity'].get(sev, 0)
        gap = max(0, target_count - current)
        status = "✓" if gap == 0 else f"⚠️ Need {gap}"
        md += f"| {sev.capitalize()} | {current} | {target_count} | {status} |\n"
        if gap > 0:
            gaps.append(f"{sev}: need {gap} more")

    md += "\n## By Fistula Type\n| Type | Current | Target | Gap |\n|------|---------|--------|-----|\n"

    for ft, target_count in type_targets.items():
        current = coverage['fistula_type'].get(ft, 0)
        gap = max(0, target_count - current)
        status = "✓" if gap == 0 else f"⚠️ Need {gap}"
        md += f"| {ft.capitalize()} | {current} | {target_count} | {status} |\n"
        if gap > 0:
            gaps.append(f"{ft}: need {gap} more")

    md += "\n## Special Cases\n| Category | Current | Target | Gap |\n|----------|---------|--------|-----|\n"

    for cat, target_count in special_targets.items():
        current = coverage['special_cases'].get(cat, 0)
        gap = max(0, target_count - current)
        status = "✓" if gap == 0 else f"⚠️ Need {gap}"
        md += f"| {cat.replace('_', ' ').title()} | {current} | {target_count} | {status} |\n"

    md += "\n## By Source\n| Source | Count |\n|--------|-------|\n"
    for src, count in sorted(coverage['source'].items(), key=lambda x: -x[1]):
        md += f"| {src} | {count} |\n"

    md += "\n## Quality Distribution\n| Score | Count |\n|-------|-------|\n"
    for q in range(5, 0, -1):
        count = coverage['quality'].get(q, 0)
        md += f"| {q} | {count} |\n"

    md += f"\n## Gaps Remaining\n"
    if gaps:
        for gap in gaps:
            md += f"- {gap}\n"
    else:
        md += "- None! All targets met.\n"

    return md

def main():
    print("=" * 60)
    print("MRI-Crohn Atlas - Case Deduplication & Analysis")
    print("=" * 60)

    # Load existing cases
    print("\n📂 Loading existing case files...")
    all_cases, files_loaded = load_existing_cases()
    print(f"   Files loaded: {files_loaded}")
    print(f"   Total raw cases: {len(all_cases)}")

    # Deduplicate
    print("\n🔍 Deduplicating cases...")
    unique_cases = deduplicate_cases(all_cases)
    print(f"   Unique perianal fistula cases: {len(unique_cases)}")
    print(f"   Duplicates/non-relevant removed: {len(all_cases) - len(unique_cases)}")

    # Standardize
    print("\n📝 Standardizing case format...")
    standardized_cases = [standardize_case(case, i) for i, case in enumerate(unique_cases)]

    # Filter for MRI findings
    mri_cases = [c for c in standardized_cases if c['has_mri_findings']]
    print(f"   Cases with MRI findings: {len(mri_cases)}")

    # Analyze coverage
    print("\n📊 Analyzing coverage...")
    coverage = analyze_coverage(standardized_cases)

    # Save master file
    master_path = TRAINING_DIR / "master_cases.json"
    with open(master_path, 'w') as f:
        json.dump({
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_cases": len(standardized_cases),
                "mri_cases": len(mri_cases),
                "description": "Master deduplicated case file for fine-tuning"
            },
            "cases": standardized_cases
        }, f, indent=2)
    print(f"\n💾 Saved master file: {master_path}")

    # Generate gap analysis
    gap_md = generate_gap_analysis(coverage)
    gap_path = TRAINING_DIR / "gap_analysis.md"
    with open(gap_path, 'w') as f:
        f.write(gap_md)
    print(f"📋 Saved gap analysis: {gap_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SCRAPE PROGRESS")
    print("=" * 60)
    print(f"Total unique cases: {len(standardized_cases)}")
    print(f"Cases with MRI findings: {len(mri_cases)}")
    print(f"\nBy source:")
    for src, count in sorted(coverage['source'].items(), key=lambda x: -x[1]):
        print(f"  - {src}: {count}")
    print(f"\nBy severity:")
    for sev, count in coverage['severity'].items():
        print(f"  - {sev}: {count}")
    print(f"\nGaps remaining:")
    total_needed = max(0, 200 - len(standardized_cases))
    if total_needed > 0:
        print(f"  ⚠️ Need {total_needed} more cases to reach 200 target")
    else:
        print(f"  ✓ Target met!")

    return standardized_cases, coverage

if __name__ == "__main__":
    cases, coverage = main()
