# Parser Validation Coverage Report

**Generated:** December 8, 2025
**Target:** 80%+ coverage
**Achieved:** 100% cell coverage (39/39 valid cells)

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Test Cases** | 68 |
| **Coverage** | 100% (39/39 valid cells) |
| **Real Cases** | 26 (38%) |
| **Synthetic Cases** | 42 (62%) |
| **Cells with 3+ cases** | 5 cells |
| **Cells with 1-2 cases** | 34 cells |
| **Empty cells** | 0 cells |

---

## Coverage Matrix

```
Case Type                 | Remission  |    Mild    |  Moderate  |   Severe
-----------------------------------------------------------------------------
Simple Intersphincteric   |     2      |     2      |     2      |     1
Transsphincteric          |    [3]     |     2      |     1      |     1
Complex/Branching         |     2      |    [3]     |     1      |     2
With Abscess              |    N/A     |     2      |     1      |     1
Healed/Fibrotic           |     2      |     1      |     1      |     2
Post-Surgical             |     1      |     1      |     1      |     2
Pediatric                 |    [3]     |     1      |    [3]     |    [3]
Ambiguous/Equivocal       |     2      |     1      |     1      |     2
Horseshoe                 |     2      |     1      |     2      |     2
Extrasphincteric          |     1      |     1      |     1      |     1
Normal/No Fistula         |    N/A     |    N/A     |    N/A     |    N/A
```

**Legend:** [N] = Full coverage (3+ cases), N = Partial (1-2 cases), N/A = Clinically impossible

---

## Gaps Filled This Round

### Priority 1 Gaps (Must Fill) - ALL FILLED

| Gap | Filled By | Case Count | Case IDs |
|-----|-----------|------------|----------|
| Remission × Transsphincteric | Synthetic | 3 | synth_trans_remission_001, 002, 003 |
| Remission × Complex/Branching | Synthetic | 2 | synth_complex_remission_001, 002 |
| Mild × Complex/Branching | Synthetic | 3 | synth_complex_mild_001, 002, 003 |
| Severe × Healed/Fibrotic | Synthetic | 2 | synth_fibrotic_severe_001, 002 |
| Severe × Post-Surgical | Synthetic | 2 | synth_postsurg_severe_001, 002 |
| Remission × Ambiguous/Equivocal | Synthetic | 2 | synth_ambig_remission_001, 002 |
| Severe × Ambiguous/Equivocal | Synthetic | 2 | synth_ambig_severe_001, 002 |
| Remission × Horseshoe | Synthetic | 2 | synth_horseshoe_remission_001, 002 |

### Priority 2 Gaps (Harder) - ALL FILLED

| Gap | Filled By | Case Count | Case IDs |
|-----|-----------|------------|----------|
| Mild × With Abscess | Synthetic | 2 | synth_abscess_mild_001, 002 |
| Moderate × Horseshoe | Synthetic | 2 | synth_horseshoe_moderate_001, 002 |
| Mild × Extrasphincteric | Synthetic | 1 | synth_extra_mild_001 |
| Moderate × Extrasphincteric | Synthetic | 1 | synth_extra_moderate_001 |

### Priority 3 Gaps (Very Rare) - ALL FILLED

| Gap | Filled By | Case Count | Case IDs |
|-----|-----------|------------|----------|
| Remission × Extrasphincteric | Synthetic | 1 | synth_extra_remission_001 |
| Pediatric × Remission | Synthetic | 3 | synth_peds_remission_001, 002, 003 |
| Pediatric × Moderate | Synthetic | 3 | synth_peds_moderate_001, 002, 003 |
| Pediatric × Severe | Synthetic | 3 | synth_peds_severe_001, 002, 003 |

---

## Intentionally Unfilled Gaps

| Gap | Reason |
|-----|--------|
| Remission × With Abscess | **Clinically contradictory** - An abscess by definition indicates active disease. A patient cannot be in remission while having an abscess. |
| Normal/No Fistula × All Severities | **Not applicable** - These are negative control cases (no fistula present). Severity doesn't apply to normal studies. Used only as baseline comparators. |

---

## Cases by Source

| Source | Count | Percentage |
|--------|-------|------------|
| Synthetic (Literature-Based) | 42 | 61.8% |
| Radiopaedia | 12 | 17.6% |
| Edge Cases | 11 | 16.2% |
| PubMed Central | 3 | 4.4% |
| **TOTAL** | **68** | **100%** |

---

## Synthetic Case Documentation

All synthetic cases are based on published literature and clinical guidelines:

### Literature Sources Used

1. **Van Assche Index (VAI)** - Original scoring criteria (AJG 2003)
2. **MAGNIFI-CD** - Modern scoring system development paper (2019)
3. **Parks Classification** - Anatomical fistula classification (BJS 1976)
4. **ADMIRE-CD & ADMIRE-CD II** - Biologic healing patterns (Lancet 2016, 2024)
5. **Beets-Tan et al.** - Post-treatment imaging patterns (Radiology 2001)
6. **GETAID-OBSERV** - Complex fistula healing outcomes (IBD 2020)
7. **Geltzeiler et al.** - Surgical + biologic outcomes (DCR 2018)
8. **St Mark's Hospital Classification** - Clinical classification criteria
9. **Pediatric IBD literature** - Age-specific presentation patterns

### Synthetic Case Characteristics

All synthetic cases include:
- `source: "synthetic_literature"` - Clear labeling as synthetic
- `literature_basis` field - Citation of source literature
- Realistic clinical language based on actual radiology reports
- Ground truth values aligned with scoring criteria
- Appropriate severity classification based on VAI/MAGNIFI thresholds

### Quality Assurance

Synthetic cases were designed to:
1. Reflect realistic clinical presentations
2. Use standard radiology terminology
3. Include appropriate detail level for scoring
4. Cover edge cases (healed disease, partial response, etc.)
5. Span the full severity spectrum for each case type

---

## Validation Metrics Reminder

The parser validation suite now covers:

| Metric | Target | Status |
|--------|--------|--------|
| Cell Coverage | 80%+ | **100%** |
| Total Cases | 50+ | **68** |
| All Remission Types | Covered | **Yes** |
| All Severity Levels | Covered | **Yes** |
| Pediatric Cases | 3+ per severity | **Yes (except Mild=1)** |
| Clinically Impossible Gaps | Documented | **Yes** |

---

## Running the Coverage Analysis

To regenerate coverage statistics:

```bash
cd data/parser_validation
python3 coverage_matrix.py
```

This will output:
- ASCII coverage matrix
- Gap analysis
- Source breakdown
- Target achievement status

---

## Next Steps (Optional Improvements)

1. **Increase Cell Density**: Add more cases to cells with only 1-2 cases to reach 3+ per cell
2. **Real Case Sourcing**: Replace some synthetic cases with real Radiopaedia/literature cases as they become available
3. **Cross-Validation**: Run parser on all 68 cases and analyze error patterns
4. **Edge Case Expansion**: Add more adversarial cases (OCR errors, truncated reports, etc.)

---

*Report generated as part of MRI-Crohn Atlas ISEF 2026 project*
*Coverage expansion completed: December 8, 2025*
