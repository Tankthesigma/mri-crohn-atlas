/**
 * MRI Report Parser for MRI-Crohn Atlas
 * Extracts structured features from free-text radiology reports
 * and calculates VAI and MAGNIFI-CD scores with traceability
 */

const MRIReportParser = (function() {
    'use strict';

    // OpenRouter API configuration
    const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

    /**
     * Get API key from config. Returns null if not configured.
     * Config should be loaded via config.js (gitignored) before this script.
     */
    function getApiKey() {
        if (typeof window !== 'undefined' && window.CONFIG && window.CONFIG.OPENROUTER_API_KEY) {
            const key = window.CONFIG.OPENROUTER_API_KEY;
            // Check it's not the placeholder
            if (key && key !== 'your-openrouter-api-key-here' && key.startsWith('sk-or-')) {
                return key;
            }
        }
        return null;
    }

    // VAI Component Weights (based on Van Assche Index)
    const VAI_WEIGHTS = {
        fistula_count: { 0: 0, 1: 1, 2: 2, '3+': 2 },  // 0-2 points
        fistula_location: { simple: 1, complex: 2, horseshoe: 2 },  // 0-2 points
        extension: { none: 0, mild: 2, moderate: 3, severe: 4 },  // 0-4 points
        t2_hyperintensity: { false: 0, mild: 4, moderate: 6, marked: 8 },  // 0-8 points
        collections: { false: 0, true: 4 },  // 0-4 points
        rectal_wall: { false: 0, true: 2 }  // 0-2 points
        // Total max: 22
    };

    // MAGNIFI-CD Component Weights
    const MAGNIFI_WEIGHTS = {
        fistula_count: { 0: 0, 1: 1, 2: 2, '3+': 3 },
        fistula_activity: { none: 0, mild: 2, moderate: 4, marked: 6 },
        collections: { false: 0, small: 2, large: 4 },
        inflammatory_mass: { false: 0, true: 3 },
        rectal_wall: { false: 0, true: 2 },
        predominant_feature: { fibrotic: 0, mixed: 2, inflammatory: 4 },
        extent: { localized: 1, moderate: 2, extensive: 3 }
        // Total max: ~25
    };

    // Feature highlighting colors
    const HIGHLIGHT_COLORS = {
        fistula_type: '#3b82f6',      // blue
        fistula_count: '#0066CC',     // blue
        t2_hyperintensity: '#f97316', // orange
        extension: '#22c55e',         // green
        collections: '#ef4444',       // red
        rectal_wall: '#ec4899',       // pink
        inflammatory_mass: '#f59e0b', // amber
        predominant_feature: '#06b6d4' // cyan
    };

    // Extraction prompt template with traceability
    const EXTRACTION_PROMPT = `You are an expert radiologist analyzing MRI findings for perianal fistulas.

Extract structured features from the following radiology report. For EACH feature you identify, include the EXACT quote from the report that supports it.

REPORT TEXT:
{report_text}

Return a JSON object with this EXACT structure:
{
    "features": {
        "fistula_count": {
            "value": <number or null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "fistula_type": {
            "value": <"intersphincteric"|"transsphincteric"|"suprasphincteric"|"extrasphincteric"|"complex"|"simple"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "t2_hyperintensity": {
            "value": <true|false|null>,
            "degree": <"mild"|"moderate"|"marked"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "extension": {
            "value": <"none"|"mild"|"moderate"|"severe"|null>,
            "description": "<description of extension pattern>",
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "collections_abscesses": {
            "value": <true|false|null>,
            "size": <"small"|"large"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "rectal_wall_involvement": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "inflammatory_mass": {
            "value": <true|false|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "predominant_feature": {
            "value": <"inflammatory"|"fibrotic"|"mixed"|null>,
            "confidence": <"high"|"medium"|"low">,
            "evidence": "<exact quote from report>"
        },
        "sphincter_involvement": {
            "internal": <true|false|null>,
            "external": <true|false|null>,
            "evidence": "<exact quote from report>"
        },
        "internal_opening": {
            "location": "<clock position or description>",
            "evidence": "<exact quote from report>"
        },
        "branching": {
            "value": <true|false|null>,
            "evidence": "<exact quote from report>"
        }
    },
    "overall_assessment": {
        "activity": <"active"|"healing"|"healed"|"chronic">,
        "severity": <"mild"|"moderate"|"severe">,
        "confidence": <"high"|"medium"|"low">
    },
    "clinical_notes": "<any additional relevant observations>"
}

IMPORTANT:
- Use null if a feature is not mentioned or cannot be determined
- The "evidence" field must contain the EXACT text from the report
- Be conservative - only report features explicitly stated
- Return ONLY valid JSON, no markdown or explanations`;

    /**
     * Check if API is configured and ready to use
     */
    function isApiConfigured() {
        return getApiKey() !== null;
    }

    /**
     * Parse a report using LLM
     */
    async function parseReport(reportText) {
        const apiKey = getApiKey();
        if (!apiKey) {
            throw new Error('API key not configured. Copy config.example.js to config.js and add your OpenRouter API key. See CLAUDE.md for instructions.');
        }

        if (!reportText || reportText.trim().length < 20) {
            throw new Error('Report text is too short. Please enter a valid MRI report.');
        }

        const prompt = EXTRACTION_PROMPT.replace('{report_text}', reportText.trim());

        try {
            const response = await fetch(OPENROUTER_URL, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': window.location.origin,
                    'X-Title': 'MRI-Crohn Atlas Parser'
                },
                body: JSON.stringify({
                    model: 'deepseek/deepseek-chat',
                    messages: [{ role: 'user', content: prompt }],
                    temperature: 0.1,
                    max_tokens: 1500
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error?.message || 'API request failed');
            }

            const data = await response.json();
            const content = data.choices?.[0]?.message?.content || '';

            // Parse JSON from response
            const jsonMatch = content.match(/\{[\s\S]*\}/);
            if (!jsonMatch) {
                throw new Error('Failed to parse LLM response');
            }

            const parsed = JSON.parse(jsonMatch[0]);
            return parsed;

        } catch (error) {
            console.error('Parse error:', error);
            throw error;
        }
    }

    /**
     * Calculate VAI score from extracted features
     */
    function calculateVAI(features) {
        let score = 0;
        const breakdown = {};

        // Fistula count (0-2)
        const count = features.fistula_count?.value;
        if (count !== null && count !== undefined) {
            if (count === 0) { breakdown.fistula_count = 0; }
            else if (count === 1) { breakdown.fistula_count = 1; score += 1; }
            else { breakdown.fistula_count = 2; score += 2; }
        }

        // Fistula location/type (0-2)
        const type = features.fistula_type?.value;
        if (type) {
            if (['simple', 'intersphincteric'].includes(type)) {
                breakdown.fistula_location = 1; score += 1;
            } else if (['transsphincteric', 'suprasphincteric', 'extrasphincteric', 'complex'].includes(type)) {
                breakdown.fistula_location = 2; score += 2;
            }
        }

        // Extension (0-4)
        const extension = features.extension?.value;
        if (extension) {
            const extScores = { none: 0, mild: 2, moderate: 3, severe: 4 };
            breakdown.extension = extScores[extension] || 0;
            score += breakdown.extension;
        }

        // T2 hyperintensity (0-8) - most important for activity
        const t2 = features.t2_hyperintensity;
        if (t2?.value === true) {
            const degree = t2.degree || 'moderate';
            const t2Scores = { mild: 4, moderate: 6, marked: 8 };
            breakdown.t2_hyperintensity = t2Scores[degree] || 6;
            score += breakdown.t2_hyperintensity;
        } else if (t2?.value === false) {
            breakdown.t2_hyperintensity = 0;
        }

        // Collections (0-4)
        if (features.collections_abscesses?.value === true) {
            breakdown.collections = 4;
            score += 4;
        } else if (features.collections_abscesses?.value === false) {
            breakdown.collections = 0;
        }

        // Rectal wall involvement (0-2)
        if (features.rectal_wall_involvement?.value === true) {
            breakdown.rectal_wall = 2;
            score += 2;
        } else if (features.rectal_wall_involvement?.value === false) {
            breakdown.rectal_wall = 0;
        }

        return { score: Math.min(score, 22), breakdown, max: 22 };
    }

    /**
     * Calculate MAGNIFI-CD score from extracted features
     */
    function calculateMAGNIFI(features) {
        let score = 0;
        const breakdown = {};

        // Fistula count (0-3)
        const count = features.fistula_count?.value;
        if (count !== null && count !== undefined) {
            if (count === 0) { breakdown.fistula_count = 0; }
            else if (count === 1) { breakdown.fistula_count = 1; score += 1; }
            else if (count === 2) { breakdown.fistula_count = 2; score += 2; }
            else { breakdown.fistula_count = 3; score += 3; }
        }

        // Fistula activity based on T2/enhancement (0-6)
        const t2 = features.t2_hyperintensity;
        if (t2?.value === true) {
            const degree = t2.degree || 'moderate';
            const actScores = { mild: 2, moderate: 4, marked: 6 };
            breakdown.fistula_activity = actScores[degree] || 4;
            score += breakdown.fistula_activity;
        } else if (t2?.value === false) {
            breakdown.fistula_activity = 0;
        }

        // Collections (0-4)
        const collections = features.collections_abscesses;
        if (collections?.value === true) {
            const size = collections.size || 'small';
            breakdown.collections = size === 'large' ? 4 : 2;
            score += breakdown.collections;
        } else if (collections?.value === false) {
            breakdown.collections = 0;
        }

        // Inflammatory mass (0-3)
        if (features.inflammatory_mass?.value === true) {
            breakdown.inflammatory_mass = 3;
            score += 3;
        } else if (features.inflammatory_mass?.value === false) {
            breakdown.inflammatory_mass = 0;
        }

        // Rectal wall involvement (0-2)
        if (features.rectal_wall_involvement?.value === true) {
            breakdown.rectal_wall = 2;
            score += 2;
        } else if (features.rectal_wall_involvement?.value === false) {
            breakdown.rectal_wall = 0;
        }

        // Predominant feature (0-4)
        const predominant = features.predominant_feature?.value;
        if (predominant) {
            const predScores = { fibrotic: 0, mixed: 2, inflammatory: 4 };
            breakdown.predominant_feature = predScores[predominant] || 2;
            score += breakdown.predominant_feature;
        }

        // Extension/extent (1-3)
        const extension = features.extension?.value;
        if (extension) {
            const extScores = { none: 0, mild: 1, moderate: 2, severe: 3 };
            breakdown.extent = extScores[extension] || 1;
            score += breakdown.extent;
        }

        return { score: Math.min(score, 25), breakdown, max: 25 };
    }

    /**
     * Apply crosswalk formula to verify scores
     */
    function applyCrosswalk(vai, fibrosis = 2) {
        // MAGNIFI-CD = 1.031 × VAI + 0.264 × Fibrosis × I(VAI ≤ 2) + 1.713
        const fibEffect = vai <= 2 ? 0.264 * fibrosis : 0;
        const predicted = 1.031 * vai + fibEffect + 1.713;
        const rmse = 1.10;
        return {
            predicted: predicted.toFixed(1),
            ci_lower: Math.max(0, predicted - 1.96 * rmse).toFixed(1),
            ci_upper: (predicted + 1.96 * rmse).toFixed(1)
        };
    }

    /**
     * Get clinical interpretation
     */
    function getInterpretation(vai, magnifi) {
        if (vai <= 2 && magnifi <= 4) {
            return {
                status: 'Healed',
                class: 'healed',
                description: 'Fistula appears healed with minimal residual activity. May show fibrotic changes.',
                color: '#22c55e'
            };
        } else if (vai <= 6 && magnifi <= 8) {
            return {
                status: 'Responding',
                class: 'responding',
                description: 'Partial response to treatment. Moderate residual inflammatory activity.',
                color: '#f59e0b'
            };
        } else if (vai <= 12 && magnifi <= 14) {
            return {
                status: 'Active',
                class: 'active',
                description: 'Active fistulizing disease requiring treatment optimization.',
                color: '#f97316'
            };
        } else {
            return {
                status: 'Severe',
                class: 'severe',
                description: 'Severe active disease with significant inflammatory burden. Urgent intervention needed.',
                color: '#ef4444'
            };
        }
    }

    /**
     * Highlight evidence in original text
     */
    function highlightEvidence(originalText, features) {
        let highlightedText = originalText;
        const highlights = [];

        // Collect all evidence quotes with their feature types
        const featureKeys = [
            'fistula_type', 'fistula_count', 't2_hyperintensity', 'extension',
            'collections_abscesses', 'rectal_wall_involvement', 'inflammatory_mass',
            'predominant_feature', 'sphincter_involvement', 'internal_opening', 'branching'
        ];

        for (const key of featureKeys) {
            const feature = features[key];
            if (feature?.evidence && feature.evidence.length > 5) {
                highlights.push({
                    key,
                    evidence: feature.evidence,
                    color: HIGHLIGHT_COLORS[key] || '#888'
                });
            }
        }

        // Sort by length (longest first) to avoid nested highlighting issues
        highlights.sort((a, b) => b.evidence.length - a.evidence.length);

        // Apply highlights
        for (const h of highlights) {
            const escapedEvidence = h.evidence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${escapedEvidence})`, 'gi');
            highlightedText = highlightedText.replace(regex,
                `<mark class="evidence-highlight" data-feature="${h.key}" style="background-color: ${h.color}20; border-bottom: 2px solid ${h.color}; padding: 0 2px;">$1</mark>`
            );
        }

        return { html: highlightedText, highlights };
    }

    /**
     * Main parse function that returns complete results
     */
    async function parse(reportText) {
        // Get LLM extraction
        const extracted = await parseReport(reportText);
        const features = extracted.features || {};

        // Calculate scores
        const vai = calculateVAI(features);
        const magnifi = calculateMAGNIFI(features);

        // Apply crosswalk for verification
        const crosswalk = applyCrosswalk(vai.score);

        // Get interpretation
        const interpretation = getInterpretation(vai.score, magnifi.score);

        // Generate highlighted text
        const highlighted = highlightEvidence(reportText, features);

        return {
            features,
            scores: {
                vai,
                magnifi,
                crosswalk
            },
            assessment: extracted.overall_assessment,
            interpretation,
            highlighted,
            clinical_notes: extracted.clinical_notes
        };
    }

    // Example reports for demo
    const EXAMPLE_REPORTS = [
        {
            title: "Active Transsphincteric Fistula with Abscess",
            severity: "severe",
            text: `MRI pelvis demonstrates a complex transsphincteric fistula originating at the 6 o'clock position of the anal canal. The tract shows marked T2 hyperintensity consistent with active inflammation. The fistula traverses both internal and external anal sphincters, extending into the left ischioanal fossa. A rim-enhancing fluid collection measuring 2.3 x 1.8 cm is identified in the left ischioanal fossa, consistent with abscess formation. The rectal wall shows focal thickening and enhancement at the level of the internal opening. No evidence of horseshoe extension. Clinical correlation recommended for surgical planning.`
        },
        {
            title: "Simple Intersphincteric Fistula - Healing",
            severity: "mild",
            text: `Follow-up MRI of the pelvis for known perianal fistula. A single intersphincteric fistulous tract is identified at the 3 o'clock position. Compared to prior study, there is decreased T2 signal intensity within the tract, suggesting healing. The tract remains confined to the intersphincteric space without extension through the external sphincter. No abscess or fluid collection. The external anal sphincter appears intact. Mild residual enhancement noted on post-contrast sequences. Findings consistent with treatment response.`
        },
        {
            title: "Complex Fistulizing Crohn's Disease",
            severity: "severe",
            text: `MRI demonstrates extensive perianal fistulizing disease in this patient with known Crohn's disease. Multiple fistulous tracts identified: (1) High transsphincteric fistula at 11 o'clock with horseshoe extension crossing posterior midline, (2) Intersphincteric fistula at 5 o'clock. Both tracts show marked T2 hyperintensity and avid post-contrast enhancement indicating active inflammation. Large inflammatory mass measuring 4.2 x 3.1 cm in the right ischioanal fossa with central necrosis. Moderate rectal wall thickening with mucosal hyperenhancement. Predominant inflammatory pattern without significant fibrosis. Findings indicate severe active perianal Crohn's disease requiring urgent gastroenterology consultation.`
        }
    ];

    // Public API
    return {
        parse,
        parseReport,
        calculateVAI,
        calculateMAGNIFI,
        applyCrosswalk,
        getInterpretation,
        highlightEvidence,
        isApiConfigured,
        EXAMPLE_REPORTS,
        HIGHLIGHT_COLORS
    };

})();

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.MRIReportParser = MRIReportParser;
}
