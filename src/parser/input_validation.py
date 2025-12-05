#!/usr/bin/env python3
"""
Input Validation Module for MRI Report Parser
Provides robust input validation and error handling

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import re
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MRIParser')


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    cleaned_input: Optional[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """Validates and sanitizes inputs for the MRI Report Parser"""

    # Minimum and maximum report lengths
    MIN_REPORT_LENGTH = 20
    MAX_REPORT_LENGTH = 50000

    # Characters that might indicate malicious input
    SUSPICIOUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'on\w+\s*=',  # onclick=, onerror=, etc.
    ]

    # Medical terms that should appear in valid MRI reports
    MEDICAL_INDICATORS = [
        'mri', 'fistula', 'tract', 'abscess', 'sphincter', 'perianal',
        't2', 'hyperintens', 'imaging', 'findings', 'pelvis', 'anal',
        'crohn', 'inflammatory', 'collection', 'enhancement'
    ]

    @classmethod
    def validate_report_text(cls, text: Any) -> ValidationResult:
        """
        Validate MRI report text input.

        Args:
            text: The input text to validate

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        warnings = []

        # Type check
        if text is None:
            return ValidationResult(
                is_valid=False,
                error_message="Report text cannot be None"
            )

        if not isinstance(text, str):
            try:
                text = str(text)
                warnings.append("Input was converted to string")
            except Exception:
                return ValidationResult(
                    is_valid=False,
                    error_message="Report text must be a string or string-convertible"
                )

        # Empty check
        if not text.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Report text cannot be empty"
            )

        # Length check - too short
        if len(text.strip()) < cls.MIN_REPORT_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_message=f"Report text too short (minimum {cls.MIN_REPORT_LENGTH} characters)"
            )

        # Length check - too long (truncate with warning)
        cleaned_text = text.strip()
        if len(cleaned_text) > cls.MAX_REPORT_LENGTH:
            cleaned_text = cleaned_text[:cls.MAX_REPORT_LENGTH]
            warnings.append(f"Report truncated to {cls.MAX_REPORT_LENGTH} characters")

        # Security check - suspicious patterns
        text_lower = cleaned_text.lower()
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_message="Report contains potentially unsafe content"
                )

        # Medical content check
        medical_terms_found = sum(1 for term in cls.MEDICAL_INDICATORS if term in text_lower)
        if medical_terms_found == 0:
            warnings.append("No medical terminology detected - may not be an MRI report")
        elif medical_terms_found < 2:
            warnings.append("Limited medical terminology - results may be unreliable")

        # Character encoding check
        try:
            cleaned_text.encode('utf-8')
        except UnicodeEncodeError:
            # Try to fix encoding issues
            cleaned_text = cleaned_text.encode('utf-8', errors='ignore').decode('utf-8')
            warnings.append("Fixed encoding issues in report text")

        # Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            cleaned_input=cleaned_text
        )

    @classmethod
    def validate_api_key(cls, api_key: Any) -> ValidationResult:
        """Validate API key format"""
        if not api_key:
            return ValidationResult(
                is_valid=False,
                error_message="API key is required"
            )

        if not isinstance(api_key, str):
            return ValidationResult(
                is_valid=False,
                error_message="API key must be a string"
            )

        # OpenRouter keys start with 'sk-or-'
        if not api_key.startswith('sk-or-'):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid API key format (expected OpenRouter key)"
            )

        if len(api_key) < 20:
            return ValidationResult(
                is_valid=False,
                error_message="API key appears incomplete"
            )

        return ValidationResult(is_valid=True, cleaned_input=api_key)

    @classmethod
    def validate_features(cls, features: Dict) -> ValidationResult:
        """Validate extracted features structure"""
        warnings = []

        if not isinstance(features, dict):
            return ValidationResult(
                is_valid=False,
                error_message="Features must be a dictionary"
            )

        # Required fields
        required_fields = ['fistula_count', 't2_hyperintensity']
        missing = [f for f in required_fields if f not in features]
        if missing:
            warnings.append(f"Missing recommended fields: {', '.join(missing)}")

        # Type validation for known fields
        if 'fistula_count' in features:
            count = features['fistula_count']
            if count is not None and not isinstance(count, (int, float)):
                try:
                    features['fistula_count'] = int(count)
                except (ValueError, TypeError):
                    features['fistula_count'] = 0
                    warnings.append("Invalid fistula_count, defaulted to 0")

        # Ensure fistula_count is non-negative
        if features.get('fistula_count', 0) < 0:
            features['fistula_count'] = 0
            warnings.append("Negative fistula_count corrected to 0")

        # Validate t2_hyperintensity_degree
        valid_degrees = ['none', 'mild', 'moderate', 'marked', None]
        if features.get('t2_hyperintensity_degree') not in valid_degrees:
            features['t2_hyperintensity_degree'] = 'moderate'
            warnings.append("Invalid T2 degree, defaulted to 'moderate'")

        # Validate extension
        valid_extensions = ['none', 'mild', 'moderate', 'severe', None]
        if features.get('extension') not in valid_extensions:
            features['extension'] = 'none'
            warnings.append("Invalid extension, defaulted to 'none'")

        # Validate boolean fields
        bool_fields = ['t2_hyperintensity', 'collections_abscesses',
                       'rectal_wall_involvement', 'inflammatory_mass']
        for field in bool_fields:
            if field in features and not isinstance(features[field], bool):
                features[field] = bool(features[field])

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            cleaned_input=None  # Features dict is modified in place
        )


def safe_calculate_scores(features: Dict, parser) -> Tuple[float, float, str]:
    """
    Safely calculate VAI and MAGNIFI scores with error handling.

    Args:
        features: Extracted features dictionary
        parser: MRIReportParser instance

    Returns:
        Tuple of (VAI score, MAGNIFI score, error message if any)
    """
    try:
        # Validate features first
        validation = InputValidator.validate_features(features)
        if validation.warnings:
            logger.warning(f"Feature validation warnings: {validation.warnings}")

        vai = parser.calculate_vai(features)
        magnifi = parser.calculate_magnifi(features)

        # Sanity checks
        if vai < 0 or vai > 22:
            logger.warning(f"VAI score {vai} out of range [0-22], clamping")
            vai = max(0, min(22, vai))

        if magnifi < 0 or magnifi > 25:
            logger.warning(f"MAGNIFI score {magnifi} out of range [0-25], clamping")
            magnifi = max(0, min(25, magnifi))

        return vai, magnifi, None

    except Exception as e:
        logger.error(f"Error calculating scores: {str(e)}")
        return 0, 0, str(e)


class ParserError(Exception):
    """Custom exception for parser errors"""
    pass


class APIError(ParserError):
    """Error related to API calls"""
    pass


class ExtractionError(ParserError):
    """Error during feature extraction"""
    pass


class ValidationError(ParserError):
    """Error during input validation"""
    pass
