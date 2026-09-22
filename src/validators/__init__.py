"""
src.validators package
Enforces structural anti-hallucination verification and citation audit.
"""

from .grounding_validator import (
    Citation,
    StructuralAuditReport,
    normalize_text,
    GroundingVerifier,
    StructuralGroundingValidator
)

__all__ = [
    "Citation",
    "StructuralAuditReport",
    "normalize_text",
    "GroundingVerifier",
    "StructuralGroundingValidator"
]
