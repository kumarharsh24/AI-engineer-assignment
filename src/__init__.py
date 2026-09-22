"""
Hasamex AI Engineer Transcript Intelligence Platform
Main package root.
"""

from .parsers import (
    DialogueTurn,
    QAUnit,
    Transcript,
    parse_transcript_text,
    parse_interview_guide,
    load_transcripts_from_dir
)
from .validators import (
    Citation,
    StructuralAuditReport,
    normalize_text,
    GroundingVerifier,
    StructuralGroundingValidator
)
from .engines import (
    InterviewAnalyzer,
    ExpertAnswer,
    GUIDE_QUESTIONS,
    COMMON_THEMES,
    DISAGREEMENTS_AND_DIVERGENCES,
    EXPERT_ANALYSIS_DATA,
    CrossTranscriptQAEngine,
    SearchResult,
    QAResponse,
    SAMPLE_QUESTIONS
)
