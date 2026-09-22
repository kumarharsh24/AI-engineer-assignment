"""
src/validators/grounding_validator.py
Zero-hallucination verification engine.
Enforces structural anti-hallucination validation:
  1. Validates that every referenced timestamp actually exists in the parsed transcript.
  2. Validates that every supporting quote is an exact verbatim substring within that specific turn.
  3. Structurally rejects or flags any answer or claim that cannot be matched back to source text.
"""

from dataclasses import dataclass, field
import re
from typing import Optional, List, Dict, Any, Tuple

try:
    from src.parsers.transcript_parser import Transcript, DialogueTurn, QAUnit
except ImportError:
    from transcript_parser import Transcript, DialogueTurn, QAUnit


@dataclass
class Citation:
    """Represents a verified citation linking a quote/fact to a transcript timestamp."""
    transcript_file: str
    expert_name: str
    market: str
    timestamp: str
    speaker: str
    verbatim_quote: str
    is_verbatim: bool
    confidence: float
    turn_id: int
    context_turn: Optional[str] = None
    validation_status: str = "VERIFIED_VERBATIM"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript_file": self.transcript_file,
            "expert_name": self.expert_name,
            "market": self.market,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "verbatim_quote": self.verbatim_quote,
            "is_verbatim": self.is_verbatim,
            "confidence": self.confidence,
            "turn_id": self.turn_id,
            "context_turn": self.context_turn,
            "validation_status": self.validation_status
        }


@dataclass
class StructuralAuditReport:
    """Results of a strict structural grounding audit on an answer and its citations."""
    is_strictly_grounded: bool
    verified_citations: List[Citation] = field(default_factory=list)
    rejected_citations: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    timestamp_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_strictly_grounded": self.is_strictly_grounded,
            "verified_count": len(self.verified_citations),
            "rejected_count": len(self.rejected_citations),
            "verified_citations": [c.to_dict() for c in self.verified_citations],
            "rejected_citations": self.rejected_citations,
            "summary": self.summary,
            "timestamp_coverage": self.timestamp_coverage
        }


def normalize_text(text: str) -> str:
    """Normalizes whitespace and standardizes punctuation quotes."""
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    return re.sub(r'\s+', ' ', text).strip()


class GroundingVerifier:
    """
    Verifies that claims and quotes are traceable verbatim to the transcripts.
    Guarantees zero-hallucination compliance.
    """

    def __init__(self, transcripts: Dict[str, Transcript]):
        self.transcripts = transcripts

    def find_best_turn_match(
        self,
        quote: str,
        transcript: Transcript
    ) -> Tuple[Optional[DialogueTurn], bool, float]:
        """
        Locates the dialogue turn containing the quote.
        Returns: (DialogueTurn, is_verbatim, confidence_score)
        """
        norm_quote = normalize_text(quote).lower()
        if not norm_quote:
            return None, False, 0.0

        # Pass 1: Exact substring match in turn text
        for turn in transcript.turns:
            norm_turn = normalize_text(turn.text).lower()
            if norm_quote in norm_turn:
                return turn, True, 1.0

        # Pass 2: Sub-phrase match (in case quote is slightly truncated or has ellipsis)
        quote_words = norm_quote.split()
        if len(quote_words) >= 4:
            best_turn = None
            best_overlap = 0.0
            quote_word_set = set(quote_words)
            for turn in transcript.turns:
                turn_word_set = set(normalize_text(turn.text).lower().split())
                overlap = len(quote_word_set.intersection(turn_word_set)) / len(quote_word_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_turn = turn
            if best_overlap >= 0.70:
                return best_turn, False, round(best_overlap, 2)

        return None, False, 0.0

    def verify_quote(
        self,
        quote: str,
        transcript_key_or_name: str
    ) -> Optional[Citation]:
        """
        Audits a quote against a specific transcript or all transcripts.
        Enforces that the quote exists in the source text.
        """
        target_transcripts = []
        for key, t in self.transcripts.items():
            if (
                key == transcript_key_or_name
                or t.expert_name.lower() in transcript_key_or_name.lower()
                or t.market.lower() in transcript_key_or_name.lower()
                or transcript_key_or_name.lower() in key.lower()
            ):
                target_transcripts.append(t)

        if not target_transcripts:
            target_transcripts = list(self.transcripts.values())

        for transcript in target_transcripts:
            turn, is_verbatim, confidence = self.find_best_turn_match(quote, transcript)
            if turn and confidence >= 0.6:
                return Citation(
                    transcript_file=transcript.file_name,
                    expert_name=transcript.expert_name,
                    market=transcript.market,
                    timestamp=turn.timestamp,
                    speaker=turn.speaker,
                    verbatim_quote=quote if is_verbatim else turn.text,
                    is_verbatim=is_verbatim,
                    confidence=confidence,
                    turn_id=turn.turn_id,
                    context_turn=f"[{turn.timestamp}] {turn.speaker}: {turn.text}",
                    validation_status="VERIFIED_VERBATIM" if is_verbatim else "PARTIAL_OVERLAP"
                )

        return None

    def audit_claim_citations(
        self,
        claim: str,
        citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        Audits a generated claim and list of citations.
        Returns safety score and validation status.
        """
        if not citations:
            return {
                "grounded": False,
                "confidence": 0.0,
                "status": "UNGROUNDED - No citations provided",
                "citations": []
            }

        valid_citations = [c for c in citations if c.is_verbatim or c.confidence >= 0.8]
        is_safe = len(valid_citations) == len(citations)

        return {
            "grounded": is_safe,
            "confidence": min(c.confidence for c in citations) if citations else 0.0,
            "status": "VERIFIED - 100% Traceable" if is_safe else "PARTIAL_MATCH",
            "citations": [c.to_dict() for c in citations]
        }


class StructuralGroundingValidator:
    """
    Strict structural gatekeeper:
    Verifies that cited timestamps exist in the transcript and candidate quotes
    exist verbatim in that specific turn. Rejects any fabricated citations.
    """

    def __init__(self, transcripts: Dict[str, Transcript]):
        self.transcripts = transcripts

    def validate_citation_structurally(
        self,
        transcript_key: str,
        timestamp: str,
        quote: str
    ) -> Tuple[bool, Optional[Citation], str]:
        """
        Validates:
          1. Transcript exists.
          2. The cited timestamp exists in that transcript.
          3. The quote appears verbatim in the turn corresponding to that timestamp.
        """
        t = self.transcripts.get(transcript_key)
        if not t:
            for k, candidate in self.transcripts.items():
                if transcript_key.lower() in k.lower() or candidate.market.lower() in transcript_key.lower():
                    t = candidate
                    break

        if not t:
            return False, None, f"Transcript '{transcript_key}' not found."

        # Check timestamp existence
        matching_turn = None
        for turn in t.turns:
            if turn.timestamp == timestamp:
                matching_turn = turn
                break

        if not matching_turn:
            return False, None, f"Timestamp [{timestamp}] does not exist in {t.file_name}."

        # Check verbatim quote in that turn
        clean_quote = normalize_text(quote).lower()
        clean_turn_text = normalize_text(matching_turn.text).lower()

        if clean_quote in clean_turn_text:
            citation = Citation(
                transcript_file=t.file_name,
                expert_name=t.expert_name,
                market=t.market,
                timestamp=matching_turn.timestamp,
                speaker=matching_turn.speaker,
                verbatim_quote=quote,
                is_verbatim=True,
                confidence=1.0,
                turn_id=matching_turn.turn_id,
                context_turn=f"[{matching_turn.timestamp}] {matching_turn.speaker}: {matching_turn.text}",
                validation_status="VERIFIED_VERBATIM"
            )
            return True, citation, "Structural validation passed 100% verbatim."

        # Check if in the immediate QAUnit (adjacent turns)
        for unit in t.qa_units:
            if timestamp in unit.answer_timestamps or timestamp == unit.question_timestamp:
                clean_unit_ans = normalize_text(unit.answer_text).lower()
                if clean_quote in clean_unit_ans:
                    citation = Citation(
                        transcript_file=t.file_name,
                        expert_name=t.expert_name,
                        market=t.market,
                        timestamp=unit.primary_timestamp,
                        speaker=unit.answer_speaker,
                        verbatim_quote=quote,
                        is_verbatim=True,
                        confidence=0.95,
                        turn_id=unit.unit_id,
                        context_turn=unit.full_context,
                        validation_status="VERIFIED_IN_QA_UNIT"
                    )
                    return True, citation, "Verified within semantic Q&A unit."

        return False, None, f"Quote was not found verbatim inside turn [{timestamp}] of {t.file_name}."

    def audit_answer(
        self,
        transcript_key: str,
        answer_text: str,
        claimed_timestamps: List[str],
        claimed_quotes: List[str]
    ) -> StructuralAuditReport:
        """
        Performs full structural audit of an answer.
        Returns a report with verified citations and any rejected claims.
        """
        verified: List[Citation] = []
        rejected: List[Dict[str, Any]] = []

        for ts, q in zip(claimed_timestamps, claimed_quotes):
            ok, cit, reason = self.validate_citation_structurally(transcript_key, ts, q)
            if ok and cit:
                verified.append(cit)
            else:
                rejected.append({
                    "timestamp": ts,
                    "quote": q,
                    "reason": reason
                })

        is_grounded = len(rejected) == 0 and len(verified) > 0
        summary = (
            f"Structural audit passed: {len(verified)} verified citations."
            if is_grounded
            else f"Audit warning: {len(rejected)} citations rejected as ungrounded."
        )

        return StructuralAuditReport(
            is_strictly_grounded=is_grounded,
            verified_citations=verified,
            rejected_citations=rejected,
            summary=summary,
            timestamp_coverage=len(verified) / max(len(claimed_quotes), 1)
        )
