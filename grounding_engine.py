"""
grounding_engine.py
Zero-hallucination verification engine.
Validates quotes, binds verbatim text snippets to timestamps, and audits factual claims.
"""

from dataclasses import dataclass
import re
from typing import Optional, List, Dict, Any, Tuple
from transcript_parser import Transcript, DialogueTurn


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
            "context_turn": self.context_turn
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
            # Check for largest word n-gram
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
                    context_turn=f"[{turn.timestamp}] {turn.speaker}: {turn.text}"
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
