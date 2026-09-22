"""
transcript_parser.py
Parses raw interview transcripts into structured data classes with turn-level metadata,
timestamps, and character offsets for grounded traceability.
"""

from dataclasses import dataclass, field
import os
import re
from typing import List, Optional, Dict, Any


@dataclass
class DialogueTurn:
    """Represents a single speaker turn in an interview transcript."""
    turn_id: int
    timestamp: str  # Format: "MM:SS" or "HH:MM:SS"
    speaker: str
    text: str
    start_char: int
    end_char: int
    line_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "line_number": self.line_number
        }


@dataclass
class Transcript:
    """Represents a fully parsed transcript with expert metadata and dialogue turns."""
    file_name: str
    expert_id: str
    expert_name: str
    role: str
    market: str
    raw_text: str
    turns: List[DialogueTurn] = field(default_factory=list)

    @property
    def full_dialogue_text(self) -> str:
        """Returns dialogue lines formatted with timestamp and speaker."""
        return "\n\n".join(
            f"[{t.timestamp}] {t.speaker}: {t.text}" for t in self.turns
        )

    def find_quote_turn(self, quote: str) -> Optional[DialogueTurn]:
        """Find the dialogue turn containing a specific quote snippet."""
        clean_quote = re.sub(r'\s+', ' ', quote.strip().lower())
        for turn in self.turns:
            clean_turn = re.sub(r'\s+', ' ', turn.text.lower())
            if clean_quote in clean_turn:
                return turn
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "expert_id": self.expert_id,
            "expert_name": self.expert_name,
            "role": self.role,
            "market": self.market,
            "turn_count": len(self.turns),
            "turns": [t.to_dict() for t in self.turns]
        }


def parse_transcript_text(raw_text: str, file_name: str = "transcript.txt") -> Transcript:
    """
    Parses raw text of an interview transcript.
    Extracts metadata from header and turns with timestamps and speakers.
    """
    lines = raw_text.splitlines()

    expert_id = "Expert"
    expert_name = "Unknown Expert"
    role = "Unknown Role"
    market = "Unknown Market"

    # Header parsing
    content_start_line = 0
    for idx, line in enumerate(lines[:10]):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Expert line: e.g. "Expert 1 – Dr. Jean Martin" or "Expert 2 - Anna Keller"
        expert_match = re.match(r"(Expert\s*\d*)\s*[\–\-:]\s*(.+)", line_clean, re.IGNORECASE)
        if expert_match:
            expert_id = expert_match.group(1).strip()
            expert_name = expert_match.group(2).strip()
            content_start_line = max(content_start_line, idx + 1)
            continue

        # Role line
        role_match = re.match(r"Role:\s*(.+)", line_clean, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).strip()
            content_start_line = max(content_start_line, idx + 1)
            continue

        # Market line
        market_match = re.match(r"Market:\s*(.+)", line_clean, re.IGNORECASE)
        if market_match:
            market = market_match.group(1).strip()
            content_start_line = max(content_start_line, idx + 1)
            continue

        # If we encounter a timestamp like "00:00" or "[00:00]", dialogue starts here
        if re.match(r"^\[?\d{1,2}:\d{2}(?::\d{2})?\]?$", line_clean):
            content_start_line = idx
            break

    # Body parsing for timestamps and speaker turns
    turns: List[DialogueTurn] = []
    current_timestamp = "00:00"
    current_speaker = ""
    current_text_parts: List[str] = []
    current_start_char = 0
    current_line_num = 1
    turn_counter = 1

    # Track character offsets across lines
    char_offset = 0
    line_offsets = []
    for line in lines:
        line_offsets.append(char_offset)
        char_offset += len(line) + 1  # +1 for newline

    i = content_start_line
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for timestamp line e.g. "00:00" or "[00:18]"
        ts_match = re.match(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?$", stripped)
        if ts_match:
            # If we had a previous turn accumulating, save it
            if current_speaker and current_text_parts:
                turn_text = " ".join(current_text_parts).strip()
                turns.append(DialogueTurn(
                    turn_id=turn_counter,
                    timestamp=current_timestamp,
                    speaker=current_speaker,
                    text=turn_text,
                    start_char=current_start_char,
                    end_char=current_start_char + len(turn_text),
                    line_number=current_line_num
                ))
                turn_counter += 1
                current_text_parts = []
                current_speaker = ""

            current_timestamp = ts_match.group(1)
            i += 1
            # Next line usually has "Speaker: Text"
            if i < len(lines):
                speaker_line = lines[i].strip()
                speaker_match = re.match(r"^([^:]+):\s*(.*)", speaker_line)
                if speaker_match:
                    current_speaker = speaker_match.group(1).strip()
                    first_text = speaker_match.group(2).strip()
                    current_start_char = line_offsets[i]
                    current_line_num = i + 1
                    if first_text:
                        current_text_parts.append(first_text)
                else:
                    current_text_parts.append(speaker_line)
            i += 1
            continue

        # Check if line begins with "Speaker: Text" without a timestamp preceding
        speaker_match = re.match(r"^([A-Za-z0-9\.\s]+):\s*(.*)", stripped)
        if speaker_match and not stripped.startswith("http"):
            # If we had an existing turn, save it
            if current_speaker and current_text_parts:
                turn_text = " ".join(current_text_parts).strip()
                turns.append(DialogueTurn(
                    turn_id=turn_counter,
                    timestamp=current_timestamp,
                    speaker=current_speaker,
                    text=turn_text,
                    start_char=current_start_char,
                    end_char=current_start_char + len(turn_text),
                    line_number=current_line_num
                ))
                turn_counter += 1
                current_text_parts = []

            current_speaker = speaker_match.group(1).strip()
            first_text = speaker_match.group(2).strip()
            current_start_char = line_offsets[i]
            current_line_num = i + 1
            if first_text:
                current_text_parts.append(first_text)
            i += 1
            continue

        # Continuation of previous turn
        if stripped and current_speaker:
            current_text_parts.append(stripped)

        i += 1

    # Add final turn
    if current_speaker and current_text_parts:
        turn_text = " ".join(current_text_parts).strip()
        turns.append(DialogueTurn(
            turn_id=turn_counter,
            timestamp=current_timestamp,
            speaker=current_speaker,
            text=turn_text,
            start_char=current_start_char,
            end_char=current_start_char + len(turn_text),
            line_number=current_line_num
        ))

    return Transcript(
        file_name=file_name,
        expert_id=expert_id,
        expert_name=expert_name,
        role=role,
        market=market,
        raw_text=raw_text,
        turns=turns
    )


def load_transcripts_from_dir(directory: str) -> Dict[str, Transcript]:
    """Loads all default transcript files from a specified directory."""
    transcripts = {}
    expected_files = [
        "Transcript_1_France.txt",
        "Transcript_2_Germany.txt",
        "Transcript_3_UK.txt"
    ]
    for fname in expected_files:
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            transcripts[fname] = parse_transcript_text(content, fname)

    # Fallback to any .txt files matching Transcript_ if standard names missing
    if not transcripts:
        for fname in sorted(os.listdir(directory)):
            if fname.endswith(".txt") and "Transcript" in fname:
                path = os.path.join(directory, fname)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                transcripts[fname] = parse_transcript_text(content, fname)

    return transcripts
