"""
transcript_parser.py
Parses raw interview transcripts into structured data classes with turn-level metadata,
timestamps, and character offsets for grounded traceability.
Automatically groups interviewer questions and expert replies into Q&A units.
Supports zero-config auto-discovery of any transcript files dropped into an input folder.
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
class QAUnit:
    """
    Groups an interviewer question with the expert's subsequent answer turn(s).
    Preserves all timestamps for both question and answer turns.
    """
    unit_id: int
    expert_name: str
    market: str
    question_timestamp: str
    question_speaker: str
    question_text: str
    answer_timestamps: List[str]
    answer_speaker: str
    answer_text: str
    answer_turns: List[DialogueTurn] = field(default_factory=list)

    @property
    def primary_timestamp(self) -> str:
        """Returns the initial timestamp when the expert begins their answer."""
        return self.answer_timestamps[0] if self.answer_timestamps else self.question_timestamp

    @property
    def full_context(self) -> str:
        """Returns the combined question and answer text with timestamps."""
        return (
            f"[{self.question_timestamp}] {self.question_speaker}: {self.question_text}\n"
            f"[{self.primary_timestamp}] {self.answer_speaker}: {self.answer_text}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "expert_name": self.expert_name,
            "market": self.market,
            "question_timestamp": self.question_timestamp,
            "question_speaker": self.question_speaker,
            "question_text": self.question_text,
            "answer_timestamps": self.answer_timestamps,
            "answer_speaker": self.answer_speaker,
            "answer_text": self.answer_text,
            "primary_timestamp": self.primary_timestamp
        }


@dataclass
class Transcript:
    """Represents a fully parsed transcript with expert metadata, dialogue turns, and Q&A units."""
    file_name: str
    expert_id: str
    expert_name: str
    role: str
    market: str
    raw_text: str
    turns: List[DialogueTurn] = field(default_factory=list)
    qa_units: List[QAUnit] = field(default_factory=list)

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

    def find_quote_qa_unit(self, quote: str) -> Optional[QAUnit]:
        """Find the Q&A unit containing a specific quote snippet."""
        clean_quote = re.sub(r'\s+', ' ', quote.strip().lower())
        for unit in self.qa_units:
            clean_ans = re.sub(r'\s+', ' ', unit.answer_text.lower())
            if clean_quote in clean_ans:
                return unit
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "expert_id": self.expert_id,
            "expert_name": self.expert_name,
            "role": self.role,
            "market": self.market,
            "turn_count": len(self.turns),
            "qa_unit_count": len(self.qa_units),
            "turns": [t.to_dict() for t in self.turns],
            "qa_units": [u.to_dict() for u in self.qa_units]
        }


def _group_turns_into_qa_units(
    turns: List[DialogueTurn],
    expert_name: str,
    market: str
) -> List[QAUnit]:
    """
    Groups interviewer prompt turns with subsequent expert answer turns into semantic Q&A units.
    """
    qa_units: List[QAUnit] = []
    unit_counter = 1

    i = 0
    while i < len(turns):
        turn = turns[i]
        # Check if this turn is an interviewer question
        is_interviewer = (
            "interviewer" in turn.speaker.lower()
            or "q:" in turn.speaker.lower()
            or "moderator" in turn.speaker.lower()
        )

        if is_interviewer:
            q_turn = turn
            ans_turns: List[DialogueTurn] = []
            j = i + 1
            # Collect all subsequent expert turns until the next interviewer question
            while j < len(turns):
                next_turn = turns[j]
                next_is_interviewer = (
                    "interviewer" in next_turn.speaker.lower()
                    or "q:" in next_turn.speaker.lower()
                    or "moderator" in next_turn.speaker.lower()
                )
                if next_is_interviewer:
                    break
                ans_turns.append(next_turn)
                j += 1

            if ans_turns:
                ans_speaker = ans_turns[0].speaker
                combined_answer_text = " ".join(t.text for t in ans_turns).strip()
                ans_timestamps = [t.timestamp for t in ans_turns]

                qa_units.append(QAUnit(
                    unit_id=unit_counter,
                    expert_name=expert_name,
                    market=market,
                    question_timestamp=q_turn.timestamp,
                    question_speaker=q_turn.speaker,
                    question_text=q_turn.text,
                    answer_timestamps=ans_timestamps,
                    answer_speaker=ans_speaker,
                    answer_text=combined_answer_text,
                    answer_turns=ans_turns
                ))
                unit_counter += 1
            i = j
        else:
            # Standalone expert turn not preceded by interviewer
            i += 1

    return qa_units


def parse_transcript_text(raw_text: str, file_name: str = "transcript.txt") -> Transcript:
    """
    Parses raw text of an interview transcript.
    Extracts metadata from header and turns with timestamps and speakers.
    Groups turns into semantic Q&A units.
    """
    lines = raw_text.splitlines()

    expert_id = "Expert"
    expert_name = "Unknown Expert"
    role = "Unknown Role"
    market = "Unknown Market"

    # Header parsing
    content_start_line = 0
    for idx, line in enumerate(lines[:12]):
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

    # Group turns into Q&A units
    qa_units = _group_turns_into_qa_units(turns, expert_name, market)

    return Transcript(
        file_name=file_name,
        expert_id=expert_id,
        expert_name=expert_name,
        role=role,
        market=market,
        raw_text=raw_text,
        turns=turns,
        qa_units=qa_units
    )


def parse_interview_guide(guide_path: str) -> List[Dict[str, Any]]:
    """
    Parses Interview_Guide.txt dynamically into a structured list of question objects.
    Enables adding/editing guide questions without code changes.
    """
    if not os.path.exists(guide_path):
        return []

    with open(guide_path, "r", encoding="utf-8") as f:
        content = f.read()

    questions = []
    lines = content.splitlines()
    for line in lines:
        line_clean = line.strip()
        m = re.match(r"^(\d+)\.\s*(.+)", line_clean)
        if m:
            num = int(m.group(1))
            q_text = m.group(2).strip()
            questions.append({
                "id": f"Q{num}",
                "number": num,
                "question": q_text,
                "topic": _extract_topic_from_question(q_text)
            })
    return questions


def _extract_topic_from_question(q_text: str) -> str:
    """Helper to derive a short readable topic label from question text."""
    lower = q_text.lower()
    if "adoption" in lower and "current" in lower:
        return "Current Adoption"
    if "barrier" in lower:
        return "Main Barriers"
    if "budget" in lower or "roi" in lower:
        return "Budgets & ROI"
    if "training" in lower or "outcome" in lower:
        return "Training & Outcomes"
    if "trend" in lower or "3–5" in lower or "3-5" in lower:
        return "Future Outlook (3-5 Years)"
    if "timeline" in lower or "decision-making" in lower or "how long" in lower:
        return "Purchasing Timeline"
    return "Market Question"


def load_transcripts_from_dir(directory: str) -> Dict[str, Transcript]:
    """
    Zero-config auto-discovery: scans directory and any input folder (e.g. input_transcripts/)
    to load all transcript files. Adding a 4th, 5th, ... 30th transcript requires no code changes.
    """
    transcripts: Dict[str, Transcript] = {}
    candidate_paths: List[str] = []

    # 1. Check input folder if present
    input_folder = os.path.join(directory, "input_transcripts")
    if os.path.exists(input_folder) and os.path.isdir(input_folder):
        for fname in sorted(os.listdir(input_folder)):
            if fname.endswith(".txt") and not fname.startswith("."):
                candidate_paths.append(os.path.join(input_folder, fname))

    # 2. Check main directory
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt") and ("transcript" in fname.lower() or fname.startswith("Transcript_")):
            full_path = os.path.join(directory, fname)
            if full_path not in candidate_paths:
                candidate_paths.append(full_path)

    # 3. Parse candidates and index by basename
    for path in candidate_paths:
        fname = os.path.basename(path)
        if fname in transcripts:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Verify this is a transcript file (has Expert header or timestamps)
            if "Expert" in content[:300] or re.search(r"\d{1,2}:\d{2}", content[:500]):
                transcripts[fname] = parse_transcript_text(content, fname)
        except Exception as e:
            print(f"Error parsing transcript {path}: {e}")

    return transcripts
