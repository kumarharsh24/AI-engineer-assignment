"""
test_app.py
Automated test suite for the Hasamex AI Engineer transcript analysis platform.
Verifies parsing, zero-hallucination quote matching, timestamp citations, and QA retrieval.
"""

import os
import pytest
from transcript_parser import (
    load_transcripts_from_dir,
    parse_transcript_text,
    Transcript,
    DialogueTurn
)
from grounding_engine import GroundingVerifier, Citation
from analyzer import (
    InterviewAnalyzer,
    GUIDE_QUESTIONS,
    COMMON_THEMES,
    DISAGREEMENTS_AND_DIVERGENCES,
    EXPERT_ANALYSIS_DATA
)
from qa_engine import CrossTranscriptQAEngine, SAMPLE_QUESTIONS


WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def transcripts():
    return load_transcripts_from_dir(WORKSPACE_DIR)


@pytest.fixture
def analyzer(transcripts):
    return InterviewAnalyzer(transcripts)


@pytest.fixture
def qa_engine(transcripts):
    return CrossTranscriptQAEngine(transcripts)


# ==========================================
# TEST 1: TRANSCRIPT PARSER
# ==========================================
def test_transcript_loading(transcripts):
    """Verify that all 3 case study transcripts are loaded and parsed."""
    assert len(transcripts) == 3
    assert "Transcript_1_France.txt" in transcripts
    assert "Transcript_2_Germany.txt" in transcripts
    assert "Transcript_3_UK.txt" in transcripts


def test_transcript_metadata(transcripts):
    """Verify expert names, roles, and markets are accurately extracted."""
    t1 = transcripts["Transcript_1_France.txt"]
    assert "Martin" in t1.expert_name
    assert "Urology" in t1.role
    assert t1.market == "France"
    assert len(t1.turns) > 5

    t2 = transcripts["Transcript_2_Germany.txt"]
    assert "Keller" in t2.expert_name
    assert "Procurement" in t2.role
    assert t2.market == "Germany"
    assert len(t2.turns) > 5

    t3 = transcripts["Transcript_3_UK.txt"]
    assert "Carter" in t3.expert_name
    assert "Urologist" in t3.role
    assert "Kingdom" in t3.market or "UK" in t3.market
    assert len(t3.turns) > 5


def test_turn_timestamps_and_offsets(transcripts):
    """Verify that turns have valid timestamps and text."""
    for fkey, t in transcripts.items():
        for turn in t.turns:
            assert turn.timestamp
            assert ":" in turn.timestamp
            assert turn.speaker
            assert len(turn.text) > 0
            assert turn.end_char > turn.start_char


# ==========================================
# TEST 2: GROUNDING & VERBATIM QUOTE MATCHER
# ==========================================
def test_quote_verification_exact(transcripts):
    """Verify that genuine quotes match 100% verbatim with correct timestamp."""
    verifier = GroundingVerifier(transcripts)

    # France quote
    q1 = "Adoption is growing, but it is still concentrated in larger academic hospitals and private centres"
    cit1 = verifier.verify_quote(q1, "Transcript_1_France.txt")
    assert cit1 is not None
    assert cit1.is_verbatim
    assert cit1.timestamp == "00:18"
    assert cit1.speaker == "Dr. Martin"

    # Germany quote
    q2 = "Cost is the first barrier. These are large capital purchases"
    cit2 = verifier.verify_quote(q2, "Transcript_2_Germany.txt")
    assert cit2 is not None
    assert cit2.is_verbatim
    assert cit2.timestamp == "01:10"
    assert cit2.speaker == "Anna Keller"

    # UK quote
    q3 = "Funding is important, but I would say training capacity is just as important"
    cit3 = verifier.verify_quote(q3, "Transcript_3_UK.txt")
    assert cit3 is not None
    assert cit3.is_verbatim
    assert cit3.timestamp == "01:05"
    assert cit3.speaker == "Dr. Carter"


def test_quote_verification_rejects_hallucination(transcripts):
    """Verify that non-existent / fabricated quotes are rejected."""
    verifier = GroundingVerifier(transcripts)
    fake_quote = "Robotic surgery in France is completely funded by private insurance companies with zero delays."
    cit = verifier.verify_quote(fake_quote, "Transcript_1_France.txt")
    assert cit is None


# ==========================================
# TEST 3: INTERVIEW GUIDE ANALYZER
# ==========================================
def test_guide_questions_count():
    """Verify all 6 interview guide questions are defined."""
    assert len(GUIDE_QUESTIONS) == 6


def test_expert_answers_and_citations(analyzer):
    """Verify that all 6 questions have answers and verified citations for each expert."""
    for fkey in ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]:
        answers = analyzer.get_analysis_for_expert(fkey)
        assert len(answers) == 6
        for ans in answers:
            assert ans.summary_answer
            assert len(ans.quotes) > 0
            assert len(ans.timestamps) > 0
            assert len(ans.citations) > 0
            # Ensure every citation is verified
            for cit in ans.citations:
                assert cit.is_verbatim
                assert cit.confidence >= 0.99


def test_comparative_matrix_dataframes(analyzer):
    """Verify that comparative and executive dataframes are populated."""
    df_comp = analyzer.build_comparison_dataframe()
    assert len(df_comp) == 6
    assert "France (Dr. Martin)" in df_comp.columns
    assert "Germany (Anna Keller)" in df_comp.columns
    assert "UK (Dr. Carter)" in df_comp.columns

    df_metrics = analyzer.get_metrics_matrix()
    assert len(df_metrics) == 6
    assert "Dimension" in df_metrics.columns


# ==========================================
# TEST 4: CROSS-TRANSCRIPT QA ENGINE
# ==========================================
def test_qa_search_relevant_results(qa_engine):
    """Verify that BM25 search returns relevant turns."""
    results = qa_engine.search("purchasing timeline", top_k=5)
    assert len(results) > 0
    # Must contain timeline mentions
    found_timeline_mention = any(
        "month" in r.turn.text.lower() or "timeline" in r.matched_snippet.lower()
        for r in results
    )
    assert found_timeline_mention


def test_qa_offline_answers(qa_engine):
    """Verify offline answer synthesis runs cleanly and produces grounded responses."""
    for q in SAMPLE_QUESTIONS[:3]:
        resp = qa_engine.answer_query_offline(q)
        assert resp.grounded
        assert resp.confidence_score > 0.4
        assert len(resp.citations) > 0
        assert len(resp.answer_text) > 50


def test_qa_irrelevant_query_handling(qa_engine):
    """Verify that unrelated out-of-domain queries do not hallucinate answers."""
    resp = qa_engine.answer_query_offline("What is the average rainfall in the Sahara desert during July?")
    assert not resp.grounded
    assert "No sufficiently relevant statements" in resp.answer_text


# ==========================================
# TEST 5: ENVIRONMENT CONFIGURATION
# ==========================================
def test_env_configuration():
    """Verify that .env.example exists with expected configuration keys."""
    env_example_path = os.path.join(WORKSPACE_DIR, ".env.example")
    assert os.path.exists(env_example_path)
    with open(env_example_path, "r") as f:
        content = f.read()
    assert "OPENAI_API_KEY" in content
    assert "OPENAI_MODEL" in content

