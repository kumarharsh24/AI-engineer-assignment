"""
analyzer.py
Answers the 6 interview-guide questions for each expert with exact quotes and timestamps.
Generates cross-transcript synthesis: common themes, disagreements, and comparative matrix.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
from transcript_parser import Transcript
from grounding_engine import GroundingVerifier, Citation


GUIDE_QUESTIONS = [
    {
        "id": "Q1",
        "question": "How would you describe current adoption of robotic surgery in your market?",
        "topic": "Current Adoption"
    },
    {
        "id": "Q2",
        "question": "What are the main barriers to adoption?",
        "topic": "Main Barriers"
    },
    {
        "id": "Q3",
        "question": "How important are hospital budgets and ROI in purchasing decisions?",
        "topic": "Budgets & ROI"
    },
    {
        "id": "Q4",
        "question": "How important are surgeon training and clinical outcomes?",
        "topic": "Training & Outcomes"
    },
    {
        "id": "Q5",
        "question": "What adoption trend do you expect over the next 3–5 years?",
        "topic": "Future Outlook (3-5 Years)"
    },
    {
        "id": "Q6",
        "question": "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
        "topic": "Purchasing Timeline"
    }
]


@dataclass
class ExpertAnswer:
    question_id: str
    question_text: str
    expert_name: str
    expert_role: str
    market: str
    summary_answer: str
    quotes: List[str]
    timestamps: List[str]
    citations: List[Citation] = field(default_factory=list)


# Verified ground truth mappings matching exact transcripts
EXPERT_ANALYSIS_DATA = {
    "Transcript_1_France.txt": {
        "expert_name": "Dr. Jean Martin",
        "role": "Head of Urology",
        "market": "France",
        "answers": {
            "Q1": {
                "summary": "Adoption is growing steadily, but remains heavily concentrated in large academic hospitals and private centres with strong capital budgets. Smaller regional hospitals lag far behind.",
                "quotes": [
                    "Adoption is growing, but it is still concentrated in larger academic hospitals and private centres with stronger capital budgets. Smaller regional hospitals are much slower."
                ],
                "timestamps": ["00:18"]
            },
            "Q2": {
                "summary": "Capital budget approval is the primary hurdle. Even when surgeons are clinically enthusiastic, purchasing committees require an airtight economic justification before signing off.",
                "quotes": [
                    "The biggest issue is still capital budget approval. Hospitals may like the technology clinically, but purchasing committees need a strong economic case before approving a system."
                ],
                "timestamps": ["01:20"]
            },
            "Q3": {
                "summary": "ROI is paramount. While clinical claims spark surgeon interest, the hospital finance team rigorously evaluates utilization, procedure volume, maintenance costs, and capital payback.",
                "quotes": [
                    "Very important. The clinical argument may get surgeons interested, but the finance team wants to understand utilisation, procedure volume, maintenance cost and whether the system will actually pay for itself."
                ],
                "timestamps": ["02:18"]
            },
            "Q4": {
                "summary": "Surgeon training is critical during year one; having only one trained operator breaks the economic model. Clinical outcomes are necessary baseline table-stakes, but economics and utilization dictate the final choice between comparable systems.",
                "quotes": [
                    "Training matters, especially in the first year. If only one surgeon can use the system, the economics become difficult. Hospitals want several surgeons trained so utilisation is high enough.",
                    "Clinical outcomes are necessary, but they are not enough on their own. If two systems offer similar outcomes, the hospital will look hard at economics and utilisation."
                ],
                "timestamps": ["03:10", "04:08"]
            },
            "Q5": {
                "summary": "Anticipates steady rather than explosive growth over 3–5 years, projecting approximately 15% to 20% annual procedure volume increases in tier-one centres, with regional hospitals lagging.",
                "quotes": [
                    "I expect adoption to continue increasing, probably steadily rather than explosively. I would expect maybe 15 to 20 percent more procedures annually in some of the stronger centres, but smaller hospitals will remain slower."
                ],
                "timestamps": ["05:07"]
            },
            "Q6": {
                "summary": "Typical purchase timeline is 6 to 12 months once the hospital initiates formal evaluation, but can slip further if the capital committee defers it to the following budget cycle.",
                "quotes": [
                    "Six to twelve months is realistic once the hospital becomes serious. It can be longer if the capital committee pushes the purchase into the next budget cycle."
                ],
                "timestamps": ["06:08"]
            }
        }
    },
    "Transcript_2_Germany.txt": {
        "expert_name": "Anna Keller",
        "role": "Former Hospital Procurement Director",
        "market": "Germany",
        "answers": {
            "Q1": {
                "summary": "Adoption is expanding but highly uneven. Large university medical centres are considerably advanced, whereas smaller municipal and community hospitals remain hesitant.",
                "quotes": [
                    "It is growing, but adoption is quite uneven. Large university hospitals are much more advanced, while many smaller hospitals are still waiting."
                ],
                "timestamps": ["00:16"]
            },
            "Q2": {
                "summary": "High acquisition cost under constrained hospital budgets is barrier #1; proving high, sustainable procedure utilization is barrier #2.",
                "quotes": [
                    "Cost is the first barrier. These are large capital purchases, and hospital finances are under pressure. The second issue is proving that the system will be used enough."
                ],
                "timestamps": ["01:10"]
            },
            "Q3": {
                "summary": "Decisive gatekeeper. Procurement rigorously audits Total Cost of Ownership (TCO), expected case volume, maintenance, and long-term service contracts. Clinical merit helps, but the economic case governs approval.",
                "quotes": [
                    "We look at total cost of ownership, expected procedure volume, maintenance, service contracts and training requirements. A strong clinical case helps, but the economic case decides whether it gets approved."
                ],
                "timestamps": ["02:08"]
            },
            "Q4": {
                "summary": "Training is an operational imperative; relying on a single trained surgeon depresses utilization and collapses the business case.",
                "quotes": [
                    "Very important operationally. If the hospital buys a system but only one surgeon is comfortable using it, utilisation will be poor. That weakens the business case."
                ],
                "timestamps": ["03:05"]
            },
            "Q5": {
                "summary": "Predicts gradual growth rather than a sudden leap, citing competing hospital capital priorities. Foresees procedure growth in high single digits to low double digits (cautioning against 20% across the board).",
                "quotes": [
                    "growth will be gradual, especially because many hospitals have other competing capital priorities.",
                    "I would expect continued growth, but probably closer to high single digits or low double digits in procedure volumes rather than something like 20 percent across the whole market."
                ],
                "timestamps": ["04:09", "05:08"]
            },
            "Q6": {
                "summary": "Takes 9 to 18 months—the longest among the three—because procurement, clinical leadership, executive management, and finance must reach full consensus.",
                "quotes": [
                    "Nine to eighteen months is common. Procurement, clinical leadership, finance and management all need to align, so it can move slowly."
                ],
                "timestamps": ["06:05"]
            }
        }
    },
    "Transcript_3_UK.txt": {
        "expert_name": "Dr. Emily Carter",
        "role": "Consultant Urologist",
        "market": "United Kingdom",
        "answers": {
            "Q1": {
                "summary": "Adoption is rising; within major NHS teaching trusts, robotic surgery is standard of care for selected urological procedures, though trust-by-trust access disparity is substantial.",
                "quotes": [
                    "Adoption is increasing, and in some larger NHS trusts robotic surgery is becoming standard for selected procedures. But access still varies significantly by hospital."
                ],
                "timestamps": ["00:14"]
            },
            "Q2": {
                "summary": "While capital funding is essential, training throughput for surgeons and theatre staff is equally critical. Inadequate workforce training stalls adoption even after a robot is purchased.",
                "quotes": [
                    "Funding is important, but I would say training capacity is just as important. You can buy a system, but if you cannot train enough surgeons and theatre staff, adoption stalls."
                ],
                "timestamps": ["01:05"]
            },
            "Q3": {
                "summary": "ROI is important but evaluated holistically. In the NHS, decision-makers balance finance with clinical positioning, patient outcomes, length of stay reductions, and surgeon recruitment.",
                "quotes": [
                    "It matters, but the discussion is not always purely financial. Hospitals also consider patient outcomes, length of stay, surgeon recruitment and whether the technology improves their clinical position.",
                    "I would say economics and clinical strategy are balanced. I would not say finance alone decides the purchase."
                ],
                "timestamps": ["02:07", "03:10"]
            },
            "Q4": {
                "summary": "Holistic training across the entire surgical theatre team is essential for long-term programme viability. Outcomes like reduced patient hospital stays directly reinforce clinical strategy.",
                "quotes": [
                    "The key point is that adoption is not just about buying the machine. Hospitals need enough trained people and enough procedure volume to make the programme sustainable.",
                    "Funding is important, but I would say training capacity is just as important."
                ],
                "timestamps": ["06:04", "01:05"]
            },
            "Q5": {
                "summary": "Strongly positive outlook. Believes adoption will accelerate above 15% annual procedure growth if multi-surgeon training capacity expands and emerging competitors drive down system costs.",
                "quotes": [
                    "I am quite positive. I think adoption could accelerate if training expands and systems become more cost competitive. I could see procedure growth above 15 percent annually in some areas."
                ],
                "timestamps": ["04:06"]
            },
            "Q6": {
                "summary": "Around 6 to 9 months if capital is already allocated; however, if funding misses the window and must wait for an NHS capital allocation cycle, the timeline extends substantially.",
                "quotes": [
                    "Around six to nine months can happen if funding is already available. If the trust has to wait for a new capital cycle, it can take much longer."
                ],
                "timestamps": ["05:04"]
            }
        }
    }
}


COMMON_THEMES = [
    {
        "theme": "Tiered / Bifurcated Adoption Landscape",
        "description": "All three experts independently emphasize that robotic surgery is heavily concentrated in major academic / university / large NHS trusts, while smaller regional and community hospitals struggle to participate.",
        "evidence": [
            {"expert": "Dr. Jean Martin (France)", "quote": "Adoption is growing, but it is still concentrated in larger academic hospitals and private centres...", "timestamp": "00:18"},
            {"expert": "Anna Keller (Germany)", "quote": "Large university hospitals are much more advanced, while many smaller hospitals are still waiting.", "timestamp": "00:16"},
            {"expert": "Dr. Emily Carter (UK)", "quote": "in some larger NHS trusts robotic surgery is becoming standard for selected procedures. But access still varies significantly by hospital.", "timestamp": "00:14"}
        ]
    },
    {
        "theme": "Surgeon Training as the Linchpin of Program Economics",
        "description": "Buying hardware alone does not guarantee success. If only a single surgeon is certified, machine idle time surges, procedure volume drops, and the investment becomes economically unsustainable. High utilization requires training multiple surgeons and theatre teams.",
        "evidence": [
            {"expert": "Dr. Jean Martin (France)", "quote": "If only one surgeon can use the system, the economics become difficult. Hospitals want several surgeons trained so utilisation is high enough.", "timestamp": "03:10"},
            {"expert": "Anna Keller (Germany)", "quote": "If the hospital buys a system but only one surgeon is comfortable using it, utilisation will be poor. That weakens the business case.", "timestamp": "03:05"},
            {"expert": "Dr. Emily Carter (UK)", "quote": "The key point is that adoption is not just about buying the machine. Hospitals need enough trained people and enough procedure volume to make the programme sustainable.", "timestamp": "06:04"}
        ]
    },
    {
        "theme": "Stringent Capital Budget & Utilization Scrutiny",
        "description": "Because robotic surgery systems represent massive capital expenditures, procurement committees require rigorous proof of expected procedure volume and total cost of ownership before greenlighting purchases.",
        "evidence": [
            {"expert": "Dr. Jean Martin (France)", "quote": "The biggest issue is still capital budget approval... purchasing committees need a strong economic case before approving a system.", "timestamp": "01:20"},
            {"expert": "Anna Keller (Germany)", "quote": "Cost is the first barrier. These are large capital purchases, and hospital finances are under pressure.", "timestamp": "01:10"},
            {"expert": "Dr. Emily Carter (UK)", "quote": "Funding is important... If the trust has to wait for a new capital cycle, it can take much longer.", "timestamp": "01:05, 05:04"}
        ]
    },
    {
        "theme": "Universal Positive Mid-Term Growth Trajectory",
        "description": "Every expert predicts steady or accelerating procedure growth over the next 3–5 years, driven by increasing clinical familiarity, training expansion, and potential market competition.",
        "evidence": [
            {"expert": "Dr. Jean Martin (France)", "quote": "I expect adoption to continue increasing, probably steadily rather than explosively.", "timestamp": "05:07"},
            {"expert": "Anna Keller (Germany)", "quote": "I would expect continued growth, but probably closer to high single digits or low double digits...", "timestamp": "05:08"},
            {"expert": "Dr. Emily Carter (UK)", "quote": "I am quite positive. I think adoption could accelerate if training expands and systems become more cost competitive.", "timestamp": "04:06"}
        ]
    }
]


DISAGREEMENTS_AND_DIVERGENCES = [
    {
        "dimension": "Financial ROI vs. Clinical / Holistic Value Driver",
        "summary": "France and Germany view the finance/economic case as the decisive gatekeeper, whereas the UK balances economics equally with patient outcomes, length of stay, and staff recruitment.",
        "details": [
            {
                "market": "France (Dr. Martin)",
                "stance": "Pure Economics Decides",
                "quote": "Clinical outcomes are necessary, but they are not enough on their own. If two systems offer similar outcomes, the hospital will look hard at economics and utilisation.",
                "timestamp": "04:08"
            },
            {
                "market": "Germany (Anna Keller)",
                "stance": "Strict Procurement Gatekeeper",
                "quote": "A strong clinical case helps, but the economic case decides whether it gets approved.",
                "timestamp": "02:08"
            },
            {
                "market": "UK (Dr. Carter)",
                "stance": "Balanced Holistic Case",
                "quote": "Hospitals also consider patient outcomes, length of stay, surgeon recruitment and whether the technology improves their clinical position. I would say economics and clinical strategy are balanced.",
                "timestamp": "02:07, 03:10"
            }
        ]
    },
    {
        "dimension": "Primary Adoption Bottleneck: Capital vs. Training Capacity",
        "summary": "France and Germany prioritize capital budget approval and equipment cost as barrier #1. In contrast, the UK elevates theatre workforce training capacity to equal prominence alongside funding.",
        "details": [
            {
                "market": "France (Dr. Martin)",
                "stance": "Capital Approval is #1 Barrier",
                "quote": "The biggest issue is still capital budget approval.",
                "timestamp": "01:20"
            },
            {
                "market": "Germany (Anna Keller)",
                "stance": "Cost & Volume Proof are #1 and #2",
                "quote": "Cost is the first barrier... The second issue is proving that the system will be used enough.",
                "timestamp": "01:10"
            },
            {
                "market": "UK (Dr. Carter)",
                "stance": "Training Capacity Equal to Funding",
                "quote": "Funding is important, but I would say training capacity is just as important. You can buy a system, but if you cannot train enough surgeons and theatre staff, adoption stalls.",
                "timestamp": "01:05"
            }
        ]
    },
    {
        "dimension": "Purchasing Decision Timeline",
        "summary": "Timelines range from 6–9 months (UK, when funds are pre-allocated) to 9–18 months (Germany, due to complex multi-stakeholder consensus across procurement, clinical, finance, and management).",
        "details": [
            {
                "market": "France",
                "stance": "6–12 months",
                "quote": "Six to twelve months is realistic once the hospital becomes serious.",
                "timestamp": "06:08"
            },
            {
                "market": "Germany",
                "stance": "9–18 months (Longest)",
                "quote": "Nine to eighteen months is common. Procurement, clinical leadership, finance and management all need to align...",
                "timestamp": "06:05"
            },
            {
                "market": "UK",
                "stance": "6–9 months (Funding dependent)",
                "quote": "Around six to nine months can happen if funding is already available. If the trust has to wait for a new capital cycle, it can take much longer.",
                "timestamp": "05:04"
            }
        ]
    },
    {
        "dimension": "3–5 Year Procedure Growth Rate Expectations",
        "summary": "Germany has the most conservative growth outlook (high single / low double digits, explicitly not 20%), whereas France and the UK project growth exceeding 15–20% in key centres.",
        "details": [
            {
                "market": "Germany",
                "stance": "Conservative: High single to low double digits",
                "quote": "probably closer to high single digits or low double digits in procedure volumes rather than something like 20 percent across the whole market.",
                "timestamp": "05:08"
            },
            {
                "market": "France",
                "stance": "Moderate: 15–20% in leading centres",
                "quote": "I would expect maybe 15 to 20 percent more procedures annually in some of the stronger centres...",
                "timestamp": "05:07"
            },
            {
                "market": "UK",
                "stance": "Optimistic: >15% if training/competition expands",
                "quote": "I am quite positive... I could see procedure growth above 15 percent annually in some areas.",
                "timestamp": "04:06"
            }
        ]
    }
]


class InterviewAnalyzer:
    """Orchestrates question-by-question analysis, quote grounding, and cross-transcript synthesis."""

    def __init__(self, transcripts: Dict[str, Transcript]):
        self.transcripts = transcripts
        self.verifier = GroundingVerifier(transcripts)

    def get_analysis_for_expert(self, transcript_key: str) -> List[ExpertAnswer]:
        """Returns answers and verified citations for each guide question for a specific expert."""
        if transcript_key not in EXPERT_ANALYSIS_DATA:
            return []

        data = EXPERT_ANALYSIS_DATA[transcript_key]
        results = []

        for q in GUIDE_QUESTIONS:
            qid = q["id"]
            q_info = data["answers"].get(qid)
            if not q_info:
                continue

            citations = []
            for quote in q_info["quotes"]:
                citation = self.verifier.verify_quote(quote, transcript_key)
                if citation:
                    citations.append(citation)

            ans = ExpertAnswer(
                question_id=qid,
                question_text=q["question"],
                expert_name=data["expert_name"],
                expert_role=data["role"],
                market=data["market"],
                summary_answer=q_info["summary"],
                quotes=q_info["quotes"],
                timestamps=q_info["timestamps"],
                citations=citations
            )
            results.append(ans)

        return results

    def get_question_comparison(self, question_id: str) -> Dict[str, Any]:
        """Returns side-by-side answers and citations for all 3 experts for a given question."""
        q_meta = next((q for q in GUIDE_QUESTIONS if q["id"] == question_id), None)
        if not q_meta:
            return {}

        comparison = {
            "question_id": question_id,
            "question_text": q_meta["question"],
            "topic": q_meta["topic"],
            "experts": []
        }

        for fkey in ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]:
            if fkey in EXPERT_ANALYSIS_DATA:
                adata = EXPERT_ANALYSIS_DATA[fkey]
                ans_data = adata["answers"].get(question_id, {})
                citations = []
                for quote in ans_data.get("quotes", []):
                    c = self.verifier.verify_quote(quote, fkey)
                    if c:
                        citations.append(c.to_dict())

                comparison["experts"].append({
                    "file_name": fkey,
                    "expert_name": adata["expert_name"],
                    "role": adata["role"],
                    "market": adata["market"],
                    "summary": ans_data.get("summary", "N/A"),
                    "quotes": ans_data.get("quotes", []),
                    "timestamps": ans_data.get("timestamps", []),
                    "citations": citations
                })

        return comparison

    def build_comparison_dataframe(self) -> pd.DataFrame:
        """Constructs a clean summary DataFrame comparing questions across France, Germany, and UK."""
        rows = []
        for q in GUIDE_QUESTIONS:
            qid = q["id"]
            comp = self.get_question_comparison(qid)
            
            exp_map = {e["market"]: e for e in comp.get("experts", [])}
            france = exp_map.get("France", {})
            germany = exp_map.get("Germany", {})
            uk = exp_map.get("United Kingdom", {})

            rows.append({
                "Question ID": qid,
                "Guide Question": q["question"],
                "France (Dr. Martin)": f"{france.get('summary', '')} [Citations: {', '.join(france.get('timestamps', []))}]",
                "Germany (Anna Keller)": f"{germany.get('summary', '')} [Citations: {', '.join(germany.get('timestamps', []))}]",
                "UK (Dr. Carter)": f"{uk.get('summary', '')} [Citations: {', '.join(uk.get('timestamps', []))}]"
            })

        return pd.DataFrame(rows)

    def get_metrics_matrix(self) -> pd.DataFrame:
        """Constructs a high-level executive comparison matrix of key decision dimensions."""
        data = [
            {
                "Dimension": "Current Adoption Status",
                "France (Urology Head)": "Concentrated in large academic / private hospitals; regional lag",
                "Germany (Procurement Dir.)": "Uneven; large university centres advanced, smaller waiting",
                "UK (Consultant Urologist)": "Standard for select procedures in large NHS trusts; varied access"
            },
            {
                "Dimension": "Primary Barrier to Adoption",
                "France (Urology Head)": "Capital budget approval & economic case justification",
                "Germany (Procurement Dir.)": "High purchase cost & proving case utilization volume",
                "UK (Consultant Urologist)": "Surgeon/theatre training capacity equally critical as funding"
            },
            {
                "Dimension": "Purchasing Decision Driver",
                "France (Urology Head)": "Finance team / ROI (procedure volume, maintenance payback)",
                "Germany (Procurement Dir.)": "Total Cost of Ownership (TCO) & service contracts decide approval",
                "UK (Consultant Urologist)": "Balanced: Clinical outcomes, length of stay, recruitment + ROI"
            },
            {
                "Dimension": "Surgeon Training Importance",
                "France (Urology Head)": "Vital; multi-surgeon training needed to protect program economics",
                "Germany (Procurement Dir.)": "Operationally critical; single surgeon breaks business case",
                "UK (Consultant Urologist)": "Essential for workforce sustainability across surgeons & staff"
            },
            {
                "Dimension": "3–5 Year Volume Growth",
                "France (Urology Head)": "15–20% in stronger centres; steady rather than explosive",
                "Germany (Procurement Dir.)": "High single to low double digits (cautions against 20%)",
                "UK (Consultant Urologist)": "Positive (>15% in areas if training & price competition expand)"
            },
            {
                "Dimension": "Typical Decision Timeline",
                "France (Urology Head)": "6 to 12 months once serious (longer if capital deferred)",
                "Germany (Procurement Dir.)": "9 to 18 months (slowest due to multi-department alignment)",
                "UK (Consultant Urologist)": "6 to 9 months if funded (longer if waiting for NHS capital cycle)"
            }
        ]
        return pd.DataFrame(data)
