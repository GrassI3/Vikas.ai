"""
Vikas.ai — LangGraph Agent Nodes
Each function is a discrete node in the reasoning graph.
Nodes read from and mutate the shared AgentState.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groq import AsyncGroq

from backend.agents.state import AgentState, Domain, RetrievedDocument, Severity
from backend.config import get_settings
from backend.knowledge.vector_db import query_knowledge_base
from backend.utils.guardrails import check_emergency, build_disclaimer

logger = logging.getLogger("vikas.agents.nodes")
settings = get_settings()

_groq_client: AsyncGroq | None = None


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# ────────────────────────────────────────────────────────────
# Node 1 — Intake & Triage
# ────────────────────────────────────────────────────────────
async def intake_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Analyse the user utterance to determine:
      • Domain (medical, mental health, civic, accessibility, general)
      • Severity (low → emergency)
      • Sentiment score
      • A short clinical summary
    """
    utterance = state.get("translated_utterance") or state.get("user_utterance", "")

    # ── Fast-path emergency keyword scan ─────────────────────
    is_emergency, matched = check_emergency(utterance)
    if is_emergency:
        logger.warning("Emergency keywords detected: %s", matched)
        return {
            "severity": Severity.EMERGENCY.value,
            "domain": Domain.MENTAL_HEALTH.value if any(
                w in matched for w in ("suicide", "kill myself", "want to die", "end my life")
            ) else Domain.MEDICAL.value,
            "sentiment_score": -1.0,
            "intake_summary": f"EMERGENCY — matched keywords: {', '.join(matched)}",
        }

    # ── LLM-based triage ────────────────────────────────────
    client = _get_groq()
    system_prompt = (
        "You are a medical-triage intake agent. The user's message may be in a regional language. "
        "Given the user's message, return ONLY a JSON object with keys: "
        "domain (medical|mental_health|civic|accessibility|general), "
        "severity (low|medium|high|emergency), sentiment_score (float -1 to 1), "
        "and intake_summary (one sentence in English). No markdown fences."
    )

    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    try:
        parsed = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        logger.error("Intake LLM response was not valid JSON — falling back to defaults")
        parsed = {
            "domain": Domain.GENERAL.value,
            "severity": Severity.LOW.value,
            "sentiment_score": 0.0,
            "intake_summary": "Unable to parse triage — proceeding with general guidance.",
        }

    return {
        "severity": parsed.get("severity", Severity.LOW.value),
        "domain": parsed.get("domain", Domain.GENERAL.value),
        "sentiment_score": parsed.get("sentiment_score", 0.0),
        "intake_summary": parsed.get("intake_summary", ""),
    }


# ────────────────────────────────────────────────────────────
# Node 2 — Retrieval & Grounding
# ────────────────────────────────────────────────────────────
async def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Build a targeted search query from the intake summary,
    then retrieve the top-k most relevant documents from ChromaDB.
    """
    utterance = state.get("translated_utterance") or state.get("user_utterance", "")
    intake_summary = state.get("intake_summary", "")
    retrieval_query = f"{intake_summary}. User said: {utterance}"

    docs = await query_knowledge_base(retrieval_query, n_results=5)

    retrieved = [
        {
            "content": d["content"],
            "source": d["source"],
            "relevance_score": d["relevance_score"],
            "metadata": d.get("metadata", {}),
        }
        for d in docs
    ]

    return {
        "retrieval_query": retrieval_query,
        "retrieved_documents": retrieved,
    }


# ────────────────────────────────────────────────────────────
# Node 3 — Reasoning (Chain-of-Thought)
# ────────────────────────────────────────────────────────────
async def reasoning_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Apply Chain-of-Thought reasoning over the retrieved documents
    to produce ranked hypotheses with confidence scores.
    """
    utterance = state.get("translated_utterance") or state.get("user_utterance", "")
    docs = state.get("retrieved_documents", [])
    intake = state.get("intake_summary", "")

    context_block = "\n\n".join(
        f"[Source: {d.get('source', 'unknown')}]\n{d.get('content', '')}"
        for d in docs
    )

    system_prompt = (
        "You are a medical reasoning agent. You MUST think step-by-step.\n"
        "Given the user's situation and the retrieved medical literature below, "
        "produce a JSON object with:\n"
        "  reasoning_chain: list[str] — each step of your logical deduction\n"
        "  hypotheses: list[{hypothesis: str, confidence: float}] — ranked possibilities\n"
        "  confidence: float — overall confidence in your top hypothesis (0-1)\n"
        "Use ONLY information from the provided documents. If the documents lack coverage, "
        "state so explicitly. No markdown fences."
    )

    user_prompt = (
        f"Intake summary: {intake}\n"
        f"User statement: {utterance}\n\n"
        f"Retrieved documents:\n{context_block}"
    )

    client = _get_groq()
    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    try:
        parsed = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        logger.error("Reasoning LLM output was not valid JSON")
        parsed = {
            "reasoning_chain": ["Unable to parse reasoning output."],
            "hypotheses": [],
            "confidence": 0.0,
        }

    return {
        "reasoning_chain": parsed.get("reasoning_chain", []),
        "hypotheses": parsed.get("hypotheses", []),
        "confidence": parsed.get("confidence", 0.0),
    }


# ────────────────────────────────────────────────────────────
# Node 4 — Synthesis & Citation
# ────────────────────────────────────────────────────────────
async def synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Compose the final human-readable response:
      • Weave in citations from source documents
      • Inject safety disclaimers when severity is HIGH or EMERGENCY
      • Format for voice delivery (short sentences, clear structure)
    """
    severity = state.get("severity", Severity.LOW.value)
    reasoning = state.get("reasoning_chain", [])
    hypotheses = state.get("hypotheses", [])
    docs = state.get("retrieved_documents", [])
    utterance = state.get("translated_utterance") or state.get("user_utterance", "")

    # ── Build citation list ──────────────────────────────────
    citations = list({d.get("source", "unknown") for d in docs if d.get("source")})

    # ── Emergency override ───────────────────────────────────
    if severity in (Severity.EMERGENCY.value, Severity.EMERGENCY):
        disclaimer = build_disclaimer(state.get("domain", Domain.GENERAL.value))
        return {
            "final_response_en": disclaimer,
            "citations": [],
            "disclaimer_injected": True,
        }

    # ── Normal synthesis via LLM ─────────────────────────────
    reasoning_text = "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(reasoning))
    hyp_text = "\n".join(
        f"  • {h.get('hypothesis', '?')} (confidence {h.get('confidence', 0):.0%})"
        for h in hypotheses
    )

    system_prompt = (
        "You are a compassionate health-information synthesis agent speaking to a patient "
        "over the phone. Produce a clear, concise spoken response that:\n"
        "  1. Summarises the most likely situation in plain language.\n"
        "  2. Cites the source of each key fact (e.g., 'According to NIH guidelines…').\n"
        "  3. Provides 1-3 actionable next steps.\n"
        "  4. Ends with a standard reminder that this is automated information, not a "
        "licensed medical opinion.\n"
        "CRITICAL INSTRUCTION: You MUST produce your final response in the exact SAME LANGUAGE that the user spoke in. "
        "Do not respond in English unless the user spoke in English. Keep sentences short for voice delivery. Do NOT use markdown."
    )

    user_prompt = (
        f"User concern: {utterance}\n\n"
        f"Reasoning chain:\n{reasoning_text}\n\n"
        f"Hypotheses:\n{hyp_text}\n\n"
        f"Source documents: {', '.join(citations)}"
    )

    client = _get_groq()
    resp = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=512,
    )

    final_text = resp.choices[0].message.content.strip()

    # ── Append standard disclaimer for HIGH severity ─────────
    disclaimer_injected = False
    if severity in (Severity.HIGH.value, Severity.HIGH):
        final_text += (
            "\n\nImportant: I am an automated informational service and not a medical "
            "professional. If your symptoms worsen, please seek immediate medical attention."
        )
        disclaimer_injected = True

    return {
        "final_response_en": final_text,
        "citations": citations,
        "disclaimer_injected": disclaimer_injected,
    }
