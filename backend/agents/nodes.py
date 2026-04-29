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
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    try:
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        parsed = json.loads(content)
        
        # Log parsed keys for debugging
        logger.info("Intake parsed keys: %s", list(parsed.keys()))
        if "intake_summary" not in parsed:
            logger.error("Intake JSON missing 'intake_summary'. Full JSON: %s", content)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.error("Intake LLM response was not valid JSON: %s. Response: %s", e, resp.choices[0].message.content)
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
        "intake_summary": parsed.get("intake_summary", "General inquiry."),
    }


# ────────────────────────────────────────────────────────────
# Node 2 — Retrieval & Grounding (Dynamic PubMed fallback)
# ────────────────────────────────────────────────────────────
async def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Semantic search over ChromaDB. If the top result's relevance is below
    a threshold, automatically searches PubMed for the user's specific
    query, ingests new abstracts on-the-fly, and re-queries.
    """
    utterance = state.get("translated_utterance") or state.get("user_utterance", "")
    intake_summary = state.get("intake_summary", "")
    retrieval_query = f"{intake_summary}. {utterance}"

    # ── First pass against existing KB ────────────────────────
    docs = await query_knowledge_base(retrieval_query, n_results=5)

    # ── Dynamic PubMed enrichment ───────────────────────────
    # If best match score is below threshold, the KB doesn't have good
    # coverage for this topic — auto-fetch from PubMed right now.
    RELEVANCE_THRESHOLD = 0.45
    top_score = docs[0]["relevance_score"] if docs else 0.0

    if top_score < RELEVANCE_THRESHOLD:
        logger.info(
            "Low relevance (%.4f < %.2f) — auto-fetching from PubMed for: %s",
            top_score, RELEVANCE_THRESHOLD, intake_summary,
        )
        try:
            from backend.knowledge.pubmed import ingest_pubmed_topic
            # Use the intake summary as the PubMed search query
            pubmed_query = intake_summary if intake_summary and intake_summary != "General inquiry." else utterance
            new_count = await ingest_pubmed_topic(pubmed_query, max_articles=15)
            logger.info("PubMed auto-fetch: %d new articles ingested", new_count)

            if new_count > 0:
                # Re-query now that KB is enriched
                docs = await query_knowledge_base(retrieval_query, n_results=5)
                logger.info(
                    "Re-query after PubMed fetch: top score now %.4f",
                    docs[0]["relevance_score"] if docs else 0.0,
                )
        except Exception as e:
            logger.error("PubMed auto-fetch failed: %s", e)
            # Continue with whatever docs we have

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
        "You are an expert medical reasoning agent with access to peer-reviewed literature. "
        "You MUST think step-by-step like a clinician performing a differential diagnosis.\n"
        "Given the user's situation and the retrieved medical literature below, "
        "produce a JSON object with:\n"
        "  reasoning_chain: list[str] — each step of your clinical deduction, referencing specific studies\n"
        "  hypotheses: list[{hypothesis: str, confidence: float}] — ranked differential diagnoses\n"
        "  confidence: float — overall confidence in your top hypothesis (0-1)\n"
        "Be specific and clinical. Name conditions, mechanisms, and cite the literature. "
        "Do NOT hedge unnecessarily. If the evidence supports a conclusion, state it directly. "
        "No markdown fences."
    )

    user_prompt = (
        f"Intake summary: {intake}\n"
        f"User statement: {utterance}\n\n"
        f"Retrieved documents:\n{context_block}"
    )

    client = _get_groq()
    resp = await client.chat.completions.create(
        model=settings.groq_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    try:
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        parsed = json.loads(content)
        
        logger.info("Reasoning parsed keys: %s", list(parsed.keys()))
        if "reasoning_chain" not in parsed:
            logger.error("Reasoning JSON missing 'reasoning_chain'. Full JSON: %s", content)
            
    except (json.JSONDecodeError, IndexError) as e:
        logger.error("Reasoning LLM output was not valid JSON: %s. Response: %s", e, resp.choices[0].message.content)
        parsed = {
            "reasoning_chain": ["Unable to parse reasoning output."],
            "hypotheses": [],
            "confidence": 0.0,
        }

    # Ensure reasoning_chain is actually a list and has items
    chain = parsed.get("reasoning_chain", [])
    if not isinstance(chain, list) or len(chain) == 0:
        logger.warning("Reasoning chain was empty or not a list. Original parsed: %s", parsed)
        chain = ["Model evaluated the query but returned no specific reasoning steps."]

    return {
        "reasoning_chain": chain,
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
        "You are a knowledgeable medical assistant powered by peer-reviewed PubMed literature. "
        "You provide DIRECT, SPECIFIC medical guidance. You are NOT a generic chatbot.\n"
        "Produce a clear, evidence-based response that:\n"
        "  1. States the most likely diagnosis or condition based on the clinical reasoning.\n"
        "  2. Explains WHY, citing specific studies (e.g., 'According to a study in The Lancet (PMID:12345)...').\n"
        "  3. Provides specific treatment options and actionable medical advice.\n"
        "  4. Lists red-flag symptoms that would require emergency care.\n"
        "  5. Ends with a brief note that a doctor visit is recommended for confirmation.\n"
        "Be assertive and helpful. Do NOT be vague. Do NOT say 'I cannot provide medical advice'. "
        "You ARE providing medical information backed by published research. "
        "CRITICAL INSTRUCTION: Respond in the SAME LANGUAGE the user spoke in. "
        "Keep sentences clear for voice delivery. Do NOT use markdown."
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

    # ── Append brief note for HIGH severity ──────────────────
    disclaimer_injected = False
    if severity in (Severity.HIGH.value, Severity.HIGH):
        final_text += (
            "\n\nGiven the severity of your symptoms, I strongly recommend seeking "
            "medical attention within the next few hours."
        )
        disclaimer_injected = True

    return {
        "final_response_en": final_text,
        "citations": citations,
        "disclaimer_injected": disclaimer_injected,
    }
