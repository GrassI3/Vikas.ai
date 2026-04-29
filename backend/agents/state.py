"""
Vikas.ai — Agent State Schema
Defines the shared, typed state that flows through every node in the LangGraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Call severity levels detected by the Intake agent."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class Domain(str, Enum):
    """Detected domain of the user's query."""
    MEDICAL = "medical"
    MENTAL_HEALTH = "mental_health"
    CIVIC = "civic"
    ACCESSIBILITY = "accessibility"
    GENERAL = "general"


@dataclass
class RetrievedDocument:
    """A single document chunk retrieved from the vector database."""
    content: str
    source: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """
    The mutable state object passed between all LangGraph nodes.

    Every node reads from and writes to this state, ensuring full
    traceability and explainability of the reasoning chain.
    """

    # ── User Input ───────────────────────────────────────────
    user_utterance: str = ""
    detected_language: str = "en"
    translated_utterance: str = ""          # English translation of user input

    # ── Intake / Triage ──────────────────────────────────────
    severity: Severity = Severity.LOW
    domain: Domain = Domain.GENERAL
    sentiment_score: float = 0.0            # -1.0 (very negative) to 1.0 (very positive)
    intake_summary: str = ""

    # ── Retrieval ────────────────────────────────────────────
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    retrieval_query: str = ""

    # ── Reasoning (Chain-of-Thought) ─────────────────────────
    reasoning_chain: list[str] = field(default_factory=list)   # Ordered list of logical steps
    confidence: float = 0.0
    hypotheses: list[dict[str, Any]] = field(default_factory=list)

    # ── Synthesis ────────────────────────────────────────────
    final_response_en: str = ""             # English response with citations
    final_response_localized: str = ""      # Translated back to user's language
    citations: list[str] = field(default_factory=list)
    disclaimer_injected: bool = False

    # ── Conversation Memory ──────────────────────────────────
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0

    # ── Metadata ─────────────────────────────────────────────
    session_id: str = ""
    error: str | None = None
