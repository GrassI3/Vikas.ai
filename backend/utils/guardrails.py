"""
Vikas.ai — Safety Guardrails & Emergency Disclaimers
Deterministic safety checks that operate independently of the LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.config import get_settings

logger = logging.getLogger("vikas.utils.guardrails")
settings = get_settings()


# ── Emergency Contact Directory ─────────────────────────────
EMERGENCY_CONTACTS: dict[str, list[dict[str, str]]] = {
    "medical": [
        {"name": "National Emergency Number (India)", "number": "112"},
        {"name": "Ambulance", "number": "108"},
        {"name": "National Health Helpline", "number": "1800-180-1104"},
    ],
    "mental_health": [
        {"name": "iCall Psychosocial Helpline", "number": "9152987821"},
        {"name": "Vandrevala Foundation 24x7", "number": "1860-2662-345"},
        {"name": "NIMHANS Helpline", "number": "080-46110007"},
        {"name": "National Emergency Number (India)", "number": "112"},
    ],
    "civic": [
        {"name": "National Emergency Number (India)", "number": "112"},
        {"name": "Women Helpline", "number": "1091"},
        {"name": "Child Helpline", "number": "1098"},
    ],
    "general": [
        {"name": "National Emergency Number (India)", "number": "112"},
    ],
}


def check_emergency(text: str) -> tuple[bool, list[str]]:
    """
    Scan text for emergency keywords using exact and pattern matching.

    Returns:
        (is_emergency, list_of_matched_keywords)
    """
    text_lower = text.lower()
    matched: list[str] = []

    for keyword in settings.emergency_keywords:
        # Use word-boundary matching to avoid false positives
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(keyword)

    is_emergency = len(matched) > 0
    if is_emergency:
        logger.warning("Emergency keywords detected: %s", matched)

    return is_emergency, matched


def build_disclaimer(domain: str = "general") -> str:
    """
    Construct a hard-coded safety disclaimer with emergency contacts.

    This response completely overrides the LLM output when an emergency
    is detected, ensuring that critical information is never hallucinated.
    """
    contacts = EMERGENCY_CONTACTS.get(domain, EMERGENCY_CONTACTS["general"])
    contact_lines = "\n".join(
        f"  • {c['name']}: {c['number']}" for c in contacts
    )

    disclaimer = (
        "I want to be very clear with you. Based on what you have described, "
        "this sounds like it could be a serious situation that needs immediate "
        "professional attention.\n\n"
        "IMPORTANT: I am an automated informational service. I am NOT a doctor, "
        "therapist, or emergency responder. I cannot provide a medical diagnosis "
        "or treatment.\n\n"
        "Please contact one of the following services right away:\n"
        f"{contact_lines}\n\n"
        "If you or someone near you is in immediate danger, please call 112 now. "
        "A trained professional can provide the help you need. You are not alone."
    )

    logger.info("Emergency disclaimer generated for domain: %s", domain)
    return disclaimer


def validate_output(response: str) -> tuple[bool, str]:
    """
    Post-generation guardrail: lightweight safety net.

    Only catches genuinely dangerous outputs like specific drug
    prescriptions with dosages. We intentionally allow the LLM to
    provide medical information, differential diagnoses, and direct
    guidance — this is the entire point of the system.

    Returns:
        (is_safe, sanitized_response)
    """
    # Only flag specific prescription dosages — everything else is allowed
    prohibited_patterns = [
        (r"\btake\s+\d+\s*(mg|ml|grams?)\s+of\s+\w+\s*(every|daily|twice|three times)\b",
         "consult a doctor for the exact dosage of this medication"),
    ]

    sanitized = response
    is_safe = True

    for pattern, replacement in prohibited_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning("Guardrail triggered — prescription pattern detected")
            is_safe = False
            sanitized = re.sub(
                pattern,
                f"[{replacement}]",
                sanitized,
                flags=re.IGNORECASE,
            )

    return is_safe, sanitized
