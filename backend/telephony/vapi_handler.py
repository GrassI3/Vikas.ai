"""
Vikas.ai — Vapi AI Telephony Handler
Processes incoming Vapi webhook events and manages the voice call lifecycle.

Vapi sends POST requests to our server URL with events like:
  • assistant-request  — ask us which assistant config to use
  • function-call      — the assistant wants to invoke a tool
  • end-of-call-report — call has ended, includes transcript + analytics
  • hang               — user or system hung up
  • speech-update      — real-time transcript updates

Reference: https://docs.vapi.ai/server-url
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agents.graph import run_pipeline
from backend.config import get_settings

logger = logging.getLogger("vikas.telephony.vapi")
settings = get_settings()

# ── In-memory session store (swap for Redis in production) ──
_sessions: dict[str, dict[str, Any]] = {}


def _get_session(call_id: str) -> dict[str, Any]:
    """Retrieve or initialize a session for a given Vapi call ID."""
    if call_id not in _sessions:
        _sessions[call_id] = {
            "session_id": call_id,
            "conversation_history": [],
            "turn_count": 0,
        }
    return _sessions[call_id]


async def handle_assistant_request(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Respond to Vapi's 'assistant-request' event with our assistant configuration.
    This defines the system prompt, voice, model, and tool definitions.
    """
    return {
        "assistant": {
            "transcriber": {
                "provider": "deepgram",
                "model": "nova-2",
                "language": "hi", # Deepgram defaults, can be switched based on region or multi
            },
            "model": {
                "provider": "groq",
                "model": settings.groq_model,
                "temperature": 0.3,
                "systemPrompt": (
                    "You are Vikas, a compassionate health and civic information assistant. "
                    "You help users who may have limited access to healthcare or "
                    "government services. You speak clearly and simply. You ALWAYS cite your "
                    "sources. You NEVER provide a medical diagnosis — you provide information "
                    "and recommend consulting a doctor. If the user is in a medical emergency, "
                    "immediately advise them to call emergency services. "
                    "When the user describes symptoms, use the 'process_query' function to "
                    "search verified medical databases and reason through the situation. "
                    "CRITICAL: The user will speak to you in their native language (e.g. Hindi, Tamil, English). "
                    "You must process their query and ALWAYS respond directly in their fluent, natural-sounding native language."
                ),
            },
            "voice": {
                "provider": "azure",
                "voiceId": "hi-IN-SwaraNeural",  # High quality Azure voice
            },
            "firstMessage": (
                "Namaste! I am Vikas, your health and information assistant. "
                "I can help you understand medical symptoms or find government services. "
                "Please describe your situation, and I will do my best to help."
            ),
            "endCallMessage": (
                "Thank you for calling. Please remember, I am an automated service. "
                "Always consult a qualified professional for important health decisions. "
                "Take care!"
            ),
            "serverUrl": None,  # Vapi will use the URL it already has
            "functions": [
                {
                    "name": "process_query",
                    "description": (
                        "Analyse the user's health concern or civic query using verified "
                        "medical databases and multi-step reasoning. Returns an evidence-based, "
                        "cited response."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_message": {
                                "type": "string",
                                "description": "The user's spoken message describing their situation.",
                            },
                        },
                        "required": ["user_message"],
                    },
                },
            ],
        }
    }


async def handle_function_call(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Handle a function call from the Vapi assistant.
    This is where the LangGraph reasoning pipeline is invoked.
    """
    message = payload.get("message", {})
    function_call = message.get("functionCall", {})
    func_name = function_call.get("name", "")
    params = function_call.get("parameters", {})
    call_id = payload.get("call", {}).get("id", "unknown")

    if func_name != "process_query":
        logger.warning("Unknown function call: %s", func_name)
        return {"result": "I'm sorry, I can only help with health and civic queries."}

    user_message = params.get("user_message", "")
    session = _get_session(call_id)

    # ── Build pipeline state ─────────────────────────────────
    state = {
        **session,
        "user_utterance": user_message,
        "translated_utterance": user_message, # Pass native text directly to LangGraph
        "detected_language": "native",
        "turn_count": session["turn_count"] + 1,
    }

    # ── Run the multi-agent reasoning pipeline ───────────────
    result = await run_pipeline(state)

    # ── Extract native response ──────────────────────────────
    # LangGraph will now output natively
    response_native = result.get("final_response_en", "I was unable to process your request.")

    # ── Update session ───────────────────────────────────────
    session["turn_count"] = state["turn_count"]
    session["conversation_history"].append({
        "role": "user",
        "content": user_message,
    })
    session["conversation_history"].append({
        "role": "assistant",
        "content": response_native,
    })

    logger.info(
        "Processed native query (call=%s, severity=%s)",
        call_id, result.get("severity", "?"),
    )

    return {"result": response_native}


async def handle_end_of_call(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle the end-of-call report — log analytics and clean up session."""
    call_id = payload.get("call", {}).get("id", "unknown")
    transcript = payload.get("transcript", "")
    duration = payload.get("call", {}).get("duration", 0)

    logger.info(
        "Call ended (id=%s, duration=%ds, transcript_length=%d chars)",
        call_id, duration, len(transcript),
    )

    # Clean up session
    _sessions.pop(call_id, None)

    return {"status": "ok"}


async def handle_vapi_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Main dispatcher for all Vapi webhook events.
    Routes to the appropriate handler based on the 'message.type' field.
    """
    message = payload.get("message", {})
    event_type = message.get("type", "unknown")

    logger.info("Received Vapi event: %s", event_type)

    handlers = {
        "assistant-request": handle_assistant_request,
        "function-call": handle_function_call,
        "end-of-call-report": handle_end_of_call,
    }

    handler = handlers.get(event_type)
    if handler:
        return await handler(payload)

    logger.debug("Unhandled Vapi event type: %s", event_type)
    return {"status": "ignored", "event": event_type}
