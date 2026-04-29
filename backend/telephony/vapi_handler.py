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
                "model": "nova-3",
                # "multi" = Deepgram multilingual — auto-detects language per utterance.
                # Matches Vapi dashboard: Provider=Deepgram, Model=Nova 3, Language=Multilingual
                # Handles Hindi, Tamil, Telugu, Kannada, English, etc.
                "language": "multi",
                "smartFormat": True,
            },
            "model": {
                "provider": "groq",
                "model": settings.groq_model,
                "temperature": 0.3,
                "systemPrompt": (
                    "Aap Vikas hain — ek sahayak swasthya aur nagrik seva sahayak. "
                    "Aap users ko asan Hindi mein medical jaankari aur government schemes ke baare mein "
                    "sahi aur saral jawaab dete hain. "
                    "Jab user koi lakshan bataye, to 'process_query' function ka upyog karein "
                    "taaki aap PubMed research databases se sahi medical jaankari prapt kar sakein. "
                    "IMPORTANT: "
                    "1. Agar user Hindi mein bole, toh Hindi mein jawab dein. "
                    "2. Agar user Tamil/Telugu/Kannada mein bole, toh usi bhasha mein jawab dein. "
                    "3. Agar user English mein bole, toh English mein jawab dein. "
                    "4. Hamesha research cite karein. "
                    "5. Agar koi emergency ho, toh turant 112 call karne ki salah dein. "
                    "Chhote, spasht vaakya use karein — yeh response phone par bol kar sunaya jayega."
                ),
            },
            "voice": {
                "provider": "azure",
                # hi-IN-SwaraNeural = Microsoft's best Hindi female neural voice
                # Alternatives: hi-IN-MadhurNeural (male), hi-IN-SwaraNeural (female)
                "voiceId": "hi-IN-SwaraNeural",
            },
            # First thing the AI says when the call connects
            "firstMessage": (
                "Namaste! Main Vikas hun, aapka swasthya aur nagrik seva sahayak. "
                "Aap mujhse Hindi, Tamil, ya English mein baat kar sakte hain. "
                "Aaj main aapki kya madad kar sakta hun?"
            ),
            # Said when the call ends
            "endCallMessage": (
                "Dhanyavaad. Apna khayal rakhen. "
                "Kisi bhi gambhir sthiti mein kisi daktar se zaroor milaen."
            ),
            "serverUrl": None,  # Vapi uses the webhook URL it already has
            "functions": [
                {
                    "name": "process_query",
                    "description": (
                        "Analyse the user's health concern or civic query using peer-reviewed "
                        "PubMed medical research and multi-step clinical reasoning. "
                        "Returns an evidence-based, cited response in the user's language."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_message": {
                                "type": "string",
                                "description": "The user's spoken message, exactly as transcribed.",
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
