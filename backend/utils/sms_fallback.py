"""
Vikas.ai — SMS / USSD Fallback Module
Provides asynchronous fallback when voice calls drop due to
network instability in low-bandwidth environments.

When a WebSocket audio stream disconnects mid-conversation,
the system formulates the pending recommendation and delivers
it via SMS, ensuring decision-support reaches the user regardless
of real-time connectivity.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.config import get_settings

logger = logging.getLogger("vikas.utils.sms_fallback")
settings = get_settings()


# ── Twilio SMS Gateway ──────────────────────────────────────
TWILIO_SMS_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


async def send_sms(
    to_number: str,
    message: str,
    from_number: str | None = None,
) -> dict[str, Any]:
    """
    Send an SMS message via the Twilio REST API.

    This is the primary fallback channel when a voice call is
    interrupted due to network instability.

    Args:
        to_number:   Recipient phone number in E.164 format (e.g. +919876543210)
        message:     The text content to send (max 1600 chars for concatenated SMS)
        from_number: Sender phone number (defaults to Vapi provisioned number)

    Returns:
        Twilio API response dict or error dict
    """
    twilio_sid = getattr(settings, "twilio_account_sid", "")
    twilio_token = getattr(settings, "twilio_auth_token", "")

    if not twilio_sid or not twilio_token:
        logger.warning("Twilio SMS credentials not configured — message not sent")
        return {"status": "skipped", "reason": "twilio_not_configured"}

    sender = from_number or getattr(settings, "twilio_phone_number", "")
    if not sender:
        logger.error("No sender phone number configured for SMS fallback")
        return {"status": "error", "reason": "no_sender_number"}

    # Truncate message to SMS limits (1600 chars for concatenated)
    if len(message) > 1600:
        message = message[:1597] + "..."

    url = TWILIO_SMS_URL.format(sid=twilio_sid)
    payload = {
        "To": to_number,
        "From": sender,
        "Body": message,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                data=payload,
                auth=(twilio_sid, twilio_token),
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(
                "SMS sent successfully (sid=%s, to=%s, segments=%s)",
                data.get("sid"),
                to_number,
                data.get("num_segments"),
            )
            return {"status": "sent", "message_sid": data.get("sid")}
    except httpx.HTTPError as e:
        logger.error("SMS send failed: %s", e)
        return {"status": "error", "reason": str(e)}


async def handle_call_disconnect(
    session: dict[str, Any],
    pending_response: str,
    user_phone: str,
) -> dict[str, Any]:
    """
    Graceful degradation handler invoked when a voice call drops.

    Retrieves the pending AI response from the session state and
    delivers it via SMS to ensure the user receives the critical
    decision-support information.

    Args:
        session:          The active session state dict
        pending_response: The AI-generated response that was not delivered
        user_phone:       The user's phone number in E.164 format

    Returns:
        SMS delivery result dict
    """
    session_id = session.get("session_id", "unknown")
    turn = session.get("turn_count", 0)

    # Prepend context header for SMS
    sms_body = (
        f"[Vikas.ai — Your call was disconnected]\n\n"
        f"{pending_response}\n\n"
        f"Reply CALL to reconnect, or dial again anytime. "
        f"This is an automated service — always consult a doctor for medical advice."
    )

    logger.info(
        "Call disconnected — sending SMS fallback (session=%s, turn=%d)",
        session_id,
        turn,
    )

    return await send_sms(user_phone, sms_body)


def format_ussd_response(text: str, max_length: int = 182) -> str:
    """
    Format a response for USSD delivery (extremely constrained channel).

    USSD messages are limited to ~182 characters. This function
    compresses the AI response into an actionable summary.

    Args:
        text:       Full response text
        max_length: Maximum character count (default 182 for GSM USSD)

    Returns:
        Compressed USSD-safe string
    """
    # Strip any markdown or special formatting
    clean = text.replace("\n", " ").replace("  ", " ").strip()

    if len(clean) <= max_length:
        return clean

    # Truncate and append action hint
    truncated = clean[: max_length - 25]
    # Cut at last complete word
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + "... Call for details."
