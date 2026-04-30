"""
Vikas.ai — OTP Authentication via Twilio SMS
Generates a 6-digit OTP and sends it via Twilio.
Falls back to logging the code when Twilio is not configured.
"""

import random
import logging
from backend.config import get_settings

logger = logging.getLogger("vikas.auth")
settings = get_settings()


def send_otp(phone: str) -> str:
    """Generate a 6-digit OTP and send it via Twilio SMS.
    Returns the OTP code (always, for storage)."""
    code = str(random.randint(100000, 999999))

    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        logger.warning("Twilio not configured — OTP for %s is: %s (dev mode)", phone, code)
        return code

    try:
        from twilio.rest import Client
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(
            body=f"Your Vikas.ai verification code is: {code}",
            from_=settings.twilio_phone_number,
            to=phone,
        )
        logger.info("OTP sent to %s", phone)
    except Exception as e:
        logger.error("Twilio SMS failed for %s: %s", phone, e)

    return code
