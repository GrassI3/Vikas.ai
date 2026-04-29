"""
Vikas.ai — Configuration Management
Centralizes all environment variables and API keys using Pydantic settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or a .env file."""

    # ── General ──────────────────────────────────────────────
    app_name: str = "Vikas.ai"
    app_version: str = "0.1.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # ── LLM / Groq (used internally by reasoning pipeline nodes) ────
    groq_api_key: str = Field(default="", description="Groq API key for fast inference")
    groq_model: str = Field(default="llama-3.1-8b-instant", description="Groq model for internal pipeline nodes (intake, reasoning, synthesis)")

    # ── LLM / Anthropic (used by Vapi as the voice orchestration LLM) ──
    anthropic_model: str = Field(default="claude-opus-4-5", description="Anthropic model name for Vapi voice assistant orchestration")

    # ── Vapi AI (Telephony Orchestration) ────────────────────
    vapi_api_key: str = Field(default="", description="Vapi AI API key")
    vapi_phone_number_id: str = Field(default="", description="Vapi provisioned phone number ID")
    vapi_assistant_id: str = Field(default="", description="Vapi assistant ID (created via dashboard or API)")

    # ── ChromaDB (Vector Database) ───────────────────────────
    chroma_persist_dir: str = Field(default="./backend/data/chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="vikas_knowledge", description="Default ChromaDB collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Local sentence-transformer embedding model")

    # ── Twilio (SMS Fallback) ────────────────────────────────
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")
    twilio_phone_number: str = Field(default="", description="Twilio provisioned phone number")

    # ── Safety Thresholds ────────────────────────────────────
    emergency_keywords: list[str] = Field(
        default=[
            "suicide", "kill myself", "want to die", "end my life",
            "heart attack", "chest pain", "can't breathe", "stroke",
            "unconscious", "severe bleeding", "overdose",
        ],
        description="Keywords that trigger immediate emergency escalation",
    )
    severity_escalation_threshold: float = Field(
        default=0.85,
        description="Confidence threshold (0-1) above which the system escalates to emergency protocol",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
