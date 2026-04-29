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

    # ── LLM / Groq ──────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key for fast inference")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Default Groq model for reasoning")

    # ── Vapi AI (Telephony Orchestration) ────────────────────
    vapi_api_key: str = Field(default="", description="Vapi AI API key")
    vapi_phone_number_id: str = Field(default="", description="Vapi provisioned phone number ID")
    vapi_assistant_id: str = Field(default="", description="Vapi assistant ID (created via dashboard or API)")

    # ── ChromaDB (Vector Database) ───────────────────────────
    chroma_persist_dir: str = Field(default="./backend/data/chroma_db", description="ChromaDB persistence directory")
    chroma_collection_name: str = Field(default="vikas_knowledge", description="Default ChromaDB collection name")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Local sentence-transformer embedding model")

    # ── Bhashini (Multilingual Translation) ──────────────────
    bhashini_api_key: str = Field(default="", description="Bhashini ULCA API key")
    bhashini_user_id: str = Field(default="", description="Bhashini user ID")
    bhashini_pipeline_id: str = Field(default="", description="Bhashini pipeline config ID")

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
