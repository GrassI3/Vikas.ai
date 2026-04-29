"""
Vikas.ai — Main FastAPI Application
Entry point for the backend server.

Exposes:
  • POST /api/vapi/webhook     — Vapi telephony webhook
  • POST /api/query            — Direct text query (for testing)
  • POST /api/ingest           — Document ingestion endpoint
  • GET  /api/health           — Health check
  • WS   /ws/transcript        — Real-time transcript stream
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.agents.graph import run_pipeline
from backend.config import get_settings
from backend.knowledge.ingest import ingest_seed_data
from backend.knowledge.vector_db import get_or_create_collection
from backend.telephony.vapi_handler import handle_vapi_webhook
from backend.utils.guardrails import validate_output

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vikas.main")
settings = get_settings()


# ── Lifespan ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("  🚀  Vikas.ai — Decision Support Assistant  v%s", settings.app_version)
    logger.info("=" * 60)

    # Ensure ChromaDB collection exists
    collection = get_or_create_collection()
    if collection.count() == 0:
        logger.info("Knowledge base empty — ingesting seed documents…")
        count = await ingest_seed_data()
        logger.info("Seed ingestion complete — %d documents loaded", count)
    else:
        logger.info("Knowledge base ready — %d documents available", collection.count())

    yield

    logger.info("Vikas.ai shutting down")


# ── App creation ────────────────────────────────────────────
app = FastAPI(
    title="Vikas.ai",
    description=(
        "Explainable, voice-driven decision support assistant for "
        "healthcare, mental wellness, and civic accessibility."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ───────────────────────────────
class QueryRequest(BaseModel):
    """Direct text query for testing without telephony."""
    message: str
    language: str = "en"
    session_id: str | None = None


class QueryResponse(BaseModel):
    """Structured response from the reasoning pipeline."""
    response: str
    language: str
    severity: str
    domain: str
    citations: list[str]
    reasoning_steps: list[str]
    confidence: float
    disclaimer_injected: bool


class IngestRequest(BaseModel):
    """Document ingestion request."""
    documents: list[dict[str, Any]]


# ── Endpoints ───────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    collection = get_or_create_collection()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "knowledge_base_docs": collection.count(),
        "groq_configured": bool(settings.groq_api_key),
        "vapi_configured": bool(settings.vapi_api_key),
        "bhashini_configured": bool(settings.bhashini_api_key),
    }


@app.post("/api/vapi/webhook", tags=["Telephony"])
async def vapi_webhook(payload: dict[str, Any]):
    """
    Vapi AI webhook endpoint.
    Receives all call events and dispatches to the appropriate handler.
    """
    return await handle_vapi_webhook(payload)


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def direct_query(request: QueryRequest):
    """
    Direct text query endpoint for testing the reasoning pipeline
    without going through the telephony layer.
    """
    session_id = request.session_id or str(uuid.uuid4())

    # ── Language handling ────────────────────────────────────
    # ── Run reasoning pipeline natively ───────────────────────
    state = {
        "session_id": session_id,
        "user_utterance": request.message,
        "translated_utterance": request.message,
        "detected_language": request.language,
        "conversation_history": [],
        "turn_count": 1,
    }

    result = await run_pipeline(state)

    # ── Post-generation guardrails ───────────────────────────
    response_native = result.get("final_response_en", "Unable to process request.")
    is_safe, sanitized = validate_output(response_native)

    return QueryResponse(
        response=sanitized,
        language=request.language,
        severity=result.get("severity", "low"),
        domain=result.get("domain", "general"),
        citations=result.get("citations", []),
        reasoning_steps=result.get("reasoning_chain", []),
        confidence=result.get("confidence", 0.0),
        disclaimer_injected=result.get("disclaimer_injected", False),
    )


@app.post("/api/ingest", tags=["Knowledge"])
async def ingest_documents(request: IngestRequest):
    """Ingest new documents into the knowledge base."""
    from backend.knowledge.vector_db import add_documents

    count = await add_documents(request.documents)
    return {"status": "ok", "documents_ingested": count}


# ── WebSocket — Live Transcript Stream ──────────────────────
_ws_clients: list[WebSocket] = []


@app.websocket("/ws/transcript")
async def transcript_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcript streaming.
    Clients connect here to receive live call transcripts and pipeline events.
    """
    await websocket.accept()
    _ws_clients.append(websocket)
    logger.info("WebSocket client connected (total: %d)", len(_ws_clients))

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug("WS received: %s", data[:100])
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (remaining: %d)", len(_ws_clients))


async def broadcast_transcript(event: dict[str, Any]):
    """Broadcast a transcript event to all connected WebSocket clients."""
    for ws in _ws_clients[:]:
        try:
            await ws.send_json(event)
        except Exception:
            _ws_clients.remove(ws)


# ── Static frontend ─────────────────────────────────────────
import pathlib

_frontend_dir = pathlib.Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(_frontend_dir / "index.html")

    app.mount("/", StaticFiles(directory=str(_frontend_dir)), name="frontend")
    logger.info("Frontend dashboard mounted from %s", _frontend_dir)


# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
