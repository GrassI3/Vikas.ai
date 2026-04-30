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
from backend.utils.db import init_db, store_otp, check_otp, get_calls_for_phone
from backend.utils.auth import send_otp

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

    # Initialize SQLite database
    init_db()

    # Ensure ChromaDB collection exists
    collection = get_or_create_collection()
    if collection.count() == 0:
        logger.info("Knowledge base empty — ingesting seed documents…")
        count = await ingest_seed_data()
        logger.info("Seed ingestion complete — %d documents loaded", count)
    else:
        logger.info("Knowledge base ready — %d documents available", collection.count())

    # ── Auto-start ngrok tunnel ──────────────────────────────
    tunnel = None
    try:
        from pyngrok import ngrok, conf
        # Use the pre-installed native ngrok binary (avoids pyngrok auto-download)
        conf.get_default().ngrok_path = (
            r"C:\Users\jaiga\AppData\Local\Microsoft\WindowsApps\ngrok.exe"
        )
        tunnel = ngrok.connect(settings.port, bind_tls=True)
        public_url = tunnel.public_url
        webhook_url = f"{public_url}/api/vapi/webhook"
        logger.info("=" * 60)
        logger.info("  🌐  PUBLIC URL: %s", public_url)
        logger.info("  📞  VAPI WEBHOOK: %s", webhook_url)
        logger.info("  👆  Paste the webhook URL into Vapi Dashboard → Server URL")
        logger.info("=" * 60)
    except Exception as e:
        logger.warning("ngrok tunnel failed (non-fatal): %s", e)
        logger.info("Tip: pip install pyngrok && ngrok config add-authtoken YOUR_TOKEN")

    yield

    # Cleanup tunnel on shutdown
    if tunnel:
        try:
            from pyngrok import ngrok as _ngrok
            _ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass
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


class OTPRequest(BaseModel):
    """OTP request — send a code to this phone."""
    phone_number: str


class OTPVerify(BaseModel):
    """OTP verification request."""
    phone_number: str
    code: str


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
        "twilio_configured": bool(settings.twilio_account_sid),
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


class PubMedRequest(BaseModel):
    """PubMed ingestion request."""
    topic: str | None = None
    all_defaults: bool = False
    max_per_topic: int = 20


@app.post("/api/pubmed/ingest", tags=["Knowledge"])
async def ingest_pubmed(request: PubMedRequest):
    """Bulk-ingest PubMed abstracts by topic or all defaults."""
    from backend.knowledge.pubmed import ingest_pubmed_topic, ingest_all_default_topics

    if request.all_defaults:
        count = await ingest_all_default_topics(max_per_topic=request.max_per_topic)
    elif request.topic:
        count = await ingest_pubmed_topic(request.topic, max_articles=request.max_per_topic)
    else:
        return {"status": "error", "message": "Provide 'topic' or set 'all_defaults' to true"}

    # Refresh health check count
    collection = get_or_create_collection()
    return {
        "status": "ok",
        "articles_ingested": count,
        "total_knowledge_base": collection.count(),
    }


# ── Authentication & Recordings ─────────────────────────────

@app.post("/api/auth/request-otp", tags=["Auth"])
async def request_otp(req: OTPRequest):
    """Send a 6-digit OTP to the given phone number."""
    code = send_otp(req.phone_number)
    store_otp(req.phone_number, code)
    return {"status": "ok", "message": "OTP sent"}


@app.post("/api/auth/verify-otp", tags=["Auth"])
async def verify_otp(req: OTPVerify):
    """Verify the OTP. Returns the user's call recordings on success."""
    if not check_otp(req.phone_number, req.code):
        return {"status": "error", "message": "Invalid or expired OTP"}

    # Try local DB first, fallback to Vapi API
    recordings = get_calls_for_phone(req.phone_number)
    if not recordings:
        recordings = await _fetch_vapi_calls(req.phone_number)
    return {"status": "ok", "recordings": recordings}


@app.post("/api/recordings", tags=["Recordings"])
async def get_recordings(req: OTPRequest):
    """Fetch recordings for a verified phone number."""
    recordings = get_calls_for_phone(req.phone_number)
    if not recordings:
        recordings = await _fetch_vapi_calls(req.phone_number)
    return {"status": "ok", "recordings": recordings}


async def _fetch_vapi_calls(phone: str) -> list[dict]:
    """Fetch call recordings from the Vapi API for a specific phone number."""
    import httpx
    from backend.utils.db import save_call

    if not settings.vapi_api_key:
        logger.warning("Vapi API key not configured, cannot fetch recordings")
        return []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.vapi.ai/call",
                headers={"Authorization": f"Bearer {settings.vapi_api_key}"},
            )
            resp.raise_for_status()
            all_calls = resp.json()

        results = []
        for call in all_calls:
            customer_phone = call.get("customer", {}).get("number", "")
            if customer_phone != phone:
                continue

            call_id = call.get("id", "")
            recording = call.get("recordingUrl", "") or call.get("artifact", {}).get("recordingUrl", "")

            raw_transcript = call.get("transcript", "") or call.get("artifact", {}).get("transcript", "")
            if isinstance(raw_transcript, list):
                transcript = "\n".join(f"{t.get('role','?')}: {t.get('text','')}" for t in raw_transcript)
            else:
                transcript = str(raw_transcript) if raw_transcript else ""

            summary = call.get("analysis", {}).get("summary", "")
            duration = call.get("duration", 0)
            created_at = call.get("createdAt", "")

            record = {
                "id": call_id,
                "phone": customer_phone,
                "recording": recording,
                "transcript": transcript,
                "summary": summary,
                "duration": duration,
                "created_at": created_at,
            }
            results.append(record)

            # Also cache in local DB for next time
            if call_id:
                save_call(call_id, customer_phone, recording, transcript, summary, duration)

        logger.info("Fetched %d calls from Vapi API for %s", len(results), phone)
        return results

    except Exception as e:
        logger.error("Failed to fetch calls from Vapi API: %s", e)
        return []


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
