"""
main.py — FastAPI inference server for the Sahaayak Triage Engine

Endpoints:
    POST /api/triage       — Run triage classification
    POST /api/xai          — Get explainability data for a prediction
    POST /api/feedback     — Submit user correction feedback
    POST /api/reload       — Hot-reload the model checkpoint
    GET  /api/health       — Health check
"""

import os
import sys
import re
import sqlite3
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.triage_net import SahaayakTriageNet, SEVERITY_CLASSES
from model.xai_engine import XAIEngine
from knowledge_graph.resolver import KnowledgeGraphResolver


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")
DB_PATH = os.path.join("db", "feedback.sqlite")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model: Optional[SahaayakTriageNet] = None
xai_engine: Optional[XAIEngine] = None
kg_resolver: Optional[KnowledgeGraphResolver] = None


def init_db():
    """Initialize the SQLite feedback database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            model_prediction TEXT NOT NULL,
            user_correction TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            used_in_epoch INTEGER DEFAULT NULL
        )
    """)
    conn.commit()
    conn.close()


def load_model() -> SahaayakTriageNet:
    """Load the model and optionally a trained checkpoint."""
    m = SahaayakTriageNet(freeze_encoder=True)
    if os.path.exists(BEST_CHECKPOINT):
        meta = m.load_checkpoint(BEST_CHECKPOINT, device=DEVICE)
        print(f"[MODEL] Loaded checkpoint: {BEST_CHECKPOINT} | meta={meta}")
    else:
        print("[MODEL] No checkpoint found. Using untrained classification head.")
    m = m.to(DEVICE)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, xai_engine, kg_resolver
    print("\n[BOOT] Initializing Sahaayak Triage Engine...")
    init_db()
    model = load_model()
    xai_engine = XAIEngine(model, device=DEVICE)
    kg_resolver = KnowledgeGraphResolver()
    print("[BOOT] Ready.\n")
    yield
    print("[SHUTDOWN] Sahaayak Triage Engine stopped.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sahaayak Triage Engine",
    description="Hybrid neural-symbolic medical triage API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TriageRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Symptom description")
    session_id: str = Field(default="anonymous", description="Session identifier")
    include_xai: bool = Field(default=False, description="Include explainability data")


class FeedbackRequest(BaseModel):
    session_id: str
    input_text: str
    model_prediction: str
    user_correction: str = Field(..., pattern="^(LOW|MODERATE|HIGH|EMERGENCY)$")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Clinical analysis builder
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {
    "LOW":       {"en": "Low",       "hi": "कम",      "mr": "कमी"},
    "MODERATE":  {"en": "Moderate",  "hi": "मध्यम",    "mr": "मध्यम"},
    "HIGH":      {"en": "High",      "hi": "गंभीर",    "mr": "गंभीर"},
    "EMERGENCY": {"en": "Emergency", "hi": "आपातकालीन", "mr": "आणीबाणी"},
}

SEVERITY_DESCRIPTION = {
    "LOW":       {"en": "This appears to be a mild condition. Monitor symptoms at home.",
                  "hi": "यह एक हल्की स्थिति प्रतीत होती है। घर पर लक्षणों की निगरानी करें।",
                  "mr": "ही सौम्य स्थिती असल्याचे दिसते. घरी लक्षणांवर लक्ष ठेवा."},
    "MODERATE":  {"en": "This needs medical attention. Please consult a doctor within 24 hours.",
                  "hi": "इसे चिकित्सा ध्यान की आवश्यकता है। कृपया 24 घंटे के भीतर डॉक्टर से परामर्श करें।",
                  "mr": "यासाठी वैद्यकीय लक्ष आवश्यक आहे. कृपया 24 तासांच्या आत डॉक्टरांचा सल्ला घ्या."},
    "HIGH":      {"en": "This is a serious condition. Seek medical care urgently.",
                  "hi": "यह एक गंभीर स्थिति है। तुरंत चिकित्सा सहायता लें।",
                  "mr": "ही गंभीर स्थिती आहे. तातडीने वैद्यकीय मदत घ्या."},
    "EMERGENCY": {"en": "EMERGENCY — Seek immediate medical attention or call 108.",
                  "hi": "आपातकाल — तुरंत चिकित्सा सहायता लें या 108 पर कॉल करें।",
                  "mr": "आणीबाणी — तातडीने वैद्यकीय मदत घ्या किंवा 108 वर कॉल करा."},
}

GENERAL_ADVICE = {
    "LOW": [
        {"en": "Rest and stay hydrated", "hi": "आराम करें और पानी पीते रहें", "mr": "विश्रांती घ्या आणि पाणी प्या"},
        {"en": "Monitor symptoms for 24-48 hours", "hi": "24-48 घंटे लक्षणों पर नज़र रखें", "mr": "24-48 तास लक्षणांवर लक्ष ठेवा"},
        {"en": "Visit a doctor if symptoms worsen", "hi": "लक्षण बिगड़ने पर डॉक्टर को दिखाएं", "mr": "लक्षणे वाढल्यास डॉक्टरांना भेटा"},
    ],
    "MODERATE": [
        {"en": "Consult a doctor within 24 hours", "hi": "24 घंटे के भीतर डॉक्टर से मिलें", "mr": "24 तासांच्या आत डॉक्टरांना भेटा"},
        {"en": "Avoid strenuous activity", "hi": "भारी शारीरिक गतिविधि से बचें", "mr": "कठोर शारीरिक हालचाली टाळा"},
        {"en": "Keep a symptom diary to share with the doctor", "hi": "डॉक्टर को दिखाने के लिए लक्षणों की डायरी रखें", "mr": "डॉक्टरांना दाखवण्यासाठी लक्षणांची डायरी ठेवा"},
    ],
    "HIGH": [
        {"en": "Seek urgent medical care — visit ER or call your doctor now", "hi": "तुरंत चिकित्सा सहायता लें — ER जाएं या डॉक्टर को कॉल करें", "mr": "तातडीने वैद्यकीय मदत घ्या — ER ला जा किंवा डॉक्टरांना कॉल करा"},
        {"en": "Do not ignore worsening symptoms", "hi": "बिगड़ते लक्षणों को नज़रअंदाज़ न करें", "mr": "वाढत्या लक्षणांकडे दुर्लक्ष करू नका"},
    ],
    "EMERGENCY": [
        {"en": "Call 108 (emergency ambulance) IMMEDIATELY", "hi": "तुरंत 108 (एम्बुलेंस) पर कॉल करें", "mr": "तातडीने 108 (ॲम्ब्युलन्स) वर कॉल करा"},
        {"en": "If available, go to the nearest Emergency Room", "hi": "यदि संभव हो तो निकटतम आपातकालीन कक्ष जाएं", "mr": "शक्य असल्यास जवळच्या आपत्कालीन कक्षात जा"},
    ],
}


def detect_language(text: str) -> str:
    """Detect if user is writing in Hindi/Marathi (Devanagari) or English."""
    devanagari = len(re.findall(r'[\u0900-\u097F]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    if devanagari > latin:
        return "hi"  # Devanagari script — could be Hindi or Marathi
    # Check for common Hindi romanized patterns
    hindi_markers = ["mein", "hai", "ko", "ka", "ki", "se", "nahi", "hoon", "raha", "rahi", "bahut", "mere", "mera"]
    text_lower = text.lower()
    hindi_score = sum(1 for m in hindi_markers if f" {m} " in f" {text_lower} ")
    if hindi_score >= 2:
        return "hi"
    return "en"


def build_clinical_analysis(kg_result: dict, model_result: dict, final_prediction: str, text: str) -> dict:
    """Build a rich, structured clinical analysis from KG + model data."""
    lang = detect_language(text)
    kg_matches = kg_result["knowledge_graph"]["matches"]
    overrides = kg_result.get("safety_overrides", [])

    # --- Symptoms Detected ---
    symptoms_detected = []
    for m in kg_matches:
        symptoms_detected.append({
            "condition": m["clinical_name"],
            "idiom_matched": m.get("idiom", ""),
            "literal": m.get("literal", ""),
            "match_confidence": m.get("match_score", 0),
            "default_severity": m.get("default_severity", "LOW"),
            "context": m.get("context", ""),
        })

    # --- Possible Causes ---
    possible_causes = []
    seen_causes = set()
    for m in kg_matches:
        name = m["clinical_name"]
        if name not in seen_causes:
            seen_causes.add(name)
            possible_causes.append({
                "condition": name,
                "severity": m.get("default_severity", "LOW"),
                "severity_label": SEVERITY_LABELS.get(m.get("default_severity", "LOW"), {}).get(lang, m.get("default_severity", "LOW")),
                "escalation_triggers": m.get("escalation_triggers", []),
            })

    # --- Related Conditions (graph edges) ---
    related_conditions = []
    seen_related = set()
    for m in kg_matches:
        for rel in m.get("related_symptoms", []):
            rel_name = rel.replace("med:", "")
            if rel_name not in seen_related:
                seen_related.add(rel_name)
                related_conditions.append(rel_name)

    # --- Recommended Actions ---
    actions = []
    # Safety actions first
    for o in overrides:
        actions.append({
            "priority": "CRITICAL",
            "action": o["action"],
            "flag": o["flag"],
        })
    # General advice based on severity
    for advice in GENERAL_ADVICE.get(final_prediction, []):
        actions.append({
            "priority": final_prediction,
            "action": advice.get(lang, advice["en"]),
        })

    return {
        "language_detected": lang,
        "severity_label": SEVERITY_LABELS.get(final_prediction, {}).get(lang, final_prediction),
        "severity_description": SEVERITY_DESCRIPTION.get(final_prediction, {}).get(lang, ""),
        "symptoms_detected": symptoms_detected,
        "possible_causes": possible_causes,
        "related_conditions": related_conditions,
        "recommended_actions": actions,
        "model_probabilities": {
            cls: round(model_result["probabilities"][cls], 4)
            for cls in ["LOW", "MODERATE", "HIGH", "EMERGENCY"]
        },
    }


@app.post("/api/triage")
async def triage(req: TriageRequest):
    """
    Core triage endpoint.

    Pipeline:
      1. Knowledge Graph resolves cultural idioms
      2. Safety Gate checks for emergency overrides
      3. Neural model classifies severity
      4. Clinical analysis builder creates a rich report
    """
    # Step 1 & 2: KG + Safety Gate
    kg_result = kg_resolver.process(req.text)

    # Step 3: Neural classification
    model_result = model.predict(req.text, device=DEVICE)

    # Step 4: Merge — safety override always wins
    final_prediction = model_result["prediction"]
    severity_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EMERGENCY": 3}

    # Upgrade if KG suggests higher
    kg_sev = kg_result.get("kg_suggested_severity", "LOW")
    if severity_rank.get(kg_sev, 0) > severity_rank.get(final_prediction, 0):
        final_prediction = kg_sev

    # Safety override is absolute
    overrides = kg_result.get("safety_overrides", [])
    if overrides:
        final_prediction = "EMERGENCY"

    # Step 5: Build rich clinical analysis
    analysis = build_clinical_analysis(kg_result, model_result, final_prediction, req.text)

    response = {
        "prediction": final_prediction,
        "model_prediction": model_result["prediction"],
        "confidence": model_result["confidence"],
        "probabilities": model_result["probabilities"],
        "knowledge_graph_matches": kg_result["knowledge_graph"]["matches"],
        "safety_overrides": overrides,
        "safety_flags": [o["flag"] for o in overrides],
        "safety_actions": [o["action"] for o in overrides],
        "session_id": req.session_id,
        "clinical_analysis": analysis,
    }

    # Optional XAI
    if req.include_xai:
        xai_result = xai_engine.explain(req.text)
        response["xai"] = {
            "tokens": xai_result["tokens"],
            "attributions": xai_result["attributions"],
            "top_contributors": xai_result["top_contributors"],
        }

    return response


@app.post("/api/xai")
async def explain(req: TriageRequest):
    """Standalone XAI endpoint for detailed explainability."""
    xai_result = xai_engine.explain(req.text)
    kg_result = kg_resolver.process(req.text)

    return {
        **xai_result,
        "knowledge_graph_matches": kg_result["knowledge_graph"]["matches"],
        "safety_overrides": kg_result["safety_overrides"],
    }


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Store user correction feedback for future retraining."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO feedback (session_id, input_text, model_prediction, user_correction)
               VALUES (?, ?, ?, ?)""",
            (req.session_id, req.input_text, req.model_prediction, req.user_correction),
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM feedback WHERE used_in_epoch IS NULL").fetchone()[0]
        return {
            "status": "recorded",
            "pending_feedback_count": count,
            "message": f"Thank you. {count} corrections pending for next retraining cycle.",
        }
    finally:
        conn.close()


@app.post("/api/reload")
async def reload_model():
    """Hot-reload the model checkpoint without restarting the server."""
    global model, xai_engine
    try:
        model = load_model()
        xai_engine = XAIEngine(model, device=DEVICE)
        return {"status": "reloaded", "checkpoint": BEST_CHECKPOINT}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    has_checkpoint = os.path.exists(BEST_CHECKPOINT)
    conn = sqlite3.connect(DB_PATH)
    feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM feedback WHERE used_in_epoch IS NULL").fetchone()[0]
    conn.close()

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "checkpoint_exists": has_checkpoint,
        "total_feedback": feedback_count,
        "pending_retraining": pending,
        "severity_classes": SEVERITY_CLASSES,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
