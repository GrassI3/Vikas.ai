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

# Force HuggingFace offline mode to prevent WinError 10054 connection crashes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    "LOW":       {"en": "Low",       "hi": "कम",      "mr": "कमी",     "bn": "কম",       "te": "తక్కువ",   "ta": "குறைவு",    "gu": "ઓછું",     "kn": "ಕಡಿಮೆ",    "ml": "കുറവ്",     "pa": "ਘੱਟ"},
    "MODERATE":  {"en": "Moderate",  "hi": "मध्यम",    "mr": "मध्यम",   "bn": "মাঝারি",   "te": "మితమైన",   "ta": "மிதமான",    "gu": "મધ્યમ",    "kn": "ಮಧ್ಯಮ",    "ml": "മിതമായ",   "pa": "ਦਰਮਿਆਨਾ"},
    "HIGH":      {"en": "High",      "hi": "गंभीर",    "mr": "गंभीर",   "bn": "উচ্চ",     "te": "అధిక",     "ta": "உயர்",      "gu": "ઉચ્ચ",     "kn": "ಹೆಚ್ಚು",     "ml": "ഉയർന്ന",    "pa": "ਉੱਚ"},
    "EMERGENCY": {"en": "Emergency", "hi": "आपातकालीन", "mr": "आणीबाणी", "bn": "জরুরী",    "te": "అత్యవసర",  "ta": "அவசரம்",    "gu": "કટોકટી",   "kn": "ತುರ್ತು",    "ml": "അടിയന്തരം", "pa": "ਐਮਰਜੈਂਸੀ"},
}

SEVERITY_DESCRIPTION = {
    "LOW": {
        "en": "This appears to be a mild condition. Monitor symptoms at home.",
        "hi": "यह एक हल्की स्थिति प्रतीत होती है। घर पर लक्षणों की निगरानी करें।",
        "mr": "ही सौम्य स्थिती असल्याचे दिसते. घरी लक्षणांवर लक्ष ठेवा.",
        "bn": "এটি একটি মৃদু অবস্থা বলে মনে হচ্ছে। বাড়িতে লক্ষণগুলি পর্যবেক্ষণ করুন।",
        "te": "ఇది తేలికపాటి పరిస్థితిగా కనిపిస్తోంది. ఇంట్లో లక్షణాలను గమనించండి.",
        "ta": "இது ஒரு லேசான நிலையாகத் தெரிகிறது. வீட்டிலேயே அறிகுறிகளைக் கண்காணிக்கவும்.",
        "gu": "આ એક હળવી સ્થિતિ હોવાનું જણાય છે. ઘરે લક્ષણોનું નિરીક્ષણ કરો.",
        "kn": "ಇದು ಸೌಮ್ಯ ಸ್ಥಿತಿ ಎಂದು ತೋರುತ್ತದೆ. ಮನೆಯಲ್ಲಿ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ.",
        "ml": "ഇതൊരു ചെറിയ അവസ്ഥയാണെന്ന് തോന്നുന്നു. വീട്ടിൽ രോഗലക്ഷണങ്ങൾ നിരീക്ഷിക്കുക.",
        "pa": "ਇਹ ਇੱਕ ਹਲਕੀ ਸਥਿਤੀ ਜਾਪਦੀ ਹੈ। ਘਰ ਵਿੱਚ ਲੱਛਣਾਂ ਦੀ ਨਿਗਰਾਨੀ ਕਰੋ।"
    },
    "MODERATE": {
        "en": "This needs medical attention. Please consult a doctor within 24 hours.",
        "hi": "इसे चिकित्सा ध्यान की आवश्यकता है। कृपया 24 घंटे के भीतर डॉक्टर से परामर्श करें।",
        "mr": "यासाठी वैद्यकीय लक्ष आवश्यक आहे. कृपया 24 तासांच्या आत डॉक्टरांचा सल्ला घ्या.",
        "bn": "এর জন্য চিকিৎসার প্রয়োজন। দয়া করে 24 ঘন্টার মধ্যে একজন ডাক্তারের পরামর্শ নিন।",
        "te": "దీనికి వైద్య సహాయం అవసరం. దయచేసి 24 గంటలలోపు వైద్యుడిని సంప్రదించండి.",
        "ta": "இதற்கு மருத்துவ கவனம் தேவை. 24 மணி நேரத்திற்குள் மருத்துவரை அணுகவும்.",
        "gu": "આ માટે તબીબી ધ્યાનની જરૂર છે. કૃપા કરીને 24 કલાકની અંદર ડૉક્ટરની સલાહ લો.",
        "kn": "ಇದಕ್ಕೆ ವೈದ್ಯಕೀಯ ಗಮನ ಅಗತ್ಯ. ದಯವಿಟ್ಟು 24 ಗಂಟೆಗಳ ಒಳಗೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "ml": "ഇതിന് വൈദ്യസഹായം ആവശ്യമാണ്. 24 മണിക്കൂറിനുള്ളിൽ ഒരു ഡോക്ടറെ സമീപിക്കുക.",
        "pa": "ਇਸ ਵੱਲ ਡਾਕਟਰੀ ਧਿਆਨ ਦੇਣ ਦੀ ਲੋੜ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ 24 ਘੰਟਿਆਂ ਦੇ ਅੰਦਰ ਡਾਕਟਰ ਦੀ ਸਲਾહ ਲਓ।"
    },
    "HIGH": {
        "en": "This is a serious condition. Seek medical care urgently.",
        "hi": "यह एक गंभीर स्थिति है। तुरंत चिकित्सा सहायता लें।",
        "mr": "ही गंभीर स्थिती आहे. तातडीने वैद्यकीय मदत घ्या.",
        "bn": "এটি একটি গুরুতর অবস্থা। জরুরি ভিত্তিতে চিকিৎসা নিন।",
        "te": "ఇది తీవ్రమైన పరిస్థితి. అత్యవసరంగా వైద్య సంరక్షణ పొందండి.",
        "ta": "இது ஒரு தீவிரமான நிலை. அவசரமாக மருத்துவ உதவியை நாடுங்கள்.",
        "gu": "આ એક ગંભીર સ્થિતિ છે. તાત્કાલિક તબીબી સંભાળ લો.",
        "kn": "ಇದು ಗಂಭೀರ ಸ್ಥಿತಿಯಾಗಿದೆ. ತುರ್ತಾಗಿ ವೈದ್ಯಕೀಯ ಆರೈಕೆಯನ್ನು ಪಡೆಯಿರಿ.",
        "ml": "ഇതൊരു ഗുരുതരമായ അവസ്ഥയാണ്. അടിയന്തരമായി വൈദ്യസഹായം തേടുക.",
        "pa": "ਇਹ ਇੱਕ ਗੰਭੀਰ ਸਥਿਤੀ ਹੈ। ਤੁਰੰਤ ਡਾਕਟਰੀ ਦੇਖਭਾਲ ਲਓ।"
    },
    "EMERGENCY": {
        "en": "EMERGENCY — Seek immediate medical attention or call 108.",
        "hi": "आपातकाल — तुरंत चिकित्सा सहायता लें या 108 पर कॉल करें।",
        "mr": "आणीबाणी — तातडीने वैद्यकीय मदत घ्या किंवा 108 वर कॉल करा.",
        "bn": "জরুরী — অবিলম্বে চিকিৎসা সহায়তা নিন বা 108 নম্বরে কল করুন।",
        "te": "అత్యవసర — తక్షణ వైద్య సహాయం తీసుకోండి లేదా 108కి కాల్ చేయండి.",
        "ta": "அவசரம் — உடனடியாக மருத்துவ உதவியை நாடுங்கள் அல்லது 108 ஐ அழைக்கவும்.",
        "gu": "કટોકટી — તાત્કાલિક તબીબી ધ્યાન લો અથવા 108 પર કૉલ કરો.",
        "kn": "ತುರ್ತು — ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ನೆರವು ಪಡೆಯಿರಿ ಅಥವಾ 108 ಗೆ ಕರೆ ಮಾಡಿ.",
        "ml": "അടിയന്തരം — ഉടനടി വൈദ്യസഹായം തേടുക അല്ലെങ്കിൽ 108 ൽ വിളിക്കുക.",
        "pa": "ਐਮਰਜੈਂਸੀ — ਤੁਰੰਤ ਡਾਕਟਰੀ ਸਹਾਇਤਾ ਲਓ ਜਾਂ 108 'ਤੇ ਕਾਲ ਕਰੋ।"
    },
}

# Merged above.

GENERAL_ADVICE = {
    "LOW": [
        {"en": "Rest and stay hydrated", "hi": "आराम करें और पानी पीते रहें", "mr": "विश्रांती घ्या आणि पाणी प्या", "bn": "বিশ্রাম নিন এবং হাইড্রেটেড থাকুন", "te": "విశ్రాంతి తీసుకోండి మరియు హైడ్రేటెడ్‌గా ఉండండి"},
        {"en": "Monitor symptoms for 24-48 hours", "hi": "24-48 घंटे लक्षणों पर नज़र रखें", "mr": "24-48 तास लक्षणांवर लक्ष ठेवा", "bn": "24-48 ঘন্টার জন্য লক্ষণ পর্যবেক্ষণ করুন", "te": "24-48 గంటల పాటు లక్షణాలను గమనించండి"},
        {"en": "Visit a doctor if symptoms worsen", "hi": "लक्षण बिगड़ने पर डॉक्टर को दिखाएं", "mr": "लक्षणे वाढल्यास डॉक्टरांना भेटा", "bn": "লক্ষণগুলি আরও খারাপ হলে ডাক্তারের কাছে যান", "te": "లక్షణాలు తీవ్రమైతే వైద్యుడిని సందర్శించండి"},
    ],
    "MODERATE": [
        {"en": "Consult a doctor within 24 hours", "hi": "24 घंटे के भीतर डॉक्टर से मिलें", "mr": "24 तासांच्या आत डॉक्टरांना भेटा", "bn": "24 ঘন্টার মধ্যে একজন ডাক্তারের পরামর্শ নিন", "te": "24 గంటలలోపు వైద్యుడిని సంప్రదించండి"},
        {"en": "Avoid strenuous activity", "hi": "भारी शारीरिक गतिविधि से बचें", "mr": "कठोर शारीरिक हालचाली टाळा", "bn": "কঠোর কার্যকলাপ এড়িয়ে চলুন", "te": "శ్రమతో కూడిన కార్యకలాపాలను నివారించండి"},
        {"en": "Keep a symptom diary to share with the doctor", "hi": "डॉक्टर को दिखाने के लिए लक्षणों की डायरी रखें", "mr": "डॉक्टरांना दाखवण्यासाठी लक्षणांची डायरी ठेवा", "bn": "ডাক্তারের সাথে শেয়ার করার জন্য একটি লক্ষণ ডায়েরি রাখুন", "te": "వైద్యుడితో పంచుకోవడానికి లక్షణాల డైరీని ఉంచండి"},
    ],
    "HIGH": [
        {"en": "Seek urgent medical care — visit ER or call your doctor now", "hi": "तुरंत चिकित्सा सहायता लें — ER जाएं या डॉक्टर को कॉल करें", "mr": "तातडीने वैद्यकीय मदत घ्या — ER ला जा किंवा डॉक्टरांना कॉल करा", "bn": "জরুরী চিকিৎসা সেবা নিন", "te": "అత్యవసర వైద్య సంరక్షణ పొందండి"},
        {"en": "Do not ignore worsening symptoms", "hi": "बिगड़ते लक्षणों को नज़रअंदाज़ न करें", "mr": "वाढत्या लक्षणांकडे दुर्लक्ष करू नका", "bn": "অবনতিশীল লক্ষণগুলি এড়িয়ে যাবেন না", "te": "తీవ్రమవుతున్న లక్షణాలను విస్మరించవద్దు"},
    ],
    "EMERGENCY": [
        {"en": "Call 108 (emergency ambulance) IMMEDIATELY", "hi": "तुरंत 108 (एम्बुलेंस) पर कॉल करें", "mr": "तातडीने 108 (ॲम्ब्युलन्स) वर कॉल करा", "bn": "অবিলম্বে 108 কল করুন", "te": "వెంటనే 108 కి కాల్ చేయండి"},
        {"en": "If available, go to the nearest Emergency Room", "hi": "यदि संभव हो तो निकटतम आपातकालीन कक्ष जाएं", "mr": "शक्य असल्यास जवळच्या आपत्कालीन कक्षात जा", "bn": "যদি সম্ভব হয়, কাছের জরুরী কক্ষে যান", "te": "వీలైతే, సమీప ఎమర్జెన్సీ గదికి వెళ్లండి"},
    ],
}


def detect_language(text: str) -> str:
    """Detect script or simple heuristics for language."""
    # This is a mock language detector for the synthetic generator.
    # In a real pipeline, we'd use fasttext language detector.
    text_lower = text.lower()
    
    # Check for indic scripts
    if re.search(r'[\u0980-\u09FF]', text): return "bn" # Bengali
    if re.search(r'[\u0C00-\u0C7F]', text): return "te" # Telugu
    if re.search(r'[\u0B80-\u0BFF]', text): return "ta" # Tamil
    if re.search(r'[\u0A80-\u0AFF]', text): return "gu" # Gujarati
    if re.search(r'[\u0C80-\u0CFF]', text): return "kn" # Kannada
    if re.search(r'[\u0D00-\u0D7F]', text): return "ml" # Malayalam
    if re.search(r'[\u0A00-\u0A7F]', text): return "pa" # Punjabi
    if re.search(r'[\u0900-\u097F]', text): return "hi" # Hindi/Marathi (Devanagari)
    
    # Check romanized heuristics
    hindi_markers = ["mein", "hai", "ko", "ka", "ki", "se", "nahi", "hoon", "raha", "rahi", "bahut", "mere"]
    if sum(1 for m in hindi_markers if f" {m} " in f" {text_lower} ") >= 2:
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


# --- Generative Model Integration ---
USE_GENERATIVE_MODEL = False  # Set to True once you finish QLoRA training on GPU

if USE_GENERATIVE_MODEL:
    try:
        from model.generative_qlora_net import GenerativeQLoRANet
        generative_model = GenerativeQLoRANet(use_4bit=True)
        print("[BOOT] Generative QLoRA Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load Generative QLoRA Model: {e}")
        USE_GENERATIVE_MODEL = False

@app.post("/api/triage")
async def triage(req: TriageRequest):
    """
    Core triage endpoint.
    If USE_GENERATIVE_MODEL is True, it queries the fine-tuned LLM which learned everything.
    Otherwise, it uses the hybrid Neural-Symbolic classification pipeline.
    """
    if USE_GENERATIVE_MODEL:
        result = generative_model.predict(req.text)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail="Generative model failed to produce valid JSON")
            
        return {
            "prediction": result["data"].get("severity", "UNKNOWN"),
            "model_prediction": result["data"].get("severity", "UNKNOWN"),
            "confidence": 0.99,
            "probabilities": {"LOW": 0.25, "MODERATE": 0.25, "HIGH": 0.25, "EMERGENCY": 0.25},
            "knowledge_graph_matches": [],
            "safety_overrides": [],
            "safety_flags": [f["flag"] for f in result["data"].get("safety_flags", [])],
            "safety_actions": [f["action"] for f in result["data"].get("safety_flags", [])],
            "session_id": req.session_id,
            "clinical_analysis": result["data"]
        }
    else:
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
