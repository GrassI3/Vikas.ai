# 🏥 Vikas.ai — Explainable, Voice-Driven Decision Support Assistant

An AI-powered **telephonic assistant** designed to provide **explainable, evidence-based** health and civic guidance to underserved populations in India — accessible via a **standard phone call**. No app. No internet. No literacy required.

> **PS #10 — Health · Social Impact**
> *AI Assistant for Underserved Decision-Making*

---

## 🎯 What It Does

1. **Just a Phone Call** — Dial a number from any phone (feature phone, smartphone, landline) and speak. Zero tech barriers.
2. **Speaks Your Language** — Supports **8+ Indian languages** natively: English, Hindi, Marathi, Tamil, Telugu, Malayalam, Punjabi, Bengali. Auto-detects and responds in the caller's language.
3. **Truthful & Cited** — Uses **RAG** (Retrieval-Augmented Generation) over PubMed peer-reviewed databases and verified medical sources. Cites research aloud during the call.
4. **Explainable** — Full **Chain-of-Thought** reasoning. The caller understands *why* the AI reached its conclusion — not a black box.
5. **Safe** — Detects emergencies via keyword scanning, injects hard-coded disclaimers, provides crisis helpline numbers, and blocks specific prescription dosages.
6. **Resilient** — If the call drops, the pending response is delivered via **SMS fallback** automatically.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────────────────────────┐
│  User Phone  │────▶│  Vapi AI        │────▶│  FastAPI Backend (Python)          │
│  (PSTN/VoIP) │◀────│  + Twilio       │◀────│                                    │
└──────────────┘     │                 │     │  ┌────────────────────────────────┐ │
                     │  Orchestration  │     │  │  LangGraph Multi-Agent Engine  │ │
                     └─────────────────┘     │  │  ┌──────────────────────────┐  │ │
                             │               │  │  │ 1. Intake & Triage       │  │ │
                     ┌───────┴───────┐       │  │  │ 2. RAG Retrieval         │  │ │
                     │               │       │  │  │ 3. Chain-of-Thought      │  │ │
              ┌──────┴─────┐  ┌──────┴─────┐ │  │  │ 4. Synthesis & Citation  │  │ │
              │  Deepgram  │  │ ElevenLabs │ │  │  └──────────────────────────┘  │ │
              │  Nova 3    │  │ Vikram V2  │ │  └────────────────────────────────┘ │
              │  (ASR)     │  │ (TTS)      │ │  ┌──────────┐ ┌─────────────────┐  │
              └────────────┘  └────────────┘ │  │ ChromaDB │ │ PubMed Live     │  │
                                             │  │ (RAG)    │ │ Evidence Fetch  │  │
                                             │  └──────────┘ └─────────────────┘  │
                                             │  ┌────────────────────────────────┐ │
                                             │  │  Safety Guardrails & SMS      │ │
                                             │  └────────────────────────────────┘ │
                                             └────────────────────────────────────┘
```

### Data Flow
1. **User calls** → Vapi/Twilio phone number connects
2. **Voice → Text** → Deepgram Nova 3 (multilingual ASR, auto language detection)
3. **Text → LLM** → Groq LLama 3.1 8B Instant processes via LangGraph pipeline
4. **RAG enrichment** → ChromaDB + live PubMed fetch for evidence grounding
5. **LLM → Voice** → ElevenLabs Multilingual V2 (Vikram voice) speaks response
6. **Safety layer** → Emergency keywords trigger hard-coded disclaimers at every stage

---

## 📞 Phone Numbers

| Number | Provider | Features |
|--------|----------|----------|
| **+1 (631) 490-9141** | Vapi | Primary line · SMS OTP Auth |
| **+1 (938) 902-2543** | Twilio | SMS OTP Auth · Recording |
| **+1 (516) 667-0818** | Vapi | Backup line |
| **+1 (989) 935-6642** | Vapi | Backup line |

> **SMS Auth**: Numbers +1 (631) and +1 (938) require Twilio SMS OTP authentication to access response recordings and call transcripts.

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd Vikas.ai
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Seed the Knowledge Base
```bash
python -m backend.knowledge.ingest --seed
```

### 4. Start the Server
```bash
python -m backend.main
# Server runs on http://localhost:8000
# ngrok tunnel auto-starts for Vapi webhook
```

### 5. Test via API
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a severe headache and fever", "language": "en"}'
```

### 6. Test via Phone
Call any of the numbers listed above and speak naturally in your preferred language.

---

## 📁 Project Structure

```
Vikas.ai/
├── backend/
│   ├── main.py                  # FastAPI entry point + ngrok auto-tunnel
│   ├── config.py                # Environment & API key management
│   ├── agents/
│   │   ├── state.py             # Shared state schema (AgentState)
│   │   ├── nodes.py             # LangGraph nodes (intake, retrieval, reasoning, synthesis)
│   │   └── graph.py             # LangGraph state machine wiring
│   ├── knowledge/
│   │   ├── vector_db.py         # ChromaDB integration & semantic search
│   │   ├── ingest.py            # Document ingestion (seed data + JSON files)
│   │   └── pubmed.py            # Live PubMed abstract fetching & ingestion
│   ├── telephony/
│   │   └── vapi_handler.py      # Vapi AI webhook dispatcher & assistant config
│   └── utils/
│       ├── guardrails.py        # Safety checks & emergency disclaimers
│       └── sms_fallback.py      # SMS/USSD fallback for dropped calls
├── frontend/
│   ├── index.html               # Dashboard HTML shell
│   ├── index.css                # Dashboard styling
│   └── app.js                   # Dashboard logic & API integration
├── .env                         # API keys and configuration
├── requirements.txt             # Python dependencies
├── test_pipeline.py             # End-to-end pipeline test script
├── create_presentation.py       # Hackathon PPTX generator
└── README.md
```

---

## 🧠 Multi-Agent Reasoning Pipeline (LangGraph)

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  1. INTAKE &    │  Domain detection, severity classification,
                    │     TRIAGE      │  sentiment analysis, emergency keyword scan
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  Emergency?     │
                    └─┬─────────────┬─┘
                  YES │             │ NO
                      ▼             ▼
           ┌──────────────┐  ┌─────────────────┐
           │  Hard-coded  │  │  2. RETRIEVAL &  │  Semantic search ChromaDB,
           │  Disclaimer  │  │     GROUNDING    │  dynamic PubMed auto-fetch
           │  + Helplines │  └────────┬────────┘
           └──────────────┘           ▼
                             ┌─────────────────┐
                             │  3. CHAIN-OF-    │  Differential diagnosis,
                             │     THOUGHT      │  hypothesis ranking,
                             │     REASONING    │  confidence scoring
                             └────────┬────────┘
                                      ▼
                             ┌─────────────────┐
                             │  4. SYNTHESIS &  │  Evidence-based response,
                             │     CITATION     │  source attribution,
                             │                  │  safety disclaimers
                             └────────┬────────┘
                                      ▼
                             ┌─────────────────┐
                             │  5. GUARDRAILS   │  Prescription blocking,
                             │                  │  output validation
                             └─────────────────┘
```

---

## 🔑 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Telephony** | Vapi AI | Voice assistant orchestration & call management |
| **Phone Numbers** | Vapi + Twilio | PSTN phone numbers, SMS OTP authentication |
| **Speech-to-Text** | Deepgram Nova 3 | Multilingual ASR with auto language detection |
| **Text-to-Speech** | ElevenLabs Multilingual V2 | Vikram voice — natural Hindi/English/regional TTS |
| **LLM Inference** | Groq (LLama 3.1 8B Instant) | Ultra-fast reasoning and response generation |
| **Agent Framework** | LangGraph | Multi-agent state machine with conditional routing |
| **Vector Database** | ChromaDB | Local semantic search for RAG retrieval |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Document and query embedding |
| **Medical Evidence** | PubMed API | Live peer-reviewed abstract fetching |
| **Backend** | FastAPI + Uvicorn | Async Python web server |
| **Tunneling** | ngrok | Secure webhook tunneling for Vapi |
| **SMS Fallback** | Twilio SMS API | Deliver responses when calls drop |

---

## 🌐 Supported Languages

| Language | Native Script | Status |
|----------|--------------|--------|
| English | English | ✅ Fully supported |
| Hindi | हिन्दी | ✅ Fully supported |
| Marathi | मराठी | ✅ Fully supported |
| Tamil | தமிழ் | ✅ Supported |
| Telugu | తెలుగు | ✅ Supported |
| Malayalam | മലയാളം | ✅ Supported |
| Punjabi | ਪੰਜਾਬੀ | ✅ Supported |
| Bengali | বাংলা | ✅ Supported |

> Vikas auto-detects the caller's language and responds natively. Code-mixing (e.g., Hinglish, Marathi-English) is supported naturally.

---

## 🛡️ Safety Features

- **Emergency keyword scanning** — Detects suicide, cardiac, trauma, and respiratory distress markers in real-time
- **Hard-coded emergency disclaimers** — Override LLM output completely during emergencies with verified helpline numbers
- **Output validation guardrails** — Block specific prescription dosages from being spoken
- **Live mental health intervention** — Guided breathing exercises and panic de-escalation during the call
- **Indian emergency contacts** — 112 (national), 108 (ambulance), iCall, Vandrevala Foundation, NIMHANS

---

## 🔐 Authentication

- **SMS OTP Auth** is enabled on the +1 (631) and +1 (938) numbers
- Callers receive a Twilio SMS OTP to verify their identity
- Authenticated sessions unlock access to:
  - Call recording playback
  - Full conversation transcripts
  - Response history

---

## 🧪 Testing

### Run the pipeline test
```bash
python test_pipeline.py
```

### Test via the web dashboard
1. Start the server: `python -m backend.main`
2. Open `http://localhost:8000` in your browser
3. Use the "Test Query" tab to send messages through the pipeline
4. Inspect reasoning chains, citations, and safety analysis in the "Pipeline Inspector"

---

## 📊 PS #10 Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|---------------|
| Conversational or form-based input flow | ✅ | Voice telephony + web dashboard |
| Reasoning engine with ranked recommendations | ✅ | LangGraph 4-node CoT pipeline |
| Safety disclaimer & resource-referral layer | ✅ | Emergency guardrails + helpline routing |
| Explainable output (not a black box) | ✅ | Full reasoning chain with cited steps |
| Risk disclaimer logic | ✅ | Severity-based disclaimer injection |
| Human-readable insights with next actions | ✅ | Actionable medical/civic guidance |
| **Bonus:** Multilingual support | ✅ | 8+ Indian languages via Deepgram + ElevenLabs |
| **Bonus:** Voice input and output | ✅ | Full telephonic voice assistant |

---

## 📜 License

MIT