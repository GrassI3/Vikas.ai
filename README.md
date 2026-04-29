# 🏥 Vikas.ai — Explainable, Voice-Driven Decision Support Assistant

An AI-powered telephonic assistant designed to provide **explainable, evidence-based** health and civic guidance to underserved populations in India — accessible via a **standard phone call**.

## 🎯 What It Does

1. **Just a Phone Call** — No apps, no internet required. Dial a number and speak.
2. **Speaks Your Language** — Supports 12+ Indian languages via the Bhashini API pipeline.
3. **Truthful & Cited** — Uses RAG (Retrieval-Augmented Generation) over verified medical databases. Cites sources aloud.
4. **Safe** — Detects emergencies, injects hard-coded disclaimers, and provides crisis helpline numbers.
5. **Explainable** — Multi-agent Chain-of-Thought reasoning, fully traceable through the LangGraph execution graph.

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────┐     ┌────────────────────────────┐
│  User Phone  │────▶│   Vapi AI   │────▶│  FastAPI Backend Server    │
│  (PSTN/VoIP) │◀────│  (Voice     │◀────│                            │
└──────────────┘     │   Gateway)  │     │  ┌────────────────────────┐│
                     └─────────────┘     │  │  LangGraph Pipeline    ││
                                         │  │  ┌──────────────────┐  ││
                                         │  │  │ 1. Intake/Triage │  ││
                                         │  │  │ 2. RAG Retrieval │  ││
                                         │  │  │ 3. CoT Reasoning │  ││
                                         │  │  │ 4. Synthesis     │  ││
                                         │  │  └──────────────────┘  ││
                                         │  └────────────────────────┘│
                                         │  ┌────────┐ ┌───────────┐  │
                                         │  │ChromaDB│ │ Bhashini  │  │
                                         │  │ (RAG)  │ │(Translate)│  │
                                         │  └────────┘ └───────────┘  │
                                         │  ┌────────────────────────┐│
                                         │  │  Safety Guardrails     ││
                                         │  └────────────────────────┘│
                                         └────────────────────────────┘
```

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
```

### 5. Test with the API
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a severe headache and fever", "language": "en"}'
```

## 📁 Project Structure

```
Vikas.ai/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Environment & API key management
│   ├── agents/
│   │   ├── state.py             # Shared state schema (AgentState)
│   │   ├── nodes.py             # LangGraph nodes (intake, retrieval, reasoning, synthesis)
│   │   └── graph.py             # LangGraph state machine wiring
│   ├── knowledge/
│   │   ├── vector_db.py         # ChromaDB integration & semantic search
│   │   └── ingest.py            # Document ingestion (seed data + JSON files)
│   ├── telephony/
│   │   └── vapi_handler.py      # Vapi AI webhook dispatcher
│   └── utils/
│       ├── guardrails.py        # Safety checks & emergency disclaimers
│       ├── bhashini.py          # Bhashini multilingual pipeline
│       └── sms_fallback.py      # SMS/USSD fallback for low-bandwidth environments
├── frontend/
│   ├── index.html               # Dashboard HTML shell
│   ├── index.css                # Dashboard styling
│   └── app.js                   # Dashboard logic & API integration
├── .env.example
├── requirements.txt
└── README.md
```

## 🔑 API Keys Required

| Service | Purpose | Get Key |
|---------|---------|---------|
| **OpenAI** | LLM reasoning + embeddings | [platform.openai.com](https://platform.openai.com) |
| **Vapi AI** | Telephony voice interface | [vapi.ai](https://vapi.ai) |
| **Bhashini** | Indian language translation | [bhashini.gov.in](https://bhashini.gov.in) |

## 🛡️ Safety Features

- **Keyword-based emergency detection** — scans for suicide, cardiac, and trauma markers
- **Hard-coded disclaimers** — override LLM output during emergencies
- **Output validation guardrails** — block direct diagnosis or prescription language
- **Indian emergency contacts** — 112, 108 (ambulance), iCall, Vandrevala Foundation

## 📜 License

MIT