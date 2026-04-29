# TECHNICAL_ARCH.md — Sahaayak-Core Triage Engine

## System Overview

Sahaayak-Core is a **hybrid neural-symbolic** medical triage engine. It combines a custom-trained PyTorch classification model with a rule-based safety layer and a cultural-idiom Knowledge Graph.

**Zero LLM wrappers.** The core inference is a 3-layer classification head trained on top of a frozen IndicBERTv2 encoder.

## Module Map

```
backend/
├── main.py                        # FastAPI server — 5 endpoints
├── train.py                       # Full training pipeline with early stopping
├── retrain.py                     # Dynamic retraining from user feedback
├── requirements.txt
│
├── model/
│   ├── triage_net.py              # SahaayakTriageNet (PyTorch nn.Module)
│   │   ├── Encoder: ai4bharat/IndicBERTv2-MLM-only (FROZEN)
│   │   ├── Head: Linear(768→256→128→4) + ReLU + Dropout
│   │   └── Helpers: predict(), save/load checkpoint
│   ├── data_loader.py             # HuggingFace dataset loading + preprocessing
│   │   ├── ESI→Sahaayak severity mapping
│   │   ├── 3 dataset sources + feedback ingestion
│   │   └── Stratified train/val/test splits + class weights
│   └── xai_engine.py              # Captum LayerIntegratedGradients wrapper
│       └── Per-token attribution scores for explainability
│
├── knowledge_graph/
│   ├── graph.jsonld                # 13 conditions × Hindi/Marathi idioms (JSON-LD)
│   ├── safety_rules.json          # 8 emergency override rules (bilingual keywords)
│   └── resolver.py                # Fuzzy-match idiom resolution + safety gate
│
├── db/
│   └── feedback.sqlite            # User correction storage (auto-created)
│
└── checkpoints/                   # Versioned model weights (auto-created)
    ├── best_model.pt
    └── v{N}.pt
```

## Data Flow

```
User Input (potentially in Hindi/Marathi)
        │
        ▼
┌─ Knowledge Graph Resolver ─────────────────────────┐
│  1. Fuzzy-match idioms → medical conditions         │
│  2. Check Symbolic Safety Gate for overrides         │
│  3. Output: normalized text + KG matches + flags     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ SahaayakTriageNet ────────────────────────────────┐
│  1. Tokenize with IndicBERTv2 tokenizer             │
│  2. Extract [CLS] embedding (768-d)                 │
│  3. Classify via 3-layer head → 4-class logits       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Merge Layer ──────────────────────────────────────┐
│  final = max(model_pred, kg_severity, safety_gate)  │
│  Safety overrides are ABSOLUTE (always EMERGENCY)   │
└─────────────────────────────────────────────────────┘
        │
        ▼
   Response JSON (prediction + confidence + XAI + KG matches)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/triage` | Core classification. Accepts `text`, returns merged prediction. |
| `POST` | `/api/xai` | Standalone XAI. Returns per-token attribution scores. |
| `POST` | `/api/feedback` | Store user corrections in SQLite for retraining. |
| `POST` | `/api/reload` | Hot-swap model checkpoint without server restart. |
| `GET`  | `/api/health` | System status, device info, feedback counts. |

## Severity Classes

| Index | Label | Color | IPHS Mapping |
|-------|-------|-------|--------------|
| 0 | `LOW` | 🟢 Green | ESI 4-5 |
| 1 | `MODERATE` | 🟡 Yellow | ESI 3 |
| 2 | `HIGH` | 🟠 Orange | ESI 2 |
| 3 | `EMERGENCY` | 🔴 Red | ESI 1 |

## Knowledge Graph Schema

Format: **JSON-LD** (lightweight, no database dependency).

Each condition node contains:
- `med:clinicalName` — Standardized medical name
- `med:defaultSeverity` — Baseline severity
- `indic:idioms[]` — Array of `{term, lang, literal, context}`
- `med:escalationTriggers[]` — Conditions that upgrade severity
- `med:relatedSymptoms[]` — Graph edges to related conditions

Resolver uses `rapidfuzz` with an 80% similarity threshold for fuzzy matching.

## Retraining Pipeline

```
User corrects AI → POST /api/feedback → SQLite
                                            │
                        python retrain.py ◀─┘
                              │
                    Load pending feedback
                    Append to training set
                    Fine-tune head (N epochs)
                    Save v{N+1}.pt
                    Mark feedback as consumed
                              │
                    POST /api/reload → Hot-swap
```

## Safety Architecture

The **Symbolic Safety Gate** is the final arbiter. It cannot be overridden by the neural model. 8 rules cover:
- Cardiac emergencies (chest pain)
- Stroke (FAST protocol)
- Loss of consciousness
- Suicidal ideation (with crisis hotlines)
- Severe hemorrhage
- Respiratory failure
- Seizures
- Anaphylaxis

Each rule includes bilingual keywords (English + Hindi transliteration).

## Explainability (XAI)

Uses **Captum `LayerIntegratedGradients`** targeting the IndicBERTv2 embedding layer.

Output per prediction:
- Per-token attribution scores (normalized 0-1)
- Top 10 contributing tokens with direction (positive/negative)
- Raw logits and class probabilities

This feeds the React Flow visualization on the frontend `/map` page.
