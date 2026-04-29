# Sahaayak-Core Triage Engine — Architecture & Training Plan

## 1. Mission Statement

Build a **bespoke, hybrid neural-symbolic triage engine** from scratch. No LLM wrappers. The system classifies patient distress into four severity tiers (`Low`, `Moderate`, `High`, `Emergency`) using a custom-trained PyTorch model, enriched by a cultural-idiom Knowledge Graph and served via FastAPI with full XAI explainability.

---

## 2. Neural Network Architecture

### 2.1 Model: `SahaayakTriageNet` — Dual-Encoder with Symbolic Gate

The architecture is a **two-stage hybrid** that combines a neural classifier with a symbolic rule engine.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SahaayakTriageNet v1                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌──────────────────┐                     │
│  │ User Input    │───▶│ Knowledge Graph   │──▶ Normalized      │
│  │ (Raw Text)    │    │ Idiom Resolver    │    Medical Text    │
│  └───────────────┘    └──────────────────┘                     │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌──────────────────────────────────────────────────────┐      │
│  │        IndicBERTv2 Encoder (Frozen / LoRA)           │      │
│  │        ai4bharat/IndicBERTv2-MLM-only                │      │
│  │        (24 Indian Languages + English)                │      │
│  └──────────────────────────────────────────────────────┘      │
│                          │                                      │
│                    [CLS] Embedding (768-d)                      │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Classification Head                      │      │
│  │  Linear(768, 256) → ReLU → Dropout(0.3)              │      │
│  │  Linear(256, 128)  → ReLU → Dropout(0.2)             │      │
│  │  Linear(128, 4)    → (Logits)                         │      │
│  └──────────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │           Symbolic Safety Gate                        │      │
│  │  Rule-based override for critical keywords:           │      │
│  │  "chest pain" / "seene mein dard" → EMERGENCY         │      │
│  │  "unconscious" / "behosh" → EMERGENCY                 │      │
│  │  "suicide" / "aatmhatya" → EMERGENCY + CRISIS FLAG    │      │
│  └──────────────────────────────────────────────────────┘      │
│                          │                                      │
│                          ▼                                      │
│              Final Prediction + Logit Map                       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Architecture?

| Decision | Rationale |
|---|---|
| **IndicBERTv2 over DistilBERT** | IndicBERTv2 is pre-trained on 20.9B tokens across 24 Indian languages (IndicCorp v2). DistilBERT only covers English. This is non-negotiable for Hindi/Marathi/regional input. |
| **Frozen encoder + trainable head** | We lack millions of medical samples. Fine-tuning the full encoder risks catastrophic forgetting. We freeze the encoder and only train the classification head (transfer learning). |
| **Symbolic Safety Gate** | Neural models can misclassify edge cases. A hard-coded rule layer guarantees that critical symptoms (chest pain, stroke signs, suicidal ideation) always escalate to `Emergency`, regardless of model confidence. This is a **patient safety non-negotiable**. |
| **No CNN-LSTM** | After research, a transformer encoder (IndicBERTv2) outperforms CNN-LSTM on text classification benchmarks for Indian languages. CNN-LSTM would require custom tokenization per language. |

### 2.3 Severity Classes (Ground Truth)

Based on the **Indian Public Health Standards (IPHS) 2022** triage color-coding and MoHFW operational guidelines:

| Class | Color Code | Description | Examples |
|---|---|---|---|
| `LOW` | 🟢 Green | Minor / Non-urgent | Mild headache, common cold, minor rash |
| `MODERATE` | 🟡 Yellow | Urgent, not life-threatening | Persistent fever >24hrs, moderate abdominal pain, dizziness |
| `HIGH` | 🟠 Orange | Emergent, potential deterioration | High fever with altered consciousness, severe dehydration, fracture |
| `EMERGENCY` | 🔴 Red | Resuscitation / Life-threatening | Chest pain, stroke symptoms, uncontrolled bleeding, unconsciousness |

---

## 3. Training Data Sources

### 3.1 Primary Datasets

| Dataset | Source | Size | Use |
|---|---|---|---|
| `olaflaitinen/fedmml-ed-triage` | HuggingFace | 87k+ records | Primary training data. ESI levels 1-5 mapped to our 4-class schema. |
| `gretelai/symptom_to_diagnosis` | HuggingFace | 1,000+ samples | Augmentation — symptom-description-to-diagnosis pairs. |
| `syntech-ai/medical-triage-500` | HuggingFace | 500 cases | Validation set — contains urgency levels and risk scores. |

### 3.2 ESI → Sahaayak Mapping

The `fedmml-ed-triage` dataset uses the Emergency Severity Index (ESI 1-5). We remap:

```
ESI 1 (Resuscitation)  → EMERGENCY
ESI 2 (Emergent)       → HIGH
ESI 3 (Urgent)         → MODERATE
ESI 4 (Less Urgent)    → LOW
ESI 5 (Non-urgent)     → LOW
```

### 3.3 Synthetic Augmentation

Generate ~2,000 additional samples by:
1. Translating English symptom descriptions to Hindi/Marathi using IndicTrans2.
2. Injecting cultural idioms from the Knowledge Graph (e.g., replace "anxiety" with "ghabrahat").
3. Adding noise variations (typos, colloquial phrasing) for robustness.

---

## 4. Knowledge Graph — Cultural Idiom Resolver

### 4.1 Structure: JSON-LD (No Neo4j Dependency)

We use a lightweight **JSON-LD** knowledge graph stored as a static file. This keeps the system zero-dependency and portable. If scale demands it later, we migrate to Neo4j.

```jsonld
{
  "@context": {
    "med": "https://sahaayak.ai/ontology/medical#",
    "indic": "https://sahaayak.ai/ontology/indic#"
  },
  "@graph": [
    {
      "@id": "med:Anxiety",
      "@type": "med:Condition",
      "med:clinicalName": "Generalized Anxiety Disorder",
      "med:defaultSeverity": "MODERATE",
      "indic:idioms": [
        { "indic:term": "ghabrahat", "indic:lang": "hi", "indic:literal": "घबराहट" },
        { "indic:term": "ghabra jaana", "indic:lang": "hi", "indic:literal": "घबरा जाना" },
        { "indic:term": "dhak dhak", "indic:lang": "hi", "indic:literal": "धक धक" }
      ],
      "med:relatedSymptoms": ["med:Palpitations", "med:ShortnessOfBreath"],
      "med:escalationTriggers": ["suicidal_ideation", "self_harm"]
    },
    {
      "@id": "med:ChestPain",
      "@type": "med:Condition",
      "med:clinicalName": "Acute Chest Pain",
      "med:defaultSeverity": "EMERGENCY",
      "indic:idioms": [
        { "indic:term": "seene mein dard", "indic:lang": "hi", "indic:literal": "सीने में दर्द" },
        { "indic:term": "chhati mein dard", "indic:lang": "hi", "indic:literal": "छाती में दर्द" }
      ]
    }
  ]
}
```

### 4.2 Idiom Resolution Pipeline

```
User: "Mere papa ko bahut ghabrahat ho rahi hai aur dhak dhak"
                              │
                              ▼
              Knowledge Graph Lookup (fuzzy match)
              "ghabrahat" → med:Anxiety
              "dhak dhak"  → med:Palpitations
                              │
                              ▼
              Normalized: "Patient experiencing anxiety and palpitations"
                              │
                              ▼
                    IndicBERTv2 Encoder
```

---

## 5. Explainability (XAI) — Captum Integration

### 5.1 Method: Layer Integrated Gradients

Using **PyTorch Captum** (`captum.attr.LayerIntegratedGradients`):

1. Target the IndicBERTv2 embedding layer.
2. Baseline: all-`[PAD]` token sequence.
3. Compute per-token attribution scores for the predicted class.
4. Aggregate across embedding dimensions → scalar importance per token.
5. Return a `LogitMap` JSON object to the frontend.

### 5.2 XAI Response Schema

```json
{
  "prediction": "MODERATE",
  "confidence": 0.87,
  "logit_map": {
    "tokens": ["severe", "headache", "24", "hours", "dizzy"],
    "attributions": [0.31, 0.42, 0.15, 0.08, 0.28],
    "top_contributors": [
      { "token": "headache", "weight": 0.42, "direction": "positive" },
      { "token": "severe", "weight": 0.31, "direction": "positive" }
    ]
  },
  "symbolic_overrides": [],
  "knowledge_graph_matches": [
    { "idiom": "chakkar", "resolved_to": "Dizziness", "lang": "hi" }
  ]
}
```

This response feeds directly into the React Flow visualization on the `/map` page.

---

## 6. System Architecture — Full Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Sahaayak System Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐       │
│   │ Next.js  │────▶│  FastAPI      │────▶│  SahaayakTriageNet  │       │
│   │ Frontend │◀────│  /api/triage  │◀────│  (PyTorch Model)    │       │
│   │ :3000    │     │  :8000        │     └─────────────────────┘       │
│   └──────────┘     ├──────────────┤              │                     │
│                    │  /api/xai     │     ┌────────┴────────┐           │
│                    │  /api/feedback│     │  Captum XAI     │           │
│                    │  /api/health  │     │  Integrated     │           │
│                    └──────┬───────┘     │  Gradients      │           │
│                           │             └─────────────────┘           │
│                    ┌──────┴───────┐                                    │
│                    │   Middleware  │                                    │
│                    ├──────────────┤                                    │
│                    │ KG Resolver  │──▶ knowledge_graph.jsonld          │
│                    │ Safety Gate  │──▶ safety_rules.json               │
│                    │ Feedback DB  │──▶ feedback.sqlite                 │
│                    └──────────────┘                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Dynamic Retraining Pipeline

### 7.1 Feedback Collection

When a user corrects the AI's triage decision:

```
POST /api/feedback
{
  "session_id": "abc123",
  "original_input": "mere papa ko bahut ghabrahat ho rahi hai",
  "model_prediction": "MODERATE",
  "user_correction": "HIGH",
  "timestamp": "2026-04-29T20:00:00Z"
}
```

This is stored in a local **SQLite** database (`feedback.sqlite`):

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    model_prediction TEXT NOT NULL,
    user_correction TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    used_in_epoch INTEGER DEFAULT NULL
);
```

### 7.2 Retraining Trigger

A CLI command (`python retrain.py`) that:
1. Queries all feedback rows where `used_in_epoch IS NULL`.
2. Appends them to the training set with the corrected label.
3. Runs N additional fine-tuning epochs on the classification head only.
4. Saves a versioned checkpoint (`checkpoints/v{N}.pt`).
5. Marks processed feedback rows with the epoch number.
6. Hot-reloads the model in the FastAPI server via a `/api/reload` endpoint.

---

## 8. File Structure

```
Sahaayak_AI_Health_Assistant/
├── backend/
│   ├── main.py                  # FastAPI server (inference + XAI + feedback)
│   ├── train.py                 # Training script
│   ├── retrain.py               # Dynamic retraining pipeline
│   ├── model/
│   │   ├── triage_net.py        # SahaayakTriageNet PyTorch module
│   │   ├── data_loader.py       # Dataset loading & preprocessing
│   │   └── xai_engine.py        # Captum Integrated Gradients wrapper
│   ├── knowledge_graph/
│   │   ├── graph.jsonld          # Cultural idiom knowledge graph
│   │   ├── resolver.py          # Fuzzy-match idiom resolution
│   │   └── safety_rules.json    # Symbolic override rules
│   ├── db/
│   │   └── feedback.sqlite      # User correction feedback store
│   ├── checkpoints/             # Versioned model weights
│   ├── requirements.txt
│   └── TECHNICAL_ARCH.md
├── frontend/                    # (Existing Next.js app)
└── PLAN.md                      # This document
```

---

## 9. Dependencies

```
# Core ML
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
captum>=0.7.0
accelerate>=0.30.0

# Serving
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0

# Knowledge Graph
rapidfuzz>=3.9.0        # Fuzzy string matching for idiom resolution
pyld>=2.0.0             # JSON-LD processing

# Data & Storage
aiosqlite>=0.20.0       # Async SQLite for feedback
pandas>=2.2.0
scikit-learn>=1.5.0     # Metrics, train/test split

# Indic NLP
indicnlp                # Indic NLP Library for preprocessing
```

---

## 10. Training Hyperparameters (Initial)

| Parameter | Value | Notes |
|---|---|---|
| Base model | `ai4bharat/IndicBERTv2-MLM-only` | Frozen encoder |
| Max sequence length | 256 tokens | Sufficient for symptom descriptions |
| Batch size | 32 | |
| Learning rate | 2e-4 | AdamW, only for classification head |
| Epochs | 15 | With early stopping (patience=3) |
| Dropout | 0.3 / 0.2 | Per layer |
| Loss function | CrossEntropyLoss with class weights | To handle class imbalance |
| Optimizer | AdamW | Weight decay 0.01 |
| Scheduler | CosineAnnealingWarmRestarts | |
| Validation split | 80/10/10 (train/val/test) | Stratified |

---

## 11. Execution Order

| Step | Action | Blocker |
|---|---|---|
| **Step 1** | ✅ Create PLAN.md (this document) | **Awaiting your approval** |
| **Step 2** | Write `triage_net.py`, `train.py`, `main.py` | Approval of Step 1 |
| **Step 3** | Build Knowledge Graph (`graph.jsonld`, `resolver.py`, `safety_rules.json`) | None |
| **Step 4** | Write XAI engine (`xai_engine.py`) | None |
| **Step 5** | Build retraining pipeline (`retrain.py`, feedback SQLite schema) | None |
| **Step 6** | Write `TECHNICAL_ARCH.md` | After all code is written |
| **Step 7** | Wire FastAPI endpoints to Next.js frontend | After backend is stable |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **IndicBERTv2 model size (~1GB)** | Use `torch.compile()` + ONNX export for inference optimization. Keep encoder frozen to reduce VRAM. |
| **Training data is English-heavy** | Synthetic augmentation via translation + idiom injection. Validate on manually curated Hindi test set. |
| **Class imbalance (few EMERGENCY samples)** | Weighted CrossEntropyLoss + SMOTE-style oversampling for minority classes. |
| **Fuzzy matching false positives** | Set a minimum similarity threshold (85%) in the KG resolver. Log all resolutions for audit. |
| **Patient safety** | The Symbolic Safety Gate is the final layer — it cannot be overridden by model output. Critical keywords always escalate. |

---

> **STATUS: AWAITING APPROVAL — Please review and confirm before I proceed to Step 2.**
