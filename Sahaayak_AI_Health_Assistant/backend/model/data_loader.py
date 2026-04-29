"""
Data loader & preprocessing for the Sahaayak Triage Engine.

Handles:
  - Loading HuggingFace datasets (fedmml-ed-triage, symptom_to_diagnosis, medical-triage-500)
  - ESI → Sahaayak severity mapping
  - Tokenization with IndicBERTv2
  - Train/Val/Test splitting
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
import pandas as pd
import json
import os

from model.triage_net import INDICBERT_MODEL_NAME, MAX_SEQ_LENGTH, SEVERITY_CLASSES


# ---------------------------------------------------------------------------
# ESI → Sahaayak Mapping
# ---------------------------------------------------------------------------

ESI_TO_SAHAAYAK = {
    1: "EMERGENCY",   # Resuscitation
    2: "HIGH",        # Emergent
    3: "MODERATE",    # Urgent
    4: "LOW",         # Less Urgent
    5: "LOW",         # Non-urgent
}

LABEL_TO_IDX = {label: idx for idx, label in enumerate(SEVERITY_CLASSES)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Triage Dataset
# ---------------------------------------------------------------------------

class TriageDataset(Dataset):
    """PyTorch Dataset wrapping tokenized symptom texts + severity labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

def load_fedmml_triage() -> pd.DataFrame:
    """
    Load the fedmml-ed-triage dataset (87k+ records) and map ESI → Sahaayak severity.

    Expected columns after processing: ['text', 'label']
    """
    try:
        ds = load_dataset("olaflaitinen/fedmml-ed-triage", split="train")
        df = ds.to_pandas()

        # The dataset has ESI levels — find the relevant columns
        # Common column names: 'chief_complaint', 'esi', 'triage_level'
        text_col = None
        esi_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "complaint" in col_lower or "symptom" in col_lower or "description" in col_lower or "text" in col_lower:
                text_col = col
            if "esi" in col_lower or "triage" in col_lower or "level" in col_lower or "severity" in col_lower:
                esi_col = col

        if text_col is None:
            # Fallback: concatenate all string columns
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if str_cols:
                text_col = str_cols[0]

        if esi_col is None:
            # Fallback: look for numeric columns that could be ESI
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            for col in num_cols:
                if df[col].min() >= 1 and df[col].max() <= 5:
                    esi_col = col
                    break

        if text_col is None or esi_col is None:
            print(f"[WARN] Could not auto-detect columns. Available: {df.columns.tolist()}")
            print(f"       Detected text_col={text_col}, esi_col={esi_col}")
            return pd.DataFrame(columns=["text", "label"])

        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str)
        result["label"] = df[esi_col].apply(
            lambda x: LABEL_TO_IDX.get(ESI_TO_SAHAAYAK.get(int(x), "LOW"), 0)
        )

        print(f"[INFO] Loaded fedmml-ed-triage: {len(result)} samples")
        return result

    except Exception as e:
        print(f"[WARN] Failed to load fedmml-ed-triage: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_symptom_diagnosis() -> pd.DataFrame:
    """
    Load the symptom-to-diagnosis dataset and assign severity heuristically.
    """
    try:
        ds = load_dataset("gretelai/symptom_to_diagnosis", split="train")
        df = ds.to_pandas()

        # Heuristic severity assignment based on diagnosis keywords
        emergency_keywords = ["heart", "stroke", "sepsis", "hemorrhage", "anaphylaxis"]
        high_keywords = ["pneumonia", "diabetes", "hepatitis", "malaria", "dengue"]
        moderate_keywords = ["flu", "bronchitis", "infection", "gastro", "migraine"]

        def assign_severity(diagnosis: str) -> int:
            d = diagnosis.lower()
            if any(k in d for k in emergency_keywords):
                return LABEL_TO_IDX["EMERGENCY"]
            elif any(k in d for k in high_keywords):
                return LABEL_TO_IDX["HIGH"]
            elif any(k in d for k in moderate_keywords):
                return LABEL_TO_IDX["MODERATE"]
            return LABEL_TO_IDX["LOW"]

        text_col = "text" if "text" in df.columns else df.columns[0]
        diag_col = "output_text" if "output_text" in df.columns else (
            "label" if "label" in df.columns else df.columns[-1]
        )

        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str)
        result["label"] = df[diag_col].apply(assign_severity)

        print(f"[INFO] Loaded symptom_to_diagnosis: {len(result)} samples")
        return result

    except Exception as e:
        print(f"[WARN] Failed to load symptom_to_diagnosis: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_medical_triage_500() -> pd.DataFrame:
    """
    Load the medical-triage-500 dataset (for validation).
    """
    try:
        ds = load_dataset("syntech-ai/medical-triage-500", split="train")
        df = ds.to_pandas()

        # Map urgency levels to our schema
        urgency_map = {
            "non-urgent": LABEL_TO_IDX["LOW"],
            "low": LABEL_TO_IDX["LOW"],
            "moderate": LABEL_TO_IDX["MODERATE"],
            "medium": LABEL_TO_IDX["MODERATE"],
            "high": LABEL_TO_IDX["HIGH"],
            "urgent": LABEL_TO_IDX["HIGH"],
            "critical": LABEL_TO_IDX["EMERGENCY"],
            "emergency": LABEL_TO_IDX["EMERGENCY"],
        }

        text_col = None
        urgency_col = None
        for col in df.columns:
            cl = col.lower()
            if "symptom" in cl or "complaint" in cl or "text" in cl or "description" in cl:
                text_col = col
            if "urgency" in cl or "severity" in cl or "triage" in cl or "level" in cl:
                urgency_col = col

        if text_col is None:
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            text_col = str_cols[0] if str_cols else None
        if urgency_col is None:
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            urgency_col = str_cols[-1] if len(str_cols) > 1 else None

        if text_col is None or urgency_col is None:
            print(f"[WARN] Could not auto-detect columns in medical-triage-500")
            return pd.DataFrame(columns=["text", "label"])

        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str)
        result["label"] = df[urgency_col].apply(
            lambda x: urgency_map.get(str(x).strip().lower(), LABEL_TO_IDX["LOW"])
        )

        print(f"[INFO] Loaded medical-triage-500: {len(result)} samples")
        return result

    except Exception as e:
        print(f"[WARN] Failed to load medical-triage-500: {e}")
        return pd.DataFrame(columns=["text", "label"])


def load_feedback_data(db_path: str = "db/feedback.sqlite") -> pd.DataFrame:
    """Load user-corrected feedback from SQLite for retraining."""
    import sqlite3

    if not os.path.exists(db_path):
        return pd.DataFrame(columns=["text", "label"])

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT input_text AS text, user_correction AS label_name FROM feedback WHERE used_in_epoch IS NULL",
            conn,
        )
        if df.empty:
            return pd.DataFrame(columns=["text", "label"])

        df["label"] = df["label_name"].apply(lambda x: LABEL_TO_IDX.get(x, 0))
        print(f"[INFO] Loaded {len(df)} feedback samples for retraining")
        return df[["text", "label"]]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Combined Data Pipeline
# ---------------------------------------------------------------------------

def build_datasets(
    include_feedback: bool = False,
    feedback_db_path: str = "db/feedback.sqlite",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[TriageDataset, TriageDataset, TriageDataset, dict]:
    """
    Build train/val/test TriageDatasets from all sources.

    Returns
    -------
    train_dataset, val_dataset, test_dataset, class_weights_dict
    """
    tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)

    # Load all sources
    dfs = [
        load_fedmml_triage(),
        load_symptom_diagnosis(),
    ]

    if include_feedback:
        dfs.append(load_feedback_data(feedback_db_path))

    # Concatenate
    combined = pd.concat([df for df in dfs if not df.empty], ignore_index=True)
    combined = combined.dropna(subset=["text", "label"])
    combined["label"] = combined["label"].astype(int)

    print(f"\n[INFO] Combined dataset: {len(combined)} total samples")
    print(f"[INFO] Class distribution:\n{combined['label'].value_counts().sort_index()}\n")

    # Stratified split
    texts = combined["text"].tolist()
    labels = combined["label"].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=(val_ratio + test_ratio), random_state=seed, stratify=labels
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=relative_test, random_state=seed, stratify=temp_labels
    )

    print(f"[INFO] Split — Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Compute class weights for imbalanced data
    from collections import Counter
    counts = Counter(train_labels)
    total = sum(counts.values())
    class_weights = {
        cls: total / (len(counts) * count) for cls, count in counts.items()
    }
    print(f"[INFO] Class weights: {class_weights}")

    return (
        TriageDataset(train_texts, train_labels, tokenizer),
        TriageDataset(val_texts, val_labels, tokenizer),
        TriageDataset(test_texts, test_labels, tokenizer),
        class_weights,
    )


def get_dataloaders(
    batch_size: int = 32,
    include_feedback: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Convenience wrapper that returns DataLoaders."""
    train_ds, val_ds, test_ds, weights = build_datasets(include_feedback=include_feedback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, weights
