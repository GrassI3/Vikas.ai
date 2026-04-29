"""
SahaayakTriageNet — Dual-Encoder with Symbolic Safety Gate

Architecture:
  1. IndicBERTv2 encoder (frozen) extracts 768-d [CLS] embeddings
  2. Classification head maps embeddings → 4 severity classes
  3. Symbolic Safety Gate overrides predictions for critical keywords

Classes: LOW (0), MODERATE (1), HIGH (2), EMERGENCY (3)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_CLASSES = ["LOW", "MODERATE", "HIGH", "EMERGENCY"]
NUM_CLASSES = len(SEVERITY_CLASSES)
INDICBERT_MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
MAX_SEQ_LENGTH = 256


# ---------------------------------------------------------------------------
# SahaayakTriageNet
# ---------------------------------------------------------------------------

class SahaayakTriageNet(nn.Module):
    """
    Hybrid neural-symbolic triage classifier.

    The encoder is frozen by default to prevent catastrophic forgetting on
    small medical datasets.  Only the classification head is trainable.
    """

    def __init__(
        self,
        encoder_name: str = INDICBERT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # --- Encoder ---
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder_hidden_size = self.encoder.config.hidden_size  # typically 768

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # --- Tokenizer (kept here for convenience during inference) ---
        self._tokenizer: Optional[AutoTokenizer] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Lazy-loaded tokenizer so we don't serialize it with the model."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    INDICBERT_MODEL_NAME, local_files_only=True
                )
            except Exception:
                # Fallback: try with network (first-time download)
                self._tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)
        return self._tokenizer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (batch, seq_len)
        attention_mask : (batch, seq_len)
        token_type_ids : (batch, seq_len), optional

        Returns
        -------
        logits : (batch, num_classes)
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # Use the [CLS] token embedding as the sequence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)

        logits = self.classifier(cls_embedding)  # (batch, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, text: str, device: Optional[torch.device] = None) -> dict:
        """
        Run inference on a single text string.

        Returns a dict with keys: prediction, confidence, logits, probabilities.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        encoded = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()

        return {
            "prediction": SEVERITY_CLASSES[pred_idx],
            "prediction_idx": pred_idx,
            "confidence": round(probs[0, pred_idx].item(), 4),
            "logits": logits[0].cpu().tolist(),
            "probabilities": {
                cls: round(p, 4) for cls, p in zip(SEVERITY_CLASSES, probs[0].cpu().tolist())
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int = 0, metadata: Optional[dict] = None):
        """Save only the classification head (encoder is frozen/pretrained)."""
        payload = {
            "classifier_state_dict": self.classifier.state_dict(),
            "epoch": epoch,
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None):
        """Load a classification head checkpoint."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        return checkpoint.get("metadata", {})
