"""
XAI Engine — Explainability via Captum Integrated Gradients

Provides per-token attribution scores for triage predictions so the
frontend can render the Visual Reasoning Map.
"""

import torch
from transformers import AutoTokenizer
from captum.attr import LayerIntegratedGradients
from typing import Optional

from model.triage_net import SahaayakTriageNet, INDICBERT_MODEL_NAME, MAX_SEQ_LENGTH, SEVERITY_CLASSES


class XAIEngine:
    """
    Wraps Captum's LayerIntegratedGradients to explain triage predictions.

    Computes per-token importance scores by integrating gradients from the
    IndicBERTv2 embedding layer to the predicted class logit.
    """

    def __init__(self, model: SahaayakTriageNet, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(INDICBERT_MODEL_NAME)

        # Target the encoder's word embedding layer
        self.lig = LayerIntegratedGradients(
            self._forward_for_captum,
            self.model.encoder.embeddings.word_embeddings,
        )

    def _forward_for_captum(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward function compatible with Captum's expected signature."""
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits

    def explain(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> dict:
        """
        Generate per-token attributions for a given input text.

        Parameters
        ----------
        text : str
            The symptom description to explain.
        target_class : int, optional
            Class index to explain. If None, uses the predicted class.
        n_steps : int
            Number of interpolation steps for Integrated Gradients.

        Returns
        -------
        dict with keys:
            - prediction: str
            - confidence: float
            - tokens: list[str]
            - attributions: list[float]  (normalized to sum to 1.0)
            - top_contributors: list[dict]
            - raw_logits: list[float]
        """
        self.model.eval()

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get prediction first
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()

        if target_class is None:
            target_class = pred_idx

        # Create baseline (all PAD tokens)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        # Compute attributions
        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=False,
        )

        # Aggregate across embedding dimension → per-token scalar
        # attributions shape: (1, seq_len, embed_dim)
        token_attributions = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)

        # Get actual tokens (strip padding)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
        actual_length = attention_mask.sum().item()
        tokens = tokens[:actual_length]
        token_attrs = token_attributions[:actual_length].cpu().detach().tolist()

        # Filter out special tokens
        filtered = [
            (tok, attr) for tok, attr in zip(tokens, token_attrs)
            if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
        ]

        if not filtered:
            filtered = [("(empty)", 0.0)]

        f_tokens, f_attrs = zip(*filtered)
        f_tokens = list(f_tokens)
        f_attrs = list(f_attrs)

        # Normalize attributions to [0, 1]
        abs_attrs = [abs(a) for a in f_attrs]
        max_attr = max(abs_attrs) if max(abs_attrs) > 0 else 1.0
        normalized = [round(a / max_attr, 4) for a in abs_attrs]

        # Build top contributors (sorted by absolute importance)
        indexed = sorted(
            enumerate(zip(f_tokens, f_attrs, normalized)),
            key=lambda x: abs(x[1][1]),
            reverse=True,
        )
        top_contributors = [
            {
                "token": tok,
                "weight": round(norm, 4),
                "direction": "positive" if raw >= 0 else "negative",
            }
            for _, (tok, raw, norm) in indexed[:10]
        ]

        return {
            "prediction": SEVERITY_CLASSES[pred_idx],
            "prediction_idx": pred_idx,
            "confidence": round(probs[0, pred_idx].item(), 4),
            "tokens": f_tokens,
            "attributions": normalized,
            "top_contributors": top_contributors,
            "raw_logits": logits[0].cpu().tolist(),
            "probabilities": {
                cls: round(p, 4)
                for cls, p in zip(SEVERITY_CLASSES, probs[0].cpu().tolist())
            },
        }
