import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

# Using a lightweight multilingual generative model
DEFAULT_GENERATIVE_MODEL = "google/mt5-small"
MAX_SEQ_LENGTH = 256
MAX_NEW_TOKENS = 512

class GenerativeTriageNet(nn.Module):
    """
    Sequence-to-Sequence (Generative) Triage Engine.
    
    Takes in a symptom description with an instruction prefix.
    Outputs a structured JSON string containing severity, descriptions, 
    and advice in the requested language.
    """
    
    def __init__(self, model_name: str = DEFAULT_GENERATIVE_MODEL):
        super().__init__()
        # We load both tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
    def predict(self, text: str, device: Optional[torch.device] = None) -> dict:
        """
        Run inference. The model generates a JSON string which we parse.
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        self.model.eval()
        
        instruction = "Analyze the following medical symptoms and provide a structured clinical assessment in JSON format: "
        prompt = f"{instruction}{text}"
        
        encoded = self.tokenizer(
            prompt,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True
            )
            
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        import json
        try:
            # Attempt to parse the generated string as JSON
            data = json.loads(generated_text)
            return {"status": "success", "data": data, "raw": generated_text}
        except json.JSONDecodeError:
            # Generative models can hallucinate invalid JSON
            return {
                "status": "error", 
                "error": "Failed to generate valid JSON", 
                "raw": generated_text,
                "data": {
                    "severity_label": "UNKNOWN",
                    "severity_description": "Failed to parse model output.",
                    "symptoms_detected": [],
                    "possible_causes": [],
                    "related_conditions": [],
                    "recommended_actions": [],
                    "model_probabilities": {"LOW": 0.25, "MODERATE": 0.25, "HIGH": 0.25, "EMERGENCY": 0.25}
                }
            }
