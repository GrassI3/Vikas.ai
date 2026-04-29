import torch
import json
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "aaditya/OpenBioLLM-Llama3-8B"
ADAPTER_PATH = "sahaayak-qlora-model"  # Path where train_qlora.py saves adapters

class GenerativeQLoRANet:
    """
    Inference Engine for the QLoRA fine-tuned Generative Triage Model.
    
    Loads the base OpenBioLLM model in 4-bit, attaches the trained LoRA adapters,
    and runs inference to generate the structured JSON output.
    """
    
    def __init__(self, use_4bit: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # In a real environment, you'd only load this if on GPU.
        # It will fail on CPU with bitsandbytes.
        print("Loading Base Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Load base
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=use_4bit,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        print(f"Attaching LoRA Adapters from {ADAPTER_PATH}...")
        try:
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        except Exception as e:
            print(f"Warning: Could not load adapters from {ADAPTER_PATH}. Running Base model only. Error: {e}")
            self.model = base_model
            
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Run generation to produce the JSON clinical report."""
        
        system_prompt = (
            "You are Sahaayak, an AI medical triage assistant for India. "
            "Analyze the patient's symptoms and respond with a JSON object containing: "
            "severity, severity_label, severity_description, symptoms_detected, "
            "possible_causes, related_conditions, recommended_actions, and safety_flags. "
            "Always respond in the language the patient uses."
        )
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temp for deterministic JSON
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract only the generated text
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        try:
            # Clean up potential markdown formatting (```json ... ```)
            clean_text = generated_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
                
            data = json.loads(clean_text)
            return {"status": "success", "data": data, "raw": generated_text}
        except json.JSONDecodeError:
            return {
                "status": "error", 
                "error": "Failed to generate valid JSON", 
                "raw": generated_text,
                "data": {
                    "severity": "UNKNOWN",
                    "severity_label": "Error",
                    "symptoms_detected": [],
                    "possible_causes": [],
                    "related_conditions": [],
                    "recommended_actions": [],
                    "safety_flags": []
                }
            }
