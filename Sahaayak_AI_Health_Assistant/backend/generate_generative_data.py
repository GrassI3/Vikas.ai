import os
import json
import random
from tqdm import tqdm

from main import build_clinical_analysis, load_model, DEVICE
from model.data_loader import load_symptom_diagnosis
from knowledge_graph.resolver import KnowledgeGraphResolver

OUTPUT_FILE = "generative_training_data.jsonl"
NUM_SAMPLES = 500  # We'll generate a small subset for demonstration

def generate_synthetic_dataset():
    print("[1/3] Loading base dataset to use as seed...")
    df = load_symptom_diagnosis()
    
    kg_resolver = KnowledgeGraphResolver()
    
    # We only need the text inputs to run through our current logic to generate the "ground truth" JSON
    texts = df['text'].tolist()
    random.shuffle(texts)
    texts = texts[:NUM_SAMPLES]
    
    print("[2/3] Loading current model to act as a teacher...")
    model = load_model()
    model.eval()
    
    print(f"[3/3] Generating structured JSON for {len(texts)} samples...")
    
    dataset = []
    
    for text in tqdm(texts):
        # 1. Run the current pipeline
        kg_result = kg_resolver.process(text)
        model_result = model.predict(text, device=DEVICE)
        
        final_prediction = model_result["prediction"]
        severity_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "EMERGENCY": 3}
        
        kg_sev = kg_result.get("kg_suggested_severity", "LOW")
        if severity_rank.get(kg_sev, 0) > severity_rank.get(final_prediction, 0):
            final_prediction = kg_sev
            
        overrides = kg_result.get("safety_overrides", [])
        if overrides:
            final_prediction = "EMERGENCY"
            
        # 2. Build the rich analysis (this is what we want the new model to output)
        analysis = build_clinical_analysis(kg_result, model_result, final_prediction, text)
        
        # We don't want the model to predict the exact "model_probabilities" float values, 
        # that's impossible. We just want it to output the text/logic.
        del analysis["model_probabilities"]
        
        # 3. Create the instruction tuning pair
        dataset.append({
            "instruction": "Analyze the following medical symptoms and provide a structured clinical assessment in JSON format.",
            "input": text,
            "output": json.dumps(analysis, ensure_ascii=False)
        })
        
    # Write to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\nDone! Wrote {len(dataset)} examples to {OUTPUT_FILE}")
    print("Example output target:")
    print(json.dumps(json.loads(dataset[0]["output"]), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate_synthetic_dataset()
