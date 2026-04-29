import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# We are using OpenBioLLM as it is the current SOTA open medical model.
# Note: For production Indian language usage, you could also swap this for
# "ai4bharat/Airavata" if you find it lacks language nuances.
MODEL_NAME = "aaditya/OpenBioLLM-Llama3-8B"
DATASET_PATH = "full_training_data.jsonl"
OUTPUT_DIR = "sahaayak-qlora-model"

def format_instruction(example):
    """
    Format the JSONL row into a standard Llama-3 instruction prompt format.
    """
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{example['system']}<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['input']}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )
    return {"text": prompt}

def main():
    print("="*50)
    print(" Sahaayak Triage Engine - QLoRA Fine-tuning")
    print("="*50)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for QLoRA training. Please run on a cloud GPU instance.")
        
    print("[1/5] Loading Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(format_instruction)
    
    # Split for validation
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    
    print(f"Loaded {len(train_data)} training and {len(val_data)} validation samples.")

    print(f"[2/5] Configuring 4-bit Quantization (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"[3/5] Loading Base Model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False
    )
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("[4/5] Setting up SFT Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("[5/5] Starting Training...")
    trainer.train()

    print("Saving the fine-tuned LoRA adapters...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! You can now deploy this adapter with the base OpenBioLLM model.")

if __name__ == "__main__":
    main()
