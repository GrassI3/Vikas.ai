import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

from model.generative_net import DEFAULT_GENERATIVE_MODEL, MAX_SEQ_LENGTH, MAX_NEW_TOKENS

DATA_FILE = "generative_training_data.jsonl"
OUTPUT_DIR = "generative_checkpoints"

def prepare_dataset(tokenizer):
    """Load JSONL data and tokenize for Seq2Seq."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    dataset = Dataset.from_list(data)
    
    def preprocess_function(examples):
        # inputs are instruction + input
        inputs = [f"{inst} {inp}" for inst, inp in zip(examples["instruction"], examples["input"])]
        targets = examples["output"]
        
        model_inputs = tokenizer(inputs, max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_NEW_TOKENS, padding="max_length", truncation=True)
            
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    # Split
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset["train"], split_dataset["test"]


def train():
    print(f"Loading {DEFAULT_GENERATIVE_MODEL}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_GENERATIVE_MODEL, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_GENERATIVE_MODEL, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_GENERATIVE_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_GENERATIVE_MODEL)
        
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(tokenizer)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Use mixed precision if on GPU
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training (Warning: This is incredibly slow on CPU!)...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}/best_model")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print("Done!")

if __name__ == "__main__":
    train()
