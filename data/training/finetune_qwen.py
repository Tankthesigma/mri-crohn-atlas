"""
CrohnBridge Parser Fine-tuning Script
=====================================
Fine-tunes Qwen2.5-0.5B-Instruct on MRI report parsing using LoRA.

Base model: Qwen/Qwen2.5-0.5B-Instruct
Dataset: sigmsisgam/crohnbridge-training
Output: sigmsisgam/crohnbridge-parser-v1

For ZeroGPU Spaces, wrap with @spaces.GPU decorator.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
import os

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "sigmsisgam/crohnbridge-training"
OUTPUT_MODEL = "sigmsisgam/crohnbridge-parser-v1"
MAX_LENGTH = 2048
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

def format_chat(example, tokenizer):
    """Format messages into chat template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def tokenize(example, tokenizer):
    """Tokenize the formatted text."""
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    print("=" * 60)
    print("CrohnBridge Parser Fine-tuning")
    print("=" * 60)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Training will be very slow!")

    # Load tokenizer
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Loaded {len(dataset)} training examples")

    # Format and tokenize
    print("\nFormatting and tokenizing...")
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names
    )
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=["text"]
    )

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    # Load model
    print(f"\nLoading model {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # Rank
        lora_alpha=32,                 # Alpha scaling
        lora_dropout=0.05,             # Dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./crohnbridge-checkpoints",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=device == "cuda",
        report_to="none",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,
        hub_strategy="end",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save and push
    print("\nSaving and pushing to HuggingFace...")
    trainer.save_model()
    trainer.push_to_hub()

    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to {OUTPUT_MODEL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
