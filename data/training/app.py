"""
CrohnBridge Fine-tuning Space
=============================
Gradio app for fine-tuning on HuggingFace ZeroGPU.

Deploy this to a HuggingFace Space with ZeroGPU enabled.
"""

import gradio as gr
import spaces
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import gc
import os
import json

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "sigmsisgam/crohnbridge-training"
OUTPUT_MODEL = "sigmsisgam/crohnbridge-parser-v1"
MAX_LENGTH = 2048
CHECKPOINT_DIR = "/tmp/crohnbridge-checkpoint"
STATE_FILE = "/tmp/crohnbridge-state.json"

def load_state():
    """Load training state from file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"total_steps": 0, "runs": 0}

def save_state(state):
    """Save training state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def format_chat(example, tokenizer):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"], truncation=True, max_length=MAX_LENGTH, padding=False
    )
    result["labels"] = result["input_ids"].copy()
    return result

@spaces.GPU(duration=120)  # 2 minutes max per call
def train_model(epochs, learning_rate, batch_size, progress=gr.Progress()):
    """Fine-tune the model on ZeroGPU with checkpoint resumption."""

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)
        return "\n".join(logs)

    try:
        # Load state
        state = load_state()
        log(f"=== Training Run #{state['runs'] + 1} ===")
        log(f"Previous total steps: {state['total_steps']}")

        progress(0, desc="Loading tokenizer...")
        log(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        progress(0.1, desc="Loading dataset...")
        dataset = load_dataset(DATASET_ID, split="train")
        log(f"Loaded {len(dataset)} training examples")

        progress(0.2, desc="Tokenizing...")
        dataset = dataset.map(lambda x: format_chat(x, tokenizer), remove_columns=dataset.column_names)
        dataset = dataset.map(lambda x: tokenize(x, tokenizer), remove_columns=["text"])
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        log(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

        progress(0.3, desc="Loading model...")

        # Check if checkpoint exists - resume from it
        if os.path.exists(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "adapter_config.json")):
            log(f"📂 Found checkpoint, resuming training...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, CHECKPOINT_DIR, is_trainable=True)
            log(f"✓ Loaded LoRA weights from checkpoint")
        else:
            log(f"🆕 Starting fresh training...")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
            progress(0.4, desc="Applying LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Optimized for short GPU duration
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=int(batch_size),
            per_device_eval_batch_size=int(batch_size),
            gradient_accumulation_steps=2,
            learning_rate=float(learning_rate),
            weight_decay=0.01,
            warmup_steps=10,
            lr_scheduler_type="linear",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,
            report_to="none",
            push_to_hub=False,
            max_steps=50,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, return_tensors="pt"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        progress(0.5, desc="Training (50 steps)...")
        log(f"Starting training (steps {state['total_steps']} → {state['total_steps'] + 50})...")
        trainer.train()

        progress(0.9, desc="Saving checkpoint...")
        log("Saving checkpoint...")

        # Save to checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        model.save_pretrained(CHECKPOINT_DIR)
        tokenizer.save_pretrained(CHECKPOINT_DIR)

        # Update state
        state['total_steps'] += 50
        state['runs'] += 1
        save_state(state)

        log(f"\n✅ Training complete!")
        log(f"📊 Total steps so far: {state['total_steps']}")
        log(f"📊 Total runs: {state['runs']}")
        log(f"\n💡 Click 'Train' again to continue, or 'Push to Hub' when done.")

        # Cleanup
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        return "\n".join(logs)

    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}\n\n" + "\n".join(logs)

@spaces.GPU(duration=60)
def push_to_hub():
    """Push the trained model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        if not os.path.exists(CHECKPOINT_DIR):
            return "❌ No checkpoint found. Train the model first!"

        state = load_state()

        api = HfApi()
        api.upload_folder(
            folder_path=CHECKPOINT_DIR,
            repo_id=OUTPUT_MODEL,
            repo_type="model",
        )
        return f"✅ Model pushed to {OUTPUT_MODEL}\nTotal training steps: {state['total_steps']}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def reset_training():
    """Reset training state and delete checkpoint."""
    import shutil

    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    return "🔄 Training reset! Checkpoint deleted. Click 'Train' to start fresh."

def get_status():
    """Get current training status."""
    state = load_state()
    checkpoint_exists = os.path.exists(CHECKPOINT_DIR)

    status = f"📊 **Training Status**\n"
    status += f"- Total steps completed: {state['total_steps']}\n"
    status += f"- Training runs: {state['runs']}\n"
    status += f"- Checkpoint exists: {'✅ Yes' if checkpoint_exists else '❌ No'}\n"

    if state['total_steps'] > 0:
        # Estimate epochs (460 samples, batch 8, grad_accum 2 = ~29 steps/epoch)
        est_epochs = state['total_steps'] / 29
        status += f"- Estimated epochs: {est_epochs:.1f}\n"

    return status

@spaces.GPU(duration=60)
def test_inference(mri_report):
    """Test the fine-tuned model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )

        # Try local checkpoint first, then hub
        source = f"Base model ({BASE_MODEL})"
        if os.path.exists(CHECKPOINT_DIR):
            try:
                model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
                state = load_state()
                source = f"Local checkpoint ({state['total_steps']} steps)"
            except:
                pass
        else:
            try:
                model = PeftModel.from_pretrained(model, OUTPUT_MODEL)
                source = f"Hub model ({OUTPUT_MODEL})"
            except:
                pass

        system = "You are a medical AI that extracts Van Assche Index and MAGNIFI-CD scores from MRI radiology reports. Analyze the report and output JSON with vai_score, magnifi_score, extracted_features, and confidence."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": mri_report}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return f"Source: {source}\n\n{response}"

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks(title="CrohnBridge Fine-tuning") as demo:
    gr.Markdown("""
    # 🏥 CrohnBridge Parser Fine-tuning

    Fine-tune Qwen2.5-0.5B-Instruct on MRI report parsing using LoRA.

    - **Base Model:** Qwen/Qwen2.5-0.5B-Instruct
    - **Dataset:** sigmsisgam/crohnbridge-training (460 examples)
    - **Output:** sigmsisgam/crohnbridge-parser-v1
    """)

    with gr.Tab("🚀 Train"):
        gr.Markdown("""
        **Continuous Training:** Each click runs 50 steps. Progress is saved between runs.
        Click multiple times to accumulate training, then push to hub when done.
        """)

        status_box = gr.Markdown(value=get_status())
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(get_status, inputs=[], outputs=status_box)

        with gr.Row():
            epochs = gr.Number(value=1, label="Epochs", minimum=1, maximum=3, visible=False)
            lr = gr.Number(value=2e-4, label="Learning Rate")
            batch = gr.Number(value=8, label="Batch Size", minimum=1, maximum=16)

        with gr.Row():
            train_btn = gr.Button("▶️ Train (+50 steps)", variant="primary")
            push_btn = gr.Button("☁️ Push to Hub", variant="secondary")
            reset_btn = gr.Button("🗑️ Reset", variant="stop")

        output = gr.Textbox(label="Training Log", lines=20)

        train_btn.click(train_model, inputs=[epochs, lr, batch], outputs=output).then(
            get_status, inputs=[], outputs=status_box
        )
        push_btn.click(push_to_hub, inputs=[], outputs=output)
        reset_btn.click(reset_training, inputs=[], outputs=output).then(
            get_status, inputs=[], outputs=status_box
        )

    with gr.Tab("🔍 Test"):
        gr.Markdown("Test the model (uses local checkpoint if available, otherwise hub model):")
        report_input = gr.Textbox(
            label="MRI Report",
            placeholder="Paste MRI findings here...",
            lines=8
        )
        test_btn = gr.Button("Parse Report")
        result = gr.Textbox(label="Model Output", lines=10)
        test_btn.click(test_inference, inputs=report_input, outputs=result)

if __name__ == "__main__":
    demo.launch()
