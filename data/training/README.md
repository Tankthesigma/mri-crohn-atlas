---
title: CrohnBridge Fine-tuning
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# CrohnBridge Parser Fine-tuning

Fine-tune Qwen2.5-0.5B-Instruct on MRI report parsing for Van Assche Index and MAGNIFI-CD scoring.

## Configuration

- **Base Model:** Qwen/Qwen2.5-0.5B-Instruct (0.5B params)
- **Method:** LoRA (Low-Rank Adaptation)
- **Dataset:** sigmsisgam/crohnbridge-training (460 examples)
- **Output:** sigmsisgam/crohnbridge-parser-v1

## Training Parameters

- Epochs: 3
- Learning Rate: 2e-4
- Batch Size: 4
- Gradient Accumulation: 4
- LoRA Rank: 16
- LoRA Alpha: 32

## Usage

1. Go to the **Train** tab
2. Adjust hyperparameters if needed
3. Click **Start Training**
4. Wait for training to complete (~30 min on ZeroGPU)
5. Test the model in the **Test** tab
