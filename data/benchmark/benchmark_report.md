# MRI Report Parser - Multi-Model Benchmark

**Date:** 2025-12-10

**Test Cases:** 30

## Summary Table

| Model | VAI MAE | MAGNIFI MAE | VAI Acc (±1) | VAI Acc (±2) | Avg Time | Cost/Case |
|-------|---------|-------------|--------------|--------------|----------|-----------|
| **qwen-2.5-72b** | **2.21** | 3.47 | **36.8%** | **52.6%** | 3.27s | $0.0001 |
| gpt-4o | 2.71 | **2.35** | 29.4% | 47.1% | **1.02s** | $0.0013 |
| claude-3.5-sonnet | 2.70 | 3.43 | 26.1% | 47.8% | 3.01s | $0.0022 |
| deepseek-v3 | 3.56 | 2.33 | 22.2% | 33.3% | 3.52s | $0.0001 |
| gemini-2.0-flash | 4.17 | 3.35 | 8.3% | 41.7% | 1.12s | $0.0001 |
| crohnbridge-v1 (50 steps) | 6.83 | 7.25 | 13.3% | 20.0% | 5.66s | **$0.00** |

## Winner Analysis

- **Best Accuracy (VAI MAE):** qwen-2.5-72b (2.21)
- **Best MAGNIFI Accuracy:** GPT-4o (2.35)
- **Fastest:** GPT-4o (1.02s)
- **Most Cost-Effective:** qwen-2.5-72b ($0.0001/case with best accuracy)
- **Free (Local):** crohnbridge-v1 (needs more training)

## Fine-Tuned Model Analysis

The CrohnBridge fine-tuned model (Qwen 0.5B + LoRA) underperforms because:

1. **Undertrained:** Only 50 training steps completed
   - Model outputs repetitive patterns (VAI: 7, 9, 10, 14)
   - Needs 200-500+ steps for convergence

2. **Model Size:** 0.5B vs 72B+ parameters
   - Small models require more training data and steps
   - May need increased LoRA rank (r=8 → r=16 or 32)

3. **Training Data:** 460 examples may be insufficient
   - Larger models leverage pre-trained medical knowledge

**Recommendation:** Continue training to 300+ steps or use Qwen 72B via API

## Model Details


### deepseek-v3

- Model ID: `deepseek/deepseek-chat`
- Valid cases: 18
- Total time: 105.6s
- Total cost: $0.0017
- Tokens used: 7360 in / 2459 out

### gpt-4o

- Model ID: `openai/gpt-4o`
- Valid cases: 17
- Total time: 30.6s
- Total cost: $0.0392
- Tokens used: 7652 in / 2003 out

### gemini-2.0-flash

- Model ID: `google/gemini-2.0-flash-001`
- Valid cases: 20
- Total time: 33.5s
- Total cost: $0.0017
- Tokens used: 7540 in / 2266 out

### claude-3.5-sonnet

- Model ID: `anthropic/claude-3.5-sonnet`
- Valid cases: 23
- Total time: 90.2s
- Total cost: $0.0668
- Tokens used: 8197 in / 2814 out

### qwen-2.5-72b

- Model ID: `qwen/qwen-2.5-72b-instruct`
- Valid cases: 19
- Total time: 98.2s
- Total cost: $0.0038
- Tokens used: 7896 in / 2618 out

## Methodology

- Each model received the same MRI reports with identical system prompts
- Temperature set to 0.1 for consistency
- Accuracy measured as percentage of scores within 1 or 2 points of ground truth
- MAE = Mean Absolute Error (lower is better)
- Cost based on OpenRouter pricing as of December 2024