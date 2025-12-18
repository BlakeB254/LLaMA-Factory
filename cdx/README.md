# CDX LLaMA-Factory Integration

This directory contains CDX Platform-specific configurations for LLaMA-Factory, optimized for NVIDIA DGX Spark (GB10 Blackwell architecture).

## Supported Training Types

| Type | Description | Example Models |
|------|-------------|----------------|
| **LLM LoRA** | Text-only language models | Llama 3, Mistral, Phi, Qwen, DeepSeek |
| **Multimodal LoRA** | Vision-language models (image+text) | Qwen-VL, LLaVA, InternVL |
| **Audio LoRA** | Speech/audio models | Qwen-Audio |

## Directory Structure

```
cdx/
├── README.md                      # This file
├── config/                        # Training configurations
│   ├── dgx-spark-base.yaml       # Base DGX Spark optimizations
│   │
│   │ # LLM Configs (text-only)
│   ├── llama3-lora-sft.yaml      # Llama 3.x LoRA SFT
│   ├── mistral-lora-sft.yaml     # Mistral LoRA SFT
│   ├── phi-lora-sft.yaml         # Phi LoRA SFT
│   ├── deepseek-lora-sft.yaml    # DeepSeek LoRA SFT
│   ├── qwen-lora-sft.yaml        # Qwen LoRA SFT
│   │
│   │ # Multimodal Configs (image+text)
│   ├── qwen-vl-lora-sft.yaml     # Qwen2.5-VL vision-language
│   ├── llava-lora-sft.yaml       # LLaVA vision-language
│   │
│   └── inference/                # Chat/inference configs
│       ├── llama3-chat.yaml
│       └── mistral-chat.yaml
│
├── datasets/                      # CDX custom datasets
│   ├── dataset_info.json         # Dataset registry
│   ├── cdx_identity.json         # CDX brand identity (text)
│   └── cdx_multimodal.json       # CDX multimodal (image+text)
│
├── scripts/                       # Utility scripts
│   ├── train.sh                  # Training launcher
│   ├── chat.sh                   # Interactive chat
│   ├── export.sh                 # Export/merge adapters
│   └── sync-models.py            # Model registry sync
│
└── outputs/                       # Training outputs (gitignored)
    ├── checkpoints/              # Model checkpoints
    └── merged/                   # Merged models
```

## Model Registry Integration

Models are stored in: `/home/codex450/cdx/resources/models/`
- LLMs: `/home/codex450/cdx/resources/models/llm/`
- Fine-tuned LoRAs: `/home/codex450/cdx/resources/models/fine-tuning/lora/`
- Merged outputs: `/home/codex450/cdx/resources/models/fine-tuning/merged/`

## Quick Start

```bash
# Activate environment
source /home/codex450/ai-tools/LLaMA-Factory/venv/bin/activate

# Run training
./cdx/scripts/train.sh llama3-lora-sft

# Chat with trained model
./cdx/scripts/chat.sh llama3-lora-sft

# Export merged model
./cdx/scripts/export.sh llama3-lora-sft
```

## DGX Spark Optimizations

- **CUDA 13.0** with Blackwell architecture support
- **bf16** precision for optimal performance
- **Unified Memory Architecture** aware configurations
- **FlashAttention-2** enabled where supported
- Optimized batch sizes for GB10 memory

## Hardware Specs

- GPU: NVIDIA GB10 (Blackwell)
- Driver: 580.95.05
- CUDA: 13.0
- Memory: Unified Memory Architecture

## Training Examples

### LLM Training (Text-only)
```bash
# Train Llama 3 with CDX identity
./cdx/scripts/train.sh llama3-lora-sft

# Train Mistral for coding
./cdx/scripts/train.sh deepseek-lora-sft
```

### Multimodal Training (Vision-Language)
```bash
# Train Qwen-VL to understand images
./cdx/scripts/train.sh qwen-vl-lora-sft

# Train LLaVA for visual QA
./cdx/scripts/train.sh llava-lora-sft
```

### Web UI (All-in-One)
```bash
# Launch graphical interface
llamafactory-cli webui
```
The Web UI provides dropdown menus for:
- Model selection
- LoRA configuration
- Dataset selection
- Training parameters

## Note on Image LoRA (SDXL/Flux)

LLaMA-Factory trains **language model** LoRAs (text and vision-language).

For **image generation LoRAs** (like Dini's SDXL LoRA), use:
- **Kohya SS** - Popular SDXL/Flux LoRA trainer
- **ai-toolkit** - SimpleTuner for Flux
- **ComfyUI training nodes**

The Dini LoRA (`dini_thirst_trap_rank32_fp16.safetensors`) was trained for SDXL image generation, which is a different type of model.
