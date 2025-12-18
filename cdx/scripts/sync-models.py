#!/usr/bin/env python3
"""
CDX Model Registry Sync Script
Syncs LLaMA-Factory outputs with the CDX model registry.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Paths
CDX_MODELS_BASE = Path("/home/codex450/cdx/resources/models")
REGISTRY_FILE = CDX_MODELS_BASE / "MODEL-REGISTRY.json"
FINE_TUNING_DIR = CDX_MODELS_BASE / "fine-tuning"
LORA_DIR = FINE_TUNING_DIR / "lora"
MERGED_DIR = FINE_TUNING_DIR / "merged"


def load_registry():
    """Load the model registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {"models": {}, "registry_version": "1.0"}


def save_registry(registry):
    """Save the model registry."""
    registry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"Registry updated: {REGISTRY_FILE}")


def get_adapter_info(adapter_dir: Path):
    """Get information about a trained adapter."""
    info = {
        "path": str(adapter_dir),
        "downloaded": True,
        "use_cases": ["fine-tuned", "custom"],
    }

    # Check for config
    config_file = adapter_dir / "adapter_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            info["lora_rank"] = config.get("r", 16)
            info["lora_alpha"] = config.get("lora_alpha", 32)
            info["base_model"] = config.get("base_model_name_or_path", "unknown")

    # Check for training args
    args_file = adapter_dir / "training_args.bin"
    if args_file.exists():
        info["training_completed"] = True

    # Get size
    total_size = sum(f.stat().st_size for f in adapter_dir.glob("**/*") if f.is_file())
    info["size_gb"] = round(total_size / (1024**3), 2)

    return info


def scan_lora_adapters():
    """Scan for trained LoRA adapters."""
    adapters = {}

    if not LORA_DIR.exists():
        return adapters

    for adapter_dir in LORA_DIR.iterdir():
        if adapter_dir.is_dir():
            name = adapter_dir.name
            adapters[f"cdx-lora-{name}"] = get_adapter_info(adapter_dir)

    return adapters


def scan_merged_models():
    """Scan for merged models."""
    models = {}

    if not MERGED_DIR.exists():
        return models

    for model_dir in MERGED_DIR.iterdir():
        if model_dir.is_dir():
            name = model_dir.name
            # Check for model files
            has_model = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
            if has_model:
                total_size = sum(f.stat().st_size for f in model_dir.glob("**/*") if f.is_file())
                models[f"cdx-merged-{name}"] = {
                    "name": f"CDX Merged {name.title()}",
                    "path": str(model_dir),
                    "size_gb": round(total_size / (1024**3), 2),
                    "downloaded": True,
                    "use_cases": ["fine-tuned", "inference", "deployment"],
                    "notes": "Merged LoRA adapter into base model"
                }

    return models


def update_registry():
    """Update the model registry with fine-tuned models."""
    registry = load_registry()

    # Ensure fine-tuning section exists
    if "fine-tuning" not in registry["models"]:
        registry["models"]["fine-tuning"] = {
            "lora-adapters": {},
            "merged-models": {}
        }

    # Scan and update
    lora_adapters = scan_lora_adapters()
    merged_models = scan_merged_models()

    registry["models"]["fine-tuning"]["lora-adapters"] = lora_adapters
    registry["models"]["fine-tuning"]["merged-models"] = merged_models

    # Update stats
    total_ft = len(lora_adapters) + len(merged_models)
    if "download_stats" in registry:
        registry["download_stats"]["fine_tuned_models"] = total_ft

    save_registry(registry)

    # Print summary
    print("\n=== CDX Fine-Tuned Models ===")
    print(f"LoRA Adapters: {len(lora_adapters)}")
    for name, info in lora_adapters.items():
        print(f"  - {name}: {info.get('size_gb', 0):.2f} GB")

    print(f"\nMerged Models: {len(merged_models)}")
    for name, info in merged_models.items():
        print(f"  - {name}: {info.get('size_gb', 0):.2f} GB")


def list_models():
    """List all models in the registry."""
    registry = load_registry()

    print("\n=== CDX Model Registry ===\n")

    for category, models in registry.get("models", {}).items():
        print(f"📁 {category.upper()}")
        if isinstance(models, dict):
            for model_name, model_info in models.items():
                if isinstance(model_info, dict):
                    name = model_info.get("name", model_name)
                    downloaded = "✓" if model_info.get("downloaded") else "○"
                    size = model_info.get("size_gb", "?")
                    print(f"  {downloaded} {name} ({size} GB)")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "sync":
            update_registry()
        elif command == "list":
            list_models()
        else:
            print(f"Unknown command: {command}")
            print("Usage: sync-models.py [sync|list]")
    else:
        update_registry()
