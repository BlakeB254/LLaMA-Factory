#!/bin/bash
# CDX LLaMA-Factory Setup Script for DGX Spark
# This script sets up the complete environment for fine-tuning on DGX Spark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CDX LLaMA-Factory Setup for NVIDIA DGX Spark            ║${NC}"
echo -e "${BLUE}║                   Blackwell GB10 Architecture                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Check prerequisites
echo -e "\n${BLUE}[1/6] Checking prerequisites...${NC}"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ NVIDIA Driver:${NC}"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# Create virtual environment
echo -e "\n${BLUE}[2/6] Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${BLUE}[3/6] Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo -e "\n${BLUE}[4/6] Installing PyTorch with CUDA 12.x support...${NC}"
echo -e "${YELLOW}Note: Using PyTorch for CUDA 12.x (compatible with CUDA 13.0)${NC}"

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install LLaMA-Factory
echo -e "\n${BLUE}[5/6] Installing LLaMA-Factory...${NC}"
cd "$SCRIPT_DIR"
pip install -e ".[torch,metrics]" --no-build-isolation

# Install additional dependencies for DGX Spark optimization
echo -e "\n${BLUE}[6/6] Installing additional dependencies...${NC}"
pip install \
    tensorboard \
    wandb \
    datasets \
    accelerate \
    peft \
    trl \
    transformers \
    sentencepiece \
    protobuf \
    bitsandbytes \
    scipy

# Optionally install flash-attn (if compatible)
echo -e "${YELLOW}Attempting to install flash-attn (may fail on some systems)...${NC}"
pip install flash-attn --no-build-isolation 2>/dev/null || echo -e "${YELLOW}flash-attn installation skipped (not critical)${NC}"

# Make scripts executable
echo -e "\n${BLUE}Making scripts executable...${NC}"
chmod +x "$SCRIPT_DIR/cdx/scripts/"*.sh
chmod +x "$SCRIPT_DIR/cdx/scripts/"*.py

# Create output directories
echo -e "\n${BLUE}Creating output directories...${NC}"
mkdir -p /home/codex450/cdx/resources/models/fine-tuning/lora
mkdir -p /home/codex450/cdx/resources/models/fine-tuning/merged
mkdir -p "$SCRIPT_DIR/cdx/outputs/checkpoints"
mkdir -p "$SCRIPT_DIR/cdx/outputs/logs"

# Test installation
echo -e "\n${BLUE}Testing installation...${NC}"
llamafactory-cli version || echo -e "${YELLOW}Version check not available${NC}"

# Final summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "${BLUE}Activation:${NC}"
echo -e "  source $VENV_DIR/bin/activate"
echo -e ""
echo -e "${BLUE}Quick Start:${NC}"
echo -e "  1. Train: ./cdx/scripts/train.sh llama3-lora-sft"
echo -e "  2. Chat:  ./cdx/scripts/chat.sh llama3-lora-sft"
echo -e "  3. Export: ./cdx/scripts/export.sh llama3-lora-sft"
echo -e ""
echo -e "${BLUE}Web UI:${NC}"
echo -e "  llamafactory-cli webui"
echo -e ""
echo -e "${BLUE}Available training configs:${NC}"
ls -1 "$SCRIPT_DIR/cdx/config/"*.yaml | xargs -n1 basename | sed 's/.yaml$//' | sed 's/^/  - /'
echo -e ""
echo -e "${YELLOW}Note: For gated models (Llama), run: huggingface-cli login${NC}"
