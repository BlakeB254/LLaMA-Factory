#!/bin/bash
# CDX LLaMA-Factory Quick Start
# Run this script to set up and launch LLaMA-Factory

set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════"
echo "         CDX LLaMA-Factory Setup for DGX Spark"
echo "═══════════════════════════════════════════════════════════════"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment exists"
fi

# Activate venv
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Check if llamafactory is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "[3/4] Installing LLaMA-Factory and dependencies..."
    pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # Install LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation

    # Install additional deps
    pip install tensorboard accelerate peft trl transformers sentencepiece bitsandbytes gradio
else
    echo "[3/4] LLaMA-Factory already installed"
fi

# Make scripts executable
chmod +x cdx/scripts/*.sh cdx/scripts/*.py 2>/dev/null || true

echo "[4/4] Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To launch the Web UI, run:"
echo "  source venv/bin/activate"
echo "  llamafactory-cli webui"
echo ""
echo "Or use the training scripts:"
echo "  ./cdx/scripts/train.sh llama3-lora-sft"
echo ""
echo "Web UI will be available at: http://localhost:7063"
echo "═══════════════════════════════════════════════════════════════"

# Ask if user wants to launch WebUI
read -p "Launch Web UI now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching LLaMA-Factory Web UI on port 7063..."
    echo "(CDX AI/ML port range: 7060-7069)"
    GRADIO_SERVER_PORT=7063 llamafactory-cli webui
fi
