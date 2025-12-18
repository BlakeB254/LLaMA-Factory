#!/bin/bash
# CDX LLaMA-Factory Training Script
# Usage: ./train.sh <config-name> [additional-args]
# Example: ./train.sh llama3-lora-sft
# Example: ./train.sh llama3-lora-sft learning_rate=1e-5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDX_DIR="$(dirname "$SCRIPT_DIR")"
LLAMAFACTORY_DIR="$(dirname "$CDX_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           CDX LLaMA-Factory Training Pipeline              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check config argument
if [ -z "$1" ]; then
    echo -e "${YELLOW}Available configurations:${NC}"
    ls -1 "$CDX_DIR/config/"*.yaml | xargs -n1 basename | sed 's/.yaml$//'
    echo ""
    echo -e "${RED}Usage: $0 <config-name> [additional-args]${NC}"
    echo -e "Example: $0 llama3-lora-sft"
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_FILE="$CDX_DIR/config/${CONFIG_NAME}.yaml"
shift  # Remove first argument, rest are passed to llamafactory-cli

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Available configurations:${NC}"
    ls -1 "$CDX_DIR/config/"*.yaml | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

# Activate virtual environment if it exists
if [ -f "$LLAMAFACTORY_DIR/venv/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$LLAMAFACTORY_DIR/venv/bin/activate"
fi

# Check GPU availability
echo -e "${BLUE}Checking GPU status...${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Clear buffer cache for DGX Spark Unified Memory optimization
echo -e "${YELLOW}Optimizing memory for DGX Spark...${NC}"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# Display training config
echo -e "${BLUE}Training Configuration:${NC}"
echo -e "  Config: ${GREEN}$CONFIG_NAME${NC}"
echo -e "  File: $CONFIG_FILE"
echo -e "  Additional args: $@"

# Create output directories
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo -e "  Output: $OUTPUT_DIR"
fi

# Run training
echo -e "\n${GREEN}Starting training...${NC}\n"
cd "$LLAMAFACTORY_DIR"

llamafactory-cli train "$CONFIG_FILE" "$@"

echo -e "\n${GREEN}Training completed!${NC}"
echo -e "Checkpoints saved to: $OUTPUT_DIR"
