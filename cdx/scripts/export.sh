#!/bin/bash
# CDX LLaMA-Factory Export/Merge Script
# Usage: ./export.sh <config-name> [output-path]
# Example: ./export.sh llama3-lora-sft
# Example: ./export.sh llama3-lora-sft /path/to/merged/model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDX_DIR="$(dirname "$SCRIPT_DIR")"
LLAMAFACTORY_DIR="$(dirname "$CDX_DIR")"

# Default export directory
DEFAULT_EXPORT_DIR="/home/codex450/cdx/resources/models/fine-tuning/merged"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       CDX LLaMA-Factory Model Export (Merge LoRA)          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <config-name> [output-path]${NC}"
    echo -e "Example: $0 llama3-lora-sft"
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_FILE="$CDX_DIR/config/${CONFIG_NAME}.yaml"
CUSTOM_OUTPUT="${2:-}"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Activate virtual environment
if [ -f "$LLAMAFACTORY_DIR/venv/bin/activate" ]; then
    source "$LLAMAFACTORY_DIR/venv/bin/activate"
fi

# Get adapter path from config
ADAPTER_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')

# Check if adapter exists
if [ ! -d "$ADAPTER_DIR" ] || [ -z "$(ls -A "$ADAPTER_DIR" 2>/dev/null)" ]; then
    echo -e "${RED}Error: No trained adapter found at: $ADAPTER_DIR${NC}"
    echo -e "Please run training first: ./train.sh $CONFIG_NAME"
    exit 1
fi

# Determine export path
if [ -n "$CUSTOM_OUTPUT" ]; then
    EXPORT_DIR="$CUSTOM_OUTPUT"
else
    EXPORT_DIR="$DEFAULT_EXPORT_DIR/$CONFIG_NAME"
fi

echo -e "${BLUE}Export Configuration:${NC}"
echo -e "  Source adapter: ${GREEN}$ADAPTER_DIR${NC}"
echo -e "  Export to: ${GREEN}$EXPORT_DIR${NC}"

# Create export directory
mkdir -p "$EXPORT_DIR"

# Create temporary export config
EXPORT_CONFIG=$(mktemp)
cat > "$EXPORT_CONFIG" << EOF
### model
model_name_or_path: $(grep "model_name_or_path:" "$CONFIG_FILE" | awk '{print $2}')
adapter_name_or_path: $ADAPTER_DIR
template: $(grep "template:" "$CONFIG_FILE" | awk '{print $2}')
trust_remote_code: true

### export
export_dir: $EXPORT_DIR
export_size: 2
export_device: auto
export_quantization_bit: null
export_legacy_format: false
EOF

echo -e "\n${GREEN}Merging LoRA adapter into base model...${NC}\n"

cd "$LLAMAFACTORY_DIR"
llamafactory-cli export "$EXPORT_CONFIG"

# Cleanup
rm -f "$EXPORT_CONFIG"

echo -e "\n${GREEN}Export completed!${NC}"
echo -e "Merged model saved to: $EXPORT_DIR"

# Update model registry
echo -e "\n${YELLOW}Remember to update the model registry if deploying this model.${NC}"
