#!/bin/bash
# CDX LLaMA-Factory Chat Script
# Usage: ./chat.sh <config-name>
# Example: ./chat.sh llama3-lora-sft

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CDX_DIR="$(dirname "$SCRIPT_DIR")"
LLAMAFACTORY_DIR="$(dirname "$CDX_DIR")"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           CDX LLaMA-Factory Interactive Chat               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check config argument
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <config-name>${NC}"
    echo -e "Example: $0 llama3-lora-sft"
    echo ""
    echo -e "Available configurations:"
    ls -1 "$CDX_DIR/config/"*.yaml | xargs -n1 basename | sed 's/.yaml$//'
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_FILE="$CDX_DIR/config/${CONFIG_NAME}.yaml"

# Activate virtual environment if it exists
if [ -f "$LLAMAFACTORY_DIR/venv/bin/activate" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$LLAMAFACTORY_DIR/venv/bin/activate"
fi

# Check for trained adapter
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo -e "${GREEN}Found trained adapter at: $OUTPUT_DIR${NC}"
    ADAPTER_ARG="adapter_name_or_path=$OUTPUT_DIR"
else
    echo -e "${BLUE}No trained adapter found, using base model only${NC}"
    ADAPTER_ARG=""
fi

cd "$LLAMAFACTORY_DIR"

echo -e "\n${GREEN}Starting chat session...${NC}"
echo -e "Type 'exit' or 'quit' to end the session.\n"

llamafactory-cli chat "$CONFIG_FILE" $ADAPTER_ARG
