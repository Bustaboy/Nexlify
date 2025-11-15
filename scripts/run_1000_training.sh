#!/bin/bash
###############################################################################
# Nexlify ML/RL 1000-Round Training Runner
# Quick start script for training the ML/RL agent
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "================================================================================"
echo "  🚀 Nexlify ML/RL 1000-Round Training"
echo "================================================================================"
echo -e "${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.11+ required, found Python $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check if in correct directory
if [ ! -f "scripts/train_ml_rl_1000_rounds.py" ]; then
    echo -e "${RED}Error: Please run this script from the Nexlify root directory${NC}"
    exit 1
fi

# Check/create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install --upgrade pip > /dev/null 2>&1

if ! pip show numpy > /dev/null 2>&1; then
    echo -e "${YELLOW}Installing dependencies (this may take a few minutes)...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p models/ml_rl_1000
mkdir -p logs
mkdir -p data
echo -e "${GREEN}✓ Directories ready${NC}"

# Parse command line arguments
AGENT_TYPE="adaptive"
BALANCE=10000
DATA_DAYS=180
SYMBOL="BTC/USDT"
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent-type)
            AGENT_TYPE="$2"
            shift 2
            ;;
        --balance)
            BALANCE="$2"
            shift 2
            ;;
        --data-days)
            DATA_DAYS="$2"
            shift 2
            ;;
        --symbol)
            SYMBOL="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --agent-type TYPE    Agent type: adaptive, ultra, basic (default: adaptive)"
            echo "  --balance AMOUNT     Initial balance (default: 10000)"
            echo "  --data-days DAYS     Days of historical data (default: 180)"
            echo "  --symbol SYMBOL      Trading symbol (default: BTC/USDT)"
            echo "  --resume FILE        Resume from checkpoint file"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Train with defaults"
            echo "  $0 --agent-type ultra --balance 50000"
            echo "  $0 --resume models/ml_rl_1000/checkpoint_ep500.pth"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "\n${BLUE}Training Configuration:${NC}"
echo -e "  Agent Type:     ${GREEN}$AGENT_TYPE${NC}"
echo -e "  Initial Balance: ${GREEN}\$$BALANCE${NC}"
echo -e "  Data Days:      ${GREEN}$DATA_DAYS${NC}"
echo -e "  Symbol:         ${GREEN}$SYMBOL${NC}"
if [ -n "$RESUME" ]; then
    echo -e "  Resume From:    ${GREEN}${RESUME#--resume }${NC}"
fi
echo ""

# Confirm before starting
read -p "Start training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled${NC}"
    exit 0
fi

# Start training
echo -e "\n${GREEN}Starting 1000-round training...${NC}\n"
echo -e "${YELLOW}Note: This will take several hours. Progress is saved every 50 episodes.${NC}"
echo -e "${YELLOW}You can safely interrupt with Ctrl+C and resume later.${NC}\n"

python3 scripts/train_ml_rl_1000_rounds.py \
    --agent-type "$AGENT_TYPE" \
    --balance "$BALANCE" \
    --data-days "$DATA_DAYS" \
    --symbol "$SYMBOL" \
    $RESUME

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}"
    echo "================================================================================"
    echo "  ✅ Training Completed Successfully!"
    echo "================================================================================"
    echo -e "${NC}"
    echo -e "Results available in: ${GREEN}models/ml_rl_1000/${NC}"
    echo -e "  - Best model: ${GREEN}best_model.pth${NC}"
    echo -e "  - Final model: ${GREEN}final_model_1000.pth${NC}"
    echo -e "  - Report: ${GREEN}training_report_1000.png${NC}"
    echo -e "  - Summary: ${GREEN}training_summary_1000.txt${NC}"
    echo ""
    echo -e "View the report:"
    echo -e "  ${BLUE}open models/ml_rl_1000/training_report_1000.png${NC}  (macOS)"
    echo -e "  ${BLUE}xdg-open models/ml_rl_1000/training_report_1000.png${NC}  (Linux)"
    echo ""
    echo -e "Read the summary:"
    echo -e "  ${BLUE}cat models/ml_rl_1000/training_summary_1000.txt${NC}"
    echo ""
else
    echo -e "\n${RED}"
    echo "================================================================================"
    echo "  ❌ Training Failed or Interrupted"
    echo "================================================================================"
    echo -e "${NC}"
    echo -e "Check logs: ${YELLOW}logs/ml_rl_1000_training.log${NC}"
    echo ""
    if [ -f "models/ml_rl_1000/checkpoint_ep50.pth" ]; then
        echo -e "Progress was saved. Resume with:"
        LATEST_CHECKPOINT=$(ls -t models/ml_rl_1000/checkpoint_ep*.pth 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo -e "  ${BLUE}$0 --resume $LATEST_CHECKPOINT${NC}"
        fi
    fi
    exit 1
fi

# Deactivate virtual environment
deactivate
