#!/bin/bash
# CMB Phase Transitions Analysis - Docker Runner
# ===============================================
#
# This script runs the complete CMB phase transition analysis
# in a Docker container and outputs results to the local results/
# directory.
#
# Usage: ./run_docker_analysis.sh
#
# Requirements: Docker installed and running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and results path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE} CMB Phase Transitions Analysis - Docker Runner${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker from https://docker.com"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}Docker daemon not running, attempting to start...${NC}"

    # On macOS, try to start Docker Desktop
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/Applications/Docker.app" ]; then
            echo -e "${BLUE}Starting Docker Desktop...${NC}"
            open -a Docker
            echo -e "${YELLOW}Waiting for Docker to start (this may take 30-60 seconds)...${NC}"

            # Wait for Docker to be ready
            for i in {1..60}; do
                if docker info &> /dev/null; then
                    echo -e "${GREEN}✓ Docker is ready!${NC}"
                    break
                fi
                echo -n "."
                sleep 2
            done

            if ! docker info &> /dev/null; then
                echo -e "\n${RED}Error: Docker failed to start after 2 minutes${NC}"
                echo "Please start Docker Desktop manually and try again"
                exit 1
            fi
        else
            echo -e "${RED}Error: Docker Desktop not found${NC}"
            echo
            echo -e "${YELLOW}You have Docker CLI but need Docker Desktop for macOS${NC}"
            echo -e "${BLUE}Install Docker Desktop:${NC}"
            echo "  1. Visit: https://docs.docker.com/desktop/install/mac-install/"
            echo "  2. Download Docker Desktop for Mac"
            echo "  3. Install and start Docker Desktop"
            echo "  4. Run this script again"
            echo
            echo -e "${GREEN}Or run locally without Docker:${NC}"
            echo "  python3 main.py --gamma --bao --all-datasets --full-validation"
            exit 1
        fi
    else
        echo -e "${RED}Error: Docker daemon not running${NC}"
        echo "Please start Docker service and try again"
        exit 1
    fi
fi

# Create results directory if it doesn't exist
echo -e "${YELLOW}Setting up results directory...${NC}"
mkdir -p "${RESULTS_DIR}"

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
if ! docker build -t cmb-analysis "${SCRIPT_DIR}"; then
    echo -e "${RED}Error: Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker image built successfully${NC}"
echo

# Run the analysis
echo -e "${YELLOW}Running complete analysis (gamma + BAO validation)...${NC}"
echo -e "${YELLOW}This may take 15-20 minutes...${NC}"
echo

if docker run -v "${RESULTS_DIR}:/app/results" cmb-analysis; then
    echo
    echo -e "${GREEN}✓ Analysis completed successfully!${NC}"
    echo
    echo -e "${BLUE}Results saved to: ${RESULTS_DIR}${NC}"
    echo
    echo "Key output files:"
    echo "  - results/gamma_theoretical.json"
    echo "  - results/bao_multi_dataset_validation.json"
    echo "  - results/complete_statistical_validation.log"
    echo
    echo -e "${GREEN}🎉 Ready for publication!${NC}"
else
    echo
    echo -e "${RED}✗ Analysis failed${NC}"
    echo "Check the Docker logs above for details"
    echo "Results directory: ${RESULTS_DIR}"
    exit 1
fi
