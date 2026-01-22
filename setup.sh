#!/bin/bash
# Setup script for Concordia Sim Framework
# Usage: ./setup.sh [--dev] [--ollama]

set -e

echo "=========================================="
echo "Concordia Sim Framework Setup"
echo "=========================================="

# Parse arguments
DEV_MODE=false
OLLAMA=false

for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            ;;
        --ollama)
            OLLAMA=true
            ;;
        --help)
            echo "Usage: ./setup.sh [--dev] [--ollama]"
            echo "  --dev     Install development dependencies"
            echo "  --ollama  Install Ollama support"
            exit 0
            ;;
    esac
done

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check Python version
if [[ $(echo "$PYTHON_VERSION < 3.11" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.11 or higher is required."
    exit 1
fi

# Determine package manager
USE_UV=false
if command -v uv &> /dev/null; then
    echo "Found uv package manager"
    USE_UV=true
else
    echo "uv not found, using pip"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."

if $USE_UV; then
    if $DEV_MODE && $OLLAMA; then
        uv sync --all-extras
    elif $DEV_MODE; then
        uv sync --extra dev
    elif $OLLAMA; then
        uv sync --extra ollama
    else
        uv sync
    fi
else
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install package
    if $DEV_MODE && $OLLAMA; then
        pip install -e ".[all]"
    elif $DEV_MODE; then
        pip install -e ".[dev]"
    elif $OLLAMA; then
        pip install -e ".[ollama]"
    else
        pip install -e .
    fi
fi

# Setup pre-commit hooks
if $DEV_MODE; then
    echo ""
    echo "Setting up pre-commit hooks..."
    if $USE_UV; then
        uv run pre-commit install
        uv run pre-commit install --hook-type commit-msg
    else
        pre-commit install
        pre-commit install --hook-type commit-msg
    fi
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your API keys."
fi

# Create outputs directory
mkdir -p outputs

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your API keys"
echo "  2. Run a test simulation:"
if $USE_UV; then
    echo "     uv run python run_experiment.py --cfg job"
else
    echo "     source .venv/bin/activate"
    echo "     python run_experiment.py --cfg job"
fi
echo ""
