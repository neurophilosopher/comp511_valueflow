# Setup script for Concordia Sim Framework (Windows PowerShell)
# Usage: .\setup.ps1 [-Dev] [-Ollama]

param(
    [switch]$Dev,
    [switch]$Ollama,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Concordia Sim Framework Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

if ($Help) {
    Write-Host "Usage: .\setup.ps1 [-Dev] [-Ollama]"
    Write-Host "  -Dev     Install development dependencies"
    Write-Host "  -Ollama  Install Ollama support"
    exit 0
}

# Check for Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python is required but not installed." -ForegroundColor Red
    exit 1
}

$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Found Python $pythonVersion"

# Check Python version
if ([float]$pythonVersion -lt 3.11) {
    Write-Host "Error: Python 3.11 or higher is required." -ForegroundColor Red
    exit 1
}

# Check for uv
$useUv = $false
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    Write-Host "Found uv package manager"
    $useUv = $true
} else {
    Write-Host "uv not found, using pip"
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..."

if ($useUv) {
    if ($Dev -and $Ollama) {
        uv sync --all-extras
    } elseif ($Dev) {
        uv sync --extra dev
    } elseif ($Ollama) {
        uv sync --extra ollama
    } else {
        uv sync
    }
} else {
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment..."
        python -m venv .venv
    }

    # Activate virtual environment
    & .\.venv\Scripts\Activate.ps1

    # Upgrade pip
    pip install --upgrade pip

    # Install package
    if ($Dev -and $Ollama) {
        pip install -e ".[all]"
    } elseif ($Dev) {
        pip install -e ".[dev]"
    } elseif ($Ollama) {
        pip install -e ".[ollama]"
    } else {
        pip install -e .
    }
}

# Setup pre-commit hooks
if ($Dev) {
    Write-Host ""
    Write-Host "Setting up pre-commit hooks..."
    if ($useUv) {
        uv run pre-commit install
        uv run pre-commit install --hook-type commit-msg
    } else {
        pre-commit install
        pre-commit install --hook-type commit-msg
    }
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Creating .env file from template..."
    Copy-Item .env.example .env
    Write-Host "Please edit .env and add your API keys." -ForegroundColor Yellow
}

# Create outputs directory
New-Item -ItemType Directory -Force -Path outputs | Out-Null

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit .env and add your API keys"
Write-Host "  2. Run a test simulation:"
if ($useUv) {
    Write-Host "     uv run python run_experiment.py --cfg job"
} else {
    Write-Host "     .\.venv\Scripts\Activate.ps1"
    Write-Host "     python run_experiment.py --cfg job"
}
Write-Host ""
