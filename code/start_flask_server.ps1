# PowerShell Script: Start Flask Backend Service
# UTF-8 Encoding

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting EchoSage Flask Backend Service" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Environment name
$ENV_NAME = "flask_backend"

# Check if running in Conda environment
Write-Host "Checking runtime environment..." -ForegroundColor Yellow

$condaEnv = $env:CONDA_DEFAULT_ENV
if ($condaEnv) {
    Write-Host "[OK] Current Conda environment: $condaEnv" -ForegroundColor Green
    
    if ($condaEnv -ne $ENV_NAME) {
        Write-Host "[WARNING] Recommended to use '$ENV_NAME' environment" -ForegroundColor Yellow
        Write-Host "   Run: conda activate $ENV_NAME" -ForegroundColor Gray
    }
} else {
    Write-Host "[WARNING] No Conda environment detected" -ForegroundColor Yellow
    Write-Host "   Recommended: conda activate $ENV_NAME" -ForegroundColor Gray
    Write-Host "   Or run: ..\setup_conda_env.ps1 to create environment" -ForegroundColor Gray
    Write-Host ""
    $response = Read-Host "Continue with current Python environment? (y/n)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Startup cancelled" -ForegroundColor Yellow
        pause
        exit 0
    }
}

Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow
if ($condaEnv -eq $ENV_NAME) {
    Write-Host "[OK] Using virtual environment, dependencies configured" -ForegroundColor Green
} else {
    Write-Host "Note: If dependencies are missing, run: pip install -r ../requirements.txt" -ForegroundColor Gray
}
Write-Host ""

# Start Flask service
Write-Host "Starting Flask server..." -ForegroundColor Yellow
Write-Host "Service URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: See API_DOCUMENTATION.md" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Gray
Write-Host ""

python flask_app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Flask service failed to start" -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. All dependencies are installed" -ForegroundColor Gray
    Write-Host "  2. Port 8000 is not occupied" -ForegroundColor Gray
    Write-Host "  3. Model files exist" -ForegroundColor Gray
    pause
    exit 1
}
