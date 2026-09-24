$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "  python scripts\generate_synthetic_data.py --scale small"
Write-Host "  python scripts\run_monday.py"
Write-Host "  python scripts\validate_monday.py"
Write-Host "  pytest -q"
