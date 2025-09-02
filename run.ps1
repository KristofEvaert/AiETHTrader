# AiETHTrader PowerShell Launcher
Write-Host "Starting AiETHTrader..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"
python main.py $args
