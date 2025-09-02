@echo off
echo Starting AiETHTrader...
call venv\Scripts\Activate.bat
python main.py %*
pause
