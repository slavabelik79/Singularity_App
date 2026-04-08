@echo off
cd /d "%~dp0"

echo Starting Singularity App...
echo Waiting for server to initialize...

start "" http://localhost:5000

python app.py

pause
