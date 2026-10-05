@echo off
setlocal
cd /d "%~dp0"

if not exist .env copy .env.example .env >nul

docker compose up --build -d
if errorlevel 1 (
    echo.
    echo De chatbot kon niet starten. Controleer of Docker Desktop geopend is.
    pause
    exit /b 1
)

timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"
echo De chatbot draait op http://localhost:8501
echo Dit venster mag dicht. Gebruik Stop-chatbot.bat om de chatbot af te sluiten.
pause
