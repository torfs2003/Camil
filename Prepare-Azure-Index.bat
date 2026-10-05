@echo off
setlocal
cd /d "%~dp0"

if not exist .env (
    copy .env.example .env >nul
    echo Er is een lokaal .env-bestand gemaakt. Vul daarin de Azure OpenAI-gegevens in en start dit bestand opnieuw.
    pause
    exit /b 0
)

docker compose run --build --rm codex-chatbot python prepare_azure_index.py
if errorlevel 1 (
    echo De index kon niet worden voorbereid. Controleer de instellingen in .env en probeer opnieuw.
    pause
    exit /b 1
)

echo De Azure-vectorindex is klaar om in de containerimage te worden opgenomen.
pause
