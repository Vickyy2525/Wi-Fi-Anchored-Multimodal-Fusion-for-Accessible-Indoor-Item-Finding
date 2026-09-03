@echo off
:: Demo launcher (Windows)
::   run.bat              → Neo4j + Streamlit
::   run.bat with-server  → also Ingest :8000
::
:: Browser: http://127.0.0.1:8501
:: Neo4j:   http://localhost:7474  (neo4j / password)

chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set VENV_DIR=venv
set MODE=%1
if "%MODE%"=="" set MODE=demo

if not exist %VENV_DIR%\Scripts\activate.bat (
    echo Creating venv...
    python -m venv %VENV_DIR%
)
call %VENV_DIR%\Scripts\activate.bat

echo.
echo [1/5] Python deps...
pip install -r requirements.txt -q

echo.
echo [2/5] .env...
if not exist .env (
    if exist .env.demo (
        copy .env.demo .env >nul
        echo Created .env from .env.demo
    ) else if exist .env.example (
        copy .env.example .env >nul
        echo Created .env from .env.example
    ) else (
        echo Missing .env / .env.demo
        pause
        exit /b 1
    )
) else (
    echo Using existing .env
)

echo.
echo [3/5] Neo4j (Docker)...
docker compose up -d
if errorlevel 1 (
    echo Docker failed. Start Docker Desktop and retry.
    pause
    exit /b 1
)

echo.
echo [4/5] VLM note: set VLM_BACKEND=ollama in .env if you use Ollama.
echo       Default Windows template often uses mock for UI-only tests.

echo.
echo [5/5] Streamlit...
echo Web UI: http://127.0.0.1:8501
echo Dataset: sensor_data_1780239297777  Query: Water bottle
echo.

if /I "%MODE%"=="with-server" (
    echo Starting Ingest on :8000 in a new window...
    start "IndoorLoc-Ingest" cmd /c "call %VENV_DIR%\Scripts\activate.bat && uvicorn server.main:app --host 0.0.0.0 --port 8000"
)

streamlit run app.py --server.address 0.0.0.0 --server.port 8501

echo.
pause
