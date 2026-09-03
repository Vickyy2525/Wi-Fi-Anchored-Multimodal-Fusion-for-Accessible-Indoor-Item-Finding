#!/bin/bash
# Demo 一鍵啟動（離線最穩路徑）
#   ./run.sh              → Neo4j + Streamlit（報告 Demo）
#   ./run.sh with-server  → 再加 Ingest Server :8000（即時上傳加分）
#
# 瀏覽器：http://127.0.0.1:8501
# Neo4j： http://localhost:7474  (neo4j / password)

set -e
cd "$(dirname "$0")"

MODE="${1:-demo}"
VENV_DIR="venv"
INGEST_PID=""

cleanup() {
    if [ -n "$INGEST_PID" ] && kill -0 "$INGEST_PID" 2>/dev/null; then
        echo ""
        echo "🛑 停止 Ingest Server (pid=$INGEST_PID)..."
        kill "$INGEST_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "========================================"
echo "  IndoorLoc Demo 啟動"
echo "========================================"

# --- venv ---
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️  建立虛擬環境..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ""
echo "📦 [1/5] 檢查 Python 依賴..."
pip3 install -r requirements.txt -q

# --- .env ---
echo ""
echo "⚙️  [2/5] 環境變數..."
if [ ! -f ".env" ]; then
    if [ -f ".env.demo" ]; then
        cp .env.demo .env
        echo "✅ 已從 .env.demo 建立 .env"
    elif [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ 已從 .env.example 建立 .env"
    else
        echo "❌ 找不到 .env / .env.demo / .env.example"
        exit 1
    fi
else
    echo "✅ 使用現有 .env（Demo 固定參數見 .env.demo）"
fi

# 強制 demo 用 ollama（不覆寫其他鍵）
if grep -q "^VLM_BACKEND=" .env; then
    sed -i '' 's/^VLM_BACKEND=.*/VLM_BACKEND=ollama/' .env 2>/dev/null || \
    sed -i 's/^VLM_BACKEND=.*/VLM_BACKEND=ollama/' .env
else
    echo "VLM_BACKEND=ollama" >> .env
fi
echo "✅ VLM_BACKEND=ollama"

# --- Neo4j ---
echo ""
echo "🗄️  [3/5] 啟動 Neo4j (Docker)..."
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ 找不到 docker。請先安裝並開啟 Docker Desktop。"
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker 未就緒。請開啟 Docker Desktop 後重試。"
    exit 1
fi
docker compose up -d
echo "✅ Neo4j：http://localhost:7474  (neo4j / password)"

# --- Ollama ---
echo ""
echo "🧠 [4/5] 檢查 Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
    echo "❌ 找不到 ollama。請安裝：https://ollama.com/"
    exit 1
fi
if ! curl -s http://localhost:11434 >/dev/null 2>&1; then
    echo "❌ Ollama 未啟動。請開啟 Ollama App 後重試。"
    exit 1
fi
echo "✅ Ollama 運作中"
ollama pull llava >/dev/null 2>&1 || true

# --- optional ingest ---
if [ "$MODE" = "with-server" ]; then
    echo ""
    echo "📡 啟動 Ingest Server :8000 ..."
    uvicorn server.main:app --host 0.0.0.0 --port 8000 &
    INGEST_PID=$!
    sleep 2
    if curl -s http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
        echo "✅ Ingest：http://127.0.0.1:8000/docs"
    else
        echo "⚠️  Ingest 可能尚未就緒，請稍後檢查 port 8000"
    fi
fi

# --- Streamlit ---
echo ""
echo "🌐 [5/5] 啟動 Streamlit..."
echo "----------------------------------------"
echo "  Web UI ：http://127.0.0.1:8501"
echo "  Dataset: sensor_data_1780239297777"
echo "  Query:   Water bottle"
echo "----------------------------------------"
echo "  Ctrl+C to stop"
echo "----------------------------------------"

streamlit run app.py --server.address 0.0.0.0 --server.port 8501
