# Wi-Fi-Anchored Multimodal Fusion for Accessible Indoor Item Finding

Find everyday indoor objects from a phone walk: camera keyframes + IMU + Wi-Fi RSSI, then search in a web UI (SigLIP, YOLO, LLaVA, Neo4j).

- Demo video: https://youtu.be/o1gum_NE7Z8
- Repository: https://github.com/Vickyy2525/Wi-Fi-Anchored-Multimodal-Fusion-for-Accessible-Indoor-Item-Finding

## Requirements

- Python 3.10+
- Docker Desktop (for Neo4j)
- macOS (recommended): [Ollama](https://ollama.com/) with `llava`
- Windows: can run with `VLM_BACKEND=mock` (UI flow only; VLM answers are simulated)

## macOS

1. Install Docker Desktop and start it.
2. Install [Ollama](https://ollama.com/) and open the app.
3. In Terminal, from this repository root:

```bash
chmod +x run.sh
./run.sh
```

4. Open http://127.0.0.1:8501
5. In the sidebar, select the bundled indoor walk capture and search `Water bottle`.

Optional live upload API:

```bash
./run.sh with-server
```

Ingest API docs: http://127.0.0.1:8000/docs

## Windows

1. Install [Python 3.10+](https://www.python.org/downloads/) and Docker Desktop, then start Docker.
2. From this repository root in Command Prompt or PowerShell:

```cmd
run.bat
```

3. Open http://127.0.0.1:8501

Default Windows setup often uses mock VLM (no GPU / no Ollama). To use local LLaVA, copy `.env.example` to `.env` and set:

```
VLM_BACKEND=ollama
```

Then install Ollama, pull `llava`, and run `run.bat` again.

## Capture data

Keep the included `sensor_data_*` folder next to the Python files (images + IMU / keyframe / Wi-Fi CSV). Choose it in the sidebar if more than one folder is present.

## First run

The first search may take several minutes while YOLO / SigLIP build the visual cache. Later runs are faster.

## Optional: manual start

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
streamlit run app.py
```
