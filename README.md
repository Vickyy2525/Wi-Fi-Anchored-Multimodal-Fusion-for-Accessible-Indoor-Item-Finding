# Indoor Item Finder

Find everyday indoor objects with a phone walk: camera keyframes + IMU + Wi-Fi RSSI, then search in a web UI (SigLIP, YOLO, LLaVA, Neo4j).

Demo video: https://youtu.be/o1gum_NE7Z8

## Requirements

- Python 3.10+
- Docker Desktop (for Neo4j)
- macOS (recommended): [Ollama](https://ollama.com/) with `llava`
- Windows: can run with `VLM_BACKEND=mock` (UI only, fake VLM answers)

## macOS

1. Install Docker Desktop and start it.
2. Install [Ollama](https://ollama.com/) and open the app.
3. In Terminal:

```bash
cd Remote
chmod +x run.sh
./run.sh
```

4. Open http://127.0.0.1:8501
5. In the sidebar, pick dataset `sensor_data_1780239297777` and search `Water bottle`.

Optional live upload API:

```bash
./run.sh with-server
```

Ingest API: http://127.0.0.1:8000/docs

## Windows

1. Install [Python 3.10+](https://www.python.org/downloads/) and Docker Desktop, then start Docker.
2. In Command Prompt or PowerShell:

```cmd
cd Remote
run.bat
```

3. Open http://127.0.0.1:8501

Default Windows setup often uses mock VLM (no GPU / no Ollama). To use local LLaVA instead, copy `.env.example` to `.env` and set:

```
VLM_BACKEND=ollama
```

Then install Ollama, pull `llava`, and run `run.bat` again.

## Dataset

Keep this folder next to the Python files:

```
sensor_data_1780239297777/
  images/
  imu_*.csv
  keyframes_*.csv
  wifi_*.csv
```

If you have another `sensor_data_*` folder, choose it in the sidebar.

## First run

The first search may take several minutes while YOLO / SigLIP build the visual cache (`feature_cache.pkl` inside the dataset folder). Later runs are faster.

## Optional: manual start

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
streamlit run app.py
```
