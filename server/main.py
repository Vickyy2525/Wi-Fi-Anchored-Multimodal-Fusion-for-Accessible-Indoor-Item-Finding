"""
FastAPI realtime ingest + window jobs.

Run:
  uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure Remote/ is on path when launched as uvicorn server.main:app
REMOTE_ROOT = Path(__file__).resolve().parents[1]
if str(REMOTE_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOTE_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from server.graph_store import get_graph_store
from server.models import (
    HealthResponse,
    ImuBatch,
    LabelBatch,
    ObjectQueryResponse,
    RunWindowRequest,
    SessionCreate,
    SessionResponse,
    WifiBatch,
)
from server import staging
from server.scheduler import start_scheduler, stop_scheduler
from server.worker import run_window_job

app = FastAPI(title="IndoorLoc Realtime Ingest", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_scheduler():
    start_scheduler()


@app.on_event("shutdown")
async def _shutdown_scheduler():
    await stop_scheduler()


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    store = get_graph_store()
    return HealthResponse(status="ok", graph_backend=store.backend_name)


@app.post("/api/v1/sessions", response_model=SessionResponse)
def create_session(body: SessionCreate = SessionCreate()):
    sid = staging.create_session(body.device_id)
    return SessionResponse(session_id=sid)


@app.post("/api/v1/sessions/{session_id}/keyframes")
async def upload_keyframe(
    session_id: str,
    file: UploadFile = File(...),
    timestamp: int = Form(...),
    jerk: float = Form(0.0),
    gyro_mag: float = Form(0.0),
    trigger_reason: str = Form("upload"),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty image")
    info = staging.save_keyframe(
        session_id,
        timestamp,
        data,
        jerk=jerk,
        gyro_mag=gyro_mag,
        trigger_reason=trigger_reason,
    )
    return {"ok": True, **info}


@app.post("/api/v1/sessions/{session_id}/wifi")
def upload_wifi(session_id: str, body: WifiBatch):
    rows = [r.model_dump() for r in body.rows]
    return {"ok": True, **staging.save_wifi_batch(session_id, rows)}


@app.post("/api/v1/sessions/{session_id}/imu")
def upload_imu(session_id: str, body: ImuBatch):
    rows = [r.model_dump() for r in body.rows]
    return {"ok": True, **staging.save_imu_batch(session_id, rows)}


@app.post("/api/v1/sessions/{session_id}/labels")
def upload_labels(session_id: str, body: LabelBatch):
    rows = [r.model_dump() for r in body.rows]
    return {"ok": True, **staging.save_label_batch(session_id, rows)}


@app.post("/api/v1/jobs/run-window")
def jobs_run_window(body: RunWindowRequest):
    result = run_window_job(body.session_id, body.window_id)
    if not result.get("ok"):
        raise HTTPException(400, detail=result)
    return result


@app.get("/api/v1/objects/{name}", response_model=ObjectQueryResponse)
def get_object(name: str):
    store = get_graph_store()
    return ObjectQueryResponse(result=store.query_object(name))
