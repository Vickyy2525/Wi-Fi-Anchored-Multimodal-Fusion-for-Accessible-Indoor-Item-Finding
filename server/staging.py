"""Staging filesystem helpers for realtime ingest."""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

WINDOW_MS = 300_000  # 5 minutes

REMOTE_ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = REMOTE_ROOT / "data" / "staging"
MEDIA_ROOT = REMOTE_ROOT / "data" / "media" / "objects"


def window_id_for_ts(timestamp_ms: int) -> int:
    return int(timestamp_ms) // WINDOW_MS


def ensure_window_dir(session_id: str, window_id: int) -> Path:
    path = STAGING_ROOT / session_id / str(window_id)
    (path / "images").mkdir(parents=True, exist_ok=True)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        write_meta(path, {
            "session_id": session_id,
            "window_id": window_id,
            "status": "receiving",
            "created_at": int(time.time() * 1000),
        })
    for name, header in (
        ("imu.csv", "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,jerk,gyro_mag"),
        ("wifi.csv", "timestamp,ssid,bssid,level,frequency"),
        ("keyframes.csv", "timestamp,image_name,jerk,gyro_mag,trigger_reason"),
        ("labels.csv", "timestamp,label"),
    ):
        fp = path / name
        if not fp.exists():
            fp.write_text(header + "\n", encoding="utf-8")
    return path


def write_meta(window_path: Path, data: dict[str, Any]) -> None:
    existing = {}
    meta_path = window_path / "meta.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(data)
    meta_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def read_meta(window_path: Path) -> dict[str, Any]:
    meta_path = window_path / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def create_session(device_id: Optional[str] = None) -> str:
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    sess_dir = STAGING_ROOT / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    (sess_dir / "session.json").write_text(
        json.dumps({
            "session_id": session_id,
            "device_id": device_id,
            "created_at": int(time.time() * 1000),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return session_id


def append_csv_rows(csv_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return len(rows)


def save_keyframe(
    session_id: str,
    timestamp: int,
    image_bytes: bytes,
    jerk: float = 0.0,
    gyro_mag: float = 0.0,
    trigger_reason: str = "upload",
) -> dict[str, Any]:
    wid = window_id_for_ts(timestamp)
    wdir = ensure_window_dir(session_id, wid)
    image_name = f"keyframe_{timestamp}.jpg"
    img_path = wdir / "images" / image_name
    img_path.write_bytes(image_bytes)
    append_csv_rows(
        wdir / "keyframes.csv",
        ["timestamp", "image_name", "jerk", "gyro_mag", "trigger_reason"],
        [{
            "timestamp": timestamp,
            "image_name": image_name,
            "jerk": jerk,
            "gyro_mag": gyro_mag,
            "trigger_reason": trigger_reason,
        }],
    )
    write_meta(wdir, {"uploaded_at": int(time.time() * 1000), "status": "receiving"})
    return {"window_id": wid, "image_name": image_name, "path": str(img_path)}


def save_wifi_batch(session_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"written": 0}
    ts = int(rows[0]["timestamp"])
    wid = window_id_for_ts(ts)
    wdir = ensure_window_dir(session_id, wid)
    n = append_csv_rows(
        wdir / "wifi.csv",
        ["timestamp", "ssid", "bssid", "level", "frequency"],
        rows,
    )
    write_meta(wdir, {"uploaded_at": int(time.time() * 1000)})
    return {"written": n, "window_id": wid}


def save_imu_batch(session_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"written": 0}
    ts = int(rows[0]["timestamp"])
    wid = window_id_for_ts(ts)
    wdir = ensure_window_dir(session_id, wid)
    n = append_csv_rows(
        wdir / "imu.csv",
        ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
         "mag_x", "mag_y", "mag_z", "jerk", "gyro_mag"],
        rows,
    )
    write_meta(wdir, {"uploaded_at": int(time.time() * 1000)})
    return {"written": n, "window_id": wid}


def save_label_batch(session_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"written": 0}
    ts = int(rows[0]["timestamp"])
    wid = window_id_for_ts(ts)
    wdir = ensure_window_dir(session_id, wid)
    n = append_csv_rows(wdir / "labels.csv", ["timestamp", "label"], rows)
    write_meta(wdir, {"uploaded_at": int(time.time() * 1000)})
    return {"written": n, "window_id": wid}


def list_window_ids(session_id: str) -> list[int]:
    sess = STAGING_ROOT / session_id
    if not sess.exists():
        return []
    ids = []
    for p in sess.iterdir():
        if p.is_dir() and p.name.isdigit():
            ids.append(int(p.name))
    return sorted(ids)


def list_sessions() -> list[str]:
    """List available staging sessions."""
    if not STAGING_ROOT.exists():
        return []
    sessions = [p.name for p in STAGING_ROOT.iterdir() if p.is_dir()]
    return sorted(sessions)


def resolve_window_to_run(session_id: str, window_id: Optional[int] = None) -> Optional[int]:
    """Pick a window to process: explicit id, else latest complete window (before current)."""
    ids = list_window_ids(session_id)
    if not ids:
        return None
    if window_id is not None:
        return window_id if window_id in ids else None
    current = window_id_for_ts(int(time.time() * 1000))
    complete = [i for i in ids if i < current]
    if complete:
        return complete[-1]
    return ids[-1]


def latest_complete_window_id(session_id: str) -> Optional[int]:
    """Latest complete window (strictly before current time window)."""
    ids = list_window_ids(session_id)
    if not ids:
        return None
    current = window_id_for_ts(int(time.time() * 1000))
    complete = [i for i in ids if i < current]
    if not complete:
        return None
    return complete[-1]


def staging_window_path(session_id: str, window_id: int) -> Path:
    return STAGING_ROOT / session_id / str(window_id)


def prepare_offline_folder(session_id: str, window_id: int) -> Path:
    """Copy/link staging window into a folder layout compatible with complete_fusion_system."""
    src = staging_window_path(session_id, window_id)
    out = REMOTE_ROOT / "data" / "work" / f"{session_id}_{window_id}"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    images = out / "images"
    images.mkdir(exist_ok=True)

    src_images = src / "images"
    if src_images.exists():
        for img in src_images.glob("*.jpg"):
            target = images / img.name
            if not target.exists():
                try:
                    os.link(img, target)
                except OSError:
                    import shutil
                    shutil.copy2(img, target)

    # Rename staging csvs to expected patterns
    import shutil
    for src_name, dst_name in (
        ("wifi.csv", f"wifi_{window_id}.csv"),
        ("imu.csv", f"imu_{window_id}.csv"),
        ("keyframes.csv", f"keyframes_{window_id}.csv"),
        ("labels.csv", "labels.csv"),
    ):
        s = src / src_name
        if s.exists():
            shutil.copy2(s, out / dst_name)
    return out
