"""5-minute window worker: dedupe, common-item search, graph upsert, media save."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from server.graph_store import get_graph_store
from server.staging import (
    MEDIA_ROOT,
    prepare_offline_folder,
    read_meta,
    resolve_window_to_run,
    staging_window_path,
    write_meta,
)
from memory_graph_adapter import resolve_graph_mode

REMOTE_ROOT = Path(__file__).resolve().parents[1]
COMMON_ITEMS_PATH = REMOTE_ROOT / "config" / "common_items.json"

_system = None


def load_common_items() -> list[str]:
    if COMMON_ITEMS_PATH.exists():
        try:
            data = json.loads(COMMON_ITEMS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    return ["水瓶", "鑰匙", "藥盒", "手機"]


def get_fusion_system(data_folder: str):
    """Lazy-load SpatialMemorySystem pointed at the window work folder."""
    global _system
    from complete_fusion_system import SpatialMemorySystem

    # Reuse process-level models; switch working data folder per window.
    if _system is None:
        _system = SpatialMemorySystem.load_models(data_folder)
    else:
        _system.data_folder = data_folder
    return _system


def _yolo_object_counts(images_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        from ultralytics import YOLO
        from PIL import Image, ImageOps
        model = YOLO("yolov8n.pt")
        for img_path in sorted(images_dir.glob("*.jpg")):
            try:
                img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
                results = model(img, verbose=False)
                counts[img_path.name] = len(results[0].boxes) if results and results[0].boxes is not None else 0
            except Exception:
                counts[img_path.name] = 0
    except Exception as exc:
        print(f"[worker] YOLO count failed: {exc}")
    return counts


def _dedupe_prefer_rich_frames(work_dir: Path, min_keep: int = 3) -> list[str]:
    """Prefer frames with more YOLO detections; drop sparse near-duplicates by keeping top-N rich frames + all if few."""
    images_dir = work_dir / "images"
    if not images_dir.exists():
        return []
    counts = _yolo_object_counts(images_dir)
    if not counts:
        return [p.name for p in images_dir.glob("*.jpg")]

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    # Keep frames with at least 1 detection, or top min_keep by count
    keep = [name for name, c in ranked if c >= 1]
    if len(keep) < min_keep:
        keep = [name for name, _ in ranked[:max(min_keep, len(ranked))]]

    keep_set = set(keep)
    for img_path in list(images_dir.glob("*.jpg")):
        if img_path.name not in keep_set:
            try:
                img_path.unlink()
            except OSError:
                pass
    print(f"[worker] dedupe: kept {len(keep_set)} / {len(counts)} frames (prefer high object count)")
    return list(keep_set)


def _parse_ts_from_name(name: str) -> Optional[int]:
    try:
        # keyframe_123.jpg
        return int(name.split("_")[1].split(".")[0])
    except Exception:
        return None


def _quality_score_rgb(img) -> float:
    """Simple sharpness + brightness quality score."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    # Gradient magnitude as sharpness proxy
    gy, gx = np.gradient(arr)
    sharp = float(np.var(np.hypot(gx, gy)))
    bright = float(np.mean(arr) / 255.0)
    return sharp * 0.01 + bright * 0.2


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _dedupe_frames_v2(work_dir: Path, system, min_keep: int = 3) -> dict[str, Any]:
    """
    Multi-signal dedupe:
      1) temporal bucket
      2) visual similarity suppression
      3) quality + object-richness score
    """
    images_dir = work_dir / "images"
    files = sorted(images_dir.glob("*.jpg"))
    if not files:
        return {"kept": [], "removed": [], "reason": "no_images"}

    temporal_sec = float(os.getenv("DEDUPE_TEMPORAL_SEC", "3"))
    sim_threshold = float(os.getenv("DEDUPE_SIM_THRESHOLD", "0.93"))
    per_bucket_keep = int(os.getenv("DEDUPE_BUCKET_KEEP", "2"))
    per_bucket_keep = max(1, per_bucket_keep)

    records = []
    for idx, img_path in enumerate(files):
        try:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            ts = _parse_ts_from_name(img_path.name)
            if ts is None:
                ts = idx * int(temporal_sec * 1000)
            bucket = int(ts // int(temporal_sec * 1000))
            quality = _quality_score_rgb(img)
            emb = system.extract_image_feature(img)
            yolo_res = system.yolo_model(img, verbose=False)
            object_count = len(yolo_res[0].boxes) if yolo_res and yolo_res[0].boxes is not None else 0
            score = quality + 0.08 * float(object_count)
            records.append(
                {
                    "name": img_path.name,
                    "path": img_path,
                    "timestamp": ts,
                    "bucket": bucket,
                    "quality": quality,
                    "object_count": object_count,
                    "score": score,
                    "embedding": emb,
                }
            )
        except Exception as exc:
            print(f"[worker] dedupe_v2 skip {img_path.name}: {exc}")

    if not records:
        return {"kept": [], "removed": [], "reason": "all_failed"}

    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)

    kept: list[dict[str, Any]] = []
    for bucket in sorted(by_bucket.keys()):
        bucket_rows = sorted(by_bucket[bucket], key=lambda x: x["score"], reverse=True)
        selected: list[dict[str, Any]] = []
        for row in bucket_rows:
            too_similar = False
            for s in selected:
                if _cosine_sim(row["embedding"], s["embedding"]) >= sim_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(row)
            if len(selected) >= per_bucket_keep:
                break
        kept.extend(selected)

    if len(kept) < min_keep:
        all_sorted = sorted(records, key=lambda x: x["score"], reverse=True)
        seen = {r["name"] for r in kept}
        for row in all_sorted:
            if row["name"] in seen:
                continue
            kept.append(row)
            seen.add(row["name"])
            if len(kept) >= min_keep:
                break

    keep_names = {r["name"] for r in kept}
    removed = []
    for p in files:
        if p.name in keep_names:
            continue
        removed.append(p.name)
        try:
            p.unlink()
        except OSError:
            pass

    print(
        f"[worker] dedupe_v2: kept {len(keep_names)} / {len(files)} "
        f"(bucket={temporal_sec}s, sim<{sim_threshold}, keep_per_bucket={per_bucket_keep})"
    )
    return {
        "kept": sorted(list(keep_names)),
        "removed": removed,
        "kept_count": len(keep_names),
        "total_count": len(files),
        "sim_threshold": sim_threshold,
        "temporal_sec": temporal_sec,
        "per_bucket_keep": per_bucket_keep,
    }


def _save_representative_image(
    object_name: str,
    src_image: Path,
    *,
    timestamp: Optional[int],
    score: float,
    keyframe: str,
    wifi_ts: Optional[int],
) -> Path:
    out_dir = MEDIA_ROOT / object_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "latest.jpg"
    meta_path = out_dir / "meta.json"
    prev_ts = None
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            prev_ts = prev.get("timestamp")
        except Exception:
            prev_ts = None
    # Keep newest by timestamp
    if prev_ts is not None and timestamp is not None and int(timestamp) < int(prev_ts):
        return dest
    shutil.copy2(src_image, dest)
    meta_path.write_text(
        json.dumps({
            "object": object_name,
            "timestamp": timestamp,
            "keyframe": keyframe,
            "score": score,
            "wifi_timestamp": wifi_ts,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest


def run_window_job(session_id: str, window_id: Optional[int] = None) -> dict[str, Any]:
    wid = resolve_window_to_run(session_id, window_id)
    if wid is None:
        return {"ok": False, "error": f"No staging window for session={session_id}"}

    wpath = staging_window_path(session_id, wid)
    if not wpath.exists():
        return {"ok": False, "error": f"Window path missing: {wpath}"}

    write_meta(wpath, {"status": "processing"})
    work_dir = prepare_offline_folder(session_id, wid)

    items = load_common_items()
    store = get_graph_store()
    graph_mode = resolve_graph_mode()
    # dual_write / neo4j_first 時 search_and_verify 已寫入 GraphStore
    skip_store_upsert = graph_mode in {"dual_write", "neo4j_first"}
    results_summary = []
    confirmed_total = 0

    try:
        system = get_fusion_system(str(work_dir))
        dedupe_report = _dedupe_frames_v2(work_dir, system, min_keep=3)
        kept = dedupe_report.get("kept", [])
        if not kept:
            # fallback to previous simple strategy if v2 fails unexpectedly
            kept = _dedupe_prefer_rich_frames(work_dir)
            dedupe_report = {"fallback": "yolo_count_only", "kept": kept}
        # Build DB after dedupe to avoid indexing removed frames
        system.build_vector_database(use_cache=False)
        results_dir = work_dir / "job_results"
        results_dir.mkdir(exist_ok=True)

        for item in items:
            print(f"[worker] searching common item: {item}")
            result = system.search_and_verify(
                item,
                top_k=3,
                results_dir=str(results_dir),
                verbose=False,
            )
            confirmed_total += int(result.get("confirmed_count") or 0)
            for cand in result.get("candidates") or []:
                if not cand.get("bound"):
                    continue
                fname = cand.get("filename")
                src = work_dir / "images" / fname
                if not src.exists():
                    continue
                if not skip_store_upsert:
                    store.upsert_binding(
                        item,
                        recognized_item=cand.get("recognized_item"),
                        location=cand.get("location_label"),
                        details=cand.get("llava_answer"),
                        keyframe=fname,
                        timestamp=cand.get("keyframe_timestamp"),
                        score=float(cand.get("score") or 0),
                        wifi_timestamp=cand.get("wifi_timestamp"),
                        wifi_fingerprint={},
                    )
                    try:
                        row = system.db_df[system.db_df["filename"] == fname].iloc[0]
                        fp = row.get("wifi_fingerprint") or {}
                        store.upsert_binding(
                            item,
                            recognized_item=cand.get("recognized_item"),
                            location=cand.get("location_label"),
                            details=cand.get("llava_answer"),
                            keyframe=fname,
                            timestamp=cand.get("keyframe_timestamp"),
                            score=float(cand.get("score") or 0),
                            wifi_timestamp=cand.get("wifi_timestamp"),
                            wifi_fingerprint=dict(fp) if isinstance(fp, dict) else {},
                        )
                    except Exception:
                        pass
                _save_representative_image(
                    item,
                    src,
                    timestamp=cand.get("keyframe_timestamp"),
                    score=float(cand.get("score") or 0),
                    keyframe=fname,
                    wifi_ts=cand.get("wifi_timestamp"),
                )
            results_summary.append({
                "item": item,
                "confirmed": result.get("confirmed_count"),
                "object_location": result.get("object_location"),
                "graph_consistency": result.get("graph_consistency"),
            })
    except Exception as exc:
        write_meta(wpath, {"status": "error", "error": str(exc)})
        return {"ok": False, "session_id": session_id, "window_id": wid, "error": str(exc)}

    # Cleanup: remove non-kept images from staging images (keep wifi/imu)
    staging_images = wpath / "images"
    if staging_images.exists() and kept:
        keep_set = set(kept)
        # Also keep representative media sources already copied
        for img in list(staging_images.glob("*.jpg")):
            if img.name not in keep_set:
                try:
                    img.unlink()
                except OSError:
                    pass

    write_meta(wpath, {
        "status": "processed",
        "kept_frames": kept,
        "confirmed_total": confirmed_total,
        "graph_backend": store.backend_name,
        "graph_mode": graph_mode,
        "dedupe_report": dedupe_report,
    })

    return {
        "ok": True,
        "session_id": session_id,
        "window_id": wid,
        "kept_frames": kept,
        "confirmed_total": confirmed_total,
        "items": results_summary,
        "dedupe_report": dedupe_report,
        "graph_backend": store.backend_name,
        "graph_mode": graph_mode,
        "meta": read_meta(wpath),
    }
