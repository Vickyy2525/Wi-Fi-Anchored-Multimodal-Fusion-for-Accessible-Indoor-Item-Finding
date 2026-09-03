"""Persistent memory graph: Neo4j preferred, FileGraphStore fallback."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

REMOTE_ROOT = Path(__file__).resolve().parents[1]
FILE_GRAPH_PATH = REMOTE_ROOT / "data" / "graph" / "memory_graph.json"


class GraphStore(ABC):
    backend_name: str = "abstract"

    @abstractmethod
    def upsert_binding(
        self,
        object_name: str,
        *,
        recognized_item: Optional[str],
        location: Optional[str],
        details: Optional[str],
        keyframe: str,
        timestamp: Optional[int],
        score: float,
        wifi_timestamp: Optional[int],
        wifi_fingerprint: dict,
    ) -> None:
        ...

    @abstractmethod
    def query_object(self, object_name: str) -> Any:
        ...


class FileGraphStore(GraphStore):
    """JSON-backed graph mirroring Object → VLM_Report → Signal_State."""

    backend_name = "file"

    def __init__(self, path: Path = FILE_GRAPH_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {"objects": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {"objects": {}}
        if "objects" not in self._data:
            self._data = {"objects": {}}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def upsert_binding(
        self,
        object_name: str,
        *,
        recognized_item: Optional[str],
        location: Optional[str],
        details: Optional[str],
        keyframe: str,
        timestamp: Optional[int],
        score: float,
        wifi_timestamp: Optional[int],
        wifi_fingerprint: dict,
    ) -> None:
        self._load()
        obj = self._data["objects"].setdefault(object_name, {"type": "Object", "name": object_name, "reports": []})
        report = {
            "type": "VLM_Report",
            "item": object_name,
            "recognized_item": recognized_item,
            "location": location,
            "details": details,
            "keyframe": keyframe,
            "timestamp": timestamp,
            "score": score,
            "signal": {
                "type": "Signal_State",
                "wifi_timestamp": wifi_timestamp,
                "fingerprint": wifi_fingerprint or {},
            },
        }
        # Keep newest by timestamp as primary; append history
        obj["reports"] = [r for r in obj["reports"] if r.get("keyframe") != keyframe]
        obj["reports"].append(report)
        obj["reports"].sort(key=lambda r: r.get("timestamp") or 0)
        obj["description"] = details
        self._save()

    def query_object(self, object_name: str) -> Any:
        self._load()
        obj = self._data["objects"].get(object_name)
        if not obj or not obj.get("reports"):
            return f"抱歉，記憶庫中沒有關於 [{object_name}] 的紀錄。"
        best = max(obj["reports"], key=lambda r: r.get("score") or 0.0)
        signal = best.get("signal") or {}
        ts = best.get("timestamp")
        time_str = str(ts) if ts is not None else "未知"
        try:
            from datetime import datetime
            if ts is not None:
                time_str = datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        return {
            "status": "Found",
            "object": object_name,
            "llava_recognized_as": best.get("recognized_item"),
            "visual_score": round(float(best.get("score") or 0), 4),
            "relation_description": f"位於 [{best.get('location')}]（推導關係：LOCATED_AT）",
            "last_seen_time": time_str,
            "matched_wifi_timestamp": signal.get("wifi_timestamp"),
            "target_wifi_fingerprint": signal.get("fingerprint"),
            "keyframe": best.get("keyframe"),
        }


class Neo4jGraphStore(GraphStore):
    backend_name = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

    def upsert_binding(
        self,
        object_name: str,
        *,
        recognized_item: Optional[str],
        location: Optional[str],
        details: Optional[str],
        keyframe: str,
        timestamp: Optional[int],
        score: float,
        wifi_timestamp: Optional[int],
        wifi_fingerprint: dict,
    ) -> None:
        fp_json = json.dumps(wifi_fingerprint or {}, ensure_ascii=False, default=str)
        vlm_id = f"llava_report_{keyframe}"
        signal_id = f"wifi_state_{wifi_timestamp}"
        cypher = """
        MERGE (o:Object {name: $object_name})
        SET o.type = 'Object', o.description = $details
        MERGE (v:VLM_Report {id: $vlm_id})
        SET v.type = 'VLM_Report',
            v.item = $object_name,
            v.recognized_item = $recognized_item,
            v.location = $location,
            v.details = $details,
            v.keyframe = $keyframe,
            v.timestamp = $timestamp,
            v.score = $score
        MERGE (s:Signal_State {id: $signal_id})
        SET s.type = 'Signal_State',
            s.wifi_timestamp = $wifi_timestamp,
            s.fingerprint_json = $fp_json
        MERGE (o)-[r1:LOCATED_AT]->(v)
        SET r1.timestamp = $timestamp, r1.location = $location, r1.score = $score, r1.keyframe = $keyframe
        MERGE (v)-[r2:DESCRIBED_AT]->(s)
        SET r2.keyframe = $keyframe
        """
        with self._driver.session() as session:
            session.run(
                cypher,
                object_name=object_name,
                details=details,
                vlm_id=vlm_id,
                recognized_item=recognized_item,
                location=location,
                keyframe=keyframe,
                timestamp=timestamp,
                score=score,
                signal_id=signal_id,
                wifi_timestamp=wifi_timestamp,
                fp_json=fp_json,
            )

    def query_object(self, object_name: str) -> Any:
        cypher = """
        MATCH (o:Object {name: $object_name})-[r:LOCATED_AT]->(v:VLM_Report)-[:DESCRIBED_AT]->(s:Signal_State)
        RETURN o, r, v, s
        ORDER BY coalesce(r.score, 0) DESC
        LIMIT 1
        """
        with self._driver.session() as session:
            rec = session.run(cypher, object_name=object_name).single()
        if not rec:
            return f"抱歉，記憶庫中沒有關於 [{object_name}] 的紀錄。"
        v, r, s = rec["v"], rec["r"], rec["s"]
        ts = v.get("timestamp")
        time_str = str(ts) if ts is not None else "未知"
        try:
            from datetime import datetime
            if ts is not None:
                time_str = datetime.fromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        fp = {}
        try:
            fp = json.loads(s.get("fingerprint_json") or "{}")
        except Exception:
            fp = {}
        return {
            "status": "Found",
            "object": object_name,
            "llava_recognized_as": v.get("recognized_item"),
            "visual_score": round(float(r.get("score") or v.get("score") or 0), 4),
            "relation_description": f"位於 [{v.get('location')}]（推導關係：LOCATED_AT）",
            "last_seen_time": time_str,
            "matched_wifi_timestamp": s.get("wifi_timestamp"),
            "target_wifi_fingerprint": fp,
            "keyframe": v.get("keyframe"),
        }


_store: Optional[GraphStore] = None


def strict_neo4j_required() -> bool:
    """Fail-fast when Neo4j is mandatory (STRICT_NEO4J or GRAPH_MODE=neo4j_first)."""
    flag = os.getenv("STRICT_NEO4J", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    mode = os.getenv("GRAPH_MODE", "").strip().lower()
    return mode == "neo4j_first"


def reset_graph_store() -> None:
    """Clear process-level store singleton (tests / mode switch)."""
    global _store
    if _store is not None and hasattr(_store, "close"):
        try:
            _store.close()
        except Exception:
            pass
    _store = None


def get_graph_store() -> GraphStore:
    global _store
    if _store is not None:
        return _store
    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "password").strip()
    require_neo4j = strict_neo4j_required()

    if uri:
        try:
            _store = Neo4jGraphStore(uri, user, password)
            print(f"[graph] Neo4j connected: {uri}")
            return _store
        except Exception as exc:
            if require_neo4j:
                raise RuntimeError(
                    f"STRICT Neo4j required but connection failed ({exc}). "
                    "請確認 docker compose up -d，以及 .env 的 NEO4J_URI。"
                ) from exc
            print(f"[graph] Neo4j unavailable ({exc}), using FileGraphStore")
    elif require_neo4j:
        raise RuntimeError(
            "STRICT Neo4j required but NEO4J_URI is empty. "
            "請在 .env 設定 NEO4J_URI=bolt://localhost:7687"
        )

    _store = FileGraphStore()
    print(f"[graph] FileGraphStore: {FILE_GRAPH_PATH}")
    return _store
