"""
Session-scoped memory graph adapter.

PR1: NetworkXSessionGraph behind a common interface.
PR2: DualWriteSessionGraph — write NetworkX + persistent GraphStore,
     then compare key query fields for consistency.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import networkx as nx

from i18n import t_lang

# Fields compared between session (NetworkX) and persistent store (Neo4j/File).
COMPARE_KEYS = (
    "status",
    "object",
    "visual_score",
    "matched_wifi_timestamp",
    "llava_recognized_as",
)


def binding_params_from_search(
    object_name: str,
    candidate: dict[str, Any],
    vlm_result: dict[str, Any],
) -> dict[str, Any]:
    """Normalize candidate + VLM output into GraphStore-compatible fields."""
    return {
        "object_name": object_name,
        "recognized_item": vlm_result.get("item") or object_name,
        "location": vlm_result.get("location") or candidate.get("location_label"),
        "details": vlm_result.get("details"),
        "keyframe": candidate["filename"],
        "timestamp": candidate.get("keyframe_timestamp"),
        "score": float(candidate.get("score") or 0.0),
        "wifi_timestamp": candidate.get("wifi_timestamp"),
        "wifi_fingerprint": candidate.get("wifi_fingerprint") or {},
    }


def resolve_graph_mode() -> str:
    """
    GRAPH_MODE:
      - networkx     : session-only NetworkX (PR1)
      - dual_write   : NetworkX + Neo4j/File, compare consistency (PR2)
      - neo4j_first  : Neo4j is source of truth for query; NetworkX scratch only (PR3)
    """
    mode = os.getenv("GRAPH_MODE", "networkx").strip().lower()
    if mode in {"networkx", "dual_write", "neo4j_first"}:
        return mode
    print(f"[graph] Unknown GRAPH_MODE={mode!r}, falling back to networkx")
    return "networkx"


def compare_query_results(primary: Any, secondary: Any) -> dict[str, Any]:
    """Compare two query_object results on shared key fields."""
    primary_dict = primary if isinstance(primary, dict) else {"status": "miss", "raw": str(primary)}
    secondary_dict = secondary if isinstance(secondary, dict) else {"status": "miss", "raw": str(secondary)}

    mismatches: list[str] = []
    for key in COMPARE_KEYS:
        pv = primary_dict.get(key)
        sv = secondary_dict.get(key)
        # Tolerant numeric compare for scores
        if key == "visual_score" and pv is not None and sv is not None:
            try:
                if abs(float(pv) - float(sv)) > 1e-3:
                    mismatches.append(key)
                continue
            except (TypeError, ValueError):
                pass
        if pv != sv:
            mismatches.append(key)

    match = len(mismatches) == 0
    return {
        "match": match,
        "mismatched_keys": mismatches,
        "session": {k: primary_dict.get(k) for k in COMPARE_KEYS if k in primary_dict or isinstance(primary, dict)},
        "store": {k: secondary_dict.get(k) for k in COMPARE_KEYS if k in secondary_dict or isinstance(secondary, dict)},
    }


class SessionMemoryGraph(ABC):
    """Ephemeral memory graph for a single search / worker window."""

    backend_name: str = "abstract"

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def add_binding(
        self,
        object_name: str,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    def query_object(self, object_name: str, lang: str = "zh") -> Any:
        ...

    @abstractmethod
    def stats(self) -> dict[str, int]:
        ...

    @abstractmethod
    def list_nearby_objects(self, exclude: str) -> list[dict[str, Any]]:
        ...

    def verify_consistency(self, object_name: str) -> Optional[dict[str, Any]]:
        """Optional dual-write consistency report; default None."""
        return None


class NetworkXSessionGraph(SessionMemoryGraph):
    """In-memory NetworkX implementation (current default behaviour)."""

    backend_name = "networkx"

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Underlying graph (debug / transitional access only)."""
        return self._graph

    def reset(self) -> None:
        self._graph = nx.MultiDiGraph()

    def add_binding(
        self,
        object_name: str,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any],
    ) -> None:
        params = binding_params_from_search(object_name, candidate, vlm_result)
        filename = params["keyframe"]
        kf_time = params["timestamp"]
        wifi_ts = params["wifi_timestamp"]
        wifi_fp = params["wifi_fingerprint"]
        score = params["score"]
        location = params["location"]
        details = params["details"]
        recognized_item = params["recognized_item"]

        signal_node_id = f"wifi_state_{wifi_ts}"
        self._graph.add_node(
            signal_node_id,
            type="Signal_State",
            fingerprint=wifi_fp,
            timestamp=wifi_ts,
        )

        vlm_node_id = f"llava_report_{filename}"
        self._graph.add_node(
            vlm_node_id,
            type="VLM_Report",
            item=object_name,
            recognized_item=recognized_item,
            location=location,
            details=details,
            keyframe=filename,
            timestamp=kf_time,
            score=score,
        )

        self._graph.add_node(
            object_name,
            type="Object",
            description=details,
        )

        self._graph.add_edge(
            object_name,
            vlm_node_id,
            relation="LOCATED_AT",
            timestamp=kf_time,
            location=location,
            keyframe=filename,
            score=score,
        )
        self._graph.add_edge(
            vlm_node_id,
            signal_node_id,
            relation="DESCRIBED_AT",
            keyframe=filename,
        )

    def query_object(self, object_name: str, lang: str = "zh") -> Any:
        graph = self._graph
        if not graph.has_node(object_name):
            return t_lang("graph_not_found", lang, item=object_name)

        best_edge = None
        best_score = -1.0
        for _, anchor_node, data in graph.edges(object_name, data=True):
            if data.get("relation") in ["ON", "NEXT_TO", "INSIDE", "LOCATED_AT"]:
                edge_score = data.get("score", 0.0) or 0.0
                if edge_score >= best_score:
                    best_score = edge_score
                    best_edge = (anchor_node, data)

        if best_edge is None:
            return t_lang("graph_no_wifi", lang, item=object_name)

        anchor_node, data = best_edge
        relation = data.get("relation")
        timestamp = data.get("timestamp")
        keyframe = data.get("keyframe") or graph.nodes[anchor_node].get("keyframe")
        anchor_location = graph.nodes[anchor_node].get("location")
        recognized_item = graph.nodes[anchor_node].get("recognized_item")

        for _, signal_node, a_data in graph.edges(anchor_node, data=True):
            if a_data.get("relation") in ["DESCRIBED_AT", "OBSERVED_AT"]:
                wifi_feature = graph.nodes[signal_node].get("fingerprint")
                wifi_ts = graph.nodes[signal_node].get("timestamp")

                time_str = t_lang("graph_time_unknown", lang)
                try:
                    if timestamp is not None:
                        time_str = datetime.fromtimestamp(int(timestamp) / 1000).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                except Exception:
                    time_str = str(timestamp)

                return {
                    "status": "Found",
                    "object": object_name,
                    "llava_recognized_as": recognized_item,
                    "visual_score": round(float(best_score), 4),
                    "relation_description": t_lang(
                        "graph_relation", lang, loc=anchor_location, rel=relation
                    ),
                    "last_seen_time": time_str,
                    "matched_wifi_timestamp": wifi_ts,
                    "target_wifi_fingerprint": wifi_feature,
                    "keyframe": keyframe,
                }
        return t_lang("graph_no_wifi", lang, item=object_name)

    def stats(self) -> dict[str, int]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }

    def list_nearby_objects(self, exclude: str) -> list[dict[str, Any]]:
        nearby: list[dict[str, Any]] = []
        for node, data in self._graph.nodes(data=True):
            if data.get("type") != "Object" or node == exclude:
                continue
            ts = None
            for _, _, edata in self._graph.edges(node, data=True):
                if edata.get("relation") == "LOCATED_AT":
                    ts = edata.get("timestamp")
                    break
            nearby.append({"name": node, "timestamp": ts})
        return nearby


class DualWriteSessionGraph(SessionMemoryGraph):
    """
    Write to NetworkX (session) and persistent GraphStore (Neo4j or File).
    Query still returns the session (NetworkX) view; verify_consistency()
    compares against the store.
    """

    backend_name = "dual_write"

    def __init__(self, session: NetworkXSessionGraph, store: Any):
        self._session = session
        self._store = store
        store_name = getattr(store, "backend_name", type(store).__name__)
        self.backend_name = f"dual_write:{store_name}"
        self._last_consistency: Optional[dict[str, Any]] = None
        self._bound_objects: set[str] = set()

    def reset(self) -> None:
        self._session.reset()
        self._last_consistency = None
        self._bound_objects.clear()

    def add_binding(
        self,
        object_name: str,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any],
    ) -> None:
        self._session.add_binding(object_name, candidate, vlm_result)
        self._bound_objects.add(object_name)
        params = binding_params_from_search(object_name, candidate, vlm_result)
        try:
            self._store.upsert_binding(
                params["object_name"],
                recognized_item=params["recognized_item"],
                location=params["location"],
                details=params["details"],
                keyframe=params["keyframe"],
                timestamp=params["timestamp"],
                score=params["score"],
                wifi_timestamp=params["wifi_timestamp"],
                wifi_fingerprint=params["wifi_fingerprint"],
            )
        except Exception as exc:
            print(f"[graph] dual_write store upsert failed: {exc}")

    def query_object(self, object_name: str, lang: str = "zh") -> Any:
        return self._session.query_object(object_name, lang=lang)

    def verify_consistency(self, object_name: str) -> Optional[dict[str, Any]]:
        session_result = self._session.query_object(object_name)
        try:
            store_result = self._store.query_object(object_name)
        except Exception as exc:
            report = {
                "match": False,
                "error": str(exc),
                "store_backend": getattr(self._store, "backend_name", "unknown"),
            }
            self._last_consistency = report
            print(f"[graph] consistency check failed to query store: {exc}")
            return report

        report = compare_query_results(session_result, store_result)
        report["store_backend"] = getattr(self._store, "backend_name", "unknown")
        report["object"] = object_name
        self._last_consistency = report
        if report["match"]:
            print(
                f"[graph] consistency OK ({report['store_backend']}): "
                f"object={object_name} score={report['session'].get('visual_score')} "
                f"wifi_ts={report['session'].get('matched_wifi_timestamp')}"
            )
        else:
            print(
                f"[graph] consistency MISMATCH ({report['store_backend']}): "
                f"keys={report['mismatched_keys']} "
                f"session={report['session']} store={report['store']}"
            )
        return report

    def stats(self) -> dict[str, int]:
        return self._session.stats()

    def list_nearby_objects(self, exclude: str) -> list[dict[str, Any]]:
        return self._session.list_nearby_objects(exclude)


class Neo4jFirstSessionGraph(SessionMemoryGraph):
    """
    PR3: persistent Neo4j is the source of truth for queries.
    NetworkX is kept only as an ephemeral scratch pad (nearby / stats / debug).
    """

    backend_name = "neo4j_first"

    def __init__(self, store: Any, scratch: NetworkXSessionGraph | None = None):
        store_name = getattr(store, "backend_name", type(store).__name__)
        if store_name != "neo4j":
            raise RuntimeError(
                f"neo4j_first requires Neo4jGraphStore, got backend={store_name!r}. "
                "請啟動 Neo4j 並設定 NEO4J_URI。"
            )
        self._store = store
        self._scratch = scratch or NetworkXSessionGraph()
        self.backend_name = f"neo4j_first:{store_name}"
        self._last_consistency: Optional[dict[str, Any]] = None

    def reset(self) -> None:
        self._scratch.reset()
        self._last_consistency = None

    def add_binding(
        self,
        object_name: str,
        candidate: dict[str, Any],
        vlm_result: dict[str, Any],
    ) -> None:
        params = binding_params_from_search(object_name, candidate, vlm_result)
        # Source of truth first
        self._store.upsert_binding(
            params["object_name"],
            recognized_item=params["recognized_item"],
            location=params["location"],
            details=params["details"],
            keyframe=params["keyframe"],
            timestamp=params["timestamp"],
            score=params["score"],
            wifi_timestamp=params["wifi_timestamp"],
            wifi_fingerprint=params["wifi_fingerprint"],
        )
        # Scratch for same-search nearby / stats only
        self._scratch.add_binding(object_name, candidate, vlm_result)

    def query_object(self, object_name: str, lang: str = "zh") -> Any:
        return self._store.query_object(object_name)

    def verify_consistency(self, object_name: str) -> Optional[dict[str, Any]]:
        """Compare Neo4j (truth) vs NetworkX scratch for this search session."""
        try:
            store_result = self._store.query_object(object_name)
        except Exception as exc:
            report = {
                "match": False,
                "error": str(exc),
                "store_backend": "neo4j",
                "source_of_truth": "neo4j",
            }
            self._last_consistency = report
            print(f"[graph] neo4j_first store query failed: {exc}")
            return report

        scratch_result = self._scratch.query_object(object_name)
        # primary = store (Neo4j), secondary = scratch
        report = compare_query_results(store_result, scratch_result)
        report["store_backend"] = "neo4j"
        report["source_of_truth"] = "neo4j"
        report["object"] = object_name
        # Remap labels for UI clarity
        report["neo4j"] = report.pop("session")
        report["scratch"] = report.pop("store")
        self._last_consistency = report
        if report["match"]:
            print(
                f"[graph] neo4j_first OK: object={object_name} "
                f"score={report['neo4j'].get('visual_score')} "
                f"wifi_ts={report['neo4j'].get('matched_wifi_timestamp')}"
            )
        else:
            print(
                f"[graph] neo4j_first vs scratch MISMATCH: "
                f"keys={report['mismatched_keys']} "
                f"neo4j={report['neo4j']} scratch={report['scratch']}"
            )
        return report

    def stats(self) -> dict[str, int]:
        # Session scratch stats for this search window
        return self._scratch.stats()

    def list_nearby_objects(self, exclude: str) -> list[dict[str, Any]]:
        return self._scratch.list_nearby_objects(exclude)


def create_session_graph(backend: str | None = None) -> SessionMemoryGraph:
    mode = (backend or resolve_graph_mode()).strip().lower()
    if mode == "networkx":
        return NetworkXSessionGraph()
    if mode == "dual_write":
        from server.graph_store import get_graph_store

        store = get_graph_store()
        print(f"[graph] GRAPH_MODE=dual_write (session=networkx, store={store.backend_name})")
        return DualWriteSessionGraph(NetworkXSessionGraph(), store)
    if mode == "neo4j_first":
        from server.graph_store import get_graph_store

        store = get_graph_store()
        print(f"[graph] GRAPH_MODE=neo4j_first (source_of_truth={store.backend_name})")
        return Neo4jFirstSessionGraph(store)
    raise ValueError(f"Unsupported session graph backend: {backend or mode}")
