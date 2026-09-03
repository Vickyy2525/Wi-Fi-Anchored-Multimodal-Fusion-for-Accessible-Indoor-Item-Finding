"""In-process scheduler for auto window jobs."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from server import staging
from server.worker import run_window_job

_task: Optional[asyncio.Task] = None
_stop_event: Optional[asyncio.Event] = None
_last_processed: dict[str, int] = {}


def _enabled() -> bool:
    return os.getenv("AUTO_WINDOW_JOB", "true").strip().lower() in {"1", "true", "yes", "on"}


def _interval_sec() -> int:
    raw = os.getenv("AUTO_WINDOW_INTERVAL_SEC", "300").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 300
    return max(30, v)


def _run_once() -> None:
    sessions = staging.list_sessions()
    if not sessions:
        return
    for session_id in sessions:
        wid = staging.latest_complete_window_id(session_id)
        if wid is None:
            continue
        prev = _last_processed.get(session_id)
        if prev is not None and wid <= prev:
            continue
        result = run_window_job(session_id, wid)
        if result.get("ok"):
            _last_processed[session_id] = int(result.get("window_id", wid))
            print(
                f"[scheduler] processed session={session_id} window={wid} "
                f"confirmed={result.get('confirmed_total')} graph={result.get('graph_backend')}"
            )
        else:
            print(f"[scheduler] run failed session={session_id} window={wid}: {result.get('error')}")


async def _loop() -> None:
    assert _stop_event is not None
    interval = _interval_sec()
    print(f"[scheduler] started interval={interval}s")
    while not _stop_event.is_set():
        try:
            _run_once()
        except Exception as exc:  # guard background loop
            print(f"[scheduler] unexpected error: {exc}")
        try:
            await asyncio.wait_for(_stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    print("[scheduler] stopped")


def start_scheduler() -> None:
    global _task, _stop_event
    if not _enabled():
        print("[scheduler] AUTO_WINDOW_JOB disabled")
        return
    if _task is not None and not _task.done():
        return
    _stop_event = asyncio.Event()
    _task = asyncio.create_task(_loop())


async def stop_scheduler() -> None:
    global _task, _stop_event
    if _task is None:
        return
    if _stop_event is not None:
        _stop_event.set()
    try:
        await _task
    finally:
        _task = None
        _stop_event = None
