"""Streamlit 自訂元件：瀏覽器語音辨識（iframe 具麥克風權限）。"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")

_speech_input = components.declare_component("speech_input", path=_FRONTEND)


def speech_to_text(
    lang: str = "zh-TW",
    start_label: str = "🎤 開始錄音",
    stop_label: str = "⏹ 停止",
    idle_hint: str = "尚未開始（按左側藍色按鈕）",
    messages: dict[str, str] | None = None,
    key=None,
    height: int = 120,
) -> str:
    """回傳辨識文字；空字串表示尚未錄音或失敗。"""
    msgs: dict[str, Any] = dict(messages or {})
    msgs.setdefault("idle_hint", idle_hint)
    value = _speech_input(
        lang=lang,
        start_label=start_label,
        stop_label=stop_label,
        idle_hint=idle_hint,
        messages=json.dumps(msgs, ensure_ascii=False),
        key=key,
        default="",
        height=height,
    )
    return (value or "").strip()
