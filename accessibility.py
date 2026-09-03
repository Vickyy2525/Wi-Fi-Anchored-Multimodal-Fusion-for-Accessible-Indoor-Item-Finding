"""使用者情境（無障礙／長輩／視覺／照護）設定與 UI 輔助。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from i18n import get_lang, profile_text, set_lang, speech_lang_code, t

REMOTE_ROOT = Path(__file__).resolve().parent
PROFILES_PATH = REMOTE_ROOT / "config" / "user_profiles.json"


def load_profile_config() -> dict[str, Any]:
    try:
        return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"profiles": [], "default_profile": "default", "voice_switch_phrases": {}}


def list_profiles() -> list[dict[str, Any]]:
    cfg = load_profile_config()
    return cfg.get("profiles") or []


def get_profile(profile_id: str | None) -> dict[str, Any]:
    profiles = list_profiles()
    pid = profile_id or load_profile_config().get("default_profile", "default")
    for p in profiles:
        if p.get("id") == pid:
            out = dict(p)
            out["label"] = profile_text(pid, "label")
            out["description"] = profile_text(pid, "description")
            out["page_title"] = profile_text(pid, "page_title")
            out["page_subtitle"] = profile_text(pid, "page_subtitle")
            return out
    return {
        "id": "default",
        "label": t("profile_default"),
        "font_scale": 1.0,
        "quick_items": [],
        "voice_input": False,
        "voice_output": False,
        "medication_panel": False,
        "emergency_panel": False,
        "page_title": t("title_default"),
        "page_subtitle": t("sub_default"),
    }


def resolve_profile_from_voice(text: str) -> str | None:
    if not text:
        return None
    raw = str(text).strip()
    t_norm = raw.replace(" ", "").lower()
    if re.search(r"\benglish\b|英文|切换英文|切換英文", raw, re.I):
        set_lang("en")
        return None
    if re.search(r"中文|chinese|切换中文|切換中文", raw, re.I):
        set_lang("zh")
        return None
    phrases = load_profile_config().get("voice_switch_phrases") or {}
    for phrase in sorted(phrases.keys(), key=len, reverse=True):
        if phrase.replace(" ", "") in t_norm or phrase.lower() in raw.lower():
            return phrases[phrase]
    en_map = {
        "standard": "default",
        "elder": "elderly",
        "elderly": "elderly",
        "vision": "vision",
        "memory": "dementia",
        "dementia": "dementia",
    }
    for word, pid in en_map.items():
        if word in raw.lower():
            return pid
    return None


def is_voice_confirm(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    norm = raw.lower().rstrip("。.!！?？")
    if get_lang() == "en":
        return bool(
            re.match(
                r"^(ok|okay|yes|yeah|yep|sure|search|start search|go|confirm|"
                r"find it|find|that's right|correct|go ahead)$",
                norm,
                re.I,
            )
        ) or norm in ("start searching",)
    compact = raw.replace(" ", "")
    return bool(
        re.match(
            r"^(好|好的|沒問題|可以|對|對的|確認|開始搜尋|搜尋|開始找|找吧|沒錯|行|嗯|開始)$",
            compact,
        )
    )


def is_voice_cancel(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if get_lang() == "en":
        return bool(re.search(r"\b(wrong|no|cancel|again|retry|not right|incorrect)\b", raw, re.I))
    return bool(re.search(r"不對|錯了|重新|再來|取消|不是", raw))


def _voice_component_messages() -> dict[str, str]:
    return {
        "idle_hint": t("voice_idle_hint"),
        "requesting_mic": t("voice_requesting_mic"),
        "recording": t("voice_recording"),
        "done": t("voice_done"),
        "no_speech": t("voice_no_speech"),
        "mic_blocked": t("voice_mic_blocked"),
        "browser_unsupported": t("voice_browser_unsupported"),
        "stopped": t("voice_stopped"),
        "no_sound": t("voice_no_sound"),
    }


def _process_voice_transcript(transcript: str) -> None:
    """依語音文字更新 session_state（需在外層 rerun）。"""
    before_lang = get_lang()
    before_profile = st.session_state.get("user_profile")

    pid = resolve_profile_from_voice(transcript)
    if get_lang() != before_lang:
        return
    if pid:
        if pid != before_profile:
            st.session_state.user_profile = pid
        return

    stage = st.session_state.get("voice_stage", "idle")

    if is_voice_cancel(transcript):
        st.session_state.voice_stage = "idle"
        st.session_state.voice_draft = ""
        st.session_state.speak_query = t("voice_cancelled")
        st.session_state.speak_force = True
        return

    if is_voice_confirm(transcript):
        draft = (st.session_state.get("voice_draft") or "").strip()
        if stage == "awaiting_confirm" and draft:
            st.session_state.query = draft
            st.session_state.auto_search_trigger = True
            st.session_state.voice_stage = "idle"
            st.session_state.speak_query = t("voice_searching")
            st.session_state.speak_force = True
        return

    item = transcript.strip()
    if not item:
        return

    st.session_state.voice_draft = item
    st.session_state.query = item
    st.session_state.voice_stage = "awaiting_confirm"
    st.session_state.speak_query = t("voice_confirm_speak", item=item)
    st.session_state.speak_force = True


def inject_profile_css(profile: dict[str, Any]) -> None:
    scale = float(profile.get("font_scale") or 1.0)
    weight = int(profile.get("font_weight") or 400)
    hc = bool(profile.get("high_contrast"))
    title_w = max(weight, 700)
    btn_w = max(weight, 600)
    border_w = 3 if hc else 2

    st.markdown(
        f"""
        <style>
        .main-title {{
            font-size: {2.2 * scale}rem !important;
            font-weight: {title_w} !important;
            letter-spacing: -0.02em;
        }}
        .sub-title {{
            font-size: {1.05 * scale}rem !important;
            font-weight: {500 if hc else 400};
        }}
        [data-theme="light"] .main-title {{ color: #0a0a0a !important; }}
        [data-theme="light"] .sub-title {{ color: #374151 !important; }}
        [data-theme="light"] .profile-bar {{
            background: #eef2f7;
            border: {border_w}px solid #1565C0;
        }}
        [data-theme="dark"] .main-title {{ color: #f9fafb !important; }}
        [data-theme="dark"] .sub-title {{ color: #e5e7eb !important; }}
        [data-theme="dark"] .profile-bar {{
            background: #1f2937;
            border: {border_w}px solid #60a5fa;
        }}
        .profile-bar {{
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
        }}
        div.stButton > button {{
            font-size: {1.05 * scale}rem !important;
            font-weight: {btn_w} !important;
            min-height: {2.6 * scale}rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_language_switcher() -> None:
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "zh"

    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button(
            t("lang_zh"),
            key="lang_zh_btn",
            type="primary" if get_lang() == "zh" else "secondary",
            use_container_width=True,
        ):
            set_lang("zh")
            st.rerun()
    with c2:
        if st.button(
            t("lang_en"),
            key="lang_en_btn",
            type="primary" if get_lang() == "en" else "secondary",
            use_container_width=True,
        ):
            set_lang("en")
            st.rerun()


def render_profile_switcher() -> dict[str, Any]:
    cfg = load_profile_config()
    profiles = cfg.get("profiles") or []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = cfg.get("default_profile", "default")
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False

    render_language_switcher()

    st.markdown('<div class="profile-bar">', unsafe_allow_html=True)
    st.markdown(f"#### 🎛️ {t('profile_bar_title')}")
    st.caption(t("profile_bar_hint"))

    cols = st.columns(min(len(profiles), 4))
    for i, p in enumerate(profiles):
        pid = p.get("id", "")
        label = f"{p.get('icon', '')} {profile_text(pid, 'label')}"
        if cols[i].button(
            label,
            key=f"profile_{pid}",
            use_container_width=True,
            type="primary" if st.session_state.user_profile == pid else "secondary",
        ):
            st.session_state.user_profile = pid
            if p.get("voice_input"):
                st.session_state.voice_enabled = True
            st.rerun()

    profile = get_profile(st.session_state.user_profile)
    st.info(f"{t('current_profile')}：**{profile.get('label')}** — {profile.get('description', '')}")

    show_voice = bool(profile.get("voice_input")) or st.session_state.voice_enabled
    if not profile.get("voice_input"):
        st.session_state.voice_enabled = st.checkbox(
            t("voice_enable"),
            value=st.session_state.voice_enabled,
            key="voice_enabled_cb",
        )
        show_voice = st.session_state.voice_enabled

    st.markdown("</div>", unsafe_allow_html=True)
    profile = dict(profile)
    profile["_show_voice"] = show_voice
    return profile


def render_voice_input_bar(profile: dict[str, Any]) -> str:
    if not profile.get("_show_voice") and not profile.get("voice_input"):
        return ""

    st.markdown(f"**🎤 {t('voice_title')}**")
    st.caption(t("voice_hint"))
    st.caption(t("voice_browser_hint"))

    try:
        from components.speech_input import speech_to_text

        transcript = speech_to_text(
            lang=speech_lang_code(),
            start_label=t("voice_start_btn"),
            stop_label=t("voice_stop_btn"),
            idle_hint=t("voice_idle_hint"),
            messages=_voice_component_messages(),
            key=f"main_speech_input_{get_lang()}",
            height=120,
        )
    except Exception as exc:
        st.warning(f"Speech: {exc}")
        transcript = ""

    if transcript and transcript != st.session_state.get("_voice_last_processed"):
        st.session_state._voice_last_processed = transcript
        _process_voice_transcript(transcript)
        st.rerun()

    draft = (st.session_state.get("voice_draft") or "").strip()
    if st.session_state.get("voice_stage") == "awaiting_confirm" and draft:
        st.info(t("voice_confirm_prompt", item=draft))
    elif not transcript:
        st.caption(t("voice_empty"))

    return (st.session_state.get("query") or "").strip()


def maybe_speak(text: str, profile: dict[str, Any], *, force: bool = False) -> None:
    if not text:
        return
    if not force and not profile.get("voice_output"):
        return
    import streamlit.components.v1 as components

    lang = speech_lang_code()
    safe = json.dumps(text, ensure_ascii=False)
    components.html(
        f"""
        <script>
        (function() {{
          const msg = {safe};
          if (!window.speechSynthesis) return;
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(msg);
          u.lang = '{lang}';
          u.rate = 0.92;
          window.speechSynthesis.speak(u);
        }})();
        </script>
        """,
        height=0,
    )


def profile_quick_items(profile: dict[str, Any], fallback: list[str]) -> list[str]:
    pid = profile.get("id", "default")
    if get_lang() == "en":
        en_items = {
            "default": ["Water bottle", "Keys", "Glasses", "Medicine", "Remote", "Charger", "Earphones", "Wallet"],
            "elderly": ["Medicine", "Glasses", "Keys", "Water bottle", "Remote", "Wallet"],
            "vision": ["Glasses", "Medicine", "Keys", "Water bottle", "Earphones", "Charger", "Wallet"],
            "dementia": ["Medicine", "Glasses", "Keys", "Wallet", "Remote", "Phone"],
        }
        return en_items.get(pid, en_items["default"])
    items = profile.get("quick_items") or fallback
    return [str(x).strip() for x in items if str(x).strip()]
