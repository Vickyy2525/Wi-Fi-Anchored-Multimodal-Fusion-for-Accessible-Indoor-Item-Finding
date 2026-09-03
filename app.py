"""
多模態空間記憶尋物系統 — Streamlit Web Demo
"""

import os
import json

import streamlit as st

from accessibility import (
    get_profile,
    inject_profile_css,
    maybe_speak,
    profile_quick_items,
    render_profile_switcher,
    render_voice_input_bar,
)
from care_service import render_care_panel
from complete_fusion_system import (
    DATA_FOLDER,
    SpatialMemorySystem,
    _json_default,
    discover_dataset_candidates,
    is_dataset_folder,
    resolve_data_folder,
)
from i18n import get_lang, t
from vlm_backend import VLM_BACKEND

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_results")
QUICK_FIND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "quick_find_items.json")
DEFAULT_QUICK_ITEMS = ["一瓶水", "鑰匙", "眼鏡", "藥", "遙控器", "充電器", "耳機", "錢包"]


def load_quick_find_items() -> list[str]:
    try:
        with open(QUICK_FIND_PATH, encoding="utf-8") as f:
            data = json.load(f)
        items = [str(x).strip() for x in data if str(x).strip()]
        return items or list(DEFAULT_QUICK_ITEMS)
    except Exception:
        return list(DEFAULT_QUICK_ITEMS)


@st.cache_resource(show_spinner="🧠 Loading models & building memory DB (first run may take minutes)…")
def get_system(data_folder: str):
    system = SpatialMemorySystem.load_models(data_folder)
    system.build_vector_database(use_cache=True)
    return system


def to_json_safe(obj):
    if isinstance(obj, dict):
        return json.loads(json.dumps(obj, ensure_ascii=False, default=_json_default))
    return obj


def _localize_conclusion(conclusion: str) -> str:
    if get_lang() != "en":
        return conclusion
    legacy = {
        "YES (同類別)": t("conclusion_yes_semantic"),
        "YES（同類別）": t("conclusion_yes_semantic"),
    }
    return legacy.get(conclusion, conclusion)


def render_candidate_card(col, candidate, large: bool = False):
    with col:
        img_path = candidate.get("result_image_path")
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(t("no_image"))

        rank = candidate.get("rank", "?")
        title = f"### {t('rank_candidate', n=rank)}" if large else f"**{t('rank_candidate', n=rank)}**"
        st.markdown(title)

        conclusion = _localize_conclusion(candidate.get("conclusion", "UNKNOWN"))
        sep = t("colon_sep")
        if candidate.get("bound"):
            st.success(f"✅ {t('verdict')}{sep}{conclusion}")
        elif conclusion == t("conclusion_no"):
            st.error(f"❌ {t('verdict')}{sep}{conclusion}")
        elif conclusion in ("NO", t("conclusion_no")):
            st.error(f"❌ {t('verdict')}{sep}{conclusion}")
        else:
            st.info(f"⚠️ {t('verdict')}{sep}{conclusion}")

        st.caption(
            f"📍 {candidate.get('location_label', '—')}  "
            f"| {t('visual_score')} {candidate.get('score', 0):.4f}  "
            f"| {t('recognized')}{sep}{candidate.get('recognized_item') or '—'}"
        )

        with st.expander(t("llava_full"), expanded=False):
            st.text(candidate.get("llava_answer", ""))


def main():
    st.set_page_config(
        page_title="Multimodal Agent",
        page_icon="🧠",
        layout="wide",
    )

    profile = render_profile_switcher()
    inject_profile_css(profile)

    candidates = discover_dataset_candidates()
    default_folder = resolve_data_folder(DATA_FOLDER)
    if default_folder not in candidates:
        candidates = [default_folder] + candidates

    # Prefer env override, then newest discovered dataset
    env_folder = os.environ.get("DATA_FOLDER", "").strip()
    if env_folder:
        env_abs = os.path.abspath(env_folder)
        if env_abs not in candidates and is_dataset_folder(env_abs):
            candidates = [env_abs] + candidates

    init_folder = default_folder
    if env_folder and is_dataset_folder(os.path.abspath(env_folder)):
        init_folder = os.path.abspath(env_folder)
    elif candidates:
        init_folder = candidates[0]

    if "dataset_selectbox" not in st.session_state:
        st.session_state.dataset_selectbox = (
            init_folder if init_folder in candidates else candidates[0]
        )
    elif st.session_state.dataset_selectbox not in candidates:
        st.session_state.dataset_selectbox = candidates[0]

    data_folder = os.path.abspath(st.session_state.dataset_selectbox)

    with st.sidebar:
        st.title(f"🧠 {t('sidebar_title')}")
        st.markdown("---")
        st.markdown("**📂 Dataset**")
        if candidates:
            selected = st.selectbox(
                t("dataset"),
                options=candidates,
                format_func=lambda p: os.path.basename(p),
                key="dataset_selectbox",
            )
            data_folder = os.path.abspath(selected)
            st.caption(data_folder)
            st.markdown(f"**{t('dataset_active')}:** `{os.path.basename(data_folder)}`")
            if st.session_state.get("_active_dataset") != data_folder:
                get_system.clear()
                st.session_state._active_dataset = data_folder
            if st.button(f"🔄 {t('dataset_reload')}", use_container_width=True, key="reload_dataset_btn"):
                get_system.clear()
                st.session_state._active_dataset = data_folder
                st.rerun()
            st.caption(t("dataset_reload_hint"))
        else:
            st.code(data_folder, language=None)

        if not os.path.exists(data_folder):
            st.error(f"❌ {t('dataset_folder_missing')}")
            st.stop()

        try:
            system = get_system(data_folder)
            db_size = len(system.db_df) if system.db_df is not None else 0
            wifi_pts = len(system.df_wifi) if system.df_wifi is not None else 0
            st.success(f"✅ {t('models_ok')}")
            st.success(f"✅ {t('db_ok', n=db_size)}")
            st.info(f"📡 {t('wifi_pts')}：{wifi_pts}")
            st.caption(f"{t('device_label')}：{system.device}")
            backend_label = {
                "mock": "🧪 Mock",
                "ollama": "🦙 Ollama LLaVA",
                "huggingface": "🤗 Hugging Face",
            }.get(VLM_BACKEND, VLM_BACKEND)
            st.info(f"VLM: {backend_label}")
        except Exception as exc:
            st.error(f"Init failed: {exc}")
            st.stop()

        st.markdown("---")
        st.caption("SigLIP + YOLO + LLaVA + Wi-Fi")

    render_care_panel(profile)

    page_title = profile.get("page_title") or t("title_default")
    page_sub = profile.get("page_subtitle") or t("sub_default")
    st.markdown(f'<p class="main-title">🔍 {page_title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{page_sub}</p>', unsafe_allow_html=True)

    if "query" not in st.session_state:
        st.session_state.query = "Water bottle" if get_lang() == "en" else "一瓶水"
    if "extra_quick_items" not in st.session_state:
        st.session_state.extra_quick_items = []

    render_voice_input_bar(profile)

    if st.session_state.get("speak_query"):
        maybe_speak(
            st.session_state.speak_query,
            profile,
            force=bool(st.session_state.pop("speak_force", False)),
        )
        del st.session_state["speak_query"]

    preset_items = profile_quick_items(profile, load_quick_find_items())
    quick_items = list(dict.fromkeys(preset_items + st.session_state.extra_quick_items))

    st.markdown(f"**{t('quick_items')}**")
    ncols = 3 if profile.get("font_scale", 1) >= 1.35 else 4
    cols = st.columns(ncols)
    for i, item in enumerate(quick_items):
        if cols[i % ncols].button(item, key=f"quick_{item}", use_container_width=True):
            st.session_state.query = item
            if profile.get("voice_output"):
                maybe_speak(item, profile)

    with st.expander(t("add_item_expander"), expanded=False):
        st.caption(t("add_item_hint"))
        new_item = st.text_input(t("item_name"), key="new_quick_item")
        add_col, clear_col = st.columns(2)
        if add_col.button(t("add_to_bar"), use_container_width=True):
            name = (new_item or "").strip()
            if not name:
                st.warning(t("need_query"))
            elif name in quick_items:
                st.session_state.query = name
            else:
                st.session_state.extra_quick_items.append(name)
                st.session_state.query = name
                st.rerun()
        if clear_col.button(t("clear_my_items"), use_container_width=True):
            st.session_state.extra_quick_items = []
            st.rerun()

    query = st.text_input(t("search_input"), key="query", placeholder=t("search_placeholder"))
    btn_label = t("search_btn") if profile.get("id") == "default" else t("search_btn_short")
    auto_search = st.session_state.pop("auto_search_trigger", False)
    search_clicked = st.button(f"🚀 {btn_label}", type="primary", use_container_width=True) or auto_search

    if search_clicked:
        if not query.strip():
            st.warning(t("need_query"))
            st.stop()

        with st.spinner(t("searching", q=query.strip())):
            try:
                result = system.search_and_verify(
                    query.strip(),
                    top_k=3,
                    results_dir=RESULTS_DIR,
                    verbose=False,
                    ui_lang=get_lang(),
                )
            except Exception as exc:
                st.error(f"{t('search_fail')}：{exc}")
                st.stop()

        if result.get("error"):
            st.error(result["error"])
            st.stop()

        bound = [c for c in result.get("candidates", []) if c.get("bound")]
        if profile.get("voice_output"):
            if bound:
                best = max(bound, key=lambda c: c.get("score") or 0)
                loc = best.get("location_label", "—")
                maybe_speak(t("speak_found", query=query, loc=loc), profile)
            else:
                maybe_speak(t("speak_not_confirmed", query=query), profile)

        st.markdown("---")
        st.subheader(f"🧩 {t('block_a')}")
        st.caption(t("block_a_cap"))

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown(f"**{t('block_a_memory')}**")
            obj_loc = result.get("object_location")
            if isinstance(obj_loc, dict):
                st.json(to_json_safe(obj_loc))
            else:
                st.warning(str(obj_loc))

        with col_a2:
            st.markdown(f"**{t('block_a_wifi')}**")
            wifi_bind = result.get("wifi_binding")
            if isinstance(wifi_bind, dict):
                fp = wifi_bind.get("wifi_fingerprint") or {}
                st.success(
                    f"{t('wifi_bound')}: {wifi_bind.get('item')} ↔ ts {wifi_bind.get('wifi_timestamp')} ↔ {len(fp)} APs"
                )
                if isinstance(fp, dict) and fp:
                    ranked = sorted(
                        ((str(b), float(lv)) for b, lv in fp.items()),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                    for bssid, level in ranked:
                        st.caption(f"`{bssid}` → {level:.0f} dBm")
                with st.expander("JSON", expanded=False):
                    st.json(to_json_safe(wifi_bind))
            else:
                st.warning(str(wifi_bind))
                st.caption(t("wifi_not_bound_hint"))

        stats = result.get("graph_stats", {})
        st.info(
            f"{t('memories_written')} {result.get('confirmed_count', 0)}  "
            f"| {t('nodes')} {stats.get('nodes', 0)}  "
            f"| {t('edges')} {stats.get('edges', 0)}"
        )

        st.markdown("---")
        st.subheader(f"🗺️ {t('block_c')}")
        st.caption(t("block_c_cap"))

        rel = result.get("relative_description")
        wifi_guide = result.get("wifi_backtrack_guide")
        if wifi_guide:
            st.success(wifi_guide)
            extra = (rel or "").replace(wifi_guide, "", 1).strip(" \n;；")
            if extra:
                st.caption(extra)
        elif rel:
            st.success(f"{t('how_to_find')}: {rel}")

        traj_path = result.get("trajectory_map_path")
        candidates = result.get("candidates", [])
        bound_list = [c for c in candidates if c.get("bound")]
        best_img = None
        if bound_list:
            best = max(bound_list, key=lambda c: c.get("score") or 0)
            best_img = best.get("result_image_path")
        elif candidates:
            best_img = candidates[0].get("result_image_path")

        col_map, col_thumb = st.columns([1.4, 1.0])
        with col_map:
            if traj_path and os.path.exists(traj_path):
                st.image(traj_path, use_container_width=True)
            else:
                st.warning(t("no_traj"))
        with col_thumb:
            st.markdown(f"**{t('compare_photo')}**")
            if best_img and os.path.exists(best_img):
                st.image(best_img, use_container_width=True)
            else:
                st.caption("—")
            st.markdown(t("map_read_steps"))

        st.markdown("---")
        st.subheader(f"👁️ {t('block_b')}")
        st.caption(t("block_b_cap"))

        if not candidates:
            st.warning(t("no_candidates"))
        else:
            cols = st.columns(3)
            large_cards = profile.get("font_scale", 1) >= 1.35
            for idx, candidate in enumerate(candidates[:3]):
                render_candidate_card(cols[idx], candidate, large=large_cards)


if __name__ == "__main__":
    main()
