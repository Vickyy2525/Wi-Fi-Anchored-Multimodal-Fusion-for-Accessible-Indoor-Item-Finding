"""UI 中英雙語字串。"""

from __future__ import annotations

import streamlit as st

STRINGS: dict[str, dict[str, str]] = {
    "lang_zh": {"zh": "中文", "en": "中文"},
    "lang_en": {"zh": "English", "en": "English"},
    "profile_bar_title": {"zh": "使用者模式", "en": "User mode"},
    "profile_bar_hint": {
        "zh": "頂部快速切換：長輩大字、視覺語音、記憶照護。語音可說「長輩模式」「English」。",
        "en": "Switch modes: Elder-friendly, Vision/voice, Memory care. Say profile name or “English”.",
    },
    "current_profile": {"zh": "目前", "en": "Active"},
    "voice_enable": {"zh": "啟用語音輸入", "en": "Enable voice input"},
    "voice_title": {"zh": "語音輸入", "en": "Voice input"},
    "voice_hint": {
        "zh": "按「開始錄音」→ 允許麥克風 → 說完按「停止」；結果會自動填入下方。",
        "en": "Click Start → allow microphone → speak → Stop. Text fills in below automatically.",
    },
    "voice_text_label": {
        "zh": "語音／文字（可編輯）",
        "en": "Voice / text (editable)",
    },
    "voice_apply": {"zh": "套用到搜尋", "en": "Apply to search"},
    "voice_read": {"zh": "朗讀搜尋詞", "en": "Read search term"},
    "voice_switched": {"zh": "已依語音切換模式", "en": "Profile switched by voice"},
    "voice_recognized": {"zh": "✅ 語音辨識成功", "en": "✅ Voice recognized"},
    "voice_empty": {"zh": "尚未辨識到語音（按藍色「開始錄音」→ 允許麥克風 → 「停止」）", "en": "No speech yet (Start → allow mic → Stop)"},
    "voice_browser_hint": {
        "zh": "請用 Chrome 或 Edge 開啟，並允許麥克風。",
        "en": "Use Chrome or Edge and allow microphone access.",
    },
    "voice_start_btn": {"zh": "🎤 開始錄音", "en": "🎤 Start recording"},
    "voice_stop_btn": {"zh": "⏹ 停止", "en": "⏹ Stop"},
    "voice_idle_hint": {
        "zh": "尚未開始（按左側藍色按鈕）",
        "en": "Not started (press the blue Start button)",
    },
    "voice_component_hint": {
        "zh": "結果會同步到下方搜尋欄",
        "en": "Result syncs to the search box below",
    },
    "voice_requesting_mic": {"zh": "正在請求麥克風權限…", "en": "Requesting microphone…"},
    "voice_recording": {"zh": "🔴 正在錄音… 說完請按「停止」。", "en": "🔴 Recording… press Stop when done."},
    "voice_done": {"zh": "✅ 完成！已同步到下方搜尋欄。", "en": "✅ Done! Synced to the search box below."},
    "voice_no_speech": {"zh": "已停止（沒有辨識到語音）。", "en": "Stopped (no speech detected)."},
    "voice_mic_blocked": {
        "zh": "麥克風被阻擋：請在網址列允許麥克風後重新整理。",
        "en": "Microphone blocked: allow it in the address bar, then refresh.",
    },
    "voice_browser_unsupported": {
        "zh": "此瀏覽器不支援語音，請改用 Chrome 或 Edge。",
        "en": "Speech not supported — use Chrome or Edge.",
    },
    "voice_stopped": {"zh": "已停止錄音。", "en": "Recording stopped."},
    "voice_no_sound": {"zh": "沒聽到聲音，請再試一次。", "en": "No speech heard — try again."},
    "voice_confirm_prompt": {
        "zh": "🔊 您要找的是「{item}」嗎？再按一次 **開始錄音**，說「好」或「開始搜尋」即自動搜尋；說「不對」可重錄。",
        "en": "🔊 Looking for **{item}**? Press **Start recording** again and say **OK** or **start search** to search; say **wrong** to re-record.",
    },
    "voice_confirm_speak": {
        "zh": "您要找的是{item}嗎？說好或開始搜尋確認，說不對重新錄音。",
        "en": "Looking for {item}? Say OK or start search to confirm, or say wrong to record again.",
    },
    "voice_searching": {"zh": "✅ 已確認，開始搜尋…", "en": "✅ Confirmed — searching…"},
    "voice_cancelled": {"zh": "已取消，請重新錄音。", "en": "Cancelled — please record again."},
    "notify_email_fail": {
        "zh": "自動寄信失敗，請改用下方「開啟郵件 App」按鈕。",
        "en": "Auto email failed — use the “Open email app” button below.",
    },
    "notify_sms_demo": {
        "zh": "自動簡訊未設定，請改用下方「開啟簡訊 App」按鈕。",
        "en": "Auto SMS not set up — use the “Open SMS app” button below.",
    },
    "care_add_and_notify_hint": {
        "zh": "💡 加完家人後，按「我需要幫忙」→ 點 **開啟郵件／簡訊** 按鈕即可通知，**不用改後台**。（進階才需設定自動代寄）",
        "en": "💡 After adding contacts, press Help → tap **Open email/SMS** — no backend setup needed. (Optional: auto-send in advanced settings.)",
    },
    "care_notify_native_hint": {
        "zh": "請點下方按鈕，會開啟您電腦/手機上的郵件或簡訊 App，按「傳送」即可完成通知。",
        "en": "Tap a button below to open your mail or SMS app, then press Send.",
    },
    "care_open_email": {"zh": "📧 用郵件 App 通知 {name}", "en": "📧 Open email app to notify {name}"},
    "care_open_sms": {"zh": "💬 用簡訊 App 通知 {name}", "en": "💬 Open SMS app to notify {name}"},
    "care_need_email": {"zh": "{name}：請補上 Email 才能用郵件通知", "en": "{name}: add an email to notify by mail"},
    "care_need_phone": {"zh": "{name}：請補上手機才能用簡訊通知", "en": "{name}: add a phone number for SMS"},
    "care_need_phone_or_email": {
        "zh": "請至少填手機或 Email 其中一項，才能通知這位家人。",
        "en": "Enter at least a phone number or email so we can notify this contact.",
    },
    "care_auto_send_optional": {"zh": "進階：自動代寄（可選，不必改 .env）", "en": "Advanced: auto-send (optional, no .env edit)"},
    "care_auto_send_hint": {
        "zh": "預設用您的郵件/簡訊 App 手動送出即可。若希望按鈕後「自動代寄」，可在這裡填一次帳號。",
        "en": "By default we open your mail/SMS app. Fill this once only if you want automatic sending.",
    },
    "care_auto_email_enable": {"zh": "啟用自動寄 Email", "en": "Enable automatic email"},
    "care_auto_sms_enable": {"zh": "啟用自動發簡訊（Twilio）", "en": "Enable automatic SMS (Twilio)"},
    "care_save_delivery": {"zh": "儲存自動代寄設定", "en": "Save auto-send settings"},
    "care_auto_email_skip": {"zh": "未啟用自動寄信", "en": "Auto email not enabled"},
    "care_auto_sms_skip": {"zh": "未啟用自動簡訊", "en": "Auto SMS not enabled"},
    "care_email_sent": {"zh": "已自動寄信至 {addr}", "en": "Email auto-sent to {addr}"},
    "care_email_failed": {"zh": "自動寄信失敗：{err}", "en": "Auto email failed: {err}"},
    "care_sms_sent": {"zh": "已自動發簡訊至 {phone}", "en": "SMS auto-sent to {phone}"},
    "quick_items": {"zh": "常會忘的物品（點一下）", "en": "Often forgotten items (tap)"},
    "add_item_expander": {"zh": "＋ 新增常忘物品", "en": "+ Add custom item"},
    "add_item_hint": {"zh": "依模式顯示不同預設；可自行新增。", "en": "Presets vary by mode; add your own items."},
    "item_name": {"zh": "物品名稱", "en": "Item name"},
    "add_to_bar": {"zh": "加入快捷列", "en": "Add shortcut"},
    "clear_my_items": {"zh": "清空我加的項目", "en": "Clear my items"},
    "search_input": {"zh": "要找的物品", "en": "Item to find"},
    "search_placeholder": {"zh": "例如：一瓶水、鑰匙、眼鏡…", "en": "e.g. water bottle, keys, glasses…"},
    "search_btn": {"zh": "啟動空間檢索", "en": "Start search"},
    "search_btn_short": {"zh": "開始找", "en": "Find"},
    "need_query": {"zh": "請先輸入要尋找的物品名稱。", "en": "Enter an item name first."},
    "searching": {"zh": "正在搜尋「{q}」…", "en": 'Searching for "{q}"…'},
    "search_fail": {"zh": "檢索失敗", "en": "Search failed"},
    "no_image": {"zh": "找不到結果圖片", "en": "Result image not found"},
    "rank_candidate": {"zh": "第 {n} 名候選", "en": "Candidate #{n}"},
    "verdict": {"zh": "判定", "en": "Verdict"},
    "location": {"zh": "位置", "en": "Location"},
    "visual_score": {"zh": "視覺吻合度", "en": "Visual score"},
    "recognized": {"zh": "辨認", "en": "Recognized"},
    "llava_full": {"zh": "LLaVA 推理全文", "en": "Full LLaVA reasoning"},
    "block_a": {"zh": "區塊 A · 認知記憶層", "en": "Block A · Memory graph"},
    "block_a_cap": {"zh": "空間記憶圖 + Wi-Fi 指紋綁定", "en": "Spatial memory + Wi-Fi fingerprint binding"},
    "block_b": {"zh": "區塊 B · 視覺感知層", "en": "Block B · Vision layer"},
    "block_b_cap": {"zh": "SigLIP Top-3 + LLaVA 判斷", "en": "SigLIP Top-3 + LLaVA verdict"},
    "block_c": {"zh": "區塊 C · 空間尋物導航圖", "en": "Block C · Navigation map"},
    "block_c_cap": {
        "zh": "虛線＝純 IMU；實線＝Wi-Fi 融合；大紅星＝在這裡找。",
        "en": "Dashed = IMU only; solid = Wi-Fi fused; red star = search here.",
    },
    "back_landmark": {"zh": "怎麼走回地標", "en": "Back to landmark"},
    "how_to_find": {"zh": "怎麼找", "en": "How to find"},
    "compare_photo": {"zh": "對照這張照片", "en": "Compare this photo"},
    "no_traj": {"zh": "找不到 IMU 資料，無法產生軌跡圖。", "en": "No IMU data; cannot draw map."},
    "no_candidates": {"zh": "沒有找到任何視覺候選。", "en": "No visual candidates found."},
    "read_map_3": {"zh": "3 步讀圖", "en": "Read the map in 3 steps"},
    "sidebar_title": {"zh": "多模態認知 Agent", "en": "Multimodal Agent"},
    "dataset": {"zh": "選擇資料集", "en": "Dataset"},
    "dataset_active": {"zh": "目前使用", "en": "Active"},
    "dataset_reload": {"zh": "重新載入資料集", "en": "Reload dataset"},
    "dataset_reload_hint": {
        "zh": "換資料集後若數字沒變，按此鈕（不必重啟整個程式）",
        "en": "If counts do not change after switching, click here (no full restart needed)",
    },
    "models_ok": {"zh": "模型載入完成", "en": "Models loaded"},
    "db_ok": {"zh": "圖形記憶庫已建立 ({n} 筆)", "en": "Visual DB ready ({n} entries)"},
    "wifi_pts": {"zh": "Wi-Fi 指紋時間點", "en": "Wi-Fi scan points"},
    "profile_default": {"zh": "一般", "en": "Standard"},
    "profile_elderly": {"zh": "長輩模式", "en": "Elder-friendly"},
    "profile_vision": {"zh": "視覺輔助", "en": "Vision assist"},
    "profile_dementia": {"zh": "記憶照護", "en": "Memory care"},
    "desc_default": {"zh": "標準介面與字級", "en": "Standard layout and font size"},
    "desc_elderly": {"zh": "更大更粗字體、高對比、常用忘物優先", "en": "Larger bold text, high contrast, common items first"},
    "desc_vision": {"zh": "語音輸入查詢、結果語音播報", "en": "Voice input and spoken results"},
    "desc_dementia": {"zh": "用藥提醒、常忘物、緊急聯絡人", "en": "Medication reminders, items, emergency contacts"},
    "title_default": {"zh": "視障輔助空間尋物系統", "en": "Indoor Item Finder"},
    "title_elderly": {"zh": "尋物助手（長輩模式）", "en": "Item Finder (Elder-friendly)"},
    "title_vision": {"zh": "語音尋物（視覺輔助）", "en": "Voice Item Finder"},
    "title_dementia": {"zh": "記憶照護尋物", "en": "Memory Care Finder"},
    "sub_default": {
        "zh": "點常用物品或輸入名稱；系統用影像＋記憶圖＋軌跡協助尋找。",
        "en": "Tap a common item or type a name; we use vision, memory graph, and path.",
    },
    "sub_elderly": {
        "zh": "大字顯示。點大按鈕找東西，也可語音說物品名稱。",
        "en": "Large text. Tap buttons or use voice to search.",
    },
    "sub_vision": {
        "zh": "用語音說要找什麼，或說模式名稱切換介面。",
        "en": "Say what to find, or say a mode name to switch UI.",
    },
    "sub_dementia": {
        "zh": "提醒吃藥、找常忘物品；需要幫忙可通知家人。",
        "en": "Medication reminders and finding items; alert contacts if needed.",
    },
    # Care panel
    "care_title": {"zh": "照護與安全", "en": "Care & safety"},
    "care_meds": {"zh": "用藥提醒", "en": "Medication reminders"},
    "care_contacts": {"zh": "緊急聯絡人", "en": "Emergency contacts"},
    "care_taken": {"zh": "我吃了", "en": "I took it"},
    "care_overdue": {"zh": "已過時未記錄服用", "en": "Overdue — not marked taken"},
    "care_due_soon": {"zh": "即將服用", "en": "Due soon"},
    "care_ok": {"zh": "狀態正常", "en": "On track"},
    "care_last": {"zh": "上次", "en": "Last taken"},
    "care_not_recorded": {"zh": "尚未記錄", "en": "Not recorded"},
    "care_notify_missed": {"zh": "通知家人：尚未按時吃藥", "en": "Notify contacts: missed medication"},
    "care_emergency_btn": {"zh": "我需要幫忙（通知所有聯絡人）", "en": "I need help (notify all contacts)"},
    "care_manage": {"zh": "管理聯絡人與用藥", "en": "Manage contacts & medication"},
    "care_save_path": {"zh": "設定儲存於 config/care_settings.local.json", "en": "Saved to config/care_settings.local.json"},
    "care_name": {"zh": "姓名", "en": "Name"},
    "care_phone": {"zh": "手機", "en": "Phone"},
    "care_email": {"zh": "Email", "en": "Email"},
    "care_add_contact": {"zh": "新增聯絡人", "en": "Add contact"},
    "care_delete": {"zh": "刪除", "en": "Delete"},
    "care_med_name": {"zh": "藥品名稱", "en": "Medicine name"},
    "care_med_times": {"zh": "每日時段（逗號分隔）", "en": "Daily times (comma-separated)"},
    "care_add_med": {"zh": "新增用藥提醒", "en": "Add medication reminder"},
    "care_smtp_ok": {"zh": "已啟用自動寄 Email", "en": "Automatic email enabled"},
    "care_smtp_missing": {
        "zh": "未啟用自動寄信（仍可用下方「開啟郵件 App」通知）",
        "en": "Auto email off (you can still use “Open email app” below)",
    },
    "care_sms_missing": {
        "zh": "未啟用自動簡訊（仍可用下方「開啟簡訊 App」通知）",
        "en": "Auto SMS off (you can still use “Open SMS app” below)",
    },
    "care_dup_contact": {"zh": "此聯絡人已存在（Email 或手機相同）", "en": "Contact already exists (same email or phone)"},
    "care_added": {"zh": "已新增", "en": "Added"},
    "care_deleted": {"zh": "已刪除", "en": "Deleted"},
    "care_no_contacts": {"zh": "尚無聯絡人，請先新增", "en": "No contacts yet — add one first"},
    "memories_written": {"zh": "本次查詢寫入", "en": "Memories written"},
    "nodes": {"zh": "節點", "en": "nodes"},
    "edges": {"zh": "邊", "en": "edges"},
    "wifi_bound": {"zh": "已綁定", "en": "Bound"},
    "wifi_not_bound_hint": {
        "zh": "若未綁定：LLaVA 未確認；同類物品（杯/瓶）可語意補綁。",
        "en": "If unbound: LLaVA rejected candidate; similar items (cup/bottle) may still bind semantically.",
    },
    "landmarks_list": {"zh": "地標清單", "en": "Landmarks"},
    "landmark_weak": {"zh": "疑似", "en": "Weak"},
    "landmark_high": {"zh": "高信心", "en": "High confidence"},
    "fusion_diag": {"zh": "融合診斷", "en": "Fusion diagnostics"},
    "traj_fusion": {"zh": "軌跡融合", "en": "Path fusion"},
    "device_label": {"zh": "運算裝置", "en": "Device"},
    "dataset_folder_missing": {"zh": "找不到資料集資料夾", "en": "Dataset folder not found"},
    "init_failed": {"zh": "系統初始化失敗", "en": "System init failed"},
    "conclusion_yes": {"zh": "YES", "en": "YES"},
    "conclusion_no": {"zh": "NO", "en": "NO"},
    "conclusion_unknown": {"zh": "UNKNOWN", "en": "UNKNOWN"},
    "conclusion_yes_semantic": {"zh": "YES（同類別）", "en": "YES (same category)"},
    "bind_reason_yes": {"zh": "結論 YES", "en": "Conclusion YES"},
    "bind_reason_semantic": {
        "zh": "同類別比對命中（辨認為「{item}」）",
        "en": 'Same-category match (recognized as "{item}")',
    },
    "err_empty_db": {"zh": "向量資料庫為空，無法檢索。", "en": "Visual database is empty; cannot search."},
    "err_no_wifi_bind": {
        "zh": "沒有找到與 [{item}] 對應的 Wi-Fi 綁定。",
        "en": "No Wi-Fi binding found for [{item}].",
    },
    "graph_not_found": {
        "zh": "抱歉，記憶庫中沒有關於 [{item}] 的紀錄。",
        "en": "Sorry, no memory record for [{item}].",
    },
    "graph_no_wifi": {
        "zh": "找到了 [{item}]，但無法推導出其 Wi-Fi 空間特徵。",
        "en": "Found [{item}], but could not infer Wi-Fi spatial features.",
    },
    "graph_relation": {
        "zh": "位於 [{loc}]（推導關係：{rel}）",
        "en": "At [{loc}] (inferred relation: {rel})",
    },
    "graph_time_unknown": {"zh": "未知", "en": "unknown"},
    "block_a_memory": {"zh": "📍 記憶圖推理", "en": "📍 Memory graph"},
    "block_a_wifi": {"zh": "📡 Wi-Fi 指紋綁定", "en": "📡 Wi-Fi binding"},
    "map_read_steps": {
        "zh": "**3 步讀圖：** 1) 紅星＝物品  2) 綠線＝回地標  3) 紫三角＝Wi-Fi",
        "en": "**3 steps:** 1) Red star = item  2) Green line = back to landmark  3) Purple = Wi-Fi bind",
    },
    "speak_found": {
        "zh": "找到{query}，位置在{loc}，請看導航圖。",
        "en": "Found {query} near {loc}. See the navigation map.",
    },
    "speak_not_confirmed": {
        "zh": "尚未確認{query}，請看候選照片。",
        "en": "Not confirmed: {query}. See candidate photos.",
    },
    "colon_sep": {"zh": "：", "en": ": "},
}

PROFILE_I18N_KEYS = {
    "default": ("profile_default", "desc_default", "title_default", "sub_default"),
    "elderly": ("profile_elderly", "desc_elderly", "title_elderly", "sub_elderly"),
    "vision": ("profile_vision", "desc_vision", "title_vision", "sub_vision"),
    "dementia": ("profile_dementia", "desc_dementia", "title_dementia", "sub_dementia"),
}


def get_lang() -> str:
    lang = st.session_state.get("ui_lang", "zh")
    return "en" if lang == "en" else "zh"


def set_lang(lang: str) -> None:
    st.session_state.ui_lang = "en" if lang == "en" else "zh"


def t(key: str, **kwargs) -> str:
    return t_lang(key, get_lang(), **kwargs)


def t_lang(key: str, lang: str, **kwargs) -> str:
    lang = "en" if lang == "en" else "zh"
    entry = STRINGS.get(key, {})
    text = entry.get(lang) or entry.get("zh") or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def profile_text(profile_id: str, field: str) -> str:
    """field: label | description | page_title | page_subtitle"""
    keys = PROFILE_I18N_KEYS.get(profile_id, PROFILE_I18N_KEYS["default"])
    mapping = {"label": 0, "description": 1, "page_title": 2, "page_subtitle": 3}
    idx = mapping.get(field, 0)
    return t(keys[idx])


def speech_lang_code() -> str:
    return "en-US" if get_lang() == "en" else "zh-TW"
