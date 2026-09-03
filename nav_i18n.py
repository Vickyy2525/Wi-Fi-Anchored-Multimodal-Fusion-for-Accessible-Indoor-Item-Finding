"""導航／軌跡圖中英字串（不依賴 Streamlit）。"""

from __future__ import annotations

NAV_STRINGS: dict[str, dict[str, str]] = {
    "imu_ghost": {"zh": "純 IMU（對照）", "en": "Pure IMU (reference)"},
    "wifi_path": {"zh": "Wi-Fi 骨架 + IMU 段", "en": "Wi-Fi skeleton + IMU segments"},
    "midpoint": {"zh": "中點", "en": "Midpoint"},
    "start": {"zh": "起點", "en": "Start"},
    "end": {"zh": "終點", "en": "End"},
    "wifi_scan": {"zh": "Wi-Fi 掃描", "en": "Wi-Fi scan"},
    "landmark_prefix": {"zh": "地標：{label}", "en": "Landmark: {label}"},
    "nearby_objects": {"zh": "附近物體", "en": "Nearby objects"},
    "search_here": {"zh": "在這裡找", "en": "Search here"},
    "search_here_query": {"zh": "在這裡找：{query}", "en": "Search here: {query}"},
    "other_candidates": {"zh": "其他可能", "en": "Other candidates"},
    "back_to_landmark_dir": {"zh": "回地標方向", "en": "Direction to landmark"},
    "walk_back_to": {"zh": "走回「{label}」", "en": "Walk back to \"{label}\""},
    "back_path_segment": {"zh": "回地標路徑段", "en": "Path back to landmark"},
    "wifi_fingerprint_bind": {"zh": "Wi-Fi 指紋綁定", "en": "Wi-Fi fingerprint bind"},
    "item_wifi_bind": {"zh": "物品↔Wi-Fi 綁定", "en": "Item ↔ Wi-Fi bind"},
    "wifi_fp_ap": {"zh": "Wi-Fi 指紋\n({n} AP)", "en": "Wi-Fi fingerprint\n({n} APs)"},
    "map_title": {"zh": "找「{query}」", "en": "Find \"{query}\""},
    "map_title_default": {"zh": "尋物導航圖", "en": "Item navigation map"},
    "axis_lr": {"zh": "左右 →", "en": "Left / Right →"},
    "axis_fwd": {"zh": "前進 ↑", "en": "Forward ↑"},
    "scale_units": {"zh": "約 {n} 單位", "en": "~{n} units"},
    "map_footer_default": {
        "zh": "看紅星位置，並對照候選照片。",
        "en": "Follow the red star and compare candidate photos.",
    },
    "guide_back_prefix": {"zh": "🧭 回地標：", "en": "🧭 Back to landmark: "},
    "dir_back_to_start": {
        "zh": "沿路徑往回（朝起點方向）走到",
        "en": "Walk back along the path (toward the start) to ",
    },
    "dir_forward_to_end": {
        "zh": "沿路徑往前（朝終點方向）走到",
        "en": "Walk forward along the path (toward the end) to ",
    },
    "dir_nearby": {"zh": "就在附近，走到", "en": "You are nearby — go to "},
    "dir_default": {"zh": "先走到", "en": "First go to "},
    "path_units": {"zh": "沿軌跡約 {n:.1f} 單位", "en": "~{n:.1f} units along the path"},
    "wifi_sim_high": {
        "zh": "Wi-Fi 指紋相似度 {pct:.0f}%（愈接近愈表示快到了）",
        "en": "Wi-Fi fingerprint similarity {pct:.0f}% (higher means you are getting close)",
    },
    "wifi_sim_hint": {
        "zh": "可用當下 Wi-Fi 指紋靠近該地標掃描時的 AP 強度分布來確認抵達",
        "en": "Use current Wi-Fi signal strength near the landmark scan to confirm arrival",
    },
    "after_landmark": {
        "zh": "到「{label}」後再對照候選照片找物品",
        "en": "At \"{label}\", compare candidate photos to find the item",
    },
    "rel_no_ts": {
        "zh": "請看圖上紅星「{query}」，並對照旁邊候選照片確認。",
        "en": "See the red star for \"{query}\" and compare candidate photos.",
    },
    "rel_star": {"zh": "紅星＝「{query}」", "en": "Red star = \"{query}\""},
    "rel_path_pct": {"zh": "（路徑約 {pct:.0f}%）", "en": " (~{pct:.0f}% along path)"},
    "rel_back": {"zh": "沿綠線／箭頭往回走到「{label}」", "en": "Follow the green line/arrow back to \"{label}\""},
    "rel_forward": {"zh": "沿綠線／箭頭往前走到「{label}」", "en": "Follow the green line/arrow forward to \"{label}\""},
    "rel_near": {"zh": "就在「{label}」附近", "en": "Near \"{label}\""},
    "rel_tail_wifi": {
        "zh": "，再用 Wi-Fi 指紋相似度確認抵達，最後對照照片找物。",
        "en": ", then confirm arrival with Wi-Fi similarity and match photos.",
    },
    "rel_tail_photo": {"zh": "。請搭配候選照片確認場景。", "en": ". Compare candidate photos to verify the scene."},
    "landmark_default": {"zh": "地標", "en": "landmark"},
    "north": {"zh": "北 ↑", "en": "N ↑"},
}

LANDMARK_LABEL_EN: dict[str, str] = {
    "door": "Door",
    "window": "Window",
    "refrigerator": "Fridge",
    "air_conditioner": "Air conditioner",
    "washing_machine": "Washing machine",
    "stove": "Stove",
    "water_dispenser": "Water dispenser",
    "television": "TV",
    "wardrobe": "Wardrobe",
    "bookshelf": "Bookshelf",
    "sofa": "Sofa",
    "desk": "Desk",
    "cabinet": "Cabinet",
    "sink": "Sink",
    "toilet": "Toilet",
    "stairs": "Stairs",
}


def nav_t(key: str, lang: str = "zh", **kwargs) -> str:
    entry = NAV_STRINGS.get(key, {})
    text = entry.get(lang) or entry.get("zh") or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def display_landmark_label(lm: dict, lang: str = "zh") -> str:
    raw = str(lm.get("label") or lm.get("name") or "?")
    if lang != "en":
        return raw
    lid = str(lm.get("id") or "")
    base = LANDMARK_LABEL_EN.get(lid)
    if not base and raw.startswith("疑似"):
        return f"Possible {raw[2:]}"
    if base:
        if lm.get("confidence") == "weak" or raw.startswith("疑似"):
            return f"Possible {base}"
        return base
    return raw
