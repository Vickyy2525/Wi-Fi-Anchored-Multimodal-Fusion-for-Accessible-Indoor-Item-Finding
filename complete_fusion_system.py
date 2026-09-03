"""
多模態空間記憶圖形系統 (Multi-modal Spatial Memory Graph System)
================================================================

本系統融合三大模態，建構一個可被「推理 / 尋物」的空間記憶：

    1. 視覺檢索 (SigLIP + YOLOv8)：把每張 keyframe 切出物件特徵向量，建立向量資料庫。
    2. Wi-Fi 跨模態時間對齊：在建庫時，為每個物件特徵綁定「拍照那一瞬間最近的 Wi-Fi 指紋」。
    3. LLaVA 常識推理 + NetworkX 記憶圖：用視覺語言模型驗證候選，並把
       (Object) -[LOCATED_AT]-> (VLM_Report) -[DESCRIBED_AT]-> (Signal_State)
       的關係實體化進記憶圖，最後可同時用「物件名稱」與「Wi-Fi 綁定」雙路徑查詢。

執行 CLI：
    python complete_fusion_system.py

執行 Web Demo：
    streamlit run app.py
"""

import os
import re
import json
import glob
import pickle
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from vlm_backend import VLM_BACKEND, ask_llava_commonsense
from trajectory import compute_trajectory, generate_trajectory_map, load_imu_data
from landmarks import build_relative_description, scan_structural_landmarks
from i18n import t_lang
from memory_graph_adapter import (
    NetworkXSessionGraph,
    create_session_graph,
)

from PIL import Image, ImageDraw, ImageOps
from transformers import AutoProcessor, AutoModel
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

DATA_FOLDER = "./sensor_data_1780239297777"

SYNONYM_GROUPS = [
    {"水", "水瓶", "水壺", "寶特瓶", "保特瓶", "瓶裝水", "礦泉水", "礦泉水瓶",
     "水杯", "杯子", "杯", "馬克杯", "保溫瓶", "保溫杯", "一瓶水", "bottle", "water", "cup", "mug"},
    {"鑰匙", "鑰匙圈", "key", "keys", "keychain"},
    {"藥", "藥盒", "藥罐", "藥瓶", "藥丸", "medicine", "pill", "pillbox", "medication"},
    {"眼鏡", "眼鏡盒", "墨鏡", "glasses", "eyeglasses", "sunglasses", "spectacles", "glasses case"},
    {"遙控器", "電視遙控器", "remote", "remote control"},
    {"充電器", "充電線", "變壓器", "charger", "cable", "adapter"},
    {"耳機", "有線耳機", "藍牙耳機", "earphone", "earphones", "earbuds", "airpods"},
    {"錢包", "皮夾", "零錢包", "wallet", "purse"},
    {"手機", "電話", "phone", "smartphone", "mobile"},
]


def _json_default(obj):
    """json.dumps 的 fallback：把 numpy / pandas 數值型別轉成原生 Python 型別。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_default(v) for k, v in obj.items()}
    return str(obj)


def is_dataset_folder(path: str) -> bool:
    """判斷是否為可用資料集資料夾（至少有 images + imu/keyframes/wifi 其一）。"""
    if not path:
        return False
    d = Path(path)
    if not d.exists() or not d.is_dir():
        return False
    if not (d / "images").exists():
        return False
    return any(d.glob("imu_*.csv")) and any(d.glob("keyframes_*.csv"))


def discover_dataset_candidates(extra_roots: list[str] | None = None) -> list[str]:
    """
    掃描可用 sensor_data_* 資料集：
      - Remote/ 目錄
      - ../data（舊資料集中繼目錄）
      - 呼叫者提供的額外 roots
    """
    remote_root = Path(__file__).resolve().parent
    project_root = remote_root.parent
    implement_root = project_root.parent
    roots = [
        remote_root,
        implement_root / "data",
    ]
    if extra_roots:
        roots.extend(Path(p) for p in extra_roots if p)

    found: dict[str, float] = {}
    for root in roots:
        if not root.exists():
            continue
        for d in root.glob("sensor_data_*"):
            if is_dataset_folder(str(d)):
                abs_path = str(d.resolve())
                try:
                    mtime = d.stat().st_mtime
                except Exception:
                    mtime = 0.0
                found[abs_path] = max(found.get(abs_path, 0.0), mtime)

    # 新到舊排序，方便預設挑最近可用資料集
    ordered = sorted(found.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p, _ in ordered]


def resolve_data_folder(preferred: str = DATA_FOLDER) -> str:
    """解析最終資料夾：優先 preferred，否則自動挑一個可用資料集。"""
    pref_abs = os.path.abspath(preferred)
    if is_dataset_folder(pref_abs):
        return pref_abs
    candidates = discover_dataset_candidates()
    if candidates:
        return candidates[0]
    return pref_abs


def load_labels_table(data_folder: str) -> pd.DataFrame:
    """
    讀取 labels 資料：
      1) labels.csv
      2) labels_*.csv（舊版命名）
    會自動合併、清理 timestamp、依時間排序。
    """
    files = []
    canonical = os.path.join(data_folder, "labels.csv")
    if os.path.exists(canonical):
        files.append(canonical)
    files.extend(sorted(glob.glob(os.path.join(data_folder, "labels_*.csv"))))
    # 去重（避免 labels.csv 也被 glob 到）
    files = list(dict.fromkeys(files))
    if not files:
        return pd.DataFrame(columns=["timestamp", "label"])

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"⚠️  讀取標籤檔失敗 {os.path.basename(f)}: {exc}")
            continue
        if "timestamp" not in df.columns or "label" not in df.columns:
            print(f"⚠️  略過非標準標籤檔 {os.path.basename(f)}（需含 timestamp,label）")
            continue
        df = df[["timestamp", "label"]].copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "label"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)
    print(f"🏷️ 載入標籤檔 {len(files)} 份，共 {len(merged)} 筆標記。")
    return merged


def _text_matches_synonym_group(text: str, group: set[str]) -> bool:
    """查詢詞或辨認詞是否落在同一同義詞群（含中文短字，如 杯 ↔ 水杯）。"""
    t = str(text).lower().strip()
    if not t:
        return False
    for g in group:
        gl = g.lower()
        if gl in t or t in gl:
            return True
        # 短辨認詞（如「杯」）可能是長同義詞的子串
        if len(t) <= 2 and t in gl:
            return True
    return False


def is_semantic_match(query, item):
    """判斷 LLaVA 辨認出的 item 是否在語意上等同於使用者查詢 query。"""
    if not item or not query:
        return False
    q = str(query).lower().strip()
    it = str(item).lower().strip()
    if not q or not it:
        return False
    if q in it or it in q:
        return True
    for group in SYNONYM_GROUPS:
        gset = {str(x).lower() for x in group}
        if _text_matches_synonym_group(q, gset) and _text_matches_synonym_group(it, gset):
            return True
    return False


def extract_json_from_text(text):
    """從一段文字中，盡力擷取出第一個合法 JSON 物件。"""
    if not text or not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index, ch in enumerate(text[start:], start):
        if ch == "\\" and in_string:
            escape = not escape
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        escape = False
    return None


def interpret_llava_conclusion(answer_text):
    """解讀 LLaVA 的常識推理結論。"""
    text = answer_text or ""

    parsed = extract_json_from_text(text)
    json_location = parsed.get("location") if isinstance(parsed, dict) else None
    json_item = parsed.get("item") if isinstance(parsed, dict) else None

    item = json_item
    if not item:
        m_item = re.search(
            r"(?:物品名稱|Item name)\s*[:：]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if m_item:
            item = m_item.group(1).strip()
            # trim trailing conclusion fragments
            item = re.split(r"\s*(?:Conclusion|結論)\s*[:：]", item, flags=re.I)[0].strip()

    is_yes = None
    matches = re.findall(
        r"(?:結論|Conclusion)\s*[:：]?\s*(YES|NO|是|否|不是)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        last = matches[-1].upper()
        is_yes = last in ("YES", "是")
    else:
        if re.search(r"\bYES\b", text, re.IGNORECASE) or "是的" in text:
            is_yes = True
        elif re.search(r"\bNO\b", text, re.IGNORECASE) or "不是" in text:
            is_yes = False

    return {
        "is_yes": is_yes,
        "location": json_location,
        "item": item,
        "details": text,
    }


def build_wifi_fingerprint_table(df):
    rows = []
    for timestamp, group in df.groupby("timestamp"):
        fingerprint = group.set_index("bssid")["level"].to_dict()
        rows.append({
            "timestamp": int(timestamp),
            "wifi_fingerprint": fingerprint,
            "raw_rows": group.to_dict(orient="records"),
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "wifi_fingerprint", "raw_rows"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def load_wifi_table(data_folder):
    wifi_files = sorted(glob.glob(os.path.join(data_folder, "wifi_*.csv")))
    if not wifi_files:
        print("⚠️  找不到 wifi_*.csv，Wi-Fi 綁定功能將以空指紋運作。")
        return pd.DataFrame(columns=["timestamp", "wifi_fingerprint", "raw_rows"])

    wifi_csv = wifi_files[-1]
    print(f"📡 載入 Wi-Fi 指紋檔：{os.path.basename(wifi_csv)}")
    df_wifi_raw = pd.read_csv(wifi_csv)
    df_wifi_raw["timestamp"] = pd.to_numeric(df_wifi_raw["timestamp"], errors="coerce")
    df_wifi_raw = df_wifi_raw.dropna(subset=["timestamp"])
    df_wifi = build_wifi_fingerprint_table(df_wifi_raw)
    print(f"📡 Wi-Fi 指紋建表完成，共 {len(df_wifi)} 個時間點。")
    return df_wifi


def find_nearest_wifi_fingerprint(kf_timestamp, df_wifi):
    if df_wifi is None or df_wifi.empty:
        return None, {}, None
    time_diff = (df_wifi["timestamp"] - kf_timestamp).abs()
    nearest_idx = time_diff.idxmin()
    row = df_wifi.loc[nearest_idx]
    fingerprint = row["wifi_fingerprint"]
    matched_time = int(row["timestamp"])
    return nearest_idx, fingerprint, matched_time


def add_memory_to_graph(graph, user_query, candidate, vlm_result):
    """Backward-compatible wrapper; prefer SessionMemoryGraph.add_binding."""
    if isinstance(graph, NetworkXSessionGraph):
        graph.add_binding(user_query, candidate, vlm_result)
        return
    # Legacy: raw NetworkX graph passed directly
    session = NetworkXSessionGraph()
    session._graph = graph  # noqa: SLF001 — transitional compat only
    session.add_binding(user_query, candidate, vlm_result)


def bind_wifi_dataframe(df_wifi, wifi_ts, user_query, vlm_result):
    if df_wifi is None or df_wifi.empty or wifi_ts is None:
        return
    for col in ("llava_item", "llava_location", "llava_raw_answer"):
        if col not in df_wifi.columns:
            df_wifi[col] = None
    mask = df_wifi["timestamp"] == wifi_ts
    df_wifi.loc[mask, "llava_item"] = user_query
    df_wifi.loc[mask, "llava_location"] = vlm_result.get("location")
    df_wifi.loc[mask, "llava_raw_answer"] = vlm_result.get("details")


def query_object_location(object_name, graph):
    """Backward-compatible wrapper; prefer SessionMemoryGraph.query_object."""
    if isinstance(graph, NetworkXSessionGraph):
        return graph.query_object(object_name)
    session = NetworkXSessionGraph()
    session._graph = graph  # noqa: SLF001 — transitional compat only
    return session.query_object(object_name)


def query_wifi_by_item(item_name, wifi_df, lang: str = "zh"):
    if wifi_df is None or wifi_df.empty or "llava_item" not in wifi_df.columns:
        return t_lang("err_no_wifi_bind", lang, item=item_name)

    matched = wifi_df[wifi_df["llava_item"].astype(str).str.contains(str(item_name), na=False)]
    if matched.empty:
        return t_lang("err_no_wifi_bind", lang, item=item_name)

    row = matched.iloc[-1]
    return {
        "item": row["llava_item"],
        "location": row["llava_location"],
        "wifi_timestamp": int(row["timestamp"]),
        "wifi_fingerprint": row["wifi_fingerprint"],
        "raw_llava_answer": row.get("llava_raw_answer"),
    }


class SpatialMemorySystem:
    """多模態空間記憶系統：封裝模型、建庫、檢索與記憶圖推理，供 CLI 與 Streamlit 共用。"""

    def __init__(self, data_folder, clip_model, processor, yolo_model, device):
        self.data_folder = os.path.abspath(data_folder)
        self.clip_model = clip_model
        self.processor = processor
        self.yolo_model = yolo_model
        self.device = device
        self.db_df = None
        self.df_wifi = None
        self.session_graph = create_session_graph()

    @classmethod
    def load_models(cls, data_folder=DATA_FOLDER):
        """載入 SigLIP + YOLO 模型（供 @st.cache_resource 快取）。"""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🚀 運算裝置: {device}")

        print("📦 載入 Google SigLIP 視覺引擎...")
        model_id = "google/siglip-base-patch16-224"
        clip_model = AutoModel.from_pretrained(model_id).to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        print("🔎 載入 YOLOv8 放大鏡...")
        yolo_model = YOLO("yolov8n.pt")

        return cls(data_folder, clip_model, processor, yolo_model, device)

    def _get_embedding_vector(self, model_output):
        if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
            tensor = model_output.pooler_output
        elif hasattr(model_output, "image_embeds") and model_output.image_embeds is not None:
            tensor = model_output.image_embeds
        elif hasattr(model_output, "text_embeds") and model_output.text_embeds is not None:
            tensor = model_output.text_embeds
        else:
            tensor = model_output[0]

        tensor = tensor.detach().cpu().squeeze()
        if tensor.dim() > 1:
            tensor = tensor[0]
        vec = tensor.numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def extract_image_feature(self, image_input):
        inputs = self.processor(images=image_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
        return self._get_embedding_vector(outputs)

    def extract_text_feature(self, text_input):
        inputs = self.processor(text=[text_input], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)
        return self._get_embedding_vector(outputs)

    def build_vector_database(self, use_cache=True):
        """建庫並快取；回傳 (db_df, df_wifi)。"""
        image_dir = os.path.join(self.data_folder, "images")
        cache_path = os.path.join(self.data_folder, "feature_cache.pkl")

        self.df_wifi = load_wifi_table(self.data_folder)

        image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("image_count") == len(image_files):
                    self.db_df = cached["db_df"]
                    print(f"⚡ 命中特徵快取 ({len(self.db_df)} 筆物件)，跳過建庫！")
                    return self.db_df, self.df_wifi
                print("ℹ️  影像數量已變動，快取失效，重新建庫...")
            except Exception as e:
                print(f"⚠️  快取載入失敗 ({e})，重新建庫...")

        labels_df = load_labels_table(self.data_folder)

        database = []
        print("🔄 開始建庫 (視覺特徵 + Wi-Fi 跨模態綁定)...")

        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)
            try:
                img_time = int(img_name.split("_")[1].split(".")[0])
                original_image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

                location_label = "Unlabeled"
                if labels_df is not None and not labels_df.empty:
                    closest_idx = (labels_df["timestamp"] - img_time).abs().idxmin()
                    if abs(labels_df.loc[closest_idx, "timestamp"] - img_time) < 30000:
                        location_label = labels_df.loc[closest_idx, "label"]

                _, wifi_fp, wifi_ts = find_nearest_wifi_fingerprint(img_time, self.df_wifi)

                yolo_results = self.yolo_model(original_image, verbose=False)
                for box in yolo_results[0].boxes.xyxy:
                    x1, y1, x2, y2 = [int(v.item()) for v in box]
                    crop_img = original_image.crop((x1, y1, x2, y2))

                    if crop_img.size[0] < 30 or crop_img.size[1] < 30:
                        continue

                    database.append({
                        "filename": img_name,
                        "keyframe_timestamp": img_time,
                        "location_label": location_label,
                        "bbox": [x1, y1, x2, y2],
                        "wifi_timestamp": wifi_ts,
                        "wifi_fingerprint": wifi_fp,
                        "vector": self.extract_image_feature(crop_img),
                    })
            except Exception as e:
                print(f"處理 {img_name} 錯誤: {e}")

        self.db_df = pd.DataFrame(database)

        export_path = os.path.join(self.data_folder, "database_export.csv")
        if not self.db_df.empty:
            export_cols = ["filename", "location_label", "bbox", "wifi_timestamp"]
            export_cols = [c for c in export_cols if c in self.db_df.columns]
            self.db_df[export_cols].to_csv(export_path, index=False)
            print(f"💾 資料庫清單已匯出至: {export_path}")

            if use_cache:
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump({"image_count": len(image_files), "db_df": self.db_df}, f)
                    print(f"⚡ 特徵快取已儲存：{os.path.basename(cache_path)}")
                except Exception as e:
                    print(f"⚠️  特徵快取儲存失敗：{e}")
        else:
            print("⚠️  資料庫為空，未匯出 CSV。")

        return self.db_df, self.df_wifi

    def search_and_verify(self, text_query, top_k=3, results_dir=None, verbose=True, ui_lang: str = "zh"):
        """視覺檢索 + LLaVA 驗證 + 記憶圖綁定，回傳結構化結果供 UI 渲染。"""
        if self.db_df is None or self.db_df.empty:
            return {
                "query": text_query,
                "error": t_lang("err_empty_db", ui_lang),
                "candidates": [],
                "confirmed_count": 0,
                "object_location": None,
                "wifi_binding": None,
                "graph_stats": {"nodes": 0, "edges": 0},
            }

        if results_dir is None:
            results_dir = os.getcwd()
        results_dir = os.path.abspath(results_dir)
        os.makedirs(results_dir, exist_ok=True)

        self.session_graph.reset()

        if verbose:
            print(f"\n🔍 [階段一] SigLIP 初篩：尋找「{text_query}」...")

        text_vector = self.extract_text_feature(text_query)
        text_features_2d = text_vector.reshape(1, -1)
        image_vectors_2d = np.vstack(self.db_df["vector"].values)

        similarities = cosine_similarity(text_features_2d, image_vectors_2d)[0]
        top_indices = np.argsort(similarities)[::-1]

        seen_files = set()
        final_candidates = []
        for idx in top_indices:
            row = self.db_df.iloc[idx]
            fname = row["filename"]
            if fname not in seen_files:
                seen_files.add(fname)
                final_candidates.append({
                    "filename": fname,
                    "keyframe_timestamp": row.get("keyframe_timestamp"),
                    "location_label": row["location_label"],
                    "bbox": row["bbox"],
                    "wifi_timestamp": row.get("wifi_timestamp"),
                    "wifi_fingerprint": row.get("wifi_fingerprint"),
                    "score": float(similarities[idx]),
                })
                if len(final_candidates) >= top_k:
                    break

        if verbose:
            print(f"\n🧠 [階段二] LLaVA 啟動常識推理...")

        candidate_results = []
        confirmed_count = 0

        for rank, res in enumerate(final_candidates, 1):
            if verbose:
                print(f"\n--- 第 {rank} 名照片：{res['filename']} ---")
                print(f"📍 位置: {res['location_label']} | 視覺吻合度: {res['score']:.4f}")

            img_full_path = os.path.join(self.data_folder, "images", res["filename"])
            img_to_show = ImageOps.exif_transpose(Image.open(img_full_path)).convert("RGB")
            draw = ImageDraw.Draw(img_to_show)

            x1, y1, x2, y2 = res["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=8)

            save_path = os.path.join(results_dir, f"result_top_{rank}.jpg")
            img_to_show.save(save_path)

            llava_answer = ask_llava_commonsense(
                save_path, text_query, res["location_label"], lang=ui_lang,
            )
            vlm_result = interpret_llava_conclusion(llava_answer)

            if verbose:
                print(f"🤖 LLaVA 辨認物品: {vlm_result.get('item') or '—'}")
                print(f"🤖 LLaVA 分析與常識判斷：\n{llava_answer}")

            semantic_hit = is_semantic_match(text_query, vlm_result.get("item"))
            should_bind = (vlm_result["is_yes"] is True) or semantic_hit

            bind_reason = None
            if should_bind:
                confirmed_count += 1
                if vlm_result["is_yes"]:
                    bind_reason = t_lang("bind_reason_yes", ui_lang)
                else:
                    bind_reason = t_lang(
                        "bind_reason_semantic", ui_lang, item=vlm_result.get("item") or "—"
                    )
                self.session_graph.add_binding(text_query, res, vlm_result)
                bind_wifi_dataframe(self.df_wifi, res.get("wifi_timestamp"), text_query, vlm_result)
                if verbose:
                    print(f"  ✅ 判定為「{text_query}」[{bind_reason}]")
            elif verbose:
                print(f"  ⛔ 判定非「{text_query}」(辨認為: {vlm_result['item']})")

            if vlm_result["is_yes"] is True:
                conclusion_label = t_lang("conclusion_yes", ui_lang)
            elif vlm_result["is_yes"] is False:
                conclusion_label = t_lang("conclusion_no", ui_lang)
            else:
                conclusion_label = t_lang("conclusion_unknown", ui_lang)
            if should_bind and vlm_result["is_yes"] is not True:
                conclusion_label = t_lang("conclusion_yes_semantic", ui_lang)

            candidate_results.append({
                "rank": rank,
                "filename": res["filename"],
                "keyframe_timestamp": res.get("keyframe_timestamp"),
                "location_label": res["location_label"],
                "score": round(res["score"], 4),
                "wifi_timestamp": res.get("wifi_timestamp"),
                "bbox": res["bbox"],
                "result_image_path": save_path,
                "llava_answer": llava_answer,
                "recognized_item": vlm_result.get("item"),
                "conclusion": conclusion_label,
                "is_yes": vlm_result["is_yes"],
                "semantic_hit": semantic_hit,
                "bound": should_bind,
                "bind_reason": bind_reason,
            })

        object_location = self.session_graph.query_object(text_query, lang=ui_lang)
        wifi_binding = query_wifi_by_item(text_query, self.df_wifi, lang=ui_lang)
        graph_stats = self.session_graph.stats()
        graph_consistency = self.session_graph.verify_consistency(text_query)

        result_payload = {
            "query": text_query,
            "candidates": candidate_results,
            "confirmed_count": confirmed_count,
            "object_location": object_location,
            "wifi_binding": wifi_binding,
            "graph_stats": graph_stats,
            "graph_backend": self.session_graph.backend_name,
            "graph_consistency": graph_consistency,
        }

        traj_path = os.path.join(results_dir, "trajectory_map.png")
        result_payload["nearby_objects"] = self.session_graph.list_nearby_objects(text_query)

        # 結構地標掃描（SigLIP；有快取則秒回）
        landmarks = []
        try:
            landmarks = scan_structural_landmarks(
                self.data_folder,
                self.extract_image_feature,
                self.extract_text_feature,
                use_cache=True,
            )
        except Exception as exc:
            print(f"[landmarks] 掃描失敗：{exc}")
        result_payload["landmarks"] = landmarks

        # 目標時間戳：取已綁定候選中分數最高者
        target_ts = None
        bound_cands = [c for c in candidate_results if c.get("bound")]
        if bound_cands:
            best = max(bound_cands, key=lambda c: c.get("score") or 0)
            target_ts = best.get("keyframe_timestamp")
        elif candidate_results:
            target_ts = candidate_results[0].get("keyframe_timestamp")

        if verbose:
            print("\n🗺️  [階段三] 軌跡與地標相對描述…")
        imu_df = load_imu_data(self.data_folder)
        traj_df = compute_trajectory(imu_df) if not imu_df.empty else None
        relative_description = build_relative_description(
            text_query, target_ts, landmarks, traj_df, lang=ui_lang,
        )
        result_payload["relative_description"] = relative_description

        map_path = generate_trajectory_map(
            self.data_folder,
            result_payload,
            traj_path,
            landmarks=landmarks,
            relative_description=relative_description,
            traj_df=traj_df,
            lang=ui_lang,
        )
        result_payload["trajectory_map_path"] = map_path

        if verbose:
            print(f"\n--- Binding 完成，共寫入 {confirmed_count} 筆多模態記憶 ---")
            if relative_description:
                print(f"📍 相對地標：{relative_description}")
            if map_path:
                print(f"🗺️  軌跡圖已儲存：{map_path}")
            else:
                print("⚠️  找不到 IMU 資料，無法產生軌跡圖。")

        return result_payload


def create_system(data_folder=DATA_FOLDER):
    """建立並初始化系統（載入模型 + 建庫）。供 Streamlit @st.cache_resource 使用。"""
    resolved = resolve_data_folder(data_folder)
    system = SpatialMemorySystem.load_models(resolved)
    system.build_vector_database(use_cache=True)
    return system


if __name__ == "__main__":
    resolved_folder = resolve_data_folder(DATA_FOLDER)
    print(f"\n🧠 VLM 推理後端：{VLM_BACKEND}")
    print(f"\n👉 系統正在尋找資料夾：{resolved_folder}")

    if is_dataset_folder(resolved_folder):
        print("✅ 成功找到資料夾！準備進入建庫程序...")

        system = create_system(resolved_folder)
        user_query = "水瓶"
        result = system.search_and_verify(user_query, top_k=3)

        print("\n" + "=" * 50)
        print("🧠 最終多模態空間記憶推理結果")
        print("=" * 50)

        print("\n=== [路徑一] 記憶圖推理查詢 (視覺 + LLaVA + Wi-Fi) ===")
        obj = result["object_location"]
        if isinstance(obj, dict):
            print(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))
        else:
            print(obj)

        print("\n=== [路徑二] Wi-Fi 綁定反查結果 ===")
        wifi = result["wifi_binding"]
        if isinstance(wifi, dict):
            print(json.dumps(wifi, indent=2, ensure_ascii=False, default=_json_default))
        else:
            print(wifi)

        stats = result["graph_stats"]
        print(f"\n=== 記憶圖統計 ===")
        print(f"後端: {result.get('graph_backend')} | 節點數: {stats['nodes']} | 邊數: {stats['edges']}")
        consistency = result.get("graph_consistency")
        if consistency:
            print(f"一致性: {'OK' if consistency.get('match') else 'MISMATCH'} "
                  f"(store={consistency.get('store_backend')})")
            if not consistency.get("match"):
                print(json.dumps(consistency, indent=2, ensure_ascii=False, default=_json_default))
        fusion = result.get("wifi_fusion_report")
        if fusion:
            print(f"Wi-Fi/IMU 融合: mode={fusion.get('mode')} applied={fusion.get('applied')} status={fusion.get('status')}")

        traj = result.get("trajectory_map_path")
        if traj:
            print(f"\n=== 軌跡圖 ===")
            print(f"🗺️  已儲存至：{traj}")
            if result.get("relative_description"):
                print(f"📍 {result['relative_description']}")
            print("   （可用預覽 App 開啟 trajectory_map.png）")
    else:
        print(f"\n❌ 【錯誤】找不到資料夾 '{resolved_folder}'！")
        cands = discover_dataset_candidates()
        if cands:
            print("📦 可用資料集候選：")
            for c in cands:
                print(f"  - {c}")
        print(f"📍 你目前終端機所在的路徑是：{os.getcwd()}")
        print("💡 請檢查：\n1. 資料夾名稱有沒有拼錯？\n2. 你是不是在錯誤的目錄底下執行 Python 檔？")