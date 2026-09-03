"""
結構地標掃描：用 SigLIP 在 keyframe 上找門／大型家電／固定傢俱等不易移動地標。

快取檔：{data_folder}/landmarks_cache.json
"""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from nav_i18n import display_landmark_label, nav_t
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity

REMOTE_ROOT = Path(__file__).resolve().parent
LANDMARKS_CONFIG = REMOTE_ROOT / "config" / "landmarks.json"

DEFAULT_MAX_IMAGES = 48
DEFAULT_SCORE_THRESHOLD = 0.12
SOFT_SCORE_FLOOR = 0.085  # below hard threshold but still plot as 「疑似」錨點


def load_landmark_vocab(config_path: Path = LANDMARKS_CONFIG) -> list[dict[str, Any]]:
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _is_enabled(item: dict[str, Any]) -> bool:
    return bool(item.get("enabled", True))


def _role(item: dict[str, Any]) -> str:
    role = str(item.get("role", "secondary")).strip().lower()
    return role if role in {"primary", "secondary"} else "secondary"


def _timestamp_from_filename(name: str) -> Optional[int]:
    try:
        part = name.split("_")[1].split(".")[0]
        return int(part)
    except Exception:
        return None


def _list_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )


def _sample_images(files: list[Path], max_images: int) -> list[Path]:
    if len(files) <= max_images:
        return files
    idx = np.linspace(0, len(files) - 1, max_images, dtype=int)
    return [files[i] for i in idx]


def _vocab_signature(vocab: list[dict[str, Any]]) -> str:
    payload = json.dumps(vocab, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path, image_count: int, vocab_sig: str) -> Optional[list[dict[str, Any]]]:
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("image_count") == image_count
            and cached.get("vocab_sig") == vocab_sig
            and isinstance(cached.get("landmarks"), list)
        ):
            return cached["landmarks"]
    except Exception:
        return None
    return None


def _save_cache(cache_path: Path, image_count: int, landmarks: list[dict[str, Any]], vocab_sig: str) -> None:
    cache_path.write_text(
        json.dumps(
            {"image_count": image_count, "vocab_sig": vocab_sig, "landmarks": landmarks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def scan_structural_landmarks(
    data_folder: str,
    extract_image_feature: Callable[[Any], np.ndarray],
    extract_text_feature: Callable[[str], np.ndarray],
    *,
    max_images: int = DEFAULT_MAX_IMAGES,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    回傳地標列表，每筆：
      id, label, marker, color, score, keyframe, timestamp, name
    """
    data_folder = os.path.abspath(data_folder)
    images_dir = Path(data_folder) / "images"
    cache_path = Path(data_folder) / "landmarks_cache.json"
    vocab = load_landmark_vocab()
    if not vocab or not images_dir.exists():
        print("[landmarks] 無地標詞彙或 images/ 不存在")
        return []

    all_files = _list_image_files(images_dir)
    if not all_files:
        return []

    active_vocab = [item for item in vocab if _is_enabled(item)]
    if not active_vocab:
        print("[landmarks] 地標詞彙皆停用，略過掃描")
        return []

    vocab_sig = _vocab_signature(active_vocab)
    if use_cache:
        cached = _load_cache(cache_path, len(all_files), vocab_sig)
        if cached is not None:
            print(f"[landmarks] 命中快取 ({len(cached)} 個地標)")
            return cached

    sample = _sample_images(all_files, max_images)
    print(f"[landmarks] 掃描 {len(sample)}/{len(all_files)} 張 keyframe…")

    text_vecs = {}
    for item in active_vocab:
        lid = item["id"]
        queries = item.get("queries") or [item.get("label", lid)]
        vecs = []
        for q in queries[:5]:
            try:
                vecs.append(extract_text_feature(str(q)))
            except Exception:
                continue
        if vecs:
            text_vecs[lid] = np.mean(np.vstack(vecs), axis=0)

    best: dict[str, dict[str, Any]] = {}
    best_any: dict[str, dict[str, Any]] = {}

    for img_path in sample:
        try:
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            full_vec = extract_image_feature(img)
            w, h = img.size
            crop = img.crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
            crop_vec = extract_image_feature(crop)
        except Exception as exc:
            print(f"[landmarks] 讀圖失敗 {img_path.name}: {exc}")
            continue

        ts = _timestamp_from_filename(img_path.name)
        for item in active_vocab:
            lid = item["id"]
            tvec = text_vecs.get(lid)
            if tvec is None:
                continue
            s1 = float(cosine_similarity(tvec.reshape(1, -1), full_vec.reshape(1, -1))[0, 0])
            s2 = float(cosine_similarity(tvec.reshape(1, -1), crop_vec.reshape(1, -1))[0, 0])
            score = max(s1, s2)
            per_item_threshold = float(item.get("score_threshold", score_threshold))
            entry = {
                "id": lid,
                "label": item.get("label", lid),
                "marker": item.get("marker", "s"),
                "color": item.get("color", "#16A085"),
                "role": _role(item),
                "score": round(score, 4),
                "keyframe": img_path.name,
                "timestamp": ts,
                "name": item.get("label", lid),
                "confidence": "high",
            }
            prev_any = best_any.get(lid)
            if prev_any is None or score > prev_any["score"]:
                best_any[lid] = dict(entry)
            if score < per_item_threshold:
                continue
            prev = best.get(lid)
            if prev is None or score > prev["score"]:
                best[lid] = entry

    landmarks = sorted(best.values(), key=lambda x: -x["score"])
    # 硬門檻不足時補「疑似」錨點；詞彙變多後允許多種大型家電／門同時出現
    soft_target = max(3, min(8, len(active_vocab)))
    if len(landmarks) < soft_target:
        for lid, entry in sorted(best_any.items(), key=lambda kv: -kv[1]["score"]):
            if lid in {x["id"] for x in landmarks}:
                continue
            if entry["score"] < SOFT_SCORE_FLOOR:
                continue
            soft = dict(entry)
            soft["confidence"] = "weak"
            soft["role"] = "secondary"
            soft["label"] = f"疑似{entry['label']}"
            soft["name"] = soft["label"]
            landmarks.append(soft)
            if len(landmarks) >= soft_target:
                break
        landmarks = sorted(landmarks, key=lambda x: -x["score"])

    if use_cache:
        _save_cache(cache_path, len(all_files), landmarks, vocab_sig)
        print(f"[landmarks] 快取已寫入：{cache_path.name}（{len(landmarks)} 個地標）")
    if not landmarks:
        print("[landmarks] 未達分數門檻，本資料集未標出結構地標")
    else:
        names = ", ".join(f"{x['label']}({x['score']})" for x in landmarks)
        print(f"[landmarks] 找到：{names}")
    return landmarks


def trajectory_arc_length_to(traj_df, timestamp: Optional[int]) -> Optional[float]:
    """從軌跡起點沿折線走到該時間戳的累積弧長。"""
    if traj_df is None or getattr(traj_df, "empty", True) or timestamp is None:
        return None
    from trajectory import interpolate_trajectory_position

    pos = interpolate_trajectory_position(traj_df, timestamp)
    if pos is None:
        return None
    xs = traj_df["x"].to_numpy(dtype=float)
    ys = traj_df["y"].to_numpy(dtype=float)
    t_arr = traj_df["timestamp"].to_numpy()
    idx = int(np.searchsorted(t_arr, int(timestamp)))
    idx = min(max(idx, 0), len(xs) - 1)
    if idx == 0:
        return 0.0
    dx = np.diff(xs[: idx + 1])
    dy = np.diff(ys[: idx + 1])
    return float(np.sum(np.hypot(dx, dy)))


def _path_fraction(traj_df, timestamp: Optional[int]) -> Optional[float]:
    target_arc = trajectory_arc_length_to(traj_df, timestamp)
    if target_arc is None or traj_df is None or getattr(traj_df, "empty", True):
        return None
    xs = traj_df["x"].to_numpy(dtype=float)
    ys = traj_df["y"].to_numpy(dtype=float)
    total = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    if total <= 1e-6:
        return 0.0
    return float(np.clip(target_arc / total, 0.0, 1.0))


def build_relative_description(
    query: str,
    target_timestamp: Optional[int],
    landmarks: list[dict[str, Any]],
    traj_df,
    lang: str = "zh",
) -> str:
    """短提示：物品在哪 + 怎麼走回最近地標。"""
    if target_timestamp is None:
        return nav_t("rel_no_ts", lang, query=query)

    frac = _path_fraction(traj_df, target_timestamp)
    anchors = [lm for lm in landmarks if lm.get("confidence", "high") != "weak"]
    if not anchors:
        anchors = landmarks

    tip = nav_t("rel_star", lang, query=query)
    if frac is not None:
        tip += nav_t("rel_path_pct", lang, pct=frac * 100)

    nearest = None
    target_arc = trajectory_arc_length_to(traj_df, target_timestamp)
    if anchors and target_arc is not None:
        scored = []
        for lm in anchors:
            lm_arc = trajectory_arc_length_to(traj_df, lm.get("timestamp"))
            if lm_arc is None:
                continue
            scored.append((abs(lm_arc - target_arc), lm, lm_arc))
        if scored:
            scored.sort(key=lambda x: x[0])
            nearest = scored[0]

    if nearest:
        lm, lm_arc = nearest[1], nearest[2]
        label = display_landmark_label(lm, lang)
        if target_arc > lm_arc + 0.2:
            how = nav_t("rel_back", lang, label=label)
        elif lm_arc > target_arc + 0.2:
            how = nav_t("rel_forward", lang, label=label)
        else:
            how = nav_t("rel_near", lang, label=label)
        sep = "。" if lang == "zh" else ". "
        return f"{tip}{sep}{how}{nav_t('rel_tail_wifi', lang)}"
    return f"{tip}{nav_t('rel_tail_photo', lang)}"