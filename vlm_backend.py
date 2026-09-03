"""
VLM 推理後端 — 策略模式 (Strategy Pattern)

支援三種模式，由 .env 的 VLM_BACKEND 切換：
    ollama      → 本地 Ollama LLaVA
    mock        → 假資料，供 UI 開發無痛測試
    huggingface → Hugging Face Serverless Inference API
"""

from __future__ import annotations

import base64
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# 載入 .env（存在則讀取；不存在也不報錯）
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

_VALID_BACKENDS = {"ollama", "mock", "huggingface"}
_warned_mock_default = False


def _yellow_warn(message: str) -> None:
    print(f"\033[33m{message}\033[0m", file=sys.stderr)


def resolve_vlm_backend() -> str:
    """解析 VLM_BACKEND；未設定或 .env 不存在時預設 mock 並印出警告。"""
    global _warned_mock_default

    raw = os.getenv("VLM_BACKEND", "").strip().lower()
    if raw in _VALID_BACKENDS:
        return raw

    if not _warned_mock_default:
        _warned_mock_default = True
        if not ENV_PATH.exists():
            _yellow_warn(
                "⚠️ 未偵測到 .env 檔案，自動啟用 Mock (假資料) 模式。"
                "若要使用真實模型，請複製 .env.example 為 .env 並設定 VLM_BACKEND。"
            )
        else:
            _yellow_warn(
                "⚠️ 未偵測到 VLM_BACKEND，自動啟用 Mock (假資料) 模式。"
                "若要使用真實模型，請設定 .env 檔案。"
            )
    return "mock"


# 模組載入時解析一次，供 UI / CLI 顯示
VLM_BACKEND = resolve_vlm_backend()


def build_commonsense_prompt(query: str, location_hint: str | None = None, lang: str = "zh") -> str:
    """共用的常識推理提示詞（三種後端一致）。"""
    use_en = str(lang).lower().startswith("en")
    loc_unknown = "Unknown location" if use_en else "未知地點"
    loc_text = location_hint if location_hint and location_hint != "Unlabeled" else loc_unknown

    if use_en:
        return (
            f"Context: photo taken at「{loc_text}」.\n"
            f"A red box highlights one object. Judge by appearance, shape, and material.\n"
            f"The user is looking for:「{query}」.\n"
            f"Rules (strict):\n"
            f"1. Answer YES if the boxed item is the same kind or function as「{query}」.\n"
            f"2. Water bottles, kettles, cups, mugs, and drink containers count as the same category for water-related queries.\n"
            f"3. Answer NO only for clearly different objects (chair, phone, plant, soap, etc.).\n"
            f"4. Unusual placement is NOT a reason to say NO.\n"
            f"IMPORTANT: Reply ONLY in English using exactly this format (no extra lines):\n"
            f"Item name: <recognized object in English>\n"
            f"Reasoning: <one or two short sentences in English>\n"
            f"Conclusion: YES or Conclusion: NO"
        )

    return (
        f"先驗知識：這張照片是在「{loc_text}」拍攝的，僅供你理解場景。\n"
        f"圖中紅色框框圈出了一個物品，請依它的外觀、形狀與材質來判斷。\n"
        f"使用者要找的是「{query}」。\n"
        f"判斷規則（非常重要，請嚴格遵守）：\n"
        f"1. 只要框內物品在『種類或功能』上等同或非常接近「{query}」，就算符合，請回答 YES。\n"
        f"2. 『水瓶、水壺、寶特瓶、瓶裝水、礦泉水瓶、水杯、保溫瓶』都屬於『裝水來喝的容器』，"
        f"視為同一類，彼此都算符合，不要因為名稱用字不同（例如『瓶』和『杯』）就回答 NO。\n"
        f"3. 只有當它明顯是完全不同功能的東西（例如洗手乳、清潔劑、盆栽、電話、電腦、椅子）時，才回答 NO。\n"
        f"4. 物品常被隨手放在不尋常的位置，所以『位置奇怪』不可以當作否定理由；"
        f"即使視覺吻合度低，也要依外觀認真辨認。\n"
        f"請務必用以下格式回答，每行開頭不要留白：\n"
        f"物品名稱：<你辨認出的物品>\n"
        f"推理：<簡短理由>\n"
        f"結論：YES 或 結論：NO"
    )


def _format_mock_answer(item: str, reason: str, is_yes: bool, location: str, lang: str = "zh") -> str:
    conclusion = "YES" if is_yes else "NO"
    if str(lang).lower().startswith("en"):
        return (
            f"Item name: {item}\n"
            f"Reasoning: [Mock] {reason} Location: {location}.\n"
            f"Conclusion: {conclusion}"
        )
    return (
        f"物品名稱：{item}\n"
        f"推理：[Mock 模式] {reason} 拍攝地點：{location}。\n"
        f"結論：{conclusion}"
    )


def simulate_mock_output(
    query: str,
    location_hint: str | None = None,
    image_path: str | None = None,
    lang: str = "zh",
) -> str:
    """Mock 假資料：依查詢關鍵字回傳固定但合理的 LLaVA 格式文字。"""
    loc = location_hint if location_hint and location_hint != "Unlabeled" else "Room 1"
    q = (query or "").lower()

    water_kw = {"水", "水瓶", "水杯", "bottle", "water", "cup", "一瓶水"}
    key_kw = {"鑰匙", "key", "keys"}
    pill_kw = {"藥", "藥盒", "medicine", "pill"}

    use_en = str(lang).lower().startswith("en")

    if any(k in q for k in water_kw):
        return _format_mock_answer(
            item="water bottle" if use_en else "水瓶",
            reason="Clear drink container with cap." if use_en else "透明塑膠瓶身與瓶蓋，屬於裝水來喝的容器。",
            is_yes=True,
            location=loc,
            lang=lang,
        )
    if any(k in q for k in key_kw):
        return _format_mock_answer(
            item="keys" if use_en else "鑰匙",
            reason="Metal keyring matches keys." if use_en else "金屬鑰匙圈，符合鑰匙外觀特徵。",
            is_yes=True,
            location=loc,
            lang=lang,
        )
    if any(k in q for k in pill_kw):
        return _format_mock_answer(
            item="pill box" if use_en else "藥盒",
            reason="Small medicine container." if use_en else "白色塑膠小盒，符合藥盒外觀。",
            is_yes=True,
            location=loc,
            lang=lang,
        )
    if any(k in q for k in {"電話", "phone"}):
        return _format_mock_answer(
            item="phone" if use_en else "電話",
            reason="Not the queried item." if use_en else "黑色無線電話本體，與查詢不符。",
            is_yes=False,
            location=loc,
            lang=lang,
        )

    return _format_mock_answer(
        item=query or ("unknown" if use_en else "未知物品"),
        reason="Mock recognition for UI test." if use_en else f"模擬辨認為「{query}」，供 UI 測試使用。",
        is_yes=True,
        location=loc,
        lang=lang,
    )


class VLMStrategy(ABC):
    """VLM 推理策略抽象介面。"""

    @abstractmethod
    def infer(
        self,
        image_path: str,
        query: str,
        location_hint: str | None = None,
        lang: str = "zh",
    ) -> str:
        """回傳 LLaVA 格式純文字（物品名稱 / 推理 / 結論）。"""


class OllamaVLMStrategy(VLMStrategy):
    """模式 A：本地 Ollama LLaVA。"""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llava")

    def infer(
        self,
        image_path: str,
        query: str,
        location_hint: str | None = None,
        lang: str = "zh",
    ) -> str:
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt = build_commonsense_prompt(query, location_hint, lang=lang)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }
            res = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            )
            res.raise_for_status()
            raw_response = res.json().get("response", "LLaVA 沒反應")
            clean_lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
            return "\n".join(clean_lines)
        except Exception as exc:
            return f"LLaVA (Ollama) 連線失敗: {exc}"


class MockVLMStrategy(VLMStrategy):
    """模式 B：Mock 假資料，零運算、零連網。"""

    def infer(
        self,
        image_path: str,
        query: str,
        location_hint: str | None = None,
        lang: str = "zh",
    ) -> str:
        return simulate_mock_output(query, location_hint, image_path, lang=lang)


class HuggingFaceVLMStrategy(VLMStrategy):
    """模式 C：Hugging Face Serverless Inference API。"""

    def __init__(self):
        self.token = os.getenv("HF_TOKEN", "").strip()
        self.model_id = os.getenv("HF_MODEL_ID", "llava-hf/llava-1.5-7b-hf").strip()
        self.api_url = os.getenv(
            "HF_API_URL",
            f"https://api-inference.huggingface.co/models/{self.model_id}",
        ).strip()
        self.timeout = int(os.getenv("HF_TIMEOUT", "120"))

    def _extract_hf_text(self, payload) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("answer") or str(first)
            return str(first)
        if isinstance(payload, dict):
            for key in ("generated_text", "answer", "text", "output"):
                if key in payload and payload[key]:
                    return str(payload[key])
            if "error" in payload:
                raise RuntimeError(payload["error"])
        return str(payload)

    def infer(
        self,
        image_path: str,
        query: str,
        location_hint: str | None = None,
        lang: str = "zh",
    ) -> str:
        if not self.token:
            return "LLaVA (HuggingFace) failed: HF_TOKEN not set in .env."

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            prompt = build_commonsense_prompt(query, location_hint, lang=lang)
            headers = {"Authorization": f"Bearer {self.token}"}

            # LLaVA 系列：image + question
            if "llava" in self.model_id.lower():
                body = {
                    "inputs": {
                        "image": base64.b64encode(image_bytes).decode("utf-8"),
                        "question": prompt,
                    },
                    "parameters": {"max_new_tokens": 256, "temperature": 0.2},
                }
            else:
                # Qwen-VL / 其他多模態：messages 格式
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                body = {
                    "inputs": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                    },
                    "parameters": {"max_new_tokens": 256, "temperature": 0.2},
                }

            res = requests.post(self.api_url, headers=headers, json=body, timeout=self.timeout)

            if res.status_code == 503:
                return "LLaVA (HuggingFace) 模型載入中，請稍後再試 (503 Model loading)."

            res.raise_for_status()
            data = res.json()
            raw = self._extract_hf_text(data)
            clean_lines = [line.strip() for line in raw.split("\n") if line.strip()]
            return "\n".join(clean_lines) if clean_lines else raw

        except Exception as exc:
            return f"LLaVA (HuggingFace) 連線失敗: {exc}"


_strategy_instance: VLMStrategy | None = None


def get_vlm_strategy(backend: str | None = None) -> VLMStrategy:
    """取得（並快取）目前使用的 VLM 策略實例。"""
    global _strategy_instance

    mode = (backend or resolve_vlm_backend()).lower()
    if mode not in _VALID_BACKENDS:
        mode = "mock"

    if _strategy_instance is not None and getattr(_strategy_instance, "_mode", None) == mode:
        return _strategy_instance

    if mode == "ollama":
        strategy = OllamaVLMStrategy()
    elif mode == "huggingface":
        strategy = HuggingFaceVLMStrategy()
    else:
        strategy = MockVLMStrategy()

    strategy._mode = mode  # type: ignore[attr-defined]
    _strategy_instance = strategy
    return strategy


def ask_llava_commonsense(
    image_path: str,
    query: str,
    location_hint: str | None = None,
    lang: str = "zh",
) -> str:
    """統一 VLM 推理入口：依 VLM_BACKEND 自動路由至對應策略。"""
    return get_vlm_strategy().infer(image_path, query, location_hint, lang=lang)
