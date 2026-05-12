"""
llm_translate.py - Dịch batch text EN -> VI qua vLLM (OpenAI-compatible API)
"""

import os
import re
import time
import requests

# ── Config từ env vars ────────────────────────────────────────────────────────
VLLM_URL     = os.getenv("VLLM_URL",     "http://100.86.64.33:8003/v1/chat/completions")
VLLM_MODEL   = os.getenv("VLLM_MODEL",   "Qwen3-14B-AWQ")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "60"))

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {VLLM_API_KEY}",
}

SYSTEM_PROMPT = (
    "You are a professional subtitle translator. "
    "Translate English subtitle lines to natural, concise Vietnamese. "
    "Keep proper nouns (product names, brand names) in English."
)


def _call_llm(prompt: str) -> str:
    payload = {
        "model":       VLLM_MODEL,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3 no-think mode
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt + "/no_think"},
        ],
    }
    resp = requests.post(VLLM_URL, json=payload, headers=HEADERS, timeout=VLLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _build_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        "Translate each English line below to Vietnamese.\n"
        "Rules:\n"
        "- Output ONLY the translations, numbered the same way\n"
        "- One translation per line, no extra text or explanations\n"
        "- Keep product/brand names in English\n\n"
        f"{numbered}"
    )


def _parse_response(response: str, expected: int, originals: list[str]) -> list[str]:
    """Tách kết quả theo số thứ tự. Fallback về EN nếu thiếu dòng."""
    lines = response.splitlines()
    results = {}
    for line in lines:
        # Khớp "1. text" hoặc "1) text"
        m = re.match(r"^(\d+)[.)]\s*(.+)", line.strip())
        if m:
            idx  = int(m.group(1)) - 1
            text = m.group(2).strip()
            if 0 <= idx < expected:
                results[idx] = text

    # Nếu không parse được số thứ tự → thử lấy theo thứ tự dòng
    if not results:
        clean = [l.strip() for l in lines if l.strip()]
        for i, text in enumerate(clean[:expected]):
            results[i] = text

    return [results.get(i, originals[i]) for i in range(expected)]


def translate_batch(texts: list[str], retry: int = 1) -> list[str]:
    """Dịch một batch. Trả về list cùng độ dài với texts."""
    prompt = _build_prompt(texts)
    for attempt in range(retry + 1):
        try:
            response = _call_llm(prompt)
            return _parse_response(response, len(texts), texts)
        except Exception as e:
            if attempt < retry:
                print(f"\n    [warn] LLM call failed: {e} — retrying...")
                time.sleep(2)
            else:
                print(f"\n    [warn] LLM failed after {retry+1} attempts: {e}")
                return list(texts)   # fallback: giữ nguyên EN


def translate_entries(entries: list[dict], batch_size: int = 20) -> list[dict]:
    """
    Dịch list entries (mỗi entry có key 'text').
    Thêm key 'text_vi' vào mỗi entry. Trả về list mới.
    """
    result = list(entries)
    total  = len(entries)

    for start in range(0, total, batch_size):
        batch  = entries[start : start + batch_size]
        texts  = [e["text"] for e in batch]
        vi     = translate_batch(texts)

        for i, vi_text in enumerate(vi):
            result[start + i]["text_vi"] = vi_text

        done = min(start + batch_size, total)
        print(f"  [{done:>4}/{total}] {batch[-1]['start']} --> {batch[-1]['end']}", end="\r")

    print()
    return result
