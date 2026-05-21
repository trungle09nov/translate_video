"""Batch translation helpers for the vLLM OpenAI-compatible API."""

from __future__ import annotations

import os
import re
import time

import requests

from pipeline_config import ADV_OCR_LANG, ADV_TARGET_LANG


VLLM_URL = os.getenv("VLLM_URL", "http://100.86.64.33:8003/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen3-14B-AWQ")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "60"))

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {VLLM_API_KEY}",
}

LANGUAGE_NAMES = {
    "de": "German",
    "deu": "German",
    "german": "German",
    "en": "English",
    "eng": "English",
    "english": "English",
    "vi": "Vietnamese",
    "vie": "Vietnamese",
    "vietnamese": "Vietnamese",
}


def _language_name(lang: str) -> str:
    value = str(lang or "").strip()
    return LANGUAGE_NAMES.get(value.lower(), value or "the target language")


def _system_prompt(source_lang: str, target_lang: str) -> str:
    source = _language_name(source_lang)
    target = _language_name(target_lang)
    return (
        "You are a professional subtitle and UI text translator. "
        f"Translate {source} text to natural, concise {target}. "
        "Keep product names, brand names, numbers, and placeholders unchanged."
    )


def _call_llm(prompt: str, source_lang: str, target_lang: str) -> str:
    payload = {
        "model": VLLM_MODEL,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": _system_prompt(source_lang, target_lang)},
            {"role": "user", "content": prompt + "/no_think"},
        ],
    }
    resp = requests.post(VLLM_URL, json=payload, headers=HEADERS, timeout=VLLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _build_prompt(texts: list[str], source_lang: str, target_lang: str) -> str:
    source = _language_name(source_lang)
    target = _language_name(target_lang)
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(texts))
    return (
        f"Translate each {source} line below to {target}.\n"
        "Rules:\n"
        "- Output ONLY the translations, numbered the same way\n"
        "- One translation per line, no extra text or explanations\n"
        "- Keep product/brand names, numbers, and placeholders unchanged\n\n"
        f"{numbered}"
    )


def _parse_response(response: str, expected: int, originals: list[str]) -> list[str]:
    lines = response.splitlines()
    results = {}
    for line in lines:
        match = re.match(r"^(\d+)[.)]\s*(.+)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            text = match.group(2).strip()
            if 0 <= idx < expected:
                results[idx] = text

    if not results:
        clean = [line.strip() for line in lines if line.strip()]
        for i, text in enumerate(clean[:expected]):
            results[i] = text

    return [results.get(i, originals[i]) for i in range(expected)]


def translate_batch(
    texts: list[str],
    retry: int = 1,
    source_lang: str = ADV_OCR_LANG,
    target_lang: str = ADV_TARGET_LANG,
) -> list[str]:
    """Translate a batch and return a list with the same length as texts."""
    prompt = _build_prompt(texts, source_lang, target_lang)
    for attempt in range(retry + 1):
        try:
            response = _call_llm(prompt, source_lang, target_lang)
            return _parse_response(response, len(texts), texts)
        except Exception as exc:
            if attempt < retry:
                print(f"\n    [warn] LLM call failed: {exc} - retrying...")
                time.sleep(2)
            else:
                print(f"\n    [warn] LLM failed after {retry + 1} attempts: {exc}")
                return list(texts)


def translate_entries(entries: list[dict], batch_size: int = 20) -> list[dict]:
    """Translate SRT entries from English to Vietnamese and fill text_vi."""
    result = list(entries)
    total = len(entries)

    for start in range(0, total, batch_size):
        batch = entries[start : start + batch_size]
        texts = [entry["text"] for entry in batch]
        translated = translate_batch(texts, source_lang="english", target_lang="vietnamese")

        for i, translated_text in enumerate(translated):
            result[start + i]["text_vi"] = translated_text

        done = min(start + batch_size, total)
        print(f"  [{done:>4}/{total}] {batch[-1]['start']} --> {batch[-1]['end']}", end="\r")

    print()
    return result
