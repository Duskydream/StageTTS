"""Shared TTS-facing text and language normalization helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_DUAL_CHOICE_SLASH_RE = re.compile(r"([\u4e00-\u9fff]{1,4})\s*[／/]\s*([\u4e00-\u9fff]{1,8})")
_EDITORIAL_GLOSS_RE = re.compile(r"[（(]\s*(?:暂译|待定|译名待定|待确认|音译|直译|意译)\s*[）)]")
_SHORT_TRAILING_ELLIPSIS_RE = re.compile(r"(?:\.{2,}|…+|——+|~+)\s*$")


def _replace_cjk_slash(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if len(right) <= len(left) + 1:
        return left
    # Keep trailing clause words when regex spans past the short synonym token.
    return left + right[len(left):]


def clean_dubbing_text(text: str) -> str:
    """Clean spoken subtitle/dubbing text without altering CJK punctuation or facts.

    - Collapses redundant whitespace.
    - Resolves accidental dual-choice slashes (e.g. ``耳环／耳钉`` -> ``耳环``).
    - Removes LLM editorial tags (e.g. ``（暂译）``).
    """
    value = "".join(ch for ch in (text or "") if ch.isprintable() or ch.isspace())
    value = _EDITORIAL_GLOSS_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    # Strip CJK dual-choice slashes iteratively in case a sentence has two.
    while _DUAL_CHOICE_SLASH_RE.search(value):
        value = _DUAL_CHOICE_SLASH_RE.sub(_replace_cjk_slash, value)
    return value.strip()


def normalize_tts_lang(value: str | None, *, default: str = "zh") -> str:
    """Canonicalize language codes accepted by GPT-SoVITS api_v2.py."""
    lang = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh", "chinese": "zh", "中文": "zh",
        "en-us": "en", "en-gb": "en", "english": "en", "英语": "en",
        "ja-jp": "ja", "japanese": "ja", "日语": "ja",
        "ko-kr": "ko", "korean": "ko", "韩语": "ko",
        "yue-hk": "yue", "cantonese": "yue", "粤语": "yue",
    }
    lang = aliases.get(lang, lang)
    supported = {"zh", "en", "ja", "ko", "yue", "auto", "", "all_zh", "all_en", "all_ja", "all_ko", "all_yue"}
    if lang not in supported:
        return default
    return lang


def prepare_tts_text(text: str, lang: str, glossary: dict[str, str] | None = None) -> str:
    """Create a TTS-facing spoken-form copy without mutating reviewed subtitles."""
    value = clean_dubbing_text(text)
    for source, target in (glossary or {}).items():
        if source and source.casefold() in value.casefold():
            value = re.sub(re.escape(source), target or source, value, flags=re.IGNORECASE)
    canon_lang = normalize_tts_lang(lang)
    if canon_lang in {"zh", "yue", "all_zh", "all_yue"}:
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", value))
        if 0 < cjk_count <= 8 and _SHORT_TRAILING_ELLIPSIS_RE.search(value):
            # Prevent GPT-SoVITS from stretching ultra-short ellipsis lines into long sighs.
            value = _SHORT_TRAILING_ELLIPSIS_RE.sub("。", value)
    if canon_lang == "zh":
        value = re.sub(r"(?<=[\u4e00-\u9fff])([A-Za-z0-9]+)", r" \1", value)
        value = re.sub(r"([A-Za-z0-9]+)(?=[\u4e00-\u9fff])", r"\1 ", value)
    return value


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
