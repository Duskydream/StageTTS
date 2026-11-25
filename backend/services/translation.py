"""Dialogue context shared by parse-time translation and duration rewrites."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Iterable

from openai import OpenAI, OpenAIError

from backend import config
from backend.agent.prompts import (
    DIALOGUE_POLISH_PROMPT,
    DIALOGUE_QA_TIGHTEN_PROMPT,
    DIRECTOR_TRANSLATE_PROMPT,
    GLOSSARY_WEB_REFINE_PROMPT,
    SCRIPT_ANALYSIS_PROMPT,
    TRANSLATION_REPAIR_PROMPT,
)
from backend.schemas import ProjectScript, Segment
from backend.services import web_search
from backend.services.text_normalization import clean_dubbing_text

EMOTIONS = {"neutral", "angry", "happy", "sad", "whisper"}
CONTEXT_INSTRUCTIONS = (
    "先通读完整对白理解剧情、角色关系和前后语气，再按原句 id 还原译文。"
    "参照相邻台词、角色语气、show_bible 与术语表；术语出现时使用固定译法。"
    "语义完整且贴合戏剧时长：中文配音舒适语速约 4.5 字/秒，译文长度应尽量落在 [min_units, max_units] 区间（偏近 max_units 的 90%），既不超时拖沓，也不要过度精简删减导致长段冷场空隙；严禁斜杠备选词。"
    "只处理 lines 中的片段 ID。"
)

# Only an explicit non-language event marker is enough evidence to keep source audio.
# Short spoken answers ("Yes", "Go", "你好", "はい", "네") and bare laughter-like
# strings ("ha ha ha") are still dialogue and must be translated/dubbed.
_NON_SPEECH_EVENT_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"[\[\(（【][^\]\)）】]{0,24}"
    r"(?:laughs?|laughter|sighs?|gasp|groans?|cry|crying|sobs?|applause|cheers?|breathing|"
    r"笑声|笑|叹气|叹息|抽泣|哭声|哭泣|掌声|喘气|呼吸|惊呼)"
    r"[^\]\)）】]{0,24}[\]\)）】]"
    r")"
    r"[\s!！.?。，,]*$",
    re.IGNORECASE,
)


class LLMNotConfiguredError(RuntimeError):
    """未配置有效的 LLM API Key。"""


class LLMServiceError(RuntimeError):
    """调用上游大模型网关失败或返回非法结果。"""


def create_openai_client() -> OpenAI:
    from backend.services.llm_settings import get_llm_snapshot
    snapshot = get_llm_snapshot()
    if not snapshot.configured():
        raise LLMNotConfiguredError("请在设置页面配置 LLM API Key、接口地址与模型名称。")
    client = OpenAI(api_key=snapshot.api_key, base_url=snapshot.base_url, timeout=config.LLM_TIMEOUT_SEC)
    client._stagetts_model = snapshot.model
    return client


def create_director_client(fallback_client: OpenAI | None = None) -> OpenAI:
    from backend.services.llm_settings import get_llm_snapshot
    snapshot = get_llm_snapshot()
    if not snapshot.has_custom_director():
        return fallback_client if fallback_client is not None else create_openai_client()
    if not snapshot.director_configured():
        raise LLMNotConfiguredError("导演模型配置不完整，请检查导演模型的 API Key 与接口地址。")
    d_url, d_model, d_key = snapshot.effective_director()
    client = OpenAI(api_key=d_key, base_url=d_url, timeout=config.LLM_TIMEOUT_SEC)
    client._stagetts_model = d_model
    return client


def client_model(client: OpenAI) -> str:
    model = getattr(client, "_stagetts_model", None)
    return model if isinstance(model, str) else config.LLM_MODEL


def _extract_json_payload(content: str, *, left_char: str, right_char: str, err_msg: str):
    fences = re.findall(r"```(json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fences:
        content = (
            next((body.strip() for lang, body in fences if lang and left_char in body and right_char in body), None)
            or next((body.strip() for _, body in fences if left_char in body and right_char in body), None)
            or fences[0][1].strip()
        )
    left, right = content.find(left_char), content.rfind(right_char)
    if left == -1 or right <= left:
        raise ValueError(err_msg)
    return json.loads(content[left:right + 1])


def complete_json_object(
    client: OpenAI,
    system: str,
    payload: dict,
    *,
    temperature: float = 0.2,
) -> dict:
    response = client.chat.completions.create(
        model=client_model(client),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=temperature,
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = _extract_json_payload(content, left_char="{", right_char="}", err_msg="LLM 未返回 JSON 对象")
    if not isinstance(parsed, dict):
        raise ValueError("LLM 未返回 JSON 对象")
    return parsed


def complete_dialogue(
    client: OpenAI,
    system: str,
    payload: dict,
    *,
    temperature: float = 0.3,
) -> list[dict]:
    response = client.chat.completions.create(
        model=client_model(client),
        messages=[
            {"role": "system", "content": system + "\n" + CONTEXT_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=temperature,
    )
    content = (response.choices[0].message.content or "").strip()
    entries = _extract_json_payload(content, left_char="[", right_char="]", err_msg="LLM 未返回 JSON 数组")
    if not isinstance(entries, list):
        raise ValueError("LLM 未返回 JSON 数组")
    return [entry for entry in entries if isinstance(entry, dict)]


def _compute_speaker_pace_scales(
    script: ProjectScript,
    acoustic_hints: dict[int, dict],
) -> dict[str, float]:
    """Measure each speaker's relative speaking rate on active audio windows."""
    if not acoustic_hints:
        return {}
    per_speaker: dict[str, list[float]] = {}
    all_rates: list[float] = []
    source_cjk = (script.source_lang or "").removeprefix("all_").split("-")[0] in {"zh", "ja", "yue", "ko"}
    for seg in script.segments:
        text = seg.orig_text.strip()
        dur = seg.end - seg.start
        hint = acoustic_hints.get(seg.id)
        if not text or dur < 1.2 or not hint or seg.speaker == "UNKNOWN":
            continue
        active_ratio = float(hint.get("energy_active_ratio") or 0.85)
        active_sec = max(0.4, dur * max(0.35, min(1.0, active_ratio)))
        units = len(re.sub(r"\s+", "", text)) if source_cjk else max(1, len(text.split()))
        rate = units / active_sec
        if 0.5 <= rate <= 16.0:
            per_speaker.setdefault(seg.speaker, []).append(rate)
            all_rates.append(rate)
    if len(all_rates) < 3:
        return {}
    global_med = statistics.median(all_rates)
    if global_med <= 0:
        return {}
    scales: dict[str, float] = {}
    for spk, rates in per_speaker.items():
        if len(rates) >= 2:
            ratio = statistics.median(rates) / global_med
            scales[spk] = round(max(0.80, min(1.22, ratio)), 3)
    return scales


class DialogueContext:
    def __init__(
        self,
        script: ProjectScript,
        *,
        acoustic_hints: dict[int, dict] | None = None,
        show_bible: dict | None = None,
    ):
        self.script = script
        if acoustic_hints is not None:
            self.acoustic_hints = acoustic_hints
        else:
            from backend.services.emotion_routing import extract_segment_acoustics

            try:
                self.acoustic_hints = extract_segment_acoustics(script, use_ser=False, allow_gpu_ser=False)
            except (OSError, ValueError):
                self.acoustic_hints = {}
        self.speaker_pace_scale = _compute_speaker_pace_scales(script, self.acoustic_hints)
        self.show_bible = {k: v for k, v in (show_bible or {}).items() if v}
        self.ordered = sorted(script.segments, key=lambda seg: (seg.start, seg.id))
        self.positions = {seg.id: index for index, seg in enumerate(self.ordered)}

    def _line(self, seg: Segment) -> dict:
        character = self.script.characters.get(seg.speaker)
        return {
            "id": seg.id,
            "speaker": seg.speaker,
            "speaker_name": character.display_name if character else seg.speaker,
            "speaking_style": character.speaking_style if character else "",
            "orig_text": seg.orig_text,
            "target_text": seg.target_text if seg.target_text != seg.orig_text else "",
        }

    def entry(self, seg: Segment, *, actual_duration: float | None = None) -> dict:
        index = self.positions[seg.id]
        slot = seg.end - seg.start
        cjk = seg.target_lang.removeprefix("all_").split("-")[0] in {"zh", "ja", "yue", "ko"}
        hint = self.acoustic_hints.get(seg.id)
        effective_speech_sec = slot
        pause_sec = 0.0
        if hint and hint.get("energy_active_ratio") is not None:
            active_ratio = float(hint["energy_active_ratio"])
            effective_speech_sec = min(slot, max(0.55, slot * min(1.0, active_ratio * 1.15)))
            pause_sec = round(max(0.0, slot - effective_speech_sec), 2)
            pace = self.speaker_pace_scale.get(seg.speaker, 1.0)
            budget = max(1, int(effective_speech_sec * (4.5 if cjk else 2.6) * pace))
        else:
            budget = max(1, int(slot * (4.6 if cjk else 2.7)))
        if actual_duration is not None:
            units = len(re.sub(r"\s+", "", seg.target_text)) if cjk else len(seg.target_text.split())
            if actual_duration > slot:
                budget = min(budget, max(1, int(units * slot / actual_duration * 0.92)))
            else:
                # For underflowing lines, allow expanding back toward natural fill (~0.88x slot).
                budget = max(budget, int(units * slot / max(0.2, actual_duration) * 0.88))
        item = {
            **self._line(seg),
            "target_lang": seg.target_lang,
            "emotion": seg.emotion,
            "slot_sec": round(slot, 3),
            "unit": "characters" if cjk else "words",
            "min_units": max(1, int(budget * 0.78)),
            "max_units": budget,
            "before": [self._line(other) for other in self.ordered[max(0, index - 2):index]],
            "after": [self._line(other) for other in self.ordered[index + 1:index + 3]],
        }
        if hint:
            acoustic_entry = {
                key: value
                for key, value in (
                    ("ser_top", hint.get("ser_top")),
                    ("ser_score", hint.get("ser_score")),
                    ("loudness_z", hint.get("rms_z")),
                    ("pitch_span_hz", hint.get("f0_range_hz")),
                    ("suggested_emotion", hint.get("suggested_emotion")),
                    ("effective_speech_sec", round(effective_speech_sec, 2) if hint.get("energy_active_ratio") is not None else None),
                    ("internal_pause_sec", pause_sec if pause_sec >= 0.35 else None),
                )
                if value is not None
            }
            if acoustic_entry:
                item["acoustic_hint"] = acoustic_entry
        line_notes = self.show_bible.get("line_notes")
        if isinstance(line_notes, dict):
            note_entry = line_notes.get(str(seg.id))
            if isinstance(note_entry, dict):
                note_text = str(note_entry.get("note") or "").strip()
                note_emotion = str(note_entry.get("emotion") or "").strip()
                if note_text:
                    item["director_note"] = note_text
                if note_emotion in EMOTIONS:
                    item["director_emotion"] = note_emotion
        return item

    def _public_show_bible(self) -> dict:
        return {k: v for k, v in self.show_bible.items() if k != "line_notes" and v}

    def payload(self, entries: Iterable[dict]) -> dict:
        lines = list(entries)
        data = {
            "source_lang": self.script.source_lang,
            "glossary": self.script.glossary,
            "constraints": {
                "fidelity": "逐句完整翻译，不得摘要、概括、省略或改成名词短语",
                "glossary": "术语按固定译法，不得同义改写",
                "spoken_form": "中文输出中的英文缩写/单字母必须是可朗读形式",
                "budget": "min_units/max_units 是软预算，语义完整优先",
            },
            "batch_ids": [line.get("id") for line in lines],
            "lines": lines,
        }
        bible = self._public_show_bible()
        if bible:
            data["show_bible"] = bible
        return data

    def whole_dialogue_payload(
        self,
        segments: Iterable[Segment],
        *,
        full_segments: Iterable[Segment] | None = None,
    ) -> dict:
        """构造带全文上下文的分批翻译请求，并保留原句 id 供结果还原。"""
        ordered = sorted(segments, key=lambda seg: (seg.start, seg.id))
        dialogue_source = sorted(full_segments or segments, key=lambda seg: (seg.start, seg.id))
        lines = [self.entry(seg) for seg in ordered]
        dialogue_lines = []
        for seg in dialogue_source:
            character = self.script.characters.get(seg.speaker)
            speaker_name = character.display_name if character else seg.speaker
            dialogue_lines.append(f"[{seg.id}] {speaker_name}: {seg.orig_text.strip()}")
        target_lang = next((seg.target_lang for seg in ordered if seg.target_lang), "zh")
        payload = self.payload(lines)
        payload.update({
            "target_lang": target_lang,
            "dialogue": "\n".join(dialogue_lines),
            "output_requirements": (
                "dialogue 仅作完整上下文，不翻译其中未列入 lines 的 id；"
                "先理解整段对白，再逐条还原到 lines 中的 id；"
                "只输出 JSON 数组，不输出完整译文正文；"
                "每项必须包含 id、target_text、emotion、action、keep_original、self_check；"
                "self_check.omitted_info 为空且 semantic_complete=true，否则会被判为漏译。"
            ),
        })
        return payload

    def polish_payload(
        self,
        segments: Iterable[Segment],
        *,
        full_segments: Iterable[Segment] | None = None,
    ) -> dict:
        """构造 Pass 2 口语化润色请求，保留核心语义与术语预算。"""
        ordered = sorted(segments, key=lambda seg: (seg.start, seg.id))
        dialogue_source = sorted(full_segments or segments, key=lambda seg: (seg.start, seg.id))
        lines = []
        for seg in ordered:
            item = self.entry(seg)
            item["draft_text"] = seg.target_text
            item.pop("target_text", None)
            lines.append(item)
        dialogue_lines = []
        for seg in dialogue_source:
            character = self.script.characters.get(seg.speaker)
            speaker_name = character.display_name if character else seg.speaker
            dialogue_lines.append(f"[{seg.id}] {speaker_name}: {seg.orig_text.strip()}")
        target_lang = next((seg.target_lang for seg in ordered if seg.target_lang), "zh")
        data: dict = {
            "source_lang": self.script.source_lang,
            "target_lang": target_lang,
            "glossary": self.script.glossary,
            "dialogue": "\n".join(dialogue_lines),
            "batch_ids": [line.get("id") for line in lines],
            "lines": lines,
            "output_requirements": (
                "仅润色 lines 中的片段 ID，保持自然口语与角色人设；"
                "保留核心事实、数字、否定与 glossary 固定译法；"
                "只输出 JSON 数组：[{\"id\": int, \"target_text\": str, \"self_check\": {\"semantic_complete\": bool, \"omitted_info\": str}}]。"
            ),
        }
        bible = self._public_show_bible()
        if bible:
            data["show_bible"] = bible
        return data


def _self_check_flags(entry: dict, *, require: bool = False) -> tuple[bool, str]:
    check = entry.get("self_check")
    if not isinstance(check, dict):
        return (False if require else True), ""
    complete = check.get("semantic_complete")
    omitted = str(check.get("omitted_info") or "").strip()
    ok = complete is True if require else complete is not False
    return ok, omitted


def _term_in_text(text: str, term: str) -> bool:
    """Apply alphanumeric boundaries only on alphanumeric edges; CJK edges match directly."""
    if not term:
        return False
    if re.search(r"[A-Za-z0-9]", term):
        prefix = r"(?<![A-Za-z0-9])" if re.match(r"^[A-Za-z0-9]", term) else ""
        suffix = r"(?![A-Za-z0-9])" if re.search(r"[A-Za-z0-9]$", term) else ""
        pattern = f"{prefix}{re.escape(term)}{suffix}"
        return re.search(pattern, text, re.IGNORECASE) is not None
    return term in text


def _fidelity_issues(
    seg: Segment,
    text: str,
    glossary: dict[str, str] | None = None,
    *,
    include_length: bool = True,
) -> list[str]:
    """Flag fidelity problems; length is a soft heuristic and can be disabled after repair."""
    source = seg.orig_text.strip()
    candidate = text.strip()
    issues: list[str] = []
    if include_length and len(source) >= 12 and len(candidate) <= max(2, int(len(source) * 0.25)):
        issues.append("译文过短，疑似摘要或名词短语")

    glossary = glossary or {}
    for term, target in glossary.items():
        if not isinstance(term, str) or not isinstance(target, str) or not term.strip() or not target.strip():
            continue
        if _term_in_text(source, term) and not _term_in_text(candidate, target):
            issues.append(f"术语“{term}”未使用固定译法“{target}”")

    for number in re.findall(r"\d+(?:[.,]\d+)*", source):
        if number not in candidate:
            issues.append(f"数字“{number}”未保留")

    for acronym in re.findall(r"\b[A-Z]{2,}(?:[-/][A-Z]{2,})*\b", source):
        if _term_in_text(candidate, acronym):
            continue
        mapped = next((target for term, target in glossary.items()
                       if isinstance(term, str) and isinstance(target, str)
                       and term.strip().casefold() == acronym.casefold()), "")
        if mapped and _term_in_text(candidate, mapped):
            continue
        issues.append(f"缩写“{acronym}”未保留或未按术语表改写")
    return issues


def _is_suspicious(seg: Segment, text: str, glossary: dict[str, str] | None = None) -> bool:
    """Heuristic only: triggers one repair, never a hard rejection by itself."""
    return bool(_fidelity_issues(seg, text, glossary))


def _repair_entries(
    client: OpenAI,
    context: DialogueContext,
    suspicious: list[tuple[Segment, dict]],
    errors: list[str] | None,
) -> dict[int, dict]:
    if not suspicious:
        return {}
    allowed_ids = {seg.id for seg, _ in suspicious}
    payload = context.payload([
        {
            **context.entry(seg),
            "previous_target_text": str(previous.get("target_text") or ""),
            "repair_reason": "、".join(_fidelity_issues(seg, str(previous.get("target_text") or ""), context.script.glossary))
            or "疑似摘要化或漏译",
        }
        for seg, previous in suspicious
    ])
    try:
        entries = complete_dialogue(client, TRANSLATION_REPAIR_PROMPT, payload)
    except (OpenAIError, ValueError, IndexError, json.JSONDecodeError):
        return {}
    repaired: dict[int, dict] = {}
    seen: set[int] = set()
    for entry in entries:
        seg_id, text = entry.get("id"), entry.get("target_text")
        if type(seg_id) is not int or seg_id not in allowed_ids:
            if errors is not None:
                errors.append(f"片段 #{seg_id} 不在修复白名单，忽略")
            continue
        if seg_id in seen:
            if errors is not None:
                errors.append(f"片段 #{seg_id} 修复结果重复，忽略后一次")
            continue
        seen.add(seg_id)
        if not isinstance(text, str) or not text.strip():
            if errors is not None:
                errors.append(f"片段 #{seg_id} 修复结果为空，保留上一版")
            continue
        repaired[seg_id] = {
            "id": seg_id,
            "target_text": text.strip(),
            # Keep the repair's self-check so a false/incomplete result cannot
            # bypass validation when it is applied.
            "self_check": entry.get("self_check"),
        }
    for seg_id in sorted(allowed_ids - seen):
        if errors is not None:
            errors.append(f"片段 #{seg_id} 未返回修复结果，保留上一版")
    return repaired


def _is_non_speech(text: str) -> bool:
    """Conservative evidence check for keeping the original source audio.

    Only an explicit bracketed/parenthesized event marker is evidence. Language
    content and bare onomatopoeia are never treated as events by length alone.
    """
    return bool(_NON_SPEECH_EVENT_RE.search(text or ""))


def _apply_translation(
    seg: Segment,
    text: str,
    entry: dict,
    *,
    preserve_metadata: bool = False,
) -> bool:
    text = clean_dubbing_text(text)
    complete, omitted = _self_check_flags(entry, require=preserve_metadata)
    if complete is False or omitted or not text:
        return False

    if preserve_metadata:
        # Repair only supplies replacement text; never reset emotion/action/keep_original.
        emotion = seg.emotion
        action = seg.action
        keep_original = seg.keep_original
    else:
        requested_emotion = entry.get("emotion")
        emotion = requested_emotion if isinstance(requested_emotion, str) and requested_emotion in EMOTIONS else seg.emotion
        requested_action = entry.get("action")
        action = requested_action.strip() if isinstance(requested_action, str) and requested_action.strip() else (seg.action or "idle")
        requested_keep = entry.get("keep_original") is True
        # A human-set keep_original is never cancelled by translation/repair.
        # The model alone cannot turn ordinary dialogue into "keep source audio".
        keep_original = seg.keep_original or (requested_keep and _is_non_speech(seg.orig_text))

    needs_resynth = (text, emotion, keep_original) != (seg.target_text, seg.emotion, seg.keep_original)
    seg.target_text = text
    seg.emotion = emotion
    seg.action = action
    seg.keep_original = keep_original
    if needs_resynth:
        seg.invalidate_audio()
    return True


def _batch_error(errors: list[str] | None, message: str) -> None:
    if errors is not None:
        errors.append(message)


def _merge_glossary_entries(
    glossary: dict[str, str],
    additions: object,
    *,
    protected_keys: set[str],
    allow_overwrite_provisional: bool = False,
) -> int:
    """Merge discovered glossary terms without clobbering user-supplied entries."""
    if not isinstance(additions, dict):
        return 0
    merged = 0
    for raw_term, raw_target in additions.items():
        if not isinstance(raw_term, str) or not isinstance(raw_target, str):
            continue
        term, target = raw_term.strip(), clean_dubbing_text(raw_target)
        if not term or not target:
            continue
        if term in protected_keys:
            continue
        if not allow_overwrite_provisional and term in glossary:
            continue
        glossary[term] = target
        merged += 1
    return merged


def _analyze_script(
    script: ProjectScript,
    segments: list[Segment],
    client: OpenAI,
    *,
    target_lang: str,
    enable_web_search: bool,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run Pass 0 script/character analysis and optional Pass 0.5 web term lookup."""
    if not segments:
        return {}
    user_glossary_keys = {k.strip() for k in script.glossary if isinstance(k, str) and k.strip()}
    dialogue_lines = []
    for seg in segments:
        character = script.characters.get(seg.speaker)
        speaker_name = character.display_name if character else seg.speaker
        dialogue_lines.append(f"[{seg.id}] {speaker_name} ({seg.speaker}): {seg.orig_text.strip()}")

    characters_summary = {
        spk: {
            "display_name": ch.display_name,
            "speaking_style": ch.speaking_style,
        }
        for spk, ch in script.characters.items()
    }
    analysis_payload = {
        "source_lang": script.source_lang,
        "target_lang": target_lang,
        "characters": characters_summary,
        "existing_glossary": script.glossary,
        "dialogue": "\n".join(dialogue_lines),
    }

    if log:
        log("Pass 0：通读全片对白，提取人物性格、称呼关系与专有名词")
    try:
        analysis = complete_json_object(client, SCRIPT_ANALYSIS_PROMPT, analysis_payload, temperature=0.2)
    except Exception as exc:
        if log:
            log(f"Pass 0 剧本分析跳过（{exc}）")
        return {}

    styles_filled = 0
    raw_styles = analysis.get("character_styles")
    if isinstance(raw_styles, dict):
        for spk, raw_style in raw_styles.items():
            if (
                isinstance(spk, str)
                and isinstance(raw_style, str)
                and raw_style.strip()
                and spk in script.characters
                and not (script.characters[spk].speaking_style or "").strip()
            ):
                script.characters[spk].speaking_style = raw_style.strip()
                styles_filled += 1

    terms_added = _merge_glossary_entries(
        script.glossary,
        analysis.get("glossary"),
        protected_keys=user_glossary_keys,
        allow_overwrite_provisional=False,
    )

    show_bible: dict = {}
    for key in ("genre", "tone"):
        val = analysis.get(key)
        if isinstance(val, str) and val.strip():
            show_bible[key] = val.strip()
    relationships = analysis.get("relationships")
    if isinstance(relationships, list):
        cleaned_rel = [str(item).strip() for item in relationships if isinstance(item, str) and item.strip()]
        if cleaned_rel:
            show_bible["relationships"] = cleaned_rel
    if isinstance(raw_styles, dict):
        cleaned_styles = {
            str(k).strip(): str(v).strip()
            for k, v in raw_styles.items()
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
        }
        if cleaned_styles:
            show_bible["character_styles"] = cleaned_styles

    raw_scenes = analysis.get("scenes")
    if isinstance(raw_scenes, list):
        cleaned_scenes = []
        for scene in raw_scenes:
            if not isinstance(scene, dict):
                continue
            scene_range = str(scene.get("range") or "").strip()
            scene_summary = str(scene.get("summary") or "").strip()
            if scene_range and scene_summary:
                cleaned_scenes.append({"range": scene_range, "summary": scene_summary})
        if cleaned_scenes:
            show_bible["scenes"] = cleaned_scenes

    raw_line_notes = analysis.get("line_notes")
    if isinstance(raw_line_notes, dict):
        cleaned_notes: dict[str, dict[str, str]] = {}
        for raw_id, raw_note in raw_line_notes.items():
            if not isinstance(raw_note, dict):
                continue
            seg_key = str(raw_id).strip()
            if not seg_key:
                continue
            note_text = str(raw_note.get("note") or "").strip()
            note_emotion = str(raw_note.get("emotion") or "").strip()
            cleaned_entry: dict[str, str] = {}
            if note_text:
                cleaned_entry["note"] = note_text
            if note_emotion in EMOTIONS:
                cleaned_entry["emotion"] = note_emotion
            if cleaned_entry:
                cleaned_notes[seg_key] = cleaned_entry
        if cleaned_notes:
            show_bible["line_notes"] = cleaned_notes

    if log:
        note_count = len(show_bible.get("line_notes") or {})
        extra = f"，导演批注 {note_count} 条" if note_count else ""
        log(f"Pass 0 完成：自动补全 {styles_filled} 个角色风格，新增 {terms_added} 条术语{extra}")

    raw_uncertain = analysis.get("uncertain_terms")
    uncertain_terms = [
        item.strip()
        for item in (raw_uncertain if isinstance(raw_uncertain, list) else [])
        if isinstance(item, str) and item.strip()
    ][:8]

    if enable_web_search and uncertain_terms:
        if log:
            log(f"Pass 0.5：联网核查 {len(uncertain_terms)} 个专有名词/典故（{', '.join(uncertain_terms)}）")
        try:
            hits = web_search.search_terms(
                uncertain_terms,
                source_lang=script.source_lang,
                target_lang=target_lang,
            )
            if hits:
                if log:
                    log(f"联网命中 {len(hits)} 个词条参考，正在结合语境校对译名")
                refine_payload = {
                    "source_lang": script.source_lang,
                    "target_lang": target_lang,
                    "uncertain_terms": uncertain_terms,
                    "existing_glossary": script.glossary,
                    "web_results": hits,
                }
                refined = complete_json_object(client, GLOSSARY_WEB_REFINE_PROMPT, refine_payload, temperature=0.2)
                verified_added = _merge_glossary_entries(
                    script.glossary,
                    refined.get("glossary"),
                    protected_keys=user_glossary_keys,
                    allow_overwrite_provisional=True,
                )
                notes = refined.get("background_notes")
                if isinstance(notes, list):
                    cleaned_notes = [str(n).strip() for n in notes if isinstance(n, str) and n.strip()]
                    if cleaned_notes:
                        show_bible["background_notes"] = cleaned_notes
                if log:
                    log(f"Pass 0.5 完成：联网确认/更新 {verified_added} 条专有名词译法")
            elif log:
                log("Pass 0.5：未检索到可用外部条目，沿用语境译法")
        except Exception as exc:
            if log:
                log(f"Pass 0.5 联网校对跳过（{exc}）")

    return show_bible


def _polish_batch(
    client: OpenAI,
    context: DialogueContext,
    batch: list[Segment],
    *,
    full_segments: list[Segment],
    log: Callable[[str], None] | None = None,
) -> int:
    """Run Pass 2 dubbing polish on completed segments, keeping Pass 1 draft on any guard failure."""
    targets = [seg for seg in batch if not seg.keep_original and seg.target_text.strip()]
    if not targets:
        return 0
    allowed_ids = {seg.id for seg in targets}
    seg_by_id = {seg.id: seg for seg in targets}
    payload = context.polish_payload(targets, full_segments=full_segments)
    try:
        entries = complete_dialogue(client, DIALOGUE_POLISH_PROMPT, payload, temperature=0.55)
    except (OpenAIError, ValueError, IndexError, json.JSONDecodeError) as exc:
        if log:
            log(f"Pass 2 口语润色跳过（保留初译）：{exc}")
        return 0

    polished_count = 0
    seen: set[int] = set()
    for entry in entries:
        seg_id = entry.get("id")
        text = entry.get("target_text")
        if type(seg_id) is not int or seg_id not in allowed_ids or seg_id in seen:
            continue
        seen.add(seg_id)
        if not isinstance(text, str) or not text.strip():
            continue
        seg = seg_by_id[seg_id]
        complete, omitted = _self_check_flags(entry, require=True)
        if not complete or omitted:
            continue
        if _fidelity_issues(seg, text, context.script.glossary, include_length=False):
            continue
        if _apply_translation(seg, text, entry, preserve_metadata=True):
            polished_count += 1
    return polished_count


def _text_quality_issues(seg: Segment, context: DialogueContext) -> list[str]:
    """Identify post-translation density or spoken-form defects for Director QA."""
    text = (seg.target_text or "").strip()
    if not text or seg.keep_original:
        return []
    entry = context.entry(seg)
    cjk = entry["unit"] == "characters"
    units = len(re.sub(r"\s+", "", text)) if cjk else len(text.split())
    max_units = int(entry["max_units"])
    slot = max(0.2, seg.end - seg.start)
    issues: list[str] = []
    if re.search(r"[\u4e00-\u9fff]\s*[／/]\s*[\u4e00-\u9fff]", text):
        issues.append("包含双选斜杠，不符合口语朗读")
    if units >= 12 and max_units >= 4 and units > int(round(max_units * 1.15)):
        issues.append(f"台词偏长（{units}/{max_units} {entry['unit']}，预计语速偏赶）")
    if slot >= 4.5 and max_units >= 15 and units < int(round(max_units * 0.70)):
        issues.append(f"台词偏短（{units}/{entry['min_units']}-{max_units} {entry['unit']}，长槽位易留大空白），请还原原文修饰与口语衔接")
    if cjk and units <= 5 and slot <= 1.2 and re.search(r"(?:\.{2,}|…+)\s*$", text):
        issues.append("短句末尾带省略号，易导致合成拖音")
    return issues


def _director_pre_qa(
    director: OpenAI,
    context: DialogueContext,
    segments: list[Segment],
    *,
    log: Callable[[str], None] | None = None,
) -> int:
    """Run targeted Director QA tightening on flagged dense or malformed lines only."""
    flagged: list[tuple[Segment, list[str]]] = []
    for seg in segments:
        reasons = _text_quality_issues(seg, context)
        if reasons:
            flagged.append((seg, reasons))
    if not flagged:
        return 0
    if log:
        log(f"Pre-TTS 导演终审：发现 {len(flagged)} 条台词偏长或含口语瑕疵，正在自动精炼")

    chunk = 12
    updated_count = 0
    for offset in range(0, len(flagged), chunk):
        part = flagged[offset:offset + chunk]
        allowed_ids = {seg.id for seg, _ in part}
        seg_by_id = {seg.id: seg for seg, _ in part}
        lines = []
        for seg, reasons in part:
            item = context.entry(seg)
            item["draft_text"] = seg.target_text
            item["qa_issues"] = reasons
            item.pop("target_text", None)
            lines.append(item)
        payload = {
            "source_lang": context.script.source_lang,
            "target_lang": part[0][0].target_lang,
            "glossary": context.script.glossary,
            "batch_ids": [item["id"] for item in lines],
            "lines": lines,
        }
        bible = context._public_show_bible()
        if bible:
            payload["show_bible"] = bible
        try:
            entries = complete_dialogue(director, DIALOGUE_QA_TIGHTEN_PROMPT, payload, temperature=0.3)
        except (OpenAIError, ValueError, IndexError, json.JSONDecodeError) as exc:
            if log:
                log(f"Pre-TTS 导演终审跳过：{exc}")
            continue
        for entry in entries:
            seg_id = entry.get("id")
            text = entry.get("target_text")
            if type(seg_id) is not int or seg_id not in allowed_ids or not isinstance(text, str):
                continue
            seg = seg_by_id[seg_id]
            complete, omitted = _self_check_flags(entry, require=True)
            if not complete or omitted:
                continue
            if _fidelity_issues(seg, text, context.script.glossary, include_length=False):
                continue
            if _apply_translation(seg, text, entry, preserve_metadata=True):
                updated_count += 1
    if log and updated_count:
        log(f"Pre-TTS 导演终审完成：自动精炼 {updated_count} 条台词")
    return updated_count


def translate_script(
    script: ProjectScript,
    client: OpenAI,
    log: Callable[[str], None] | None = None,
    errors: list[str] | None = None,
    *,
    enable_analysis: bool = False,
    enable_polish: bool = False,
    enable_web_search: bool = False,
    director_client: OpenAI | None = None,
) -> list[int]:
    active_director = director_client or client
    if log and director_client is not None and director_client is not client:
        log(f"双层 LLM 模式：导演={client_model(active_director)} / 翻译={client_model(client)}")
    acoustic_hints: dict[int, dict] = {}
    if script.media.vocals_wav:
        try:
            from backend.services.emotion_routing import extract_segment_acoustics

            acoustic_hints = extract_segment_acoustics(script, use_ser=True, allow_gpu_ser=False)
        except Exception:
            acoustic_hints = {}
    ordered_segments = [seg for seg in sorted(script.segments, key=lambda seg: (seg.start, seg.id)) if seg.orig_text.strip()]
    if not ordered_segments:
        return []

    target_lang = next((seg.target_lang for seg in ordered_segments if seg.target_lang), "zh")
    show_bible: dict = {}
    if enable_analysis or enable_web_search:
        show_bible = _analyze_script(
            script,
            ordered_segments,
            active_director,
            target_lang=target_lang,
            enable_web_search=enable_web_search,
            log=log,
        )
        if show_bible:
            script.show_bible = show_bible
    elif script.show_bible:
        show_bible = dict(script.show_bible)

    context = DialogueContext(script, acoustic_hints=acoustic_hints, show_bible=show_bible)
    segments = [seg for seg in context.ordered if seg.orig_text.strip()]

    chunk_size = max(1, config.TRANSLATION_CHUNK_SIZE)
    batch_count = (len(segments) + chunk_size - 1) // chunk_size
    if log:
        log(f"使用完整 {len(segments)} 条对白作为上下文，按 {chunk_size} 条分批翻译（共 {batch_count} 批）")

    translated: list[int] = []
    completed_ids: set[int] = set()
    previous_ids: set[int] = set()
    all_ids = {seg.id for seg in segments}
    future_ids = set(all_ids)

    for offset in range(0, len(segments), chunk_size):
        batch = segments[offset:offset + chunk_size]
        batch_ids = {seg.id for seg in batch}
        future_ids -= batch_ids
        batch_payload = context.whole_dialogue_payload(batch, full_segments=segments)
        if log:
            log(f"调用导演 LLM 通读全文并翻译片段 #{batch[0].id}..#{batch[-1].id}（{len(batch)} 条）")
        try:
            entries = complete_dialogue(client, DIRECTOR_TRANSLATE_PROMPT, batch_payload)
        except (OpenAIError, ValueError, IndexError, json.JSONDecodeError) as exc:
            _batch_error(errors, f"片段 #{batch[0].id}..#{batch[-1].id} 翻译失败：{exc}")
            if log:
                log("本批翻译失败，保留当前台词")
            previous_ids |= batch_ids
            continue

        responded: set[int] = set()
        suspicious: list[tuple[Segment, dict]] = []
        for entry in entries:
            seg_id = entry.get("id")
            text = entry.get("target_text")
            if type(seg_id) is not int:
                _batch_error(errors, f"批次 #{batch[0].id}..#{batch[-1].id} 返回非法 id，忽略")
                continue
            if seg_id not in batch_ids:
                if seg_id in previous_ids:
                    kind = "属于已处理批次（重复/越批返回）"
                elif seg_id in future_ids:
                    kind = "属于后续批次（提前返回）"
                else:
                    kind = "不在本次任务片段内"
                _batch_error(errors, f"片段 #{seg_id} {kind}，忽略")
                continue
            if seg_id in responded:
                _batch_error(errors, f"片段 #{seg_id} 在当前批次重复返回，忽略后一次")
                continue
            responded.add(seg_id)
            if not isinstance(text, str) or not text.strip():
                _batch_error(errors, f"片段 #{seg_id} 译文为空或类型错误，保留上一版")
                continue

            seg = next(segment for segment in batch if segment.id == seg_id)
            complete, omitted = _self_check_flags(entry)
            if not complete or omitted or _is_suspicious(seg, text, script.glossary):
                suspicious.append((seg, entry))
                continue
            if _apply_translation(seg, text, entry):
                completed_ids.add(seg.id)
                translated.append(seg.id)

        for seg_id in sorted(batch_ids - responded):
            _batch_error(errors, f"片段 #{seg_id} 未在本批次返回，保留上一版")

        previous_ids |= batch_ids
        if suspicious:
            repaired = _repair_entries(client, context, suspicious, errors)
            for seg, entry in suspicious:
                repair_entry = repaired.get(seg.id)
                text = repair_entry.get("target_text") if repair_entry else None
                issues = _fidelity_issues(seg, text or "", script.glossary, include_length=False)
                reasons = "、".join(issues) if text else "未返回修复结果"
                hard_issues = [issue for issue in issues if not issue.startswith("译文过短")]
                self_ok, self_note = _self_check_flags(repair_entry or {}, require=bool(repair_entry))
                if self_note:
                    hard_issues.append(f"self_check 标注遗漏：{self_note}")
                if not self_ok:
                    hard_issues.append("self_check.semantic_complete 非 true")
                if not text or hard_issues:
                    _batch_error(errors, f"片段 #{seg.id} 修复后仍存在明确漏译或未通过自检（{reasons}），保留上一版")
                    continue
                # Repair output is validated and applied without changing any metadata.
                if seg.id not in completed_ids and _apply_translation(seg, text, repair_entry, preserve_metadata=True):
                    completed_ids.add(seg.id)
                    translated.append(seg.id)

        if enable_polish:
            polish_candidates = [seg for seg in batch if seg.id in completed_ids and not seg.keep_original]
            if polish_candidates:
                polished = _polish_batch(active_director, context, polish_candidates, full_segments=segments, log=log)
                if log and polished:
                    log(f"Pass 2 完成：口语润色片段 #{batch[0].id}..#{batch[-1].id} 共 {polished}/{len(polish_candidates)} 条")

    if enable_polish and completed_ids:
        qa_candidates = [seg for seg in segments if seg.id in completed_ids and not seg.keep_original]
        _director_pre_qa(active_director, context, qa_candidates, log=log)

    return sorted(set(translated))
