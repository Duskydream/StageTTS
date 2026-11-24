"""Dialogue context shared by parse-time translation and duration rewrites."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

from openai import OpenAI, OpenAIError

from backend import config
from backend.agent.prompts import DIRECTOR_TRANSLATE_PROMPT, TRANSLATION_REPAIR_PROMPT
from backend.schemas import ProjectScript, Segment

EMOTIONS = {"neutral", "angry", "happy", "sad", "whisper"}
CONTEXT_INSTRUCTIONS = (
    "先通读完整对白理解剧情、角色关系和前后语气，再按原句 id 还原译文。"
    "参照相邻台词、角色语气和术语表；术语出现时使用固定译法。"
    "忠实优先：不得摘要、概括、省略、删词或改成名词短语；min_units/max_units 只是软预算。"
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


def client_model(client: OpenAI) -> str:
    model = getattr(client, "_stagetts_model", None)
    return model if isinstance(model, str) else config.LLM_MODEL


def complete_dialogue(client: OpenAI, system: str, payload: dict) -> list[dict]:
    response = client.chat.completions.create(
        model=client_model(client),
        messages=[
            {"role": "system", "content": system + "\n" + CONTEXT_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    content = (response.choices[0].message.content or "").strip()
    fences = re.findall(r"```(json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fences:
        content = (
            next((body.strip() for lang, body in fences if lang and "[" in body and "]" in body), None)
            or next((body.strip() for _, body in fences if "[" in body and "]" in body), None)
            or fences[0][1].strip()
        )
    left, right = content.find("["), content.rfind("]")
    if left == -1 or right <= left:
        raise ValueError("LLM 未返回 JSON 数组")
    entries = json.loads(content[left:right + 1])
    if not isinstance(entries, list):
        raise ValueError("LLM 未返回 JSON 数组")
    return [entry for entry in entries if isinstance(entry, dict)]


class DialogueContext:
    def __init__(self, script: ProjectScript, *, acoustic_hints: dict[int, dict] | None = None):
        self.script = script
        self.acoustic_hints = acoustic_hints or {}
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
        budget = max(1, int(slot * (5.0 if cjk else 2.8)))
        if actual_duration is not None:
            units = len(re.sub(r"\s+", "", seg.target_text)) if cjk else len(seg.target_text.split())
            budget = min(budget, max(1, int(units * slot / actual_duration * 0.95)))
        item = {
            **self._line(seg),
            "target_lang": seg.target_lang,
            "emotion": seg.emotion,
            "slot_sec": round(slot, 3),
            "unit": "characters" if cjk else "words",
            "min_units": max(1, int(budget * 0.55)),
            "max_units": budget,
            "before": [self._line(other) for other in self.ordered[max(0, index - 2):index]],
            "after": [self._line(other) for other in self.ordered[index + 1:index + 3]],
        }
        hint = self.acoustic_hints.get(seg.id)
        if hint:
            acoustic_entry = {
                key: value
                for key, value in (
                    ("ser_top", hint.get("ser_top")),
                    ("ser_score", hint.get("ser_score")),
                    ("loudness_z", hint.get("rms_z")),
                    ("pitch_span_hz", hint.get("f0_range_hz")),
                    ("suggested_emotion", hint.get("suggested_emotion")),
                )
                if value is not None
            }
            if acoustic_entry:
                item["acoustic_hint"] = acoustic_entry
        return item

    def payload(self, entries: Iterable[dict]) -> dict:
        lines = list(entries)
        return {
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


def _self_check_flags(entry: dict, *, require: bool = False) -> tuple[bool, str]:
    check = entry.get("self_check")
    if not isinstance(check, dict):
        return (False if require else True), ""
    complete = check.get("semantic_complete")
    omitted = str(check.get("omitted_info") or "").strip()
    ok = complete is True if require else complete is not False
    return ok, omitted


def _term_in_text(text: str, term: str) -> bool:
    """Latin terms need word boundaries; CJK terms are checked as substrings."""
    if not term:
        return False
    if re.search(r"[A-Za-z0-9]", term):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
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
    text = text.strip()
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


def translate_script(
    script: ProjectScript,
    client: OpenAI,
    log: Callable[[str], None] | None = None,
    errors: list[str] | None = None,
) -> list[int]:
    acoustic_hints: dict[int, dict] = {}
    if script.media.vocals_wav:
        try:
            from backend.services.emotion_routing import extract_segment_acoustics

            acoustic_hints = extract_segment_acoustics(script, use_ser=True, allow_gpu_ser=False)
        except Exception:
            acoustic_hints = {}
    context = DialogueContext(script, acoustic_hints=acoustic_hints)
    segments = [seg for seg in context.ordered if seg.orig_text.strip()]
    if not segments:
        return []

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
        if not suspicious:
            continue

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

    return sorted(set(translated))
