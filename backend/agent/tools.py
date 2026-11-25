"""Agent 五大确定性原子工具封装与 OpenAI Function Calling Schema。

所有工具均通过读写 workspace/{project_id}/script.json 保持全链路单一真相源。
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import threading
import wave
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from weakref import WeakValueDictionary

from backend import config
from backend.schemas import ProjectScript, Segment, SegmentPatch, StemVolumes, SubtitleBurnMode, SubtitleLayout, SynthesizeResponse
from backend.services.media_mixer import export_project_deliverables
from backend.services.synthesis import _check_duration, compute_tail_slack_map, synthesize_with_duration_check
from backend.services.task_runner import GpuBusyError, gpu_task_guard
from backend.services.tts_client import TTSServiceUnavailableError

_project_locks = WeakValueDictionary()
_project_locks_guard = threading.Lock()


@contextmanager
def project_task_guard(project_id: str) -> Iterator[None]:
    """同一项目的读改写操作不可并行，其他项目仍可保存和导出。"""
    with _project_locks_guard:
        lock = _project_locks.get(project_id)
        if lock is None:
            lock = _project_locks[project_id] = threading.RLock()
    if not lock.acquire(blocking=False):
        raise GpuBusyError("该项目正在解析、配音或导出，请在当前操作完成后重试。")
    try:
        yield
    finally:
        lock.release()


def _script_path(project_id: str) -> Path:
    path = (config.WORKSPACE_DIR / project_id / "script.json").resolve()
    workspace_root = config.WORKSPACE_DIR.resolve()
    if workspace_root not in path.parents:
        raise ValueError(f"非法的 project_id: {project_id}")
    return path


def load_project_script(project_id: str) -> ProjectScript:
    """读取并校验指定项目的 script.json。"""
    path = _script_path(project_id)
    if not path.exists():
        raise FileNotFoundError(f"未找到项目剧本文件: {path}")
    return ProjectScript.model_validate_json(path.read_text(encoding="utf-8"))


def save_project_script(script: ProjectScript) -> Path:
    """将校验后的 ProjectScript 持久化至 script.json。"""
    path = _script_path(script.project_id)
    with project_task_guard(script.project_id):
        # Revalidate mutated models as well as request payloads before persistence.
        validated = ProjectScript.model_validate(script.model_dump())
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            draft = Path(temp) / path.name
            draft.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
            draft.replace(path)
    return path


# ---------------------------------------------------------------------------
# 5 个确定性原子工具（见 ARCHITECTURE.md 第 5 节）
# ---------------------------------------------------------------------------


def query_script(
    project_id: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    speaker: Optional[str] = None,
) -> Dict[str, Any]:
    """按时间范围或说话人筛选查询剧本片段（只读，不修改文件）。"""
    script = load_project_script(project_id)
    matched = []
    for seg in script.segments:
        if start_sec is not None and seg.end < start_sec:
            continue
        if end_sec is not None and seg.start > end_sec:
            continue
        if speaker is not None and seg.speaker != speaker:
            continue
        matched.append(seg.model_dump())
    return {"count": len(matched), "segments": matched}


def batch_update_segments(
    project_id: str,
    updates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """应用片段补丁，并将影响音频的片段标为待合成。"""
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        seg_map = {seg.id: seg for seg in script.segments}
        updated_ids: List[int] = []
        marked_pending_ids: List[int] = []

        resynth_fields = {"speaker", "target_text", "target_lang", "emotion", "keep_original", "speed_factor"}

        for raw_patch in updates:
            patch = SegmentPatch.model_validate(raw_patch)
            if patch.id not in seg_map:
                raise KeyError(f"片段 ID {patch.id} 不存在于项目 {project_id}")
            seg = seg_map[patch.id]
            patch_data = patch.model_dump(exclude={"id"}, exclude_none=True)
            if "speaker" in patch_data:
                if patch_data["speaker"] not in script.characters:
                    raise ValueError("请先新建要指派的角色")
                if patch_data["speaker"] == "UNKNOWN":
                    patch_data["speaker_needs_review"] = True
                else:
                    patch_data.setdefault("speaker_needs_review", False)
                    if not patch_data["speaker_needs_review"]:
                        patch_data["speaker_fallback_reason"] = None
            if seg.speaker == "UNKNOWN" and "speaker" not in patch_data:
                patch_data["speaker_needs_review"] = True
            if any(field in patch_data and patch_data[field] != getattr(seg, field)
                   for field in ("orig_text", "start", "end", "speaker")):
                patch_data["segment_timing_needs_review"] = True
            if "orig_text" in patch_data and patch_data["orig_text"] != seg.orig_text:
                patch_data.setdefault("alignment_needs_review", True)
            needs_resynth = False
            for field_name, value in patch_data.items():
                if getattr(seg, field_name) != value:
                    setattr(seg, field_name, value)
                    if field_name in resynth_fields:
                        needs_resynth = True
            validated = Segment.model_validate(seg.model_dump())
            seg.speaker_needs_review = validated.speaker_needs_review
            seg.speaker_fallback_reason = validated.speaker_fallback_reason
            updated_ids.append(seg.id)
            if needs_resynth:
                seg.invalidate_audio()
            if needs_resynth and not seg.keep_original:
                marked_pending_ids.append(seg.id)

        save_project_script(script)
    return {"updated_ids": updated_ids, "marked_pending_ids": marked_pending_ids}


def synthesize_project(
    project_id: str,
    segment_ids: Optional[List[int]] = None,
    *,
    auto_reflect: bool = True,
) -> SynthesizeResponse:
    """合成并保存最终剧本，期间避免其他 GPU 任务切换服务。"""
    with project_task_guard(project_id), gpu_task_guard():
        script = load_project_script(project_id)
        try:
            synthesized_ids, overflow_segments, errors = synthesize_with_duration_check(
                script, segment_ids, auto_reflect=auto_reflect,
            )
        except TTSServiceUnavailableError:
            save_project_script(script)
            raise
        save_project_script(script)
    return SynthesizeResponse(
        project_id=project_id,
        synthesized_ids=synthesized_ids,
        overflow_segments=overflow_segments,
        errors=errors,
        script=script,
    )


def synthesize_segments(
    project_id: str,
    segment_ids: Optional[List[int]] = None,
    *,
    auto_reflect: bool = True,
) -> Dict[str, Any]:
    result = synthesize_project(project_id, segment_ids, auto_reflect=auto_reflect)
    return result.model_dump(exclude={"project_id", "script"})


def update_mix_settings(
    project_id: str,
    bgm_volume: Optional[float] = None,
    singing_volume: Optional[float] = None,
    music_volume: Optional[float] = None,
    effects_volume: Optional[float] = None,
    auto_ducking: Optional[bool] = None,
) -> Dict[str, Any]:
    """更新背景混音主音量、四轨子音量与自动闪避开关（0.0 ~ 2.0）并持久化 script.json。"""
    def _clamp(val: float) -> float:
        return round(max(0.0, min(2.0, float(val))), 3)

    with project_task_guard(project_id):
        script = load_project_script(project_id)
        if bgm_volume is not None:
            script.media.bgm_volume = _clamp(bgm_volume)
        if any(v is not None for v in (singing_volume, music_volume, effects_volume)):
            current = script.media.stem_volumes or StemVolumes()
            script.media.stem_volumes = StemVolumes(
                singing=_clamp(singing_volume) if singing_volume is not None else current.singing,
                music=_clamp(music_volume) if music_volume is not None else current.music,
                effects=_clamp(effects_volume) if effects_volume is not None else current.effects,
            )
        if auto_ducking is not None:
            script.media.auto_ducking = bool(auto_ducking)
        save_project_script(script)
    return {
        "bgm_volume": script.media.bgm_volume,
        "stem_volumes": script.media.stem_volumes.model_dump() if script.media.stem_volumes else None,
        "auto_ducking": script.media.auto_ducking,
    }


def render_final_video(
    project_id: str,
    burn_watermark: bool = False,
    burn_subtitles: SubtitleBurnMode = "none",
    export_ass_sidecar: Optional[SubtitleLayout] = None,
    auto_ducking: Optional[bool] = None,
    auto_fit_overflow: bool = False,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """触发 pydub 拼轨与 FFmpeg 混音封装，输出成片路径并回写 script.json。"""
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        result = export_project_deliverables(
            script,
            burn_watermark=burn_watermark,
            burn_subtitles=burn_subtitles,
            export_ass_sidecar=export_ass_sidecar,
            auto_ducking=auto_ducking,
            auto_fit_overflow=auto_fit_overflow,
            output_filename=output_filename,
            output_dir=output_dir,
        )
        script.media.output_video = result["output_video"]
        save_project_script(script)
    return result


def _wav_edge_and_activity_metrics(wav_path: Path) -> tuple[float, float, float] | None:
    """Measure (lead_silence_sec, tail_silence_sec, active_ratio) on a synthesized PCM16 WAV."""
    if not wav_path.is_file():
        return None
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            nframes = wf.getnframes()
            if rate <= 0 or channels <= 0 or width != 2 or nframes <= 0:
                return None
            raw = wf.readframes(nframes)
    except (OSError, wave.Error, EOFError):
        return None

    hop = max(1, round(rate * 0.02))
    hop_sec = hop / float(rate)
    samples = array("h")
    samples.frombytes(raw)
    stride = hop * channels
    rms_list: list[float] = []
    for pos in range(0, len(samples) - stride + 1, stride):
        sq = 0.0
        for idx in range(pos, pos + stride, channels):
            v = sum(samples[idx + ch] for ch in range(channels)) / (channels * 32768.0)
            sq += v * v
        rms_list.append(math.sqrt(sq / hop))
    if not rms_list:
        return None
    peak = max(rms_list)
    if peak <= 1e-4:
        return (round(len(rms_list) * hop_sec, 3), 0.0, 0.0)
    gate = max(10 ** (-44.0 / 20.0), peak * 0.06)
    active_indices = [idx for idx, val in enumerate(rms_list) if val >= gate]
    if not active_indices:
        return (round(len(rms_list) * hop_sec, 3), 0.0, 0.0)
    lead = round(active_indices[0] * hop_sec, 3)
    tail = round((len(rms_list) - 1 - active_indices[-1]) * hop_sec, 3)
    active_ratio = round(len(active_indices) / float(len(rms_list)), 3)
    return lead, tail, active_ratio


def _split_text_near_ratio(text: str, ratio: float) -> tuple[str, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return "", ""
    target = max(1, min(len(cleaned) - 1, round(len(cleaned) * max(0.2, min(0.8, ratio)))))
    candidates = [m.end() for m in re.finditer(r"[。！？；.!?;、，,:：]", cleaned)]
    valid = [pos for pos in candidates if 2 <= pos <= len(cleaned) - 2]
    if valid:
        cut = min(valid, key=lambda pos: abs(pos - target))
    else:
        cut = target
    left = cleaned[:cut].strip()
    right = cleaned[cut:].strip()
    if not left or not right:
        return cleaned[:target].strip(), cleaned[target:].strip()
    return left, right


def inspect_project_quality(project_id: str) -> Dict[str, Any]:
    """全片剧本与已合成 WAV 诊断体检，返回按严重度排序的可执行问题清单。"""
    script = load_project_script(project_id)
    ordered = sorted(script.segments, key=lambda seg: (seg.start, seg.id))
    tail_slacks = compute_tail_slack_map(ordered, script.media.duration)
    issues: List[Dict[str, Any]] = []

    for idx, seg in enumerate(ordered):
        slot = max(0.05, seg.end - seg.start)
        lang_prefix = seg.target_lang.removeprefix("all_").split("-")[0]
        cjk = lang_prefix in {"zh", "ja", "yue", "ko"}
        text = (seg.target_text or "").strip()
        units = len(re.sub(r"\s+", "", text)) if cjk else len(text.split())
        suggested_max = max(1, int(slot * (4.6 if cjk else 2.7)))
        suggested_min = max(1, int(suggested_max * 0.78))
        next_seg = ordered[idx + 1] if idx + 1 < len(ordered) else None
        seg_codes: list[str] = []
        details: list[str] = []

        if not seg.keep_original and text:
            if re.search(r"[\u4e00-\u9fff]\s*[／/]\s*[\u4e00-\u9fff]", text):
                seg_codes.append("FORBIDDEN_TEXT_PATTERN")
                details.append("台词含斜杠双选词（／）")
            if cjk and units <= 5 and slot <= 1.2 and re.search(r"(?:\.{2,}|…+)\s*$", text):
                seg_codes.append("FORBIDDEN_TEXT_PATTERN")
                details.append("短句末尾带省略号易拖长")
            if cjk and units >= 12 and (units / slot) > 5.3 and units > int(suggested_max * 1.15):
                seg_codes.append("TEXT_DENSE")
                details.append(f"译文偏密（{units}字/{slot:.1f}s≈{units / slot:.1f}字/秒，建议<={suggested_max}字）")
            elif cjk and slot >= 4.5 and suggested_max >= 15 and units < int(suggested_max * 0.70):
                seg_codes.append("TEXT_SPARSE")
                details.append(f"长槽位译文偏短（{units}/{suggested_min}-{suggested_max}字，易留冷场空隙）")

        if slot > 12.0 and re.search(r"[。！？；.!?;、，,]", seg.orig_text or ""):
            seg_codes.append("OVERLONG_SEGMENT")
            details.append(f"片段长达 {slot:.1f}s，建议用 split_segment 拆分")
        if (
            next_seg is not None
            and seg.speaker == next_seg.speaker
            and seg.speaker != "UNKNOWN"
            and 0 <= (next_seg.start - seg.end) <= 0.65
            and (seg.end - seg.start < 1.2 or units <= 4)
            and (next_seg.end - seg.start) <= 10.5
        ):
            seg_codes.append("TINY_FRAGMENT")
            details.append(f"与同角色 #{next_seg.id} 间距仅 {next_seg.start - seg.end:.2f}s，建议 merge_segments")

        if not seg.keep_original and seg.status == "ready" and seg.tts_wav:
            check = _check_duration(seg, tail_slack=tail_slacks.get(seg.id, 0.0))
            if check is not None:
                code = "DURATION_OVERFLOW" if check.duration_state == "too_long" else "DURATION_UNDERFLOW"
                seg_codes.append(code)
                details.append(f"实际 {check.actual_duration:.2f}s / 槽位 {slot:.2f}s (fill={check.fill_ratio:.2f})")
            if (seg.applied_speed_factor or 1.0) >= 1.12:
                seg_codes.append("HEAVY_SPEEDUP")
                details.append(f"被迫加速 {seg.applied_speed_factor:.2f}x")
            if seg.actual_duration and next_seg is not None and (seg.start + seg.actual_duration) > next_seg.start - 0.02:
                seg_codes.append("TIMELINE_COLLISION")
                details.append(f"配音结尾({seg.start + seg.actual_duration:.2f}s)侵入 #{next_seg.id}({next_seg.start:.2f}s)")
            try:
                wav_path = config.resolve_media_path(seg.tts_wav)
                metrics = _wav_edge_and_activity_metrics(wav_path)
                if metrics is not None:
                    lead, tail, act = metrics
                    dur = seg.actual_duration or 0.0
                    if lead > 0.42 or tail > 0.50 or (dur >= 1.3 and act < 0.35):
                        seg_codes.append("AUDIO_ANOMALY")
                        details.append(f"WAV异常(lead={lead:.2f}s, tail={tail:.2f}s, active={act:.2f})")
            except ValueError:
                pass

        if seg_codes:
            severity = "high" if any(c in {"FORBIDDEN_TEXT_PATTERN", "TIMELINE_COLLISION", "AUDIO_ANOMALY", "DURATION_OVERFLOW"} for c in seg_codes) else "medium"
            issues.append({
                "segment_id": seg.id,
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "slot_sec": round(slot, 3),
                "max_units": suggested_max,
                "orig_text": seg.orig_text,
                "target_text": seg.target_text,
                "actual_duration": seg.actual_duration,
                "applied_speed_factor": seg.applied_speed_factor,
                "codes": seg_codes,
                "severity": severity,
                "details": "；".join(details),
            })

    issues.sort(key=lambda item: (0 if item["severity"] == "high" else 1, item["segment_id"]))
    by_code: Dict[str, int] = {}
    for item in issues:
        for code in item["codes"]:
            by_code[code] = by_code.get(code, 0) + 1
    return {
        "project_id": project_id,
        "segment_count": len(ordered),
        "issue_count": len(issues),
        "code_counts": by_code,
        "issues": issues,
    }


def split_segment(
    project_id: str,
    segment_id: int,
    split_time_sec: Optional[float] = None,
    left_orig_text: Optional[str] = None,
    right_orig_text: Optional[str] = None,
    left_target_text: Optional[str] = None,
    right_target_text: Optional[str] = None,
) -> Dict[str, Any]:
    """将过长片段一分为二，自动或按指定时间点与文案切分并失效旧音频。"""
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        seg = next((item for item in script.segments if item.id == segment_id), None)
        if seg is None:
            raise KeyError(f"片段 ID {segment_id} 不存在")
        slot = seg.end - seg.start
        if slot < 1.6:
            raise ValueError(f"片段 #{segment_id} 时长仅 {slot:.2f}s，过短无法切分")

        split_at = split_time_sec
        if split_at is None and script.media.vocals_wav:
            try:
                from workers.moss_segmentation import _find_silence_valleys, _read_mono_envelope, _speech_threshold

                vocals_path = config.resolve_media_path(script.media.vocals_wav)
                env, hop, _ = _read_mono_envelope(vocals_path)
                thresh = _speech_threshold(env)
                if env and thresh is not None:
                    valleys = _find_silence_valleys(env, hop, seg.start, seg.end, thresh, min_silence_sec=0.18)
                    if valleys:
                        mid = (seg.start + seg.end) * 0.5
                        split_at = min(valleys, key=lambda v: abs(v[0] - mid) - min(0.3, v[1]) * 0.5)[0]
            except Exception:
                split_at = None
        if split_at is None:
            split_at = round((seg.start + seg.end) * 0.5, 3)
        split_at = round(float(split_at), 3)
        if not (seg.start + 0.5 <= split_at <= seg.end - 0.5):
            raise ValueError(f"切分时间点 {split_at} 必须位于 [{seg.start + 0.5:.2f}, {seg.end - 0.5:.2f}] 内")

        ratio = (split_at - seg.start) / slot
        if left_orig_text is None or right_orig_text is None:
            auto_lo, auto_ro = _split_text_near_ratio(seg.orig_text, ratio)
            left_orig = (left_orig_text if left_orig_text is not None else auto_lo).strip()
            right_orig = (right_orig_text if right_orig_text is not None else auto_ro).strip()
        else:
            left_orig, right_orig = left_orig_text.strip(), right_orig_text.strip()

        if left_target_text is None or right_target_text is None:
            auto_lt, auto_rt = _split_text_near_ratio(seg.target_text, ratio)
            left_tgt = (left_target_text if left_target_text is not None else auto_lt).strip()
            right_tgt = (right_target_text if right_target_text is not None else auto_rt).strip()
        else:
            left_tgt, right_tgt = left_target_text.strip(), right_target_text.strip()

        new_id = max((item.id for item in script.segments), default=-1) + 1
        right_seg = Segment(
            id=new_id,
            start=split_at,
            end=seg.end,
            speaker=seg.speaker,
            orig_text=right_orig,
            target_text=right_tgt,
            target_lang=seg.target_lang,
            emotion=seg.emotion,
            action=seg.action,
            keep_original=seg.keep_original,
            speed_factor=seg.speed_factor,
            alignment_granularity=seg.alignment_granularity,
            timing_source=seg.timing_source,
            segment_timing_needs_review=True,
            alignment_needs_review=True,
            speaker_needs_review=seg.speaker_needs_review,
            speaker_fallback_reason=seg.speaker_fallback_reason,
        )
        seg.end = split_at
        seg.orig_text = left_orig
        seg.target_text = left_tgt
        seg.words = []
        seg.segment_timing_needs_review = True
        seg.alignment_needs_review = True
        seg.invalidate_audio()
        right_seg.invalidate_audio()

        script.segments.append(right_seg)
        script.segments.sort(key=lambda item: (item.start, item.id))
        save_project_script(script)
    return {
        "left_id": seg.id,
        "right_id": new_id,
        "split_time_sec": split_at,
        "marked_pending_ids": [] if seg.keep_original else [seg.id, new_id],
    }


def merge_segments(
    project_id: str,
    segment_ids: List[int],
) -> Dict[str, Any]:
    """合并 2 条及以上同说话人相邻片段为一个连贯片段并失效旧音频。"""
    if len(segment_ids) < 2:
        raise ValueError("merge_segments 至少需要 2 个片段 ID")
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        by_id = {seg.id: seg for seg in script.segments}
        for sid in segment_ids:
            if sid not in by_id:
                raise KeyError(f"片段 ID {sid} 不存在")
        targets = sorted((by_id[sid] for sid in dict.fromkeys(segment_ids)), key=lambda item: (item.start, item.id))
        if len(targets) < 2:
            raise ValueError("merge_segments 需要至少 2 个不同片段")
        speakers = {seg.speaker for seg in targets}
        if len(speakers) != 1:
            raise ValueError("只能合并同一说话人的片段")
        merged_dur = targets[-1].end - targets[0].start
        if merged_dur > 16.0:
            raise ValueError(f"合并后总时长 {merged_dur:.2f}s 超过 16s 上限")

        def _join(parts: list[str]) -> str:
            acc = ""
            for p in parts:
                p = (p or "").strip()
                if not p:
                    continue
                if not acc:
                    acc = p
                elif re.search(r"[A-Za-z0-9]$", acc) and re.match(r"[A-Za-z0-9]", p):
                    acc = f"{acc} {p}"
                else:
                    acc = f"{acc}{p}"
            return acc

        keeper = targets[0]
        removed_ids = {seg.id for seg in targets[1:]}
        keeper.end = targets[-1].end
        keeper.orig_text = _join([seg.orig_text for seg in targets])
        keeper.target_text = _join([seg.target_text for seg in targets])
        keeper.words = []
        keeper.segment_timing_needs_review = True
        keeper.alignment_needs_review = True
        keeper.keep_original = all(seg.keep_original for seg in targets)
        keeper.invalidate_audio()

        if script.evidence is not None:
            for unit in [*script.evidence.translation_units, *script.evidence.performance_units]:
                if any(sid in removed_ids for sid in unit.segment_ids):
                    unit.segment_ids = list(dict.fromkeys(
                        keeper.id if sid in removed_ids else sid for sid in unit.segment_ids
                    ))
            keeper.translation_unit_ids = list(dict.fromkeys(
                tid for seg in targets for tid in seg.translation_unit_ids
            ))
            keeper.performance_unit_ids = list(dict.fromkeys(
                pid for seg in targets for pid in seg.performance_unit_ids
            ))
            keeper.source_token_ids = list(dict.fromkeys(
                tok for seg in targets for tok in seg.source_token_ids
            ))
            keeper.source_turn_ids = list(dict.fromkeys(
                turn for seg in targets for turn in seg.source_turn_ids
            ))

        script.segments = [seg for seg in script.segments if seg.id not in removed_ids]
        save_project_script(script)
    return {
        "kept_id": keeper.id,
        "removed_ids": sorted(removed_ids),
        "merged_duration": round(keeper.end - keeper.start, 3),
        "marked_pending_ids": [] if keeper.keep_original else [keeper.id],
    }


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "inspect_project_quality": inspect_project_quality,
    "query_script": query_script,
    "batch_update_segments": batch_update_segments,
    "split_segment": split_segment,
    "merge_segments": merge_segments,
    "synthesize_segments": synthesize_segments,
    "update_mix_settings": update_mix_settings,
    "render_final_video": render_final_video,
}


# ---------------------------------------------------------------------------
# OpenAI Function Calling 工具定义列表
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_project_quality",
            "description": "对全片剧本、文本密度、斜杠硬伤、切分长度及已合成 WAV 时长/首尾静音进行全面体检。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_script",
            "description": "按时间范围或说话人筛选查询当前剧本片段信息，修改前应先查询确认片段 ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "start_sec": {"type": "number", "description": "可选，起始时间过滤（秒）"},
                    "end_sec": {"type": "number", "description": "可选，结束时间过滤（秒）"},
                    "speaker": {"type": "string", "description": "可选，说话人 Key，如 SPEAKER_00"},
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_update_segments",
            "description": "批量修改剧本片段的角色、台词、情绪、动作、原声保留或语速，并将内容变更片段标记为待合成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "updates": {
                        "type": "array",
                        "description": "待修改片段补丁列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer", "description": "片段 ID"},
                                "speaker": {"type": "string"},
                                "target_text": {"type": "string"},
                                "target_lang": {"type": "string"},
                                "emotion": {
                                    "type": "string",
                                    "enum": ["neutral", "angry", "happy", "sad", "whisper"],
                                },
                                "action": {"type": "string"},
                                "keep_original": {"type": "boolean"},
                                "speed_factor": {"type": "number"},
                            },
                            "required": ["id"],
                        },
                    },
                },
                "required": ["project_id", "updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_segment",
            "description": "将过长片段（>12s）按静音谷或指定时间拆分为前后两段并标记待重合成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "segment_id": {"type": "integer", "description": "待拆分的片段 ID"},
                    "split_time_sec": {"type": "number", "description": "可选，切分绝对时间点（秒）；留空则自动探测静音谷"},
                    "left_orig_text": {"type": "string", "description": "可选，左半段原文"},
                    "right_orig_text": {"type": "string", "description": "可选，右半段原文"},
                    "left_target_text": {"type": "string", "description": "可选，左半段译文"},
                    "right_target_text": {"type": "string", "description": "可选，右半段译文"},
                },
                "required": ["project_id", "segment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_segments",
            "description": "合并 2 条及以上同说话人相邻片段，消除 1 秒左右过短孤立碎片。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "待合并的相邻同角色片段 ID 列表",
                    },
                },
                "required": ["project_id", "segment_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_segments",
            "description": "按角色分组批量合成指定片段（留空则合成所有待合成片段）并返回时长超标检查报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "segment_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "可选，指定要重合成的片段 ID 列表",
                    },
                },
                "required": ["project_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_mix_settings",
            "description": "动态调节背景伴奏混音音量（0.0 ~ 2.0）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "bgm_volume": {"type": "number", "description": "BGM 音量系数（0.0 到 2.0）"},
                },
                "required": ["project_id", "bgm_volume"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_final_video",
            "description": "触发 pydub 拼轨与 FFmpeg 混音封装，导出最终成片 output.mp4。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "项目 ID"},
                    "burn_watermark": {
                        "type": "boolean",
                        "description": "是否烧录“AI 合成语音”硬水印（默认 false 使用无损流拷贝）",
                    },
                },
                "required": ["project_id"],
            },
        },
    },
]
