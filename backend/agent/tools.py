"""Agent 五大确定性原子工具封装与 OpenAI Function Calling Schema。

所有工具均通过读写 workspace/{project_id}/script.json 保持全链路单一真相源。
"""

from __future__ import annotations

import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from weakref import WeakValueDictionary

from backend import config
from backend.schemas import ProjectScript, Segment, SegmentPatch, SynthesizeResponse
from backend.services.media_mixer import mix_and_render
from backend.services.synthesis import synthesize_with_duration_check
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
    bgm_volume: float,
) -> Dict[str, Any]:
    """更新背景伴奏音量系数（限 0.0 ~ 2.0）并持久化 script.json。"""
    clamped = max(0.0, min(2.0, float(bgm_volume)))
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        script.media.bgm_volume = clamped
        save_project_script(script)
    return {"bgm_volume": clamped}


def render_final_video(
    project_id: str,
    burn_watermark: bool = False,
    burn_subtitles: str = "none",
    auto_fit_overflow: bool = False,
) -> Dict[str, Any]:
    """触发 pydub 拼轨与 FFmpeg 混音封装，输出成片路径并回写 script.json。"""
    with project_task_guard(project_id):
        script = load_project_script(project_id)
        output_video, duration = mix_and_render(
            script,
            burn_watermark=burn_watermark,
            burn_subtitles=burn_subtitles,  # type: ignore[arg-type]
            auto_fit_overflow=auto_fit_overflow,
        )
        script.media.output_video = output_video
        save_project_script(script)
    return {
        "output_video": output_video,
        "duration": duration,
        "burn_watermark": burn_watermark,
        "burn_subtitles": burn_subtitles,
    }


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "query_script": query_script,
    "batch_update_segments": batch_update_segments,
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
